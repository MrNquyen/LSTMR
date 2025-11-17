import torch
import fasttext
from torch import nn
from project.modules.multimodal_embedding import OCREmbedding, ObjEmbedding, WordEmbedding
# from project.modules.geo_relationship import GeoRelationship
from project.modules.geo_relationship_batch_processing import GeoRelationship
from project.modules.decoder import Decoder
from utils.registry import registry
from utils.utils import count_nan, save_tensor_txt, save_list_txt, check_requires_grad
from utils.module_utils import _batch_gather
from torch.nn import functional as F
from icecream import ic
import math


class LSTMR(nn.Module):
    def __init__(self):
        super().__init__()
        self.model_config = registry.get_config("model_attributes")
        self.device = registry.get_args("device")
        self.writer = registry.get_writer("common")
        self.max_length = self.model_config["text_embedding"]["max_length"]
        self.hidden_size = self.model_config["hidden_size"]
        self.build()
    #-- BUILD
    def build(self):
        self.writer.LOG_INFO("=== Build model params ===")
        self.build_model_params()
        
        self.writer.LOG_INFO("=== Build writer ===")
        self.build_writer()

        self.writer.LOG_INFO("=== Build model layers ===")
        self.build_layers()
        self.build_model_init()

        self.writer.LOG_INFO("=== Build model outputs ===")
        self.build_ouput()

        self.writer.LOG_INFO("=== Build adjust learning rate ===")
        self.adjust_lr()
    

    def build_writer(self):
        self.writer = registry.get_writer("common")


    def build_model_params(self):
        self.ocr_config = self.model_config["ocr"]
        self.obj_config = self.model_config["obj"]
        self.num_ocr, self.num_obj = self.ocr_config["num_ocr"], self.obj_config["num_obj"]
        self.dim_ocr, self.dim_obj = self.ocr_config["dim"], self.obj_config["dim"]

    def build_model_init(self):
        #~ Finetune module is the module has lower lr than others module
        self.finetune_modules = []
        self.build_fasttext_model()


    def build_fasttext_model(self):
        self.fasttext_model = fasttext.load_model(self.model_config["fasttext_bin"])

    def build_ouput(self):
        # Num choices = num vocab
        self.num_choices = self.word_embedding.common_vocab.get_size()
        self.classifier = nn.Linear(
            in_features=self.hidden_size,
            out_features=self.num_choices
        )
        self.ptr_net = OcrPtrNet(self.hidden_size)


    def build_layers(self):
        self.ocr_embedding = OCREmbedding()
        self.obj_embedding = ObjEmbedding()
        self.geo_relationship = GeoRelationship()
        self.word_embedding = WordEmbedding()
        self.decoder = Decoder()
        self.dropout = nn.Dropout(self.model_config["dropout"])

        #-- Init Hidden State
        self.init_h = nn.Linear(self.hidden_size, self.hidden_size)
        self.init_c = nn.Linear(self.hidden_size, self.hidden_size)
    

    def adjust_lr(self):
        #~ Word Embedding
        # self.add_finetune_modules(self.word_embedding)
        pass


    #-- ADJUST LEARNING RATE
    def add_finetune_modules(self, module: nn.Module):
        self.finetune_modules.append({
            'module': module,
            'lr_scale': self.model_config["adjust_optimizer"]["lr_scale"],
        })


    def get_optimizer_parameters(self, config_optimizer):
        """
            -----
            Function:
                - Modify learning rate
                - Fine-tuning layer has lower learning rate than others
        """
        optimizer_param_groups = []
        base_lr = config_optimizer["params"]["lr"]
        scale_lr = config_optimizer["lr_scale"]
        base_lr = float(base_lr)
        
        # collect all the parameters that need different/scaled lr
        finetune_params_set = set()
        for m in self.finetune_modules:
            optimizer_param_groups.append({
                "params": list(m['module'].parameters()),
                "lr": base_lr * scale_lr
            })
            finetune_params_set.update(list(m['module'].parameters()))
        # remaining_params are those parameters w/ default lr
        remaining_params = [
            p for p in self.parameters() if p not in finetune_params_set
        ]
        # put the default lr parameters at the beginning
        # so that the printed lr (of group 0) matches the default lr
        optimizer_param_groups.insert(0, {"params": remaining_params})
        
        return optimizer_param_groups
        

    def init_random(self, shape):
        return torch.normal(
            mean=0.0, 
            std=0.1, 
            size=shape
        ).to(self.device)


    def init_hidden_state(self, obj_features, ocr_features):
        """
        Creates the initial hidden and cell states for the decoder's LSTM based on the encoded images.

        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :return: hidden state, cell state
        """

        input_features = torch.concat([
            obj_features,
            ocr_features
        ], dim=1)
        fuse_features = torch.mean(input_features, dim=1)

        h = self.init_h(fuse_features)  # (batch_size, decoder_dim)
        c = self.init_c(fuse_features)
        return h, c
        
    #-- FORWARD
    def forward(
            self,
            batch
        ):
        obj_embed = self.obj_embedding(batch["list_obj_feat"])
        ocr_embed = self.ocr_embedding(
            ocr_feat=batch["list_ocr_feat"].to(self.device),
            ocr_boxes=batch["list_ocr_boxes"],
            ocr_tokens=batch["list_ocr_tokens"]
        )

        ocr_mask = batch["ocr_mask"]
        obj_mask = batch["obj_mask"]

        #-- Common embed and ocr embed
        batch_size = obj_embed.size(0)
        # common_vocab_embed = self.word_embedding.common_vocab.get_vocab_embedding()
        common_vocab_embed = self.classifier.weight
        common_vocab_embed = common_vocab_embed.unsqueeze(0).expand(batch_size, -1, -1)

        vocab_embed = torch.concat([
            common_vocab_embed,
            ocr_embed
        ], dim=1).to(self.device)
        vocab_size = vocab_embed.size(1)

        # -- Training or Evaluating
        pad_idx = self.word_embedding.common_vocab.get_pad_index()
        start_idx = self.word_embedding.common_vocab.get_start_index() 

        # prev_h = self.init_random(shape=(batch_size, self.hidden_size))
        # prev_c = self.init_random(shape=(batch_size, self.hidden_size))

        prev_h, prev_c = self.init_hidden_state(obj_embed, ocr_embed)
        prev_attended_vector = self.init_random((batch_size, self.hidden_size))

        prev_inds = torch.full((batch_size, self.max_length), pad_idx).to(self.device)
        prev_inds[:, 0] = start_idx

        scores = torch.full((batch_size, self.max_length, vocab_size), pad_idx).to(self.device).to(torch.float32)
          
        if self.training:
            #~ Get inds of all token in the sentences
            caption_inds = self.word_embedding.get_prev_inds(
                sentences=batch["list_captions"],
                ocr_tokens=batch["list_ocr_tokens"]
            ) # BS, max_length, 1
            # ic(caption_inds)

            #~ Get inds of all token in the sentences
            caption_embed = _batch_gather(
                x=vocab_embed,
                inds=caption_inds
            )
            for step in range(self.max_length):
                prev_word_inds = caption_inds[:, step]
                prev_word_embed = caption_embed[:, step, :]
                cur_h, cur_c, prev_attended_vector = self.decoder(
                    obj_features=obj_embed,
                    ocr_features=ocr_embed,
                    ocr_mask=ocr_mask,
                    obj_mask=obj_mask,
                    prev_hidden_state=prev_h,
                    prev_cell_state=prev_c,
                    prev_attended_vector=prev_attended_vector,
                    prev_word_embed=prev_word_embed
                )
                prev_h = cur_h
                prev_c = cur_c

                results = {
                    "hidden_state": cur_h,
                    "prev_word_inds": torch.flatten(prev_word_inds),
                    "vocab_size": self.num_choices,
                    "ocr_feat": ocr_embed,
                    "ocr_boxes": batch["list_ocr_boxes"],
                    "ocr_mask": ocr_mask
                }
                score = self.forward_output(results=results) # BS, num_common + num_ocr, 1
                scores[:, step, :] = score.permute(0, 2, 1).squeeze(1) # BS, 1, num_common + num_ocr
            return scores, caption_inds
            
        else:
            with torch.no_grad():
                for step in range(0, self.max_length, 1):
                    prev_word_embed = _batch_gather(
                        x=vocab_embed,
                        inds=prev_inds
                    )[:, step-1, :]

                    cur_h, cur_c = self.decoder(
                        obj_features=obj_embed,
                        ocr_features=ocr_embed,
                        ocr_mask=ocr_mask,
                        obj_mask=obj_mask,
                        prev_hidden_state=prev_h,
                        prev_cell_state=prev_c,
                        prev_attended_vector=prev_attended_vector,
                        prev_word_embed=prev_word_embed
                    )
                    prev_h = cur_h
                    prev_c = cur_c

                    results = {
                        "hidden_state": cur_h,
                        "prev_word_inds": torch.flatten(prev_inds[:, step-1]), # Previous idx
                        "vocab_size": self.num_choices,
                        "ocr_feat": ocr_embed,
                        "ocr_boxes": batch["list_ocr_boxes"],
                        "ocr_mask": ocr_mask
                    }
                    score = self.forward_output(results=results) # BS, num_common + num_ocr, 1
                    scores[:, step, :] = score.permute(0, 2, 1).squeeze(1) # BS, 1, num_common + num_ocr
                    argmax_inds = score.argmax(dim=1) # BS, 1
                    # ic(argmax_inds.shape)
                    prev_inds[:, step] = argmax_inds.squeeze(-1)
                return scores, prev_inds

    def forward_output(self, results):
        """
        Calculate scores for ocr tokens and common word at each timestep

            Parameters:
            ----------
            results: dict
                - The result output of decoder step

            Return:
            ----------
        """
        #~ All value
        hidden_state = results["hidden_state"]
        prev_inds = results["prev_word_inds"]


        common_vocab_size = results["vocab_size"]
        ocr_feat = results["ocr_feat"]
        ocr_boxes = results["ocr_boxes"] # BS, num_ocr, 4
        ocr_mask = results["ocr_mask"]

        num_ocr = ocr_boxes.size(1)

        #~ Leverage scores
            #~~ Get prev word boxes inds
        prev_boxes_inds = torch.tensor([
            ind - common_vocab_size
            if ind >= common_vocab_size
            else -1
            for ind in prev_inds
        ]).unsqueeze(1).to(self.device)

            #~~ Lookup for suitable boxes
        
        # Find where idx = -1 
        # Assign 0 to batch gather and assign zero array for later
        prev_boxes_inds_tmp = prev_boxes_inds.clone()
        prev_boxes_inds[prev_boxes_inds == -1] = 0
        prev_boxes = _batch_gather(
            x=ocr_boxes,
            inds=prev_boxes_inds
        ).squeeze(1) # BS, 1, 4

            #~~ Geo relation embed
        geo_relation_embed = torch.stack([
            torch.zeros((num_ocr, self.hidden_size)).to(self.device)
            if ind == torch.tensor([-1]).to(self.device) # If no geo relationship - Fill with zeros tensor
            else self.geo_relationship(
                target=prev_box,
                ocr_boxes=ocr_boxes[id, :, :].squeeze()
            )
            for id, (ind, prev_box) in enumerate(zip(prev_boxes_inds_tmp, prev_boxes))
        ]).to(self.device)

            #~~ Geo relation embed
        augmented_features = torch.concat([
            ocr_feat, geo_relation_embed
        ], dim=-1)

            #~~ Calculate common vocab scores

        fixed_scores = self.classifier(hidden_state).unsqueeze(-1) #  BS, num_vocab, 1
        ocr_scores = self.ptr_net(
            hidden_state=hidden_state,
            augmented_features=augmented_features, 
            attention_mask=ocr_mask,
        )

            #~~ Classify scores
        scores = torch.concat([
            fixed_scores,
            ocr_scores
        ], dim=1)
        return scores # BS, num_common + num_ocr, 1


# ----- RELATION AWARE POINTER NETWORK -----
class OcrPtrNet(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.linear_hidden_state = nn.Linear(
            in_features=hidden_size,
            out_features=hidden_size
        )

        self.linear_augmented_feat = nn.Linear(
            in_features=hidden_size * 2,
            out_features=hidden_size
        )

    def forward(self, hidden_state, augmented_features, attention_mask):
        """
            Calculate scores for ocr tokens and common word 
            at each timestep using bilinear pooling operation

            Parameters:
            ----------
            hidden_state: BS, hidden_state
                - The latest hidden state in decoder step
                
            augmented_features: BS, num_ocr, hidden_size * 2
                - Augmented features adding geo relationship to ocr features
            
            common_scores: BS, vocab_size, 1
                - Scores for common vocab
            
            Return:
            ----------
        """
        linear_pooling_ht = self.linear_hidden_state(hidden_state).unsqueeze(-1) # BS, hidden_size, 1
        linear_pooling_aug_feat = self.linear_augmented_feat(augmented_features) # BS, num_ocr, hidden_size

        extended_attention_mask = (1.0 - attention_mask) * -10000.0
        extended_attention_mask = extended_attention_mask.unsqueeze(-1)

        ocr_scores = torch.bmm(
            input=linear_pooling_aug_feat,
            mat2=linear_pooling_ht
        ) # BS, num_ocr, 1

        ocr_scores = ocr_scores / math.sqrt(self.hidden_size)
        ocr_scores = ocr_scores + extended_attention_mask
        
        return ocr_scores
 


