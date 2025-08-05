import torch
import fasttext
from torch import nn
from modules.multimodal_embedding import OCREmbedding, ObjEmbedding, WordEmbedding
from modules.geo_relationship import GeoRelationship
from modules.decoder import Decoder
from utils.registry import registry
from utils.module_utils import _batch_gather
from torch.nn import functional as F


class LSTMR(nn.Module):
    def __init__(self):
        super().__init__()
        self.model_config = registry.get_config("model_attributes")
        self.device = registry.get_args("device")
        self.max_length = self.model_config["text_embedding"]["max_length"]
        self.hidden_state = self.model_config["hidden_state"]

    #-- BUILD
    def build(self):
        self.writer.LOG_INFO("=== Build model params ===")
        self.build_model_params()
        
        self.writer.LOG_INFO("=== Build writer ===")
        self.build_writer()

        self.writer.LOG_INFO("=== Build model layers ===")
        self.build_layers()

        self.writer.LOG_INFO("=== Build model outputs ===")
        self.build_ouput()

        self.writer.LOG_INFO("=== Build adjust learning rate ===")
        self.adjust_lr()
    

    def build_writer(self):
        self.writer = registry.get_writer("common")


    def build_model_init(self):
        #~ Finetune module is the module has lower lr than others module
        self.finetune_modules = []
        self.build_fasttext_model()


    def build_fasttext_model(self):
        self.fasttext_model = fasttext.load_model(self.config["fasttext_bin"])

    def build_ouput(self):
        # Num choices = num vocab
        self.num_choices = self.word_embedding.common_vocab.get_size()
        self.classifier = nn.Linear(
            in_features=self.hidden_size,
            out_features=self.num_choices
        )
        self.ptr_net = OcrPtrNet(self.hidden_state)

    def build_layers(self):
        self.ocr_embedding = OCREmbedding()
        self.obj_embedding = ObjEmbedding()
        self.geo_relationship = GeoRelationship()
        self.word_embedding = WordEmbedding()
        self.decoder = Decoder()

    
    def adjust_lr(self):
        #~ Word Embedding
        self.add_finetune_modules(self.word_embedding)


    #-- ADJUST LEARNING RATE
    def add_finetune_modules(self, module: nn.Module):
        self.finetune_modules.append({
            'module': module,
            'lr_scale': self.config["adjust_optimizer"]["lr_scale"],
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
        
        # check_overlap(finetune_params_set, remaining_params)
        return optimizer_param_groups
        
        
    #-- FORWARD
    def forward(
            self,
            batch
        ):

        obj_embed = self.obj_embedding(batch["list_obj_feat"])
        ocr_embed = self.ocr_embedding(
            ocr_feat=batch["list_ocr_feat"],
            ocr_boxes=batch["list_ocr_boxes"],
            ocr_tokens=batch["list_ocr_tokens"]
        )
          
        #-- Common embed and ocr embed
        common_vocab_embed = self.word_embedding.common_vocab.get_vocab_embedding()
        vocab_embed = torch.concat([
            common_vocab_embed,
            ocr_embed
        ]).to(self.device)
        vocab_size = vocab_embed.size(1)

        # -- Training or Evaluating
        pad_idx = self.word_embedding.common_vocab.get_pad_index()
        start_idx = self.word_embedding.common_vocab.get_start_index() 
        batch_size = obj_embed.size(0)

        prev_h = None
        prev_c = None
        prev_inds = torch.full((batch_size, self.max_length), pad_idx).to(self.device)
        prev_inds[:, 0] = start_idx

        scores = torch.full((batch_size, self.max_length, vocab_size), pad_idx).to(self.device)
          
        if self.training:
            #~ Get inds of all token in the sentences
            caption_inds = self.word_embedding.get_prev_inds(
                sentences=batch["list_captions"],
                ocr_tokens=batch["list_ocr_tokens"]
            ) # BS, max_length, 1

            #~ Get inds of all token in the sentences
            caption_embed = _batch_gather(
                x=vocab_embed,
                inds=prev_inds
            )
            for step in range(self.max_length):
                prev_inds = caption_inds[:, step]
                prev_word_embed = caption_embed[:, step, :]
                cur_h, cur_c = self.decoder(
                    obj_features=obj_embed,
                    ocr_features=ocr_embed,
                    prev_hidden_state=prev_h,
                    prev_cell_state=prev_c,
                    prev_word_embed=prev_word_embed
                )
                prev_h = cur_h
                prev_c = cur_c
                results = {
                    "hidden_state": cur_h,
                    "prev_inds": torch.flatten(prev_inds),
                    "vocab_size": self.num_choices,
                    "ocr_feat": ocr_embed,
                    "ocr_boxes": batch["list_ocr_boxes"]
                }
                score = self.forward_output(results=results) # BS, num_common + num_ocr, 1
                scores[:, step, :] = score.permute(0, 2, 1) # BS, 1, num_common + num_ocr
            return scores, caption_inds
        else:
            for step in range(1, self.max_length, 1):
                prev_word_embed = _batch_gather(
                    x=vocab_embed,
                    inds=prev_inds
                )[:, step-1, :]
                cur_h, cur_c = self.decoder(
                    obj_features=obj_embed,
                    ocr_features=ocr_embed,
                    prev_hidden_state=prev_h,
                    prev_cell_state=prev_c,
                    prev_word_embed=prev_word_embed
                )
                prev_h = cur_h
                prev_c = cur_c
                results = {
                    "hidden_state": cur_h,
                    "prev_inds": torch.flatten(prev_inds),
                    "vocab_size": self.num_choices,
                    "ocr_feat": ocr_embed,
                    "ocr_boxes": batch["list_ocr_boxes"]
                }
                score = self.forward_output(results=results) # BS, num_common + num_ocr, 1
                scores[:, step, :] = score.permute(0, 2, 1) # BS, 1, num_common + num_ocr
                argmax_inds = score.argmax(dim=1) # BS, 1
                prev_inds[:, step] = argmax_inds
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
        prev_inds = results["prev_inds"]
        common_vocab_size = results["vocab_size"]
        ocr_feat = results["ocr_feat"]
        ocr_boxes = results["ocr_boxes"] # BS, num_ocr, 4
        num_ocr = ocr_boxes.size(1)

        #~ Leverage scores
            #~~ Get prev word boxes inds
        prev_boxes_inds = torch.tensor([
            ind - common_vocab_size
            if ind >= common_vocab_size
            else -1
            for ind in prev_inds
        ]).unsqeeze(1)

            #~~ Lookup for suitable boxes
        prev_boxes = _batch_gather(
            x=ocr_boxes,
            inds=prev_boxes_inds
        ).squeeze(1) # BS, 1, 4

            #~~ Geo relation embed
        geo_relation_embed = torch.stack([
            self.geo_relationship(
                target=prev_box,
                ocr_boxes=ocr_boxes[id, :, :].squeeze()
            )
            if ind == -1 # If no geo relationship - Fill with zeros tensor
            else torch.zeros((num_ocr, self.hidden_state))
            for id, (ind, prev_box) in enumerate(zip(prev_boxes_inds, prev_boxes))
        ]).to(self.device)

            #~~ Geo relation embed
        augmented_features = torch.concat([
            ocr_feat, geo_relation_embed
        ], dim=-1)

            #~~ Calculate common vocab scores
        fixed_scores = self.classifier(hidden_state).unsqueeze(-1) #  BS, num_vocab, 1
        scores = self.ptr_net(
            augmented_features=augmented_features, 
            common_scores=fixed_scores,
        )
        return scores

# ----- RELATION AWARE POINTER NETWORK -----
class OcrPtrNet(nn.Module):
    def __init__(self, hidden_state):
        super().__init__()
        self.linear_hidden_state = nn.Linear(
            in_features=hidden_state,
            out_features=hidden_state
        )

        self.linear_augmented_feat = nn.Linear(
            in_features=hidden_state * 2,
            out_features=hidden_state
        )

    def forward(self, hidden_state, augmented_features, common_scores):
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

        ocr_scores = torch.bmm(
            input=linear_pooling_aug_feat,
            mat2=linear_pooling_ht
        ) # BS, num_ocr, 1

        #-- Maximum probabilities scores
        scores = F.sigmoid(torch.concat([
            common_scores,
            ocr_scores
        ], dim=1))
        return scores # BS, num_common + num_ocr, 1 


