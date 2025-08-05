import torch
from torch import nn
from typing import List
from transformers import AutoModel, AutoTokenizer, AutoConfig
from utils.module_utils import fasttext_embedding_module, _batch_padding_string
from utils.phoc.build_phoc import build_phoc
from utils.layers import L2Norm
from utils.registry import registry
from utils.vocab import PretrainedVocab, OCRVocab


#----------Word Embedding----------
class WordEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        #-- Load config and args
        self.model_config = registry.get_config("model_attributes")
        self.text_embedding_config = self.model_config["text_embedding"]
        self.device = registry.get_args("device")
        #-- Load pretrained model (phobert-base)
        self.load_pretrained()

        #-- Load params
        vocab_path=self.text_embedding_config["common_vocab"]
        self.max_length = self.text_embedding_config["max_length"]
        self.common_vocab = PretrainedVocab(
            model=self.model,
            tokenizer=self.tokenizer,
            vocab_file=vocab_path
        )

    def load_pretrained(self):
        self.roberta_model_name = self.config["model_decoder"]
        roberta_config = AutoConfig.from_pretrained(self.roberta_model_name)
        roberta_config.num_attention_heads = self.config["mutimodal_transformer"]["nhead"]
        roberta_config.num_hidden_layers = self.config["mutimodal_transformer"]["num_layers"]
        roberta_model = AutoModel.from_pretrained(
            self.roberta_model_name, 
            config=roberta_config
        )
        roberta_model.gradient_checkpointing_enable()
        self.model = roberta_model
        self.tokenizer = AutoTokenizer.from_pretrained(self.roberta_model_name)
        

    def get_prev_inds(self, sentences, ocr_tokens):
        """
            Use to get inds of each token of the caption sentences

            Parameters:
            ----------
            sentences: List[str]
                - Caption of the images
            
            ocr_tokens: List[List[str]]

            Return:
            ----------
            prev_ids: Tensor:
                - All inds of all word in the sentences 
        """
        ocr_vocab_object = OCRVocab(ocr_tokens=ocr_tokens)

        start_token = self.common_vocab.get_start_token()
        end_token = self.common_vocab.get_end_token()
        pad_token = self.common_vocab.get_pad_token()
        
        sentences_tokens = [
            sentence.split(" ")
            for sentence in sentences
        ]
        sentences_tokens = _batch_padding_string(
            sequences=sentences_tokens,
            max_length=self.config["max_length"],
            pad_value=pad_token,
            return_mask=False
        )
        sentences_tokens = [
            [start_token] + sentence_tokens[:self.max_length - 2] + [end_token]
            for sentence_tokens in sentences_tokens
        ]

        # Get prev_inds
        prev_ids = [
            [
                self.common_vocab.get_size() + ocr_vocab_object[sen_id].get_word_idx(token)
                if token in ocr_tokens[sen_id]
                # else ocr_vocab_object[sen_id].get_word_idx(token)
                else self.common_vocab.get_word_idx(token)
                for token in sentence_tokens
            ] 
            for sen_id, sentence_tokens in enumerate(sentences_tokens)
        ]
        return torch.tensor(prev_ids).to(self.device)



#----------Embedding----------
class BaseEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = registry.get_config("model_attributes")
        self.device = registry.get_args("device")
        self.l2norm = L2Norm()
        self.layer_norm = nn.LayerNorm(normalized_shape=self.config["hidden_size"])
        self.activation = nn.ReLU()

    def forward(self):
        NotImplemented


#----------Embedding Obj----------
class ObjEmbedding(BaseEmbedding):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(
            in_features=self.config["obj"]["dim"],
            out_features=self.config["hidden_size"]
        )
    

    def forward(
            self,
            obj_feat 
        ):
        """
            Function:
            ---------
            This module is for object embedding

            Parameters:
            -----------
                - obj_feat: (BS, M, 1024)
                    + Features for each objects 
        """
        return self.activation(
            self.layer_norm(
                self.projection(
                    self.l2norm(obj_feat)
                )
            )
        )


#----------Embedding OBJ----------
class OCREmbedding(BaseEmbedding):
    def __init__(self):
        super().__init__()
        self.projection_feat = nn.Linear(
            in_features=self.config["ocr"]["concat_dim"],
            out_features=self.config["hidden_size"]
        )

        self.projection_bbox = nn.Linear(
            in_features=8,
            out_features=self.config["hidden_size"]
        )
    
    # Nên để ở Dataset lúc dataloader để tiết kiệm thời gian 
    def convert_box(self, ocr_boxes):
        """
            Function:
            ---------
            This module convert boxes from (1) - (2)
                (1) [x_min, y_min, x_max, y_max] format 
                (2) [x1, x2, x3, x4, y1, y2, y3, y4] format 

            Parameters:
            -----------
                - ocr_boxes: (BS, N, 4)
                    + Bounding box for each ocr tokens 
        """
        new_boxes = []
        for boxes in ocr_boxes:
            x_min, y_min, x_max, y_max = boxes[1], boxes[2], boxes[3], boxes[4]
            x1, x2, x3, x4 = x_min, x_min, x_max, x_max
            y1, y2, y3, y4 = y_min, y_max, y_max, y_min
            new_boxes.append([x1, x2, x3, x4, y1, y2, y3, y4])
        new_boxes = torch.tensor(new_boxes).to(self.device)
        return new_boxes


    def phoc_embedding(self, words: List[str]):
        """
            :params words:  List of word needed to embedded
        """
        phoc_embed = [
            
            build_phoc(token=word) 
            for word in words
        ]
        
        return torch.tensor(phoc_embed).to(self.device)
    

    def fasttext_embedding(self, words: List[str]):
        """
            :params words:  List of word needed to embedded
        """
        fasttext_embedding = [
            fasttext_embedding_module(
                model=self.fasttext_model,
                word=word
            ) 
            for word in words
        ]
        return torch.tensor(fasttext_embedding).to(self.device)

    
    def forward(
            self, 
            ocr_feat,
            ocr_boxes,
            ocr_tokens
        ):
        """
            Function:
            ---------
            This module embdding ocr features
            
            Parameters:
            -----------
                - ocr_boxes: (BS, N, 4)
                    + Bounding box for each ocr tokens 
                - ocr_feat: (BS, N, 4)
                    + Features for each ocr token 
                - ocr_token: (BS, N, 4)
                    + ocr_token for each images 
        """
        # Finding ocr main
        fasttext_embed = torch.stack([self.fasttext_embedding(words=tokens) for tokens in ocr_tokens])
        phoc_embed = torch.stack([self.phoc_embedding(words=tokens)  for tokens in ocr_tokens])
        ocr_main = torch.concat(
            [
                self.l2norm(ocr_feat),
                self.l2norm(fasttext_embed),
                self.l2norm(phoc_embed)
            ], dim=-1
        )

        # Finding ocr embed
        boxes_embed = self.activation(
            self.layer_norm(
                self.projection_bbox(ocr_boxes)
            )
        )

        ocr_main_proj = self.activation(
            self.layer_norm(
                self.projection_feat(ocr_main)
            )
        )

        ocr_embed = boxes_embed + ocr_main_proj
        return ocr_embed