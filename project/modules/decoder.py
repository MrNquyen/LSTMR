import torch
from torch import nn
from project.modules.attention import LSTMAttention
from torch.nn import functional as F
from utils.registry import registry 
from icecream import ic

class DecoderBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = registry.get_args("device")
        self.model_config = registry.get_config("model_attributes")
        self.hidden_size = self.model_config["hidden_size"]


class Decoder(DecoderBase):
    def __init__(self):
        super().__init__()
        
        #-- Layers
        self.lstm_cell = nn.LSTMCell(
            input_size=self.hidden_size*3,
            hidden_size=self.hidden_size,
            device=self.device,
            dtype=torch.float32
        )

        self.attention = LSTMAttention(
            input_dim=self.hidden_size,
            hidden_size=self.hidden_size
        )

    def forward(
            self, 
            obj_features,
            ocr_features,
            obj_mask,
            ocr_mask,
            prev_hidden_state,
            prev_cell_state,
            prev_word_embed
        ):
        #-- Concat OCR features and Obj features
        mask = torch.cat([obj_mask, ocr_mask], dim=1).unsqueeze(-1) 
        input_features = torch.concat([
            obj_features,
            ocr_features
        ], dim=1)
        num_features = input_features.size(1)

        #-- Fuse OCR and Obj features with mean
        fuse_features = torch.mean(input_features, dim=1)

        #-- Calculate attention scores
        attention_scores = self.attention(
            prev_hidden_state=prev_hidden_state,
            input_features=input_features,
            mask=mask
        )

        #==== DEBUG ===-
        # debug_scores = attention_scores.squeeze(-1)  # [B, L]
        # for b in range(mask.size(0)):
        #     pad_idx = (mask[b] == 0).nonzero(as_tuple=True)[0]  # indices of pads
        #     nan_idx = (debug_scores[b].isnan()).nonzero(as_tuple=True)[0]  # NaN positions

        #     if pad_idx.numel() > 0:
        #         print(f"\nBatch {b} - pad idx: {pad_idx.tolist()}")
        #         print("Attention scores at pad positions:",
        #             debug_scores[b, pad_idx].tolist())

        #     if nan_idx.numel() > 0:
        #         print(f"Batch {b} - NaN at positions: {nan_idx.tolist()}")
        #         print("NaN values:", debug_scores[b, nan_idx].tolist())
        #==== DEBUG ===-

        prev_hidden_state_attention = input_features * attention_scores
        attended_vector = torch.sum(prev_hidden_state_attention, dim=1)

        #-- Input to cell
        cell_inputs = torch.concat([
            fuse_features,
            attended_vector,
            prev_word_embed
        ], dim=-1)

        ht, ct = self.lstm_cell(
            cell_inputs,
            (prev_hidden_state, prev_cell_state)
        )
        return ht, ct # Hidden_state, cell_state

    