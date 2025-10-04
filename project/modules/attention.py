import torch
from torch import nn
from icecream import ic
import torch.nn.functional as F
from utils.registry import registry

class LSTMAttention(nn.Module):
    def __init__(self, input_dim, hidden_size):
        super().__init__()
        self.device = registry.get_args("device")
        self._mask_value = -1e9
        self.activation = nn.Tanh()
        self.softmax = nn.Softmax()
        self.w = nn.Linear(
            in_features=hidden_size,
            out_features=1
        )
        self.hidden_state_linear = nn.Linear(
            in_features=input_dim,
            out_features=hidden_size
        )
        self.input_linear = nn.Linear(
            in_features=input_dim,
            out_features=hidden_size
        )

    def forward(
            self,
            prev_hidden_state,
            input_features,
            mask,
        ):
        """
            Function:
            ---------
            This module calculate attention scores for each ocr-obj in LSTMCell 
            based on last hidden state and input features

            Parameters:
            -----------
                - prev_hidden_state: BS, hidden_size
                - input_features: BS, M+N, input_dim
                    + Object and OCR Features

            Output:
            -------
            
            Attention scores table with (BS, M+N, 1)

        """
        scores = self.w(
            self.activation(
                self.hidden_state_linear(prev_hidden_state.unsqueeze(1)) + \
                self.input_linear(input_features)
            )
        ).to(self.device)
        # extended_attention_mask = (1.0 - mask) * -10000.0
        # extended_attention_mask = extended_attention_mask
        # scores = scores + extended_attention_mask

        scores = scores.masked_fill(mask == 0, self._mask_value)
        attn_weights = self.softmax(scores)

        #-- Avoid nan
        # attn_weights = torch.where(
        #     attn_weights.isnan(),
        #     torch.zeros_like(attn_weights),
        #     attn_weights
        # )
        return attn_weights
        

# class LSTMAttention(nn.Module):
#     def __init__(self, input_dim, hidden_size):
#         super().__init__()
#         self._mask_value = -1e9
#         self.activation = nn.Tanh()
#         self.softmax = nn.Softmax()
#         self.w = nn.Linear(
#             in_features=hidden_size,
#             out_features=1
#         )
#         self.hidden_state_linear = nn.Linear(
#             in_features=input_dim,
#             out_features=hidden_size
#         )
#         self.input_linear = nn.Linear(
#             in_features=input_dim,
#             out_features=hidden_size
#         )

#     def forward(
#             self,
#             prev_hidden_state,
#             input_features,
#             mask,
#         ):
#         """
#             Function:
#             ---------
#             This module calculate attention scores for each ocr-obj in LSTMCell 
#             based on last hidden state and input features

#             Parameters:
#             -----------
#                 - prev_hidden_state: BS, hidden_size
#                 - input_features: BS, M+N, input_dim
#                     + Object and OCR Features

#             Output:
#             -------
            
#             Attention scores table with (BS, M+N, 1)

#         """
#         B, L, _ = input_features.size()

#         # project hidden state and expand to (B, 1, H)
#         h_proj = self.hidden_state_linear(prev_hidden_state)            # (B, H)
#         h_proj = h_proj.unsqueeze(1)                                    # (B, 1, H)

#         # project input features to same hidden dim
#         x_proj = self.input_linear(input_features)                      # (B, L, H)

#         # combined -> non-linearity -> score
#         combined = self.activation(h_proj + x_proj)                     # (B, L, H)
#         scores = self.w(combined)                                       # (B, L, 1)

#         # normalize mask shape
#         if mask.dim() == 2:
#             mask2 = mask.unsqueeze(-1)                                  # (B, L, 1)
#         else:
#             mask2 = mask

#         # mask logits (use large negative value to avoid NaNs)
#         scores = scores.masked_fill(mask2 == 0, self._mask_value)

#         # softmax over locations (dim=1)
#         attn_weights = F.softmax(scores, dim=1)                         # (B, L, 1)

#         # if sample has all masked positions -> softmax can be NaN; replace by zeros
#         # attn_weights = torch.where(
#         #     attn_weights != attn_weights,   # isnan test
#         #     torch.zeros_like(attn_weights),
#         #     attn_weights,
#         # )
#         # ic(attn_weights)

#         return attn_weights
        

