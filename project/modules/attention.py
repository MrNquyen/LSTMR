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