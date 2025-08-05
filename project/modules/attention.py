import torch
from torch import nn

class LSTMAttention(nn.Module):
    def __init__(self, input_dim, hidden_size):
        super().__init__()
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
            input_features
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
        return self.softmax(
            self.w(
                self.activation(
                    self.hidden_state_linear(prev_hidden_state) + \
                    self.input_linear(input_features)
                )
            )
        )

