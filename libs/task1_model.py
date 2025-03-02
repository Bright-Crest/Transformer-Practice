from torch import nn

class Model(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.PReLU(),
            nn.Linear(32, 64),
            nn.PReLU(),
            nn.Linear(64, 16),
            nn.PReLU(),
            nn.Linear(16, output_dim)
        )

    def forward(self, X):
        X = self.fc(X)
        return X


import numpy as np
import torch
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(self, max_len, d_model):
        super().__init__()
        self.d_model = d_model
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).float().unsqueeze(1)
        angular_speed = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * angular_speed) # even dimensions
        pe[:, 1::2] = torch.cos(position * angular_speed) # odd dimensions
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        # x is N, L, D
        # pe is 1, maxlen, D
        scaled_x = x * np.sqrt(self.d_model)
        encoded = scaled_x + self.pe[:, :x.size(1), :]
        return encoded


class TransformerModel(nn.Module):
    def __init__(self, transformer, input_len, target_len, n_features, is_with_pe=True):
        super().__init__()
        self.transf = transformer
        self.input_len = input_len
        self.target_len = target_len
        self.trg_masks = self.transf.generate_square_subsequent_mask(self.target_len)
        self.n_features = n_features
        self.proj = nn.Linear(n_features, self.transf.d_model)
        self.linear = nn.Linear(self.transf.d_model, n_features)
        
        max_len = max(self.input_len, self.target_len)
        self.is_with_pe = is_with_pe
        if is_with_pe:
            self.pe = PositionalEncoding(max_len, self.transf.d_model)
        self.norm = nn.LayerNorm(self.transf.d_model)
                
    def preprocess(self, seq):
        seq = self.proj(seq)
        if self.is_with_pe:
            seq = self.pe(seq)
        return self.norm(seq)
    
    def encode_decode(self, source, target, source_mask=None, target_mask=None):
        # Projections
        src = self.preprocess(source)
        tgt = self.preprocess(target)

        out = self.transf(src, tgt, 
                          src_key_padding_mask=source_mask, 
                          memory_key_padding_mask=source_mask,
                          tgt_mask=target_mask)

        # Linear
        out = self.linear(out) # N, L, F
        return out
        
    def predict(self, source_seq, source_mask=None):
        inputs = source_seq[:, -1:]
        for i in range(self.target_len):
            out = self.encode_decode(source_seq, inputs, 
                                     source_mask=source_mask,
                                     target_mask=self.trg_masks[:i+1, :i+1])
            out = torch.cat([inputs, out[:, -1:, :]], dim=-2)
            inputs = out.detach()
        outputs = out[:, 1:, :]
        return outputs
        
    def forward(self, X, source_mask=None):
        self.trg_masks = self.trg_masks.type_as(X)
        if source_mask:
            source_mask = source_mask.type_as(X)
        source_seq = X[:, :self.input_len, :]

        if self.training:            
            shifted_target_seq = X[:, self.input_len-1:-1, :]
            outputs = self.encode_decode(source_seq, shifted_target_seq, 
                                         source_mask=source_mask, 
                                         target_mask=self.trg_masks)
        else:
            outputs = self.predict(source_seq, source_mask)
            
        return outputs


from torch import nn


# class CNNModel(nn.Module):
#     def __init__(self, input_dim, output_dim):
#         super().__init__()
#         self.cnn = nn.Sequential(
#             # (1, input_dim, 1) => (16, input_dim - 2, 1)
#             nn.Conv2d(1, 16, (3, 1), stride=(1, 0), padding=(0, 0), padding_mode='circular'),
#             nn.ReLU(),
#             # (16, input_dim - 2, 1) => 
#             nn.MaxPool2d(2),
#             nn.Conv1d(16, 32, 3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.MaxPool1d(2),
#         )

class CNNModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.cnn = nn.Sequential(
            # (1, input_dim) => (16, input_dim - 2)
            nn.Conv1d(1, 16, 3, stride=1),
            nn.ReLU(),
            # => (32, input_dim - 4)
            nn.Conv1d(16, 32, 3, stride=1),
            nn.ReLU(),
        )
        self.fc = nn.Linear(32 * (input_dim - 4), output_dim)

    def forward(self, X):
        X = X.unsqueeze(1)
        X = self.cnn(X)
        X = X.view(X.shape[0], -1)
        X = self.fc(X)
        return X
                    
