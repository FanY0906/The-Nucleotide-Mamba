from conv_mamba_block import conv_mamba_block
from classification import classification
import torch.nn as nn
import torch





class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.n_vocab = 7
        self.d_model = 128
        self.d_hidden = 384
        self.drop_out_rate = 0.1
        self.num_encoder = 2
        self.seq_len = 593
        self.feedforward_factor = 1
        self.n_heads = 8
        self.embedding = nn.Embedding(self.n_vocab, self.d_model, padding_idx=self.n_vocab - 1)
        self.encoder = conv_mamba_block(self.d_model, self.d_hidden, self.drop_out_rate, self.feedforward_factor, self.n_heads, self.seq_len)
        self.encoders = nn.ModuleList([
            copy.deepcopy(self.encoder)
            for _ in range(self.num_encoder)])
        self.classification = classification(self.seq_len, self.d_model, self.d_hidden)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
        
    def forward(self, x):
        x = self.embedding(x)
        for encoder in self.encoders:
            x = encoder(x)
        x = x.permute(0, 2, 1)
        x = self.classification(x)
        
        return x

    
