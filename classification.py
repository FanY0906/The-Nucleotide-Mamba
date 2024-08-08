import torch
import torch.nn as nn

class swish(nn.Module):
    def __init__(self):
        super(swish, self).__init__()

    def forward(self, x):
        return (x * torch.exp(x)) / (torch.exp(x) + 1)


class classification(torch.nn.Module):
    def __init__(self, seq_len, d_model, d_hidden):
        super(classification, self).__init__()
        self.pool = nn.MaxPool1d(seq_len, 1)
        self.d_model = d_model
        self.fc1 = nn.Linear(in_features=d_model, out_features=d_hidden)
        self.bn1 = nn.BatchNorm1d(num_features=d_hidden)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(in_features=d_hidden, out_features=2)
        self.bn2 = nn.BatchNorm1d(num_features=2)
        self.out_act = nn.Sigmoid()

    def forward(self, x):
        x = self.pool(x)
        x = x.contiguous().view(-1, self.d_model)
        x = self.fc1(x)
        x = self.gelu(self.bn1(x))
        x = self.fc2(x)
        x = self.out_act(self.bn2(x))
        return x
    
    
