import sys
import torch
import torch.nn as nn

# Patch the model before loading the script
import mil.model

class GatedAttentionMIL(nn.Module):
    def __init__(self, in_dim=2048, dim=256, att=128, dropout=0.25):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(in_dim, dim), nn.ReLU(), nn.Dropout(dropout))
        self.V = nn.Linear(dim, att)
        self.U = nn.Linear(dim, att)
        self.w = nn.Linear(att, 1)

    def forward(self, patches):
        h = self.proj(patches)
        logit = self.w(torch.tanh(self.V(h)) * torch.sigmoid(self.U(h)))
        a = torch.softmax(logit, dim=0)
        z = (a * h).sum(0)
        return z, a.squeeze(-1)

mil.model.GatedAttentionMIL = GatedAttentionMIL

from mil.diag_fusion_collapse import main

if __name__ == '__main__':
    sys.argv = ['diag_fusion_collapse.py', '--epochs', '2', '--samples', '1']
    main()
