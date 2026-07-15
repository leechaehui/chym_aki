"""ABMIL — 슬라이드 수준 다중 구조 지표 회귀."""
import torch
import torch.nn as nn

HE_TARGETS = ["fibrosisRatio", "atrophyRatio",
               "tubularInjury", "inflammation", "artHyalinosis"]
MT_TARGETS = ["fibrosisRatio", "atrophyRatio", "artHyalinosis"]

TARGETS   = HE_TARGETS
N_TARGETS = len(HE_TARGETS)


class GatedAttention(nn.Module):
    def __init__(self, in_dim: int = 1024, hidden: int = 256):
        super().__init__()
        self.V = nn.Linear(in_dim, hidden)
        self.U = nn.Linear(in_dim, hidden)
        self.w = nn.Linear(hidden, 1, bias=False)

    def forward(self, H: torch.Tensor):
        a = torch.tanh(self.V(H)) * torch.sigmoid(self.U(H))
        a = self.w(a)
        a = torch.softmax(a, dim=0)
        z = (a * H).sum(dim=0)
        return z, a.squeeze(1)


class ABMIL(nn.Module):
    """
    입력: [N, in_dim] 패치 피처
    출력: (preds [n_targets], attn [N])
    """
    def __init__(self, in_dim: int = 1024, hidden: int = 256,
                 dropout: float = 0.25, n_targets: int = None):
        super().__init__()
        if n_targets is None:
            n_targets = N_TARGETS
        self.feat_proj = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.LayerNorm(in_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.attention = GatedAttention(in_dim, hidden)
        self.struct_head = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, n_targets),
            nn.Sigmoid(),
        )

    def forward(self, H: torch.Tensor):
        H = self.feat_proj(H)
        z, attn = self.attention(H)
        preds = self.struct_head(z)
        return preds, attn
