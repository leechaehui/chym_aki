import torch
import sys
import numpy as np
sys.path.insert(0, "c:/team/chym_aki/Pathology_model")
from mil.cdss_v5_viz_heatmap40 import load_model
from mil.train import load_bags
import pandas as pd

device="cpu"
m = load_model("vfind40", 2, 768, device)
idx40 = pd.read_csv("C:/team/chym_aki/data/embeddings/ctranspath/index.csv")
idx40 = idx40[idx40.magnification == "40"]
bags = load_bags(["30-11163"], idx40, keep={"HE", "PAS", "MT"})

mats = [torch.from_numpy(bags["30-11163"][s]) for s in ["HE", "PAS", "MT"]]
x_base = torch.cat(mats, dim=0)
h = m.proj(x_base)
a = (torch.softmax(m.attn_ais(h), 0) + torch.softmax(m.attn_cds(h), 0) + torch.softmax(m.attn_ins(h), 0)) / 3.0

lengths = [len(m) for m in mats]
a_split = torch.split(a, lengths, dim=0)

for s, attn_t in zip(["HE", "PAS", "MT"], a_split):
    attn = attn_t.detach().numpy().reshape(-1)
    vmin, vmax = np.percentile(attn, 5), np.percentile(attn, 99)
    print(f"[{s}] N={len(attn)}")
    print(f"  min={attn.min():.6f}, max={attn.max():.6f}")
    print(f"  5th={vmin:.6f}, 99th={vmax:.6f}, 99.9th={np.percentile(attn, 99.9):.6f}")
