import torch
from mil.model import StainAwareMIL

def test_gradient_flow():
    model = StainAwareMIL(in_dim=128, dim=64, att=32, fusion_att=32, silver_mode="off")
    model.train()
    
    # Dummy bag with different number of patches
    bag = {
        "HE": torch.randn(5000, 128, requires_grad=True),
        "PAS": torch.randn(1000, 128, requires_grad=True),
        "MT": torch.randn(1000, 128, requires_grad=True),
    }
    
    out = model(bag)
    
    # Fake loss
    loss = out["immune"].sum() + out["chronic"].sum()
    loss.backward()
    
    print("--- Gradient Check ---")
    print("HE proj grad:", model.encoders["HE"].proj[0].weight.grad.abs().mean().item())
    print("PAS proj grad:", model.encoders["PAS"].proj[0].weight.grad.abs().mean().item())
    print("MT proj grad:", model.encoders["MT"].proj[0].weight.grad.abs().mean().item())
    
    print("\n--- Stain Contrib (Fusion Attention) ---")
    print(out["stain_contrib"])

if __name__ == "__main__":
    test_gradient_flow()
