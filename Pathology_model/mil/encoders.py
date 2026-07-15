"""
Foundation Encoder Registry (graceful availability)

정책: 시작 전 각 모델 다운로드 가능 여부를 확인. 접근 불가(토큰/권한/가중치 없음)
모델은 자동 제외하고 가능한 encoder만 사용. 모델 접근 불가로 전체 실험 중단 금지.

우선순위: resnet50 > ctranspath > uni > virchow

각 encoder: build -> (model, embed_dim, preprocess(resize+normalize)) , 모두 frozen.
"""
import os

import torch.nn as nn

# 우선순위 순서 (UNI/Virchow는 사용 안 함 — 제거됨)
PRIORITY = ["resnet50", "ctranspath", "dinov2"]


class _ConvStem(nn.Module):
    """CTransPath patch_embed: 3->embed//8->embed//4->embed conv stem (총 stride4).
    timm 1.0 swin 호환: output_fmt='NHWC', grid_size 제공."""

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96,
                 norm_layer=None, flatten=True, output_fmt="NHWC", bias=True,
                 strict_img_size=True, dynamic_img_pad=False, **kwargs):
        super().__init__()
        img_size = (img_size, img_size) if isinstance(img_size, int) else img_size
        self.img_size = img_size
        self.patch_size = (patch_size, patch_size)
        self.grid_size = (img_size[0] // patch_size, img_size[1] // patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = False
        self.output_fmt = output_fmt
        stem, in_d, out_d = [], in_chans, embed_dim // 8
        for _ in range(2):
            stem += [nn.Conv2d(in_d, out_d, 3, 2, 1, bias=False),
                     nn.BatchNorm2d(out_d), nn.ReLU(inplace=True)]
            in_d, out_d = out_d, out_d * 2
        stem += [nn.Conv2d(in_d, embed_dim, 1)]
        self.proj = nn.Sequential(*stem)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x).permute(0, 2, 3, 1)   # NCHW -> NHWC
        return self.norm(x)

# timm hf-hub id (모두 비gated 공개)
HF_TIMM_IDS = {
    "ctranspath": "1aurent/swin_tiny_patch4_window7_224.CTransPath",
    "dinov2": "timm/vit_base_patch14_dinov2.lvd142m",
}
EMBED_DIM = {"resnet50": 2048, "ctranspath": 768, "dinov2": 768}


def _hf_token():
    return (os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
            or _cached_token())


def _cached_token():
    try:
        from huggingface_hub import HfFolder
        return HfFolder.get_token()
    except Exception:
        return None


def check_availability(name):
    """(available, reason). 메타데이터가 아니라 '실제 가중치 파일 접근'으로 검증."""
    if name == "resnet50":
        try:
            import torchvision  # noqa
            return True, "torchvision 내장(ImageNet)"
        except Exception as e:
            return False, f"torchvision 없음: {e}"

    if name in HF_TIMM_IDS:
        repo = HF_TIMM_IDS[name]
        token = _hf_token()
        # config.json 실제 다운로드 시도(gated면 GatedRepoError) -> 메타데이터 false-positive 방지
        try:
            from huggingface_hub import hf_hub_download
            hf_hub_download(repo, "config.json", token=token)
            return True, f"가중치 접근 가능: {repo}"
        except Exception as e:
            et = type(e).__name__
            hint = " (HF_TOKEN+접근승인 필요)" if et == "GatedRepoError" else ""
            return False, f"{repo} 접근불가 ({et}){hint} -> 제외"

    return False, "알 수 없는 encoder"


def available_encoders():
    """우선순위 순으로 사용 가능한 encoder 이름 리스트."""
    out = []
    for n in PRIORITY:
        ok, _ = check_availability(n)
        if ok:
            out.append(n)
    return out


def build_encoder(name, device, img_size=224):
    """frozen encoder + embed_dim + preprocess(tensor[B,3,H,W] uint8->정규화) 반환.
    img_size: 인코더 입력 해상도(기본 224). ctranspath는 img_size로 재인스턴스화(Swin은 relative
    position bias라 224 가중치 호환). 512 등 고해상 입력 시 세포 단위 디테일 보존."""
    import torch
    import torch.nn.functional as F

    def make_preprocess(mean, std, size=224):
        m = torch.tensor(mean, device=device).view(1, 3, 1, 1)
        s = torch.tensor(std, device=device).view(1, 3, 1, 1)

        def pp(uint8_bhwc):  # numpy (B,H,W,3) uint8
            t = torch.from_numpy(uint8_bhwc).to(device).permute(0, 3, 1, 2).float().div_(255)
            t = F.interpolate(t, size=size, mode="bilinear", align_corners=False)
            return (t - m) / s
        return pp

    if name == "resnet50":
        from torchvision.models import resnet50, ResNet50_Weights
        m = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        m.fc = torch.nn.Identity()
        pp = make_preprocess([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], size=img_size)
        model = m
    elif name == "ctranspath":
        # CTransPath = (repo config의 swin 구조) + 커스텀 ConvStem patch_embed.
        # hf-hub 경로가 올바른 downsample 구조를 만들고, embed_layer만 ConvStem으로 override.
        import timm
        model = timm.create_model(f"hf-hub:{HF_TIMM_IDS['ctranspath']}", pretrained=True,
                                  num_classes=0, embed_layer=_ConvStem, img_size=img_size)
        pp = make_preprocess([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], size=img_size)
    elif name == "dinov2":
        # DINOv2 ViT-B/14 (lvd142m). patch14 -> 224 입력엔 dynamic_img_size로 pos-embed 보간.
        import timm
        model = timm.create_model(f"hf-hub:{HF_TIMM_IDS['dinov2']}", pretrained=True,
                                  num_classes=0, dynamic_img_size=True)
        pp = make_preprocess([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], size=img_size)
    else:
        raise ValueError(name)

    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    return model, EMBED_DIM[name], pp


if __name__ == "__main__":
    print("=== Encoder 가용성 점검 (우선순위순) ===")
    avail = []
    for n in PRIORITY:
        ok, reason = check_availability(n)
        print(f"  [{'O' if ok else 'X'}] {n:11} {reason}")
        if ok:
            avail.append(n)
    print(f"\n사용 가능 encoder: {avail}")
