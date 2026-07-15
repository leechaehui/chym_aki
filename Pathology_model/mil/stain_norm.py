"""
Stain 정규화 (순수 numpy/cv2/skimage; torch·SPAMS 불필요)

제공:
 - ReinhardNormalizer : LAB 평균/표준편차 매칭 (stain-agnostic, 모든 stain 적용 가능)
 - MacenkoNormalizer  : OD 공간 stain 벡터 분리 후 reference에 매칭 (H&E류 2-stain 가정)
 - nmi(img)           : Normalized Median Intensity (stain 일관성 지표용)

모든 입출력은 RGB uint8 (H,W,3).
"""
import numpy as np

# ---------- 공통 OD 변환 ----------
def rgb_to_od(I, Io=240):
    I = I.astype(np.float64)
    I = np.maximum(I, 1.0)
    return -np.log(I / Io)


def od_to_rgb(OD, Io=240):
    rgb = Io * np.exp(-OD)
    return np.clip(rgb, 0, 255).astype(np.uint8)


def _tissue_mask_od(OD, beta=0.15):
    """OD가 너무 낮은(배경/투명) 픽셀 제외 마스크."""
    return ~np.any(OD < beta, axis=1)


# ---------- Reinhard ----------
class ReinhardNormalizer:
    """LAB 색공간에서 채널별 평균/표준편차를 reference에 맞춤."""

    def __init__(self):
        self.t_mean = None
        self.t_std = None

    @staticmethod
    def _rgb2lab(rgb):
        import cv2
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float64)
        return lab

    @staticmethod
    def _lab2rgb(lab):
        import cv2
        lab = np.clip(lab, 0, 255).astype(np.uint8)
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    def _stats(self, rgb, mask=None):
        lab = self._rgb2lab(rgb)
        flat = lab.reshape(-1, 3)
        if mask is not None:
            flat = flat[mask.reshape(-1)]
        return flat.mean(0), flat.std(0) + 1e-6

    def fit(self, target_rgb, mask=None):
        self.t_mean, self.t_std = self._stats(target_rgb, mask)
        return self

    def transform(self, rgb, mask=None):
        lab = self._rgb2lab(rgb)                      # LAB 변환 1회만(기존 2회→1회)
        flat = lab.reshape(-1, 3)
        s_mean = flat.mean(0); s_std = flat.std(0) + 1e-6
        out = (lab - s_mean) / s_std * self.t_std + self.t_mean
        return self._lab2rgb(out)


# ---------- Macenko ----------
class MacenkoNormalizer:
    """
    Macenko et al. 2009. OD 공간에서 SVD로 stain 벡터(3x2)와 최대 농도를 추정,
    source를 target의 stain 행렬·농도 스케일에 맞춰 재구성.
    """

    def __init__(self, Io=240, beta=0.15, alpha=1, sample=20000, seed=0):
        self.Io = Io
        self.beta = beta
        self.alpha = alpha
        self.sample = sample           # stain 추정용 tissue 픽셀 서브샘플 수
        self.rng = np.random.default_rng(seed)
        self.t_HE = None
        self.t_maxC = None

    def _stain_matrix(self, ODhat):
        """tissue OD 서브샘플로 stain 벡터(3x2) 추정."""
        _, V = np.linalg.eigh(np.cov(ODhat.T))
        V = V[:, [2, 1]]
        if V[0, 0] < 0:
            V[:, 0] *= -1
        if V[0, 1] < 0:
            V[:, 1] *= -1
        phi = np.arctan2(*(ODhat @ V).T[::-1])
        minPhi, maxPhi = np.percentile(phi, [self.alpha, 100 - self.alpha])
        v1 = V @ np.array([np.cos(minPhi), np.sin(minPhi)])
        v2 = V @ np.array([np.cos(maxPhi), np.sin(maxPhi)])
        return (np.array([v1, v2]).T if v1[0] > v2[0] else np.array([v2, v1]).T).astype(np.float32)

    def _fit_od(self, rgb):
        OD = rgb_to_od(rgb, self.Io).reshape(-1, 3).astype(np.float32)
        tissue = OD[_tissue_mask_od(OD, self.beta)]
        if tissue.shape[0] < 10:
            raise ValueError("조직 픽셀 부족(거의 배경 타일)")
        sub = tissue if tissue.shape[0] <= self.sample else \
            tissue[self.rng.integers(0, tissue.shape[0], self.sample)]
        HE = self._stain_matrix(sub)                       # 추정=서브샘플
        C = np.linalg.pinv(HE) @ OD.T                       # 재구성 농도=전체(matmul, BLAS)
        maxC = np.percentile(C, 99, axis=1)
        return HE, C, maxC

    def fit(self, target_rgb):
        HE, _, maxC = self._fit_od(target_rgb)
        self.t_HE, self.t_maxC = HE, maxC
        return self

    def transform(self, rgb):
        h, w = rgb.shape[:2]
        _, C, maxC = self._fit_od(rgb)
        C *= (self.t_maxC / np.maximum(maxC, 1e-6))[:, None]
        OD_norm = (self.t_HE @ C).T
        return od_to_rgb(OD_norm.reshape(h, w, 3), self.Io)


# ---------- stain 일관성 지표 ----------
def nmi(rgb, Io=240, beta=0.15):
    """Normalized Median Intensity: 조직 픽셀 OD 합의 중앙값. 슬라이드간 CV가 낮을수록 일관적."""
    OD = rgb_to_od(rgb, Io).reshape(-1, 3)
    m = _tissue_mask_od(OD, beta)
    if m.sum() == 0:
        return np.nan
    return float(np.median(OD[m].sum(1)))
