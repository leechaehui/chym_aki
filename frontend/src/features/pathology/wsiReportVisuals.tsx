import { useState } from "react";
import type { WsiStain } from "@/types/wsi";
import { wsiService } from "@/services/wsiService";
import { useAuthenticatedImage } from "@/hooks/useAuthenticatedImage";

/** WsiAttnOverlay 와 협진 회신에 실린 요약 오버레이 모두를 받는 최소 형태. */
export interface OverlayLite {
  cx: number;
  cy: number;
  r: number;
  weight: number;
  px?: number;
  py?: number;
  psize?: number;
}

/**
 * 대표 병변 ROI 크롭 URL — 최상위 attention 오버레이(가중치 max) 영역.
 * 정확 좌표(px/py/psize>0)가 있을 때만. 없으면 null(가짜 위치 표시 금지).
 */
export function roiCropUrl(stain: WsiStain, slideId: string, overlays: OverlayLite[]): string | null {
  if (!overlays?.length) return null;
  const top = overlays.reduce((a, b) => (b.weight > a.weight ? b : a));
  if (!top.psize || top.psize <= 0 || top.px === undefined || top.py === undefined) return null;
  const crop = top.psize * 3; // 패치 주변 문맥 포함(3배)
  const x = Math.max(0, top.px + top.psize / 2 - crop / 2);
  const y = Math.max(0, top.py + top.psize / 2 - crop / 2);
  return wsiService.patchUrl(stain, slideId, x, y, crop, crop);
}

/**
 * Attention Heatmap — 슬라이드 썸네일 위에 attention 오버레이(가중치)를 점으로 렌더.
 * cy 는 슬라이드 '너비' 기준 정규화라, 썸네일 natural 종횡비로 높이 비율을 환산해 배치.
 */
export function AttentionHeatmap({ label, slideId, overlays }: { label: string; slideId?: string; overlays: OverlayLite[] }) {
  const src = useAuthenticatedImage(slideId ? wsiService.thumbnailUrl(slideId) : null);
  const [aspect, setAspect] = useState(1); // naturalWidth / naturalHeight
  return (
    <figure className="flex flex-col gap-1">
      <figcaption className="text-[10px] font-medium text-muted-foreground">{label}</figcaption>
      {slideId && src ? (
        <div className="relative w-full overflow-hidden rounded border border-border bg-black/5">
          <img
            src={src}
            alt={`${label} attention heatmap`}
            className="block w-full"
            onLoad={(e) => { const t = e.currentTarget; if (t.naturalHeight) setAspect(t.naturalWidth / t.naturalHeight); }}
          />
          {overlays.map((o, i) => (
            <span
              key={i}
              className="pointer-events-none absolute rounded-full"
              style={{
                left: `${o.cx * 100}%`,
                top: `${o.cy * aspect * 100}%`,
                width: `${o.r * 2 * 100}%`,
                aspectRatio: "1",
                transform: "translate(-50%,-50%)",
                background: `rgba(239,68,68,${Math.min(0.75, 0.12 + o.weight * 0.63)})`,
              }}
            />
          ))}
        </div>
      ) : (
        <div className="flex aspect-square w-full items-center justify-center rounded border border-dashed border-border bg-muted/20 text-[10px] text-muted-foreground">
          {slideId ? "불러오는 중…" : "미분석"}
        </div>
      )}
    </figure>
  );
}

/** 대표 병변 ROI 썸네일 — 인증 필요 이미지라 fetch+blob(useAuthenticatedImage) 로 로드. */
export function RoiThumbnail({ label, url, analyzed }: { label: string; url: string | null; analyzed: boolean }) {
  const src = useAuthenticatedImage(url);
  return (
    <figure className="flex flex-col gap-1">
      <figcaption className="text-[10px] font-medium text-muted-foreground">{label} ROI</figcaption>
      {url && src ? (
        <img src={src} alt={`${label} 대표 병변 ROI`} className="aspect-square w-full rounded border border-border bg-muted/30 object-cover" />
      ) : (
        <div className="flex aspect-square w-full items-center justify-center rounded border border-dashed border-border bg-muted/20 px-1 text-center text-[10px] text-muted-foreground">
          {url ? "불러오는 중…" : analyzed ? "좌표 없음" : "미분석"}
        </div>
      )}
    </figure>
  );
}
