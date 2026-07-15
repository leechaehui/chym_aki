import { useEffect, useRef, useState } from "react";
import OpenSeadragon from "openseadragon";
import { getToken } from "@/services/http";
import { Download, Eye, EyeOff, Home, Layers, Minus, Plus, Square, SquareDashed, Settings2, X, ZoomIn } from "lucide-react";
import type { WsiAttnOverlay, WsiMetric, WsiStain, WsiTissue } from "@/types/wsi";
import { wsiService } from "@/services/wsiService";
import { cn } from "@/lib/cn";
import { useAuthenticatedImage } from "@/hooks/useAuthenticatedImage";

type AnnotPoint = { x: number; y: number };
type Annotation  = { type: "pen" | "region" | "ruler"; pts: AnnotPoint[]; color: string };

interface Props {
  stain:          WsiStain;
  slideId?:       string;
  dziUrl?:        string;
  attnOverlays?:  WsiAttnOverlay[];
  metrics?:       WsiMetric[];
  tool?:          string;
  drawColor?:     string;
  tissues?:       WsiTissue[];
  selectedTissue?: number;
  slide_w?:       number;
  slide_h?:       number;
}

const LAYER_COLORS: Record<string, string> = {
  fibrosisRatio: "96,165,250",
  atrophyRatio:  "251,191,36",
  tubularInjury: "248,113,113",
  inflammation:  "74,222,128",
};
const GLOBAL_COLOR = "251,191,36";

/** weight(0~1) → jet(무지개) 컬러 "r,g,b": 낮음=파랑 → 초록 → 노랑 → 높음=빨강.
 *  attention 을 강도별 무지개로 표시(가장 중요=빨강). blocky/smooth 공통(색 일치). 시각화 전용. */
function jetColor(t: number): string {
  const x = Math.max(0, Math.min(1, t));
  const c = (v: number) => Math.round(255 * Math.max(0, Math.min(1, v)));
  return `${c(1.5 - Math.abs(4 * x - 3))},${c(1.5 - Math.abs(4 * x - 2))},${c(1.5 - Math.abs(4 * x - 1))}`;
}

const TARGET_LABEL: Record<string, string> = {
  fibrosisRatio: "간질 섬유화",
  atrophyRatio:  "세뇨관 위축",
  tubularInjury: "세뇨관 손상",
  inflammation:  "간질 염증",
};

const STAIN_LABEL: Record<WsiStain, string> = { HE: "H&E", MT: "Masson Trichrome", PAS: "PAS" };

function buildTooltipHtml(
  o: WsiAttnOverlay,
  metrics: WsiMetric[],
): string {
  const pct = Math.round(o.weight * 100);
  let rows = "";

  if (o.contrib && Object.keys(o.contrib).length > 0) {
    const sorted = Object.entries(o.contrib).sort(([, a], [, b]) => b - a).slice(0, 3);
    const metricMap = Object.fromEntries(metrics.map((m) => [m.key, m]));
    rows = sorted.map(([key, c]) => {
      const m       = metricMap[key];
      const label   = TARGET_LABEL[key] ?? key;
      const valStr  = m ? `${m.value}${m.unit}` : "";
      // 막대는 텍스트와 같은 값(지표 자체의 심각도)을 나타내야 함 — contrib(기여도)와 스케일이 다르다.
      const barW    = m ? Math.round(m.value) : Math.round(c * 100);
      const barColor = c >= 0.7 ? "rgba(255,255,255,0.75)" : c >= 0.4 ? "rgba(255,255,255,0.5)" : "rgba(255,255,255,0.28)";
      return `
        <div style="margin-bottom:5px;">
          <div style="display:flex;justify-content:space-between;font-size:11px;margin-bottom:2px;">
            <span style="color:rgba(255,255,255,0.70);">${label}</span>
            <span style="color:rgba(255,255,255,0.75);font-weight:600;">${valStr}</span>
          </div>
          <div style="height:4px;background:rgba(255,255,255,0.12);border-radius:2px;">
            <div style="height:4px;width:${barW}%;background:${barColor};border-radius:2px;"></div>
          </div>
        </div>`;
    }).join("");
  } else {
    const sorted = [...metrics].sort((a, b) => b.value - a.value).slice(0, 3);
    rows = sorted.map((m) => {
      const barW = Math.round(m.value);
      const rgb  = LAYER_COLORS[m.key] ?? GLOBAL_COLOR;
      return `
        <div style="margin-bottom:5px;">
          <div style="display:flex;justify-content:space-between;font-size:11px;margin-bottom:2px;">
            <span style="color:rgba(255,255,255,0.85);">${m.label}</span>
            <span style="color:rgba(255,255,255,0.75);font-weight:600;">${m.value}${m.unit}</span>
          </div>
          <div style="height:4px;background:rgba(255,255,255,0.12);border-radius:2px;">
            <div style="height:4px;width:${barW}%;background:rgba(${rgb},0.8);border-radius:2px;"></div>
          </div>
        </div>`;
    }).join("");
  }

  const titleText  = o.weight >= 0.75 ? "고주목 병변 구역" : o.weight >= 0.45 ? "중간 주목 구역" : "보조 참조 구역";
  const titleAlpha = o.weight >= 0.75 ? 1.0 : o.weight >= 0.45 ? 0.82 : 0.65;
  return `
    <div style="font-size:11px;font-weight:600;color:rgba(255,255,255,${titleAlpha});margin-bottom:6px;">${titleText}</div>
    <div style="font-size:10px;color:rgba(255,255,255,0.45);margin-bottom:7px;">지표별 기여도</div>
    ${rows}
    <div style="font-size:10px;color:rgba(255,255,255,0.35);border-top:1px solid rgba(255,255,255,0.08);padding-top:5px;margin-top:2px;">
      AI 주목도 <span style="color:rgba(255,255,255,0.85);font-weight:600;">${pct}%</span>
    </div>
  `;
}

export function WSIViewer({
  stain,
  slideId,
  dziUrl,
  attnOverlays = [],
  metrics = [],
  tool = "pointer",
  drawColor = "#f97316",
  tissues = [],
  selectedTissue = 0,
  slide_w = 0,
  slide_h = 0,
}: Props) {
  const containerRef   = useRef<HTMLDivElement>(null);
  const viewerRef      = useRef<OpenSeadragon.Viewer | null>(null);
  const annotationsRef = useRef<Annotation[]>([]);
  const [zoom, setZoom]               = useState(1);
  const [error, setError]             = useState<string | null>(null);
  const [loading, setLoading]         = useState(false);
  
  // 시각화 설정 상태
  const [showHeatmap, setShowHeatmap] = useState(true);
  const [showBoxes,   setShowBoxes]   = useState(true);
  
  // 개발자/디버그 모드 상태
  const [devMode, setDevMode]         = useState(false);
  const [devBlocky, setDevBlocky]     = useState(false);
  const [devGrid, setDevGrid]         = useState(false);

  // PAS 전용: 종합 주목도 vs 지표별(contrib) 주목도 히트맵 전환. "all"=종합.
  const [attnTarget, setAttnTarget]   = useState<string>("all");

  // 클릭된 패치 상세 보기 상태
  const [clickedPatch, setClickedPatch] = useState<WsiAttnOverlay | null>(null);
  // <img src>는 Authorization 헤더를 못 실어서 fetch+blob 으로 인증된 패치 이미지를 받는다.
  const clickedPatchUrl = clickedPatch?.px !== undefined && clickedPatch?.py !== undefined && clickedPatch?.psize !== undefined
    ? wsiService.patchUrl(stain, slideId ?? "", clickedPatch.px, clickedPatch.py, clickedPatch.psize, clickedPatch.psize)
    : null;
  const clickedPatchSrc = useAuthenticatedImage(clickedPatchUrl);

  // ROI 좌표 변환 헬퍼
  const getRoiCoords = (o: WsiAttnOverlay) => {
    if (tissues.length === 0 || !slide_w || selectedTissue === undefined) {
      return { cx: o.cx, cy: o.cy, r: o.r };
    }
    const t = tissues[selectedTissue];
    if (!t) return { cx: o.cx, cy: o.cy, r: o.r };

    // 정규화 좌표 -> 전체 원본 픽셀 좌표 역산
    const x = o.cx * slide_w;
    const y = o.cy * slide_w; // mapper.py와 동일하게 cx, cy 모두 slide_w 기준 정규화됨
    const size = o.r * 2 * slide_w;

    // ROI 기준 뷰포트 좌표로 전환 (ROI의 가로 너비 t.w 기준 정규화)
    const cx_roi = (x - t.x) / t.w;
    const cy_roi = (y - t.y) / t.w;
    const r_roi = (size / 2) / t.w;

    return { cx: cx_roi, cy: cy_roi, r: r_roi };
  };

  // ── OSD 초기화 ────────────────────────────────────────────────────────────
  useEffect(() => {
    if (!containerRef.current || !dziUrl) return;
    viewerRef.current?.destroy();
    viewerRef.current = null;
    setError(null);
    setLoading(true);
    annotationsRef.current = [];
    setClickedPatch(null);

    const viewer = OpenSeadragon({
      element:               containerRef.current,
      tileSources:           dziUrl,
      loadTilesWithAjax:     true,
      ajaxHeaders:           { Authorization: `Bearer ${getToken() ?? ""}` },
      showNavigator:         true,
      navigatorPosition:     "TOP_RIGHT",
      navigatorSizeRatio:    0.18,
      showNavigationControl: false,
      gestureSettingsMouse:  { clickToZoom: false, dblClickToZoom: true, scrollToZoom: true },
      minZoomImageRatio:     0.4,
      maxZoomPixelRatio:     8,
      visibilityRatio:       0.3,
      animationTime:         0.4,
      blendTime:             0.1,
      constrainDuringPan:    false,
      crossOriginPolicy:     "Anonymous",
    });

    viewer.addHandler("open", () => {
      setLoading(false);
      setZoom(Math.round(viewer.viewport.getZoom() * 100));
    });
    viewer.addHandler("open-failed", () => {
      setLoading(false);
      setError("슬라이드를 불러올 수 없습니다 — 백엔드 로그를 확인하세요");
      viewer.destroy();
    });
    viewer.addHandler("zoom", () => {
      const z = viewer.viewport?.getZoom();
      if (z) setZoom(Math.round(z * 100));
    });

    viewerRef.current = viewer;
    return () => { viewer.destroy(); viewerRef.current = null; };
  }, [dziUrl]);

  // 지표별 주목도(contrib) 히트맵 전환은 PAS(내 Task-Attention MIL 모델) 전용.
  // 다른 stain/모델은 종합 주목도(weight)만 노출한다.
  const contribKeys = new Set<string>();
  attnOverlays.forEach((o) => { if (o.contrib) Object.keys(o.contrib).forEach((k) => contribKeys.add(k)); });
  const targetOptions = metrics.filter((m) => contribKeys.has(m.key));
  const perTargetAvailable = stain === "PAS" && targetOptions.length > 0;
  const heatTarget = perTargetAvailable ? attnTarget : "all";

  // ── AI 주목 오버레이 (canvas, pointer-events:none) ────────────────────────
  useEffect(() => {
    const viewer = viewerRef.current;
    const container = containerRef.current;
    if (!viewer || !container || !showHeatmap || attnOverlays.length === 0) return;

    const canvas = document.createElement("canvas");
    canvas.style.cssText = "position:absolute;top:0;left:0;pointer-events:none;z-index:5;";
    canvas.width  = container.clientWidth;
    canvas.height = container.clientHeight;
    container.appendChild(canvas);
    const ctx = canvas.getContext("2d")!;

    const tooltip = document.createElement("div");
    tooltip.style.cssText = [
      "position:absolute;z-index:999;",
      "background:rgba(10,12,18,0.94);border:1px solid rgba(255,255,255,0.18);",
      "border-radius:8px;padding:8px 11px;pointer-events:none;display:none;",
      "max-width:210px;box-shadow:0 4px 20px rgba(0,0,0,0.7);backdrop-filter:blur(4px);",
    ].join("");
    container.appendChild(tooltip);

    function getPxR(o: WsiAttnOverlay, customR?: number): number {
      if (!viewer!.viewport) return 7;
      const rVal = customR !== undefined ? customR : o.r;
      return Math.max(7, viewer!.viewport.deltaPixelsFromPointsNoRotate(
        new OpenSeadragon.Point(rVal * 0.5, 0),
      ).x);
    }

    function draw() {
      if (!viewer!.viewport) return;
      const w = container!.clientWidth;
      const h = container!.clientHeight;
      if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
      ctx.clearRect(0, 0, w, h);

      attnOverlays.forEach((o) => {
        // 종합 모드는 weight(태스크 평균), 지표별 모드는 해당 지표의 contrib(태스크 attention, p99 정규화).
        const weight = heatTarget === "all" ? o.weight : (o.contrib?.[heatTarget] ?? 0);
        // 지표별 모드는 컷을 낮춰(0.02) 옅은 기여도 패치까지 표시. 종합은 0.05 유지.
        if (weight < (heatTarget === "all" ? 0.05 : 0.02)) return;
        const rgb = GLOBAL_COLOR;
        const heatRgb = jetColor(weight);   // 강도별 무지개(최고=빨강). blocky/smooth 공통.

        const rCoords = getRoiCoords(o);
        const pt  = viewer!.viewport.viewportToViewerElementCoordinates(
          new OpenSeadragon.Point(rCoords.cx, rCoords.cy),
        );
        const rad = getPxR(o, rCoords.r);

        if (devMode && devBlocky) {
          // 1) 디버그 모드 - Blocky Attention (사각형 그리드 형태로 꽉 채워 그림)
          const tl = viewer!.viewport.viewportToViewerElementCoordinates(
            new OpenSeadragon.Point(rCoords.cx - rCoords.r, rCoords.cy - rCoords.r),
          );
          const br = viewer!.viewport.viewportToViewerElementCoordinates(
            new OpenSeadragon.Point(rCoords.cx + rCoords.r, rCoords.cy + rCoords.r),
          );
          
          ctx.save();
          ctx.fillStyle = `rgba(${heatRgb}, 0.62)`;   // jet: 색이 강도, alpha 고정(무지개 선명)
          ctx.fillRect(tl.x, tl.y, br.x - tl.x, br.y - tl.y);
          
          if (devGrid) {
            ctx.strokeStyle = `rgba(${rgb}, 0.35)`;
            ctx.lineWidth = 1;
            ctx.strokeRect(tl.x, tl.y, br.x - tl.x, br.y - tl.y);
          }
          ctx.restore();
        } else {
          // 2) 기본 사용자 모드 - Smooth Attention Heatmap (matplotlib 'jet' smooth 처럼).
          // 반지름을 키우고, 중심=hot(빨강)→가장자리=cool(초록/파랑)로 강도별 jet 을 '통과'시켜
          // Gaussian blur 로 채색한 연속 히트필드처럼 보이게 한다(각 blob 이 미니 무지개 halo).
          const R = rad * 2.4;
          ctx.save();
          ctx.beginPath();
          const grad = ctx.createRadialGradient(pt.x, pt.y, 0, pt.x, pt.y, R);
          grad.addColorStop(0,    `rgba(${jetColor(weight)}, 0.85)`);         // 중심: 실제 강도색(최고=빨강)
          grad.addColorStop(0.45, `rgba(${jetColor(weight * 0.6)}, 0.5)`);    // 중간: 한 단계 cool
          grad.addColorStop(1,    `rgba(${jetColor(weight * 0.15)}, 0)`);     // 가장자리: cool→투명
          ctx.fillStyle = grad;
          ctx.arc(pt.x, pt.y, R, 0, 2 * Math.PI);
          ctx.fill();
          ctx.restore();
        }
      });
    }

    function onMove(e: MouseEvent) {
      if (!viewer!.viewport) return;
      const rect = container!.getBoundingClientRect();
      const mx = e.clientX - rect.left;
      const my = e.clientY - rect.top;
      let hit: WsiAttnOverlay | null = null;
      for (const o of attnOverlays) {
        const rCoords = getRoiCoords(o);
        const pt = viewer!.viewport.viewportToViewerElementCoordinates(
          new OpenSeadragon.Point(rCoords.cx, rCoords.cy),
        );
        if (Math.hypot(mx - pt.x, my - pt.y) <= getPxR(o, rCoords.r)) { hit = o; break; }
      }
      if (hit) {
        tooltip.innerHTML = buildTooltipHtml(hit, metrics);
        tooltip.style.left    = `${mx + 12}px`;
        tooltip.style.top     = `${my - 10}px`;
        tooltip.style.display = "block";
      } else {
        tooltip.style.display = "none";
      }
    }
    const onLeave = () => { tooltip.style.display = "none"; };

    container.addEventListener("mousemove",  onMove);
    container.addEventListener("mouseleave", onLeave);
    viewer.addHandler("viewport-change", draw);
    viewer.addHandler("resize",          draw);

    if (viewer.isOpen()) draw();
    else viewer.addOnceHandler("open", draw);

    return () => {
      container.removeEventListener("mousemove",  onMove);
      container.removeEventListener("mouseleave", onLeave);
      viewer.removeHandler("viewport-change", draw);
      viewer.removeHandler("resize",          draw);
      canvas.remove();
      tooltip.remove();
    };
  }, [attnOverlays, showHeatmap, metrics, devMode, devBlocky, devGrid, selectedTissue, tissues, heatTarget]);

  // ── OSD 캔버스 클릭 이벤트 리스너 (패치 클릭 핸들러) ──────────────────────
  useEffect(() => {
    const viewer = viewerRef.current;
    if (!viewer || attnOverlays.length === 0) return;

    function getPxR(o: WsiAttnOverlay, customR?: number): number {
      if (!viewer!.viewport) return 7;
      const rVal = customR !== undefined ? customR : o.r;
      return Math.max(7, viewer!.viewport.deltaPixelsFromPointsNoRotate(
        new OpenSeadragon.Point(rVal * 0.5, 0),
      ).x);
    }

    const onCanvasClick = (event: any) => {
      // 드래그가 아닌 순수 클릭인 경우에만 감지
      if (!event.quick) return;
      if (tool !== "pointer") return;

      const mx = event.position.x;
      const my = event.position.y;

      let hit: WsiAttnOverlay | null = null;
      for (const o of attnOverlays) {
        const rCoords = getRoiCoords(o);
        const pt = viewer.viewport.viewportToViewerElementCoordinates(
          new OpenSeadragon.Point(rCoords.cx, rCoords.cy),
        );
        const rad = getPxR(o, rCoords.r);
        if (Math.hypot(mx - pt.x, my - pt.y) <= rad) {
          hit = o;
          break;
        }
      }
      if (hit) {
        setClickedPatch(hit);
      }
    };

    viewer.addHandler("canvas-click", onCanvasClick);
    return () => {
      viewer.removeHandler("canvas-click", onCanvasClick);
    };
  }, [attnOverlays, tool, selectedTissue, tissues, slide_w]);

  // ── 병변 박스 오버레이 (상위 8개 패치 강조 마커) ───────────────────────────
  useEffect(() => {
    const viewer    = viewerRef.current;
    const container = containerRef.current;
    if (!viewer || !container || !showBoxes || attnOverlays.length === 0) return;

    const canvas = document.createElement("canvas");
    canvas.style.cssText = "position:absolute;top:0;left:0;pointer-events:none;z-index:4;";
    canvas.width  = container.clientWidth;
    canvas.height = container.clientHeight;
    container.appendChild(canvas);
    const ctx = canvas.getContext("2d")!;

    const topOverlays = [...attnOverlays]
      .filter((o) => o.weight >= 0.45)
      .sort((a, b) => b.weight - a.weight)
      .slice(0, 8);

    function draw() {
      if (!viewer!.viewport) return;
      const cw = container!.clientWidth;
      const ch = container!.clientHeight;
      if (canvas.width !== cw || canvas.height !== ch) { canvas.width = cw; canvas.height = ch; }
      ctx.clearRect(0, 0, cw, ch);

      topOverlays.forEach((o) => {
        const weight = o.weight;
        if (weight < 0.1) return;

        // contrib(패치별 병변 기여도)는 슬라이드 전체 지표 비율을 attention 가중치로 나눈 값이라
        // 패치마다 실제로 다른 병변을 가리키는 정보가 아님 — "1등 지표"를 자동으로 골라 라벨을
        // 붙이면 마치 그 패치가 그 병변이라고 판별한 것처럼 오인시킨다. 그래서 항상 전체 기여도만 표기.
        const categoryScore = weight;
        const rgb       = GLOBAL_COLOR;
        const labelText = "예측 기여도";

        const rCoords = getRoiCoords(o);
        const tl = viewer!.viewport.viewportToViewerElementCoordinates(
          new OpenSeadragon.Point(rCoords.cx - rCoords.r, rCoords.cy - rCoords.r),
        );
        const br = viewer!.viewport.viewportToViewerElementCoordinates(
          new OpenSeadragon.Point(rCoords.cx + rCoords.r, rCoords.cy + rCoords.r),
        );

        const MIN_BOX = 24;
        const rawW = br.x - tl.x;
        const rawH = br.y - tl.y;
        const bw = Math.max(rawW, MIN_BOX);
        const bh = Math.max(rawH, MIN_BOX);
        const ctr = viewer!.viewport.viewportToViewerElementCoordinates(
          new OpenSeadragon.Point(rCoords.cx, rCoords.cy),
        );
        const bx = ctr.x - bw / 2;
        const by = ctr.y - bh / 2;

        // 아래 범례(낮음/중간/높음)와 동일한 3단계 표기 — 툴팁 제목 임계값(0.75/0.45)과도 통일.
        const tier = weight >= 0.75 ? 2 : weight >= 0.45 ? 1 : 0;
        const TIER_ALPHA = [0.40, 0.65, 0.90];
        const TIER_WIDTH = [1.5, 2, 2.5];
        const TIER_DASH: number[][] = [[10, 6], [6, 3], []];

        ctx.save();

        ctx.setLineDash(TIER_DASH[tier]);
        ctx.strokeStyle = `rgba(${rgb},${TIER_ALPHA[tier]})`;
        ctx.lineWidth   = TIER_WIDTH[tier];
        ctx.strokeRect(bx + 0.5, by + 0.5, bw - 1, bh - 1);
        ctx.setLineDash([]);

        const corner = Math.min(bw, bh, 16);
        ctx.strokeStyle = `rgba(${rgb},${Math.min(TIER_ALPHA[tier] + 0.3, 1)})`;
        ctx.lineWidth   = 2;
        ctx.beginPath();
        ctx.moveTo(bx, by + corner); ctx.lineTo(bx, by); ctx.lineTo(bx + corner, by);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(bx + bw - corner, by + bh); ctx.lineTo(bx + bw, by + bh); ctx.lineTo(bx + bw, by + bh - corner);
        ctx.stroke();

        const pct  = Math.round(categoryScore * 100);
        const text = `${labelText} ${pct}%`;
        ctx.font = "bold 10px sans-serif";
        const tw    = ctx.measureText(text).width;
        const bpad  = 4;
        const bh2   = 15;
        if (bw > tw + 10 && bh > bh2 + 8) {
          ctx.fillStyle = `rgba(${rgb},0.88)`;
          ctx.fillRect(bx + 1, by + 1, tw + bpad * 2, bh2);
          ctx.fillStyle    = "rgba(0,0,0,0.9)";
          ctx.textBaseline = "middle";
          ctx.fillText(text, bx + 1 + bpad, by + 1 + bh2 / 2);
        }
        ctx.restore();
      });
    }

    viewer.addHandler("viewport-change", draw);
    viewer.addHandler("resize",          draw);
    if (viewer.isOpen()) draw();
    else viewer.addOnceHandler("open", draw);

    return () => {
      viewer.removeHandler("viewport-change", draw);
      viewer.removeHandler("resize",          draw);
      canvas.remove();
    };
  }, [attnOverlays, showBoxes, selectedTissue, tissues]);

  // ── 드로잉 도구 (Ruler / Drawing) ──────────────────────────────────────────
  useEffect(() => {
    const viewer = viewerRef.current;
    const container = containerRef.current;
    if (!viewer || !container) return;

    const isDrawTool = tool !== "pointer";
    viewer.setMouseNavEnabled(!isDrawTool);
    if (!isDrawTool || !slideId) return;

    const isEraser = tool === "eraser";
    const canvas = document.createElement("canvas");
    canvas.style.cssText = `position:absolute;top:0;left:0;pointer-events:auto;z-index:6;cursor:${isEraser ? "cell" : "crosshair"};`;
    canvas.width  = container.clientWidth;
    canvas.height = container.clientHeight;
    container.appendChild(canvas);
    const ctx = canvas.getContext("2d")!;

    let drawing = false;
    let current: Annotation | null = null;

    function toVP(e: MouseEvent): AnnotPoint {
      const rect = container!.getBoundingClientRect();
      const p = viewer!.viewport.viewerElementToViewportCoordinates(
        new OpenSeadragon.Point(e.clientX - rect.left, e.clientY - rect.top),
      );
      return { x: p.x, y: p.y };
    }
    function toSC(p: AnnotPoint) {
      const s = viewer!.viewport.viewportToViewerElementCoordinates(new OpenSeadragon.Point(p.x, p.y));
      return { x: s.x, y: s.y };
    }
    function annCenter(ann: Annotation) {
      if (ann.pts.length === 0) return { x: 0, y: 0 };
      if (ann.pts.length === 1) return ann.pts[0];
      if (ann.type === "pen") {
        const sx = ann.pts.reduce((a, p) => a + p.x, 0) / ann.pts.length;
        const sy = ann.pts.reduce((a, p) => a + p.y, 0) / ann.pts.length;
        return { x: sx, y: sy };
      }
      return { x: (ann.pts[0].x + ann.pts[1].x) / 2, y: (ann.pts[0].y + ann.pts[1].y) / 2 };
    }

    function redraw() {
      if (!viewer!.viewport) return;
      const w = container!.clientWidth; const h = container!.clientHeight;
      if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
      ctx.clearRect(0, 0, w, h);

      [...annotationsRef.current, ...(current ? [current] : [])].forEach((ann) => {
        ctx.save();
        ctx.lineCap = "round"; ctx.lineJoin = "round";
        ctx.lineWidth = 2; ctx.strokeStyle = ann.color; ctx.fillStyle = ann.color;

        if (ann.type === "pen" && ann.pts.length > 1) {
          ctx.beginPath();
          const s0 = toSC(ann.pts[0]); ctx.moveTo(s0.x, s0.y);
          ann.pts.slice(1).forEach((p) => { const s = toSC(p); ctx.lineTo(s.x, s.y); });
          ctx.stroke();

        } else if (ann.type === "region" && ann.pts.length >= 2) {
          const s = toSC(ann.pts[0]); const e = toSC(ann.pts[1]);
          ctx.strokeRect(s.x, s.y, e.x - s.x, e.y - s.y);
          ctx.globalAlpha = 0.12; ctx.fillRect(s.x, s.y, e.x - s.x, e.y - s.y); ctx.globalAlpha = 1;

        } else if (ann.type === "ruler" && ann.pts.length >= 2) {
          const s = toSC(ann.pts[0]); const e = toSC(ann.pts[1]);
          ctx.setLineDash([6, 3]);
          ctx.beginPath(); ctx.moveTo(s.x, s.y); ctx.lineTo(e.x, e.y); ctx.stroke();
          ctx.setLineDash([]);
          [s, e].forEach((p) => { ctx.beginPath(); ctx.arc(p.x, p.y, 4, 0, 2 * Math.PI); ctx.fill(); });
          const dx = ann.pts[1].x - ann.pts[0].x; const dy = ann.pts[1].y - ann.pts[0].y;
          const imgW = viewer!.world.getItemAt(0)?.getContentSize().x ?? 1;
          const distPx = Math.round(Math.sqrt(dx * dx + dy * dy) * imgW);
          const label = `${distPx}px`;
          const mx = (s.x + e.x) / 2; const my = (s.y + e.y) / 2 - 10;
          ctx.font = "bold 11px sans-serif";
          const tw = ctx.measureText(label).width;
          ctx.fillStyle = "rgba(0,0,0,0.75)";
          ctx.fillRect(mx - tw / 2 - 4, my - 9, tw + 8, 16);
          ctx.fillStyle = ann.color; ctx.textAlign = "center";
          ctx.fillText(label, mx, my + 2); ctx.textAlign = "start";
        }
        ctx.restore();
      });
    }

    function onDown(e: MouseEvent) {
      if (e.button !== 0) return;
      if (isEraser) {
        if (!viewer!.viewport) return;
        const rect = container!.getBoundingClientRect();
        const mx = e.clientX - rect.left;
        const my = e.clientY - rect.top;
        let minDist = Infinity; let minIdx = -1;
        annotationsRef.current.forEach((ann, i) => {
          const c = annCenter(ann); const sc = toSC(c);
          const d = Math.hypot(mx - sc.x, my - sc.y);
          if (d < minDist) { minDist = d; minIdx = i; }
        });
        if (minIdx >= 0 && minDist <= 30) {
          annotationsRef.current.splice(minIdx, 1);
          redraw();
        }
        return;
      }
      drawing = true;
      current = { type: tool as Annotation["type"], pts: [toVP(e)], color: drawColor };
    }
    function onMove(e: MouseEvent) {
      if (!drawing || !current) return;
      const vp = toVP(e);
      if (current.type === "pen") current.pts.push(vp);
      else if (current.pts.length < 2) current.pts.push(vp);
      else current.pts[1] = vp;
      redraw();
    }
    function onUp() {
      if (!drawing || !current) return;
      drawing = false;
      if (current.pts.length >= 2) annotationsRef.current.push(current);
      current = null;
      redraw();
    }
    function onWheel(e: WheelEvent) {
      e.preventDefault(); e.stopPropagation();
      if (!viewer!.viewport) return;
      const rect = container!.getBoundingClientRect();
      const refPt = viewer!.viewport.viewerElementToViewportCoordinates(
        new OpenSeadragon.Point(e.clientX - rect.left, e.clientY - rect.top),
      );
      viewer!.viewport.zoomBy(e.deltaY < 0 ? 1.3 : 1 / 1.3, refPt, true);
    }

    canvas.addEventListener("mousedown", onDown);
    canvas.addEventListener("mousemove", onMove);
    canvas.addEventListener("mouseup",   onUp);
    canvas.addEventListener("wheel",     onWheel, { passive: false });
    viewer.addHandler("viewport-change", redraw);

    if (viewer.isOpen()) redraw();
    else viewer.addOnceHandler("open", redraw);

    return () => {
      canvas.removeEventListener("mousedown", onDown);
      canvas.removeEventListener("mousemove", onMove);
      canvas.removeEventListener("mouseup",   onUp);
      canvas.removeEventListener("wheel",     onWheel);
      viewer.removeHandler("viewport-change", redraw);
      canvas.remove();
      if (viewerRef.current) viewerRef.current.setMouseNavEnabled(true);
    };
  }, [tool, slideId, dziUrl, drawColor]);

  // 현재 화면(타일 + AI 오버레이 + 주석)에 겹쳐진 canvas 들을 z-index 순서로 합성해 PNG로 저장.
  function saveAsImage() {
    const container = containerRef.current;
    if (!container) return;
    const canvases = Array.from(container.querySelectorAll("canvas"));
    if (canvases.length === 0) return;

    const out = document.createElement("canvas");
    out.width = container.clientWidth;
    out.height = container.clientHeight;
    const ctx = out.getContext("2d");
    if (!ctx) return;

    canvases
      .map((c) => ({ c, z: parseInt(c.style.zIndex || "0", 10) }))
      .sort((a, b) => a.z - b.z)
      .forEach(({ c }) => ctx.drawImage(c, 0, 0, out.width, out.height));

    out.toBlob((blob) => {
      if (!blob) return;
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `${slideId ?? "wsi"}_${stain}_${Date.now()}.png`;
      a.click();
      URL.revokeObjectURL(url);
    });
  }

  const attnRgb   = GLOBAL_COLOR;
  const attnLabel = heatTarget === "all"
    ? "전체 주목도"
    : `${metrics.find((m) => m.key === heatTarget)?.label ?? heatTarget} 주목도`;

  if (!slideId || !dziUrl) {
    return (
      <div className="flex h-[420px] items-center justify-center rounded-lg border border-border bg-[#15100d] text-sm text-white/40">
        <div className="text-center">
          <Layers className="mx-auto mb-2 size-10 opacity-40" />
          <p>슬라이드를 선택하면 WSI가 표시됩니다</p>
        </div>
      </div>
    );
  }

  return (
    <div className="relative overflow-hidden rounded-lg border border-border bg-[#15100d]" style={{ height: 420 }}>
      {loading && (
        <div className="absolute inset-0 z-20 flex items-center justify-center bg-[#15100d]">
          <div className="text-center text-white/60">
            <div className="mx-auto mb-3 size-8 animate-spin rounded-full border-2 border-white/20 border-t-white/70" />
            <p className="text-xs">조직 타일 로딩 중...</p>
          </div>
        </div>
      )}
      {error && (
        <div className="absolute inset-0 z-20 flex items-center justify-center bg-[#15100d]">
          <div className="text-center text-white/50">
            <Layers className="mx-auto mb-2 size-8 opacity-40" />
            <p className="text-sm">{error}</p>
          </div>
        </div>
      )}

      <div ref={containerRef} className="h-full w-full" />

      {/* 줌 표시 + 도구 배지 */}
      {!error && (
        <div className="pointer-events-none absolute left-2 top-2 z-10 flex items-center gap-1.5 rounded-md bg-black/55 px-2 py-1 text-[11px] font-medium text-white/85">
          {zoom}% · {STAIN_LABEL[stain]}
          {tissues.length > 0 && ` · Tissue ${selectedTissue + 1}`}
          {tool !== "pointer" && (
            <span className="ml-1 rounded bg-white/10 px-1 text-[10px] text-white/60">
              {{ pen: "펜", region: "영역", ruler: "측정", eraser: "지우기" }[tool] ?? tool}
            </span>
          )}
        </div>
      )}

      {/* 현재 화면 이미지로 저장 — 우측 상단은 OSD 네비게이터(미니맵)라 겹치지 않게 하단에 배치. */}
      {!error && (
        <button
          onClick={saveAsImage}
          title="현재 화면을 이미지 파일로 저장"
          className="absolute bottom-2 right-2 z-10 flex items-center gap-1 rounded-md bg-black/55 px-2 py-1 text-[11px] font-medium text-white/85 hover:bg-black/70"
        >
          <Download className="size-3.5" /> 이미지 저장
        </button>
      )}

      {/* 시각화 토글 버튼 패널 */}
      {!error && attnOverlays.length > 0 && (
        <div className="absolute left-2 top-9 z-10 flex flex-col gap-1">
          <button
            onClick={() => setShowHeatmap((v) => !v)}
            className="flex items-center gap-1 rounded-md bg-black/55 px-2 py-1 text-[10px] text-white/80 hover:bg-black/70"
          >
            {showHeatmap ? <Eye className="size-3" /> : <EyeOff className="size-3" />}
            AI 주목 영역 {showHeatmap ? "ON" : "OFF"}
          </button>
          <button
            onClick={() => setShowBoxes((v) => !v)}
            className="flex items-center gap-1 rounded-md bg-black/55 px-2 py-1 text-[10px] text-white/80 hover:bg-black/70"
          >
            {showBoxes ? <Square className="size-3" /> : <SquareDashed className="size-3" />}
            병변 박스 {showBoxes ? "ON" : "OFF"}
          </button>

          {/* 지표별 주목도 히트맵 전환 (PAS · 내 Task-Attention MIL 전용) */}
          {perTargetAvailable && showHeatmap && (
            <select
              value={attnTarget}
              onChange={(e) => setAttnTarget(e.target.value)}
              title="지표별 주목도 히트맵 (PAS 전용)"
              className="rounded-md bg-black/55 px-2 py-1 text-[10px] text-white/80 outline-none hover:bg-black/70"
            >
              <option value="all">종합 주목도</option>
              {targetOptions.map((m) => (
                <option key={m.key} value={m.key}>{m.label} 주목도</option>
              ))}
            </select>
          )}

          {/* 디버그/개발자 모드 기어 */}
          <button
            onClick={() => setDevMode((v) => !v)}
            className={cn(
              "flex items-center gap-1 rounded-md px-2 py-1 text-[10px] font-semibold transition-colors",
              devMode ? "bg-amber-600/90 text-white hover:bg-amber-700" : "bg-black/55 text-white/60 hover:text-white"
            )}
          >
            <Settings2 className="size-3" />
            개발자 모드 {devMode ? "ON" : "OFF"}
          </button>
          
          {/* 개발자 모드 세부 토글 */}
          {devMode && (
            <div className="flex flex-col gap-1 rounded-md bg-amber-950/80 border border-amber-500/20 p-1.5 mt-0.5">
              <label className="flex items-center gap-1.5 text-[9px] text-amber-200/90 cursor-pointer">
                <input
                  type="checkbox"
                  checked={devBlocky}
                  onChange={(e) => setDevBlocky(e.target.checked)}
                  className="rounded text-amber-600 bg-amber-950 border-amber-500/40"
                />
                Blocky Attention
              </label>
              <label className="flex items-center gap-1.5 text-[9px] text-amber-200/90 cursor-pointer">
                <input
                  type="checkbox"
                  checked={devGrid}
                  disabled={!devBlocky}
                  onChange={(e) => setDevGrid(e.target.checked)}
                  className="rounded text-amber-600 bg-amber-950 border-amber-500/40 disabled:opacity-40"
                />
                Patch Grid
              </label>
            </div>
          )}
        </div>
      )}

      {/* 범례 */}
      {!error && attnOverlays.length > 0 && showHeatmap && (
        <div className="pointer-events-none absolute bottom-2 left-2 z-10 flex items-center gap-2 rounded-md bg-black/55 px-2 py-1 text-[10px] text-white/70">
          {(["낮음", "중간", "높음"] as const).map((label, i) => {
            const alpha  = [0.06, 0.09, 0.12][i];
            const sAlpha = [0.40, 0.65, 0.90][i];
            const sw     = [1.5,  2,    2.5][i];
            const dash   = i === 0 ? "10 6" : i === 1 ? "6 3" : "";
            return (
              <span key={label} className="flex items-center gap-1">
                <svg width="14" height="14" viewBox="0 0 14 14" className="shrink-0">
                  <circle cx="7" cy="7" r="5.5"
                    fill={`rgba(${attnRgb},${alpha})`}
                    stroke={`rgba(${attnRgb},${sAlpha})`}
                    strokeWidth={sw}
                    strokeDasharray={dash}
                  />
                </svg>
                {label}
              </span>
            );
          })}
          <span className="text-white/40">·</span>
          <span style={{ color: `rgba(${attnRgb},0.90)` }}>{attnLabel}</span>
          <span className="text-white/35">({attnOverlays.length}패치)</span>
        </div>
      )}

      {/* 줌 컨트롤 */}
      {!error && (
        <div className="absolute right-[148px] top-1/2 z-10 flex -translate-y-1/2 flex-col items-center gap-1 rounded-full bg-black/45 p-1.5">
          <button
            onClick={() => viewerRef.current?.viewport.zoomBy(2, undefined, true)}
            className="flex size-7 items-center justify-center rounded-full bg-white/10 text-white hover:bg-white/20"
          >
            <Plus className="size-4" />
          </button>
          <button
            onClick={() => viewerRef.current?.viewport.goHome(true)}
            className="flex size-7 items-center justify-center rounded-full bg-white/10 text-white hover:bg-white/20"
          >
            <Home className="size-3.5" />
          </button>
          <button
            onClick={() => viewerRef.current?.viewport.zoomBy(0.5, undefined, true)}
            className="flex size-7 items-center justify-center rounded-full bg-white/10 text-white hover:bg-white/20"
          >
            <Minus className="size-4" />
          </button>
        </div>
      )}

      {/* ── 클릭된 패치 상세 모달 (ROI Patch Zoom) ─────────────────────────── */}
      {clickedPatch && (
        <div className="absolute inset-0 z-30 flex items-center justify-center bg-black/70 p-4 animate-in fade-in zoom-in duration-200">
          <div className="relative w-full max-w-sm rounded-lg border border-border bg-[#100c0a] p-4 text-white shadow-2xl">
            <button
              onClick={() => setClickedPatch(null)}
              className="absolute right-3 top-3 rounded-md bg-white/10 p-1 text-white/60 hover:bg-white/20 hover:text-white"
            >
              <X className="size-4" />
            </button>
            <h4 className="flex items-center gap-1.5 text-xs font-bold text-amber-500 mb-3">
              <ZoomIn className="size-3.5" />
              Original Patch 확대 검사
            </h4>

            <div className="flex gap-4">
              {/* 패치 이미지 렌더링 */}
              {clickedPatchUrl ? (
                <div className="size-32 shrink-0 overflow-hidden rounded-md border border-border bg-black/50">
                  {clickedPatchSrc ? (
                    <img
                      src={clickedPatchSrc}
                      alt="Patch Crop"
                      className="h-full w-full object-cover"
                    />
                  ) : (
                    <div className="h-full w-full flex items-center justify-center text-[10px] text-white/30">
                      로딩 중...
                    </div>
                  )}
                </div>
              ) : (
                <div className="size-32 shrink-0 flex items-center justify-center rounded-md border border-border bg-[#1a1512] text-[10px] text-white/30">
                  좌표 매핑 유실
                </div>
              )}

              {/* 상세 세부 지표 */}
              <div className="flex-1 min-w-0 flex flex-col justify-between py-0.5">
                <div>
                  <div className="text-[10px] text-white/40 mb-0.5">패치 시작 위치 (Lv 0)</div>
                  <div className="text-xs font-medium text-white/80 font-mono">
                    X: {clickedPatch.px?.toLocaleString() ?? "-"} / Y: {clickedPatch.py?.toLocaleString() ?? "-"}
                  </div>
                </div>

                <div className="mt-2.5">
                  <div className="flex items-center justify-between text-[10px] text-white/40 mb-1">
                    <span>AI Attention 가중치</span>
                    <span className="font-bold text-white/95">{Math.round(clickedPatch.weight * 100)}%</span>
                  </div>
                  <div className="h-1.5 w-full bg-white/10 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-amber-500 rounded-full"
                      style={{ width: `${Math.round(clickedPatch.weight * 100)}%` }}
                    />
                  </div>
                </div>

                {clickedPatch.contrib && Object.keys(clickedPatch.contrib).length > 0 && (
                  <div className="mt-2 text-[10px] text-white/50 border-t border-white/5 pt-1">
                    <span className="font-semibold text-white/70">병변별 추정 기여</span>
                    <div className="flex flex-wrap gap-x-2 gap-y-0.5 mt-0.5">
                      {Object.entries(clickedPatch.contrib).slice(0, 3).map(([k, val]) => (
                        <span key={k}>
                          {TARGET_LABEL[k] ?? k}: <strong className="text-white/80">{Math.round(val * 100)}%</strong>
                        </span>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
