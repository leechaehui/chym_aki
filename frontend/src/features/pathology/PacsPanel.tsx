import { useEffect, useState } from "react";
import { Database, RefreshCw, CheckCircle2, Clock, AlertCircle } from "lucide-react";
import type { WsiSlide, WsiStain } from "@/types/wsi";
import { wsiService, type CacheStatus } from "@/services/wsiService";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { LoadingSpinner } from "@/components/common/LoadingSpinner";
import { EmptyState } from "@/components/common/EmptyState";
import { useAuthenticatedImage } from "@/hooks/useAuthenticatedImage";
import { cn } from "@/lib/cn";

const STAINS: WsiStain[] = ["HE", "MT", "PAS"];

function StatusIcon({ status }: { status: CacheStatus["status"] }) {
  if (status === "ready") return <CheckCircle2 className="size-3.5 text-success" />;
  if (status === "downloading") return <Clock className="size-3.5 animate-pulse text-warning" />;
  if (status === "error") return <AlertCircle className="size-3.5 text-destructive" />;
  return <Clock className="size-3.5 text-muted-foreground" />;
}

/** WSI 썸네일 — 8001 병리과 인증이 필요해 <img src>가 아닌 인증된 fetch(blob URL)로 로드. */
function SlideThumbnail({ url, alt }: { url: string; alt: string }) {
  const src = useAuthenticatedImage(url);
  const [failed, setFailed] = useState(false);

  if (!src || failed) return null;
  return (
    <img
      src={src}
      alt={alt}
      className="h-full w-full object-cover"
      onError={() => setFailed(true)}
    />
  );
}

export function PacsPanel() {
  const [stain, setStain] = useState<WsiStain>("HE");
  const [slides, setSlides] = useState<WsiSlide[]>([]);
  const [loading, setLoading] = useState(false);
  const [preparing, setPreparing] = useState<Record<string, CacheStatus>>({});

  function loadSlides() {
    setLoading(true);
    wsiService
      .listSlides(stain)
      .then((r) => setSlides(r.slides))
      .catch(() => setSlides([]))
      .finally(() => setLoading(false));
  }

  useEffect(() => { loadSlides(); }, [stain]);

  async function handlePrepare(slide: WsiSlide) {
    setPreparing((p) => ({ ...p, [slide.slide_id]: { status: "downloading", downloaded_mb: 0 } }));
    try {
      const st = await wsiService.prepare(slide.slide_id);
      setPreparing((p) => ({ ...p, [slide.slide_id]: st }));
      if (st.status !== "ready") {
        const timer = setInterval(async () => {
          const s = await wsiService.cacheStatus(slide.slide_id).catch(() => null);
          if (!s) { clearInterval(timer); return; }
          setPreparing((p) => ({ ...p, [slide.slide_id]: s }));
          if (s.status === "ready" || s.status === "error") clearInterval(timer);
        }, 3000);
      }
    } catch {
      setPreparing((p) => ({ ...p, [slide.slide_id]: { status: "error", message: "요청 실패" } }));
    }
  }

  const thumbUrl = (slide: WsiSlide) => wsiService.thumbnailUrl(slide.slide_id, 160);

  return (
    <Card>
      <CardHeader className="flex-row flex-wrap items-center gap-2">
        <Database className="size-4 text-primary" />
        <CardTitle>PACS 슬라이드 관리</CardTitle>
        <div className="ml-auto flex items-center gap-2">
          {STAINS.map((s) => (
            <button
              key={s}
              onClick={() => setStain(s)}
              className={cn(
                "rounded-md px-3 py-1 text-xs font-medium",
                stain === s ? "bg-primary text-primary-foreground" : "bg-secondary text-muted-foreground",
              )}
            >
              {s}
            </button>
          ))}
          <Button size="sm" variant="outline" onClick={loadSlides} disabled={loading}>
            <RefreshCw className={cn("size-3.5", loading && "animate-spin")} />
          </Button>
        </div>
      </CardHeader>
      <CardContent>
        {loading ? (
          <LoadingSpinner label="슬라이드 목록 불러오는 중" />
        ) : slides.length === 0 ? (
          <EmptyState icon={Database} title="슬라이드 없음" description="PACS 서버에 등록된 슬라이드가 없습니다." />
        ) : (
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-4">
            {slides.map((slide) => {
              const cacheState = preparing[slide.slide_id];
              const isReady = slide.cached || cacheState?.status === "ready";
              return (
                <div
                  key={slide.slide_id}
                  className="flex flex-col gap-2 rounded-lg border border-border p-2"
                >
                  <div className="relative aspect-video overflow-hidden rounded bg-muted">
                    <SlideThumbnail url={thumbUrl(slide)} alt={slide.case_code || slide.slide_id} />
                    {isReady && (
                      <span className="absolute right-1 top-1 rounded-full bg-success/90 px-1.5 py-0.5 text-[9px] font-bold text-white">
                        캐시됨
                      </span>
                    )}
                  </div>
                  <div className="min-w-0">
                    <p className="truncate text-xs font-medium text-foreground">
                      {slide.case_code || slide.slide_id.slice(0, 12) + "…"}
                    </p>
                    <p className="text-[10px] text-muted-foreground">{slide.stain}</p>
                  </div>
                  <div className="flex items-center gap-1.5">
                    {cacheState && <StatusIcon status={cacheState.status} />}
                    {!isReady && (
                      <Button
                        size="sm"
                        variant="outline"
                        className="h-6 px-2 text-[10px]"
                        onClick={() => handlePrepare(slide)}
                        disabled={cacheState?.status === "downloading"}
                      >
                        {cacheState?.status === "downloading" ? "다운로드 중..." : "PACS에서 받기"}
                      </Button>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </CardContent>
    </Card>
  );
}
