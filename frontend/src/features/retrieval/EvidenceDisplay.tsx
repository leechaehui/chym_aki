import { useState } from "react";
import { Microscope, Star } from "lucide-react";
import type { RetrievalHit } from "@/types/retrieval";
import { wsiService } from "@/services/wsiService";
import { useAuthenticatedImage } from "@/hooks/useAuthenticatedImage";
import { cn } from "@/lib/cn";

/** 단일 프로토타입 Evidence 카드 — Label/Similarity/Confidence/대표 WSI 만(진단·reasoning 없음). */
export function EvidenceDisplay({ hit, rank }: { hit: RetrievalHit; rank: number }) {
  const [imgOk, setImgOk] = useState(true);
  const thumbUrl = hit.representativeWsi
    ? wsiService.thumbnailUrl(hit.representativeWsi.slideId, 240)
    : null;
  const src = useAuthenticatedImage(thumbUrl);

  return (
    <div className="flex gap-3 rounded-lg border border-border bg-card p-3">
      {/* 대표 WSI (medoid) */}
      <div className="flex size-24 shrink-0 items-center justify-center overflow-hidden rounded-md border border-border bg-muted">
        {src && imgOk ? (
          <img src={src} alt={hit.representativeWsi!.slideId}
               className="size-full object-cover" onError={() => setImgOk(false)} />
        ) : (
          <Microscope className="size-7 text-muted-foreground/50" />
        )}
      </div>

      <div className="min-w-0 flex-1">
        <div className="flex items-center gap-2">
          <span className="text-[10px] font-bold text-muted-foreground">#{rank}</span>
          <span className="truncate text-sm font-semibold text-foreground">{hit.label}</span>
          {hit.isRare && (
            <span className="flex items-center gap-0.5 rounded bg-warning/15 px-1 text-[9px] font-bold text-warning">
              <Star className="size-2.5" /> 희귀
            </span>
          )}
        </div>

        {/* 유사도 / 신뢰도 막대 */}
        <div className="mt-1.5 space-y-1">
          <Metric label="유사도" value={hit.similarity} tone="bg-primary" />
          <Metric label="신뢰도" value={hit.confidence} tone="bg-success" />
        </div>

        <div className="mt-1.5 flex flex-wrap gap-x-3 gap-y-0.5 text-[10px] text-muted-foreground">
          <span>KDIGO {hit.kdigoBand ?? "—"}</span>
          <span>원인 {hit.etiologyHint ?? "—"}</span>
          <span>eGFR~{hit.egfrMean ?? "—"}</span>
          <span>구성원 {hit.nMembers}명</span>
          {hit.representativeWsi && <span>WSI {hit.representativeWsi.slideId}</span>}
        </div>
      </div>
    </div>
  );
}

function Metric({ label, value, tone }: { label: string; value: number; tone: string }) {
  return (
    <div className="flex items-center gap-2">
      <span className="w-16 shrink-0 text-[10px] text-muted-foreground">{label}</span>
      <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-muted">
        <div className={cn("h-full rounded-full", tone)} style={{ width: `${Math.round(value * 100)}%` }} />
      </div>
      <span className="w-10 shrink-0 text-right text-[10px] font-semibold text-foreground">
        {value.toFixed(3)}
      </span>
    </div>
  );
}
