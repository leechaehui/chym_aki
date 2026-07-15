import { useEffect, useState } from "react";
import { Search, X } from "lucide-react";
import type { IcuPatientSearchResult, ModelMetrics } from "@/types";
import { icuMonitorService } from "@/services/icuMonitorService";
import { fakeKoreanName } from "@/lib/fakeName";
import { riskBandFor } from "@/lib/riskLabel";
import { cn } from "@/lib/cn";
import { PatientQuickView } from "./PatientQuickView";

/** 긴 careunit 풀네임 → 약어(괄호 안). */
function shortUnit(u: string): string {
  const m = u.match(/\(([^)]+)\)/);
  return m ? m[1] : u;
}

/**
 * ICU 코호트 환자 검색 — Stay ID · Subject ID · 병동으로 조회.
 * 입력 + [검색]/[취소] 버튼(또는 Enter)으로 명시 조회하고, 결과 클릭 시 Quick View 를 연다.
 * ICU 모니터·신장내과 대시보드 양쪽에서 재사용하는 자족형 컴포넌트.
 */
export function IcuPatientSearch({ className, onAddStay, isAdding }: { className?: string; onAddStay?: (stayId: number) => void; isAdding?: boolean }) {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<IcuPatientSearchResult[]>([]);
  const [searching, setSearching] = useState(false);
  const [searched, setSearched] = useState(false);
  const [quickViewPatient, setQuickViewPatient] = useState<IcuPatientSearchResult | null>(null);
  const [metrics, setMetrics] = useState<ModelMetrics | null>(null);

  useEffect(() => {
    icuMonitorService.modelMetrics().then(setMetrics).catch(() => {});
  }, []);

  function runSearch() {
    const q = query.trim();
    if (q.length < 1) return;
    setSearching(true);
    setSearched(true);
    icuMonitorService
      .search(q, 12)
      .then(setResults)
      .catch(() => setResults([]))
      .finally(() => setSearching(false));
  }

  function clearSearch() {
    setQuery("");
    setResults([]);
    setSearched(false);
  }

  return (
    <div className={className}>
      <div className="flex items-center gap-2">
        <div className="flex flex-1 items-center gap-2 rounded-lg border border-border bg-card px-3 py-2">
          <Search className="size-4 text-muted-foreground" />
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && runSearch()}
            placeholder="환자 검색 — 이름(예: 오준현) · Stay ID · Subject ID · 병동(MICU 등)"
            className="w-full bg-transparent text-sm outline-none placeholder:text-muted-foreground"
          />
        </div>
        <button
          onClick={runSearch}
          disabled={searching || query.trim().length < 1}
          className="rounded-lg bg-primary px-4 py-2 text-sm font-medium text-primary-foreground hover:bg-primary/90 disabled:opacity-50"
        >
          {searching ? "검색 중…" : "검색"}
        </button>
        <button
          onClick={clearSearch}
          disabled={!query && results.length === 0}
          className="rounded-lg bg-secondary px-4 py-2 text-sm font-medium text-foreground hover:bg-muted disabled:opacity-50"
        >
          취소
        </button>
      </div>

      {/* 결과 패널 */}
      {searched && (
        <div className="mt-2 overflow-hidden rounded-lg border border-border bg-card">
          {searching ? (
            <p className="px-3 py-3 text-center text-[12px] text-muted-foreground">검색 중…</p>
          ) : results.length === 0 ? (
            <p className="px-3 py-3 text-center text-[12px] text-muted-foreground">검색 결과가 없습니다.</p>
          ) : (
            <ul className="max-h-72 overflow-y-auto">
              {results.map((r) => {
                const band = riskBandFor(r.riskScore, r.stage);
                return (
                  <li key={r.stayId}>
                    <button
                      onClick={() => setQuickViewPatient(r)}
                      className="flex w-full items-center justify-between gap-2 border-b border-border/40 px-3 py-2 text-left text-[12px] last:border-0 hover:bg-muted/50"
                    >
                      <span>
                        <b>{fakeKoreanName(r.subjectId)}</b>
                        <span className="ml-2 text-muted-foreground">
                          Stay {r.stayId} · {shortUnit(r.careunit)} · {r.age ?? "—"}/{r.gender} · {r.stage}
                        </span>
                      </span>
                      <span className={cn("shrink-0 font-semibold tabular-nums", band.textClass)}>
                        {band.dot} {band.label} {r.riskScore}
                      </span>
                    </button>
                  </li>
                );
              })}
            </ul>
          )}
        </div>
      )}

      <PatientQuickView
        patient={quickViewPatient}
        metrics={metrics}
        onClose={() => setQuickViewPatient(null)}
        onAddStay={onAddStay}
        isAdding={isAdding}
      />
    </div>
  );
}
