import { useEffect, useState } from "react";
import { Search, ShieldAlert, Info, HeartPulse } from "lucide-react";
import type { RetrievalQuery, RetrievalResult } from "@/types/retrieval";
import type { IcuAkiPatient } from "@/types";
import { retrievalService } from "@/services/retrievalService";
import { icuMonitorService } from "@/services/icuMonitorService";
import { PageHeader } from "@/components/common/PageHeader";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Label, Select } from "@/components/ui/input";
import { EmptyState } from "@/components/common/EmptyState";
import { LoadingSpinner } from "@/components/common/LoadingSpinner";
import { EvidenceDisplay } from "./EvidenceDisplay";

type Form = {
  kdigoStage: string; egfr: string; proteinuriaMgG: string; a1cPct: string;
  age: string; sex: string; diabetes: string; hypertension: string;
  oliguria: string; crTrendSlope: string; etiologyHint: string;
};

const EMPTY: Form = {
  kdigoStage: "", egfr: "", proteinuriaMgG: "", a1cPct: "", age: "", sex: "",
  diabetes: "", hypertension: "", oliguria: "", crTrendSlope: "", etiologyHint: "unknown",
};

// 디버깅용 샘플(객관 임상값만, etiology 미입력).
const SAMPLES: Record<string, Form> = {
  "당뇨·저eGFR·고단백 (stage3)": { ...EMPTY, kdigoStage: "3", egfr: "35", proteinuriaMgG: "1200",
    a1cPct: "9.0", age: "65", sex: "Female", diabetes: "true", hypertension: "true",
    oliguria: "1", crTrendSlope: "0.8" },
  "비당뇨·정상·보존 (stage1)": { ...EMPTY, kdigoStage: "1", egfr: "95", proteinuriaMgG: "80",
    a1cPct: "5.4", age: "40", sex: "Male", diabetes: "false", hypertension: "false",
    oliguria: "0", crTrendSlope: "0.1" },
};

// 원인(추정) — 값은 백엔드 코드 유지, 표시는 한국어.
const ETIO_LABEL: Record<string, string> = {
  unknown: "미상", ATI: "ATI (급성세뇨관손상)", AIN: "AIN (급성간질신염)",
  DKD: "DKD (당뇨신증)", prerenal: "신전성", postrenal: "신후성",
};

function toQuery(f: Form): RetrievalQuery {
  const num = (s: string) => (s.trim() === "" ? null : Number(s));
  const boolOf = (s: string) => (s === "" ? null : s === "true");
  return {
    kdigoStage: num(f.kdigoStage), egfr: num(f.egfr), proteinuriaMgG: num(f.proteinuriaMgG),
    a1cPct: num(f.a1cPct), age: num(f.age), oliguria: num(f.oliguria),
    crTrendSlope: num(f.crTrendSlope), sex: f.sex || null,
    diabetes: boolOf(f.diabetes), hypertension: boolOf(f.hypertension),
    etiologyHint: f.etiologyHint || "unknown", k: 5,
  };
}

export function RetrievalDashboard() {
  const [form, setForm] = useState<Form>(SAMPLES["당뇨·저eGFR·고단백 (stage3)"]);
  const [result, setResult] = useState<RetrievalResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [icu, setIcu] = useState<IcuAkiPatient[]>([]);
  const [stay, setStay] = useState<string>("");

  // ICU AKI 환자 목록(자동 concept 추출용). frozen flow STEP1.
  useEffect(() => {
    icuMonitorService.list({ limit: 40 }).then(setIcu).catch(() => setIcu([]));
  }, []);

  // ICU 환자 선택 → stayId 로 MIMIC concept 자동 추출 검색.
  async function runByStay(stayId: number) {
    setLoading(true); setError(null);
    try {
      setResult(await retrievalService.query({ stayId, k: 5 }));
    } catch (e) {
      setError(e instanceof Error ? e.message : "검색 실패");
    } finally {
      setLoading(false);
    }
  }

  const set = (k: keyof Form) => (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) =>
    setForm((p) => ({ ...p, [k]: e.target.value }));

  async function run() {
    setLoading(true); setError(null);
    try {
      setResult(await retrievalService.query(toQuery(form)));
    } catch (e) {
      setError(e instanceof Error ? e.message : "검색 실패");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="mx-auto max-w-[1100px]">
      <PageHeader title="병리 참조 검색"
        subtitle="임상 표현형이 유사한 KPMP 프로토타입을 참조로 검색 · 진단 아님" />

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-[360px_1fr]">
        {/* 입력: Clinical Concept */}
        <Card className="h-fit">
          <CardHeader className="gap-2">
            <CardTitle>임상 개념</CardTitle>
            <div className="flex flex-wrap gap-1">
              {Object.keys(SAMPLES).map((k) => (
                <button key={k} onClick={() => setForm(SAMPLES[k])}
                  className="flex items-center gap-1 rounded bg-secondary px-2 py-1 text-[10px] text-muted-foreground hover:text-foreground">
                  {k}
                </button>
              ))}
            </div>
            {/* STEP1 — ICU AKI 환자 선택 → MIMIC concept 자동 추출 */}
            <div>
              <Label>
                <span className="flex items-center gap-1"><HeartPulse className="size-3 text-destructive" /> ICU AKI 환자 (자동)</span>
              </Label>
              <Select
                value={stay}
                onChange={(e) => { setStay(e.target.value); if (e.target.value) runByStay(Number(e.target.value)); }}
                className="h-8 text-xs"
                disabled={loading || icu.length === 0}
              >
                <option value="">{icu.length === 0 ? "불러오는 중…" : "ICU 환자 선택 → 자동 검색"}</option>
                {icu.map((p) => (
                  <option key={p.stayId} value={p.stayId}>
                    {p.stayId} · {p.careunit} · {p.stage} · {p.age ?? "—"}{p.gender?.[0] ?? ""}
                  </option>
                ))}
              </Select>
              <p className="mt-1 text-[10px] text-muted-foreground">선택 시 KDIGO·eGFR·Cr trend 등이 MIMIC에서 자동 추출됩니다. 아래는 수동 입력.</p>
            </div>
          </CardHeader>
          <CardContent className="grid grid-cols-2 gap-2">
            <Num label="KDIGO 단계" v={form.kdigoStage} on={set("kdigoStage")} ph="0-3" />
            <Num label="eGFR" v={form.egfr} on={set("egfr")} ph="ml/min" />
            <Num label="단백뇨" v={form.proteinuriaMgG} on={set("proteinuriaMgG")} ph="mg/g" />
            <Num label="당화혈색소(HbA1c)" v={form.a1cPct} on={set("a1cPct")} ph="%" />
            <Num label="나이" v={form.age} on={set("age")} ph="세" />
            <Field label="성별">
              <Select value={form.sex} onChange={set("sex")} className="h-8 text-xs">
                <option value="">—</option><option value="Male">남</option><option value="Female">여</option>
              </Select>
            </Field>
            <Field label="당뇨">
              <Select value={form.diabetes} onChange={set("diabetes")} className="h-8 text-xs">
                <option value="">—</option><option value="true">예</option><option value="false">아니오</option>
              </Select>
            </Field>
            <Field label="고혈압">
              <Select value={form.hypertension} onChange={set("hypertension")} className="h-8 text-xs">
                <option value="">—</option><option value="true">예</option><option value="false">아니오</option>
              </Select>
            </Field>
            <Num label="소변감소(0-2)" v={form.oliguria} on={set("oliguria")} ph="0-2" />
            <Num label="Cr 추세" v={form.crTrendSlope} on={set("crTrendSlope")} ph="기울기" />
            <Field label="원인(추정)">
              <Select value={form.etiologyHint} onChange={set("etiologyHint")} className="h-8 text-xs">
                {Object.entries(ETIO_LABEL).map(([v, ko]) =>
                  <option key={v} value={v}>{ko}</option>)}
              </Select>
            </Field>
            <div className="col-span-2 mt-1">
              <Button onClick={run} disabled={loading} className="w-full">
                <Search className="size-4" /> {loading ? "검색 중…" : "참조 검색"}
              </Button>
            </div>
          </CardContent>
        </Card>

        {/* 결과: Evidence */}
        <div className="flex flex-col gap-3">
          {error && <div className="rounded-lg border border-destructive/30 bg-destructive/5 p-3 text-xs text-destructive">{error}</div>}
          {loading ? <LoadingSpinner label="검색 중" />
            : !result ? <EmptyState icon={Search} title="검색 대기" description="임상값을 입력하고 ‘참조 검색’을 누르세요." />
            : result.ood.isOod ? (
              <Card><CardContent className="flex items-center gap-2 py-8 text-sm text-warning">
                <ShieldAlert className="size-5" /> {result.ood.message ?? "신뢰할 만한 병리 참조를 찾지 못했습니다."}
              </CardContent></Card>
            ) : (
              <>
                <div className="flex flex-wrap items-center gap-2 rounded-lg border border-border bg-muted/30 px-3 py-2 text-[11px] text-muted-foreground">
                  <span className="font-semibold text-foreground">임상 개념</span>
                  <span>KDIGO {result.concept.kdigoBand}</span>
                  <span>eGFR {result.concept.egfr ?? "—"}</span>
                  <span>단백뇨 {result.concept.proteinuria ?? "—"}</span>
                  <span>원인 {result.concept.etiologyHint}</span>
                  <span className="ml-auto">완전도 {(result.concept.completeness * 100).toFixed(0)}%</span>
                  {result.ood.score != null && <span>분포외(OOD) {result.ood.score.toFixed(2)}</span>}
                </div>
                {result.hits.map((h, i) => <EvidenceDisplay key={h.prototypeId} hit={h} rank={i + 1} />)}
                <p className="flex items-center gap-1 text-[10px] text-muted-foreground/70">
                  <Info className="size-3" /> {result.referenceNotice}
                </p>
              </>
            )}
        </div>
      </div>
    </div>
  );
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return <div><Label>{label}</Label>{children}</div>;
}
function Num({ label, v, on, ph }: { label: string; v: string; on: React.ChangeEventHandler<HTMLInputElement>; ph?: string }) {
  return (
    <Field label={label}>
      <input type="number" value={v} onChange={on} placeholder={ph}
        className="h-8 w-full rounded-md border border-border bg-background px-2 text-xs text-foreground" />
    </Field>
  );
}
