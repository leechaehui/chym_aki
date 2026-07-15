import { useRef, useState } from "react";
import { Sparkles, Loader2, Mic, Square, ShieldCheck, ShieldAlert } from "lucide-react";
import type { AiDraftResult, RiskTier } from "@/types";
import { icuMonitorService } from "@/services/icuMonitorService";
import { voiceService } from "@/services/voiceService";
import { useNotificationStore } from "@/store/notificationStore";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/input";
import { StatusBadge } from "@/components/common/StatusBadge";
import { EmptyState } from "@/components/common/EmptyState";

const TIER_TONE: Record<RiskTier, "danger" | "warning" | "success"> = {
  HIGH: "danger",
  MEDIUM: "warning",
  LOW: "success",
};
const TIER_LABEL: Record<RiskTier, string> = { HIGH: "고위험", MEDIUM: "중등도", LOW: "안정" };
const COMPONENT_LABEL: Record<string, string> = {
  creatinine_trend: "Cr 추세",
  urine_output_drop: "소변량 감소",
  diagnosis_risk_weight: "진단 가중(AKI모델)",
  vitals_instability: "활력징후 불안정",
};

/**
 * 실제 ICU stay 기반 AI 진료 초안(SOAP) — 음성/텍스트 입력 → 백엔드 근거기반 파이프라인.
 * 데모 환자가 아니라 stay_id 로 실데이터(Cr/소변량/모델예측)를 사용한다.
 * 근거(evidence) 칩·breakdown 텍스트는 가독성을 위해 고대비로 표시한다.
 */
export function IcuSoapDraftPanel({ stayId }: { stayId: number }) {
  const [transcript, setTranscript] = useState("");
  const [result, setResult] = useState<AiDraftResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [recording, setRecording] = useState(false);
  const [transcribing, setTranscribing] = useState(false);
  const notify = useNotificationStore((s) => s.notify);

  const recorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);

  async function startRecording() {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const recorder = new MediaRecorder(stream);
      chunksRef.current = [];
      recorder.ondataavailable = (e) => e.data.size > 0 && chunksRef.current.push(e.data);
      recorder.onstop = async () => {
        stream.getTracks().forEach((t) => t.stop());
        await transcribe(new Blob(chunksRef.current, { type: "audio/webm" }));
      };
      recorder.start();
      recorderRef.current = recorder;
      setRecording(true);
    } catch {
      notify({ severity: "WARNING", department: "nephrology", title: "마이크 사용 불가", message: "마이크 권한이 없거나 장치를 찾을 수 없습니다. 텍스트로 입력해 주세요." });
    }
  }

  function stopRecording() {
    recorderRef.current?.stop();
    setRecording(false);
  }

  async function transcribe(blob: Blob) {
    setTranscribing(true);
    try {
      const res = await voiceService.transcribe(blob);
      if (res.transcript.trim()) {
        setTranscript((prev) => (prev ? `${prev} ${res.transcript}` : res.transcript));
      } else {
        notify({ severity: "INFO", department: "nephrology", title: "음성 인식 결과 없음", message: `STT 엔진(${res.engine}) — whisper 미설치 환경입니다. 대화 내용을 텍스트로 입력해 주세요.` });
      }
    } catch {
      notify({ severity: "WARNING", department: "nephrology", title: "전사 실패", message: "음성 전사 중 오류가 발생했습니다." });
    } finally {
      setTranscribing(false);
    }
  }

  async function generate() {
    setLoading(true);
    try {
      const res = await icuMonitorService.draftForStay(stayId, transcript);
      setResult(res);
    } catch (e) {
      notify({ severity: "WARNING", department: "nephrology", title: "초안 생성 실패", message: e instanceof Error ? e.message : "검증에 실패했거나 오류가 발생했습니다." });
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="flex flex-col gap-3">
      <div className="flex flex-col gap-2">
        <div className="flex items-center gap-2">
          {recording ? (
            <Button onClick={stopRecording} variant="destructive" size="sm" className="self-start">
              <Square className="size-4" /> 녹음 중지
            </Button>
          ) : (
            <Button onClick={startRecording} variant="outline" size="sm" disabled={transcribing} className="self-start">
              {transcribing ? <Loader2 className="size-4 animate-spin" /> : <Mic className="size-4" />}
              {transcribing ? "전사 중…" : "음성 녹음"}
            </Button>
          )}
          {recording && (
            <span className="flex items-center gap-1.5 text-[11px] font-medium text-danger">
              <span className="size-2 animate-pulse rounded-full bg-danger" /> 녹음 중…
            </span>
          )}
        </div>
        <Textarea
          value={transcript}
          onChange={(e) => setTranscript(e.target.value)}
          placeholder="환자와의 대화 내용을 입력하거나 음성으로 녹음하세요. (예: 어제부터 소변량이 줄고 부종이 생겼습니다…)"
          className="min-h-20"
        />
      </div>

      <div className="flex items-center gap-2">
        <Button onClick={generate} disabled={loading || !transcript.trim()} size="sm" className="self-start">
          {loading ? <Loader2 className="size-4 animate-spin" /> : <Sparkles className="size-4" />}
          AI 진료 초안 생성
        </Button>
        {!transcript.trim() && <span className="text-[11px] text-muted-foreground">대화 내용을 입력해야 초안을 생성할 수 있습니다.</span>}
      </div>

      {result ? (
        <div className="flex flex-col gap-3">
          <RiskCard result={result} />
          <SoapGrid result={result} />
          <ProblemList result={result} />
          <ValidationBar result={result} />
        </div>
      ) : (
        <EmptyState
          icon={Sparkles}
          title="아직 생성된 초안이 없습니다"
          description="음성 녹음 또는 텍스트 입력 후 'AI 진료 초안 생성'을 누르면 근거기반 SOAP·문제목록·위험도가 표시됩니다."
        />
      )}
    </div>
  );
}

/** CDSS 위험도 — tier 배지 + 점수 + 컴포넌트 기여도 breakdown. */
function RiskCard({ result }: { result: AiDraftResult }) {
  const { risk } = result;
  const pct = Math.round(risk.risk_score * 100);
  return (
    <div className="rounded-lg border border-border bg-card p-3">
      <div className="mb-2 flex items-center justify-between">
        <p className="text-[11px] font-semibold text-primary">CDSS 위험도 (rule + AI 모델)</p>
        <div className="flex items-center gap-2">
          <StatusBadge label={`${TIER_LABEL[risk.tier]} · ${pct}`} tone={TIER_TONE[risk.tier]} />
          {risk.nephrology_trigger && <StatusBadge label="신장내과 호출" tone="danger" />}
        </div>
      </div>
      <div className="flex flex-col gap-1.5">
        {risk.breakdown.map((c) => (
          <div key={c.component} className="flex items-center gap-2 text-[11px]">
            <span className="w-28 shrink-0 font-medium text-foreground">{COMPONENT_LABEL[c.component] ?? c.component}</span>
            <div className="h-2 flex-1 overflow-hidden rounded-full bg-muted">
              <div className="h-full rounded-full bg-primary" style={{ width: `${Math.round((c.contribution / c.weight) * 100)}%` }} />
            </div>
            <span className="w-10 shrink-0 text-right font-semibold tabular-nums text-foreground">{c.contribution.toFixed(2)}</span>
            <span className="hidden w-44 shrink-0 truncate text-foreground/70 sm:inline" title={c.explanation}>{c.explanation}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

/** SOAP — S/O(추출) + A/P(근거 매핑). */
function SoapGrid({ result }: { result: AiDraftResult }) {
  const { soap } = result;
  return (
    <div className="grid gap-2 sm:grid-cols-2">
      <SoapText label="S · 주관적 (전사)" text={soap.S} />
      <SoapText label="O · 객관적 (검사)" text={soap.O} />
      <SoapEvidence label="A · 평가" text={soap.A.assessment ?? ""} statements={soap.A.statements} />
      <SoapEvidence label="P · 계획" text={soap.P.plan ?? ""} statements={soap.P.statements} />
    </div>
  );
}

function SoapText({ label, text }: { label: string; text: string }) {
  return (
    <div className="rounded-lg border border-border bg-muted/30 p-3">
      <p className="mb-1 text-[11px] font-semibold text-primary">{label}</p>
      <p className="text-xs leading-relaxed text-foreground">{text}</p>
    </div>
  );
}

/** A/P — 진술별 근거(evidence) 칩 표시(hallucination 방지 가시화). 고대비 칩. */
function SoapEvidence({ label, text, statements }: { label: string; text: string; statements: readonly { statement: string; evidence: readonly string[] }[] }) {
  return (
    <div className="rounded-lg border border-border bg-muted/30 p-3">
      <p className="mb-1 text-[11px] font-semibold text-primary">{label}</p>
      {statements.length === 0 ? (
        <p className="text-xs italic text-muted-foreground">{text}</p>
      ) : (
        <ul className="flex flex-col gap-2">
          {statements.map((st, i) => (
            <li key={i} className="text-xs leading-relaxed">
              <span className="font-medium text-foreground">{st.statement}</span>
              <span className="mt-1 flex flex-wrap gap-1">
                {st.evidence.map((ev, j) => (
                  <span key={j} className="rounded border border-primary/30 bg-primary/10 px-1.5 py-0.5 text-[11px] font-medium text-foreground" title="근거">
                    근거: {ev}
                  </span>
                ))}
              </span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

/** Problem List — A 기반 문제 + 신뢰도. */
function ProblemList({ result }: { result: AiDraftResult }) {
  if (result.problemList.length === 0) return null;
  return (
    <div className="rounded-lg border border-border bg-card p-3">
      <p className="mb-2 text-[11px] font-semibold text-primary">Problem List (A 기반)</p>
      <ul className="flex flex-col gap-1.5">
        {result.problemList.map((p, i) => (
          <li key={i} className="flex items-center justify-between gap-2 text-xs">
            <span className="font-medium text-foreground">{p.problem}</span>
            <span className="flex items-center gap-2">
              <span className="text-[10px] text-muted-foreground">근거 {p.evidence.length}</span>
              <StatusBadge label={`신뢰도 ${Math.round(p.confidence * 100)}%`} tone="primary" />
            </span>
          </li>
        ))}
      </ul>
    </div>
  );
}

/** 검증 결과 — 안전성(근거/재현성) 통과 여부. */
function ValidationBar({ result }: { result: AiDraftResult }) {
  const { validation } = result;
  if (validation.passed) {
    return (
      <p className="flex items-center gap-1.5 text-[11px] text-success">
        <ShieldCheck className="size-3.5" /> 안전성 검증 통과 — SOAP 근거 필수·CDSS 재현성 확인됨
      </p>
    );
  }
  const errors = validation.validators.flatMap((v) => v.errors);
  return (
    <div className="rounded-lg border border-danger/40 bg-danger/5 p-2">
      <p className="flex items-center gap-1.5 text-[11px] font-medium text-danger">
        <ShieldAlert className="size-3.5" /> 검증 실패 — 초안이 저장되지 않았습니다
      </p>
      {errors.slice(0, 3).map((e, i) => (
        <p key={i} className="mt-0.5 text-[10px] text-muted-foreground">• {e}</p>
      ))}
    </div>
  );
}
