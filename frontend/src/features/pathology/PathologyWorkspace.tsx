import { useEffect, useMemo, useRef, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { Microscope, Save, CheckCircle2, Reply, Eye, FileText, MessagesSquare, Clock, Loader2, Database, RefreshCw, Sparkles } from "lucide-react";
import type { Consult, ConsultStatus, PathologyResult } from "@/types";
import { CONSULT_STATUS_LABEL, URGENCY_LABEL } from "@/types";
import type { WsiAnalysisResult, WsiSlide, WsiStain } from "@/types/wsi";
import { useConsultStore } from "@/store/consultStore";
import { useNotificationStore } from "@/store/notificationStore";
import { useAuthStore } from "@/store/authStore";
import { useChatStore } from "@/store/chatStore";
import { eventBus } from "@/features/notification/events";
import { pathologyService } from "@/services/pathologyService";
import { wsiService, type CacheStatus, type ExtractStatus } from "@/services/wsiService";
import { PacsPanel } from "./PacsPanel";
import { consultStatusTone, urgencyTone } from "@/lib/statusTone";
import { formatDateTime, formatRelative } from "@/lib/format";
import { PageHeader } from "@/components/common/PageHeader";
import { StatusBadge } from "@/components/common/StatusBadge";
import { LoadingSpinner } from "@/components/common/LoadingSpinner";
import { EmptyState } from "@/components/common/EmptyState";
import { Timeline, type TimelineItem } from "@/components/common/Timeline";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Textarea, Label, Select } from "@/components/ui/input";
import { cn } from "@/lib/cn";
import { WSIViewer } from "./WSIViewer";
import { ToolboxRail } from "./ToolboxRail";
import { VunoResultPanel } from "./VunoResultPanel";
import { buildStandardizedReport } from "./reportDraft";
import { SignaturePad } from "@/components/ui/SignaturePad";
import { authService } from "@/services/authService";
import { AttentionHeatmap, RoiThumbnail, roiCropUrl } from "./wsiReportVisuals";

type StatusFilter = "all" | "pending" | "done";
type Tab = "viewer" | "report" | "consult" | "pacs";

const TABS: { key: Tab; label: string; icon: typeof Eye }[] = [
  { key: "viewer",  label: "Viewer",       icon: Eye          },
  { key: "report",  label: "Report",       icon: FileText     },
  { key: "consult", label: "Consultation", icon: MessagesSquare },
  { key: "pacs",    label: "PACS",         icon: Database     },
];

const STAIN_LABEL: Record<WsiStain, string> = { HE: "H&E", MT: "MT", PAS: "PAS" };

// 대기중 → 진행중 → 판독완료 → 회신완료 순 — 아직 처리 안 된 건이 목록 위로 오도록.
const STATUS_ORDER: Record<ConsultStatus, number> = { requested: 0, in_progress: 1, read: 2, replied: 3 };

function toTimeline(c: Consult): TimelineItem[] {
  return c.timeline.map((e) => ({ id: e.id, title: e.label, meta: formatDateTime(e.at), description: e.actor, done: true }));
}

/**
 * 병리과 판독 — WSI 뷰어·AI 분석·리포트가 핵심(탭 전환). 협진은 핵심 영역에서 내려
 * 별도 'Consultation' 탭으로 분리하고, 헤더 배지로 대기 건수만 알린다.
 */
export function PathologyWorkspace() {
  const allConsults  = useConsultStore((s) => s.items);
  const consults     = useMemo(() => allConsults.filter((c) => c.kind === "pathology"), [allConsults]);
  const loadConsults = useConsultStore((s) => s.load);
  const reply                = useConsultStore((s) => s.reply);
  const notify               = useNotificationStore((s) => s.notify);
  const user                 = useAuthStore((s) => s.user);
  const openForConsultReply  = useChatStore((s) => s.openForConsultReply);

  const [searchParams, setSearchParams] = useSearchParams();
  const [results, setResults]           = useState<PathologyResult[]>([]);
  const [loading, setLoading]           = useState(true);
  const [selectedId, setSelectedId]     = useState<string | null>(null);
  const [filter, setFilter]             = useState<StatusFilter>("all");
  const [tab, setTab]                   = useState<Tab>("viewer");

  // 보고서 편집 상태
  const [findings, setFindings]               = useState("");
  const [diagnosis, setDiagnosis]             = useState("");

  // 전자 서명(프로필에 없을 때 판독 화면에서 즉시 등록)
  const setUser                               = useAuthStore((s) => s.setUser);
  const [sigDraft, setSigDraft]               = useState<string | null>(null);
  const [savingSig, setSavingSig]             = useState(false);

  // 드로잉 도구
  const [tool, setTool]           = useState("pointer");
  const [drawColor, setDrawColor] = useState("#f97316");
  const [recommendation, setRecommendation]   = useState("");

  // ── WSI 상태 ──────────────────────────────────────────────────────────────
  const [wsiStain, setWsiStain]           = useState<WsiStain>("HE");
  const [wsiSlides, setWsiSlides]         = useState<WsiSlide[]>([]);
  const [wsiLoading, setWsiLoading]       = useState(false);
  const [wsiLoadError, setWsiLoadError]   = useState<string | null>(null);
  const [selectedSlide, setSelectedSlide] = useState<WsiSlide | null>(null);
  const [wsiResult, setWsiResult]         = useState<WsiAnalysisResult | null>(null);
  const [selectedTissue, setSelectedTissue] = useState<number>(0);
  const [wsiAnalyzing, setWsiAnalyzing]   = useState(false);
  const [cacheStatus, setCacheStatus]     = useState<CacheStatus | null>(null);
  const pollingRef                        = useRef<ReturnType<typeof setInterval> | null>(null);
  const [extractStatus, setExtractStatus] = useState<ExtractStatus | null>(null);
  const extractPollingRef                 = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    Promise.all([loadConsults(), pathologyService.list()])
      .then(([, r]) => setResults(r))
      .catch(() => {})
      .finally(() => setLoading(false));
  }, [loadConsults]);

  useEffect(() => {
    if (!selectedId && consults.length) setSelectedId(consults[0].id);
  }, [consults, selectedId]);

  // 헤더 배지 딥링크(?tab=consult)
  useEffect(() => {
    const t = searchParams.get("tab");
    if (t === "viewer" || t === "report" || t === "consult") {
      setTab(t);
      searchParams.delete("tab");
      setSearchParams(searchParams, { replace: true });
    }
  }, [searchParams, setSearchParams]);

  // 알림 딥링크(?consult=<id>)
  useEffect(() => {
    const wanted = searchParams.get("consult");
    if (!wanted || consults.length === 0) return;
    if (consults.some((c) => c.id === wanted)) {
      setSelectedId(wanted);
      setFilter("all");
      setTab("viewer");
    }
    searchParams.delete("consult");
    setSearchParams(searchParams, { replace: true });
  }, [searchParams, consults, setSearchParams]);

  const result  = useMemo(() => results.find((r) => r.consultId === selectedId) ?? null, [results, selectedId]);
  const consult = consults.find((c) => c.id === selectedId) ?? null;

  // DB 결과 변경 시 보고서 편집 상태 초기화
  useEffect(() => {
    setFindings(result?.report.findings ?? "");
    setDiagnosis(result?.report.diagnosis ?? "");
    setRecommendation("");
  }, [result]);

  // ── 슬라이드 캐시 polling ──────────────────────────────────────────────────
  function stopPolling() {
    if (pollingRef.current) {
      clearInterval(pollingRef.current);
      pollingRef.current = null;
    }
  }

  async function startPrepare(slide: WsiSlide) {
    stopPolling();
    // PACS 케이스는 로컬 manifest 전용 prepare/cache-status 대상이 아님(항상 error).
    // PacsDicomTileSource 가 dzi 요청 시 온디맨드로 다운로드하므로 바로 ready 취급.
    if (slide.is_pacs) {
      setCacheStatus({ status: "ready" });
      return;
    }
    if (slide.cached) {
      setCacheStatus({ status: "ready" });
      return;
    }
    setCacheStatus({ status: "downloading", downloaded_mb: 0 });
    try {
      const st = await wsiService.prepare(slide.slide_id);
      setCacheStatus(st);
      if (st.status === "ready") return;
    } catch {
      setCacheStatus({ status: "error", message: "prepare 실패" });
      return;
    }
    pollingRef.current = setInterval(async () => {
      try {
        const st = await wsiService.cacheStatus(slide.slide_id);
        setCacheStatus(st);
        if (st.status === "ready" || st.status === "error") stopPolling();
      } catch {
        stopPolling();
      }
    }, 3000);
  }

  useEffect(() => () => stopPolling(), []);

  // ── 피처(.pt) 추출 polling — has_features=false 슬라이드 전용 ───────────────
  function stopExtractPolling() {
    if (extractPollingRef.current) {
      clearInterval(extractPollingRef.current);
      extractPollingRef.current = null;
    }
  }

  async function startExtract(slide: WsiSlide) {
    stopExtractPolling();
    setExtractStatus({ status: "downloading" });
    try {
      const st = await wsiService.extractFeatures(slide.slide_id, wsiStain, slide.case_code);
      setExtractStatus(st);
      if (st.status === "ready" || st.status === "error") return;
    } catch {
      setExtractStatus({ status: "error", message: "피처 추출 시작 실패" });
      return;
    }
    extractPollingRef.current = setInterval(async () => {
      try {
        const st = await wsiService.extractStatus(slide.slide_id, wsiStain);
        setExtractStatus(st);
        if (st.status === "ready" || st.status === "error") stopExtractPolling();
      } catch {
        stopExtractPolling();
      }
    }, 2000);
  }

  useEffect(() => () => stopExtractPolling(), []);

  // (PAS 단일 stain 전환으로 검체 교차-stain 매칭 불필요 — HE 짝 없이도 PAS 분석 가능해 관련 로직 제거.)

  // Viewer 탭 진입 or stain 변경 → 슬라이드 목록 로드
  function loadSlides() {
    setWsiLoading(true);
    setWsiLoadError(null);
    setSelectedSlide(null);
    setWsiResult(null);
    setSelectedTissue(0);
    setCacheStatus(null);
    stopPolling();
    wsiService.listSlides(wsiStain)
      .then((r) => setWsiSlides(r.slides))
      .catch((e: unknown) => {
        setWsiSlides([]);
        const msg = e instanceof Error ? e.message : String(e);
        setWsiLoadError(msg);
      })
      .finally(() => setWsiLoading(false));
  }

  useEffect(() => {
    if (tab !== "viewer") return;
    loadSlides();
  }, [wsiStain, tab]);

  // 슬라이드 선택(드롭다운 onChange 와 매핑 자동선택이 공유하는 로직).
  function selectSlide(slide: WsiSlide | null) {
    setSelectedSlide(slide);
    setWsiResult(null);
    setSelectedTissue(0);
    stopExtractPolling();
    setExtractStatus(null);
    if (slide) startPrepare(slide);
  }

  // 환자(consult.patientMrn) ↔ WSI 슬라이드 매핑(chym.phase2_wsi_mapping, 수동 큐레이션) — 있으면
  // 그 환자 슬라이드만, 없으면(대부분의 데모 환자) 전체 목록으로 폴백.
  const [wsiMapping, setWsiMapping] = useState<{ slide_id: string; stain: string }[]>([]);

  // 매핑이 로드되면 매핑된 stain으로 자동 전환(HE 우선), 그 슬라이드 목록이 로드되면 자동 선택.
  useEffect(() => {
    if (wsiMapping.length === 0) return;
    const order: WsiStain[] = ["HE", "PAS", "MT"];
    const mappedStains = new Set(wsiMapping.map((m) => m.stain));
    const preferred = order.find((s) => mappedStains.has(s));
    if (preferred && preferred !== wsiStain) setWsiStain(preferred);
  }, [wsiMapping]);

  useEffect(() => {
    if (wsiMapping.length === 0 || wsiSlides.length === 0) return;
    const mappedId = wsiMapping.find((m) => m.stain === wsiStain)?.slide_id;
    if (!mappedId || selectedSlide?.slide_id === mappedId) return;
    const slide = wsiSlides.find((s) => s.slide_id === mappedId);
    if (slide) selectSlide(slide);
  }, [wsiSlides, wsiMapping, wsiStain]);

  useEffect(() => {
    if (!consult) { setWsiMapping([]); return; }
    pathologyService.getWsiMapping(consult.patientMrn).then(setWsiMapping);
  }, [consult?.id]);

  // 요약 스트립용 — HE/MT/PAS 각각 이미 분석(캐시)된 결과가 있으면 조회(새 분석은 안 돌림).
  // Viewer 탭 진입과 무관하게, 매핑만 있으면 바로 채워진다.
  const [stainResults, setStainResults] = useState<Partial<Record<WsiStain, WsiAnalysisResult>>>({});
  useEffect(() => {
    setStainResults({});
    for (const { slide_id, stain } of wsiMapping) {
      const s = stain as WsiStain;
      wsiService.getResult(s, slide_id)
        .then((r) => setStainResults((prev) => ({ ...prev, [s]: r })))
        .catch(() => {}); // 아직 분석 안 됨/캐시 없음 — 조용히 무시
    }
  }, [wsiMapping]);

  // PAS 는 HE 없이는 CdssEngine이 항상 ABSTAIN 이라, 실제로 결과가 나올 수 있는(HE 짝이 있는)
  // 슬라이드만 드롭다운에 보여준다. HE/MT 는 단독 분석(ABMIL) 가능하니 그대로 전부 노출.
  const visibleWsiSlides = useMemo(() => {
    const mappedIds = new Set(
      wsiMapping.filter((m) => m.stain === wsiStain).map((m) => m.slide_id),
    );
    let list = wsiSlides;
    if (mappedIds.size > 0) list = list.filter((s) => mappedIds.has(s.slide_id));
    return list;   // PAS 단일 stain — HE 짝 불필요, 전체 노출(HE/MT 와 동일)
  }, [wsiSlides, wsiStain, wsiMapping]);

  const filtered = consults
    .filter((c) => {
      if (filter === "pending") return c.status === "requested" || c.status === "in_progress";
      if (filter === "done")    return c.status === "read" || c.status === "replied";
      return true;
    })
    .slice()
    .sort((a, b) => STATUS_ORDER[a.status as ConsultStatus] - STATUS_ORDER[b.status as ConsultStatus]);

  const pendingCount = useMemo(
    () => consults.filter((c) => c.status === "requested" || c.status === "in_progress").length,
    [consults],
  );

  async function saveDraft(status: "draft" | "final") {
    if (!consult) return;
    try {
      const saved = await pathologyService.saveReport(consult.id, { findings, diagnosis, status });
      setResults((prev) => {
        const idx = prev.findIndex((r) => r.consultId === consult.id);
        if (idx === -1) return [...prev, saved];
        const next = [...prev];
        next[idx] = saved;
        return next;
      });
      eventBus.publish({ type: "pathology.readSaved", patientName: consult.patientName });

      // 판독 완료 시: 요청한 신장내과 전문의에게 병리 리포트를 자동 회신(전송).
      // 이미 회신된 건은 중복 전송하지 않는다(재전송은 Consultation 탭 수동 회신 사용).
      if (status === "final" && consult.status !== "replied") {
        // 신장내과 수신 리포트에 서명·ROI·heatmap·신뢰도를 함께 싣는다(회신에 snapshot).
        const analysis = (["HE", "MT", "PAS"] as WsiStain[])
          .filter((s) => stainResults[s])
          .map((s) => {
            const r = stainResults[s]!;
            return {
              stain: s,
              slideId: r.slide_id,
              status: r.report.status,
              uncertainty: r.report.uncertainty ?? null,
              confidences: r.metrics.filter((m) => m.confidence != null).map((m) => ({ label: m.label, value: m.confidence as number })),
              overlays: r.attn_overlays.map((o) => ({ cx: o.cx, cy: o.cy, r: o.r, weight: o.weight, px: o.px, py: o.py, psize: o.psize })),
            };
          });
        await reply(consult.id, {
          findings,
          diagnosis,
          recommendation: recommendation.trim() || "병리 판독 결과 회신 — 임상 소견과 종합 판단 바랍니다.",
          author: user?.name ?? "병리과",
          repliedAt: new Date().toISOString(),
          signaturePath: user?.signaturePath ?? null,
          analysis,
        });
        eventBus.publish({ type: "pathology.resultArrived", patientName: consult.patientName, mrn: consult.patientMrn });
        notify({
          severity: "INFO", department: "pathology", title: "판독 완료 · 자동 회신",
          message: `${consult.patientName} 병리 판독을 ${consult.requestedBy}님(신장내과)께 전송했습니다.`,
        });
      } else {
        notify({
          severity: "INFO", department: "pathology",
          title: status === "final" ? "판독 완료" : "임시 저장",
          message: `${consult.patientName} 소견/최종 진단을 저장했습니다.`,
        });
      }
    } catch {
      notify({ severity: "CRITICAL", department: "pathology", title: "저장 실패", message: "병리 보고서 저장 중 오류가 발생했습니다." });
    }
  }

  /**
   * 캐시된 WSI 정량 분석(stainResults)의 **실제 모델 descriptor** 로 표준 병리 보고서
   * 초안을 자동 작성. 소견=Specimen·Gross·Microscopic(구획별), 진단=Diagnosis·Banff·Comment.
   * %→Banff 매핑·미평가 항목 표기는 reportDraft.buildStandardizedReport 참조.
   */
  function autoDraftReport() {
    if (!consult) return;
    const stains = (["HE", "MT", "PAS"] as WsiStain[]).filter((s) => stainResults[s]);
    if (stains.length === 0) {
      notify({
        severity: "WARNING", department: "pathology", title: "분석 결과 없음",
        message: "먼저 Viewer 탭에서 WSI 분석을 실행하세요. 캐시된 정량 분석이 있어야 자동 작성이 가능합니다.",
      });
      return;
    }
    if ((findings.trim() || diagnosis.trim()) &&
        !window.confirm("이미 입력된 소견/최종 진단이 있습니다. AI 초안으로 덮어쓸까요?")) return;

    const draft = buildStandardizedReport({
      patientName: consult.patientName,
      patientMrn: consult.patientMrn,
      results: stainResults,
    });
    setFindings(draft.findings);
    setDiagnosis(draft.diagnosis);
    notify({
      severity: "INFO", department: "pathology", title: "AI 표준 보고서 초안 작성",
      message: `${stains.map((s) => STAIN_LABEL[s]).join("·")} 정량 분석 기반 표준 보고서 초안을 작성했습니다. 검토 후 판독 완료하세요.`,
    });
  }

  /** stain 결과에서 대표 병변 ROI 크롭 URL(공유 헬퍼). */
  function stainRoiUrl(s: WsiStain): string | null {
    const r = stainResults[s];
    return r ? roiCropUrl(s, r.slide_id, r.attn_overlays) : null;
  }

  /** 프로필에 서명이 없을 때 판독 화면에서 즉시 등록(회원가입 서명 경로 재사용). */
  async function saveSignature() {
    if (!sigDraft) return;
    setSavingSig(true);
    try {
      const updated = await authService.updateSignature(sigDraft);
      setUser(updated);
      setSigDraft(null);
      notify({ severity: "INFO", department: "pathology", title: "전자 서명 등록", message: "판독의 서명이 프로필에 저장되었습니다." });
    } catch {
      notify({ severity: "CRITICAL", department: "pathology", title: "서명 저장 실패", message: "서명 저장 중 오류가 발생했습니다." });
    } finally {
      setSavingSig(false);
    }
  }

  function sendReply() {
    if (!consult) return;
    reply(consult.id, {
      findings,
      diagnosis,
      recommendation,
      author: user?.name ?? "병리과",
      repliedAt: new Date().toISOString(),
    });
    eventBus.publish({ type: "pathology.resultArrived", patientName: consult.patientName, mrn: consult.patientMrn });
    notify({ severity: "INFO", department: "pathology", title: "협진 회신 전송", message: `${consult.patientName} 협진 회신을 전송했습니다.` });

    // 회신 내용을 채팅 초안으로 채워 요청자와 채팅 창을 자동으로 연결
    const draft = [
      `[협진 회신 · ${consult.patientName}(${consult.patientMrn})]`,
      `소견: ${findings}`,
      `진단: ${diagnosis}`,
      `권고: ${recommendation}`,
    ].join("\n");
    openForConsultReply(consult.requestedBy, draft);
  }

  async function runAnalysis() {
    if (!selectedSlide) return;
    if (!selectedSlide.has_features && extractStatus?.status !== "ready") {   // 버튼 disabled 와 동일 조건 — 추출 완료(ready)면 백엔드가 방금 만든 .pt 를 읽어 분석 가능
      notify({ severity: "WARNING", department: "pathology", title: "AI 분석 불가",
               message: "이 슬라이드는 사전 임베딩이 없어 AI 분석을 실행할 수 없습니다 (PACS/신규 슬라이드). 뷰어 열람만 가능합니다." });
      return;
    }
    setWsiAnalyzing(true);
    try {
      const r = await wsiService.analyze(selectedSlide.slide_id, wsiStain);
      setWsiResult(r);
      setStainResults((prev) => ({ ...prev, [wsiStain]: r }));
    } catch {
      notify({ severity: "CRITICAL", department: "pathology", title: "분석 실패", message: "WSI 분석 중 오류가 발생했습니다." });
    } finally {
      setWsiAnalyzing(false);
    }
  }

  // 병리과 화면 전용 — pathology/admin 외 역할은 차단(UX). 실제 접근 차단은 백엔드 8001/8010 가 강제.
  if (user && user.role !== "pathology" && user.role !== "admin") {
    return (
      <div className="mx-auto max-w-[1400px]">
        <PageHeader title="병리과 판독" subtitle="WSI 판독 · AI 정량 분석 · 리포트 작성" />
        <div className="mt-12 text-center text-sm text-muted-foreground">
          병리과 권한이 필요한 화면입니다. (현재 권한: {user.role})
        </div>
      </div>
    );
  }

  if (loading) return <LoadingSpinner label="판독 목록 불러오는 중" />;

  const reportReady = !!findings.trim() && !!diagnosis.trim();
  // 병리 리포트는 반드시 AI 분석이 끝난 뒤에만 작성 가능(요구사항).
  // 분석 완료 = stain 결과가 하나 이상 존재 + 현재 분석 진행 중이 아님.
  const analyzedStains = (["HE", "MT", "PAS"] as WsiStain[]).filter((s) => stainResults[s]);
  const analysisComplete = analyzedStains.length > 0 && !wsiAnalyzing;

  return (
    <div className="mx-auto max-w-[1400px]">
      <PageHeader title="병리과 판독" subtitle="WSI 판독 · AI 정량 분석 · 리포트 작성" />

      {/* 탭 */}
      <div className="mb-4 flex flex-wrap gap-1 border-b border-border">
        {TABS.map((t) => {
          const active = tab === t.key;
          return (
            <button
              key={t.key}
              onClick={() => setTab(t.key)}
              className={cn(
                "-mb-px flex items-center gap-1.5 border-b-2 px-3 py-2 text-sm font-medium transition-colors",
                active ? "border-primary text-primary" : "border-transparent text-muted-foreground hover:text-foreground",
              )}
            >
              <t.icon className="size-4" />
              {t.label}
              {t.key === "consult" && pendingCount > 0 && (
                <span className="ml-0.5 flex min-w-4 items-center justify-center rounded-full bg-destructive px-1 text-[9px] font-bold text-white">
                  {pendingCount}
                </span>
              )}
            </button>
          );
        })}
      </div>

      <div className="grid grid-cols-1 gap-4 xl:grid-cols-[240px_1fr]">
        {/* 좌 레일: 판독 목록 + Annotation */}
        <div className="flex flex-col gap-4">
          <Card className="h-fit">
            <CardHeader className="gap-2">
              <CardTitle>판독 목록</CardTitle>
              <div className="flex gap-1">
                {(["all", "pending", "done"] as StatusFilter[]).map((f) => (
                  <button
                    key={f}
                    onClick={() => setFilter(f)}
                    className={cn(
                      "rounded-md px-2 py-1 text-[11px] font-medium",
                      filter === f ? "bg-primary text-primary-foreground" : "bg-secondary text-muted-foreground",
                    )}
                  >
                    {f === "all" ? "전체" : f === "pending" ? "대기/진행" : "완료"}
                  </button>
                ))}
              </div>
            </CardHeader>
            <CardContent className="p-2">
              {filtered.length === 0 ? (
                <EmptyState title="판독 건이 없습니다" />
              ) : (
                filtered.map((c: Consult) => (
                  <button
                    key={c.id}
                    onClick={() => setSelectedId(c.id)}
                    className={cn(
                      "mb-1 flex w-full flex-col gap-1 rounded-md border px-3 py-2 text-left transition-colors",
                      c.id === selectedId ? "border-primary/40 bg-primary/5" : "border-transparent hover:bg-muted/60",
                    )}
                  >
                    <div className="flex items-center justify-between gap-1">
                      <span className="text-sm font-medium text-foreground">{c.patientName}</span>
                      <StatusBadge label={CONSULT_STATUS_LABEL[c.status as ConsultStatus]} tone={consultStatusTone[c.status]} />
                    </div>
                    <span className="truncate text-[11px] text-muted-foreground">{c.diagnosis}</span>
                    <div className="flex items-center justify-between">
                      <StatusBadge label={URGENCY_LABEL[c.urgency]} tone={urgencyTone[c.urgency]} />
                      <span className="text-[10px] text-muted-foreground/70">{formatRelative(c.requestedAt)}</span>
                    </div>
                  </button>
                ))
              )}
            </CardContent>
          </Card>

          {tab === "viewer" && (
            <Card className="h-fit">
              <CardHeader><CardTitle>Annotation</CardTitle></CardHeader>
              <CardContent>
                <ToolboxRail
                  tool={tool}
                  onToolChange={setTool}
                  drawColor={drawColor}
                  onColorChange={setDrawColor}
                />
              </CardContent>
            </Card>
          )}
        </div>

        {/* 메인: 탭 콘텐츠 */}
        {consult ? (
          <div className="flex min-w-0 flex-col gap-4">
            {/* 요약 스트립 — Report 탭만 HE/MT/PAS 정량 지표 전체(캐시된 분석, 없으면 미분석).
                그 외(Viewer/Consultation/PACS)는 맥락상 무관해서 환자 카드만. */}
            {tab !== "report" ? (
              <div className="rounded-lg border border-border bg-card px-3 py-2">
                <p className="text-[10px] text-muted-foreground">환자</p>
                <p className="text-base font-bold text-foreground">{consult.patientName}</p>
                <p className="text-[10px] text-muted-foreground/70">{consult.patientMrn}</p>
              </div>
            ) : (
              <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
                <div className="rounded-lg border border-border bg-card px-3 py-2">
                  <p className="text-[10px] text-muted-foreground">환자</p>
                  <p className="text-base font-bold text-foreground">{consult.patientName}</p>
                  <p className="text-[10px] text-muted-foreground/70">{consult.patientMrn}</p>
                </div>
                {(["HE", "MT", "PAS"] as WsiStain[]).map((s) => {
                  const r = stainResults[s];
                  return (
                    <div key={s} className="rounded-lg border border-border bg-card px-3 py-2">
                      <p className="mb-1 text-[10px] text-muted-foreground">{STAIN_LABEL[s]}</p>
                      {r ? (
                        <div className="flex flex-col gap-0.5">
                          {r.metrics.map((m) => (
                            <div key={m.key} className="flex items-center justify-between gap-2 text-[11px]">
                              <span className="text-muted-foreground">{m.label}</span>
                              <span className="font-semibold text-foreground">
                                {m.value !== null ? `${m.value.toFixed(1)}${m.unit}` : "—"}
                              </span>
                            </div>
                          ))}
                        </div>
                      ) : (
                        <p className="text-xs text-muted-foreground">미분석</p>
                      )}
                    </div>
                  );
                })}
              </div>
            )}

            {/* ① Viewer + AI Analysis (통합) */}
            {tab === "viewer" && (
              <Card>
                <CardHeader className="flex-row flex-wrap items-center justify-between gap-2">
                  <CardTitle>WSI Viewer</CardTitle>
                  <div className="flex items-center gap-2">
                    <Select
                      value={wsiStain}
                      onChange={(e) => setWsiStain(e.target.value as WsiStain)}
                      className="h-8 w-20 text-xs"
                    >
                      <option value="HE">H&amp;E</option>
                      <option value="MT">MT</option>
                      <option value="PAS">PAS</option>
                    </Select>
                    <Select
                      value={selectedSlide?.slide_id ?? ""}
                      onChange={(e) => {
                        const slide = visibleWsiSlides.find((s) => s.slide_id === e.target.value) ?? null;
                        selectSlide(slide);
                      }}
                      className="h-8 w-44 text-xs"
                      disabled={wsiLoading}
                    >
                      <option value="">
                        {wsiLoading ? "불러오는 중..." : visibleWsiSlides.length === 0 ? "슬라이드 없음" : "슬라이드 선택"}
                      </option>
                      {visibleWsiSlides.map((s) => (
                        <option key={s.slide_id} value={s.slide_id}>
                          {s.case_code || s.slide_id}{s.cached ? "" : " ↓"}
                        </option>
                      ))}
                    </Select>
                    
                    {/* Tissue 분리 검출에 따른 Tissue 선택 UI */}
                    {wsiResult?.tissues && wsiResult.tissues.length > 1 && (
                      <Select
                        value={selectedTissue}
                        onChange={(e) => setSelectedTissue(Number(e.target.value))}
                        className="h-8 w-32 text-xs border-amber-500/30 text-amber-600 font-semibold bg-amber-500/5 hover:bg-amber-500/10"
                        title="원하는 신장 조직 영역을 선택하세요"
                      >
                        {wsiResult.tissues.map((t, idx) => (
                          <option key={t.id} value={idx}>
                            Tissue {idx + 1} ({Math.round(t.area / 1000000)}M px)
                          </option>
                        ))}
                      </Select>
                    )}

                    <Button
                      size="sm"
                      variant="outline"
                      onClick={loadSlides}
                      disabled={wsiLoading}
                      title="슬라이드 목록 새로고침"
                    >
                      {wsiLoading ? <Loader2 className="size-3.5 animate-spin" /> : <RefreshCw className="size-3.5" />}
                    </Button>
                    <Button
                      size="sm"
                      onClick={runAnalysis}
                      disabled={!selectedSlide || wsiAnalyzing
                        || (!selectedSlide?.has_features && extractStatus?.status !== "ready")}
                    >
                      {wsiAnalyzing && <Loader2 className="size-3.5 animate-spin" />}
                      AI 분석
                    </Button>
                  </div>
                </CardHeader>
                <CardContent>
                  <div className="flex gap-4">
                    <div className="min-w-0 flex-1">
                      {/* PACS 슬라이드 목록 로드 오류 */}
                      {wsiLoadError && (
                        <div className="mb-2 rounded-md border border-destructive/30 bg-destructive/5 px-3 py-2 text-xs text-destructive">
                          슬라이드 목록 오류: {wsiLoadError}
                        </div>
                      )}
                      {/* 다운로드 진행 중 오버레이 */}
                      {selectedSlide && cacheStatus && cacheStatus.status === "downloading" && (
                        <div className="mb-2 flex items-center gap-2 rounded-md border border-border bg-muted/60 px-3 py-2 text-xs text-muted-foreground">
                          <Loader2 className="size-3.5 animate-spin shrink-0" />
                          PACS에서 슬라이드 다운로드 중...
                          {"downloaded_mb" in cacheStatus && (
                            <span className="ml-auto font-medium text-foreground">{cacheStatus.downloaded_mb} MB</span>
                          )}
                        </div>
                      )}
                      {selectedSlide && cacheStatus?.status === "error" && (
                        <div className="mb-2 rounded-md border border-destructive/30 bg-destructive/5 px-3 py-2 text-xs text-destructive">
                          {"message" in cacheStatus ? cacheStatus.message : "다운로드 오류"}
                        </div>
                      )}
                      {/* 피처(.pt) 없는 슬라이드 — 온디맨드 패치추출 트리거/진행률 */}
                      {selectedSlide && !selectedSlide.has_features && extractStatus?.status !== "ready" && (
                        <div className="mb-2 flex items-center gap-2 rounded-md border border-amber-500/30 bg-amber-500/5 px-3 py-2 text-xs">
                          {!extractStatus || extractStatus.status === "not_started" ? (
                            <>
                              <span className="text-muted-foreground">AI 분석용 피처가 없는 슬라이드입니다.</span>
                              <Button
                                size="sm" variant="outline" className="ml-auto h-6 px-2 text-[10px]"
                                onClick={() => startExtract(selectedSlide)}
                              >
                                피처 추출 시작
                              </Button>
                            </>
                          ) : extractStatus.status === "error" ? (
                            <span className="text-destructive">{extractStatus.message}</span>
                          ) : (
                            <>
                              <Loader2 className="size-3.5 animate-spin shrink-0 text-amber-600" />
                              <span className="text-muted-foreground">
                                {extractStatus.status === "downloading" && "PACS 다운로드 중..."}
                                {extractStatus.status === "extracting" && "조직 패치 탐색 중..."}
                                {extractStatus.status === "encoding" && "패치 인코딩 중..."}
                              </span>
                              {extractStatus.status === "encoding" && (
                                <span className="ml-auto font-medium text-foreground">
                                  {extractStatus.progress}/{extractStatus.total}
                                </span>
                              )}
                            </>
                          )}
                        </div>
                      )}
                      <WSIViewer
                        stain={wsiStain}
                        slideId={selectedSlide?.slide_id}
                        dziUrl={selectedSlide && cacheStatus?.status === "ready"
                          ? wsiService.pacsDziUrl(selectedSlide.slide_id)
                          : undefined}
                        attnOverlays={wsiResult?.attn_overlays}
                        metrics={wsiResult?.metrics}
                        tool={tool}
                        drawColor={drawColor}
                        tissues={wsiResult?.tissues}
                        selectedTissue={selectedTissue}
                        slide_w={wsiResult?.slide_w}
                        slide_h={wsiResult?.slide_h}
                      />
                      <p className="mt-2 text-[11px] text-muted-foreground">
                        휠: 확대/축소 · 드래그: 이동
                      </p>
                    </div>
                    <div className="w-56 shrink-0 border-l border-border pl-4">
                      <VunoResultPanel
                        result={wsiResult}
                        selectedStain={wsiStain}
                      />
                    </div>
                  </div>
                </CardContent>
              </Card>
            )}

            {/* v6.1 예측 패널은 retrieval 전환으로 legacy/ 로 이동(숨김). features/pathology/legacy/ 참조. */}

            {/* PACS — 실 PACS 게이트웨이 연동(병리과 전용, BFF). 전체 기능은 PacsPanel. */}
            {tab === "pacs" && <PacsPanel />}

            {/* ② Report */}
            {tab === "report" && (
              <Card>
                <CardHeader className="flex-row flex-wrap items-center justify-between gap-2">
                  <CardTitle>병리 보고서</CardTitle>
                  <Button
                    variant="subtle"
                    size="sm"
                    onClick={autoDraftReport}
                    disabled={!analysisComplete}
                    title={analysisComplete ? "AI 분석 결과로 소견·진단 초안을 자동 작성" : "AI 분석 완료 후 사용 가능"}
                  >
                    <Sparkles className="size-3.5" /> AI 자동 작성
                  </Button>
                </CardHeader>
                <CardContent className="flex flex-col gap-3">
                  {/* 분석 완료 게이트 — 반드시 AI 분석이 끝난 뒤에만 보고서 작성 가능 */}
                  {!analysisComplete && (
                    <div className="flex items-center gap-2 rounded-lg border border-warning/40 bg-warning/5 px-3 py-2 text-[11px] text-warning">
                      <Clock className="size-4 shrink-0" />
                      <span>
                        {wsiAnalyzing
                          ? "AI 분석이 진행 중입니다. 분석이 끝나면 보고서를 작성할 수 있습니다."
                          : "먼저 Viewer 탭에서 AI 분석을 실행·완료하세요. 분석이 끝나야 병리 리포트를 작성할 수 있습니다."}
                      </span>
                    </div>
                  )}
                  {/* ① 검사 정보 (Header) — 협진 정보에서 도출(읽기 전용) */}
                  <div className="rounded-lg border border-border bg-muted/30 p-3">
                    <p className="mb-1.5 text-[11px] font-semibold text-muted-foreground">① 검사 정보</p>
                    <dl className="grid grid-cols-2 gap-x-4 gap-y-1 text-[11px] sm:grid-cols-3">
                      <div><dt className="inline text-muted-foreground">환자 </dt><dd className="inline font-medium text-foreground">{consult.patientName}</dd></div>
                      <div><dt className="inline text-muted-foreground">등록번호 </dt><dd className="inline font-medium text-foreground">{consult.patientMrn}</dd></div>
                      <div><dt className="inline text-muted-foreground">검체 </dt><dd className="inline font-medium text-foreground">Kidney, needle biopsy</dd></div>
                      <div><dt className="inline text-muted-foreground">의뢰 </dt><dd className="inline font-medium text-foreground">{consult.requestedBy}</dd></div>
                      <div className="col-span-2 sm:col-span-1"><dt className="inline text-muted-foreground">의뢰일 </dt><dd className="inline font-medium text-foreground">{formatDateTime(consult.requestedAt)}</dd></div>
                    </dl>
                  </div>
                  <div>
                    <Label>② 소견 · Microscopic Description</Label>
                    <p className="mb-1 text-[10px] text-muted-foreground">Specimen · Gross · 구획별(사구체/세뇨관/간질/혈관) 현미경 소견</p>
                    <Textarea value={findings} onChange={(e) => setFindings(e.target.value)} disabled={!analysisComplete} placeholder="AI 자동 작성 또는 직접 기술하세요 — 사구체/세뇨관/간질/혈관 순." className="min-h-40 font-mono text-xs leading-relaxed" />
                  </div>
                  {/* 대표 병변 이미지 — stain별 최상위 attention ROI 크롭 */}
                  <div>
                    <Label>대표 병변 이미지 (AI attention ROI)</Label>
                    <div className="mt-1 grid grid-cols-3 gap-2">
                      {(["HE", "MT", "PAS"] as WsiStain[]).map((s) => (
                        <RoiThumbnail key={s} label={STAIN_LABEL[s]} url={stainRoiUrl(s)} analyzed={!!stainResults[s]} />
                      ))}
                    </div>
                    <p className="mt-1 text-[10px] text-muted-foreground/70">모델 attention 최상위 패치 영역(정확 좌표 있는 경우) — 병리 전문의 확인용.</p>
                  </div>

                  {/* Attention Heatmap + AI 예측 신뢰도 — stain별 실제 분석 결과 */}
                  <div>
                    <Label>Attention Heatmap · AI 예측 신뢰도 (stain별)</Label>
                    <div className="mt-1 grid grid-cols-3 gap-2">
                      {(["HE", "MT", "PAS"] as WsiStain[]).map((s) => {
                        const r = stainResults[s];
                        const status = r?.report.status;
                        return (
                          <div key={s} className="flex flex-col gap-1">
                            <AttentionHeatmap label={`${STAIN_LABEL[s]}`} slideId={r?.slide_id} overlays={r?.attn_overlays ?? []} />
                            {r ? (
                              <div className="rounded border border-border px-1.5 py-1 text-[10px]">
                                <span className={cn("font-semibold", status === "ALLOW" ? "text-success" : "text-warning")}>
                                  {status === "ALLOW" ? "신뢰 확보 (ALLOW)" : "신뢰 부족 (ABSTAIN)"}
                                </span>
                                {(() => {
                                  const confs = r.metrics.filter((m) => m.confidence != null);
                                  if (confs.length > 0) {
                                    return (
                                      <dl className="mt-0.5 flex flex-col gap-0.5">
                                        {confs.map((m) => (
                                          <div key={m.key} className="flex justify-between gap-1">
                                            <dt className="truncate text-muted-foreground">{m.label}</dt>
                                            <dd className="shrink-0 font-medium tabular-nums text-foreground">{Math.round((m.confidence ?? 0) * 100)}%</dd>
                                          </div>
                                        ))}
                                        {r.report.uncertainty != null && (
                                          <p className="text-muted-foreground/70">슬라이드 불확실성 {r.report.uncertainty.toFixed(2)}</p>
                                        )}
                                      </dl>
                                    );
                                  }
                                  return <p className="mt-0.5 text-muted-foreground/70">회귀 예측 · descriptor 신뢰도 미제공(ABMIL)</p>;
                                })()}
                              </div>
                            ) : (
                              <p className="text-center text-[10px] text-muted-foreground">미분석</p>
                            )}
                          </div>
                        );
                      })}
                    </div>
                    <p className="mt-1 text-[10px] text-muted-foreground/70">PAS(CDSS)는 descriptor별 보정 신뢰도 + 슬라이드 불확실성을 제공합니다. HE/MT(ABMIL)는 회귀 모델이라 신뢰도가 없어 예측값(위 소견)만 사용합니다.</p>
                  </div>

                  <div>
                    <Label>③ 최종 진단 · Diagnosis &amp; Banff</Label>
                    <p className="mb-1 text-[10px] text-muted-foreground">진단 결론 + Banff 점수(모델 평가 축만) + Comment</p>
                    <Textarea value={diagnosis} onChange={(e) => setDiagnosis(e.target.value)} disabled={!analysisComplete} placeholder="최종 병리 진단·Banff·Comment를 입력하세요." className="min-h-32 font-mono text-xs leading-relaxed" />
                  </div>
                  {/* ④ 판독의 전자 서명 — 회원가입 시 등록한 프로필 서명 활용(없으면 즉시 등록) */}
                  <div className="rounded-lg border border-border p-3">
                    <p className="mb-2 text-[11px] font-semibold text-muted-foreground">④ 판독의 전자 서명</p>
                    <div className="flex flex-wrap items-end justify-between gap-3">
                      <div className="text-[11px] text-muted-foreground">
                        <p>판독의 <span className="font-medium text-foreground">{user?.name ?? "—"}</span></p>
                        <p>{user?.department ?? "병리과"}</p>
                        <p className="mt-1 text-[10px] text-muted-foreground/70">판독 완료 시 본 서명이 보고서에 적용됩니다.</p>
                      </div>
                      {user?.signaturePath ? (
                        <img
                          src={user.signaturePath}
                          alt="전자 서명"
                          className="h-14 max-w-[180px] rounded border border-border bg-white object-contain px-2"
                        />
                      ) : (
                        <div className="flex flex-col items-end gap-1.5">
                          <span className="text-[10px] text-warning">프로필에 서명이 없습니다 — 아래에 서명 후 등록하세요.</span>
                          <SignaturePad onSign={setSigDraft} />
                          <Button variant="subtle" size="sm" onClick={saveSignature} disabled={!sigDraft || savingSig}>
                            <Save className="size-3.5" /> {savingSig ? "저장 중…" : "서명 등록"}
                          </Button>
                        </div>
                      )}
                    </div>
                  </div>
                  <div className="flex flex-wrap items-center justify-end gap-2">
                    {!reportReady && (
                      <span className="mr-auto text-[11px] text-muted-foreground">소견·최종 진단을 입력하면 Consultation 탭에서 협진 회신을 보낼 수 있습니다.</span>
                    )}
                    <Button variant="outline" size="sm" onClick={() => saveDraft("draft")} disabled={!analysisComplete}>
                      <Save className="size-3.5" /> 임시 저장
                    </Button>
                    <Button
                      variant="success" size="sm" onClick={() => saveDraft("final")}
                      disabled={!user?.signaturePath || !analysisComplete}
                      title={!analysisComplete ? "AI 분석 완료 후 판독할 수 있습니다" : !user?.signaturePath ? "프로필에 서명이 없어 판독 완료할 수 없습니다" : undefined}
                    >
                      <CheckCircle2 className="size-3.5" /> 판독 완료
                    </Button>
                  </div>
                </CardContent>
              </Card>
            )}

            {/* ③ Consultation */}
            {tab === "consult" && (
              <Card>
                <CardHeader className="flex-row flex-wrap items-center gap-2">
                  <MessagesSquare className="size-4 text-primary" />
                  <CardTitle>협진 회신</CardTitle>
                  <StatusBadge label={CONSULT_STATUS_LABEL[consult.status]} tone={consultStatusTone[consult.status]} dot />
                  <StatusBadge label={URGENCY_LABEL[consult.urgency]} tone={urgencyTone[consult.urgency]} />
                </CardHeader>
                <CardContent className="flex flex-col gap-3">
                  <div className="text-[11px] text-muted-foreground">
                    {consult.patientMrn} · 요청 {formatDateTime(consult.requestedAt)} · {consult.requestedBy}
                  </div>
                  <div>
                    <p className="text-[11px] font-semibold text-muted-foreground">주요 검사</p>
                    <p className="text-sm text-foreground">{consult.keyLabs}</p>
                  </div>
                  <div>
                    <p className="text-[11px] font-semibold text-muted-foreground">요청 사유</p>
                    <p className="text-sm leading-relaxed text-foreground">{consult.reason}</p>
                  </div>

                  <Timeline items={toTimeline(consult)} />

                  {consult.reply ? (
                    <div className="rounded-lg border border-success/30 bg-success/5 p-3">
                      <p className="mb-2 text-[11px] font-semibold text-success">회신 완료 · {consult.reply.author}</p>
                      <dl className="flex flex-col gap-2 text-xs">
                        <div>
                          <dt className="text-[11px] text-muted-foreground">소견</dt>
                          <dd className="text-foreground">{consult.reply.findings}</dd>
                        </div>
                        <div>
                          <dt className="text-[11px] text-muted-foreground">최종 진단</dt>
                          <dd className="font-medium text-foreground">{consult.reply.diagnosis}</dd>
                        </div>
                        <div>
                          <dt className="text-[11px] text-muted-foreground">권고사항</dt>
                          <dd className="text-foreground">{consult.reply.recommendation}</dd>
                        </div>
                      </dl>
                      <p className="mt-2 text-[10px] text-muted-foreground">회신 {formatDateTime(consult.reply.repliedAt)}</p>
                    </div>
                  ) : (
                    <>
                      <div>
                        <Label>권고사항 (협진 회신용)</Label>
                        <Textarea value={recommendation} onChange={(e) => setRecommendation(e.target.value)} placeholder="신장내과에 전달할 권고사항." className="min-h-16" />
                      </div>
                      <div className="flex flex-wrap items-center justify-end gap-2">
                        {!reportReady && (
                          <span className="mr-auto flex items-center gap-1 text-[11px] text-warning">
                            <Clock className="size-3.5" /> Report 탭에서 소견·최종 진단을 먼저 입력하세요.
                          </span>
                        )}
                        <Button size="sm" onClick={sendReply} disabled={!reportReady || !recommendation.trim()}>
                          <Reply className="size-3.5" /> 협진 회신
                        </Button>
                      </div>
                    </>
                  )}
                </CardContent>
              </Card>
            )}
          </div>
        ) : (
          <Card>
            <CardContent>
              <EmptyState icon={Microscope} title="판독 건을 선택하세요" description="좌측 목록에서 판독할 건을 선택하면 WSI 와 분석 결과가 표시됩니다." />
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  );
}
