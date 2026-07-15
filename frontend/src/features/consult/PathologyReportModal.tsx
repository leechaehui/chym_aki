import type { ReactNode } from "react";
import { FileText } from "lucide-react";
import type { Consult } from "@/types";
import type { WsiStain } from "@/types/wsi";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { formatDateTime } from "@/lib/format";
import { AttentionHeatmap, RoiThumbnail, roiCropUrl } from "@/features/pathology/wsiReportVisuals";

/** 리포트 섹션(검증 리포트와 동일한 번호·구분선 스타일). */
function Section({ index, title, subtitle, children }: { index: number; title: string; subtitle?: string; children: ReactNode }) {
  return (
    <section className="mt-5 first:mt-1">
      <div className="mb-2 flex items-baseline gap-2 border-b border-border pb-1.5">
        <span className="flex size-5 items-center justify-center rounded-full bg-primary/10 text-xs font-bold text-primary">{index}</span>
        <h3 className="text-base font-semibold text-foreground">{title}</h3>
        {subtitle && <span className="text-xs font-normal text-muted-foreground">· {subtitle}</span>}
      </div>
      {children}
    </section>
  );
}

/**
 * 병리 판독 리포트(신장내과 수신용) — 병리과가 회신한 소견·진단·권고를 정식 리포트 형식으로 표시.
 * 병리 리포트 본문은 표준 포맷(Specimen·Microscopic·Diagnosis·Banff·Comment)이라 pre-wrap 로 구조 유지.
 */
export function PathologyReportModal({
  consult,
  open,
  onOpenChange,
}: {
  consult: Consult;
  open: boolean;
  onOpenChange: (open: boolean) => void;
}) {
  const reply = consult.reply;
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-h-[88vh] max-w-2xl overflow-y-auto">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <FileText className="size-4 text-primary" /> 병리 판독 리포트 · {consult.patientName}
          </DialogTitle>
        </DialogHeader>

        {!reply ? (
          <p className="py-6 text-center text-sm text-muted-foreground">아직 병리과 회신이 도착하지 않았습니다.</p>
        ) : (
          <>
            {/* 1. 환자/검체 정보 */}
            <Section index={1} title="환자 정보">
              <div className="flex flex-wrap gap-x-5 gap-y-1.5 text-sm text-muted-foreground">
                <span>환자 <b className="text-foreground">{consult.patientName}</b></span>
                <span>등록번호 <b className="text-foreground">{consult.patientMrn}</b></span>
                <span>검체 <b className="text-foreground">Kidney, needle biopsy</b></span>
                <span>협진 요청 <b className="text-foreground">{consult.requestedBy}</b></span>
              </div>
            </Section>

            {/* 2. 소견 (Microscopic Description) */}
            <Section index={2} title="소견" subtitle="Microscopic Description">
              <pre className="whitespace-pre-wrap rounded-lg border border-border bg-muted/30 p-3.5 font-mono text-[13px] leading-6 text-foreground">
                {reply.findings || "—"}
              </pre>
            </Section>

            {/* 3. 최종 진단 & Banff */}
            <Section index={3} title="최종 진단" subtitle="Diagnosis & Banff">
              <pre className="whitespace-pre-wrap rounded-lg border border-primary/40 bg-primary/5 p-3.5 font-mono text-[13px] font-medium leading-6 text-foreground">
                {reply.diagnosis || "—"}
              </pre>
            </Section>

            {/* 4. 권고사항 */}
            {reply.recommendation && (
              <Section index={4} title="권고사항" subtitle="신장내과 전달">
                <p className="whitespace-pre-wrap rounded-lg border border-border p-3.5 text-sm leading-6 text-foreground">
                  {reply.recommendation}
                </p>
              </Section>
            )}

            {/* 5. AI 분석 근거 — stain별 대표 병변 ROI · Attention Heatmap · 신뢰도 */}
            <Section index={5} title="AI 분석 근거" subtitle="대표 병변 · Heatmap · 신뢰도">
              {reply.analysis && reply.analysis.length > 0 ? (
                <>
                  <div className="grid grid-cols-3 gap-3">
                    {reply.analysis.map((a) => {
                      const stain = a.stain as WsiStain;
                      const overlays = a.overlays ?? [];
                      const roi = roiCropUrl(stain, a.slideId, overlays);
                      // 신뢰 확보/부족 판정은 선택적 예측 게이트가 있는 CDSS(신뢰도 제공)에만 의미가 있다.
                      // ABMIL 회귀(HE/MT)는 abstention이 없어 status가 항상 ALLOW이므로 "신뢰 확보"로 표기하지
                      // 않고 중립 라벨("회귀 예측")로 표시한다 — 신뢰도 미제공과의 모순을 없앤다.
                      const hasConfidence = !!a.confidences && a.confidences.length > 0;
                      return (
                        <div key={a.stain} className="flex flex-col gap-1.5">
                          <AttentionHeatmap label={a.stain} slideId={a.slideId} overlays={overlays} />
                          <RoiThumbnail label={a.stain} url={roi} analyzed />
                          <div className="rounded-lg border border-border px-2 py-1.5 text-[11px]">
                            {hasConfidence ? (
                              <>
                                <span className={a.status === "ALLOW" ? "font-semibold text-success" : "font-semibold text-warning"}>
                                  {a.status === "ALLOW" ? "신뢰 확보" : a.status === "ABSTAIN" ? "신뢰 부족" : "—"}
                                </span>
                                <dl className="mt-1 flex flex-col gap-0.5">
                                  {a.confidences!.map((c) => (
                                    <div key={c.label} className="flex justify-between gap-1">
                                      <dt className="truncate text-muted-foreground">{c.label}</dt>
                                      <dd className="shrink-0 font-semibold tabular-nums text-foreground">{Math.round(c.value * 100)}%</dd>
                                    </div>
                                  ))}
                                  {a.uncertainty != null && <p className="text-muted-foreground/70">불확실성 {a.uncertainty.toFixed(2)}</p>}
                                </dl>
                              </>
                            ) : (
                              <>
                                <span className="font-semibold text-muted-foreground">회귀 예측</span>
                                <p className="mt-1 text-muted-foreground/70">신뢰도 게이트 없음(ABMIL 회귀)</p>
                              </>
                            )}
                          </div>
                        </div>
                      );
                    })}
                  </div>
                  <p className="mt-1.5 text-[11px] text-muted-foreground/70">PAS(CDSS)만 descriptor별 신뢰도 제공 · HE/MT(ABMIL)는 회귀 예측.</p>
                </>
              ) : (
                <p className="rounded-lg border border-dashed border-border bg-muted/20 p-3 text-sm text-muted-foreground">
                  이 리포트에는 AI 분석 근거가 없습니다. 병리과가 <b>Viewer에서 AI 분석을 실행한 뒤 판독 완료</b>한 리포트에만
                  대표 병변 ROI·Heatmap·신뢰도가 함께 첨부됩니다.
                </p>
              )}
            </Section>

            {/* 6. 판독의 서명 */}
            <Section index={6} title="판독의">
              <div className="flex items-center justify-between gap-3 rounded-lg border border-border px-3.5 py-2.5 text-sm">
                <div>
                  <p className="text-muted-foreground">판독의 <span className="font-semibold text-foreground">{reply.author}</span> · 병리과</p>
                  <p className="mt-0.5 text-xs text-muted-foreground/70">전자 서명 완료 · 회신 {formatDateTime(reply.repliedAt)}</p>
                </div>
                {reply.signaturePath ? (
                  <img src={reply.signaturePath} alt="판독의 서명" className="h-14 max-w-[180px] rounded border border-border bg-white object-contain px-2" />
                ) : (
                  <span className="rounded bg-success/10 px-2.5 py-1 text-xs font-semibold text-success">판독 완료</span>
                )}
              </div>
            </Section>
          </>
        )}
      </DialogContent>
    </Dialog>
  );
}
