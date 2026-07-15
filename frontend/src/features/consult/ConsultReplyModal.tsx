import { useEffect, useState } from "react";
import { Reply } from "lucide-react";
import type { Consult } from "@/types";
import { URGENCY_LABEL } from "@/types";
import { useConsultStore } from "@/store/consultStore";
import { useNotificationStore } from "@/store/notificationStore";
import { useAuthStore } from "@/store/authStore";
import { eventBus } from "@/features/notification/events";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription, DialogFooter } from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Textarea, Label } from "@/components/ui/input";
import { InfoCard } from "@/components/common/InfoCard";

/**
 * 신장 협진(응급/ICU → 신장내과) 회신 모달.
 * 신장내과가 응급 협진 건에 소견/판단/권고를 작성해 회신한다 — 전송 시 협진 스토어에
 * 회신을 등록(상태 replied 전이)하고, 도메인 이벤트를 발행해 응급의학과에 회신 도착 알림을 보낸다.
 * 회신 본문(ConsultReply) 구조는 병리 협진과 동일 스키마를 재사용한다.
 */
export function ConsultReplyModal({
  consult,
  open,
  onOpenChange,
}: {
  consult: Consult | null;
  open: boolean;
  onOpenChange: (open: boolean) => void;
}) {
  const reply = useConsultStore((s) => s.reply);
  const notify = useNotificationStore((s) => s.notify);
  const user = useAuthStore((s) => s.user);
  const [findings, setFindings] = useState("");
  const [diagnosis, setDiagnosis] = useState("");
  const [recommendation, setRecommendation] = useState("");
  const [submitting, setSubmitting] = useState(false);

  // 대상 협진이 바뀌면 입력 초기화(기존 회신이 있으면 프리필).
  useEffect(() => {
    setFindings(consult?.reply?.findings ?? "");
    setDiagnosis(consult?.reply?.diagnosis ?? "");
    setRecommendation(consult?.reply?.recommendation ?? "");
  }, [consult]);

  if (!consult) return null;

  const valid = findings.trim().length >= 5 && diagnosis.trim() && recommendation.trim();

  async function submit() {
    if (!consult || !valid) return;
    setSubmitting(true);
    await reply(consult.id, {
      findings: findings.trim(),
      diagnosis: diagnosis.trim(),
      recommendation: recommendation.trim(),
      author: user?.name ?? "신장내과",
      repliedAt: new Date().toISOString(),
    });

    // 회신자(신장내과) 본인 확인용 토스트.
    notify({ severity: "INFO", department: "nephrology", title: "협진 회신 전송", message: `${consult.patientName} 응급 협진 회신을 전송했습니다.` });
    setSubmitting(false);
    onOpenChange(false);
  }

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>응급 협진 회신 · {consult.patientName}</DialogTitle>
          <DialogDescription>응급의학과 요청에 대한 신장내과 평가/권고를 작성해 회신합니다.</DialogDescription>
        </DialogHeader>

        <div className="mb-4 rounded-lg border border-border bg-muted/40 p-3">
          <InfoCard
            items={[
              { label: "병상", value: consult.bedLabel ?? consult.patientMrn },
              { label: "환자명", value: consult.patientName },
              { label: "진단명", value: consult.diagnosis },
              { label: "긴급도", value: URGENCY_LABEL[consult.urgency] },
              { label: "주요 검사", value: consult.keyLabs },
              { label: "요청 사유", value: consult.reason },
            ]}
          />
        </div>

        <div className="flex flex-col gap-3">
          <div>
            <Label>평가 소견</Label>
            <Textarea value={findings} onChange={(e) => setFindings(e.target.value)} placeholder="신기능·전해질·수액상태 등 평가 소견을 기술하세요." className="min-h-24" />
          </div>
          <div>
            <Label>임상 판단</Label>
            <Textarea value={diagnosis} onChange={(e) => setDiagnosis(e.target.value)} placeholder="AKI 단계·원인 추정 등 임상 판단." className="min-h-16" />
          </div>
          <div>
            <Label>처치 권고</Label>
            <Textarea value={recommendation} onChange={(e) => setRecommendation(e.target.value)} placeholder="응급 투석 적응증·수액/약물 조정 등 처치 권고." className="min-h-16" />
          </div>
        </div>

        <DialogFooter>
          <Button variant="outline" size="sm" onClick={() => onOpenChange(false)}>
            취소
          </Button>
          <Button size="sm" onClick={submit} disabled={!valid || submitting}>
            <Reply className="size-3.5" /> 응급의학과로 회신
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
