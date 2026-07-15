import { useEffect, useState } from "react";
import { Send } from "lucide-react";
import type { ConsultUrgency, Patient } from "@/types";
import { URGENCY_LABEL } from "@/types";
import { draftConsultFromLabs } from "./consultDraft";
import { useConsultStore } from "@/store/consultStore";
import { useNotificationStore } from "@/store/notificationStore";
import { useAuthStore } from "@/store/authStore";
import { useChatStore } from "@/store/chatStore";
import { eventBus } from "@/features/notification/events";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription, DialogFooter } from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Textarea, Select, Label } from "@/components/ui/input";
import { InfoCard } from "@/components/common/InfoCard";

/**
 * 병리 협진 요청 모달 — 환자 정보 자동 입력 + 사유/긴급도 입력 후 병리과로 전송.
 * 전송 시 협진 스토어에 등록하고, 도메인 이벤트를 발행한다(Observer) — 병리과 대상 알림은
 * 이벤트 구독자가 자동 생성한다(요청자는 어떤 알림이 만들어지는지 몰라도 됨, 결합도↓).
 */
export function ConsultRequestModal({
  patient,
  open,
  onOpenChange,
}: {
  patient: Patient;
  open: boolean;
  onOpenChange: (open: boolean) => void;
}) {
  const request = useConsultStore((s) => s.request);
  const notify = useNotificationStore((s) => s.notify);
  const user = useAuthStore((s) => s.user);
  const notifyConsultRequest = useChatStore((s) => s.notifyConsultRequest);
  const [reason, setReason] = useState("");
  const [urgency, setUrgency] = useState<ConsultUrgency>("routine");
  const [submitting, setSubmitting] = useState(false);

  // 모달이 열릴 때(또는 환자 변경 시) 검사 결과 기반 초안으로 긴급도·사유를 자동 채운다.
  // 이후 임상의가 자유롭게 수정 가능(초안일 뿐). 전송 후 submit 에서 초기화된다.
  useEffect(() => {
    if (!open) return;
    const draft = draftConsultFromLabs(patient);
    setUrgency(draft.urgency);
    setReason(draft.reason);
  }, [open, patient.mrn]); // eslint-disable-line react-hooks/exhaustive-deps -- 열릴 때 1회 초안화

  const keyLabs = patient.labs
    .filter((l) => ["cr", "egfr", "k", "upcr"].includes(l.key))
    .map((l) => `${l.label} ${l.value}`)
    .join(" · ");

  async function submit() {
    setSubmitting(true);
    const created = await request({
      patientMrn: patient.mrn,
      patientName: patient.name,
      diagnosis: patient.diagnosis,
      keyLabs,
      reason: reason.trim(),
      urgency,
      requestedBy: user?.name ?? "신장내과",
    });
    // 도메인 이벤트 발행 → 병리과 알림은 구독자(Observer)가 자동 생성.
    // 생성된 협진 id 를 함께 실어 알림이 해당 협진으로 딥링크되게 한다.
    if (urgency === "emergency") {
      eventBus.publish({ type: "consult.urgentRead", patientName: patient.name, consultId: created.id });
    } else {
      eventBus.publish({ type: "consult.requested", patientName: patient.name, mrn: patient.mrn, urgency: URGENCY_LABEL[urgency], consultId: created.id });
    }
    // 병리과 의사에게 채팅 알림 자동 전송(패널 미열기 — 백그라운드 발송)
    notifyConsultRequest(patient.name, URGENCY_LABEL[urgency], keyLabs).catch(() => {});
    // 요청자(신장내과) 본인 확인용 토스트
    notify({ severity: "INFO", department: "nephrology", title: "전송되었습니다", message: `${patient.name} 환자 병리 협진 요청이 병리과로 전송되었습니다.` });
    setSubmitting(false);
    setReason("");
    setUrgency("routine");
    onOpenChange(false);
  }

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>병리 협진 요청</DialogTitle>
          <DialogDescription>환자 정보·검사 결과 기반으로 긴급도와 사유 초안이 자동 작성됩니다. 확인 후 수정·전송하세요.</DialogDescription>
        </DialogHeader>

        <div className="mb-4 rounded-lg border border-border bg-muted/40 p-3">
          <InfoCard
            items={[
              { label: "환자번호", value: patient.mrn },
              { label: "환자명", value: patient.name },
              { label: "진단명", value: patient.diagnosis },
              { label: "주요 검사", value: keyLabs },
            ]}
          />
        </div>

        <div className="flex flex-col gap-3">
          <div>
            <Label>긴급도</Label>
            <Select value={urgency} onChange={(e) => setUrgency(e.target.value as ConsultUrgency)}>
              {(Object.keys(URGENCY_LABEL) as ConsultUrgency[]).map((u) => (
                <option key={u} value={u}>
                  {URGENCY_LABEL[u]}
                </option>
              ))}
            </Select>
          </div>
          <div>
            <Label>협진 요청 사유</Label>
            <Textarea
              value={reason}
              onChange={(e) => setReason(e.target.value)}
              placeholder="감별이 필요한 임상 질문, 의심 진단 등을 기술하세요."
              className="min-h-24"
            />
          </div>
        </div>

        <DialogFooter>
          <Button variant="outline" size="sm" onClick={() => onOpenChange(false)}>
            취소
          </Button>
          <Button size="sm" onClick={submit} disabled={reason.trim().length < 5 || submitting}>
            <Send className="size-3.5" /> 병리과로 전송
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
