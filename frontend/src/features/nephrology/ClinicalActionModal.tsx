import { type ReactNode, useState } from "react";
import { AlertTriangle, CheckCircle2, Phone } from "lucide-react";
import type { PatientIdentity } from "@/types";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription, DialogFooter } from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { fakeKoreanName } from "@/lib/fakeName";

const ACTIONS = [
  { id: "notify",   label: "담당의 즉시 호출",      icon: Phone },
  { id: "protocol", label: "AKI 프로토콜 시작",     icon: AlertTriangle },
  { id: "ack",      label: "알람 확인 완료",         icon: CheckCircle2 },
] as const;

type ActionId = typeof ACTIONS[number]["id"];

export function ClinicalActionModal({
  patient,
  onComplete,
  trigger,
}: {
  patient: PatientIdentity;
  onComplete: () => void;
  trigger: ReactNode;
}) {
  const [open, setOpen] = useState(false);
  const [selected, setSelected] = useState<ActionId | null>(null);
  const name = patient.subjectId ? fakeKoreanName(patient.subjectId) : `Stay ${patient.stayId}`;

  function handleConfirm() {
    setOpen(false);
    setSelected(null);
    onComplete();
  }

  return (
    <>
      <div onClick={() => setOpen(true)}>{trigger}</div>
      <Dialog open={open} onOpenChange={setOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              <AlertTriangle className="size-4 text-destructive" />
              긴급 임상 조치
            </DialogTitle>
            <DialogDescription>
              {name} 환자에 대해 즉각 조치를 선택하세요.
            </DialogDescription>
          </DialogHeader>

          <div className="flex flex-col gap-2">
            {ACTIONS.map(({ id, label, icon: Icon }) => (
              <button
                key={id}
                onClick={() => setSelected(id)}
                className={`flex items-center gap-3 rounded-lg border px-4 py-3 text-sm font-medium transition-colors text-left ${
                  selected === id
                    ? "border-primary bg-primary/10 text-primary"
                    : "border-border hover:bg-muted/60 text-foreground"
                }`}
              >
                <Icon className="size-4 shrink-0" />
                {label}
              </button>
            ))}
          </div>

          <DialogFooter>
            <Button variant="outline" size="sm" onClick={() => setOpen(false)}>
              취소
            </Button>
            <Button size="sm" onClick={handleConfirm} disabled={!selected}>
              확인 · 조치 완료
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  );
}
