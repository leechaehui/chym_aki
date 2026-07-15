import { type ReactNode, useState } from "react";
import { Network, Activity, TrendingUp } from "lucide-react";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription, DialogFooter } from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";

const MOCK_FEATURES = [
  { label: "Creatinine 48h 변화율", value: "+0.84 mg/dL", risk: "high" },
  { label: "소변량 (최근 6h)", value: "0.28 mL/kg/h", risk: "high" },
  { label: "eGFR", value: "18 mL/min", risk: "high" },
  { label: "BUN/Cr 비율", value: "22.1", risk: "medium" },
  { label: "칼륨", value: "5.4 mEq/L", risk: "medium" },
  { label: "혈압 평균", value: "88 mmHg", risk: "medium" },
];

export function NephrologyTriageModal({
  patientId,
  desc,
  trigger,
}: {
  patientId: string;
  desc: string;
  trigger: ReactNode;
}) {
  const [open, setOpen] = useState(false);

  return (
    <>
      <div onClick={() => setOpen(true)}>{trigger}</div>
      <Dialog open={open} onOpenChange={setOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              <Network className="size-4 text-primary" />
              XGBoost 트리아지 — {patientId}
            </DialogTitle>
            <DialogDescription>{desc}</DialogDescription>
          </DialogHeader>

          <div className="flex flex-col gap-3">
            <div className="flex items-center gap-3 rounded-lg border border-destructive/30 bg-destructive/5 px-4 py-3">
              <TrendingUp className="size-5 text-destructive" />
              <div>
                <p className="text-sm font-bold text-destructive">고위험 · Stage 3 예측</p>
                <p className="text-[11px] text-muted-foreground">AKI 중증 확률 87% · 신속 개입 권장</p>
              </div>
            </div>

            <div>
              <p className="mb-1.5 text-xs font-semibold text-muted-foreground">주요 기여 피처</p>
              <ul className="flex flex-col gap-1">
                {MOCK_FEATURES.map((f) => (
                  <li key={f.label} className="flex items-center justify-between text-xs">
                    <span className="flex items-center gap-1.5">
                      <Activity className={`size-3 ${f.risk === "high" ? "text-destructive" : "text-warning"}`} />
                      {f.label}
                    </span>
                    <span className="font-semibold tabular-nums text-foreground">{f.value}</span>
                  </li>
                ))}
              </ul>
            </div>
          </div>

          <DialogFooter>
            <Button size="sm" onClick={() => setOpen(false)}>확인</Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  );
}
