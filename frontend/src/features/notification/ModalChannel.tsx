import { useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { AlertOctagon } from "lucide-react";
import { useNotificationStore } from "@/store/notificationStore";
import { useAuthStore } from "@/store/authStore";
import { fireAlertAudit } from "@/lib/alertAudit";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription, DialogFooter } from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";

/**
 * CRITICAL 채널 — 즉시 의사결정이 필요한 위급 모달.
 * 닫기 불가(배경 클릭/ESC/X 차단), 확인/에스컬레이션/바로가기만 가능. 부서 큐의 맨 앞 1건을 띄운다.
 * CDSS alert(alertId 보유)는 표시 시 VIEWED, 확인 시 ACKNOWLEDGED, 에스컬레이션 시 ESCALATED 를 기록한다.
 */
export function ModalChannel() {
  const dept = useAuthStore((s) => s.user?.role);
  const current = useNotificationStore((s) => s.modalQueue.find((n) => n.department === dept) ?? null);
  const dismiss = useNotificationStore((s) => s.dismissModal);
  const markRead = useNotificationStore((s) => s.markRead);
  const navigate = useNavigate();

  // 모달이 표시되면 VIEWED 기록(1회성 — fireAlertAudit 내부 fired flag로 중복 방지).
  useEffect(() => {
    if (current?.alertId) fireAlertAudit(current.alertId, "VIEWED", dept);
  }, [current?.alertId, dept]);

  if (!current) return null;

  function resolve(action: "ACKNOWLEDGED" | "ESCALATED") {
    if (!current) return;
    fireAlertAudit(current.alertId, action, dept);
    markRead(current.id);
    dismiss();
  }

  return (
    <Dialog open onOpenChange={() => {}}>
      <DialogContent hideClose onInteractOutside={(e) => e.preventDefault()} onEscapeKeyDown={(e) => e.preventDefault()} className="max-w-md">
        <DialogHeader>
          <div className="mb-1 flex size-11 items-center justify-center rounded-full bg-destructive/10 text-destructive">
            <AlertOctagon className="size-6" />
          </div>
          <DialogTitle>{current.title}</DialogTitle>
          <DialogDescription>{current.message}</DialogDescription>
        </DialogHeader>
        <DialogFooter>
          {current.link && (
            <Button
              variant="destructive"
              size="sm"
              onClick={() => {
                navigate(current.link!);
                resolve("ACKNOWLEDGED");
              }}
            >
              바로가기
            </Button>
          )}
          {current.alertId && (
            <Button variant="outline" size="sm" onClick={() => resolve("ESCALATED")}>
              에스컬레이션
            </Button>
          )}
          <Button variant={current.link ? "outline" : "destructive"} size="sm" onClick={() => resolve("ACKNOWLEDGED")}>
            확인
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
