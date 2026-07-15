import { useState } from "react";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger, DialogFooter } from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Network, FileCheck, Loader2, XCircle } from "lucide-react";
import { ConsultTriageReportView, type ConsultTriageData } from "./ConsultTriageReportView";
import { useQuery } from "@tanstack/react-query";
import { icuMonitorService } from "@/services/icuMonitorService";
import { useNotificationStore } from "@/store/notificationStore";

export function ConsultTriageModal({
  trigger,
  patientId,
  desc,
  onAccept,
}: {
  trigger: React.ReactNode;
  patientId: string;
  desc: string;
  onAccept?: () => void;
}) {
  const [open, setOpen] = useState(false);
  const [isRejecting, setIsRejecting] = useState(false);
  const [rejectReason, setRejectReason] = useState("");
  const notify = useNotificationStore((state) => state.notify);

  // 실데이터 연동 (React Query)
  const { data, isLoading, error } = useQuery({
    queryKey: ["consultTriage", patientId],
    queryFn: () => icuMonitorService.consultTriage(Number(patientId)),
    enabled: open && !isNaN(Number(patientId)), // 모달이 열릴 때 + ID가 숫자(stayId)일 때만 Fetch
  });

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger asChild>{trigger}</DialogTrigger>
      <DialogContent className="max-w-4xl max-h-[90vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2 text-primary">
            <Network className="size-5" />
            CDSS 응급실 트리아지 (KTS 기반)
            <span className="ml-2 text-sm font-normal text-muted-foreground border-l pl-2">
              대상: {patientId} ({desc})
            </span>
          </DialogTitle>
        </DialogHeader>

        <div className="py-2 min-h-[200px] flex flex-col justify-center">
          {isLoading && (
            <div className="flex flex-col items-center justify-center text-muted-foreground gap-2">
              <Loader2 className="size-6 animate-spin" />
              <span>실시간 KTS 점수 계산 중...</span>
            </div>
          )}
          {error && (
            <div className="text-center text-red-500 font-medium">
              환자의 트리아지 정보를 불러올 수 없습니다. (데이터 부족)
            </div>
          )}
          {data && <ConsultTriageReportView data={data} />}
        </div>

        <DialogFooter className="mt-4 border-t pt-4">
          {isRejecting ? (
            <div className="flex flex-col w-full gap-3">
              <Textarea 
                placeholder="반려 사유를 상세히 입력해 주세요 (예: 이미 타과 협진 중, 불필요한 알람 등)"
                value={rejectReason}
                onChange={(e) => setRejectReason(e.target.value)}
                className="min-h-[80px]"
              />
              <div className="flex gap-2 justify-end">
                <Button variant="ghost" onClick={() => setIsRejecting(false)}>취소</Button>
                <Button variant="destructive" onClick={async () => {
                  try {
                    await icuMonitorService.rejectConsult(Number(patientId), rejectReason);
                    notify({
                      title: "협진 반려 완료",
                      message: `[${patientId}] 환자의 신장내과 협진이 반려되었습니다. (DB 저장 완료, 사유: ${rejectReason || '없음'})`,
                      severity: "WARNING",
                      department: "nephrology"
                    });
                    setOpen(false);
                    setIsRejecting(false);
                    setRejectReason("");
                  } catch (e) {
                    notify({ title: "오류 발생", message: "반려 처리에 실패했습니다.", severity: "CRITICAL", department: "nephrology" });
                  }
                }}>
                  반려 확정
                </Button>
              </div>
            </div>
          ) : (
            <div className="flex gap-3 w-full">
              <Button variant="outline" onClick={() => setIsRejecting(true)} className="flex-1 text-red-600 hover:text-red-700 hover:bg-red-50">
                <XCircle className="size-4 mr-2" />
                협진 반려 (사유 입력)
              </Button>
              <Button onClick={async () => {
                try {
                  await icuMonitorService.acceptConsult(Number(patientId));
                  notify({
                    title: "협진 수락 완료",
                    message: `[${patientId}] 환자의 신장내과 협진을 수락했습니다. (DB 저장 완료)`,
                    severity: "INFO",
                    department: "nephrology"
                  });
                  setOpen(false);
                  if (onAccept) onAccept();
                } catch (e) {
                  notify({ title: "오류 발생", message: "수락 처리에 실패했습니다.", severity: "CRITICAL", department: "nephrology" });
                }
              }} className="flex-1 gap-2 bg-primary text-white">
                <FileCheck className="size-4" />
                협진 수락 및 차트 보기
              </Button>
            </div>
          )}
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
