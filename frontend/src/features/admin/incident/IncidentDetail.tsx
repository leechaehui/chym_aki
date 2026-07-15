import { useEffect, useState } from "react";
import { ArrowLeft, CheckCircle2, Lock, FileText, AlertCircle, Printer, UserPlus, Sparkles } from "lucide-react";
import { type Incident, incidentService } from "@/services/incidentService";
import { formatDateTime } from "@/lib/format";
import { useAuthStore } from "@/store/authStore";
import { authService } from "@/services/authService";
import { type Account } from "@/types";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Select } from "@/components/ui/input";
import { useNotificationStore } from "@/store/notificationStore";
import { SignaturePad } from "@/components/ui/SignaturePad";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogFooter } from "@/components/ui/dialog";

const BACKEND_URL = (import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000/api").replace("/api", "");

const STATUS_CONFIG = {
  OPEN: { label: "신규", icon: AlertCircle, color: "text-red-500 bg-red-50 border-red-200" },
  INVESTIGATING: { label: "원인 파악 중", icon: AlertCircle, color: "text-amber-500 bg-amber-50 border-amber-200" },
  RESOLVED: { label: "조치 중 (서명 대기)", icon: CheckCircle2, color: "text-amber-600 bg-amber-50 border-amber-200" },
  SIGNED: { label: "서명 완료", icon: FileText, color: "text-purple-500 bg-purple-50 border-purple-200" },
  LOCKED: { label: "승인 완료", icon: Lock, color: "text-slate-500 bg-slate-50 border-slate-200" },
};

export function IncidentDetail({ incidentId, onBack }: { incidentId: string; onBack: () => void }) {
  const user = useAuthStore((s) => s.user);
  const [inc, setInc] = useState<Incident | null>(null);
  const [loading, setLoading] = useState(true);
  const [analyzing, setAnalyzing] = useState(false);
  
  // 상태 변경 관련
  const [actionRootCause, setActionRootCause] = useState("");
  const [actionTaken, setActionTaken] = useState("");
  const [assigneeId, setAssigneeId] = useState("");
  const [accounts, setAccounts] = useState<Account[]>([]);
  const notify = useNotificationStore((s) => s.notify);

  // 전자서명 모달 상태
  const [showSignModal, setShowSignModal] = useState(false);
  const [signatureBase64, setSignatureBase64] = useState<string>("");
  const [savingSignature, setSavingSignature] = useState(false);

  const loadData = async () => {
    setLoading(true);
    try {
      const [data, accs] = await Promise.all([
        incidentService.get(incidentId),
        authService.listAccounts()
      ]);
      setInc(data);
      setAccounts(accs);
      if (data.rootCause) setActionRootCause(data.rootCause);
      if (data.actionTaken) setActionTaken(data.actionTaken);
      if (data.assignedToUserId) setAssigneeId(data.assignedToUserId);
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadData();
  }, [incidentId]);

  // INVESTIGATING 상태인데 원인 분석 내역이 없다면 자동으로 분석(또는 재분석) 수행
  useEffect(() => {
    if (inc?.status === "INVESTIGATING" && !inc?.rootCause && !analyzing) {
      handleAnalyze();
    }
  }, [inc?.status, inc?.rootCause]);

  if (loading || !inc) return <div className="p-8 text-center">불러오는 중...</div>;

  const SConf = STATUS_CONFIG[inc.status];

  const handleAssign = async () => {
    if (assigneeId === "UNASSIGNED") {
      try {
        await incidentService.assign(inc.id, null as any, null as any);
        notify({ severity: "INFO", department: "admin", title: "담당자 지정 해제", message: "담당자가 미배정 상태로 변경되었습니다." });
        loadData();
      } catch (e: any) {
        notify({ severity: "WARNING", department: "admin", title: "오류", message: e.message || "실패했습니다." });
      }
      return;
    }
    
    const acc = accounts.find(a => a.id === assigneeId);
    if (!acc) return;
    try {
      await incidentService.assign(inc.id, acc.id, acc.name);
      notify({ severity: "INFO", department: "admin", title: "담당자 지정", message: "담당자가 지정되었습니다." });
      loadData();
    } catch (e: any) {
      notify({ severity: "WARNING", department: "admin", title: "오류", message: e.message || "실패했습니다." });
    }
  };

  const handleStatusChange = async (newStatus: string) => {
    try {
      await incidentService.updateStatus(inc.id, newStatus);
      notify({ severity: "INFO", department: "admin", title: "상태 변경", message: `상태가 ${newStatus}(으)로 변경되었습니다.` });
      await loadData();
      
      if (newStatus === "INVESTIGATING") {
        handleAnalyze();
      }
    } catch (e: any) {
      notify({ severity: "WARNING", department: "admin", title: "오류", message: e.message || "실패했습니다." });
    }
  };

  const handleAnalyze = async () => {
    setAnalyzing(true);
    try {
      const result = await incidentService.analyze(inc.id);
      setActionRootCause(result.rootCause);
      setActionTaken(result.actionTaken);
      notify({ severity: "INFO", department: "admin", title: "AI 분석 완료", message: "장애 원인과 조치 내역이 자동 생성되었습니다." });
    } catch (e: any) {
      notify({ severity: "WARNING", department: "admin", title: "분석 실패", message: e.message || "AI 분석에 실패했습니다." });
    } finally {
      setAnalyzing(false);
    }
  };

  const handleResolve = async () => {
    if (!actionRootCause || !actionTaken) {
      alert("원인과 조치내역을 모두 입력하세요.");
      return;
    }
    try {
      await incidentService.resolve(inc.id, actionRootCause, actionTaken);
      notify({ severity: "INFO", department: "admin", title: "조치 완료", message: "조치 내역이 등록되었습니다." });
      loadData();
    } catch (e: any) {
      notify({ severity: "WARNING", department: "admin", title: "오류", message: e.message || "실패했습니다." });
    }
  };

  const handleSign = async () => {
    try {
      await incidentService.sign(inc.id);
      notify({ severity: "INFO", department: "admin", title: "서명 완료", message: "전자서명이 완료되었습니다." });
      loadData();
    } catch (e: any) {
      if (e.message?.includes("서명이 없습니다")) {
        setShowSignModal(true);
      } else {
        notify({ severity: "WARNING", department: "admin", title: "오류", message: e.message || "서명에 실패했습니다." });
      }
    }
  };

  const handleSaveSignatureAndSign = async () => {
    if (!signatureBase64) {
      alert("서명을 입력해주세요.");
      return;
    }
    setSavingSignature(true);
    try {
      // 1. 서버에 내 서명 등록
      const updatedUser = await authService.updateSignature(signatureBase64);
      // zustand 등 전역 사용자 상태 동기화가 필요하지만, 
      // 현재 플로우에서는 백엔드에 서명이 저장되었으므로 incidentService.sign()을 재호출 가능.
      
      // 2. Incident 전자서명 처리
      await incidentService.sign(inc.id);
      notify({ severity: "INFO", department: "admin", title: "서명 완료", message: "새로운 서명이 프로필에 등록되고 해당 장애 조치에 서명되었습니다." });
      
      setShowSignModal(false);
      loadData();
    } catch (e: any) {
      notify({ severity: "WARNING", department: "admin", title: "오류", message: e.message || "서명 등록 및 적용에 실패했습니다." });
    } finally {
      setSavingSignature(false);
    }
  };

  const handleLock = async () => {
    if (!confirm("승인 및 잠금 처리 후에는 수정할 수 없습니다. 계속하시겠습니까?")) return;
    try {
      await incidentService.lock(inc.id);
      notify({ severity: "INFO", department: "admin", title: "최종 승인", message: "최종 승인 및 잠금 처리되었습니다." });
      loadData();
    } catch (e: any) {
      notify({ severity: "WARNING", department: "admin", title: "오류", message: e.message || "실패했습니다." });
    }
  };

  const handlePrint = () => {
    // 보고서 페이지(별도 탭)에서 PDF 다운로드/인쇄를 제공한다.
    window.open(`#/admin/incident/print/${inc.id}`, '_blank');
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <Button variant="ghost" size="icon" onClick={onBack}><ArrowLeft className="size-4" /></Button>
          <div>
            <h2 className="text-xl font-bold">{inc.incidentNo}</h2>
            <p className="text-sm text-slate-500">운영 장애 상세 정보</p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          {inc.status === "LOCKED" && (
            <Button variant="outline" className="gap-2" onClick={handlePrint}>
              <Printer className="size-4" />
              보고서 PDF 다운로드
            </Button>
          )}
          <span className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full text-sm font-medium border ${SConf.color}`}>
            <SConf.icon className="size-4" />
            {SConf.label}
          </span>
        </div>
      </div>

      <div className="grid grid-cols-3 gap-6">
        <div className="col-span-2 space-y-6">
          <Card>
            <CardHeader>
              <CardTitle className="text-lg">장애 개요</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="text-xs font-semibold text-slate-500 uppercase tracking-wider">최초 발생</label>
                  <p className="font-medium">{formatDateTime(inc.firstOccurredAt)}</p>
                </div>
                <div>
                  <label className="text-xs font-semibold text-slate-500 uppercase tracking-wider">최근 발생</label>
                  <p className="font-medium">{formatDateTime(inc.lastOccurredAt)}</p>
                </div>
                <div>
                  <label className="text-xs font-semibold text-slate-500 uppercase tracking-wider">모듈</label>
                  <p className="font-medium">{inc.moduleName}</p>
                </div>
                <div>
                  <label className="text-xs font-semibold text-slate-500 uppercase tracking-wider">엔드포인트</label>
                  <p className="font-medium">{inc.endpoint || '-'}</p>
                </div>
                <div className="col-span-2">
                  <label className="text-xs font-semibold text-slate-500 uppercase tracking-wider">발생 횟수</label>
                  <p className="font-medium">{inc.occurrenceCount} 회</p>
                </div>
              </div>
              
              <div>
                <label className="text-xs font-semibold text-slate-500 uppercase tracking-wider">오류 메시지</label>
                <div className="mt-1 p-3 bg-red-50 text-red-900 rounded-md border border-red-100 font-mono text-sm overflow-x-auto whitespace-pre-wrap">
                  {inc.errorMessage}
                </div>
              </div>

              <div>
                <label className="text-xs font-semibold text-slate-500 uppercase tracking-wider">Stack Trace</label>
                <div className="mt-1 p-3 bg-slate-900 text-slate-300 rounded-md font-mono text-xs overflow-x-auto max-h-64 whitespace-pre-wrap">
                  {inc.stackTrace || 'None'}
                </div>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="text-lg">조치 내역</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              {inc.status === "LOCKED" || inc.status === "SIGNED" || inc.status === "RESOLVED" ? (
                <>
                  <div>
                    <label className="text-xs font-semibold text-slate-500 uppercase tracking-wider">원인 분석 (Root Cause)</label>
                    <div className="mt-1 p-3 bg-slate-50 rounded-md border border-slate-200 text-sm whitespace-pre-wrap min-h-20">
                      {inc.rootCause}
                    </div>
                  </div>
                  <div>
                    <label className="text-xs font-semibold text-slate-500 uppercase tracking-wider">조치 사항 (Action Taken)</label>
                    <div className="mt-1 p-3 bg-slate-50 rounded-md border border-slate-200 text-sm whitespace-pre-wrap min-h-20">
                      {inc.actionTaken}
                    </div>
                  </div>
                  <div className="flex justify-between items-center text-sm text-slate-500 bg-slate-50 p-3 rounded-md mt-4">
                    <span>조치자: <strong>{inc.resolvedByName}</strong></span>
                    <span>조치일시: {inc.resolvedAt ? formatDateTime(inc.resolvedAt) : '-'}</span>
                  </div>
                </>
              ) : (
                <>
                  <div className="flex items-center justify-between mb-2">
                    <label className="text-sm font-semibold block">원인 분석</label>
                  </div>
                  <div>
                    <textarea 
                      className="w-full border border-slate-300 rounded-md p-2 text-sm focus:ring-2 focus:ring-blue-500 outline-none"
                      rows={4}
                      value={actionRootCause}
                      onChange={(e) => setActionRootCause(e.target.value)}
                      placeholder="발생 원인을 상세히 기록하세요..."
                    />
                  </div>
                  <div className="mt-4">
                    <label className="text-sm font-semibold mb-1 block">조치 사항</label>
                    <textarea 
                      className="w-full border border-slate-300 rounded-md p-2 text-sm focus:ring-2 focus:ring-blue-500 outline-none"
                      rows={4}
                      value={actionTaken}
                      onChange={(e) => setActionTaken(e.target.value)}
                      placeholder="재발 방지 대책 및 조치 내역을 기록하세요..."
                    />
                  </div>
                  <div className="flex justify-end pt-2">
                    <Button onClick={handleResolve} className="gap-2">
                      <CheckCircle2 className="size-4" />
                      조치 내용 저장
                    </Button>
                  </div>
                </>
              )}
            </CardContent>
          </Card>
        </div>

        <div className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle className="text-md">담당자 지정</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {inc.assignedToName ? (
                  <div className="flex items-center gap-3 p-3 bg-blue-50 text-blue-900 border border-blue-100 rounded-md mb-2">
                    <div className="size-8 bg-blue-200 rounded-full flex items-center justify-center font-bold text-blue-700">
                      {inc.assignedToName.charAt(0)}
                    </div>
                    <div>
                      <p className="text-sm font-bold">{inc.assignedToName}</p>
                      <p className="text-xs opacity-80">담당 조사관</p>
                    </div>
                  </div>
                ) : (
                  (inc.status === "LOCKED" || inc.status === "SIGNED") && (
                    <div className="text-sm text-slate-500 italic p-4 text-center border border-dashed border-slate-200 rounded-md">
                      담당자가 지정되지 않았습니다.
                    </div>
                  )
                )}
                
                {inc.status !== "LOCKED" && inc.status !== "SIGNED" && (
                  <div className="flex gap-2">
                    <Select value={assigneeId} onChange={(e) => setAssigneeId(e.target.value)} className="flex-1 h-9 text-sm">
                      <option value="">담당자 선택...</option>
                      <option value="UNASSIGNED">미배정</option>
                      {accounts.filter(a => a.approval === "approved" && a.role === "admin").map(a => (
                        <option key={a.id} value={a.id}>{a.name} ({a.department})</option>
                      ))}
                    </Select>
                    <Button variant="outline" size="sm" onClick={handleAssign} disabled={!assigneeId}>
                      지정
                    </Button>
                  </div>
                )}
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="text-md">승인 및 워크플로우</CardTitle>
              <CardDescription>현재 상태에 따른 다음 단계 조치</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              {inc.status === "OPEN" && (
                <Button className="w-full" onClick={() => handleStatusChange("INVESTIGATING")}>
                  조사 시작 (INVESTIGATING)
                </Button>
              )}
              
              {inc.status === "INVESTIGATING" && (
                <div className="space-y-3">
                  <div className="text-sm text-amber-600 bg-amber-50 p-3 rounded-md border border-amber-100">
                    현재 원인 파악 중입니다. AI가 분석한 조치 내역을 검토하고 완료해 주세요.
                  </div>
                  {(!inc.rootCause || inc.rootCause === "") && (
                    <Button variant="outline" className="w-full" onClick={handleAnalyze} disabled={analyzing}>
                      {analyzing ? "AI 분석 중..." : "AI 원인 분석 다시 생성"}
                    </Button>
                  )}
                </div>
              )}

              {inc.status === "RESOLVED" && (
                <div className="space-y-3">
                  <div className="text-sm text-blue-600 bg-blue-50 p-3 rounded-md border border-blue-100">
                    조치 내용이 임시 저장되었습니다. 본인 인증(전자서명)을 진행하여 조치를 완료하세요.
                  </div>
                  <Button className="w-full" onClick={handleSign}>
                    내 서명 적용하여 조치 완료하기 (SIGNED)
                  </Button>
                  <p className="text-xs text-slate-500 text-center mt-2">
                    프로필에 등록된 서명이 사용됩니다.
                  </p>
                </div>
              )}

              {inc.status === "SIGNED" && (
                <div className="space-y-3">
                  <div className="flex flex-col items-center justify-center p-4 border border-slate-200 rounded-md bg-slate-50 mb-4">
                    <span className="text-xs text-slate-500 mb-2">서명자: {inc.resolvedByName}</span>
                    {inc.signaturePath ? (
                      <img src={`${BACKEND_URL}${inc.signaturePath}`} alt="Signature" className="h-16 mix-blend-multiply" />
                    ) : (
                      <span className="text-sm italic">서명 누락</span>
                    )}
                    <span className="text-xs text-slate-400 mt-2">{inc.signedAt && formatDateTime(inc.signedAt)}</span>
                  </div>
                  <Button className="w-full" variant="destructive" onClick={handleLock}>
                    최종 승인 및 잠금 (LOCKED)
                  </Button>
                </div>
              )}
              
              {inc.status === "LOCKED" && (
                <div className="flex flex-col items-center justify-center p-6 border border-slate-200 rounded-md bg-slate-50">
                  <Lock className="size-8 text-slate-400 mb-2" />
                  <h3 className="font-bold text-slate-700">이 문서는 잠금 처리되었습니다.</h3>
                  <p className="text-xs text-slate-500 text-center mt-2">
                    감사(Audit) 및 규정 준수를 위해 더 이상 내용을 수정할 수 없습니다.<br/>
                    {inc.lockedAt && formatDateTime(inc.lockedAt)}
                  </p>
                  
                  {inc.signaturePath && (
                    <div className="mt-4 pt-4 border-t border-slate-200 w-full flex flex-col items-center">
                      <span className="text-xs text-slate-500 mb-2">전자 서명</span>
                      <img src={`${BACKEND_URL}${inc.signaturePath}`} alt="Signature" className="h-12 mix-blend-multiply" />
                    </div>
                  )}
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      </div>

      <Dialog open={showSignModal} onOpenChange={setShowSignModal}>
        <DialogContent className="max-w-md">
          <DialogHeader>
            <DialogTitle>전자서명 등록 및 서명</DialogTitle>
          </DialogHeader>
          <div className="py-4 space-y-4">
            <p className="text-sm text-slate-500">
              프로필에 등록된 전자서명이 없습니다. 즉석에서 서명을 입력하면 프로필에 안전하게 저장되며 현재 문서에 서명됩니다.
            </p>
            <div className="border border-slate-200 rounded-md bg-white">
              <SignaturePad onSign={(val) => setSignatureBase64(val || "")} />
            </div>
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setShowSignModal(false)} disabled={savingSignature}>취소</Button>
            <Button onClick={handleSaveSignatureAndSign} disabled={savingSignature || !signatureBase64}>
              {savingSignature ? "처리 중..." : "등록 및 서명하기"}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
