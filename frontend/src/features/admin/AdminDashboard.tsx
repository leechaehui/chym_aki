import { useEffect, useMemo, useState } from "react";
import { LayoutDashboard, Users, UserCheck, LogIn, ShieldCheck, Check, X, History, Zap } from "lucide-react";
import type { Account, ApprovalStatus, Role } from "@/types";
import { ROLE_LABEL, APPROVAL_LABEL } from "@/types";
import { authService, ApprovalHistoryOut } from "@/services/authService";
import { api } from "@/services/http";
import { useNotificationStore } from "@/store/notificationStore";
import { approvalTone } from "@/lib/statusTone";
import { formatDateTime } from "@/lib/format";
import { PageHeader } from "@/components/common/PageHeader";
import { StatCard } from "@/components/common/StatCard";
import { DataTable, type Column } from "@/components/common/DataTable";
import { StatusBadge } from "@/components/common/StatusBadge";
import { SearchInput } from "@/components/common/SearchInput";
import { ConfirmModal } from "@/components/common/ConfirmModal";
import { LoadingSpinner } from "@/components/common/LoadingSpinner";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Select } from "@/components/ui/input";
import { Modal } from "@/components/common/Modal";
import { Link } from "react-router-dom";

import { IncidentList } from "./incident/IncidentList";

type Filter = "all" | ApprovalStatus;
type Tab = "users" | "incidents";

/** 관리자 콘솔 — KPI · 직원 계정 승인/권한 관리 및 운영 장애 관리. */
export function AdminDashboard() {
  const [activeTab, setActiveTab] = useState<Tab>("users");
  const [accounts, setAccounts] = useState<(Account & { rejectionReason?: string | null })[]>([]);
  const [loading, setLoading] = useState(true);
  const [query, setQuery] = useState("");
  const [filter, setFilter] = useState<Filter>("all");
  const [confirm, setConfirm] = useState<{ id: string; action: ApprovalStatus } | null>(null);
  
  const [rejectModalOpen, setRejectModalOpen] = useState(false);
  const [rejectTargetId, setRejectTargetId] = useState<string | null>(null);
  const [rejectionReason, setRejectionReason] = useState("");
  const [rejectionError, setRejectionError] = useState("");
  
  const [historyModalOpen, setHistoryModalOpen] = useState(false);
  const [historyData, setHistoryData] = useState<ApprovalHistoryOut[]>([]);
  const [historyLoading, setHistoryLoading] = useState(false);
  const [uptimeInfo, setUptimeInfo] = useState<{ value: string; rate: string; tone: "success" | "warning" | "danger" | "neutral" }>({ value: "확인 중", rate: "...", tone: "neutral" });
  
  const notify = useNotificationStore((s) => s.notify);

  useEffect(() => {
    loadAccounts();
    
    api.get("/admin/telemetry/stats")
      .then((stats: any) => {
        const errorRate = stats.error_rate || 0;
        const uptimeRate = (100 - errorRate).toFixed(1);
        let tone: "success" | "warning" | "danger" = "success";
        let value = "정상";
        
        if (errorRate >= 10) { tone = "danger"; value = "위험"; }
        else if (errorRate >= 1) { tone = "warning"; value = "주의"; }
        
        setUptimeInfo({ value, rate: uptimeRate, tone });
      })
      .catch(() => {
        setUptimeInfo({ value: "오류", rate: "조회 불가", tone: "danger" });
      });
  }, []);
  
  function loadAccounts() {
    authService.listAccounts()
      .then((a) => setAccounts(a))
      .catch(() => {})
      .finally(() => setLoading(false));
  }

  const kpi = useMemo(() => {
    const total = accounts.length;
    const pending = accounts.filter((a) => a.approval === "pending").length;
    const todayStr = new Date().toISOString().split("T")[0];
    const today = accounts.filter((a) => a.lastLoginAt?.startsWith(todayStr)).length;
    return { total, pending, today };
  }, [accounts]);

  const filtered = useMemo(
    () =>
      accounts.filter(
        (a) =>
          (filter === "all" || a.approval === filter) &&
          (a.name.includes(query) || a.username.includes(query) || a.department.includes(query)),
      ),
    [accounts, filter, query],
  );

  function setApproval(id: string, approval: ApprovalStatus, reason?: string) {
    const acc = accounts.find((a) => a.id === id);
    setAccounts((prev) => prev.map((a) => (a.id === id ? { ...a, approval, rejectionReason: reason ?? a.rejectionReason } : a)));
    authService
      .setApproval(id, approval, reason)
      .then(() => {
        notify({ severity: "INFO", department: "admin", title: "계정 처리", message: `${acc?.name} 계정을 ${APPROVAL_LABEL[approval]} 처리했습니다.` });
        if (approval === "rejected") {
          setRejectModalOpen(false);
          setRejectionReason("");
        } else {
          setConfirm(null);
        }
      })
      .catch((e: Error) => {
        setAccounts((prev) => prev.map((a) => (a.id === id ? { ...a, approval: acc?.approval ?? a.approval, rejectionReason: acc?.rejectionReason } : a)));
        notify({ severity: "WARNING", department: "admin", title: "처리 실패", message: e.message });
      });
  }

  function handleRejectSubmit() {
    if (!rejectionReason.trim()) {
      setRejectionError("거부 사유를 입력해주세요.");
      return;
    }
    if (rejectionReason.length > 1000) {
      setRejectionError("거부 사유는 1000자를 초과할 수 없습니다.");
      return;
    }
    if (rejectTargetId) {
      setApproval(rejectTargetId, "rejected", rejectionReason.trim());
    }
  }
  
  function openRejectModal(id: string) {
    setRejectTargetId(id);
    setRejectionReason("");
    setRejectionError("");
    setRejectModalOpen(true);
  }

  function openHistoryModal(id: string) {
    setHistoryModalOpen(true);
    setHistoryLoading(true);
    authService.getApprovalHistory(id).then(data => {
      setHistoryData(data);
      setHistoryLoading(false);
    }).catch(err => {
      notify({ severity: "WARNING", department: "admin", title: "조회 실패", message: err.message });
      setHistoryLoading(false);
      setHistoryModalOpen(false);
    });
  }

  function changeRole(id: string, role: Role) {
    setAccounts((prev) => prev.map((a) => (a.id === id ? { ...a, role } : a)));
    notify({ severity: "INFO", department: "admin", title: "권한 변경", message: "사용자 권한을 변경했습니다." });
  }

  const columns: Column<Account & { rejectionReason?: string | null }>[] = [
    { key: "name", header: "이름", render: (a) => <span className="font-medium">{a.name}</span> },
    { key: "username", header: "아이디", render: (a) => <span className="text-muted-foreground">{a.username}</span> },
    {
      key: "role",
      header: "직군",
      render: (a) => (
        <Select value={a.role} onChange={(e) => changeRole(a.id, e.target.value as Role)} className="h-7 w-28 text-xs">
          {(Object.keys(ROLE_LABEL) as Role[]).map((r) => (
            <option key={r} value={r}>
              {ROLE_LABEL[r]}
            </option>
          ))}
        </Select>
      ),
    },
    { key: "department", header: "소속" },
    {
      key: "approval",
      header: "상태",
      render: (a) => (
        <div className="flex flex-col items-start gap-1">
          <StatusBadge label={APPROVAL_LABEL[a.approval]} tone={approvalTone[a.approval]} dot />
          {a.approval === "rejected" && a.rejectionReason && (
            <span className="text-[10px] text-muted-foreground">사유: {a.rejectionReason.length > 15 ? a.rejectionReason.substring(0, 15) + "..." : a.rejectionReason}</span>
          )}
          <button onClick={() => openHistoryModal(a.id)} className="flex items-center gap-1 text-[10px] text-blue-500 hover:underline">
            <History className="size-3" /> 승인/거부 이력 보기
          </button>
        </div>
      ),
    },
    {
      key: "createdAt",
      header: "가입일",
      render: (a) => <span className="text-xs text-muted-foreground">{formatDateTime(a.createdAt)}</span>,
    },
    {
      key: "lastLoginAt",
      header: "최근 로그인",
      render: (a) => <span className="text-xs text-muted-foreground">{a.lastLoginAt ? formatDateTime(a.lastLoginAt) : "기록 없음"}</span>,
    },
    {
      key: "actions",
      header: "처리",
      align: "right",
      render: (a) =>
        a.approval === "pending" ? (
          <div className="flex justify-end gap-1">
            <Button size="sm" variant="success" onClick={() => setConfirm({ id: a.id, action: "approved" })}>
              <Check className="size-3.5" /> 승인
            </Button>
            <Button size="sm" variant="outline" onClick={() => openRejectModal(a.id)}>
              <X className="size-3.5" /> 거부
            </Button>
          </div>
        ) : (
          <span className="text-xs text-muted-foreground">—</span>
        ),
    },
  ];

  if (loading) return <LoadingSpinner label="데이터 불러오는 중" />;

  return (
    <div className="mx-auto max-w-6xl">
      <PageHeader 
        icon={LayoutDashboard} 
        title="관리자 콘솔" 
        subtitle="시스템 통합 관리" 
        actions={
          <Link to="/admin/server-monitoring" className="flex items-center gap-2 px-4 py-2 bg-indigo-600 text-white rounded-md text-sm font-medium hover:bg-indigo-700 transition-colors shadow-sm">
            <Zap className="size-4" /> 서버 모니터링
          </Link>
        }
      />

      <div className="mb-6 border-b border-slate-200">
        <nav className="-mb-px flex space-x-8">
          <button
            onClick={() => setActiveTab("users")}
            className={`whitespace-nowrap pb-4 px-1 border-b-2 font-medium text-sm ${
              activeTab === "users" ? "border-blue-500 text-blue-600" : "border-transparent text-slate-500 hover:text-slate-700 hover:border-slate-300"
            }`}
          >
            직원 관리
          </button>
          <button
            onClick={() => setActiveTab("incidents")}
            className={`whitespace-nowrap pb-4 px-1 border-b-2 font-medium text-sm ${
              activeTab === "incidents" ? "border-blue-500 text-blue-600" : "border-transparent text-slate-500 hover:text-slate-700 hover:border-slate-300"
            }`}
          >
            운영 장애(Incident) 관리
          </button>
        </nav>
      </div>

      {activeTab === "users" && (
        <>
          <div className="mb-5 grid grid-cols-2 gap-3 lg:grid-cols-4">
            <StatCard label="전체 사용자" value={kpi.total} unit="명" icon={Users} tone="accent" />
            <StatCard label="승인 대기" value={kpi.pending} unit="건" icon={UserCheck} tone="warning" hint="처리 필요" />
            <StatCard label="오늘 로그인" value={kpi.today} unit="명" icon={LogIn} tone="primary" />
            <StatCard label="시스템 상태" value={uptimeInfo.value} icon={ShieldCheck} tone={uptimeInfo.tone} hint={`가동률 ${uptimeInfo.rate}${uptimeInfo.rate !== "조회 불가" && uptimeInfo.rate !== "..." ? "%" : ""}`} />
          </div>

          <Card>
            <CardHeader className="flex-row items-center justify-between">
              <CardTitle>직원 관리</CardTitle>
              <div className="flex items-center gap-2">
                <Select value={filter} onChange={(e) => setFilter(e.target.value as Filter)} className="h-8 w-32 text-xs">
                  <option value="all">전체 상태</option>
                  <option value="pending">승인 대기</option>
                  <option value="approved">승인됨</option>
                  <option value="rejected">거부됨</option>
                </Select>
                <SearchInput value={query} onChange={setQuery} placeholder="이름·아이디·소속" className="w-52" />
              </div>
            </CardHeader>
            <CardContent className="p-0">
              <DataTable
                columns={columns}
                data={filtered}
                rowKey={(a) => a.id}
                rowTone={(a) => (a.approval === "pending" ? "warning" : null)}
                emptyTitle="해당 조건의 계정이 없습니다"
              />
            </CardContent>
          </Card>

          <ConfirmModal
            open={confirm !== null}
            onOpenChange={(o) => !o && setConfirm(null)}
            title="계정 승인"
            description="해당 계정의 로그인을 허용합니다. 계속하시겠습니까?"
            confirmLabel="승인"
            tone="success"
            onConfirm={() => confirm && setApproval(confirm.id, confirm.action)}
          />
          
          <Modal open={rejectModalOpen} onOpenChange={setRejectModalOpen} title="가입 거부">
            <div className="flex flex-col gap-4 p-4">
              <div>
                <label className="text-sm font-medium">거부 사유 입력</label>
                <textarea
                  value={rejectionReason}
                  onChange={(e) => {
                    setRejectionReason(e.target.value);
                    setRejectionError("");
                  }}
                  className="mt-1 w-full rounded-md border border-slate-300 p-2 text-sm"
                  rows={4}
                  placeholder="거부 사유를 상세히 적어주세요."
                />
                {rejectionError && <p className="mt-1 text-xs text-red-500">{rejectionError}</p>}
              </div>
              <div className="flex justify-end gap-2">
                <Button variant="outline" onClick={() => setRejectModalOpen(false)}>취소</Button>
                <Button variant="destructive" onClick={handleRejectSubmit}>거부</Button>
              </div>
            </div>
          </Modal>

          <Modal open={historyModalOpen} onOpenChange={setHistoryModalOpen} title="승인/거부 이력">
            <div className="p-4">
              {historyLoading ? (
                <div className="flex justify-center p-4"><LoadingSpinner /></div>
              ) : historyData.length === 0 ? (
                <p className="text-sm text-slate-500">이력이 없습니다.</p>
              ) : (
                <div className="flex flex-col gap-3">
                  {historyData.map((h, i) => (
                    <div key={h.id} className="border-l-2 border-slate-200 pl-3">
                      <p className="text-xs text-slate-400">{formatDateTime(h.createdAt)} ({h.actorType})</p>
                      <p className="text-sm font-medium">
                        {h.oldStatus ? `${APPROVAL_LABEL[h.oldStatus as ApprovalStatus]} → ` : "초기 상태 설정: "}
                        {APPROVAL_LABEL[h.newStatus as ApprovalStatus]}
                      </p>
                      {h.reason && <p className="mt-1 text-xs text-slate-600 bg-slate-50 p-1.5 rounded">사유: {h.reason}</p>}
                    </div>
                  ))}
                </div>
              )}
            </div>
          </Modal>
        </>
      )}

      {activeTab === "incidents" && <IncidentList />}
    </div>
  );
}
