import { useEffect, useMemo } from "react";
import { useNavigate } from "react-router-dom";
import { Bell, User, Clock, Siren, Microscope } from "lucide-react";
import { useAuthStore } from "@/store/authStore";
import { useNotificationStore } from "@/store/notificationStore";
import { useConsultStore } from "@/store/consultStore";
import { ROLE_LABEL } from "@/types";

/** 헤더 협진 배지 — 대기 건수 알림(클릭 시 협진 화면 이동). */
function ConsultBadge({
  icon: Icon,
  label,
  count,
  tone,
  onClick,
}: {
  icon: typeof Clock;
  label: string;
  count: number;
  tone: "warning" | "danger";
  onClick: () => void;
}) {
  if (count === 0) return null;
  const cls = tone === "danger" ? "bg-destructive/10 text-destructive hover:bg-destructive/15" : "bg-warning/15 text-warning hover:bg-warning/25";
  return (
    <button
      onClick={onClick}
      className={`flex items-center gap-1.5 rounded-full px-2.5 py-1 text-[11px] font-semibold transition-colors ${cls}`}
      title={`${label} ${count}건`}
    >
      <Icon className="size-3.5" />
      {label} {count}건
    </button>
  );
}

/** 상단바 — 현재 화면 제목 + 협진 대기 배지 + 알림 벨(미읽음 배지) + 로그인 사용자. */
export function TopBar({ title }: { title: string }) {
  const navigate = useNavigate();
  const user = useAuthStore((s) => s.user);
  const setDrawerOpen = useNotificationStore((s) => s.setDrawerOpen);
  const unread = useNotificationStore((s) => s.items.filter((n) => !n.read && n.department === user?.role).length);

  const consults = useConsultStore((s) => s.items);
  const loadConsults = useConsultStore((s) => s.load);
  useEffect(() => {
    if (consults.length === 0) loadConsults();
  }, [consults.length, loadConsults]);

  // 직군별 협진 대기 집계 — 신장내과는 수신 응급 협진, 병리과는 판독 대기 협진.
  const badges = useMemo(() => {
    const pending = (kind: "nephrology" | "pathology") =>
      consults.filter((c) => c.kind === kind && (c.status === "requested" || c.status === "in_progress"));
    if (user?.role === "nephrology") {
      const p = pending("nephrology");
      return { wait: p.length, emergency: p.filter((c) => c.urgency === "emergency").length };
    }
    if (user?.role === "pathology") {
      return { path: pending("pathology").length };
    }
    return {};
  }, [consults, user?.role]);

  return (
    <header className="flex h-14 shrink-0 items-center justify-between border-b border-border bg-card px-5">
      <div className="flex items-center gap-2">
        <span className="size-2 rounded-full bg-primary" />
        <h1 className="text-sm font-semibold text-foreground">
          {user ? `${user.name}님 환영합니다` : "CHYM"}
          <span className="ml-2 font-normal text-muted-foreground">· {title}</span>
        </h1>
      </div>
      <div className="flex items-center gap-3">
        {/* 협진 대기 배지 — 직군별 */}
        {user?.role === "nephrology" && (
          <div className="flex items-center gap-1.5">
            <ConsultBadge icon={Clock} label="협진 대기" count={badges.wait ?? 0} tone="warning" onClick={() => navigate("/nephrology/consults")} />
            <ConsultBadge icon={Siren} label="응급 협진" count={badges.emergency ?? 0} tone="danger" onClick={() => navigate("/nephrology/consults")} />
          </div>
        )}
        {user?.role === "pathology" && (
          <ConsultBadge icon={Microscope} label="병리 협진" count={badges.path ?? 0} tone="warning" onClick={() => navigate("/pathology?tab=consult")} />
        )}

        <button
          onClick={() => setDrawerOpen(true)}
          className="relative flex size-9 items-center justify-center rounded-lg text-muted-foreground hover:bg-muted hover:text-foreground"
          title="알림"
        >
          <Bell className="size-[18px]" />
          {unread > 0 && (
            <span className="absolute right-1 top-1 flex min-w-4 items-center justify-center rounded-full bg-destructive px-1 text-[9px] font-bold text-white">
              {unread > 9 ? "9+" : unread}
            </span>
          )}
        </button>
        {user && (
          <button 
            className="flex items-center gap-2 text-left hover:opacity-80 transition-opacity" 
            onClick={() => navigate("/profile")}
            title="내 프로필"
          >
            <div className="text-right">
              <p className="text-xs font-semibold text-foreground">{user.name}</p>
              <p className="text-[10px] text-muted-foreground">{ROLE_LABEL[user.role]}</p>
            </div>
            <div className="flex size-8 items-center justify-center rounded-full bg-secondary text-secondary-foreground">
              <User className="size-4" />
            </div>
          </button>
        )}
      </div>
    </header>
  );
}
