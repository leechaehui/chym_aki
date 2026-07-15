import { NavLink } from "react-router-dom";
import { Activity, LogOut, MessageCircle } from "lucide-react";
import { useAuthStore } from "@/store/authStore";
import { useChatStore } from "@/store/chatStore";
import { ROLE_NAV } from "@/routes/navConfig";
import { ROLE_LABEL } from "@/types";
import { cn } from "@/lib/cn";

/** 좌측 세로 네비게이션 — 직군 접근 화면 + 채팅 + 로그아웃. */
export function Sidebar() {
  const user = useAuthStore((s) => s.user);
  const logout = useAuthStore((s) => s.logout);
  const panelOpen = useChatStore((s) => s.panelOpen);
  const togglePanel = useChatStore((s) => s.togglePanel);
  if (!user) return null;

  return (
    <nav className="flex w-[76px] shrink-0 flex-col items-center gap-1 bg-sidebar py-3 text-sidebar-foreground">
      <div className="mb-3 flex size-10 items-center justify-center rounded-md bg-primary text-white">
        <Activity className="size-5" />
      </div>

      {ROLE_NAV[user.role].map((item) => (
        <NavLink
          key={item.path}
          to={item.path}
          end={item.end}
          className={({ isActive }) =>
            cn(
              "flex w-[60px] flex-col items-center gap-1 rounded-lg py-2 text-[10px] font-medium transition-colors",
              isActive ? "bg-sidebar-accent/20 text-white" : "text-sidebar-foreground hover:bg-white/5 hover:text-white",
            )
          }
        >
          <item.icon className="size-[18px]" />
          <span>{item.label}</span>
        </NavLink>
      ))}

      <div className="flex-1" />

      <button
        onClick={togglePanel}
        title="채팅"
        className={cn(
          "flex w-[60px] flex-col items-center gap-1 rounded-lg py-2 text-[10px] font-medium transition-colors",
          panelOpen ? "bg-sidebar-accent/20 text-white" : "text-sidebar-foreground hover:bg-white/5 hover:text-white",
        )}
      >
        <MessageCircle className="size-[18px]" />
        <span>채팅</span>
      </button>

      <button
        onClick={logout}
        title="로그아웃"
        className="flex w-[60px] flex-col items-center gap-1 rounded-lg py-2 text-[10px] text-sidebar-foreground hover:bg-white/5 hover:text-white"
      >
        <LogOut className="size-[18px]" />
        <span>로그아웃</span>
      </button>
      <span className="mt-1 text-[9px] text-sidebar-foreground/60">{ROLE_LABEL[user.role]}</span>
    </nav>
  );
}
