import { useEffect, useState } from "react";
import { Outlet, useLocation } from "react-router-dom";
import { Sidebar } from "./Sidebar";
import { TopBar } from "./TopBar";
import { NotificationOutlet } from "@/features/notification/NotificationOutlet";
import { ChatPanel } from "@/features/chat/ChatPanel";
import { useNotificationStore } from "@/store/notificationStore";
import { useAuthStore } from "@/store/authStore";
import { notificationService } from "@/services/notificationService";
import { demoService } from "@/services/demoService";
import { Settings, Play, RefreshCw, AlertTriangle, X } from "lucide-react";
import { queryClient } from "@/lib/queryClient";

/** 경로 → 상단바 제목. */
const TITLE: Record<string, string> = {
  "/admin": "관리자 콘솔",
  "/nephrology/consults": "신장내과 · 협진 센터",
  "/nephrology": "신장내과 · 대시보드",
  "/pathology": "병리과 · 판독",
};

/**
 * 인증 후 앱 셸 — 상단 알림 배너(전폭) + (사이드바 · 상단바 · 콘텐츠) + 전역 알림 채널 + 데모 시뮬레이터.
 */
export function AppLayout() {
  const { pathname } = useLocation();
  const role = useAuthStore((s) => s.user?.role);
  const hydrate = useNotificationStore((s) => s.hydrate);
  const startObserver = useNotificationStore((s) => s.startObserver);
  const clearAll = useNotificationStore((s) => s.clearAll);
  const setSimulating = useNotificationStore((s) => s.setSimulating);

  const [demoOpen, setDemoOpen] = useState(false);
  const [loading, setLoading] = useState<string | null>(null);
  const [countdown, setCountdown] = useState<number | null>(null);

  useEffect(() => {
    if (!role) return;
    notificationService.history(role).then(hydrate).catch(() => {});
    const stop = startObserver(role);
    return stop;
  }, [role, hydrate, startObserver]);

  const handleDemoAction = async (action: "setup" | "advance" | "trigger") => {
    if (loading !== null || countdown !== null) return;

    // 버튼 클릭 시 패널 닫기(카운트다운 진행 바는 상단에 계속 보임)
    setDemoOpen(false);

    // 3초 데이터 수집 및 분석 딜레이 연출
    setCountdown(3);
    const interval = setInterval(() => {
      setCountdown((prev) => {
        if (prev === null || prev <= 1) {
          clearInterval(interval);
          return null;
        }
        return prev - 1;
      });
    }, 1000);

    setTimeout(async () => {
      try {
        setLoading(action);
        if (action === "setup") {
          await demoService.setup();
          clearAll();
        } else if (action === "advance") {
          setSimulating(true);
          try {
            await demoService.advanceHour();
          } finally {
            // 웹소켓을 통한 이벤트 수신 딜레이를 고려하여 1.5초 후 팝업 억제 해제
            setTimeout(() => {
              setSimulating(false);
            }, 1500);
          }
        } else if (action === "trigger") {
          await demoService.triggerEvent();
        }
        
        // 🚨 전체 새로고침 없이 React Query 캐시를 백그라운드 리프레시하여 수치를 스무스하게 갱신!
        await queryClient.invalidateQueries();
        // React Query를 구독하지 않는 화면(ICU AKI 모니터링 등 로컬 fetch 방식)도 갱신되도록 전역 이벤트 브로드캐스트.
        window.dispatchEvent(new Event("chym:demo-refresh"));
      } catch (err) {
        alert("시뮬레이션 실행 오류: " + (err instanceof Error ? err.message : String(err)));
      } finally {
        setLoading(null);
      }
    }, 3000);
  };

  const title = Object.entries(TITLE).find(([p]) => pathname.startsWith(p))?.[1] ?? "CHYM";

  return (
    <div className="flex h-full flex-col">
      {/* 3초 카운트다운 진행 바 연출 (실시간 데이터 스트리밍 연출) */}
      {countdown !== null && (
        <div className="fixed top-0 left-0 right-0 z-[10000] h-1.5 w-full bg-muted">
          <div
            className="h-full bg-primary"
            style={{
              animation: "progress-bar 3s linear forwards",
              width: "0%",
            }}
          />
          <style>{`
            @keyframes progress-bar {
              0% { width: 0%; }
              100% { width: 100%; }
            }
          `}</style>
        </div>
      )}

      <NotificationOutlet />
      <div className="flex min-h-0 flex-1">
        <Sidebar />
        <div className="flex min-w-0 flex-1 flex-col">
          <TopBar title={title} />
          <main className="flex-1 overflow-y-auto bg-background p-5">
            <Outlet />
          </main>
        </div>
      </div>
      <ChatPanel />

      {/* 데모 시뮬레이터 플로팅 버튼 & 패널 (응급의학과 · 신장내과 역할 사용자에게 노출) */}
      {(role === "emergency" || role === "nephrology") && (
      <div className="fixed bottom-4 right-4 z-[9999] flex flex-col items-end gap-2">
        {demoOpen && (
          <div className="w-64 rounded-xl border border-border/80 bg-popover p-4 shadow-xl backdrop-blur-md">
            <style>{`
              @keyframes shimmer-effect {
                0% { background-position: 0% 50%; }
                50% { background-position: 100% 50%; }
                100% { background-position: 0% 50%; }
              }
              .animate-shimmer-blue {
                background: linear-gradient(270deg, #2563eb, #1d4ed8, #3b82f6, #1d4ed8);
                background-size: 400% 400%;
                animation: shimmer-effect 1.5s ease infinite;
              }
              .animate-shimmer-rose {
                background: linear-gradient(270deg, #e11d48, #be123c, #f43f5e, #be123c);
                background-size: 400% 400%;
                animation: shimmer-effect 1.5s ease infinite;
              }
            `}</style>
            <div className="mb-3 flex items-center justify-between border-b border-border/60 pb-2">
              <div className="flex items-center gap-1.5 font-semibold text-foreground">
                <Settings className="size-4 animate-spin text-primary" style={{ animationDuration: "3s" }} />
                <span>시연 시뮬레이터</span>
              </div>
              <button
                onClick={() => setDemoOpen(false)}
                className="rounded-md p-1 text-muted-foreground hover:bg-accent hover:text-foreground"
              >
                <X className="size-3.5" />
              </button>
            </div>

            <div className="flex flex-col gap-2">
              <button
                disabled={loading !== null || countdown !== null}
                onClick={() => handleDemoAction("setup")}
                className={`flex items-center justify-center gap-2 rounded-lg px-3 py-2 text-xs font-semibold text-white transition active:scale-95 duration-75 disabled:cursor-not-allowed ${
                  (loading === "setup" || (countdown !== null && loading === null))
                    ? "bg-slate-700 animate-pulse opacity-90"
                    : "bg-emerald-600 hover:bg-emerald-700"
                }`}
              >
                <RefreshCw className={`size-3.5 ${(loading === "setup" || countdown !== null) ? "animate-spin" : ""}`} />
                <span>
                  {countdown !== null && loading === null
                    ? `⏳ 데이터 동기화 중... (${countdown}s)`
                    : "🔄 시나리오 초기화 (Setup)"}
                </span>
              </button>

              <button
                disabled={loading !== null || countdown !== null}
                onClick={() => handleDemoAction("advance")}
                className={`flex items-center justify-center gap-2 rounded-lg px-3 py-2 text-xs font-semibold text-white transition active:scale-95 duration-75 disabled:cursor-not-allowed ${
                  (loading === "advance" || (countdown !== null && loading === null))
                    ? "animate-shimmer-blue opacity-90"
                    : "bg-blue-600 hover:bg-blue-700"
                }`}
              >
                <Play className={`size-3.5 ${(loading === "advance" || countdown !== null) ? "animate-spin" : ""}`} />
                <span>
                  {countdown !== null && loading === null
                    ? `⏳ 시간 스트리밍 중... (${countdown}s)`
                    : "⏰ 12시간 진행 (12h Advance)"}
                </span>
              </button>

              <button
                disabled={loading !== null || countdown !== null}
                onClick={() => handleDemoAction("trigger")}
                className={`flex items-center justify-center gap-2 rounded-lg px-3 py-2 text-xs font-semibold text-white transition active:scale-95 duration-75 disabled:cursor-not-allowed ${
                  (loading === "trigger" || (countdown !== null && loading === null))
                    ? "animate-shimmer-rose opacity-90"
                    : "bg-rose-600 hover:bg-rose-700"
                }`}
              >
                <AlertTriangle className={`size-3.5 ${(loading === "trigger" || countdown !== null) ? "animate-bounce" : ""}`} />
                <span>
                  {countdown !== null && loading === null
                    ? `⏳ AI 추론 연산 중... (${countdown}s)`
                    : "🚨 오준현 환자 악화 트리거"}
                </span>
              </button>
            </div>
          </div>
        )}

        <button
          onClick={() => setDemoOpen(!demoOpen)}
          className="flex items-center gap-1.5 rounded-full bg-primary px-4 py-2 text-xs font-bold text-white shadow-lg transition hover:scale-105 hover:bg-primary/90"
        >
          <Settings className="size-4" />
          <span>데모 시뮬레이션</span>
        </button>
      </div>
      )}
    </div>
  );
}
