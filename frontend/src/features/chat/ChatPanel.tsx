import { useMemo, useState } from "react";
import { X, ChevronLeft, Search } from "lucide-react";
import { useAuthStore } from "@/store/authStore";
import { useChatStore } from "@/store/chatStore";
import { cn } from "@/lib/cn";
import { ROLE_LABEL } from "@/types";
import type { ChatUser } from "@/types";
import { ChatRoom } from "./ChatRoom";

const ROLE_COLOR: Record<string, string> = {
  nephrology: "bg-blue-50 text-blue-700",
  pathology:  "bg-green-50 text-green-700",
  admin:      "bg-purple-50 text-purple-700",
};

function Avatar({ name, role }: { name: string; role: string }) {
  return (
    <div className={cn(
      "flex size-9 shrink-0 items-center justify-center rounded-full text-sm font-semibold",
      ROLE_COLOR[role] ?? "bg-muted text-muted-foreground",
    )}>
      {name[0]}
    </div>
  );
}

export function ChatPanel() {
  const panelOpen   = useChatStore((s) => s.panelOpen);
  const closePanel  = useChatStore((s) => s.closePanel);
  const activeRoomId = useChatStore((s) => s.activeRoomId);
  const closeRoom   = useChatStore((s) => s.closeRoom);
  const openRoom    = useChatStore((s) => s.openRoom);
  const users       = useChatStore((s) => s.users);
  const rooms       = useChatStore((s) => s.rooms);
  const messages    = useChatStore((s) => s.messages);
  const me          = useAuthStore((s) => s.user);

  const [query, setQuery] = useState("");
  const [tab, setTab] = useState<"chats" | "staff">("staff");

  const getPeer = (room: { memberIds: string }): ChatUser | undefined => {
    const peerId = room.memberIds.split("_").find((id) => id !== me?.id);
    return users.find((u) => u.id === peerId);
  };

  // 최근 대화방 — 메시지가 실제로 오간 방만, 마지막 메시지 시간순 정렬
  const recentRooms = useMemo(() => {
    return rooms
      .map((r) => {
        const msgs = messages[r.id] ?? [];
        const last = msgs[msgs.length - 1];
        return { room: r, peer: getPeer(r), last };
      })
      .filter((r) => r.peer && r.last)
      .sort((a, b) => {
        const ta = a.last?.createdAt ?? a.room.createdAt;
        const tb = b.last?.createdAt ?? b.room.createdAt;
        return tb.localeCompare(ta);
      });
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [rooms, messages, users, me]);

  // 직원 탭 — 카톡 친구탭처럼 대화 여부와 무관하게 전체 직원 목록(검색 포함)
  const staffList = useMemo(() => {
    return users
      .filter((u) => u.id !== me?.id)
      .filter((u) => !query || u.name.includes(query) || ROLE_LABEL[u.role as keyof typeof ROLE_LABEL]?.includes(query));
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [users, me, query]);

  const filteredRooms = useMemo(() => {
    if (!query) return recentRooms;
    return recentRooms.filter((r) =>
      r.peer?.name.includes(query) || ROLE_LABEL[r.peer?.role as keyof typeof ROLE_LABEL]?.includes(query),
    );
  }, [recentRooms, query]);

  const activeRoom = rooms.find((r) => r.id === activeRoomId);
  const activePeer = activeRoom ? getPeer(activeRoom) : undefined;

  function handleOpenRoom(user: ChatUser) {
    openRoom(user.id);
  }

  function formatTime(iso: string) {
    const d = new Date(iso);
    const now = new Date();
    if (d.toDateString() === now.toDateString()) {
      return d.toLocaleTimeString("ko-KR", { hour: "2-digit", minute: "2-digit" });
    }
    return d.toLocaleDateString("ko-KR", { month: "numeric", day: "numeric" });
  }

  return (
    <>
      {panelOpen && (
        <div className="fixed inset-0 z-30 bg-black/10 lg:hidden" onClick={closePanel} />
      )}

      <div className={cn(
        "fixed right-0 top-0 z-40 flex h-full w-[340px] flex-col border-l border-border bg-card transition-transform duration-300",
        panelOpen ? "translate-x-0" : "translate-x-full",
      )}>
        {/* 헤더 */}
        <div className="flex items-center gap-2 border-b border-border px-4 py-3">
          {activeRoomId && (
            <button
              onClick={() => closeRoom()}
              className="flex size-7 items-center justify-center rounded-md text-muted-foreground hover:bg-muted"
            >
              <ChevronLeft className="size-4" />
            </button>
          )}
          <span className="flex-1 text-sm font-semibold text-foreground">
            {activeRoomId && activePeer ? activePeer.name : "채팅"}
          </span>
          {activeRoomId && activePeer && (
            <span className="text-[11px] text-muted-foreground">
              {ROLE_LABEL[activePeer.role as keyof typeof ROLE_LABEL] ?? activePeer.role}
            </span>
          )}
          <button
            onClick={closePanel}
            className="flex size-7 items-center justify-center rounded-md text-muted-foreground hover:bg-muted"
          >
            <X className="size-4" />
          </button>
        </div>

        {/* 본문 */}
        {activeRoomId ? (
          <div className="flex min-h-0 flex-1 flex-col overflow-hidden p-3">
            <ChatRoom key={activeRoomId} roomId={activeRoomId} />
          </div>
        ) : (
          <div className="flex min-h-0 flex-1 flex-col overflow-hidden">
            {/* 탭 — 카카오톡처럼 채팅/직원 분리 */}
            <div className="flex border-b border-border">
              {([
                { key: "staff", label: "직원" },
                { key: "chats", label: "채팅" },
              ] as const).map((t) => (
                <button
                  key={t.key}
                  onClick={() => setTab(t.key)}
                  className={cn(
                    "flex-1 py-2.5 text-sm font-semibold transition-colors",
                    tab === t.key
                      ? "border-b-2 border-primary text-primary"
                      : "text-muted-foreground hover:text-foreground",
                  )}
                >
                  {t.label}
                </button>
              ))}
            </div>

            {/* 검색 */}
            <div className="px-3 py-2 border-b border-border">
              <div className="flex items-center gap-2 rounded-md border border-border bg-background px-3 py-1.5">
                <Search className="size-3.5 shrink-0 text-muted-foreground" />
                <input
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  placeholder="이름 또는 부서"
                  className="flex-1 bg-transparent text-sm outline-none placeholder:text-muted-foreground"
                />
              </div>
            </div>

            <div className="flex-1 overflow-y-auto">
              {/* 채팅 탭 — 대화 중인 방 목록 */}
              {tab === "chats" && (
                filteredRooms.length > 0 ? (
                  filteredRooms.map(({ room, peer, last }) => (
                    <button
                      key={room.id}
                      onClick={() => handleOpenRoom(peer!)}
                      className="flex w-full items-center gap-3 px-4 py-2.5 text-left hover:bg-muted/60 transition-colors"
                    >
                      <Avatar name={peer!.name} role={peer!.role} />
                      <div className="min-w-0 flex-1">
                        <div className="flex items-baseline justify-between gap-1">
                          <span className="text-sm font-medium text-foreground">{peer!.name}</span>
                          {last && (
                            <span className="shrink-0 text-[10px] text-muted-foreground">
                              {formatTime(last.createdAt)}
                            </span>
                          )}
                        </div>
                        <p className="truncate text-[11px] text-muted-foreground">
                          {last ? last.body : ROLE_LABEL[peer!.role as keyof typeof ROLE_LABEL] ?? peer!.role}
                        </p>
                      </div>
                    </button>
                  ))
                ) : (
                  <p className="px-4 py-8 text-center text-sm text-muted-foreground">
                    {query ? "검색 결과가 없습니다." : "대화 중인 채팅이 없습니다."}
                  </p>
                )
              )}

              {/* 직원 탭 — 대화 여부와 무관한 전체 직원 목록 */}
              {tab === "staff" && (
                staffList.length > 0 ? (
                  staffList.map((u) => (
                    <button
                      key={u.id}
                      onClick={() => handleOpenRoom(u)}
                      className="flex w-full items-center gap-3 px-4 py-2.5 text-left hover:bg-muted/60 transition-colors"
                    >
                      <Avatar name={u.name} role={u.role} />
                      <div className="min-w-0 flex-1">
                        <span className="text-sm font-medium text-foreground">{u.name}</span>
                        <p className="text-[11px] text-muted-foreground">
                          {ROLE_LABEL[u.role as keyof typeof ROLE_LABEL] ?? u.role}
                        </p>
                      </div>
                    </button>
                  ))
                ) : (
                  <p className="px-4 py-8 text-center text-sm text-muted-foreground">검색 결과가 없습니다.</p>
                )
              )}
            </div>
          </div>
        )}
      </div>
    </>
  );
}
