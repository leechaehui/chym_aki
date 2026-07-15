import { useAuthStore } from "@/store/authStore";
import { useChatStore } from "@/store/chatStore";
import { cn } from "@/lib/cn";
import { ROLE_LABEL } from "@/types";
import type { ChatUser } from "@/types";

const ROLE_COLOR: Record<string, string> = {
  admin: "bg-purple-100 text-purple-700",
  emergency: "bg-red-100 text-red-700",
  nephrology: "bg-blue-100 text-blue-700",
  pathology: "bg-green-100 text-green-700",
};

function UserRow({ user, isActive }: { user: ChatUser; isActive: boolean }) {
  const openRoom = useChatStore((s) => s.openRoom);

  return (
    <button
      onClick={() => openRoom(user.id)}
      className={cn(
        "flex w-full items-center gap-3 rounded-lg px-3 py-2 text-left transition-colors",
        isActive
          ? "bg-primary/10 text-primary"
          : "hover:bg-muted text-foreground",
      )}
    >
      <div className="flex size-8 shrink-0 items-center justify-center rounded-full bg-muted text-xs font-bold text-muted-foreground">
        {user.name[0]}
      </div>
      <div className="min-w-0 flex-1">
        <p className="truncate text-sm font-medium">{user.name}</p>
        <span
          className={cn(
            "inline-block rounded px-1 py-0.5 text-[10px] font-medium",
            ROLE_COLOR[user.role] ?? "bg-gray-100 text-gray-600",
          )}
        >
          {ROLE_LABEL[user.role as keyof typeof ROLE_LABEL] ?? user.role}
        </span>
      </div>
    </button>
  );
}

export function ChatSidebar() {
  const me = useAuthStore((s) => s.user);
  const users = useChatStore((s) => s.users);
  const rooms = useChatStore((s) => s.rooms);
  const activeRoomId = useChatStore((s) => s.activeRoomId);

  function getPeerIdFromRoom(memberIds: string): string {
    const ids = memberIds.split("_");
    return ids.find((id) => id !== me?.id) ?? ids[0];
  }

  const activeRoom = rooms.find((r) => r.id === activeRoomId);
  const activePeerId = activeRoom ? getPeerIdFromRoom(activeRoom.memberIds) : null;

  return (
    <div className="flex w-56 shrink-0 flex-col gap-1 rounded-xl border border-border bg-card p-2">
      <p className="px-2 py-1 text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">
        의료진
      </p>
      {users.length === 0 && (
        <p className="px-2 text-xs text-muted-foreground">사용자 없음</p>
      )}
      {users.map((u) => (
        <UserRow key={u.id} user={u} isActive={u.id === activePeerId} />
      ))}
    </div>
  );
}
