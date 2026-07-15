import { useEffect } from "react";
import { PageHeader } from "@/components/common/PageHeader";
import { useChatStore } from "@/store/chatStore";
import { ChatSidebar } from "./ChatSidebar";
import { ChatRoom } from "./ChatRoom";

export function ChatPage() {
  const loadUsers = useChatStore((s) => s.loadUsers);
  const loadRooms = useChatStore((s) => s.loadRooms);
  const closeRoom = useChatStore((s) => s.closeRoom);
  const activeRoomId = useChatStore((s) => s.activeRoomId);

  useEffect(() => {
    loadUsers();
    loadRooms();
    return () => closeRoom();
  }, [loadUsers, loadRooms, closeRoom]);

  return (
    <div className="w-full">
      <PageHeader title="채팅" subtitle="의료진 간 실시간 메시지" />
      <div className="flex gap-4" style={{ height: "calc(100vh - 180px)" }}>
        <ChatSidebar />
        {activeRoomId ? (
          <ChatRoom roomId={activeRoomId} />
        ) : (
          <div className="flex flex-1 items-center justify-center rounded-xl border border-border bg-card text-sm text-muted-foreground">
            왼쪽에서 대화 상대를 선택하세요
          </div>
        )}
      </div>
    </div>
  );
}
