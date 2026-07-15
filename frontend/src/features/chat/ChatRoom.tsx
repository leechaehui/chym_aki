import { useEffect, useRef, useState } from "react";
import { Send } from "lucide-react";
import { useAuthStore } from "@/store/authStore";
import { useChatStore } from "@/store/chatStore";
import { cn } from "@/lib/cn";
import type { ChatMessage } from "@/types";

function Bubble({ msg, isMine }: { msg: ChatMessage; isMine: boolean }) {
  const time = new Date(msg.createdAt).toLocaleTimeString("ko-KR", {
    hour: "2-digit",
    minute: "2-digit",
  });

  return (
    <div className={cn("flex flex-col gap-0.5", isMine ? "items-end" : "items-start")}>
      {!isMine && (
        <span className="px-1 text-[11px] text-muted-foreground">{msg.senderName}</span>
      )}
      <div className="flex items-end gap-1.5">
        {isMine && <span className="text-[10px] text-muted-foreground">{time}</span>}
        <div
          className={cn(
            "max-w-[280px] break-words rounded-2xl px-3.5 py-2 text-sm",
            isMine
              ? "rounded-br-sm bg-primary text-primary-foreground"
              : "rounded-bl-sm bg-muted text-foreground",
          )}
        >
          {msg.body}
        </div>
        {!isMine && <span className="text-[10px] text-muted-foreground">{time}</span>}
      </div>
    </div>
  );
}

export function ChatRoom({ roomId }: { roomId: string }) {
  const me = useAuthStore((s) => s.user);
  const messages = useChatStore((s) => s.messages[roomId] ?? []);
  const sendMessage = useChatStore((s) => s.sendMessage);
  const [input, setInput] = useState("");
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  function handleSend() {
    const body = input.trim();
    if (!body) return;
    sendMessage(body);
    setInput("");
  }

  function handleKeyDown(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  }

  return (
    <div className="flex flex-1 flex-col rounded-xl border border-border bg-card overflow-hidden">
      <div className="flex-1 overflow-y-auto p-4 space-y-3">
        {messages.length === 0 && (
          <p className="text-center text-xs text-muted-foreground pt-8">
            대화를 시작해보세요.
          </p>
        )}
        {messages.map((msg) => (
          <Bubble key={msg.id} msg={msg} isMine={msg.senderId === me?.id} />
        ))}
        <div ref={bottomRef} />
      </div>

      <div className="border-t border-border p-3 flex gap-2">
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="메시지 입력 (Enter 전송, Shift+Enter 줄바꿈)"
          rows={1}
          className="flex-1 resize-none rounded-lg border border-border bg-background px-3 py-2 text-sm outline-none focus:ring-1 focus:ring-primary"
        />
        <button
          onClick={handleSend}
          disabled={!input.trim()}
          className="flex items-center justify-center rounded-lg bg-primary px-3 text-primary-foreground disabled:opacity-40 hover:bg-primary/90 transition-colors"
        >
          <Send className="size-4" />
        </button>
      </div>
    </div>
  );
}
