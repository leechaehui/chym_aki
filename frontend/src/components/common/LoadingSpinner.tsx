import { Loader2 } from "lucide-react";
import { cn } from "@/lib/cn";

/** 로딩 스피너 — 비동기 대기 표시. */
export function LoadingSpinner({ className, label }: { className?: string; label?: string }) {
  return (
    <div className={cn("flex flex-col items-center justify-center gap-2 py-10 text-muted-foreground", className)}>
      <Loader2 className="size-6 animate-spin text-primary" />
      {label && <span className="text-xs">{label}</span>}
    </div>
  );
}
