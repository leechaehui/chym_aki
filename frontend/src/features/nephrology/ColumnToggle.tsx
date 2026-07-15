import { useEffect, useRef, useState } from "react";
import { ChevronDown } from "lucide-react";
import { cn } from "@/lib/cn";

export interface ColumnDef {
  readonly key: string;
  readonly label: string;
  /** 식별자 등 필수 컬럼은 숨길 수 없게 잠근다. */
  readonly locked?: boolean;
}

/**
 * 표 컬럼 표시/숨김 상태를 localStorage 로 보존하는 훅.
 * 숨긴 컬럼 키 집합만 저장한다(기본은 전체 표시).
 */
export function useColumnVisibility(storageKey: string, columns: ColumnDef[]) {
  const [hidden, setHidden] = useState<Set<string>>(() => {
    try {
      const raw = localStorage.getItem(storageKey);
      return new Set<string>(raw ? (JSON.parse(raw) as string[]) : []);
    } catch {
      return new Set<string>();
    }
  });

  useEffect(() => {
    try {
      localStorage.setItem(storageKey, JSON.stringify([...hidden]));
    } catch {
      /* 저장 실패는 무시(프라이빗 모드 등) */
    }
  }, [storageKey, hidden]);

  const toggle = (key: string) =>
    setHidden((prev) => {
      const next = new Set(prev);
      if (next.has(key)) next.delete(key);
      else next.add(key);
      return next;
    });

  const isVisible = (key: string) => !hidden.has(key);
  const shownCount = columns.filter((c) => isVisible(c.key)).length;
  return { isVisible, toggle, shownCount };
}

/** 체크박스 팝오버로 표시할 컬럼을 의료진이 직접 고른다. */
export function ColumnToggle({
  columns,
  isVisible,
  toggle,
  shownCount,
}: {
  columns: ColumnDef[];
  isVisible: (key: string) => boolean;
  toggle: (key: string) => void;
  shownCount: number;
}) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    const onDown = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener("mousedown", onDown);
    return () => document.removeEventListener("mousedown", onDown);
  }, [open]);

  return (
    <div ref={ref} className="relative">
      <button
        onClick={() => setOpen((v) => !v)}
        aria-haspopup="listbox"
        aria-expanded={open}
        className="flex items-center gap-1.5 rounded-md border border-border bg-card px-2.5 py-1 text-[11px] font-medium text-foreground hover:bg-muted"
        title="표시할 컬럼 선택"
      >
        컬럼 {shownCount}/{columns.length}
        <ChevronDown className={cn("size-3.5 transition-transform", open && "rotate-180")} />
      </button>
      {open && (
        <div className="absolute right-0 z-30 mt-1 w-52 rounded-md border border-border bg-card p-1 shadow-lg">
          <div className="px-2 py-1 text-[10px] font-medium uppercase tracking-wide text-muted-foreground">
            표시할 컬럼
          </div>
          {columns.map((c) => (
            <label
              key={c.key}
              className={cn(
                "flex items-center gap-2 rounded px-2 py-1.5 text-[12px]",
                c.locked ? "cursor-not-allowed opacity-60" : "cursor-pointer hover:bg-muted",
              )}
            >
              <input
                type="checkbox"
                checked={isVisible(c.key)}
                disabled={c.locked}
                onChange={() => !c.locked && toggle(c.key)}
                className="accent-primary"
              />
              {c.label}
            </label>
          ))}
        </div>
      )}
    </div>
  );
}
