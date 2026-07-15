import { Search } from "lucide-react";
import { Input } from "@/components/ui/input";
import { cn } from "@/lib/cn";

/** 검색 입력 — 아이콘 + controlled value. 목록 필터에 재사용. */
export function SearchInput({
  value,
  onChange,
  placeholder = "검색",
  className,
}: {
  value: string;
  onChange: (v: string) => void;
  placeholder?: string;
  className?: string;
}) {
  return (
    <div className={cn("relative", className)}>
      <Search className="pointer-events-none absolute left-2.5 top-1/2 size-4 -translate-y-1/2 text-muted-foreground" />
      <Input value={value} onChange={(e) => onChange(e.target.value)} placeholder={placeholder} className="pl-8" />
    </div>
  );
}
