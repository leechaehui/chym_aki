import { Eraser, MousePointer2, Pen, Ruler, Square } from "lucide-react";
import { cn } from "@/lib/cn";

const TOOLS = [
  { key: "pointer", icon: MousePointer2, label: "선택 / 이동" },
  { key: "pen",     icon: Pen,           label: "자유 주석" },
  { key: "region",  icon: Square,        label: "영역 표시" },
  { key: "ruler",   icon: Ruler,         label: "거리 측정" },
  { key: "eraser",  icon: Eraser,        label: "지우기" },
] as const;

const HINT: Record<string, string> = {
  pen:    "슬라이드에 드래그해 주석을 그립니다",
  region: "드래그해 영역을 표시합니다",
  ruler:  "두 점을 잇는 거리를 측정합니다",
  eraser: "주석을 클릭하면 삭제됩니다",
};

// 병리 검체 마킹(gross specimen inking)에 실제로 쓰는 표준 색 조합.
const PRESET_COLORS = ["#000000", "#2563eb", "#16a34a", "#eab308", "#f97316", "#dc2626"];

export function ToolboxRail({
  tool,
  onToolChange,
  drawColor,
  onColorChange,
}: {
  tool:          string;
  onToolChange:  (t: string) => void;
  drawColor:     string;
  onColorChange: (c: string) => void;
}) {
  const showColorPicker = tool === "pen" || tool === "region" || tool === "ruler";

  return (
    <div className="flex flex-col gap-4">
      {/* 도구 버튼 */}
      <div>
        <p className="mb-2 text-[11px] font-semibold uppercase tracking-wide text-muted-foreground">Toolbox</p>
        <div className="flex flex-wrap gap-1.5">
          {TOOLS.map((t) => (
            <button
              key={t.key}
              onClick={() => onToolChange(t.key)}
              title={t.label}
              className={cn(
                "flex size-9 items-center justify-center rounded-md border transition-colors",
                tool === t.key
                  ? "border-primary bg-primary/10 text-primary"
                  : "border-border text-muted-foreground hover:bg-muted",
              )}
            >
              <t.icon className="size-4" />
            </button>
          ))}
        </div>
        {tool !== "pointer" && (
          <p className="mt-1.5 text-[10px] text-muted-foreground">{HINT[tool]}</p>
        )}
      </div>

      {/* 색상 피커 */}
      {showColorPicker && (
        <div>
          <p className="mb-2 text-[11px] font-semibold uppercase tracking-wide text-muted-foreground">색상</p>
          <div className="flex flex-wrap items-center gap-1.5">
            {PRESET_COLORS.map((c) => (
              <button
                key={c}
                title={c}
                onClick={() => onColorChange(c)}
                className={cn(
                  "size-6 rounded-full border-2 transition-transform hover:scale-110",
                  drawColor === c ? "border-white scale-110" : "border-transparent",
                )}
                style={{ background: c }}
              />
            ))}
            <label className="relative cursor-pointer" title="커스텀 색상">
              <input
                type="color"
                value={drawColor}
                onChange={(e) => onColorChange(e.target.value)}
                className="absolute inset-0 h-full w-full cursor-pointer opacity-0"
              />
              <div
                className={cn(
                  "flex size-6 items-center justify-center rounded-full border-2 text-[9px] font-bold",
                  PRESET_COLORS.includes(drawColor) ? "border-transparent" : "scale-110 border-white",
                )}
                style={{ background: drawColor }}
              >
                {!PRESET_COLORS.includes(drawColor) && (
                  <span className="drop-shadow text-white">+</span>
                )}
              </div>
            </label>
          </div>
        </div>
      )}
    </div>
  );
}
