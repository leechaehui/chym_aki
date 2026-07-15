import * as React from "react";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "@/lib/cn";

/** Badge — 상태/분류 표시용 알약. tone 으로 의미 색상 지정. */
const badgeVariants = cva("inline-flex items-center gap-1 rounded-full px-2.5 py-0.5 text-[11px] font-semibold", {
  variants: {
    tone: {
      neutral: "bg-secondary text-secondary-foreground",
      info: "bg-info/10 text-info",
      success: "bg-success/10 text-success",
      warning: "bg-warning/15 text-warning",
      high: "bg-high/10 text-high",
      danger: "bg-destructive/10 text-destructive",
      primary: "bg-primary/10 text-primary",
    },
  },
  defaultVariants: { tone: "neutral" },
});

export interface BadgeProps extends React.HTMLAttributes<HTMLSpanElement>, VariantProps<typeof badgeVariants> {}

export function Badge({ className, tone, ...props }: BadgeProps) {
  return <span className={cn(badgeVariants({ tone }), className)} {...props} />;
}
