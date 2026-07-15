import { NavLink, Outlet } from "react-router-dom";
import { NEPHROLOGY_TABS } from "@/routes/navConfig";
import { cn } from "@/lib/cn";

/**
 * 신장내과 워크스페이스 셸 — 세그먼트 컨트롤 탭으로 세부 화면을 전환한다.
 */
export function NephrologyLayout() {
  return (
    <div className="w-full">
      <div className="mb-4 inline-flex rounded-lg bg-secondary p-1 gap-0.5">
        {NEPHROLOGY_TABS.map((t) => (
          <NavLink
            key={t.path}
            to={t.path}
            end={t.end}
            className={({ isActive }) =>
              cn(
                "rounded-md px-4 py-1.5 text-sm font-medium transition-colors whitespace-nowrap",
                isActive
                  ? "bg-card text-foreground shadow-sm"
                  : "text-muted-foreground hover:text-foreground",
              )
            }
          >
            {t.label}
          </NavLink>
        ))}
      </div>

      <Outlet />
    </div>
  );
}
