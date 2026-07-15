import type { LucideIcon } from "lucide-react";
import { LayoutDashboard, Stethoscope, Microscope, MessagesSquare, HeartPulse } from "lucide-react";
import type { Role } from "@/types";

export interface NavItem {
  path: string;
  label: string;
  icon: LucideIcon;
  /** NavLink 정확 일치(하위 경로에서 상위 메뉴가 활성으로 잡히지 않게). */
  end?: boolean;
}

/**
 * 직군별 네비게이션·기본 진입 경로. 각 직군은 자신의 부서 화면만 본다(접근 분리).
 * 배열 첫 항목이 로그인 직후 기본 경로.
 */
export const ROLE_NAV: Record<Role, NavItem[]> = {
  admin: [{ path: "/admin", label: "관리자", icon: LayoutDashboard }],
  nephrology: [{ path: "/nephrology", label: "신장내과", icon: Stethoscope }],
  pathology: [{ path: "/pathology", label: "병리과", icon: Microscope }],
};

/**
 * 신장내과 워크스페이스 상단 탭(병리과 판독 화면과 동일한 가로 탭 패턴).
 * 사이드바에는 '신장내과' 한 항목만 두고, 세부 화면은 이 탭으로 전환한다.
 */
export const NEPHROLOGY_TABS: NavItem[] = [
  { path: "/nephrology", label: "대시보드", icon: Stethoscope, end: true },
  { path: "/nephrology/icu-aki", label: "ICU AKI 모니터링", icon: HeartPulse },
  { path: "/nephrology/consults", label: "협진 센터", icon: MessagesSquare },
];

/** 직군 기본 진입 경로. */
export function homePathFor(role: Role): string {
  return ROLE_NAV[role][0].path;
}
