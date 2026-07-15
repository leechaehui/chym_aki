import { Navigate } from "react-router-dom";
import type { Role } from "@/types";
import { useAuthStore } from "@/store/authStore";
import { homePathFor } from "./navConfig";

/**
 * 인증 가드 — 미로그인 시 /login 으로, 직군 불일치 시 자신의 홈으로 보낸다.
 * 라우트 단위로 접근 권한을 강제(관심사 분리: 화면은 권한을 신경 쓰지 않음).
 */
export function ProtectedRoute({ allow, children }: { allow: Role; children: React.ReactNode }) {
  const user = useAuthStore((s) => s.user);
  if (!user) return <Navigate to="/login" replace />;
  if (user.role !== allow) return <Navigate to={homePathFor(user.role)} replace />;
  return <>{children}</>;
}
