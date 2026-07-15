import { createHashRouter, Navigate } from "react-router-dom";
import { useAuthStore } from "@/store/authStore";
import { homePathFor } from "./navConfig";
import { ProtectedRoute } from "./ProtectedRoute";
import { AppLayout } from "@/layouts/AppLayout";
import { LoginPage } from "@/features/auth/LoginPage";
import { AdminDashboard } from "@/features/admin/AdminDashboard";
import { NephrologyLayout } from "@/features/nephrology/NephrologyLayout";
import { NephrologyWorkspace } from "@/features/nephrology/NephrologyWorkspace";
import { ConsultationCenter } from "@/features/consult/ConsultationCenter";
import { IcuAkiMonitor } from "@/features/nephrology/IcuAkiMonitor";
import { PathologyWorkspace } from "@/features/pathology/PathologyWorkspace";
import { ProfilePage } from "@/features/auth/ProfilePage";
import { IncidentReportPrint } from "@/features/admin/incident/IncidentReportPrint";
import { TelemetryAdmin } from "@/features/admin/TelemetryAdmin";

/** 로그인 상태면 자신의 홈으로, 아니면 로그인 화면으로 보내는 인덱스 리디렉터. */
function IndexRedirect() {
  const user = useAuthStore((s) => s.user);
  return <Navigate to={user ? homePathFor(user.role) : "/login"} replace />;
}

/** 로그인 화면 가드 — 이미 로그인했으면 홈으로. */
function LoginGate() {
  const user = useAuthStore((s) => s.user);
  if (user) return <Navigate to={homePathFor(user.role)} replace />;
  return <LoginPage />;
}

/**
 * 라우터 — HashRouter 사용(정적/오프라인 배포에서도 동작: dist 를 file:// 로 열어도 OK).
 * 보호 라우트는 AppLayout 아래에 두고 직군 가드로 접근을 분리한다.
 */
export const router = createHashRouter([
  { path: "/login", element: <LoginGate /> },
  { 
    path: "/admin/incident/print/:id", 
    element: <ProtectedRoute allow="admin"><IncidentReportPrint /></ProtectedRoute> 
  },
  {
    path: "/",
    element: <AppLayout />,
    children: [
      { index: true, element: <IndexRedirect /> },
      { path: "profile", element: <ProfilePage /> },
      { path: "admin", element: <ProtectedRoute allow="admin"><AdminDashboard /></ProtectedRoute> },
      { path: "admin/server-monitoring", element: <ProtectedRoute allow="admin"><TelemetryAdmin /></ProtectedRoute> },
      {
        path: "nephrology",
        element: <ProtectedRoute allow="nephrology"><NephrologyLayout /></ProtectedRoute>,
        children: [
          { index: true, element: <NephrologyWorkspace /> },
          { path: "consults", element: <ConsultationCenter /> },
          { path: "icu-aki", element: <IcuAkiMonitor /> },
        ],
      },
      { path: "pathology", element: <ProtectedRoute allow="pathology"><PathologyWorkspace /></ProtectedRoute> },
    ],
  },
  { path: "*", element: <Navigate to="/" replace /> },
]);
