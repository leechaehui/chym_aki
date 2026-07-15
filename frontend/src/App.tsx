import { RouterProvider } from "react-router-dom";
import { QueryClientProvider } from "@tanstack/react-query";
import { queryClient } from "@/lib/queryClient";
import { router } from "@/routes";

/**
 * 앱 루트 — 라우터.
 * 알림 채널(Toast/Banner/Drawer/Modal)은 라우터 컨텍스트가 필요하므로 AppLayout 의
 * NotificationOutlet 에서 렌더한다(useNavigate 사용).
 * 라이트 모드 전용(다크모드 미지원).
 */
export function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <RouterProvider router={router} />
    </QueryClientProvider>
  );
}
