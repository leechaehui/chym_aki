import { create } from "zustand";
import { persist, createJSONStorage } from "zustand/middleware";
import type { SessionUser } from "@/types";
import { authService } from "@/services/authService";

interface AuthState {
  user: SessionUser | null;
  loading: boolean;
  error: string | null;
  login: (username: string, password: string) => Promise<boolean>;
  logout: () => void;
  setUser: (user: SessionUser) => void;
}

/**
 * 인증 전역 상태. 로그인 사용자를 보관하고 localStorage 에 영속화한다(새로고침 유지).
 * UI 에는 SessionUser(비밀번호 없음)만 노출 — 민감정보가 전역 상태로 새지 않게 한다.
 */
export const useAuthStore = create<AuthState>()(
  persist(
    (set) => ({
      user: null,
      loading: false,
      error: null,
      async login(username, password) {
        set({ loading: true, error: null });
        try {
          const user = await authService.login(username, password);
          set({ user, loading: false });
          return true;
        } catch (e) {
          set({ error: (e as Error).message, loading: false });
          throw e; // Rethrow to let the component handle it
        }
      },
      logout() {
        authService.logout(); // 저장된 JWT 토큰 폐기
        set({ user: null, error: null });
      },
      setUser(user) {
        set({ user });
      },
    }),
    { name: "chym-auth", partialize: (s) => ({ user: s.user }), storage: createJSONStorage(() => sessionStorage) },
  ),
);

// JWT 자동 갱신 실패 시 강제 로그아웃 — http.ts 의 auth:expired 이벤트 수신.
if (typeof window !== "undefined") {
  window.addEventListener("auth:expired", () => {
    useAuthStore.getState().logout();
  });
}
