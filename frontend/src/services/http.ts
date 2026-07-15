/**
 * HTTP 클라이언트 — FastAPI 백엔드 연동 지점.
 *
 * 모든 서비스(Facade)는 이 클라이언트를 통해 백엔드와 통신한다.
 * - JWT 토큰을 보관/주입하고 localStorage 에 영속화(새로고침 유지)한다.
 * - 토큰 만료 시 저장된 자격증명으로 자동 재발급(silent refresh)한다.
 * - 비-2xx 응답은 백엔드의 {detail} 메시지로 Error 를 던진다
 *   → 기존 store/컴포넌트의 try/catch 흐름이 그대로 동작한다.
 */

/** API 베이스 URL — Vite 환경변수로 주입(기본: 로컬 백엔드). */
const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000/api";

const TOKEN_KEY = "chym-token";
const CRED_KEY = "chym-cred";

// localStorage → sessionStorage 마이그레이션: 기존 영속 세션 제거
localStorage.removeItem(TOKEN_KEY);
localStorage.removeItem(CRED_KEY);
localStorage.removeItem("chym-auth");

let accessToken: string | null = sessionStorage.getItem(TOKEN_KEY);
let isRefreshing = false;
let refreshPromise: Promise<string | null> | null = null;

export function setToken(token: string): void {
  accessToken = token;
  sessionStorage.setItem(TOKEN_KEY, token);
}

export function clearToken(): void {
  accessToken = null;
  sessionStorage.removeItem(TOKEN_KEY);
  sessionStorage.removeItem(CRED_KEY);
}

export function getToken(): string | null {
  return accessToken;
}

/** 로그인 성공 시 자격증명 저장 (자동 갱신용). */
export function saveCredentials(username: string, password: string): void {
  const encoded = btoa(encodeURIComponent(`${username}:${password}`));
  sessionStorage.setItem(CRED_KEY, encoded);
}

/** 저장된 자격증명 복원. */
function getCredentials(): { username: string; password: string } | null {
  const encoded = sessionStorage.getItem(CRED_KEY);
  if (!encoded) return null;
  try {
    const decoded = decodeURIComponent(atob(encoded));
    const [username, ...rest] = decoded.split(":");
    return { username, password: rest.join(":") };
  } catch {
    return null;
  }
}

/** JWT payload 에서 만료시간 추출 (초 단위 Unix timestamp). */
function getTokenExp(token: string): number | null {
  try {
    let base64Url = token.split(".")[1];
    let base64 = base64Url.replace(/-/g, "+").replace(/_/g, "/");
    let pad = base64.length % 4;
    if (pad) {
      base64 += "=".repeat(4 - pad);
    }
    const payload = JSON.parse(atob(base64));
    return payload.exp ?? null;
  } catch {
    return null;
  }
}

/** 토큰이 만료됐거나 60초 이내 만료 예정인지 확인. */
function isTokenExpired(token: string): boolean {
  const exp = getTokenExp(token);
  if (exp === null) return true;
  // 60초 여유를 두고 미리 갱신
  return Date.now() / 1000 >= exp - 60;
}

/** 자격증명으로 새 토큰 발급 (silent refresh). */
async function silentRefresh(): Promise<string | null> {
  const cred = getCredentials();
  if (!cred) return null;

  try {
    const res = await fetch(`${API_BASE}/auth/login`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(cred),
    });
    if (!res.ok) return null;
    const data = await res.json();
    const newToken = data.accessToken as string;
    setToken(newToken);
    return newToken;
  } catch {
    return null;
  }
}

/** 유효한 토큰 확보 (만료 시 자동 갱신). 동시 요청 방지를 위해 단일 promise 공유. */
async function ensureValidToken(): Promise<string | null> {
  if (!accessToken) return null;
  if (!isTokenExpired(accessToken)) return accessToken;

  // 이미 갱신 중이면 기존 promise 대기
  if (isRefreshing && refreshPromise) {
    return refreshPromise;
  }

  isRefreshing = true;
  refreshPromise = silentRefresh().finally(() => {
    isRefreshing = false;
    refreshPromise = null;
  });
  return refreshPromise;
}

export class ApiError extends Error {
  code?: string;
  reason?: string;

  constructor(message: string, code?: string, reason?: string) {
    super(message);
    this.name = "ApiError";
    this.code = code;
    this.reason = reason;
  }
}

/** 백엔드 에러 응답 형태. */
interface ErrorBody {
  detail?: string | { code?: string; reason?: string };
}

async function parseError(res: Response): Promise<Error> {
  try {
    const body = (await res.json()) as ErrorBody;
    if (body.detail && typeof body.detail === "object") {
      return new ApiError(body.detail.reason || body.detail.code || "요청 실패", body.detail.code, body.detail.reason);
    }
    return new ApiError((body.detail as string) || res.statusText);
  } catch {
    return new ApiError(res.statusText || `요청 실패 (${res.status})`);
  }
}

async function request<T>(
  method: string,
  path: string,
  body?: unknown,
): Promise<T> {
  // 요청 전 토큰 유효성 확인 + 자동 갱신
  const token = await ensureValidToken();

  const headers: Record<string, string> = {};
  if (token) headers.Authorization = `Bearer ${token}`;
  if (body !== undefined) headers["Content-Type"] = "application/json";

  const res = await fetch(`${API_BASE}${path}`, {
    method,
    headers,
    body: body !== undefined ? JSON.stringify(body) : undefined,
  });

  // 401 응답 시 한 번 더 갱신 시도
  if (res.status === 401) {
    if (token) {
      const refreshed = await silentRefresh();
      if (refreshed) {
        // 새 토큰으로 재시도
        const retryHeaders: Record<string, string> = {};
        retryHeaders.Authorization = `Bearer ${refreshed}`;
        if (body !== undefined) retryHeaders["Content-Type"] = "application/json";

        const retryRes = await fetch(`${API_BASE}${path}`, {
          method,
          headers: retryHeaders,
          body: body !== undefined ? JSON.stringify(body) : undefined,
        });
        if (!retryRes.ok) throw await parseError(retryRes);
        if (retryRes.status === 204) return undefined as T;
        return (await retryRes.json()) as T;
      }
    }
    // 갱신 실패 또는 토큰이 애초에 없는 경우 → 로그인 페이지로
    clearToken();
    window.dispatchEvent(new CustomEvent("auth:expired"));
    throw await parseError(res);
  }

  if (!res.ok) throw await parseError(res);
  if (res.status === 204) return undefined as T;
  return (await res.json()) as T;
}

/** multipart/form-data 전송(음성 업로드 등). Content-Type 은 브라우저가 설정. */
async function requestForm<T>(path: string, form: FormData): Promise<T> {
  const token = await ensureValidToken();
  const headers: Record<string, string> = {};
  if (token) headers.Authorization = `Bearer ${token}`;
  const res = await fetch(`${API_BASE}${path}`, { method: "POST", headers, body: form });
  if (!res.ok) throw await parseError(res);
  return (await res.json()) as T;
}

export const api = {
  get: <T>(path: string) => request<T>("GET", path),
  post: <T>(path: string, body?: unknown) => request<T>("POST", path, body),
  put: <T>(path: string, body?: unknown) => request<T>("PUT", path, body),
  patch: <T>(path: string, body?: unknown) => request<T>("PATCH", path, body),
  postForm: <T>(path: string, form: FormData) => requestForm<T>(path, form),
};
