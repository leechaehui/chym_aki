import type { Account, ApprovalStatus, Role, SessionUser } from "@/types";
import { api, clearToken, setToken, saveCredentials } from "./http";

/** 가입 입력. */
export interface SignUpInput {
  username: string;
  password: string;
  name: string;
  role: Role;
  department: string;
}

/** 로그인 응답(백엔드). */
interface TokenResponse {
  accessToken: string;
  tokenType: string;
  user: SessionUser;
}

/** 관리자 계정 응답(비밀번호 미포함). */
export interface AccountResponse {
  id: string;
  username: string;
  name: string;
  role: Role;
  department: string;
  approval: ApprovalStatus;
  createdAt: string | null;
  lastLoginAt: string | null;
  rejectionReason: string | null;
  approvedBy: string | null;
  approvedAt: string | null;
  rejectedBy: string | null;
  rejectedAt: string | null;
}

export interface ApprovalHistoryOut {
  id: number;
  userId: string;
  adminId: string | null;
  actorType: string;
  oldStatus: string | null;
  newStatus: string;
  reason: string | null;
  createdAt: string;
}

/**
 * 인증 서비스 (Facade). 로그인/가입/계정관리 — FastAPI 백엔드 연동.
 * 로그인 성공 시 JWT 토큰을 http 클라이언트에 저장한다(이후 요청 자동 인증).
 */
class AuthService {
  /** 로그인 — 토큰 저장 후 세션 사용자 반환. 실패 시 백엔드 메시지로 reject. */
  async login(username: string, password: string): Promise<SessionUser> {
    const res = await api.post<TokenResponse>("/auth/login", { username, password });
    setToken(res.accessToken);
    saveCredentials(username, password);
    return res.user;
  }

  /** 가입 — 승인 대기 상태로 계정 생성. */
  async signUp(input: SignUpInput): Promise<void> {
    await api.post<{ message: string }>("/auth/signup", input);
  }

  /** 로그아웃 — 저장된 토큰 폐기. */
  logout(): void {
    clearToken();
  }

  /** 전체 계정 조회(관리자). 백엔드는 비밀번호를 노출하지 않으므로 빈 문자열로 채운다. */
  async listAccounts(): Promise<(Account & { rejectionReason?: string | null })[]> {
    const rows = await api.get<AccountResponse[]>("/auth/accounts");
    return rows.map((r) => ({
      id: r.id,
      username: r.username,
      password: "",
      name: r.name,
      role: r.role,
      department: r.department,
      approval: r.approval,
      createdAt: r.createdAt ?? "",
      lastLoginAt: r.lastLoginAt,
      rejectionReason: r.rejectionReason,
    }));
  }

  /** 계정 승인/거부(관리자). */
  async setApproval(userId: string, approval: ApprovalStatus, rejectionReason?: string): Promise<void> {
    await api.patch<AccountResponse>(`/auth/accounts/${userId}/approval`, { approval, rejectionReason });
  }

  /** 계정 승인/거부 이력 조회(관리자). */
  async getApprovalHistory(userId: string): Promise<ApprovalHistoryOut[]> {
    return await api.get<ApprovalHistoryOut[]>(`/auth/accounts/${userId}/approval_history`);
  }

  /** 서명 업데이트. */
  async updateSignature(signatureBase64: string): Promise<SessionUser> {
    return await api.post<SessionUser>("/auth/me/signature", { signature_base64: signatureBase64 });
  }
}

export const authService = new AuthService();
