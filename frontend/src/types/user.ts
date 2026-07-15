/** 사용자/계정 도메인 타입. */

/** 직군. 접근 가능한 화면·권한이 직군에 따라 달라진다. */
export type Role = "admin" | "nephrology" | "pathology";

/** 계정 승인 상태 — 관리자 화면에서 승인/거부. */
export type ApprovalStatus = "pending" | "approved" | "rejected";

export const ROLE_LABEL: Record<Role, string> = {
  admin: "관리자",
  nephrology: "신장내과",
  pathology: "병리과",
};

export const APPROVAL_LABEL: Record<ApprovalStatus, string> = {
  pending: "승인 대기",
  approved: "승인됨",
  rejected: "거부됨",
};

/**
 * 계정 엔티티(저장소/서비스 내부 표현).
 * password 는 프론트 데모 한계상 평문 — 실제 서비스는 백엔드 해시가 필수다.
 */
export interface Account {
  readonly id: string;
  readonly username: string;
  readonly password?: string;
  readonly name: string;
  readonly role: Role;
  readonly department: string;
  readonly approval: ApprovalStatus;
  readonly createdAt: string;
  readonly lastLoginAt: string | null;
  readonly signaturePath?: string | null;
}

/** UI 로 노출되는 세션 사용자(비밀번호 제외 — 정보 은닉). */
export interface SessionUser {
  readonly id: string;
  readonly username: string;
  readonly name: string;
  readonly role: Role;
  readonly department: string;
  readonly signaturePath?: string | null;
}

export function toSessionUser(a: Account): SessionUser {
  return { 
    id: a.id, 
    username: a.username, 
    name: a.name, 
    role: a.role, 
    department: a.department,
    signaturePath: a.signaturePath 
  };
}
