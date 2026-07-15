/** 순수 검증 함수 — 폼 유효성. 빈 문자열이면 통과(에러 미표시), 형식 위반 시 메시지 반환. */

export function validateUsername(v: string): string | null {
  if (!v) return null;
  if (v.length < 4) return "아이디는 4자 이상이어야 합니다.";
  if (!/^[a-zA-Z0-9_]+$/.test(v)) return "영문/숫자/밑줄만 사용할 수 있습니다.";
  return null;
}

export function validatePassword(v: string): string | null {
  if (!v) return null;
  if (v.length < 8) return "비밀번호는 8자 이상이어야 합니다.";
  if (!/[A-Za-z]/.test(v) || !/[0-9]/.test(v)) return "영문과 숫자를 모두 포함해야 합니다.";
  return null;
}

export function validateName(v: string): string | null {
  if (!v) return null;
  if (v.length < 2) return "이름은 2자 이상이어야 합니다.";
  return null;
}

export function validateConfirm(pw: string, confirm: string): string | null {
  if (!confirm) return null;
  if (pw !== confirm) return "비밀번호가 일치하지 않습니다.";
  return null;
}
