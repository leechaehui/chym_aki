import { useState } from "react";
import kidneyIcon from "@/assets/kidney_icon.png";
import { useNavigate } from "react-router-dom";
import { Loader2 } from "lucide-react";
import type { Role } from "@/types";
import { ROLE_LABEL } from "@/types";
import { useAuthStore } from "@/store/authStore";
import { useNotificationStore } from "@/store/notificationStore";
import { authService } from "@/services/authService";
import { ApiError } from "@/services/http";
import { homePathFor } from "@/routes/navConfig";
import { validateUsername, validatePassword, validateName, validateConfirm } from "@/lib/validation";
import { Button } from "@/components/ui/button";
import { Input, Label, Select } from "@/components/ui/input";
import { cn } from "@/lib/cn";
import { Modal } from "@/components/common/Modal";
import { SignaturePad } from "@/components/ui/SignaturePad";

type Mode = "login" | "signup";

/** 필드 + 인라인 검증 메시지. */
function Field({ label, error, ok, children }: { label: string; error?: string | null; ok?: boolean; children: React.ReactNode }) {
  return (
    <div>
      <Label>{label}</Label>
      {children}
      {error && <p className="mt-1 text-[11px] text-destructive">{error}</p>}
      {!error && ok && <p className="mt-1 text-[11px] text-success">사용 가능</p>}
    </div>
  );
}

/**
 * 인증 화면 — 로그인 / 가입 토글. 실제 프론트 유효성 검증으로 동작하며
 * 데모 자동채우기 버튼은 두지 않는다(실서비스 흐름 재현).
 */
export function LoginPage() {
  const navigate = useNavigate();
  const login = useAuthStore((s) => s.login);
  const loading = useAuthStore((s) => s.loading);
  const error = useAuthStore((s) => s.error);
  const notify = useNotificationStore((s) => s.notify);

  const [mode, setMode] = useState<Mode>("login");
  const [signupDone, setSignupDone] = useState<string | null>(null);
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  
  const [regUsername, setRegUsername] = useState("");
  const [regPassword, setRegPassword] = useState("");
  const [name, setName] = useState("");
  const [confirm, setConfirm] = useState("");
  const [role, setRole] = useState<Role | "other">("nephrology");
  const [department, setDepartment] = useState(ROLE_LABEL["nephrology"]);
  const [submitting, setSubmitting] = useState(false);
  const [signupError, setSignupError] = useState<string | null>(null);
  const [signatureBase64, setSignatureBase64] = useState<string | null>(null);

  const [rejectModalOpen, setRejectModalOpen] = useState(false);
  const [rejectReason, setRejectReason] = useState("");

  const uErr = validateUsername(username);
  const pErr = validatePassword(password);
  
  const regUErr = validateUsername(regUsername);
  const regPErr = validatePassword(regPassword);
  const nErr = validateName(name);
  const cErr = validateConfirm(regPassword, confirm);

  const loginValid = username.length >= 4 && password.length >= 8 && !uErr && !pErr;
  const signupValid = regUsername.length >= 4 && regPassword.length >= 8 && !regUErr && !regPErr && !nErr && !cErr && name.length >= 2 && confirm.length > 0 && department.length > 0 && signatureBase64 !== null;

  async function handleLogin(e: React.FormEvent) {
    e.preventDefault();
    try {
      const ok = await login(username, password);
      if (ok) {
        const user = useAuthStore.getState().user!;
        notify({ severity: "INFO", department: user.role, title: "로그인", message: `${user.name}님 환영합니다.` });
        navigate(homePathFor(user.role));
      }
    } catch (err) {
      if (err instanceof ApiError) {
        if (err.code === "ACCOUNT_REJECTED") {
          setRejectReason(err.reason || "사유가 등록되지 않았습니다.");
          setRejectModalOpen(true);
        } else if (err.code === "ACCOUNT_PENDING") {
          // You could also show a modal or just leave the text in the error state
        }
      }
    }
  }

  async function handleSignup(e: React.FormEvent) {
    e.preventDefault();
    setSignupError(null);
    setSignupDone(null);
    setSubmitting(true);
    try {
      const submitRole: Role = role === "other" ? "nephrology" : role;
      await authService.signUp({ username: regUsername, password: regPassword, name, role: submitRole, department, signatureBase64: signatureBase64! } as any);
      setSignupDone("가입 요청이 접수되었습니다. 관리자 승인 후 로그인하세요.");
      setMode("login");
      setRegPassword("");
      setConfirm("");
    } catch (err) {
      setSignupError((err as Error).message);
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div className="flex h-full items-center justify-center bg-muted p-4">
      <div className="w-full max-w-sm">
        <div className="mb-6 flex flex-col items-center gap-2">
          <img src={kidneyIcon} width="56" height="56" alt="RENAI" />
          <h1 className="text-xl font-bold text-foreground">RENAI</h1>
          <p className="text-xs text-muted-foreground">신장 질환 AI 임상 지원 시스템</p>
        </div>

        <div className="rounded-xl border border-border bg-card p-5 shadow-sm">
          <div className="mb-4 flex rounded-lg bg-secondary p-1">
            {(["login", "signup"] as Mode[]).map((m) => (
              <button
                key={m}
                onClick={() => setMode(m)}
                className={cn(
                  "flex-1 rounded-md py-1.5 text-xs font-semibold transition-colors",
                  mode === m ? "bg-card text-foreground shadow-sm" : "text-muted-foreground",
                )}
              >
                {m === "login" ? "로그인" : "회원가입"}
              </button>
            ))}
          </div>

          {mode === "login" ? (
            <form onSubmit={handleLogin} className="flex flex-col gap-3">
              <Field label="아이디" error={uErr}>
                <Input value={username} onChange={(e) => setUsername(e.target.value.replace(/\s/g, ""))} placeholder="아이디" autoComplete="username" />
              </Field>
              <Field label="비밀번호" error={pErr}>
                <Input
                  type="password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  placeholder="비밀번호"
                  autoComplete="current-password"
                />
              </Field>
              {signupDone && <p className="rounded-md bg-success/10 px-3 py-2 text-[11px] text-success">{signupDone}</p>}
              {error && <p className="rounded-md bg-destructive/10 px-3 py-2 text-[11px] text-destructive">
                {error === "ACCOUNT_PENDING" ? "가입 승인 대기 중입니다." : error}
              </p>}
              <Button type="submit" disabled={!loginValid || loading} className="mt-1 w-full">
                {loading && <Loader2 className="size-4 animate-spin" />}
                로그인
              </Button>
            </form>
          ) : (
            <form onSubmit={handleSignup} className="flex flex-col gap-3">
              <Field label="이름" error={nErr} ok={!!name}>
                <Input value={name} onChange={(e) => setName(e.target.value)} placeholder="실명" />
              </Field>
              <Field label="아이디" error={regUErr} ok={!!regUsername}>
                <Input value={regUsername} onChange={(e) => setRegUsername(e.target.value.replace(/\s/g, ""))} placeholder="영문/숫자 4자 이상" />
              </Field>
              <Field label="비밀번호" error={regPErr} ok={!!regPassword}>
                <Input type="password" value={regPassword} onChange={(e) => setRegPassword(e.target.value)} placeholder="영문+숫자 8자 이상" />
              </Field>
              <Field label="비밀번호 확인" error={cErr} ok={!!confirm && !cErr}>
                <Input type="password" value={confirm} onChange={(e) => setConfirm(e.target.value)} placeholder="비밀번호 재입력" />
              </Field>
              <div className="grid grid-cols-2 gap-2">
                <Field label="직군">
                  <Select 
                    value={role} 
                    onChange={(e) => {
                      const newRole = e.target.value as Role | "other";
                      setRole(newRole);
                      if (newRole !== "other") {
                        setDepartment(ROLE_LABEL[newRole]);
                      } else {
                        setDepartment("");
                      }
                    }}
                  >
                    {(Object.keys(ROLE_LABEL) as Role[]).map((r) => (
                      <option key={r} value={r}>
                        {ROLE_LABEL[r]}
                      </option>
                    ))}
                    <option value="other">기타 (직접 입력)</option>
                  </Select>
                </Field>
                <Field label={role === "other" ? "소속 (직접 입력)" : "소속"}>
                  <Input
                    value={department}
                    onChange={(e) => setDepartment(e.target.value)}
                    placeholder={role === "other" ? "직군/소속을 직접 입력" : "예: 신장내과"}
                  />
                </Field>
              </div>
              {role === "other" && (
                <p className="-mt-1 text-[10px] text-muted-foreground">직군은 관리자 승인 시 배정됩니다. 소속을 직접 입력하세요.</p>
              )}
              
              <Field label="서명 (필수)" error={!signatureBase64 && submitting ? "서명을 입력해주세요" : null}>
                <SignaturePad onSign={setSignatureBase64} />
              </Field>

              {signupError && (
                <p className="rounded-md bg-destructive/10 px-3 py-2 text-[11px] text-destructive">{signupError}</p>
              )}
              <Button type="submit" disabled={!signupValid || submitting} className="mt-1 w-full">
                {submitting && <Loader2 className="size-4 animate-spin" />}
                가입 요청
              </Button>
            </form>
          )}
        </div>
        <p className="mt-4 text-center text-[10px] text-muted-foreground">
          데모 — 가입 요청은 관리자 승인 후 로그인 가능합니다.
        </p>
      </div>
      
      <Modal open={rejectModalOpen} onOpenChange={setRejectModalOpen} title="가입 승인 거부">
        <div className="flex flex-col items-center gap-4 p-6 text-center">
          <div className="rounded-lg bg-red-50 p-4 text-sm font-medium text-red-800 w-full text-left whitespace-pre-wrap">
            {rejectReason}
          </div>
          <p className="text-sm text-slate-600">
            재신청이 필요할 경우 관리자에게 문의하시기 바랍니다.
          </p>
          <Button onClick={() => setRejectModalOpen(false)} className="w-full mt-2">확인</Button>
        </div>
      </Modal>
    </div>
  );
}
