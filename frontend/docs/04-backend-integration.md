# 백엔드 연동 포인트 설명

현재는 **Mock 기반**이지만, 실제 백엔드로 교체할 때 **화면 코드를 건드리지 않도록**
서비스 레이어(`src/services`)를 단일 경계로 두었다. 모든 데이터 접근은 이 레이어를 통과한다.

## 교체 지점 = `src/services/*`

| 서비스 | 현재(Mock) | 교체 후(예시) |
|--------|-----------|----------------|
| `authService.login()` | in-memory 계정 비교 | `POST /api/auth/login` → JWT |
| `authService.signUp()` | 배열 push | `POST /api/auth/signup` |
| `patientService.list()` | mock 배열 | `GET /api/patients` |
| `patientService.generateDraft()` | 규칙 기반 더미 SOAP | `POST /api/ai/soap`(LLM) |
| `bedService.listBeds()` | mock 배열 | `GET /api/beds` |
| `consultService.request()/reply()` | 객체 생성 | `POST /api/consults`, `PATCH /api/consults/:id` |
| `pathologyService.list()` | mock 결과 | `GET /api/pathology/results` |
| `notificationService.subscribe()` | `setTimeout` 푸시 | **WebSocket/SSE** 구독 |

서비스 메서드 **시그니처는 그대로 두고 본문만 교체**하면 store·컴포넌트는 무수정이다.
`services/http.ts` 의 `delay()` 를 실제 `fetch` 래퍼로 바꾸는 것이 시작점이다.

## 실시간 알림 (WebSocket 전환)

`notificationService.subscribe(onPush)` 는 해제 함수를 반환하는 형태로 이미 설계됐다.

```ts
// 현재(Mock)
subscribe(onPush) { const t = setTimeout(...); return () => clearTimeout(t); }

// 전환 후
subscribe(onPush) {
  const ws = new WebSocket(import.meta.env.VITE_WS_URL);
  ws.onmessage = (e) => onPush(JSON.parse(e.data));
  return () => ws.close();
}
```

`AppLayout` 이 마운트 시 구독하고 언마운트 시 해제하므로, 전환 시 레이아웃 코드도 무수정이다.

## 인증/보안 시 해야 할 일 (프론트 데모 한계)

- 현재 비밀번호는 **평문 Mock** — 실제 서비스는 백엔드 해시(bcrypt/argon2) + HTTPS 필수.
- `authStore` 는 SessionUser(비밀번호 없음)만 보관. 토큰 도입 시 httpOnly 쿠키 권장(localStorage 토큰은 XSS 노출).
- 라우트 가드(`ProtectedRoute`)는 UX 용. **실제 권한 검증은 서버에서** 재확인해야 한다.

## WSI 뷰어

`WSIViewer` 는 합성 마커 데모. 실제 기가픽셀 WSI(.svs)는 브라우저가 직접 못 여므로
**타일 서버(OpenSlide/Deep Zoom)** + OpenSeadragon 등으로 이 컴포넌트 내부만 교체한다.

## 환경 변수
`VITE_API_URL`, `VITE_WS_URL` 등을 `.env` 로 주입(현재는 미사용, 교체 시 도입).
