# CHYM-AKI 프론트엔드 정리

> 신장내과 · 병리과 · 응급의학과 · 관리자 4개 직군이 사용하는 **의료 통합 의사결정 지원 시스템(CDSS)** 프론트엔드.
> 단순 화면이 아니라 **실제 병원 시스템 수준의 아키텍처**(SOLID · 재사용 컴포넌트 · 전역 상태 · 이벤트 기반 알림 · 백엔드 교체 가능 구조)로 구현.

---

## 1. 기술 스택

| 구분 | 기술 |
|------|------|
| 코어 | React 18, TypeScript, Vite 6 |
| 스타일 | Tailwind CSS v4 (`@theme` 토큰 + `.dark` 다크모드) |
| UI 컴포넌트 | shadcn 스타일 (Radix Dialog/Slot, CVA variant) |
| 라우팅 | React Router 7 (HashRouter — 정적/오프라인 배포 호환) |
| 상태관리 | Zustand (+ persist) |
| 차트 | Recharts |
| 아이콘 | lucide-react |

---

## 2. 폴더 구조 (feature-oriented)

```
src/
├── types/        # 도메인 타입(단일 진실 공급원)
├── mock/         # Mock 데이터(부서 간 참조 일관성)
├── services/     # 서비스 레이어(Facade) = 백엔드 교체 지점
├── store/        # Zustand 전역 상태 (auth/notification/consult/ui)
├── lib/          # 순수 유틸 (cn, format, validation, eventBus, notificationFactory)
├── components/
│   ├── ui/       # 디자인 시스템 원자 (Button/Card/Input/Badge/Dialog)
│   └── common/   # 재사용 조립체 (DataTable/StatCard/PageHeader/Timeline 등)
├── layouts/      # 앱 셸 (Sidebar/TopBar/AppLayout)
├── routes/       # 라우터 + 직군 가드(ProtectedRoute) + 네비 설정
└── features/     # 부서/도메인별 기능 단위
    ├── auth / admin / emergency / nephrology / pathology
    ├── consult        # 협진(신장↔병리 공유)
    └── notification   # 알림 채널/이벤트
```

**의존 방향은 항상 `features → common → ui` 단방향.** `ui`는 어떤 도메인도 import 하지 않아 다른 프로젝트로 이식 가능.

---

## 3. 컴포넌트 설계 (재사용성)

3계층으로 분리:

| 계층 | 위치 | 책임 |
|------|------|------|
| UI 프리미티브 | `components/ui` | 디자인 시스템 원자, 도메인 무지 |
| 공통 컴포넌트 | `components/common` | 도메인 맥락 있는 재사용 조립체 |
| 기능 컴포넌트 | `features/*` | 부서 화면, 위 두 계층을 조립 |

재사용 기법:
- **CVA variant 캡슐화** — `<Button variant="destructive">`처럼 의미만 지정 (색상 하드코딩 제거 → OCP)
- **제네릭 컴포넌트** — `DataTable<T>`가 직원·응급환자·협진 목록을 한 컴포넌트로 재사용 (DRY)
- **합성 패턴** — `Card`/`Dialog` 계열을 작은 조각으로 조합
- **상태→색상 매핑 분리** — `lib/statusTone.ts`가 도메인 상태를 배지 색상으로 일관 매핑

---

## 4. 상태 관리 (Zustand)

전역/지역 경계를 지켜 전역 상태 비대화 방지.

| 스토어 | 보관 상태 | 비고 |
|--------|-----------|------|
| `authStore` | 로그인 사용자(SessionUser) | persist 영속, **비밀번호 미보관**(정보 은닉) |
| `notificationStore` | 알림 채널 버킷 | Severity 시스템의 중심 |
| `consultStore` | 협진 목록·요청·회신 | 신장내과·병리과 **공유**(교차 연동) |
| `uiStore` | 테마(light/dark) | persist 영속 |

> ⚠️ **함정 기록**: Zustand 셀렉터에서 `.filter()`로 새 배열을 반환하면 `useSyncExternalStore`가 무한 루프(React #185)에 빠진다. → 스토어에서는 안정적 참조(`s.items`)만 선택하고 `filter`는 렌더 단계에서 수행.

---

## 5. ⭐ 알림 Severity 시스템 (Factory + Observer)

핵심 설계 포인트. **알림은 `severity` 하나만 가지며, 그 값이 표현 채널을 자동 결정**한다.

### Severity → 채널 매핑 (Factory)

| Severity | 채널(컴포넌트) | 특성 |
|----------|----------------|------|
| `INFO` | **Toast** | 우하단, 4초 자동 소멸 |
| `WARNING` | **Banner** | 화면 상단 고정 |
| `ACTION_REQUIRED` | **Drawer** | 우측 슬라이드 패널, 자동 열림 |
| `CRITICAL` | **Modal** | 닫기 불가, 확인만 |

- `NotificationFactory.create(input)` → id/시간/read 자동 채움
- `NotificationOutlet`의 **채널→컴포넌트 레지스트리**가 severity에 맞는 컴포넌트를 렌더 (새 채널 추가 = 맵 한 줄, OCP)
- 호출부는 `notify({ severity, department, title, message })`만 알면 됨 — 어떤 컴포넌트로 뜨는지 모름(캡슐화)
- 각 채널은 로그인 사용자의 `department`로 필터 → 부서별 알림 분리

### 이벤트 → 알림 (Observer)

도메인 액션은 알림을 직접 만들지 않고 **이벤트만 발행**한다.

```
[발행] 협진 요청 / 병리 결과 등록 / AKI 위험도 변경
        eventBus.publish({ type: "consult.requested", ... })
                 │  (발행자는 구독자를 모름 = 결합도↓)
[구독] notificationStore.startObserver()
        eventBus.subscribe(e => notify(eventToNotification(e)))
                 │  eventToNotification = 이벤트→(severity·부서·문구) 규칙표
[렌더] Factory가 severity로 채널 결정 → 해당 채널 컴포넌트 표시
```

- 실시간 수신은 `notificationService.simulate(dept)`가 부서별 데모 이벤트를 시간차 발행 → **백엔드 연동 시 WebSocket 구독으로 교체**(`eventBus.publish`는 그대로)

### 부서별 알림 적용

| 부서 | Toast (INFO) | Banner (WARNING) | Drawer (ACTION_REQUIRED) | Modal (CRITICAL) |
|------|-------------|------------------|--------------------------|------------------|
| 응급의학과 | 환자 등록 완료 · SOAP 생성 완료 | ICU 병상 부족 | — | AKI 고위험 / 응급 투석 고려 |
| 신장내과 | 병리 결과 도착 | — | 협진 요청 도착 | AKI Stage 3 / Cr 급상승 |
| 병리과 | 판독 저장 완료 | — | 신규 협진 요청 | 긴급 판독 요청 |

---

## 6. 라우팅 & 권한 (RBAC)

- `HashRouter` — `dist`를 `file://` 또는 정적 호스팅에서 열어도 동작
- `ProtectedRoute`가 **직군 가드**: 미로그인 → `/login`, 직군 불일치 → 본인 홈으로 리디렉션
- 화면 코드는 권한을 신경 쓰지 않음(관심사 분리)

---

## 7. 부서 화면 (구역 세분화 — 중요 정보 상단)

### 관리자
KPI(전체/대기/오늘 로그인/시스템) · 직원 계정 승인·거부·권한 변경 테이블

### 응급의학과
병상 KPI(ICU 포함) · 4구역 병상 보드 · 응급환자 위험도순 리스트

### 신장내과 (구역 세분화)
```
[ ① 최상단 전폭: 환자 핵심 요약바 ]  ← 가장 중요
   식별정보 + AKI 단계/위험 배지 + 핵심검사(Cr·eGFR·BUN·K) 칩(상승↑/하강↓ 화살표)
[ ② 좌상단 메인: 검사결과 ] [ 우상단: 추이 차트(Recharts) ]
[ ③ 하단: AI SOAP 초안 | 협진 현황(타임라인+회신) ]
```
- 환자 목록은 좌측 레일(위험도 배지 표시)
- AKI 위험도는 eGFR/Cr/K 기반 휴리스틱으로 단계·색상 자동 판정

### 병리과 (VUNO Med-PathQuant 스타일)
```
[ 좌 레일: 협진 목록 + Annotation/Toolbox + Layers 토글 ]
[ 중앙: ① 판정 요약 스트립(사구체 수·경화 수·경화 비율) ← 상단 중요정보
        ② WSI 뷰어(미니맵 + 수직 줌 슬라이더 + 스케일바 + 레이어 오버레이)
        ③ 병리 보고서(소견/진단/권고 + 임시저장/판독완료/협진회신) ]
[ 우: AI 분석 결과 — Model 선택 + Result(분류 확률 막대) + Area Statistics(구조 밀도 막대) ]
```
- 디지털 병리 기업 **뷰노(VUNO Med-PathQuant™)** 뷰어 UI를 참조

---

## 8. 협진 교차 연동 (Observer 흐름)

```
신장내과: 협진 요청 → eventBus.publish(consult.requested)
                         ↓ (Observer)
병리과:   Drawer "신규 협진 요청" 알림 + 판독 → 회신
                         ↓ eventBus.publish(pathology.resultArrived)
신장내과: Toast "병리 결과 도착" + ConsultStatusCard에 타임라인·회신 표시 (루프 종료)
```

---

## 9. 백엔드 연동 포인트

현재 Mock 기반이지만 **화면 코드를 건드리지 않고** 백엔드로 교체 가능하도록 `services/`를 단일 경계로 둠.

| 서비스 | 현재(Mock) | 교체 후 |
|--------|-----------|---------|
| `authService` | in-memory 계정 | `POST /api/auth/*` (JWT) |
| `patientService.generateDraft` | 규칙 기반 더미 SOAP | `POST /api/ai/soap` (LLM) |
| `consultService` | 객체 생성 | `POST/PATCH /api/consults` |
| `notificationService.simulate` | setTimeout 발행 | **WebSocket/SSE 구독** |

- 서비스 시그니처는 그대로 두고 내부만 교체 → store·컴포넌트 무수정

---

## 10. 데모 계정

| 직군 | 아이디 | 비밀번호 |
|------|--------|----------|
| 관리자 | `admin` | `Admin2026!` |
| 응급의학과 | `er_kim` | `Emer2026!` |
| 신장내과 | `neph_hong` | `Neph2026!` |
| 병리과 | `path_lee` | `Path2026!` |

> 비밀번호는 프론트 데모 한계상 평문 Mock. 실제 서비스는 백엔드 해시 필수.

---

## 11. 면접 예상 질문 답변

- **컴포넌트를 어떻게 분리했나요?** → 3계층(ui/common/feature) 단방향 의존 + SRP
- **재사용성은?** → CVA variant, 제네릭 `DataTable`, 합성 패턴, 상태→색상 매핑 분리
- **상태 관리는?** → 전역/지역 경계 원칙 + Zustand 셀렉터 구독 + persist
- **알림 시스템은?** → severity 1축 모델 + **Factory**(severity→채널 캡슐화) + **Observer**(이벤트→알림 자동 생성)로 발행부와 표현부 분리
- **왜 이런 구조인가요?** → 디자인 시스템을 도메인에서 떼어내 이식성 확보, 알림 표현 정책을 Factory/규칙표에 모아 변경 영향 범위 최소화

---

## 12. 실행

```bash
npm install
npm run dev        # 개발 서버 (http://localhost:5174)
npm run build      # 타입체크 + 프로덕션 빌드 → dist/ (정적/오프라인 배포 가능)
```
