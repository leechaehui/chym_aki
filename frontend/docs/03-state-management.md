# 상태 관리 구조 설명

## 전역 vs 지역 상태 분리 원칙

- **전역(Zustand)**: 여러 화면이 공유하거나 새로고침 후에도 유지돼야 하는 상태.
- **지역(useState)**: 한 컴포넌트 안에서만 의미 있는 UI 상태(모달 열림, 탭, 검색어, 정렬, 폼 입력).

이 경계를 지켜 전역 상태의 비대화를 막았다.

## 전역 스토어 (`src/store`)

| 스토어 | 보관 상태 | 비고 |
|--------|-----------|------|
| `authStore` | 로그인 사용자(SessionUser) | `persist` 미들웨어로 localStorage 영속. **비밀번호는 미보관**(정보 은닉) |
| `notificationStore` | 알림 목록·토스트·배너·모달 큐·센터 열림 | 알림 정책 5단계 라우팅의 중심 |
| `consultStore` | 협진 목록·선택·요청/회신 | 신장내과·병리과가 **공유**(교차 연동) |
| `uiStore` | 테마(light/dark) | `persist` 영속, `applyTheme` 로 `<html>.dark` 동기화 |

### 왜 Zustand인가
- Redux 대비 보일러플레이트가 적고, Context 대비 불필요한 리렌더가 적다(셀렉터 구독).
- 컴포넌트는 필요한 조각만 구독한다. **주의**: 셀렉터가 매번 새 배열/객체를 반환하면 `useSyncExternalStore` 가 무한 루프(React #185)에 빠지므로, 스토어에서는 안정적 참조(`s.items`)만 선택하고 `filter`/`map` 은 렌더에서 수행한다.

## 알림 Severity 시스템 (Factory + Observer)

### Severity → 채널 (NotificationFactory)
알림은 `severity` 하나를 가지며, 그 값이 표현 채널을 결정한다. 매핑은 `lib/notificationFactory.ts` 의 `SEVERITY_META` 한 곳에 있다.

| Severity | 채널(컴포넌트) | 특성 | 예시 |
|----------|----------------|------|------|
| `INFO` | **Toast** (`ToastChannel`) | 우하단, 4초 자동소멸 | 판독 저장 완료, 병리 결과 도착 |
| `WARNING` | **Banner** (`BannerChannel`) | 상단 고정 | ICU 병상 부족 |
| `ACTION_REQUIRED` | **Drawer** (`DrawerChannel`) | 우측 슬라이드 패널, 자동 열림 | 신규/협진 요청 도착 |
| `CRITICAL` | **Modal** (`ModalChannel`) | 닫기 불가, 확인만 | AKI 고위험, 긴급 판독 요청 |

- **Factory(생성)**: `NotificationFactory.create(input)` 가 id·시간·read 를 채워 알림 객체를 만든다.
- **Factory(렌더링)**: `NotificationOutlet` 의 `채널→컴포넌트 레지스트리`가 severity 의 채널에 해당하는 컴포넌트를 렌더한다. 새 채널 추가 = 맵에 한 줄(OCP).
- 호출부는 `notify({ severity, department, title, message })` 만 알면 되고, 어떤 컴포넌트로 뜨는지는 모른다(캡슐화).
- 각 채널은 로그인 사용자의 `department` 로 필터링한다(부서별 알림 분리).

### 이벤트 → 알림 (Observer)
도메인 액션은 알림을 직접 만들지 않고 **이벤트만 발행**한다(`lib/eventBus.ts`, `features/notification/events.ts`).

```
[발행] 협진 요청/병리 결과 등록/AKI 위험도 변경 등
        eventBus.publish({ type: "consult.requested", ... })
                         │  (발행자는 구독자를 모름 = 결합도↓)
[구독] notificationStore.startObserver():
        eventBus.subscribe(e => notify(eventToNotification(e)))
                         │  eventToNotification = 이벤트→(severity·부서·문구) 규칙표
[렌더] NotificationFactory 가 severity 로 채널 결정 → 해당 채널 컴포넌트가 표시
```

- `eventToNotification` 규칙표가 "무슨 일(이벤트)"을 "어떤 알림(severity)"으로 바꿀지 한 곳에서 관리한다.
- 실시간 수신은 `notificationService.simulate(dept)`(부서별 데모 이벤트를 시간차 발행)가 대신하며, 백엔드 연동 시 WebSocket 구독으로 교체한다(`eventBus.publish` 는 그대로).

### 부서별 적용(요약)
- **응급의학과**: INFO 환자 등록 완료·SOAP 생성 완료 / WARNING ICU 병상 부족 / CRITICAL AKI 고위험
- **신장내과**: INFO 병리 결과 도착 / ACTION_REQUIRED 협진 요청 도착 / CRITICAL AKI Stage 3·Cr 급상승
- **병리과**: INFO 판독 저장 완료 / ACTION_REQUIRED 신규 협진 요청 / CRITICAL 긴급 판독 요청

## 교차 연동 흐름 (협진)

```
신장내과: consultStore.request()  →  eventBus.publish(consult.requested)
                                        ↓ (Observer)
병리과:   notify(ACTION_REQUIRED) → Drawer "신규 협진 요청" + consultStore.reply()
                                        ↓
          eventBus.publish(pathology.resultArrived)  →  notify(INFO, nephrology)
                                        ↓
신장내과: Toast "병리 결과 도착" + ConsultStatusCard 타임라인·회신 (루프 종료)
```

## 면접 답변 요약
- *"상태 관리는 어떻게 했나요?"* → 전역/지역 경계 원칙 + Zustand 셀렉터 구독 + persist 영속.
- *"알림은 어떻게 설계했나요?"* → severity 1축 모델 + **Factory**(severity→채널 캡슐화) + **Observer**(이벤트→알림 자동 생성)로 발행부와 표현부를 분리.
- *"왜 이렇게 했나요?"* → 화면/도메인 코드는 "무슨 일이 일어났는지"만 알리고, 알림의 표현 정책은 Factory/규칙표에 모아 변경 영향 범위를 좁혔다.
