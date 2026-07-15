# ICU/ER → 신장내과 협진: Consult 엔티티 기반 워크플로우 전환

## 1. 개요

기존 ICU/응급실 → 신장내과 협진은 **1회성 알림(`aki.emergencyConsult` 이벤트 → CRITICAL 모달)** 으로
끝나, 요청 이력·상태 추적·회신이 불가능했다. 이를 **병리과 협진과 동일한 `Consult` 엔티티 기반
워크플로우**로 전환했다.

```
변경 전:  BedDetailModal → aki.emergencyConsult 발행 → 신장내과 CRITICAL 모달 → 종료
변경 후:  BedDetailModal → Consult 레코드 생성 → 신장내과 워크리스트 등록 → 상태 추적 → 회신
                         └→ aki.emergencyConsult 발행 → CRITICAL 모달 (기존 유지)
```

## 2. 설계 원칙 — 기존 구조 최대 재사용

| 재사용 자산 | 방식 |
| --- | --- |
| `Consult` 엔티티 | `kind` 판별자(`"pathology"` \| `"nephrology"`)만 추가, 나머지 필드 공유 |
| `consultStore` (zustand) | `request`/`reply` 그대로 사용, `accept` 액션만 추가 |
| `consultService` | 공유 모듈 store 그대로, 부서 인식형 타임라인으로 일반화 |
| `ConsultStatus` | 동일 enum 공유. 신장 협진은 `requested→in_progress→replied` 3단계만 사용 |
| `ConsultReply` | 소견/진단/권고 스키마 그대로 재사용 (라벨만 신장내과 맥락으로 표기) |
| `Timeline` / `InfoCard` / `StatusBadge` / `consultStatusTone` | 공용 컴포넌트·톤 매핑 그대로 |

### 상태 매핑 (요구사항 ↔ 구현 enum)

| 요구사항 | 구현(enum) | 라벨 | 전이 시점 |
| --- | --- | --- | --- |
| PENDING | `requested` | 대기중 | 응급의학과 협진 요청 시 |
| IN_PROGRESS | `in_progress` | 진행중 | 신장내과 "협진 접수" 시 (`accept`) |
| COMPLETED | `replied` | 회신완료 | 신장내과 "협진 회신" 전송 시 (`reply`) |

> 병리 협진의 `read`(판독완료) 단계는 신장 협진에서 미사용.

## 3. 변경된 파일 목록

### 신규 (2)
| 파일 | 역할 |
| --- | --- |
| `src/features/consult/EmergencyConsultInbox.tsx` | 신장내과 측 응급 협진 워크리스트(목록·상태·접수·회신 진입) |
| `src/features/consult/ConsultReplyModal.tsx` | 신장내과 회신 작성 모달(소견/판단/권고 → 회신) |

### 수정 (8)
| 파일 | 변경 내용 |
| --- | --- |
| `src/types/consult.ts` | `ConsultKind`·`CONSULT_KIND_LABEL` 추가, `Consult`에 `kind`·`bedLabel?` 필드 추가, 상태 매핑 주석 |
| `src/mock/consults.ts` | 기존 3건에 `kind:"pathology"` 부여, 신장 협진(ICU/ER) 시드 2건 추가 |
| `src/services/consultService.ts` | 입력에 `kind`·`bedLabel` 추가, 부서 인식형 타임라인, `accept()`(in_progress 전이) 추가, `reply()` 부서 일반화 |
| `src/store/consultStore.ts` | `accept(consultId, actor)` 액션 추가 |
| `src/features/notification/events.ts` | `consult.nephReplied` 이벤트 추가(→응급의학과 알림), `consult.arrivedNeph`에 `consultId` 딥링크 추가 |
| `src/features/emergency/EmergencyDashboard.tsx` | `requestConsult`에서 Consult 레코드 생성 추가(CRITICAL 모달 이벤트는 유지) |
| `src/features/nephrology/NephrologyWorkspace.tsx` | `EmergencyConsultInbox` 전폭 배치, 서브타이틀에 "응급 협진" 추가 |
| `src/features/consult/ConsultStatusCard.tsx` | 병리 협진만 표시하도록 `kind==="pathology"` 필터 |
| `src/features/pathology/PathologyWorkspace.tsx` | 같은 store 공유 격리 — `kind==="pathology"` 필터 |

## 4. 데이터 흐름

### (A) 요청 — 응급의학과 → 신장내과
```
EmergencyDashboard.requestConsult(bed, detail)
 ├─ consultStore.request({ kind:"nephrology", patientMrn:bed.label, bedLabel,
 │                         diagnosis, keyLabs, reason, urgency:"emergency", requestedBy })
 │    └─ consultService.request() → 공유 store 에 Consult(status:"requested") 적재
 │         → zustand items 갱신 (신장내과 인박스에 즉시 등장)
 ├─ eventBus.publish("aki.emergencyConsult")  ← 기존 CRITICAL 모달 (유지)
 │    └─ eventToNotification → severity:CRITICAL, dept:nephrology → 신장내과 Modal
 ├─ eventBus.publish("consult.arrivedNeph", { consultId })  ← 워크리스트 도착 알림
 │    └─ eventToNotification → severity:ACTION_REQUIRED, dept:nephrology
 │         link:/nephrology?consult=<id> → 클릭 시 인박스 해당 건 자동 선택
 └─ notify(INFO)  ← 요청자(응급의학과) 본인 확인 토스트
```

### (B) 접수 — 신장내과 (PENDING → IN_PROGRESS)
```
EmergencyConsultInbox "협진 접수" 클릭
 └─ consultStore.accept(id, userName)
      └─ consultService.accept() → status:"in_progress", timeline += "신장내과 접수"
```

### (C) 회신 — 신장내과 → 응급의학과 (IN_PROGRESS → COMPLETED)
```
ConsultReplyModal.submit()
 ├─ consultStore.reply(id, { findings, diagnosis, recommendation, author, repliedAt })
 │    └─ consultService.reply() → status:"replied", reply 등록, timeline += "협진 회신"
 ├─ eventBus.publish("consult.nephReplied")
 │    └─ eventToNotification → severity:ACTION_REQUIRED, dept:emergency → 응급의학과 알림
 └─ notify(INFO)  ← 회신자(신장내과) 본인 확인 토스트
```

### 교차 연동 핵심
`consultService` 의 모듈 레벨 `store` 를 응급/신장/병리 세 화면이 공유한다. 따라서 응급의학과에서
생성한 Consult 가 신장내과 인박스에 즉시 나타나고, 신장내과 회신이 같은 레코드에 반영된다.
`kind` 필터로 병리 협진 화면(PathologyWorkspace / ConsultStatusCard)과 격리된다.

## 5. 테스트 시나리오

### 사전 계정 (mock)
- 응급의학과: `er_kim` / `Emer2026!`
- 신장내과: `neph_hong` / `Neph2026!`
- 병리과: `path_lee` / `Path2026!`

### TC-1 응급 협진 요청 → Consult 생성 + CRITICAL 모달 동시 (요구 #6)
1. `er_kim` 로그인 → 병상 보드에서 AKI 위험(빨강) 병상 `ER-8` 클릭.
2. BedDetailModal 우측 "신장내과 응급 협진 요청" 클릭.
3. **기대:** 응급의학과 INFO 토스트("신장내과 협진 요청 전송") + 신장내과 측 CRITICAL 모달
   ("응급 협진 요청 · 노발열 (ER-08)") + 신장내과 ACTION_REQUIRED 도착 알림("협진 요청 도착") 발생.
4. `neph_hong` 로그인 → 상단 "응급 협진 요청 (ICU/ER)" 인박스에 **노발열 ER-08 / 대기중** 신규 등장.
   → ✅ 검증 완료 (대기 1·진행 1·완료 1)

### TC-2 상태 추적: PENDING → IN_PROGRESS (요구 #3)
1. `neph_hong` 인박스에서 대기중 건 선택 → "협진 접수" 클릭.
2. **기대:** 상태 배지 `대기중 → 진행중`, 카운트 `대기 -1 / 진행 +1`, 타임라인에 "신장내과 접수" 추가.
   → ✅ 검증 완료 (대기 0·진행 2·완료 0)

### TC-3 협진 회신: IN_PROGRESS → COMPLETED (요구 #5)
1. 진행중 건 선택 → "협진 회신" → 모달에서 평가 소견/임상 판단/처치 권고 입력 → "응급의학과로 회신".
2. **기대:** 상태 `진행중 → 회신완료`, 상세에 "신장내과 회신" 블록(소견/판단/권고) 노출,
   응급의학과로 `consult.nephReplied` 알림(ACTION_REQUIRED) 발행.
   → ✅ 검증 완료 (대기 0·진행 1·완료 1, 회신 블록 표시)
3. 검증: 5자 미만 소견·빈 필드 시 "응급의학과로 회신" 버튼 비활성.

### TC-4 워크리스트 표시 (요구 #4)
1. `neph_hong` 인박스에 `kind==="nephrology"` 협진만 표시되는지 확인(병리 협진 미혼입).
2. 상단 카운트(대기/진행/완료)와 목록 상태 배지 일치 확인.
   → ✅ 검증 완료

### TC-5 병리 협진 회귀 — 격리 확인 (요구 #1)
1. `neph_hong` 환자 선택 → "병리 협진 요청" → 병리과로 전송(기존 흐름 정상).
2. `path_lee` 로그인 → 병리과 협진 목록에 **신장 협진(ICU/ER)이 섞이지 않음** 확인.
3. 신장내과 하단 "병리 협진 현황" 카드에 신장 협진이 섞이지 않음 확인.

### TC-6 알림 딥링크
1. 응급의학과로 발행된 `consult.nephReplied` 알림 클릭 → `/emergency` 이동.
2. 신장 협진 도착 알림(`consult.arrivedNeph`, consultId 포함) 클릭 → `/nephrology?consult=<id>` →
   인박스에서 해당 건 자동 선택. → ✅ 검증 완료 (노발열 ER-08 자동 선택)

### 회귀(자동)
- `npm run typecheck` (tsc --noEmit) 통과. 콘솔 런타임 에러 0건.
