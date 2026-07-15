# 컴포넌트 구조 · 재사용성 설명

## 3계층 컴포넌트 분리

| 계층 | 위치 | 책임 | 예시 |
|------|------|------|------|
| **UI 프리미티브** | `components/ui` | 디자인 시스템 원자. 도메인 무지(無知). | Button, Card, Input, Badge, Dialog |
| **공통 컴포넌트** | `components/common` | 도메인 맥락이 있는 재사용 조립체. | DataTable, StatCard, PageHeader, StatusBadge, Timeline, NotificationCenter |
| **기능 컴포넌트** | `features/*` | 특정 부서 화면. 위 두 계층을 조립. | NephrologyWorkspace, PathologyWorkspace |

의존 방향은 항상 **features → common → ui** (단방향). ui 는 어떤 도메인도 import 하지 않으므로 다른 프로젝트로 그대로 이식 가능하다.

## 재사용성을 어떻게 확보했나

1. **variant 캡슐화(CVA)** — `Button`/`Badge` 는 `class-variance-authority` 로 시각 변형을 props 로 노출한다. 호출부는 `<Button variant="destructive">` 처럼 *의미*만 지정한다(색상 하드코딩 제거 → OCP).
2. **제네릭 컴포넌트** — `DataTable<T>` 는 `Column<T>[]` 설정과 데이터만 받는다. 직원 목록·응급환자·협진 목록이 같은 테이블을 재사용한다(DRY).
3. **합성(composition)** — `Card`/`CardHeader`/`CardContent`, `Dialog` 계열은 작은 조각으로 쪼개 자유 조합한다. `ConfirmModal` 은 `Dialog` 를 감싼 특화 케이스.
4. **상태값 → 표현 매핑 분리** — `lib/statusTone.ts` 가 도메인 상태(협진 상태/위험도/승인)를 배지 색상으로 매핑한다. 화면 전반의 색상 규칙이 한 곳에서 일관되게 관리된다.

## 컴포넌트를 어떻게 분리했나 (SRP)

- **표현/로직 분리**: 데이터 fetch·상태는 feature 컨테이너가, 그리기는 `common`/`ui` 가 담당.
- **차트 격리**: Recharts 의존은 `LabTrendChart` 한 곳에만. 차트 라이브러리 교체 시 영향 범위가 격리된다.
- **WSI 뷰어 격리**: 줌/팬/레이어 토글 상호작용은 `WSIViewer` 가 캡슐화. 실제 타일 서버 연동 시 이 컴포넌트 내부만 교체.

## 면접 답변 요약

- *"컴포넌트를 어떻게 분리했나요?"* → 3계층(ui/common/feature) 단방향 의존 + SRP.
- *"재사용성을 어떻게 고려했나요?"* → CVA variant, 제네릭 DataTable, 합성 패턴, 상태→색상 매핑 분리.
- *"왜 이런 구조인가요?"* → 디자인 시스템(ui)을 도메인에서 떼어내 이식성과 테스트 용이성을 확보.
