# 폴더 구조 설명

CHYM-AKI 프론트엔드는 **역할(부서)별 기능 + 공통 인프라**를 분리한 feature-oriented 구조다.
"화면을 어떻게 분리했나"라는 질문에 폴더만 보고 답할 수 있도록 설계했다.

```
src/
├── App.tsx              # 라우터 + 전역 토스트 + 테마 동기화(앱 루트)
├── main.tsx             # 진입점(StrictMode + 스타일 로드)
├── index.css            # Tailwind v4 엔트리 + 디자인 토큰(라이트/다크)
│
├── types/               # 도메인 타입(단일 진실 공급원) — user/patient/bed/consult/notification/pathology
├── mock/                # Mock 데이터(부서 간 참조 일관성 유지)
├── services/            # 서비스 레이어(Facade) — Promise 반환, 백엔드 교체 지점
├── store/               # Zustand 전역 상태 — auth/notification/consult/ui
├── lib/                 # 순수 유틸 — cn, format, validation, statusTone
│
├── components/
│   ├── ui/              # shadcn 스타일 프리미티브(Button/Card/Input/Badge/Dialog)
│   └── common/          # 재사용 컴포넌트(DataTable/StatCard/PageHeader/Timeline/…)
│
├── layouts/             # 앱 셸(Sidebar/TopBar/AppLayout)
├── routes/              # 라우터 설정 + 직군 가드 + 네비 설정
│
└── features/            # 부서/도메인별 기능 단위
    ├── auth/            # 로그인·가입
    ├── admin/           # 관리자 콘솔
    ├── emergency/       # 응급의학과(병상 보드)
    ├── nephrology/      # 신장내과(검사·추이·AI초안)
    ├── pathology/       # 병리과(WSI·정량분석·보고서)
    ├── consult/         # 협진(요청 모달·현황 카드) — 신장↔병리 공유
    └── notification/    # 알림 표현 단계(배너/모달)
```

## 설계 의도

- **features/** 는 부서 단위로 응집(high cohesion). 한 부서 화면을 고치면 그 폴더만 본다.
- **components/ui vs components/common**: ui 는 디자인 시스템 원자(프로젝트 무관), common 은 도메인 맥락이 있는 재사용 조립체.
- **types/mock/services/store** 를 features 밖으로 빼서 부서 간 공유(협진처럼 부서가 교차하는 기능)를 자연스럽게 했다.
- 깊은 상대경로 대신 `@/*` 별칭(`tsconfig` + `vite` resolve)으로 이동/리팩터링 비용을 낮췄다.
