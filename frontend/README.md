# CHYM-AKI · 의료 통합 의사결정 지원 시스템 (프론트엔드)

신장내과 · 병리과 · 응급의학과 · 관리자가 사용하는 의료 의사결정 지원 시스템의 프론트엔드.
단순 화면이 아닌 **실제 병원 시스템 수준의 아키텍처**(SOLID, 재사용 컴포넌트, 전역 상태, 백엔드 교체 가능 구조)로 구현했다.

## 기술 스택

React 18 · TypeScript · Vite · Tailwind CSS v4 · shadcn 스타일 UI(Radix) · React Router(Hash) · Zustand · Recharts · lucide-react

## 실행

```bash
npm install
npm run dev        # 개발 서버 (http://localhost:5174)
npm run build      # 타입체크 + 프로덕션 빌드 → dist/ (base './' 정적/오프라인 배포 가능)
npm run typecheck  # tsc --noEmit
```

## 데모 계정 (가입 요청은 관리자 승인 후 로그인)

| 직군 | 아이디 | 비밀번호 |
|------|--------|----------|
| 관리자 | `admin` | `Admin2026!` |
| 응급의학과 | `er_kim` | `Emer2026!` |
| 신장내과 | `neph_hong` | `Neph2026!` |
| 병리과 | `path_lee` | `Path2026!` |

> 비밀번호는 프론트 데모 한계상 평문 Mock 이다. 실제 서비스는 백엔드 해시가 필수.

## 주요 기능

- **관리자**: KPI · 직원 계정 승인/거부/권한 변경
- **응급의학과**: 병상 KPI(ICU 포함) · 구역별 병상 보드 · 응급환자 위험도순 리스트
- **신장내과**: 환자 목록 · 검사 결과(정상범위 강조) · 추이 차트(Recharts) · AI 진료 초안(SOAP) · 병리 협진 요청 · 협진 회신 확인
- **병리과**: 협진 목록 · WSI 뷰어(줌/팬/레이어 토글) · AI 정량 분석 · 병리 보고서 · 협진 회신
- **협진**: 신장내과 ↔ 병리과 요청→접수→분석→판독→회신 타임라인 + 상태 관리
- **알림**: 5단계 정책(토스트/사이드/배너/모달/크리티컬) + 알림 센터 + 실시간 구독(Mock)
- **공통**: 다크모드 · 반응형(Desktop 우선) · EMR 스타일

## 문서

- [폴더 구조](docs/01-folder-structure.md)
- [컴포넌트 구조 · 재사용성](docs/02-component-structure.md)
- [상태 관리 구조](docs/03-state-management.md)
- [백엔드 연동 포인트](docs/04-backend-integration.md)
