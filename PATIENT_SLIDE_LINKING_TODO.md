# 환자 ↔ WSI 슬라이드 연동 — 완료 (2026-07-02)

**목표**: 병리과 화면에서 좌측 환자(협진 건) 선택 시, 우측 Viewer 탭의 슬라이드 드롭다운이 해당 환자 슬라이드만 보이도록 필터링.

**상태**: 구현 완료. 판독목록 4명 중 매핑 데이터가 있는 환자는 자동 필터링, 없는 환자는 기존처럼 전체 목록 폴백.

---

## 최종 구조

```
좌측 판독목록 환자 선택(consult.patientMrn)
        │
        ▼
GET /api/pathology/wsi-mapping/{patientMrn}   (8010, api/pathology.py)
        │  chym.phase2_wsi_mapping 조회(select slide_id, stain_type where subject_id=...)
        ▼
프론트(PathologyWorkspace.tsx) — 매핑 있으면 wsiSlides 를 slide_id 로 필터링
        │  매핑 없으면(대부분의 데모 환자) 그대로 전체 목록 폴백
        ▼
8001(wsi_main.py) 은 그대로 DB 모름 — PACS 전용 원칙 유지(B안 채택)
```

- **8001은 DB를 계속 모른다** — `chym.phase2_wsi_mapping` 조회는 8010에서만 하고, 8001엔 slide_id만 넘어간다. `wsi/README.md`의 "8001은 임상 백엔드와 분리" 원칙 그대로 유지.
- 매핑이 없는 환자(대부분)는 필터링 없이 전체 PACS 슬라이드 목록이 그대로 보인다 — 깨지지 않고 graceful fallback.

## 매핑 근거 — PACS `/api/cases` 직접 조회로 확인

판독목록 4명 중 PACS에 실제로 연결 가능한 건 `ICU-39475797` 하나뿐이었다(description에 `"이정희 (ICU-39475797)"`로 이미 명시돼 있었음). 나머지 3명은 PACS 어디에도 힌트가 없어 **HE+PAS 매칭 가능한 12명 환자 풀**(아래) 중 임의로 배정했다 — 실제 임상적 연관성은 없는 데모용 배정.

HE+PAS 둘 다 있는 12명(환자ID는 PACS case_code의 `(NN-NNNNN)` 표기 기준):
```
29-10398, 30-10018, 30-10929, 30-11033, 30-11090, 32-10034,
34-10050, 34-10184, 34-10209, 34-10240, 34-10306, 34-10331
```

## `chym.phase2_wsi_mapping` insert 내역 (8건)

| subject_id (patientMrn) | 배정 PACS 환자 | 비고 |
|---|---|---|
| `AKI-100244` (사구체신염 의증, 단백뇨) | `34-10184` | HE·PAS 둘 다 has_features=true, **ALLOW 확인됨** |
| `AKI-100258` (CKD on AKI) | `34-10306` | HE·PAS 둘 다 has_features=true, **ALLOW 확인됨** |
| `AKI-100231` (AKI stage 2) | `30-10929` | 아직 피처 추출 전(has_features=false) — 첫 분석 시 온디맨드 추출됨 |
| `ICU-39475797` (AKI Stage 2-3, 모델예측) | `30-10018` | PACS description에 이미 연결돼 있던 것. PAS 피처는 아직 추출 전 |

각 환자당 HE 1장 + PAS 1장(대표 슬라이드, 연속절편 중 1장만 선택)을 insert. `file_path`는 컬럼이 NOT NULL이라 case_code 텍스트를 그대로 넣음(실제 파일경로 아님, PACS 기반이라 로컬 경로 없음).

## 수정된 파일

- `backend/api/pathology.py` — `GET /pathology/wsi-mapping/{subject_id}` 추가(raw SQL, `chym.phase2_wsi_mapping` 조회)
- `frontend/src/services/pathologyService.ts` — `getWsiMapping(patientMrn)` 추가
- `frontend/src/features/pathology/PathologyWorkspace.tsx` — `consult` 변경 시 매핑 조회(`wsiMapping` state), `visibleWsiSlides`에서 매핑 있으면 필터링

## 남은 참고사항

- **`chym.wsi_metadata`(26건, HE only, dzi_available 전부 false)는 이 용도로 쓰지 않음** — WSI 뷰어용이 아니라 `chym_aki_ai/data4/cbr_index/`(유사증례 검색) 참조 테이블로 추정. 혼동 주의.
- `AKI-100231`/`ICU-39475797`의 PAS는 아직 온디맨드 추출 전이라, 처음 열람 시 "피처 추출 시작" 버튼을 눌러야 함(정상 흐름).
- 나머지 3명(30-11033, 30-11090, 32-10034 등)은 아직 아무 판독목록 환자에도 배정 안 함 — 필요하면 추가 insert만 하면 됨.
 