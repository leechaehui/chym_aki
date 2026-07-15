# WSI 추론 서버 (포트 8001)

병리 WSI 판독용 추론 서버. 임상 백엔드(8010)와 **별 프로세스**로 동작한다(무거운 torch/openslide 격리 = 장애·의존성 격리).

> **다른 PC 연동 체크리스트** — 코드/번들은 그대로 옮겨도 되지만, 아래 4가지(① 런타임 env ② 데이터 ③ DB ④ config.json 경로)는 PC마다 갖춰야 한다.

---

## 1. 런타임 (conda env `chym_proj`)

WSI 서버는 `backend/.venv`(임상 백엔드용, 경량)가 **아니라** torch/openslide 가 있는 conda env 로 돈다.

```bash
conda create -n chym_proj python=3.10
conda activate chym_proj
pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU 빌드
pip install openslide-python openslide-bin                            # WSI 타일(DZI/썸네일)
pip install wsidicom                                                  # PACS DICOM WSI (썸네일/DZI)
pip install fastapi "uvicorn[standard]" pydantic numpy pandas
```

확인: `python -c "import torch, openslide, wsidicom, fastapi, uvicorn; print('ok')"`

## 2. 데이터 (repo에 없음 — 별도 배치)

| 데이터 | 위치(기본) | 비고 |
|---|---|---|
| 임베딩 npy + index.csv | `<data_root>/data/embeddings/ctranspath/` | 추론 입력(필수) |
| 원본 SVS | `<svs_root>/{wsi_he,wsi_mt,wsi_pas,wsi_silver}/` | 뷰어 타일/썸네일용. 없으면 뷰어만 graceful degrade |
| 모델 가중치 | `Pathology_model/models/cdss_shadow/ordinal_ms.pt` | repo에 포함(복사 시 따라옴) |

## 3. 경로 설정 = `Pathology_model/config.json` (머신별 단 1곳)

코드에 절대경로 하드코딩 없음. PC가 바뀌면 **이 파일만** 수정한다.

```json
{
  "data_root": "c:/dev/chym_aki",                 // 임베딩 루트
  "wsi_svs_root": "D:/chym_aki_data/backup_flawed" // SVS 루트
}
```
- `pkg_root`(코드/모델)는 파일 위치에서 자동 도출 → 설정 불필요.
- 환경변수로 덮어쓰기 가능: `CHYM_WSI_DATA_ROOT`, `CHYM_WSI_SVS_ROOT`, `CHYM_WSI_PKG_ROOT`, `CHYM_WSI_DEVICE`.

## 4. DB (임상 백엔드 8010용 — 판독 목록 데이터)

판독 목록(협진)은 8010 백엔드가 PostgreSQL 에서 제공한다.
- `backend/.env` 의 `DATABASE_URL` 가 가리키는 PostgreSQL 필요.
- 최초 1회 시드: `start_dev.bat` 이 `seed()` 실행(users 가 비어있을 때만). **users 는 있는데 협진이 비면** 아래로 보강:
  ```bash
  cd backend && .venv/Scripts/python -c "from core.database import SessionLocal; from db.seed_clinical import _consultations,_pathology; d=SessionLocal();[d.merge(x) for x in _consultations()];[d.merge(x) for x in _pathology()]; d.commit()"
  ```

---

## 실행

`start_dev.bat` 이 8010·8001·5174 를 함께 띄운다. WSI 기동 시 `chym_proj` python 을 **자동 탐지**한다(anaconda/miniconda 의 USERPROFILE·LOCALAPPDATA·ProgramData 위치 + `conda info --base`). 위치가 특이하면 `set CHYM_WSI_PYTHON=...\python.exe` 로 지정. 못 찾으면 경고만 찍고 8010/5174 는 정상 기동(WSI만 생략).

WSI 만 수동 실행:
```bash
cd backend
<chym_proj>\python.exe -m uvicorn wsi_main:app --host 0.0.0.0 --port 8001
```
헬스체크: `curl http://127.0.0.1:8001/health`

## 엔드포인트 (프론트 `/wsi-api` → 8001 프록시)

- `GET /slides?stain=HE|MT` — 슬라이드 목록
- `POST /analyze {slide_id, stain, use_cache}` — 추론(Banff 3등급·QC·ensemble·attention)
- `GET /result/{stain}/{slide_id}` — 캐시 결과
- `GET /dzi/{stain}/{slide_id}`, `GET /thumbnail/{stain}/{slide_id}` — 타일/썸네일(SVS 필요)

## 트러블슈팅

| 증상 | 원인 / 해결 |
|---|---|
| 모든 API ~220ms 고정 지연 | Windows `localhost` IPv6 폴백. `vite.config.ts` 프록시 target 을 **127.0.0.1** 로(localhost 금지) |
| 뷰어 안 뜸("SVS 없음") | `wsi_svs_root` 경로/파일 부재. config.json 의 `wsi_svs_root` 확인 |
| 판독 목록 비어있음 | 협진 미시드 → 위 4번 보강 스크립트 실행 |
| `/analyze` 503 | torch/가중치 로드 실패 → chym_proj env·`ordinal_ms.pt` 확인 |

## 아키텍처

`api`(Controller) → `services`(use-case) → `domain`(ports/strategy/mapper) → `infra`(repo/engine adapter/tile/cache/factory). 엔진은 지연주입 — 가중치 로드 실패해도 목록/타일은 동작(장애 격리).
