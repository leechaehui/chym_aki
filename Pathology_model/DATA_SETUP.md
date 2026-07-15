# 병리 모델 데이터 셋업 (8001 WSI `/analyze` 용)

SVN에는 **코드 + 운영 가중치만** 들어있습니다. 대용량 임베딩·manifest는 저장소에 넣지 않으므로,
팀원은 아래 데이터를 공유 드라이브/NAS에서 받아 로컬에 두고 환경변수로 위치를 지정해야 합니다.

## SVN이 제공하는 것 (`svn update` 시 자동)
- `Pathology_model/mil/*.py` — 모델 코드(TaskAttentionMIL·CdssEngine 등)
- `Pathology_model/config.json` — 라우터 임계·경로 설정
- `Pathology_model/models/cdss_shadow/ordinal_ms.pt` — 운영 가중치(24MB, 5-seed ensemble)
- `backend/wsi/` — 8001 추론서버

## 팀원이 별도로 준비할 데이터
| 데이터 | 크기 | 놓을 위치 |
| --- | --- | --- |
| `data/embeddings/ctranspath/` (index.csv + 환자별 `.npy`) | ~168MB | `<DATA_ROOT>/data/embeddings/ctranspath/` |
| `patches_manifest.csv` | ~49MB | `Pathology_model/artifacts/patches_manifest.csv` |

- `<DATA_ROOT>`는 아무 폴더나 가능(공유 드라이브 권장). 그 아래 `data/embeddings/ctranspath/` 구조를 유지할 것.
- `index.csv`의 `npy_path` 컬럼은 `<DATA_ROOT>` 기준 상대경로다.
- `patches_manifest.csv`만 예외적으로 `Pathology_model/artifacts/` 아래에 둔다(코드가 `pkg_root/artifacts`로 고정 참조).

## 환경변수 설정 (필수)
임베딩 데이터를 둔 `<DATA_ROOT>`를 가리키게 한다. **미설정 시 `config.json`의 `c:/team/chym_aki` 기본값을 써서 다른 PC에선 데이터를 못 찾는다.**

```powershell
# 예: 공유 드라이브 Z:\chym_data 아래에 data/embeddings/... 를 둔 경우
setx CHYM_WSI_DATA_ROOT "Z:\chym_data"
```

- 필요 시 추가 오버라이드: `CHYM_WSI_PKG_ROOT`(모델 코드 루트), `CHYM_WSI_SVS_ROOT`(원본 SVS — 오버레이/조직검출용, 없으면 graceful degrade).
- 설정은 새 셸부터 적용(`setx`는 현재 셸에 즉시 반영 안 됨).

## 확인
```powershell
# 8001 기동 후 (conda chym_proj)
# /slides 에 목록이 뜨고, has_features=true 인 슬라이드에서 AI 분석이 돌면 정상.
```

> 타일 뷰어(PACS 포함)는 임베딩 없이도 동작한다. 위 데이터는 **AI 분석(`/analyze`)** 에만 필요.
