# WSI 서버(8001) GPU 머신 세팅

`wsi_main.py`(8001)만 GPU 서버에서 돌리기 위한 세팅. 코드는 SVN 으로 받고, SVN 에 없는 3가지(conda 환경, hibou-L 가중치, `.env`)만 여기서 챙긴다.

## 1. 코드

```bash
svn checkout svn://192.168.0.18/teamproj/chym_aki /opt/chym_aki
cd /opt/chym_aki
```

## 2. conda 환경

```bash
conda create -n chym_wsi python=3.13 -y
conda activate chym_wsi

conda install -y pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia   # nvidia-smi 지원 버전에 맞게 12.1 조정

pip install -r requirements-wsi-gpu.txt
```

`requirements-wsi-gpu.txt`의 `transformers==5.12.1` 고정 필수 — 3번 참고.

확인: `python -c "import torch; print(torch.cuda.is_available())"`

## 3. hibou-L(MT 임베딩) 가중치

허브(`histai/hibou-L`)를 그냥 받으면 `transformers` 5.x 와 호환이 깨져서(코드 4곳 에러), 개발 머신에서 이미 패치해둔 스냅샷을 그대로 복사해서 쓴다:

```bash
rsync -avP <개발머신>:/c/dev/chym_aki/aki_wsi_ai/hibou_L_weights/ /opt/chym_aki/aki_wsi_ai/hibou_L_weights/
```

## 4. `backend/.env`

```bash
CHYM_WSI_DATA_ROOT=/opt/chym_aki/aki_wsi
CHYM_WSI_SVS_ROOT=/opt/chym_aki/aki_wsi
CHYM_WSI_CACHE_DIR=/opt/chym_aki/pacs_cache
CHYM_WSI_DEVICE=cuda

HIBOU_L_LOCAL_PATH=/opt/chym_aki/aki_wsi_ai/hibou_L_weights

PACS_BASE_URL=http://192.168.0.47:8080
PACS_SERVICE_ID=TEAM3_WSI
PACS_SERVICE_API_KEY=<8010 .env 값 그대로>
PACS_EMPLOYEE_ID=path_lee

JWT_SECRET=<8010 .env 값 그대로 — 다르면 401>
JWT_ALGORITHM=HS256
```

## 5. 실행

```bash
cd backend
chmod +x start_wsi_gpu.sh   # 최초 1회
./start_wsi_gpu.sh
curl http://127.0.0.1:8001/health
```
