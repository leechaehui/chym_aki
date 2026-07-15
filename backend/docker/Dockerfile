# ==========================================================
# CHYM-AKI Backend 이미지
# 빌드 컨텍스트는 backend/ 디렉터리를 기준으로 한다.
#   docker build -f docker/Dockerfile -t chym-aki-backend .
# ==========================================================
FROM python:3.12-slim

# 파이썬 런타임 최적화 + psycopg2 빌드용 최소 패키지.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# 1) 의존성 레이어 캐시 — requirements 변경시에만 재설치.
COPY requirements.txt .
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc libpq-dev \
    && pip install --upgrade pip \
    && pip install -r requirements.txt \
    && apt-get purge -y gcc \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

# 2) 애플리케이션 소스.
COPY . .

EXPOSE 8000

# 10. 앱 실행 (기본 포트 8010)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8010"]
