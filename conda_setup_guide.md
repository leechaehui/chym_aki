# 🐍 Conda 가상환경 생성 및 서비스 실행 가이드

이 가이드는 SVN에서 프로젝트를 처음 Checkout 받았거나 다른 PC에서 프로젝트를 세팅할 때, Anaconda/Miniconda(Conda) 가상환경을 생성하고 데이터베이스(PostgreSQL) 초기화 및 백엔드/프론트엔드 서비스를 구동하는 절차를 설명합니다.

모든 작업은 **PowerShell** 또는 **CMD** 터미널을 열고 진행해 주세요.

---

## 📌 SVN Checkout 이후 최초 설치 및 설정 (필수)
SVN에는 라이브러리 폴더(`node_modules`, `.venv` 등)와 개인 설정 파일(`.env`)이 업로드되지 않으므로, 처음 소스코드를 다운로드(Checkout)받은 후 반드시 다음 단계를 거쳐야 합니다.

### 1. 환경 설정 파일 (`.env`) 구성
프로젝트 루트 폴더(`chym_aki`) 기준으로 백엔드와 프론트엔드 각각의 환경 변수 파일을 복사 및 설정합니다.

* **백엔드 설정**:
  `backend/.env.example` 파일을 복사하여 **`backend/.env`** 파일을 생성하고, 실제 DB 접속 정보(예: `192.168.0.20` 등)를 입력합니다.
* **프론트엔드 설정**:
  `frontend/.env.example` 파일을 복사하여 **`frontend/.env`** 파일을 생성합니다.

### 2. 프론트엔드 패키지 설치 (`npm install`)
* **프론트엔드 디렉터리로 이동**:
  ```powershell
  cd c:\dev\workspace_proj\chym_aki\frontend

  
  ```
* **필수 라이브러리 설치**:
  ```powershell
  npm install
  ```

---

## 1단계. Conda 가상환경 생성 및 활성화
Python 3.10 버전을 사용하는 `chym_aki` 가상환경을 구축하고 활성화합니다.

```powershell
# 1. Python 3.10 기반 가상환경 생성
conda create -n chym_aki python=3.10 -y

# 2. 생성한 가상환경 활성화
conda activate chym_aki
```

---

## 2단계. 백엔드 종속성 패키지 설치
백엔드 폴더로 이동하여 필요한 라이브러리들을 설치합니다.

```powershell
# 1. 백엔드 디렉터리로 이동
cd d:\dev\workspace_proj\chym_aki\backend

# 2. requirements.txt에 명시된 패키지 설치
pip install -r requirements.txt
```

---

## 3단계. DB 모델 생성 및 초기 데이터(시드) 입력
원격 DB 서버(`192.168.0.20`)에 테이블을 생성하고 기본 시드 데이터를 주입합니다. (Conda 가상환경이 활성화된 상태여야 합니다.)

```powershell
# 1. DB 스키마 및 모델 생성
python -c "from core.database import init_db; init_db()"

# 2. 기본 사용자 및 환자 시드 데이터 적재
python -c "from db.seed import seed"

# 3. 텔레메트리 테이블 스펙 적용
python -c "from core.database import SessionLocal; from sqlalchemy import text; db=SessionLocal(); sql=open('telemetry/db_schema.sql', encoding='utf-8').read(); db.execute(text(sql)); db.commit(); db.close()"
```

---

## 4단계. 서버 구동

### ① 백엔드 서버 구동
백엔드 폴더(`backend`) 내에서 가상환경이 활성화된 터미널에 아래 명령어를 실행하여 uvicorn 서버를 실행합니다.
```powershell
uvicorn main:app --host 0.0.0.0 --port 8010
```

### ② 프론트엔드 서버 구동
**새로운 터미널 창**을 열고 프론트엔드 폴더로 이동한 후 Vite 개발 서버를 실행합니다. (Node.js 기반이므로 conda 활성화는 불필요합니다.)
```powershell
# 1. 프론트엔드 디렉터리로 이동
cd d:\dev\workspace_proj\chym_aki\frontend

# 2. 패키지 설치 (최초 1회만 필요)
npm install

# 3. Vite 개발 서버 실행
npm run dev
```

---

## 5단계. 브라우저 접속 확인
두 서버가 정상적으로 시작되면 아래 주소를 통해 접속하실 수 있습니다.
* **프론트엔드 웹 UI 대시보드**: [http://localhost:5174/](http://localhost:5174/)
* **백엔드 API 문서 (Swagger)**: [http://localhost:8010/docs](http://localhost:8010/docs)

---

## 🛠️ SVN 협업 가이드 (Update / Commit / Delete)

Windows 환경에서는 **TortoiseSVN (마우스 우클릭 GUI)** 또는 **Command Line (SVN CLI)**를 사용해 버전 관리를 진행합니다.

### 1. Update (최신 코드 가져오기)
작업을 시작하기 전이나 다른 팀원의 수정 사항을 내 컴퓨터에 반영할 때 실행합니다.
* **TortoiseSVN (GUI)**: 프로젝트 루트 폴더(`chym_aki`) 빈 곳 마우스 우클릭 ➡️ **SVN Update** 클릭
* **CLI (터미널)**:
  ```powershell
  svn update
  ```

### 2. Commit (작업 내용 저장소에 올리기)
내가 수정한 코드를 SVN 서버에 업로드합니다.
> [!IMPORTANT]
> 라이브러리 캐시 폴더인 `.venv`, `node_modules`, `__pycache__` 등은 절대 커밋에 포함시키지 마세요.
* **TortoiseSVN (GUI)**:
  1. 프로젝트 루트 폴더 마우스 우클릭 ➡️ **SVN Commit...** 클릭
  2. 변경된 파일 목록에서 커밋할 항목만 선택(체크)합니다.
  3. 상단 Message 란에 작업 내역(예: `feat: Conda 설치 가이드 추가`)을 상세히 적고 **OK**를 누릅니다.
* **CLI (터미널)**:
  ```powershell
  # 1. 새로 생성된 파일이 있다면 SVN 추적 대상으로 등록
  svn add <파일명 또는 폴더명>

  # 2. 변경 파일 커밋
  svn commit -m "작업 내용 메시지"
  ```

### 3. Delete (파일/폴더 삭제하기)
SVN 저장소에서도 해당 파일이 안전하게 지워지도록 삭제 처리합니다. (윈도우 탐색기에서 그냥 삭제할 경우 SVN 상에는 여전히 남아있을 수 있으므로 주의해야 합니다.)
* **TortoiseSVN (GUI)**:
  1. 삭제할 파일 마우스 우클릭 ➡️ **TortoiseSVN** ➡️ **Delete** 클릭
  2. 부모 폴더 또는 루트 폴더 마우스 우클릭 ➡️ **SVN Commit...**을 실행하여 삭제 내역을 서버에 반영합니다.
* **CLI (터미널)**:
  ```powershell
  # 1. SVN 삭제 등록
  svn delete <삭제할 파일명>

  # 2. 삭제 처리 커밋
  svn commit -m "delete: 불필요한 파일 정리"
  ```
