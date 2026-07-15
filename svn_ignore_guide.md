# 📁 CHYM-AKI 프로젝트 SVN Ignore 설정 가이드

이 가이드는 SVN 저장소에 프로젝트 코드를 올릴 때, 불필요한 라이브러리 폴더, 가상환경, 개인 설정 파일, 캐시 파일들이 업로드되지 않도록 **Ignore(버전 관리 제외)** 목록을 설정하는 방법과 대상을 안내합니다.

---

## 📌 Ignore 설정 대상 목록

`chym_aki` 프로젝트 구조 기준으로 SVN 저장소에서 반드시 제외되어야 하는 항목들입니다.

| 분류 | 대상 폴더 / 파일 | 설명 |
| :--- | :--- | :--- |
| **의존성 폴더** | `backend/.venv/`<br>`frontend/node_modules/` | 로컬 가상환경 및 Node.js 패키지 폴더 (각 개발자가 `pip/npm install`로 설치하므로 커밋 제외) |
| **환경 설정** | `backend/.env`<br>`frontend/.env` | 각 개발자 환경에 따른 로컬 환경 변수 파일 (IP, DB 주소, JWT 비밀값 등 민감 정보 포함) |
| **빌드/캐시** | `frontend/dist/`<br>`**/__pycache__/`<br>`**/*.pyc`<br>`.pytest_cache/` | 프론트엔드 배포용 빌드 폴더 및 파이썬 컴파일/테스트 캐시 파일 |
| **IDE 설정** | `.idea/`<br>`.vscode/`<br>`.claude/` | PyCharm, VS Code 등의 로컬 에디터 설정 및 도구 로그 폴더 |
| **로그 / 백업** | `backup/`<br>`backend/*.log`<br>`frontend/*.log`<br>`*.log` | 로컬 서버 로그 파일, 백업 스크립트로 생성된 백업 폴더들 |
| **업로드 파일** | `backend/uploads/signatures/*` | 개발 및 테스트 시 업로드되는 의사 자필 서명 이미지 데이터 (폴더 자체는 남겨두되, 내부 파일은 제외) |

---

## 🛠️ SVN Ignore 적용 방법

Git의 `.gitignore` 파일과 달리, SVN은 폴더의 **속성(Property)**으로 ignore 규칙을 관리합니다. **TortoiseSVN(GUI)** 또는 **SVN CLI(명령줄)** 중 편한 방법을 선택하여 설정해 주세요.

### 방법 1. TortoiseSVN (GUI) 사용법 (추천)
가장 간단하고 직관적인 방법입니다.

1. **폴더 전체 제외 설정**:
   * 설정할 폴더(예: `backend/.venv`)를 마우스 우클릭합니다.
   * **[TortoiseSVN]** ➡️ **[Unversion and add to ignore list]** ➡️ **[.venv]** 를 클릭합니다.
   * `frontend/node_modules`, `.idea`, `.vscode` 폴더 등도 동일한 방식으로 적용합니다.

2. **개별 파일 또는 확장자 제외 설정**:
   * 설정할 파일(예: `backend/.env` 또는 `*.log` 파일)을 마우스 우클릭합니다.
   * **[TortoiseSVN]** ➡️ **[Unversion and add to ignore list]** ➡️ **[.env]** (또는 **[*.log]**)를 클릭합니다.

3. **속성 메뉴에서 직접 추가하는 방법 (전체 일괄 등록 시 유용)**:
   * 프로젝트 루트 폴더(`chym_aki`) 빈 곳 마우스 우클릭 ➡️ **[Properties]**를 누릅니다.
   * **[New...]** ➡️ **[Ignore]**를 클릭하거나 **`svn:ignore`** 속성을 선택합니다.
   * 아래의 텍스트 상자에 무시할 패턴들을 줄바꿈하여 복사-붙여넣기 한 후 **[OK]**를 누릅니다:
     ```text
     .venv
     node_modules
     .env
     dist
     __pycache__
     *.pyc
     .pytest_cache
     .idea
     .vscode
     .claude
     backup
     *.log
     ```

---

### 방법 2. SVN Command Line (CLI) 사용법
터미널 환경에서 명령어로 적용하고 싶을 때 사용합니다.

```powershell
# 1. 프로젝트 루트 폴더로 이동
cd d:\dev\workspace_proj\chym_aki

# 2. 루트 폴더의 svn:ignore 속성에 무시할 폴더/파일 추가
# (TortoiseSVN이 없는 리눅스/빌드서버 환경 등에서 활용)
svn propset svn:ignore -F - . <<EOF
.venv
node_modules
.env
dist
__pycache__
*.pyc
.pytest_cache
.idea
.vscode
.claude
backup
*.log
EOF

# 3. 설정된 ignore 속성 확인
svn proplist -v
```

---

## ⚠️ 커밋 전 최종 점검
위 설정을 모두 마치고 처음 **SVN Commit** 버튼을 눌렀을 때, 커밋 목록 파일 리스트에 `.venv` 폴더나 `node_modules` 폴더의 무수한 파일들이 나타나지 않고 **순수 프로젝트 소스 코드**만 깔끔하게 뜨는지 확인한 뒤 커밋을 완료해 주시면 됩니다.
 