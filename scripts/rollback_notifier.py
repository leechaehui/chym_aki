import os
import re
import getpass
import subprocess
from datetime import datetime

# Regex mappings: (Pattern, Translated Error, Recommended Action)
PIP_TRANSLATIONS = [
    (r'(?i)No matching distribution found for (.*)', r'해당 버전의 패키지를 찾을 수 없습니다: \1', '요구사항 파일(requirements.txt)의 패키지 이름과 버전을 확인하고, 팀원과 Python 버전을 맞춰보세요.'),
    (r'(?i)Could not find a version that satisfies the requirement (.*)', r'조건을 만족하는 패키지 버전을 찾을 수 없습니다: \1', '해당 패키지가 더 이상 제공되지 않거나 스펠링 오류일 수 있습니다. PyPI에서 패키지명을 확인하세요.'),
    (r'(?i)metadata generation failed', r'패키지 메타데이터 생성 실패 (버전 호환성 확인 필요)', '빌드 도구(wheel, setuptools)를 업데이트하거나, 호환되는 라이브러리 버전을 확인하세요.'),
]

NPM_TRANSLATIONS = [
    (r'(?i)ERESOLVE', r'의존성 버전 충돌 (package.json ERESOLVE)', 'npm install --legacy-peer-deps 명령을 수동으로 사용해보거나, package.json의 패키지 버전들이 서로 호환되는지 확인하세요.'),
    (r'(?i)ENOENT', r'파일이나 폴더를 찾을 수 없습니다 (ENOENT)', 'package.json 파일이 존재하지 않거나, 폴더 구조가 변경되었는지 확인하세요.'),
    (r'(?i)EACCES', r'권한 오류가 발생했습니다 (EACCES)', '관리자 권한으로 터미널을 실행하거나 npm 캐시 폴더의 소유권을 확인하세요.'),
]

DB_TRANSLATIONS = [
    (r'(?i)table "?(.*?)"? already exists', r'테이블 [\1]이(가) 이미 존재합니다 (스키마 충돌)', '기존 데이터베이스 파일(backend/chym_aki.db 등)을 삭제하고 다시 실행해보세요.'),
    (r'(?i)no such table: "?(.*?)"?($|\s.*)', r'테이블 [\1]을(를) 찾을 수 없습니다 (스키마 누락)', '데이터베이스 초기화 스크립트에 해당 테이블 생성 로직이 있는지 확인하세요.'),
    (r'(?i)UNIQUE constraint failed: (.*)', r'데이터 중복 오류 (고유 제약조건 위반): \1', '초기 시드 데이터에 이미 존재하는 데이터가 포함되어 있습니다. 시드 로직을 점검하세요.'),
    (r'(?i)FOREIGN KEY constraint failed', r'참조 데이터 없음 (외래 키 제약조건 위반)', '연관된 테이블의 데이터가 먼저 생성되어야 합니다. 데이터 삽입 순서를 확인하세요.'),
    (r'(?i)database is locked', r'데이터베이스가 잠겨 있습니다 (다른 프로세스 사용 중)', 'DBeaver 같은 DB 툴이 열려있거나 다른 터미널에서 서버가 실행 중일 수 있습니다. 모두 닫고 재시도하세요.'),
]

def parse_error():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    logs = {
        'DB': os.path.join(base_dir, 'backend', 'db_error.log'),
        'Backend (PIP)': os.path.join(base_dir, 'backend', 'pip_error.log'),
        'Frontend (NPM)': os.path.join(base_dir, 'frontend', 'npm_error.log'),
    }

    source = "Unknown"
    raw_error = "알 수 없는 설치/설정 오류"
    translated = raw_error
    action = "팀원에게 문의하여 패키지나 DB 버전, 그리고 소스코드 병합 중 발생한 충돌 여부를 확인하세요."

    for src, path in logs.items():
        if os.path.exists(path) and os.path.getsize(path) > 0:
            source = src
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
                if not lines: continue
                
                if src == 'Frontend (NPM)':
                    err_lines = [l for l in lines if 'npm ERR!' in l]
                    raw_error = err_lines[-1].strip() if err_lines else lines[-1].strip()
                    rules = NPM_TRANSLATIONS
                else:
                    raw_error = lines[-1].strip()
                    rules = DB_TRANSLATIONS if src == 'DB' else PIP_TRANSLATIONS
            
            translated = raw_error
            for pattern, repl, act in rules:
                if re.search(pattern, translated):
                    translated = re.sub(pattern, repl, translated)
                    action = act
                    break
            break

    return source, raw_error, translated, action

def write_audit_log(source, raw_error, translated, action):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    log_path = os.path.join(base_dir, 'rollback_audit.log')
    
    user = getpass.getuser()
    try:
        git_user = subprocess.check_output(['git', 'config', 'user.name'], stderr=subprocess.DEVNULL).decode('utf-8').strip()
        if git_user:
            user = f"{user} ({git_user})"
    except Exception:
        pass

    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_entry = f"[{now}] User: {user} | Source: {source} | Error: {translated} | Action: {action} | Raw: {raw_error}\n"
    
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(log_entry)

def show_popup(source, translated, action):
    msg = (
        "서버 시작 중 오류가 발생하여 설정(의존성 설치/DB 초기화)이 중단되었습니다.\n\n"
        f"[{source} 오류]\n"
        f"{translated}\n\n"
        f"[조치 사항]\n"
        f"{action}"
    )
    ps_script = (
        "Add-Type -AssemblyName PresentationCore,PresentationFramework; "
        "[System.Windows.MessageBox]::Show('"
        + msg.replace("'", "''")
        + "', 'RENAI 알림', 'OK', 'Error')"
    )
    subprocess.run(["powershell", "-NoProfile", "-Command", ps_script])

if __name__ == '__main__':
    src, raw, trans, act = parse_error()
    write_audit_log(src, raw, trans, act)
    show_popup(src, trans, act)