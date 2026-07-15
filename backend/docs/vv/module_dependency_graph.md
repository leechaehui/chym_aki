# Module Dependency Graph (작업지시서 9 + Verification 3.1)

레이어별 `import` 관계를 AST 로 추출. 상위→하위 의존만 허용한다.

```mermaid
graph TD
    ai_draft[ai_draft (ML)] --> core[core (Infra)]
    api[api (Controller)] --> core[core (Infra)]
    api[api (Controller)] --> models[models (Domain)]
    api[api (Controller)] --> schemas[schemas (DTO)]
    api[api (Controller)] --> services[services (BL)]
    core[core (Infra)] --> models[models (Domain)]
    core[core (Infra)] --> repositories[repositories (DB)]
    models[models (Domain)] --> core[core (Infra)]
    repositories[repositories (DB)] --> core[core (Infra)]
    repositories[repositories (DB)] --> models[models (Domain)]
    services[services (BL)] --> ai_draft[ai_draft (ML)]
    services[services (BL)] --> core[core (Infra)]
    services[services (BL)] --> models[models (Domain)]
    services[services (BL)] --> nlp[nlp]
    services[services (BL)] --> repositories[repositories (DB)]
    services[services (BL)] --> schemas[schemas (DTO)]
    services[services (BL)] --> stt[stt]
    services[services (BL)] --> validator[validator (V&V)]
    stt[stt] --> core[core (Infra)]
    validator[validator (V&V)] --> services[services (BL)]
```

## 레이어링 위반 점검

- 함수 내부 지연 import(예: `init_db()` 의 `import models`)는 메타데이터 등록용이므로 제외.

✅ 미허용 위반 없음 — 모든 의존이 상위→하위 방향(레이어 경계 준수).

### 선언된 예외(허용)

- core → models (deps.py): FastAPI 의존성 조립(인증 사용자 반환) — composition root
- core → repositories (deps.py): FastAPI 의존성 조립(UserRepository 위임) — composition root
