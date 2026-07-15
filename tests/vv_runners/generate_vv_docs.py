"""V&V 필수 산출물 생성기 (작업지시서 9).

코드에서 직접 추출(introspection)해 항상 최신 상태를 보장한다:
  - docs/vv/api_endpoint_list.md        : OpenAPI 기반 엔드포인트 목록
  - docs/vv/database_schema.md          : SQLAlchemy 메타데이터 기반 스키마(+ 인덱스/FK)
  - docs/vv/module_dependency_graph.md  : 레이어 import 그래프 + 레이어링 위반 점검

실행:
  cd backend && set PYTHONPATH=. && set SEED_ON_STARTUP=false ^
    && .venv\\Scripts\\python.exe tests\\vv_runners\\generate_vv_docs.py
"""
from __future__ import annotations

import ast
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent        # <repo>/tests/vv_runners
BACKEND = HERE.parent.parent / "backend"       # <repo>/backend
sys.path.insert(0, str(BACKEND))
os.environ.setdefault("SEED_ON_STARTUP", "false")

OUT = BACKEND / "docs" / "vv"

# 레이어 정의 + 허용 의존(상위 → 하위만 허용). 숫자가 클수록 상위 레이어.
LAYER_RANK = {
    "api": 5,
    "services": 4,
    "validator": 4,
    "ai_draft": 3,
    "nlp": 3,
    "stt": 3,
    "repositories": 2,
    "schemas": 2,
    "models": 1,
    "core": 0,
}


# --------------------------------------------------------------------------
# 1) API 엔드포인트 목록
# --------------------------------------------------------------------------
def gen_endpoints() -> int:
    from main import app

    spec = app.openapi()
    rows = []
    for path, ops in spec["paths"].items():
        for method, op in ops.items():
            rows.append(
                (op.get("tags", ["?"])[0], method.upper(), path, op.get("summary", ""))
            )
    rows.sort(key=lambda r: (r[0], r[2]))

    lines = [
        "# API Endpoint List (작업지시서 9)",
        "",
        f"총 **{len(rows)}** 개 엔드포인트. OpenAPI(`app.openapi()`)에서 자동 추출.",
        "",
        "| Tag | Method | Path | Summary |",
        "|---|---|---|---|",
    ]
    for tag, method, path, summary in rows:
        lines.append(f"| {tag} | {method} | `{path}` | {summary} |")
    lines.append("")
    (OUT / "api_endpoint_list.md").write_text("\n".join(lines), encoding="utf-8")
    return len(rows)


# --------------------------------------------------------------------------
# 2) 데이터베이스 스키마
# --------------------------------------------------------------------------
def gen_schema() -> int:
    import models  # noqa: F401 (메타데이터 등록)
    from core.database import Base

    md = Base.metadata
    lines = [
        "# Database Schema (작업지시서 9)",
        "",
        f"총 **{len(md.tables)}** 테이블. SQLAlchemy 메타데이터에서 자동 추출.",
        "",
    ]
    for t in md.sorted_tables:
        lines.append(f"## `{t.name}`")
        lines.append("")
        lines.append("| Column | Type | Null | Key |")
        lines.append("|---|---|---|---|")
        for c in t.columns:
            key = "PK" if c.primary_key else ""
            if c.foreign_keys:
                key = (key + " FK→" + list(c.foreign_keys)[0].target_fullname).strip()
            lines.append(
                f"| {c.name} | {c.type} | {'Y' if c.nullable else 'N'} | {key} |"
            )
        if t.indexes:
            idx = "; ".join(
                f"`{ix.name}`({', '.join(col.name for col in ix.columns)})"
                + (" UNIQUE" if ix.unique else "")
                for ix in sorted(t.indexes, key=lambda i: i.name)
            )
            lines.append("")
            lines.append(f"**Indexes:** {idx}")
        lines.append("")
    (OUT / "database_schema.md").write_text("\n".join(lines), encoding="utf-8")
    return len(md.tables)


# --------------------------------------------------------------------------
# 3) 모듈 의존성 그래프 + 레이어링 위반 점검
# --------------------------------------------------------------------------
# 선언된 예외(composition root). core.deps 는 FastAPI 의존성 조립 지점으로,
# 인증/인가를 위해 models·repositories 를 참조하는 것이 의도된 설계다.
# (DB 쿼리는 repository 안에 캡슐화되어 있어 '쿼리 누출'은 아니다.)
ALLOWED_EXCEPTIONS = {
    ("core", "models", "deps.py"): "FastAPI 의존성 조립(인증 사용자 반환) — composition root",
    ("core", "repositories", "deps.py"): "FastAPI 의존성 조립(UserRepository 위임) — composition root",
}


def _layer_of(module_path: str) -> str | None:
    top = module_path.split(".")[0]
    return top if top in LAYER_RANK else None


def _module_level_imports(tree: ast.Module):
    """함수/메서드 내부의 지연 import 는 제외하고 모듈 레벨 import 만 수집.

    (예: core/database.init_db() 안의 `import models` 는 메타데이터 등록용
     지연 import 이므로 레이어 의존으로 보지 않는다.)
    """
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            yield node
        elif isinstance(node, (ast.If, ast.Try)):
            for sub in ast.walk(node):
                if isinstance(sub, (ast.Import, ast.ImportFrom)) and sub is not node:
                    yield sub


def gen_dependency_graph() -> int:
    edges: set[tuple[str, str]] = set()
    violations: list[str] = []
    accepted: list[str] = []

    for layer in LAYER_RANK:
        d = BACKEND / layer
        if not d.is_dir():
            continue
        for py in d.glob("*.py"):
            try:
                tree = ast.parse(py.read_text(encoding="utf-8"))
            except SyntaxError:
                continue
            for node in _module_level_imports(tree):
                targets = []
                if isinstance(node, ast.ImportFrom) and node.module:
                    targets = [_layer_of(node.module)]
                elif isinstance(node, ast.Import):
                    targets = [_layer_of(n.name) for n in node.names]
                for target in targets:
                    if not target or target == layer:
                        continue
                    edges.add((layer, target))
                    if LAYER_RANK[layer] < LAYER_RANK[target]:
                        key = (layer, target, py.name)
                        if key in ALLOWED_EXCEPTIONS:
                            note = f"{layer} → {target} ({py.name}): {ALLOWED_EXCEPTIONS[key]}"
                            if note not in accepted:
                                accepted.append(note)
                        else:
                            viol = f"{layer} → {target} ({py.name})"
                            if viol not in violations:
                                violations.append(viol)

    lines = [
        "# Module Dependency Graph (작업지시서 9 + Verification 3.1)",
        "",
        "레이어별 `import` 관계를 AST 로 추출. 상위→하위 의존만 허용한다.",
        "",
        "```mermaid",
        "graph TD",
    ]
    label = {
        "api": "api (Controller)",
        "services": "services (BL)",
        "validator": "validator (V&V)",
        "ai_draft": "ai_draft (ML)",
        "nlp": "nlp",
        "stt": "stt",
        "repositories": "repositories (DB)",
        "schemas": "schemas (DTO)",
        "models": "models (Domain)",
        "core": "core (Infra)",
    }
    for src, dst in sorted(edges):
        lines.append(f"    {src}[{label.get(src, src)}] --> {dst}[{label.get(dst, dst)}]")
    lines.append("```")
    lines.append("")
    lines.append("## 레이어링 위반 점검")
    lines.append("")
    lines.append("- 함수 내부 지연 import(예: `init_db()` 의 `import models`)는 메타데이터 등록용이므로 제외.")
    lines.append("")
    if violations:
        lines.append("> ⚠️ 하위 레이어가 상위 레이어를 import (SRP/의존 역전 위반):")
        lines.append("")
        for v in violations:
            lines.append(f"- {v}")
    else:
        lines.append("✅ 미허용 위반 없음 — 모든 의존이 상위→하위 방향(레이어 경계 준수).")
    lines.append("")
    if accepted:
        lines.append("### 선언된 예외(허용)")
        lines.append("")
        for a in accepted:
            lines.append(f"- {a}")
        lines.append("")
    (OUT / "module_dependency_graph.md").write_text("\n".join(lines), encoding="utf-8")
    return len(violations)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    n_ep = gen_endpoints()
    n_tbl = gen_schema()
    n_viol = gen_dependency_graph()
    print(f"[OK] endpoints={n_ep}  tables={n_tbl}  layering_violations={n_viol}")
    print(f"     → {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
