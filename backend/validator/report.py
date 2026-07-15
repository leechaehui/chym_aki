"""ValidationReport — 검증 결과 종합 + Markdown/JSON 직렬화.

작업지시서 9 의 'AI model validation report' 산출물 형식을 표준화한다.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class ValidationReport:
    title: str
    generated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )
    dataset: dict[str, Any] = field(default_factory=dict)
    sections: dict[str, Any] = field(default_factory=dict)
    known_failures: list[str] = field(default_factory=list)

    def add(self, key: str, value: Any) -> None:
        self.sections[key] = value

    def to_json(self) -> str:
        return json.dumps(
            {
                "title": self.title,
                "generated_at": self.generated_at,
                "dataset": self.dataset,
                "sections": self.sections,
                "known_failures": self.known_failures,
            },
            ensure_ascii=False,
            indent=2,
        )

    # --- Markdown 렌더링 ----------------------------------------------------
    def _render_binary(self, name: str, m: dict) -> list[str]:
        lines: list[str] = []
        lines.append(
            f"- N={m['n']}, positives={m['positives']} "
            f"(prevalence {m['prevalence']:.1%})"
        )
        lines.append(
            f"- **AUROC {m['auroc']:.4f}** · **AUPRC {m['auprc']:.4f}** · "
            f"Brier {m['brier']:.4f} · ECE {m['ece']:.4f}"
        )
        lines.append("")
        lines.append("| 확률 bin | n | 평균예측 | 관측빈도 |")
        lines.append("|---|---:|---:|---:|")
        for b in m["calibration"]:
            lines.append(
                f"| {b['bin']} | {b['count']} | {b['mean_predicted']:.3f} | "
                f"{b['observed_rate']:.3f} |"
            )
        lines.append("")
        return lines

    def to_markdown(self) -> str:
        L: list[str] = [f"# {self.title}", "", f"_생성: {self.generated_at}_", ""]
        if self.dataset:
            L += ["## 데이터셋", ""]
            for k, v in self.dataset.items():
                L.append(f"- **{k}**: {v}")
            L.append("")

        for key, val in self.sections.items():
            L += [f"## {key}", ""]
            if isinstance(val, dict) and "auroc" in val and "calibration" in val:
                L += self._render_binary(key, val)
            elif isinstance(val, list):
                L.append("```json")
                L.append(json.dumps(val, ensure_ascii=False, indent=2))
                L.append("```")
                L.append("")
            else:
                L.append("```json")
                L.append(json.dumps(val, ensure_ascii=False, indent=2))
                L.append("```")
                L.append("")

        if self.known_failures:
            L += ["## Known failure cases", ""]
            for f in self.known_failures:
                L.append(f"- {f}")
            L.append("")
        return "\n".join(L)
