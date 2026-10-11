"""Compact downloadable reports for subtype-aware research inference."""

from __future__ import annotations

from html import escape
from typing import Any


def btxrd_result_to_html(result: Any) -> str:
    """Render prediction metadata as a self-contained, non-clinical HTML report."""

    rows = "".join(
        "<tr><td>{}</td><td>{:.4f}</td></tr>".format(
            escape(str(row.get("name", ""))), float(row.get("probability", 0.0))
        )
        for row in result.subtype_top_k
    )
    quality = result.quality if isinstance(result.quality, dict) else {}
    warning_text = ", ".join(str(item) for item in quality.get("warnings", [])) or "none"
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>BTXRD research inference</title>
<style>body{{font-family:system-ui;max-width:760px;margin:2rem auto;padding:0 1rem}}
table{{border-collapse:collapse;width:100%}}
td,th{{border:1px solid #ddd;padding:.45rem;text-align:left}}
.notice{{background:#fff3cd;padding:.75rem}}</style></head><body>
<h1>BTXRD HOSP-BoneNet research result</h1>
<div class="notice">Research use only; not a diagnosis.</div>
<p><b>Subtype:</b> {escape(str(result.subtype_prediction))}<br>
<b>Tumour probability:</b> {float(result.binary_probability):.4f}<br>
<b>Confidence:</b> {float(result.binary_confidence):.4f}<br>
<b>Uncertainty:</b> {float(result.binary_uncertainty):.4f}</p>
<h2>Subtype ranking</h2><table><tr><th>Subtype</th><th>Probability</th></tr>{rows}</table>
<h2>Uncertainty</h2><p>Binary set: {escape(str(result.binary_conformal_set))}<br>
Subtype set: {escape(str(result.subtype_conformal_set))}</p>
<h2>Input quality</h2><p>Status: {escape(str(quality.get("status", "unknown")))}<br>
Warnings: {escape(warning_text)}</p>
<h2>Provenance</h2><p>Alias: {escape(str(result.model_alias))}<br>
Run: {escape(str(result.run_id))}<br>Checkpoint SHA-256: {escape(str(result.checkpoint_sha256))}</p>
</body></html>"""


__all__ = ["btxrd_result_to_html"]
