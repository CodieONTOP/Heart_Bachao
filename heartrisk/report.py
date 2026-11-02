"""Build a printable, self-contained HTML summary for one risk estimate."""

from __future__ import annotations

import html
from datetime import UTC, datetime

from heartrisk import config

_CSS = """
:root { color-scheme: light; }
* { box-sizing: border-box; }
body {
  margin: 0 auto;
  max-width: 780px;
  padding: 32px 28px 48px;
  font-family: "Segoe UI", system-ui, -apple-system, sans-serif;
  color: #0F172A;
  line-height: 1.5;
  background: #FFFFFF;
}
h1 { font-size: 1.7rem; margin: 0 0 4px; }
h2 { font-size: 1.05rem; margin: 28px 0 10px; border-bottom: 1px solid #E2E8F0; padding-bottom: 6px; }
p { margin: 6px 0; }
.muted { color: #64748B; font-size: 0.88rem; }
.hero {
  display: flex; flex-wrap: wrap; gap: 18px; align-items: center;
  border: 1px solid #E2E8F0; border-radius: 12px; padding: 18px 20px; margin-top: 18px;
  background: #F8FAFC;
}
.probability { font-size: 2.6rem; font-weight: 700; line-height: 1; }
.badge { border-radius: 999px; padding: 5px 12px; font-size: 0.82rem; font-weight: 600; }
.badge-low { background: #D1FAE5; color: #065F46; }
.badge-moderate { background: #FEF3C7; color: #92400E; }
.badge-high { background: #FEE2E2; color: #991B1B; }
.facts { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; margin-top: 16px; }
.fact { border: 1px solid #E2E8F0; border-radius: 10px; padding: 10px 12px; }
.fact span { display: block; color: #64748B; font-size: 0.76rem; text-transform: uppercase; letter-spacing: 0.04em; }
.fact strong { font-size: 1.05rem; }
table { width: 100%; border-collapse: collapse; font-size: 0.9rem; }
th, td { text-align: left; padding: 6px 8px; border-bottom: 1px solid #F1F5F9; vertical-align: top; }
th { width: 55%; color: #475569; font-weight: 500; }
ul.drivers { list-style: none; margin: 0; padding: 0; }
ul.drivers li { display: grid; grid-template-columns: 1fr auto; gap: 4px 12px; padding: 6px 0; border-bottom: 1px solid #F1F5F9; }
.driver-label { grid-column: 1; }
.driver-value { grid-column: 2; font-variant-numeric: tabular-nums; font-weight: 600; }
.bar { grid-column: 1 / -1; height: 6px; border-radius: 999px; display: block; }
.bar-up { background: #DC2626; }
.bar-down { background: #0F766E; }
.note { background: #FFF7ED; border: 1px solid #FED7AA; border-radius: 10px; padding: 12px 14px; }
.foot { margin-top: 28px; color: #64748B; font-size: 0.8rem; }
@media print {
  body { max-width: none; padding: 0; }
  .hero, .fact, .note { break-inside: avoid; }
}
"""

_BAND_SENTENCE = {
    "low": "The calibrated probability is below the model's screening threshold. This does not rule out disease.",
    "moderate": "The calibrated probability is near the model's screening threshold. Consider discussing risk factors with a clinician.",
    "high": "The calibrated probability is at or above the model's screening threshold. This is a prompt for clinical follow-up, not a diagnosis.",
}


def _pct(value: float, digits: int = 1) -> str:
    return f"{float(value) * 100:.{digits}f}%"


def _signed_pct_points(value: float, digits: int = 1) -> str:
    return f"{float(value) * 100:+.{digits}f} pts"


def _format_value(feature: str, value) -> str:
    if feature in config.BINARY_COLUMNS:
        if isinstance(value, str):
            return "Yes" if value.strip().casefold() in {"yes", "1", "true"} else "No"
        return "Yes" if float(value) >= 0.5 else "No"
    if feature in config.NUMERIC_COLUMNS:
        return f"{float(value):g}"
    return str(value)


def _profile_table(patient: dict, features: list[str]) -> str:
    rows = []
    for feature in features:
        if feature not in patient:
            continue
        label = html.escape(config.FEATURE_LABELS.get(feature, feature))
        value = html.escape(_format_value(feature, patient[feature]))
        rows.append(f"<tr><th>{label}</th><td>{value}</td></tr>")
    return "<table>" + "".join(rows) + "</table>"


def _driver_list(items: list[dict], direction: str, empty_text: str) -> str:
    if not items:
        return f'<p class="muted">{html.escape(empty_text)}</p>'
    largest = max(abs(float(item["shap"])) for item in items) or 1.0
    bar_class = "bar-up" if direction == "up" else "bar-down"
    rows = []
    for item in items:
        shap_value = float(item["shap"])
        width = max(4.0, 100.0 * abs(shap_value) / largest)
        label = html.escape(str(item["label"]))
        value = html.escape(str(item["value"]))
        rows.append(
            "<li>"
            f'<span class="driver-label">{label} ({value})</span>'
            f'<span class="driver-value">{shap_value:+.3f}</span>'
            f'<span class="bar {bar_class}" style="width:{width:.1f}%"></span>'
            "</li>"
        )
    return '<ul class="drivers">' + "".join(rows) + "</ul>"


def _what_if_section(what_if: dict | None) -> str:
    if not what_if:
        return ""
    movers = what_if.get("movers", [])[:5]
    rows = "".join(
        "<li>"
        f'<span class="driver-label">{html.escape(str(item["label"]))}</span>'
        f'<span class="driver-value">{float(item["delta"]):+.3f}</span>'
        "</li>"
        for item in movers
    )
    movers_html = f'<ul class="drivers">{rows}</ul>' if rows else ""
    return (
        "<h2>What-if comparison</h2>"
        f"<p>Updated probability: <strong>{_pct(what_if['probability'])}</strong> "
        f"({_signed_pct_points(what_if['delta'])}).</p>"
        "<p class=\"muted\">Change in the top grouped SHAP contributions:</p>"
        f"{movers_html}"
    )


def _lower_profile_section(lower_profile: dict | None) -> str:
    if not lower_profile:
        return ""
    steps = lower_profile.get("steps", [])
    if not steps:
        return (
            "<h2>Explored lower-risk profile</h2>"
            '<p class="muted">No further realistic improvements were found within the search limits.</p>'
        )
    items = []
    for index, step in enumerate(steps, start=1):
        changes = "; ".join(
            f"{html.escape(config.FEATURE_LABELS.get(feature, feature))} → "
            f"{html.escape(_format_value(feature, value))}"
            for feature, value in step["changes"].items()
        )
        items.append(
            f"<li>{index}. {changes} — estimated probability "
            f"{_pct(step['probability'])} ({_signed_pct_points(step['delta'])})</li>"
        )
    return (
        "<h2>Explored lower-risk profile</h2>"
        f"<p>Reached <strong>{_pct(lower_profile['probability'])}</strong> "
        f"({_signed_pct_points(lower_profile['delta'])}).</p>"
        f"<ul>{''.join(items)}</ul>"
    )


def build_patient_report(
    result: dict,
    explanation: dict,
    patient: dict,
    what_if: dict | None = None,
    lower_profile: dict | None = None,
    generated_at: datetime | None = None,
) -> str:
    """Return a printable HTML snapshot of one assessment.

    All user-controlled values are HTML-escaped before rendering.
    """
    generated = generated_at or datetime.now(UTC)
    features = result.get("features") or [feature for feature in config.ALL_FEATURES if feature in patient]
    band = result.get("risk_band", "")
    band_label = result.get("risk_band_label") or config.RISK_BAND_LABELS.get(band, band)
    band_class = {"low": "badge-low", "moderate": "badge-moderate", "high": "badge-high"}.get(
        band, "badge-moderate"
    )
    feature_set = result.get("feature_set")
    feature_set_text = f"{feature_set.title()} · {len(features)} factors" if feature_set else f"{len(features)} factors"

    increasing = explanation.get("top_increasing", [])[:6]
    decreasing = explanation.get("top_decreasing", [])[:6]
    band_sentence = _BAND_SENTENCE.get(band, "Review this estimate with a clinician.")

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Heart risk summary</title>
<style>{_CSS}</style>
</head>
<body>
<p class="muted">Educational risk summary · generated {generated.strftime("%Y-%m-%d %H:%M UTC")}</p>
<h1>Heart risk summary</h1>
<p class="muted">{html.escape(feature_set_text)}</p>

<div class="hero">
  <div>
    <p class="muted">Calibrated probability</p>
    <p class="probability">{_pct(result["probability"])}</p>
  </div>
  <div>
    <span class="badge {band_class}">{html.escape(str(band_label))}</span>
    <p class="muted">Screening threshold {_pct(result["threshold"])} · raw score {_pct(result["raw_probability"])}</p>
  </div>
</div>
<p>{html.escape(band_sentence)}</p>

<div class="facts">
  <div class="fact"><span>Calibrated probability</span><strong>{_pct(result["probability"])}</strong></div>
  <div class="fact"><span>Decision threshold</span><strong>{_pct(result["threshold"])}</strong></div>
  <div class="fact"><span>Raw model score</span><strong>{_pct(result["raw_probability"])}</strong></div>
</div>

<h2>Profile</h2>
{_profile_table(patient, features)}

<h2>What moved this estimate?</h2>
<p>Grouped SHAP contributions relative to the model baseline of
{float(explanation.get("base_value", 0.0)):+.3f} log-odds. Positive values push risk up.</p>
<h3>Factors increasing risk</h3>
{_driver_list(increasing, "up", "No factor is pushing this profile above the baseline.")}
<h3>Factors decreasing risk</h3>
{_driver_list(decreasing, "down", "No factor is pulling this profile below the baseline.")}
{_what_if_section(what_if)}
{_lower_profile_section(lower_profile)}

<h2>Important limitations</h2>
<div class="note">
  <p><strong>This is not a medical device or a diagnosis.</strong> {html.escape(config.DISCLAIMER)}</p>
  <p class="muted">The model is trained on cross-sectional survey data, so it describes associations
  rather than causal effects or future clinical events. What-if changes are model-based
  explorations, not treatment recommendations. Performance is not uniform across subgroups.</p>
</div>

<p class="foot">Data: CDC BRFSS 2020 cleaned extract, CC BY 4.0. Educational demonstration only.</p>
</body>
</html>
"""
