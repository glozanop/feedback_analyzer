# build_report.py (fixed booleans)
import argparse, json, base64
from pathlib import Path
import pandas as pd

def read_csv(p):
    try: return pd.read_csv(p)
    except Exception as e: return f"Error reading CSV: {e}"

def read_json(p):
    try: return json.loads(p.read_text(encoding="utf-8"))
    except Exception as e: return f"Error reading JSON: {e}"

def b64img(p):
    try:
        ext = p.suffix.lower().lstrip(".") or "png"
        mime = "png" if ext=="png" else "jpeg" if ext in {"jpg","jpeg"} else "gif"
        return f"data:image/{mime};base64," + base64.b64encode(p.read_bytes()).decode()
    except: return None

def table(df, n=50):
    if isinstance(df, pd.DataFrame): return df.head(n).to_html(classes="table", index=False, border=0)
    if isinstance(df, str): return f'<pre class="code">{df}</pre>'
    return "<p class='muted'>No data.</p>"

def code(txt):
    return f"<pre class='code'>{txt}</pre>" if txt else "<p class='muted'>No data.</p>"

def _is_df(x):      return isinstance(x, pd.DataFrame)
def _is_dict(x):    return isinstance(x, dict)
def _is_list(x):    return isinstance(x, list)
def _has_df(x):     return _is_df(x)                     # DataFrame present
def _has_json(x):   return _is_dict(x) or _is_list(x)    # JSON-like present
def _has_str(x):    return isinstance(x, str) and len(x) > 0
def _has_img(x):    return isinstance(x, str) and x.startswith("data:image")

def main(in_dir: Path, out_html: Path, title: str):
    g = {p.name.lower(): p for p in in_dir.rglob("*") if p.is_file()}
    def get(name): return g.get(name.lower())

    # ------------ load inputs ------------
    themes           = read_csv(get("themes_tagged_feedback.csv")) if get("themes_tagged_feedback.csv") else None
    theme_consol     = read_json(get("theme_consolidation.json")) if get("theme_consolidation.json") else None

    weekly           = read_csv(get("trend_weekly_volume.csv")) if get("trend_weekly_volume.csv") else None
    monthly          = read_csv(get("trend_monthly_volume.csv")) if get("trend_monthly_volume.csv") else None
    weekly_img       = b64img(get("trend_weekly_chart.png")) if get("trend_weekly_chart.png") else None

    trend_summary    = None
    if get("trend_summary.json"):
        js = read_json(get("trend_summary.json"))
        trend_summary = json.dumps(js, indent=2, ensure_ascii=False) if not isinstance(js, str) else js
    elif get("trend_summary.txt"):
        trend_summary = get("trend_summary.txt").read_text(encoding="utf-8")

    anomaly          = read_json(get("anomaly_report.json")) if get("anomaly_report.json") else None
    anomaly_img      = b64img(get("trend_weekly_chart_with_anomalies.png")) if get("trend_weekly_chart_with_anomalies.png") else None

    seg_tier         = read_csv(get("segmentation_by_tier.csv")) if get("segmentation_by_tier.csv") else None
    seg_spend        = read_csv(get("segmentation_by_spend.csv")) if get("segmentation_by_spend.csv") else None
    seg_appr         = read_csv(get("segmentation_by_approval_ratio.csv")) if get("segmentation_by_approval_ratio.csv") else None

    neg_ratio        = read_csv(get("negative_sentiment_ratio.csv")) if get("negative_sentiment_ratio.csv") else None
    stat_sum         = read_csv(get("statistical_summary.csv")) if get("statistical_summary.csv") else None

    recs             = read_json(get("strategic_recommendations.json")) if get("strategic_recommendations.json") else None
    expl_json        = read_json(get("explainability_samples.json")) if get("explainability_samples.json") else None

    # ------------ css & helpers ------------
    css = """
    :root { --bg:#0b0f15; --card:#111827; --card2:#0f172a; --text:#e5e7eb; --muted:#9ca3af; --border:#1f2937; }
    *{box-sizing:border-box} html,body{margin:0;padding:0;background:var(--bg);color:var(--text);font-family:ui-sans-serif,-apple-system,Segoe UI,Roboto,Arial}
    .topbar{width:100%;background:linear-gradient(90deg,#6d28d9,#9333ea,#a855f7);color:#fff;font-weight:800;letter-spacing:.3px;padding:14px 20px;font-size:18px;position:sticky;top:0;z-index:10;box-shadow:0 6px 20px rgba(0,0,0,.25)}
    .wrapper{display:grid;grid-template-columns:260px 1fr;min-height:100vh}
    .aside{padding:24px;border-right:1px solid var(--border);position:sticky;top:0;height:100vh;overflow:auto;background:linear-gradient(180deg,rgba(17,24,39,.9),rgba(15,23,42,.9))}
    .brand{font-weight:800;font-size:20px;margin-bottom:8px}.brand small{color:var(--muted);font-weight:500;display:block;margin-top:4px}
    .toc{margin-top:16px;display:flex;flex-direction:column;gap:8px}
    .toc a{color:var(--text);text-decoration:none;padding:10px 12px;border-radius:10px;border:1px solid var(--border)}
    .main{padding:28px}.grid{display:grid;gap:16px;grid-template-columns:repeat(12,minmax(0,1fr))}
    .card{grid-column:span 12;background:radial-gradient(1000px 200px at 0% -10%,rgba(96,165,250,.10),transparent 40%),var(--card);border:1px solid var(--border);border-radius:16px;padding:18px;box-shadow:0 10px 30px rgba(0,0,0,.25)}
    .card h2{margin:0 0 8px;font-size:18px}.muted{color:var(--muted);font-size:14px}
    .table{width:100%;border-collapse:collapse;margin-top:8px}.table th,.table td{padding:8px 10px;border-bottom:1px solid var(--border);text-align:left;font-size:13px}
    .image{width:100%;border-radius:12px;border:1px solid var(--border)}
    .code{background:var(--card2);border:1px solid var(--border);border-radius:12px;padding:12px;white-space:pre-wrap;overflow:auto;font-size:13px;line-height:1.4}
    .two{display:grid;gap:16px}@media(min-width:1024px){.two{grid-template-columns:1fr 1fr}}
    """

    def sec(id, title, sub=""):
        return f'<div id="{id}" class="card"><h2>{title}</h2>' + (f"<div class='muted'>{sub}</div>" if sub else "")
    def end(): return "</div>"

    # ------------ TOC (use explicit booleans) ------------
    toc = "".join(
        f'<a href="#{sid}">{label}</a>'
        for sid, label, ok in [
            ("theme-mining", "Theme Mining",      _has_df(themes) or _has_json(theme_consol)),
            ("trends",       "Theme Trends",      _has_df(weekly) or _has_df(monthly) or _has_img(weekly_img) or _has_str(trend_summary)),
            ("anomalies",    "Anomaly Detection", _has_json(anomaly) or _has_img(anomaly_img)),
            ("segmentation", "Segmentation",      _has_df(seg_tier) or _has_df(seg_spend) or _has_df(seg_appr)),
            ("quality",      "Quality Metrics",   _has_df(neg_ratio) or _has_df(stat_sum)),
            ("recs",         "Recommendations",   _has_json(recs)),
            ("explain",      "Explainability",    _has_df(themes) or _has_json(expl_json)),
        ] if ok
    )

    # ------------ HTML skeleton ------------
    html = [f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1" />
    <title>{title}</title><style>{css}</style></head><body>
    <div class="topbar">{title}<span class="sub" style="font-weight:500;opacity:.85;font-size:12px;margin-left:8px">automated report</span></div>
    <div class="wrapper">
      <aside class="aside">
        <div class="brand">Feedback-Genie<small>Report builder</small></div>
        <div class="toc">{toc or "<span class='muted'>No sections</span>"}</div>
      </aside>
      <main class="main">
    """ ]

    # Executive summary
    html += ['<div class="grid"><div class="card"><h2>Executive Summary</h2>', code(trend_summary or "No trend summary available."), "</div></div>"]

    # Sections (use explicit checks)
    if _has_df(themes) or _has_json(theme_consol):
        html += [sec("theme-mining","Theme Mining","Tagged feedback and consolidated categories")]
        if _has_df(themes): html += ["<h3>themes_tagged_feedback.csv (preview)</h3>", table(themes, 50)]
        if _has_json(theme_consol):
            pretty = json.dumps(theme_consol, indent=2, ensure_ascii=False) if not isinstance(theme_consol, str) else theme_consol
            html += ["<h3>theme_consolidation.json</h3>", f"<pre class='code'>{pretty}</pre>"]
        html += [end()]

    if _has_df(weekly) or _has_df(monthly) or _has_img(weekly_img) or _has_str(trend_summary):
        html += [sec("trends","Theme Trends","Weekly & Monthly volumes")]
        if _has_img(weekly_img): html += ["<h3>Weekly Trend Chart</h3>", f'<img src="{weekly_img}" class="image" alt="weekly chart" />']
        if _has_df(weekly):      html += ["<h3>trend_weekly_volume.csv (preview)</h3>", table(weekly, 50)]
        if _has_df(monthly):     html += ["<h3>trend_monthly_volume.csv (preview)</h3>", table(monthly, 50)]
        html += [end()]

    if _has_json(anomaly) or _has_img(anomaly_img):
        html += [sec("anomalies","Anomaly Detection","Report & annotated chart")]
        if _has_json(anomaly):
            pretty = json.dumps(anomaly, indent=2, ensure_ascii=False) if not isinstance(anomaly, str) else anomaly
            html += ["<h3>anomaly_report.json</h3>", f"<pre class='code'>{pretty}</pre>"]
        if _has_img(anomaly_img): html += ["<h3>Weekly Trend with Anomalies</h3>", f'<img src="{anomaly_img}" class="image" alt="anomaly chart" />']
        html += [end()]

    if _has_df(seg_tier) or _has_df(seg_spend) or _has_df(seg_appr):
        html += [sec("segmentation","Segmentation","Tier, spend, and approval cohorts")]
        if _has_df(seg_tier):  html += ["<h3>segmentation_by_tier.csv</h3>", table(seg_tier, 50)]
        if _has_df(seg_spend): html += ["<h3>segmentation_by_spend.csv</h3>", table(seg_spend, 50)]
        if _has_df(seg_appr):  html += ["<h3>segmentation_by_approval_ratio.csv</h3>", table(seg_appr, 50)]
        html += [end()]

    if _has_df(neg_ratio) or _has_df(stat_sum):
        html += [sec("quality","Quality Metrics","Negative sentiment ratio & statistical summary")]
        if _has_df(neg_ratio): html += ["<h3>negative_sentiment_ratio.csv</h3>", table(neg_ratio, 50)]
        if _has_df(stat_sum):  html += ["<h3>statistical_summary.csv</h3>", table(stat_sum, 50)]
        html += [end()]

    if _has_json(recs):
        pretty = json.dumps(recs, indent=2, ensure_ascii=False) if not isinstance(recs, str) else recs
        html += [sec("recs","Recommendations","AI-generated strategic actions"),
                 "<h3>strategic_recommendations.json</h3>", f"<pre class='code'>{pretty}</pre>", end()]

    if _has_df(themes) or _has_json(expl_json):
        html += [sec("explain","Explainability","Examples mapping to themes")]
        if _has_df(themes):
            col = next((c for c in ["theme_category","concise_theme","theme"] if c in themes.columns), None)
            if col:
                html += ['<div class="two">']
                for t in themes[col].value_counts().head(5).index:
                    subset = themes[themes[col]==t]
                    cols = [c for c in ["message", col, "sentiment"] if c in subset.columns]
                    if cols:
                        html += [f"<div><h3>{t}</h3>", table(subset[cols], 3), "</div>"]
                html += ['</div>']
            else:
                html += ["<p class='muted'>No theme column detected.</p>"]
        if _has_json(expl_json):
            html += ["<h3>explainability_samples.json</h3>", f"<pre class='code'>{json.dumps(expl_json, indent=2, ensure_ascii=False)}</pre>"]
        html += [end()]

    html += ["</main></div></body></html>"]
    out_html.write_text("".join(html), encoding="utf-8")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", default="Outputs")
    ap.add_argument("--out", default="Notebook/Feedback-Analysis.html")
    ap.add_argument("--title", default="Feedback-Genie Analysis")
    args = ap.parse_args()
    main(Path(args.inputs), Path(args.out), args.title)
