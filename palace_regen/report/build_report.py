#!/usr/bin/env python3
"""Build the Palace recreation report as one self-contained HTML page.

Usage: python3 -I build_report.py palace-recreation.md figs palace-recreation.html
The Markdown holds the prose; {{FIG:name|caption}} inlines figs-dir/name.svg and {{TABLE:name}}
renders a table from figs-dir/tables.json (written by palace_regen/report_figs.py).
"""
import datetime
import html
import json
import pathlib
import re
import sys

import markdown

SRC, FIGS, OUT = (pathlib.Path(a).resolve() for a in sys.argv[1:4])
T = json.loads((FIGS / "tables.json").read_text())

LIGHT = ("--bg:#f7f6f2;--surface:#ffffff;--text:#1f2328;--muted:#5b616b;--border:#e2dfd6;--rule:#1f2328;"
         "--accent:#245a8c;--accent-soft:#e5edf6;--code-bg:#efede6;--th-bg:#f2f0ea;--good:#15803d;--bad:#b91c1c;")
DARK = ("--bg:#121417;--surface:#1a1d21;--text:#e6e4df;--muted:#a1a7b0;--border:#2d3137;--rule:#e6e4df;"
        "--accent:#8ab6e4;--accent-soft:#1c2937;--code-bg:#23272d;--th-bg:#20242a;--good:#4ade80;--bad:#f87171;")


def pct(v, nd=1):
    return "&ndash;" if v is None else f"{100 * v:.{nd}f}%"


def table(head, rows, cls=""):
    th = "".join(f"<th>{h}</th>" for h in head)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f'<div class="table-wrap"><table class="{cls}"><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table></div>'


def mesh_label(tag):
    m = re.match(r"es_p1_uni([\d.]+)t", tag)
    if m:
        return f"uniform, {m.group(1)}&thinsp;t"
    m = re.match(r"es_p1_pts([\d.]+)-([\d.]+)t", tag)
    if m:
        return f"graded, {m.group(1)}&thinsp;t at trace, {m.group(2)}&thinsp;t at walls"
    return tag


def t_designs():
    rows = []
    for r in T["designs"]:
        rows.append([f"{r['row']}", (r["split"] or "").replace("_", " "), f"{r['t_over_h']:.3f}",
                     f"{r['w_over_h']:.2f}", f"{r['eps_r']:.2f}", pct(r["direct_scaled"]),
                     pct(r["fixed_raw"]), pct(r["fixed_scaled"]), pct(r["fixed_drop3"]),
                     f"{pct(r['chord_scaled'])} <span class=sub>{mesh_label(r['best_mesh'])}</span>"])
    return table(["Row", "Split", "t/h", "w/h", "&epsilon;<sub>r</sub>", "Palace E",
                  "Chord V, raw", "Chord V", "Chord V without 3 worst pixels", "Best of 10 meshes"],
                 rows, "num")


def t_ratio():
    rows = [[f"{r['row']}", f"{r['t_over_h']:.3f}", f"{r['ratio_data']:.3f}",
             f"{r['ratio_direct']:.3f}" if r["ratio_direct"] is not None else "&ndash;",
             f"{r['ratio_fixed']:.3f}"] for r in T["designs"]]
    return table(["Row", "t/h", "Dataset", "Palace E", "Chord V"], rows, "num")


def t_mesh():
    order = [0, 4862, 3592, 8053, 52, 1, 500, 8317, 7185]
    th = {r["row"]: r["t_over_h"] for r in T["designs"]}
    rows = []
    def key(m):
        nums = [float(x) for x in re.findall(r"[\d.]+(?=-|t)", m["tag"].split("_p1_")[1])]
        return (not m["tag"].startswith("es_p1_uni"), nums)

    for m in sorted(T["mesh_scan"], key=key):
        vals = [m["scaled"].get(str(r), m["scaled"].get(r)) for r in order]
        rows.append([mesh_label(m["tag"])] + [pct(v) if v is not None else "&ndash;" for v in vals])
    return table(["Mesh"] + [f"{r}<br><span class=sub>t/h {th[r]:.3f}</span>" for r in order], rows, "num")


def t_solvers():
    names = {"es_p1_ref1": "Electrostatic, order 1", "es_p3_ref0.5": "Electrostatic, order 3, finer",
             "bm1GHz_p1_ref1": "Mode solve, 1 GHz", "bm10GHz_p1_ref1": "Mode solve, 10 GHz",
             "bm30GHz_p1_ref1": "Mode solve, 30 GHz", "bm10GHz_p3_ref0.5": "Mode solve, 10 GHz, order 3"}
    rows = []
    for tag, vals in T["solvers"].items():
        v = list(vals.values())
        if not v:
            continue
        v0 = vals.get("0", vals.get(0))
        rows.append([names.get(tag, tag), pct(sorted(v)[len(v) // 2]), pct(v0)])
    return table(["Solver (Palace E sampled at the grid)", "Median over 9 designs", "Row 0"], rows, "num")


def t_sig():
    lab = {"data": "Dataset", "direct_es_p1_uni1t": "Palace E, 1&thinsp;t mesh",
           "direct_es_p1_uni2t": "Palace E, 2&thinsp;t mesh", "chord_es_p1_uni1t": "Chord V, 1&thinsp;t mesh",
           "chord_es_p1_uni2t": "Chord V, 2&thinsp;t mesh"}
    rows = []
    for r, s in T["signatures"].items():
        for k in ("data", "direct_es_p1_uni1t", "direct_es_p1_uni2t", "chord_es_p1_uni1t", "chord_es_p1_uni2t"):
            if k in s:
                v = s[k]
                rows.append([f"{r}", lab[k], f"{v['ground_Ey']:.2f}", f"{v['ground_Ez']:.2f}",
                             f"{v['air_Ey']:.2f}", f"{v['air_Ez']:.2f}"])
    return table(["Row", "Source", "Rows 0&ndash;10, E<sub>y</sub>", "Rows 0&ndash;10, E<sub>z</sub>",
                  "Rows 34&ndash;53, E<sub>y</sub>", "Rows 34&ndash;53, E<sub>z</sub>"], rows, "num")


TABLES = {"designs": t_designs, "ratio": t_ratio, "mesh": t_mesh, "solvers": t_solvers, "sig": t_sig}


def fig(name, caption):
    svg = (FIGS / f"{name}.svg").read_text()
    svg = re.sub(r"<\?xml[^>]*\?>\s*", "", svg)
    svg = re.sub(r"<!DOCTYPE[^>]*>\s*", "", svg)
    svg = re.sub(r'(<svg[^>]*?)\swidth="[^"]*"\sheight="[^"]*"', r"\1", svg, count=1)
    cap = markdown.markdown(caption)[3:-4]
    return f'<figure id="fig-{name}"><div class="svgbox">{svg}</div><figcaption>{cap}</figcaption></figure>'


md_src = SRC.read_text()
tokens = {}


def stash(m):
    key = f"TOKEN{len(tokens)}X"
    kind, arg = m.group(1), m.group(2)
    if kind == "FIG":
        name, _, cap = arg.partition("|")
        tokens[key] = fig(name.strip(), cap.strip())
    else:
        tokens[key] = TABLES[arg.strip()]()
    return f"\n\n{key}\n\n"


md_src = re.sub(r"\{\{(FIG|TABLE):([^}]*)\}\}", stash, md_src)
md = markdown.Markdown(extensions=["extra", "toc", "sane_lists"], extension_configs={"toc": {"toc_depth": "2-3"}})
body = md.convert(md_src)
for k, v in tokens.items():
    body = body.replace(f"<p>{k}</p>", v).replace(k, v)

TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Palace Data Recreation</title>
<style>
:root { color-scheme: light dark; LIGHT_TOKENS }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { DARK_TOKENS } }
:root[data-theme="dark"] { DARK_TOKENS }
*, *::before, *::after { box-sizing: border-box; }
html { scroll-padding-top: 20px; -webkit-text-size-adjust: 100%; }
body { margin: 0; background: var(--bg); color: var(--text);
  font: 16px/1.65 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; }
.layout { display: grid; grid-template-columns: 250px minmax(0, 1fr); gap: 52px; max-width: 1240px; margin: 0 auto; padding: 0 32px; }
nav.side { position: sticky; top: 0; align-self: start; max-height: 100vh; overflow-y: auto; padding: 36px 4px 36px 0; font-size: 13.5px; }
nav.side summary { font-size: 11.5px; letter-spacing: .08em; text-transform: uppercase; color: var(--muted); font-weight: 650; cursor: pointer; margin-bottom: 8px; list-style: none; }
nav.side summary::-webkit-details-marker { display: none; }
nav.side ul { list-style: none; margin: 0; padding: 0; }
nav.side ul ul { padding-left: 10px; border-left: 1px solid var(--border); margin: 2px 0 10px 2px; }
nav.side a { display: block; padding: 3px 0 3px 8px; color: var(--muted); text-decoration: none; line-height: 1.35; border-radius: 4px; }
nav.side a:hover { color: var(--accent); background: var(--accent-soft); }
nav.side .toc > ul > li > a { color: var(--text); font-weight: 650; padding-left: 0; margin-top: 10px; }
main { min-width: 0; max-width: 860px; padding: 40px 0 120px; }
header.doc { padding-bottom: 22px; border-bottom: 1px solid var(--border); }
.eyebrow { font-size: 12px; letter-spacing: .09em; text-transform: uppercase; color: var(--muted); font-weight: 650; }
header.doc h1 { font-size: 36px; line-height: 1.12; margin: 8px 0 10px; letter-spacing: -.01em; }
header.doc p { color: var(--muted); margin: 0; }
.chips { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 14px; }
.chip { font-size: 12.5px; color: var(--accent); background: var(--accent-soft); padding: 3px 10px; border-radius: 999px; }
main h2 { font-size: 25px; line-height: 1.2; margin: 64px 0 12px; padding-top: 14px; border-top: 3px solid var(--rule); }
main h3 { font-size: 18px; line-height: 1.35; margin: 34px 0 8px; }
p, ul, ol { margin: 10px 0; }
li { margin: 5px 0; }
a { color: var(--accent); text-underline-offset: 2px; }
strong { font-weight: 650; }
code { font: .86em/1.4 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; background: var(--code-bg); padding: 1px 5px; border-radius: 4px; overflow-wrap: anywhere; }
pre { background: var(--code-bg); padding: 14px 16px; border-radius: 8px; overflow-x: auto; }
pre code { background: none; padding: 0; font-size: 12.8px; line-height: 1.55; overflow-wrap: normal; }
.table-wrap { overflow-x: auto; margin: 16px 0 22px; border: 1px solid var(--border); border-radius: 8px; background: var(--surface); }
table { border-collapse: collapse; width: 100%; font-size: 13.5px; line-height: 1.45; }
th, td { padding: 7px 11px; text-align: left; vertical-align: top; border-bottom: 1px solid var(--border); }
th { background: var(--th-bg); font-weight: 650; }
table.num td { font-variant-numeric: tabular-nums; }
tbody tr:last-child td { border-bottom: 0; }
.sub { font-weight: 400; color: var(--muted); font-size: 12px; }
figure { margin: 22px 0 30px; }
.svgbox { border: 1px solid var(--border); border-radius: 8px; background: #fff; padding: 8px; overflow-x: auto; }
.svgbox svg { display: block; width: 100%; height: auto; }
figcaption { font-size: 13.5px; color: var(--muted); margin-top: 8px; line-height: 1.5; }
.lead { font-size: 17px; }
.verdict { margin: 24px 0 8px; padding: 6px 22px 14px; background: var(--surface); border: 1px solid var(--border); border-left: 4px solid var(--accent); border-radius: 8px; }
footer { margin-top: 80px; padding-top: 16px; border-top: 1px solid var(--border); font-size: 13px; color: var(--muted); }
@media (max-width: 1000px) {
  .layout { grid-template-columns: minmax(0, 1fr); gap: 0; padding: 0 16px; }
  nav.side { position: static; max-height: none; padding: 16px 0 12px; border-bottom: 1px solid var(--border); }
  main { padding-top: 24px; }
  header.doc h1 { font-size: 30px; }
}
@media print { nav.side { display: none; } .layout { display: block; } }
</style>
</head>
<body>
<div class="layout">
<nav class="side" aria-label="Contents"><details id="tocbox" open><summary>Contents</summary>{{TOC}}</details></nav>
<main>
<header class="doc">
<div class="eyebrow">CS 6180 &middot; Final project &middot; Data provenance</div>
<h1>Recreating the staff data with Palace</h1>
<p>Prepared on {{BUILT}}, after Jiaji Huang's note that the fields come from Palace.</p>
<div class="chips"><span class="chip">Palace v0.19.0, built locally</span><span class="chip">9 designs re-simulated</span><span class="chip">Branch palace-rebuild</span></div>
</header>
{{BODY}}
<footer>Built {{BUILT}} by palace_regen/report/build_report.py from palace-recreation.md and the outputs of palace_regen/report_figs.py.</footer>
</main>
</div>
<script>
(function () {
  try { if (window.matchMedia("(max-width: 1000px)").matches) document.getElementById("tocbox").removeAttribute("open"); } catch (e) {}
})();
</script>
</body>
</html>
""".replace("LIGHT_TOKENS", LIGHT).replace("DARK_TOKENS", DARK)

page = TEMPLATE.replace("{{TOC}}", md.toc).replace("{{BODY}}", body).replace("{{BUILT}}", datetime.date.today().isoformat())
OUT.write_text(page)
print(f"wrote {OUT} ({len(page) / 1e6:.2f} MB)")
