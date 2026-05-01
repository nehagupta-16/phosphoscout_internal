#!/usr/bin/env python3
"""
Static Site Builder for Mutation Reports Viewer.
Generates a GitHub Pages-compatible static site with 3D AlphaFold structure
visualization and clickable mutation reports. Loads env vars from .env,
fetches gene summaries via OpenAI when not cached, and outputs to docs/.
PDB structures are loaded from the AlphaFold public CDN at runtime.
"""

import os
import re
import json
import glob
import html as html_mod
import shutil
import urllib.parse
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
REPORTS_DIR = BASE_DIR / "phosphoscout_code" / "generated_artifacts" / "reports" / "html_reports"
STRUCTURES_DIR = BASE_DIR / "phosphoscout_code" / "structures"
UNIPROT_CACHE_FILE = Path(__file__).resolve().parent / "cache" / "uniprot_id_cache.json"
SUMMARIES_CACHE_FILE = Path(__file__).resolve().parent / "cache" / "gene_summaries_cache.json"
ENV_FILE = BASE_DIR / "phosphoscout_code" / ".env"
OUTPUT_DIR = Path(__file__).resolve().parent / "docs"


def load_dotenv(path):
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


load_dotenv(ENV_FILE)


# ── Report Parsing ────────────────────────────────────────────────────────────

def parse_reports():
    genes = {}
    scan_dirs = []
    for category in ["IDG", "oncogene"]:
        cat_dir = REPORTS_DIR / category
        if cat_dir.exists():
            scan_dirs.append((cat_dir, category))
    if not scan_dirs and REPORTS_DIR.exists():
        scan_dirs.append((REPORTS_DIR, "uncategorized"))

    for cat_dir, category in scan_dirs:
        for fname in sorted(os.listdir(cat_dir)):
            if not fname.endswith(".html"):
                continue
            m = re.match(r"^(.+?)_p[._]([A-Z])(\d+)([A-Z])(?:_(.+))?\.html$", fname)
            if not m:
                print(f"  ⚠ skipped: {fname}")
                continue
            gene = m.group(1)
            orig_aa, pos, new_aa = m.group(2), int(m.group(3)), m.group(4)
            notation = f"p.{orig_aa}{pos}{new_aa}"
            if gene not in genes:
                genes[gene] = {"mutations": [], "categories": set()}
            genes[gene]["categories"].add(category)
            rel_path = f"reports/{category}/{urllib.parse.quote(fname)}" if category != "uncategorized" else f"reports/{urllib.parse.quote(fname)}"
            genes[gene]["mutations"].append(
                {
                    "notation": notation,
                    "position": pos,
                    "orig_aa": orig_aa,
                    "new_aa": new_aa,
                    "filename": fname,
                    "category": category,
                    "report_url": rel_path,
                }
            )
    for g in genes.values():
        g["categories"] = sorted(g["categories"])
    return genes


# ── UniProt / structure helpers ───────────────────────────────────────────────

def load_uniprot_cache():
    if UNIPROT_CACHE_FILE.exists():
        return json.loads(UNIPROT_CACHE_FILE.read_text())
    return {}


def get_uniprot_id(gene, cache):
    return cache.get(f"{gene}|9606")


def find_pdb_file(uniprot_id):
    if not uniprot_id:
        return None
    exact = STRUCTURES_DIR / f"AF-{uniprot_id}-F1-model_v6.pdb"
    if exact.exists():
        return exact
    matches = glob.glob(str(STRUCTURES_DIR / f"AF-{uniprot_id}*-F1-model_v6.pdb"))
    return Path(matches[0]) if matches else None


def parse_pdb_residues(pdb_path):
    from Bio.PDB import PDBParser
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("prot", str(pdb_path))
    residues = set()
    for chain in structure[0]:
        for res in chain:
            if res.id[0] == " ":
                residues.add(res.id[1])
    return sorted(residues)


def build_structure_info(genes, uniprot_cache):
    info = {}
    for gene in genes:
        uid = get_uniprot_id(gene, uniprot_cache)
        pdb_path = find_pdb_file(uid) if uid else None
        if pdb_path:
            residues = parse_pdb_residues(pdb_path)
            info[gene] = {
                "uniprot_id": uid,
                "pdb_file": pdb_path.name,
                "res_min": min(residues) if residues else None,
                "res_max": max(residues) if residues else None,
                "res_count": len(residues),
                "residues": set(residues),
            }
        else:
            info[gene] = {
                "uniprot_id": uid,
                "pdb_file": None,
                "res_min": None,
                "res_max": None,
                "res_count": 0,
                "residues": set(),
            }
    return info


def generate_gene_summaries(genes):
    cache = {}
    if SUMMARIES_CACHE_FILE.exists():
        try:
            cache = json.loads(SUMMARIES_CACHE_FILE.read_text())
        except Exception:
            pass

    missing = [g for g in genes if g not in cache]
    if not missing:
        return cache

    print(f"│  Requesting OpenAI summaries for {len(missing)} genes …")
    try:
        from openai import OpenAI

        client = OpenAI()
        for i in range(0, len(missing), 25):
            batch = missing[i : i + 25]
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": (
                            "For each gene below, write exactly one sentence (≤20 words) "
                            "about its primary biological role. Return JSON: gene→summary.\n\n"
                            + ", ".join(batch)
                        ),
                    }
                ],
                response_format={"type": "json_object"},
                temperature=0.2,
            )
            result = json.loads(resp.choices[0].message.content)
            cache.update(result)
            print(f"│    batch {i // 25 + 1} done")
        SUMMARIES_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        SUMMARIES_CACHE_FILE.write_text(json.dumps(cache, indent=2))
    except Exception as e:
        print(f"│  ⚠ OpenAI error: {e}")
        for g in missing:
            cache.setdefault(g, "Summary unavailable.")
    return cache


# ── CSS (shared) ──────────────────────────────────────────────────────────────

CSS_COMMON = """
:root {
  --bg: #f1f5f9; --card: #ffffff; --text: #0f172a; --muted: #64748b;
  --primary: #2563eb; --primary-light: #3b82f6; --accent: #0ea5e9;
  --danger: #ef4444; --warning: #f59e0b; --success: #10b981;
  --border: #e2e8f0; --shadow: 0 1px 3px rgba(0,0,0,.06), 0 4px 12px rgba(0,0,0,.04);
  --radius: 14px;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body { background: var(--bg); color: var(--text); font-family: 'Inter', ui-sans-serif, system-ui, -apple-system, sans-serif; line-height: 1.6; }
a { color: var(--primary); text-decoration: none; }
a:hover { text-decoration: underline; }
.container { max-width: 1280px; margin: 0 auto; padding: 0 24px; }
header {
  background: linear-gradient(135deg, #1e3a5f 0%, #2563eb 50%, #0ea5e9 100%);
  color: white; padding: 32px 0; position: relative; overflow: hidden;
}
header::after {
  content: ''; position: absolute; inset: 0;
  background: url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%23ffffff' fill-opacity='0.05'%3E%3Cpath d='M36 34v-4h-2v4h-4v2h4v4h2v-4h4v-2h-4zm0-30V0h-2v4h-4v2h4v4h2V6h4V4h-4zM6 34v-4H4v4H0v2h4v4h2v-4h4v-2H6zM6 4V0H4v4H0v2h4v4h2V6h4V4H6z'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E");
}
header .container { position: relative; z-index: 1; }
header h1 { font-size: 2rem; font-weight: 800; letter-spacing: -0.03em; }
header .subtitle { opacity: 0.85; margin-top: 4px; font-size: 1rem; }
.badge {
  display: inline-block; padding: 2px 10px; border-radius: 999px;
  font-size: 0.75rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.04em;
}
.badge-IDG { background: #dbeafe; color: #1d4ed8; }
.badge-oncogene { background: #fce7f3; color: #be185d; }
.badge-uncategorized { background: #f1f5f9; color: #475569; }
"""


# ── Homepage HTML ─────────────────────────────────────────────────────────────

def render_homepage(genes, summaries, struct_info):
    sorted_genes = sorted(genes.keys())
    total_mut = sum(len(g["mutations"]) for g in genes.values())

    cards_html = ""
    for gene in sorted_genes:
        data = genes[gene]
        summary = html_mod.escape(summaries.get(gene, ""))
        n = len(data["mutations"])
        badges = " ".join(
            f'<span class="badge badge-{c}">{c}</span>' for c in data["categories"]
        )
        si = struct_info.get(gene, {})
        has_struct = si.get("pdb_file") is not None
        struct_icon = "✓" if has_struct else "✗"
        struct_cls = "yes" if has_struct else "no"
        positions = sorted({m["position"] for m in data["mutations"]})
        pos_preview = ", ".join(str(p) for p in positions[:6])
        if len(positions) > 6:
            pos_preview += f" … +{len(positions) - 6}"

        cards_html += f"""
        <a href="gene/{urllib.parse.quote(gene)}/index.html" class="gene-card" data-gene="{html_mod.escape(gene.lower())}">
          <div class="card-top">
            <h3>{html_mod.escape(gene)}</h3>
            <div class="badges">{badges}</div>
          </div>
          <p class="summary">{summary}</p>
          <div class="card-meta">
            <span class="meta-item"><strong>{n}</strong> mutation{"s" if n != 1 else ""}</span>
            <span class="meta-item struct-{struct_cls}">{struct_icon} structure</span>
          </div>
          <div class="positions">pos: {pos_preview}</div>
        </a>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Mutation Reports Viewer</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap" rel="stylesheet">
<style>
{CSS_COMMON}
main {{ padding: 28px 0 60px; }}
.search-bar {{
  margin: 0 0 24px; display: flex; gap: 12px; align-items: center;
}}
.search-bar input {{
  flex: 1; padding: 12px 18px; border: 1px solid var(--border); border-radius: 10px;
  font-size: 1rem; background: var(--card); outline: none; transition: border .15s;
}}
.search-bar input:focus {{ border-color: var(--primary); box-shadow: 0 0 0 3px rgba(37,99,235,.12); }}
.stats {{ font-size: .9rem; color: var(--muted); white-space: nowrap; }}
.gene-grid {{
  display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
  gap: 16px;
}}
.gene-card {{
  display: flex; flex-direction: column; background: var(--card);
  border: 1px solid var(--border); border-radius: var(--radius);
  padding: 18px 20px; text-decoration: none !important; color: var(--text);
  transition: transform .12s, box-shadow .12s; box-shadow: var(--shadow);
}}
.gene-card:hover {{
  transform: translateY(-3px);
  box-shadow: 0 6px 24px rgba(0,0,0,.09);
  border-color: var(--primary-light);
}}
.card-top {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px; }}
.card-top h3 {{ font-size: 1.2rem; font-weight: 800; letter-spacing: -.02em; }}
.summary {{ font-size: .88rem; color: var(--muted); margin-bottom: 10px; flex: 1; }}
.card-meta {{ display: flex; gap: 14px; font-size: .82rem; margin-bottom: 4px; }}
.meta-item {{ display: flex; align-items: center; gap: 4px; }}
.struct-yes {{ color: var(--success); }}
.struct-no {{ color: var(--muted); }}
.positions {{ font-size: .78rem; color: var(--muted); font-family: 'SF Mono', 'Fira Code', monospace; }}
</style>
</head>
<body>
<header>
  <div class="container">
    <h1>🧬 Mutation Reports Viewer</h1>
    <p class="subtitle">{len(genes)} genes · {total_mut} mutation reports · AlphaFold 3D structures</p>
  </div>
</header>
<main>
  <div class="container">
    <div class="search-bar">
      <input type="text" id="q" placeholder="Search genes …" autocomplete="off">
      <span class="stats">{len(genes)} genes</span>
    </div>
    <div class="gene-grid" id="grid">{cards_html}
    </div>
  </div>
</main>
<script>
document.getElementById('q').addEventListener('input', function() {{
  const q = this.value.toLowerCase();
  document.querySelectorAll('.gene-card').forEach(c => {{
    c.style.display = c.dataset.gene.includes(q) ? '' : 'none';
  }});
}});
</script>
</body>
</html>"""


# ── Gene Viewer Page (static version) ────────────────────────────────────────

def render_gene_page(gene, gene_data, struct_entry, summaries):
    mutations = gene_data["mutations"]
    summary = html_mod.escape(summaries.get(gene, ""))
    struct_residues = struct_entry.get("residues", set())
    has_struct = struct_entry.get("pdb_file") is not None
    uid = struct_entry.get("uniprot_id", "")
    pdb_filename = struct_entry.get("pdb_file", "")

    by_pos = {}
    for m in mutations:
        by_pos.setdefault(m["position"], []).append(m)

    positions_in = sorted(p for p in by_pos if p in struct_residues)
    positions_out = sorted(p for p in by_pos if p not in struct_residues)

    pos_reports = {}
    for pos, muts in by_pos.items():
        pos_reports[str(pos)] = [
            {"notation": m["notation"], "url": "../../" + m["report_url"], "category": m["category"]}
            for m in muts
        ]

    table_rows = ""
    for m in sorted(mutations, key=lambda x: x["position"]):
        in_s = "✓" if m["position"] in struct_residues else "—"
        table_rows += f"""
        <tr class="mut-row" data-pos="{m['position']}">
          <td class="pos-cell">{m['position']}</td>
          <td><strong>{html_mod.escape(m['notation'])}</strong></td>
          <td><span class="badge badge-{m['category']}">{m['category']}</span></td>
          <td class="centered">{in_s}</td>
          <td><a href="../../{m['report_url']}" target="_blank" class="rpt-link">View Report →</a></td>
        </tr>"""

    res_range = ""
    if has_struct:
        res_range = f"residues {struct_entry['res_min']}–{struct_entry['res_max']} ({struct_entry['res_count']} aa)"

    # Build the AlphaFold CDN URL for the PDB file
    alphafold_pdb_url = ""
    if pdb_filename:
        alphafold_pdb_url = f"https://alphafold.ebi.ac.uk/files/{pdb_filename}"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html_mod.escape(gene)} – Structure Viewer</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap" rel="stylesheet">
<script src="https://3Dmol.org/build/3Dmol-min.js"></script>
<style>
{CSS_COMMON}
.back {{ display: inline-flex; align-items: center; gap: 6px; color: rgba(255,255,255,.85); font-weight: 600; }}
.back:hover {{ color: white; text-decoration: none; }}
.gene-title {{ font-size: 2rem; font-weight: 800; margin: 4px 0 2px; }}
.meta-line {{ opacity: .8; font-size: .9rem; }}
.layout {{ display: grid; grid-template-columns: 1fr 380px; gap: 20px; padding: 24px 0 60px; }}
@media (max-width: 900px) {{ .layout {{ grid-template-columns: 1fr; }} }}
.viewer-wrap {{
  background: #0f172a; border-radius: var(--radius); overflow: hidden;
  box-shadow: var(--shadow); position: relative; min-height: 520px;
}}
#viewer {{ width: 100%; height: 520px; }}
.viewer-controls {{
  position: absolute; top: 12px; right: 12px; display: flex; gap: 8px; z-index: 5;
}}
.ctrl-btn {{
  padding: 6px 14px; border-radius: 8px; border: 1px solid rgba(255,255,255,.2);
  background: rgba(15,23,42,.7); color: white; font-size: .82rem; cursor: pointer;
  backdrop-filter: blur(6px); transition: background .15s;
}}
.ctrl-btn:hover {{ background: rgba(37,99,235,.6); }}
.ctrl-btn.active {{ background: var(--primary); border-color: var(--primary); }}
.sidebar {{
  display: flex; flex-direction: column; gap: 16px;
}}
.panel {{
  background: var(--card); border: 1px solid var(--border); border-radius: var(--radius);
  box-shadow: var(--shadow); overflow: hidden;
}}
.panel-header {{
  padding: 14px 18px; font-weight: 700; font-size: .95rem;
  border-bottom: 1px solid var(--border); background: #f8fafc;
}}
.legend {{ padding: 14px 18px; display: flex; flex-direction: column; gap: 8px; font-size: .85rem; }}
.legend-item {{ display: flex; align-items: center; gap: 10px; }}
.legend-dot {{
  width: 14px; height: 14px; border-radius: 50%; flex-shrink: 0;
}}
.mut-table {{ width: 100%; border-collapse: collapse; font-size: .85rem; }}
.mut-table th {{ text-align: left; padding: 10px 12px; background: #f8fafc; border-bottom: 1px solid var(--border);
  font-weight: 700; position: sticky; top: 0; }}
.mut-table td {{ padding: 9px 12px; border-bottom: 1px solid var(--border); }}
.mut-table tbody tr:hover {{ background: #f1f5f9; }}
.mut-table tbody tr {{ cursor: pointer; transition: background .1s; }}
.centered {{ text-align: center; }}
.pos-cell {{ font-family: 'SF Mono', 'Fira Code', monospace; font-weight: 600; }}
.rpt-link {{ font-weight: 600; font-size: .82rem; white-space: nowrap; }}
.no-struct {{ padding: 40px; text-align: center; color: var(--muted); }}
.mut-scroll {{ max-height: 420px; overflow-y: auto; }}
#report-popup {{
  position: fixed; z-index: 100; background: var(--card);
  border: 1px solid var(--border); border-radius: 10px; padding: 10px 0;
  box-shadow: 0 8px 30px rgba(0,0,0,.18); min-width: 220px;
  animation: popIn .12s ease-out;
}}
@keyframes popIn {{ from {{ opacity: 0; transform: scale(.95); }} to {{ opacity: 1; transform: scale(1); }} }}
#report-popup a {{
  display: block; padding: 8px 18px; font-size: .88rem; font-weight: 600; transition: background .1s;
}}
#report-popup a:hover {{ background: #f1f5f9; text-decoration: none; }}
#report-popup .popup-header {{
  padding: 6px 18px 8px; font-size: .78rem; color: var(--muted);
  font-weight: 700; text-transform: uppercase; letter-spacing: .04em;
  border-bottom: 1px solid var(--border); margin-bottom: 4px;
}}
#hover-label {{
  position: absolute; z-index: 10; pointer-events: none;
  background: rgba(15,23,42,.88); color: white; padding: 5px 12px;
  border-radius: 8px; font-size: .82rem; font-weight: 600;
  backdrop-filter: blur(4px); display: none; white-space: nowrap;
}}
.loading-overlay {{
  position: absolute; inset: 0; display: flex; align-items: center; justify-content: center;
  background: rgba(15,23,42,.85); color: white; font-size: .95rem; z-index: 6;
}}
.loading-overlay.hidden {{ display: none; }}
</style>
</head>
<body>
<header>
  <div class="container">
    <a href="../../index.html" class="back">← All Genes</a>
    <div class="gene-title">{html_mod.escape(gene)}</div>
    <p class="meta-line">
      {summary}
      {"&nbsp;·&nbsp; UniProt: " + html_mod.escape(uid) if uid else ""}
      {"&nbsp;·&nbsp; " + res_range if res_range else ""}
    </p>
  </div>
</header>
<main>
  <div class="container">
    <div class="layout">
      <div>
        <div class="viewer-wrap">
          <div class="viewer-controls">
            <button class="ctrl-btn active" id="btn-labels" onclick="toggleLabels()">Labels</button>
            <button class="ctrl-btn" id="btn-surface" onclick="toggleSurface()">Surface</button>
            <button class="ctrl-btn" onclick="resetView()">Reset</button>
          </div>
          <div id="hover-label"></div>
          <div id="viewer">
            {"" if has_struct else '<div class="no-struct">No AlphaFold structure available for this gene.</div>'}
            {'<div class="loading-overlay" id="loading">Loading structure from AlphaFold…</div>' if has_struct else ""}
          </div>
        </div>
      </div>
      <div class="sidebar">
        <div class="panel">
          <div class="panel-header">Legend</div>
          <div class="legend">
            <div class="legend-item">
              <span class="legend-dot" style="background:linear-gradient(135deg,#ef4444,#f97316)"></span>
              Mutation site (in structure) – click to open report
            </div>
            <div class="legend-item">
              <span class="legend-dot" style="background:#94a3b8"></span>
              Mutation site (outside structure range)
            </div>
            <div class="legend-item">
              <span class="legend-dot" style="background:linear-gradient(135deg,#6366f1,#06b6d4)"></span>
              Protein backbone (spectrum coloring)
            </div>
          </div>
        </div>
        <div class="panel" style="flex:1; display:flex; flex-direction:column; min-height:0;">
          <div class="panel-header">Mutations ({len(mutations)})</div>
          <div class="mut-scroll">
            <table class="mut-table">
              <thead><tr><th>Pos</th><th>Change</th><th>Set</th><th>3D</th><th>Report</th></tr></thead>
              <tbody>{table_rows}</tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  </div>
</main>

<script>
const GENE = {json.dumps(gene)};
const POSITIONS_IN  = {json.dumps(positions_in)};
const POSITIONS_OUT = {json.dumps(positions_out)};
const ALL_POSITIONS = {json.dumps(sorted(by_pos.keys()))};
const POS_REPORTS   = {json.dumps(pos_reports)};
const HAS_STRUCT    = {json.dumps(has_struct)};
const PDB_URL       = {json.dumps(alphafold_pdb_url)};

let viewer, labelsVisible = true, surfaceId = null, surfaceOn = false;
const labels = [];

function closePopup() {{
  const p = document.getElementById('report-popup');
  if (p) p.remove();
}}
document.addEventListener('click', e => {{
  if (!e.target.closest('#report-popup')) closePopup();
}});

function showPopup(x, y, pos) {{
  closePopup();
  const reports = POS_REPORTS[String(pos)];
  if (!reports || !reports.length) return;
  if (reports.length === 1) {{
    window.open(reports[0].url, '_blank');
    return;
  }}
  const div = document.createElement('div');
  div.id = 'report-popup';
  div.innerHTML = '<div class="popup-header">Position ' + pos + '</div>';
  reports.forEach(r => {{
    const a = document.createElement('a');
    a.href = r.url;
    a.target = '_blank';
    a.innerHTML = '<span class="badge badge-' + r.category + '">' + r.category + '</span> ' + r.notation;
    div.appendChild(a);
  }});
  div.style.left = Math.min(x, window.innerWidth - 260) + 'px';
  div.style.top  = Math.min(y, window.innerHeight - 200) + 'px';
  document.body.appendChild(div);
}}

function toggleLabels() {{
  labelsVisible = !labelsVisible;
  labels.forEach(l => viewer.removeLabel(l));
  labels.length = 0;
  if (labelsVisible) addLabels();
  viewer.render();
  document.getElementById('btn-labels').classList.toggle('active', labelsVisible);
}}

function toggleSurface() {{
  if (surfaceOn && surfaceId !== null) {{
    viewer.removeSurface(surfaceId);
    surfaceId = null;
  }} else {{
    surfaceId = viewer.addSurface($3Dmol.SurfaceType.VDW,
      {{ opacity: 0.15, color: 'white' }}, {{}});
  }}
  surfaceOn = !surfaceOn;
  viewer.render();
  document.getElementById('btn-surface').classList.toggle('active', surfaceOn);
}}

function resetView() {{
  viewer.zoomTo();
  viewer.render();
}}

function addLabels() {{
  POSITIONS_IN.forEach(pos => {{
    const reports = POS_REPORTS[String(pos)] || [];
    const txt = reports.map(r => r.notation).join(', ');
    const lab = viewer.addLabel(txt, {{
      fontSize: 11, fontColor: 'white',
      backgroundColor: 'rgba(239,68,68,0.85)',
      borderRadius: 6, padding: 3,
      showBackground: true, backgroundOpacity: 0.85,
      alignment: 'bottomCenter', inFront: true,
    }}, {{ resi: pos, atom: 'CA' }});
    labels.push(lab);
  }});
}}

async function initViewer() {{
  if (!HAS_STRUCT) return;
  const container = document.getElementById('viewer');
  viewer = $3Dmol.createViewer(container, {{
    backgroundColor: '#0f172a', antialias: true,
  }});

  try {{
    const resp = await fetch(PDB_URL);
    if (!resp.ok) throw new Error('HTTP ' + resp.status);
    const pdb = await resp.text();
    document.getElementById('loading').classList.add('hidden');

    viewer.addModel(pdb, 'pdb');

    viewer.setStyle({{}}, {{ cartoon: {{ color: 'spectrum', opacity: 0.88 }} }});

    POSITIONS_IN.forEach(pos => {{
      viewer.addStyle({{ resi: pos }}, {{
        stick: {{ colorscheme: 'orangeCarbon', radius: 0.18 }},
      }});
      viewer.addStyle({{ resi: pos }}, {{
        sphere: {{ radius: 0.55, color: '#ef4444', opacity: 0.45 }},
      }});
    }});

    addLabels();

    viewer.setClickable({{ resi: POSITIONS_IN }}, true, function(atom, v, event) {{
      showPopup(event.clientX, event.clientY, atom.resi);
    }});

    const hoverEl = document.getElementById('hover-label');
    viewer.setHoverable({{ resi: POSITIONS_IN }}, true,
      function(atom, v, event) {{
        const reports = POS_REPORTS[String(atom.resi)] || [];
        hoverEl.textContent = reports.map(r => r.notation).join(' / ') + '  — click to view';
        hoverEl.style.display = 'block';
        const rect = container.getBoundingClientRect();
        hoverEl.style.left = (event.clientX - rect.left + 14) + 'px';
        hoverEl.style.top  = (event.clientY - rect.top  - 10) + 'px';
      }},
      function() {{ hoverEl.style.display = 'none'; }}
    );

    viewer.zoomTo();
    viewer.render();
    viewer.zoom(0.9);
    viewer.render();
  }} catch(e) {{
    document.getElementById('loading').textContent = 'Failed to load structure: ' + e.message;
  }}
}}

document.querySelectorAll('.mut-row').forEach(row => {{
  row.addEventListener('click', function(e) {{
    if (e.target.closest('.rpt-link')) return;
    const pos = parseInt(this.dataset.pos);
    if (viewer && POSITIONS_IN.includes(pos)) {{
      viewer.zoomTo({{ resi: pos }}, 500);
      viewer.render();
    }}
  }});
}});

initViewer();
</script>
</body>
</html>"""


# ── Build ─────────────────────────────────────────────────────────────────────

def main():
    print("┌─ Static Site Builder ─────────────────────────┐")

    # Clean output (preserve .git so the repo isn't destroyed)
    if OUTPUT_DIR.exists():
        for item in OUTPUT_DIR.iterdir():
            if item.name == ".git":
                continue
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()
    else:
        OUTPUT_DIR.mkdir(parents=True)

    # Parse data
    print("│  Parsing reports …")
    genes = parse_reports()
    print(f"│  {len(genes)} genes, {sum(len(g['mutations']) for g in genes.values())} mutations")

    print("│  Loading UniProt cache …")
    uniprot_cache = load_uniprot_cache()

    print("│  Parsing AlphaFold PDB structures (Biopython) …")
    struct_info = build_structure_info(genes, uniprot_cache)
    with_struct = sum(1 for v in struct_info.values() if v["pdb_file"])
    print(f"│  {with_struct}/{len(genes)} genes have structures")

    print("│  Generating gene summaries (OpenAI) …")
    summaries = generate_gene_summaries(genes)

    # Generate homepage
    print("│  Generating homepage …")
    (OUTPUT_DIR / "index.html").write_text(render_homepage(genes, summaries, struct_info), encoding="utf-8")

    # Generate gene pages
    print(f"│  Generating {len(genes)} gene pages …")
    for gene in sorted(genes.keys()):
        gene_dir = OUTPUT_DIR / "gene" / gene
        gene_dir.mkdir(parents=True, exist_ok=True)
        html_content = render_gene_page(gene, genes[gene], struct_info.get(gene, {}), summaries)
        (gene_dir / "index.html").write_text(html_content, encoding="utf-8")

    # Copy report HTML files
    print("│  Copying report HTML files …")
    copied = 0
    has_category_dirs = False
    for category in ["IDG", "oncogene"]:
        cat_src = REPORTS_DIR / category
        cat_dst = OUTPUT_DIR / "reports" / category
        if not cat_src.exists():
            continue
        has_category_dirs = True
        cat_dst.mkdir(parents=True, exist_ok=True)
        for fname in os.listdir(cat_src):
            if fname.endswith(".html"):
                shutil.copy2(cat_src / fname, cat_dst / fname)
                copied += 1
    if not has_category_dirs and REPORTS_DIR.exists():
        dst = OUTPUT_DIR / "reports"
        dst.mkdir(parents=True, exist_ok=True)
        for fname in os.listdir(REPORTS_DIR):
            src_file = REPORTS_DIR / fname
            if src_file.is_file() and fname.endswith(".html"):
                shutil.copy2(src_file, dst / fname)
                copied += 1
    print(f"│  Copied {copied} report files")

    # Create .nojekyll (GitHub Pages needs this for folders starting with _)
    (OUTPUT_DIR / ".nojekyll").touch()

    total_files = sum(1 for _ in OUTPUT_DIR.rglob("*.html"))
    print(f"│  Done! {total_files} HTML files in {OUTPUT_DIR}")
    print("└───────────────────────────────────────────────┘")
    print(f"\nTo deploy: push the 'docs/' folder to GitHub and enable Pages from Settings > Pages > Source: docs/")


if __name__ == "__main__":
    main()
