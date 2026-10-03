"""Build the GitHub Pages site (docs/) from the prototype page.

The public site does NOT re-host METR's run data. docs/data.json keeps only this project's outputs (IRT item bank,
model parameters, back-test and design results, task audit statistics). The per-agent success counts that replay mode
needs are recomputed in the browser from METR's public runs file, fetched at a pinned commit from METR's repository.

Usage (from the repo root, after `python prototype/export.py`):
    python prototype/build_pages.py
"""

import json
import pathlib
import re

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
DOCS = ROOT / "docs"
METR_SHA = "52cb829c7a2efb2d659285c4b1768d191d97f8d2"
METR_RUNS = (f"https://raw.githubusercontent.com/METR/eval-analysis-public/{METR_SHA}"
             "/reports/time-horizon-1-1/data/raw/runs.jsonl")
REPO_URL = "https://github.com/{owner}/horizon-irt"

LOADER = r'''
// Replay data is not re-hosted: rebuild each agent's per-task success counts from METR's public runs file.
async function loadRates(D) {
  const res = await fetch(D.meta.metr_runs_url);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  const text = await res.text();
  const idx = new Map(D.tasks.id.map((id, t) => [id, t]));
  const rates = {};
  for (const a of D.agents) rates[a.name] = {};
  for (const line of text.split("\n")) {
    if (!line) continue;
    const r = JSON.parse(line);
    const byAgent = rates[r.alias];
    const t = idx.get(r.task_id);
    if (!byAgent || t === undefined) continue;
    const cell = byAgent[t] || (byAgent[t] = [0, 0]);
    cell[0] += r.score_binarized ? 1 : 0;
    cell[1] += 1;
  }
  return rates;
}
'''


def main(owner="OWNER"):
    data = json.loads((HERE / "web" / "data.json").read_text())
    data.pop("rates", None)
    data["meta"]["metr_runs_url"] = METR_RUNS
    data["meta"]["metr_sha"] = METR_SHA
    DOCS.mkdir(exist_ok=True)
    (DOCS / "data.json").write_text(json.dumps(data, separators=(",", ":")))

    src = (HERE / "web" / "index.html").read_text()
    head_end = src.index("</style>") + len("</style>")
    head, body = src[:head_end], src[head_end:]
    repo = REPO_URL.format(owner=owner)

    old_load = 'try { D = await (await fetch("data.json")).json(); }'
    assert old_load in body
    body = body.replace(old_load, '''try {
    D = await (await fetch("data.json")).json();
    $("loading").textContent = "Loading METR's public run data (15 MB) from github.com/METR/eval-analysis-public…";
    try { D.rates = await loadRates(D); }
    catch (e) { D.rates = null; D.ratesError = String(e); }
  }''')
    body = body.replace("// ---------- session ----------", LOADER + "\n// ---------- session ----------")
    # Without replay data the page still works in hypothetical-agent mode.
    old_sel = '''  sel.value = ags.find(a => a.name.startsWith("Claude Opus 4.5")) ? ags.find(a => a.name.startsWith("Claude Opus 4.5")).name : ags[0].name;'''
    assert old_sel in body
    body = body.replace(old_sel, old_sel + '''
  if (!D.rates) {
    sel.querySelectorAll('optgroup[label^="Replay"] option').forEach(o => o.disabled = true);
    sel.value = "__hypo";
    $("agentNote").textContent = `Replay is unavailable because METR's run data could not be loaded (${D.ratesError}). Hypothetical agents still work.`;
  }''')
    footer = f'''
<footer class="wrap" style="margin-top:28px;border-top:1px solid var(--rule);padding-block:14px 0;color:var(--muted);font-size:12.5px;display:flex;flex-direction:column;gap:6px">
  <p>Code, decision logs and the full study: <a href="{repo}">{repo.replace("https://", "")}</a> (MIT licence for code).</p>
  <p>Data: METR, <a href="https://github.com/METR/eval-analysis-public">eval-analysis-public</a> (commit <span class="mono">{METR_SHA[:7]}</span>), loaded directly from METR's repository at view time and not re-hosted here. Cite <a href="https://arxiv.org/abs/2503.14499">Kwa et al. (2025)</a>. The difficulty anchor follows <a href="https://arxiv.org/abs/2602.07267">BRIDGE (Liu et al., 2026)</a>. This is an independent project, not affiliated with or endorsed by METR.</p>
</footer>'''
    body = body.replace('<div id="tip" hidden></div>', footer + '\n<div id="tip" hidden></div>')
    html = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
            '<meta name="description" content="Adaptive, cost-aware measurement of AI agents\' 50% time horizons on METR\'s task suite.">\n'
            + head.replace("<title>HorizonCAT</title>", "<title>HorizonCAT</title>")
            + '\n<style>html,body{margin:0}</style>\n</head>\n<body>\n' + body + '\n</body>\n</html>\n')
    (DOCS / "index.html").write_text(html)
    (DOCS / ".nojekyll").write_text("")
    print("wrote", DOCS / "index.html", "and", DOCS / "data.json", f"({(DOCS / 'data.json').stat().st_size / 1e3:.0f} kB)")


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else "OWNER")
