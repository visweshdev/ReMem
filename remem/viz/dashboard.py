"""Generate a single-file HTML dashboard for a ReMem ledger (no server, no external assets).

    python -m remem.viz.dashboard ~/.remem/default/ledger.json -o memory.html
    python -m remem.viz.dashboard results/runs/distractor_seed0_C_remem/ledger.json -o dash.html

Shows every memory with its state, usefulness statistics and supersession links; click a row for its full
lifecycle and recent events. Filter by state or search text. Works offline in any browser.
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

from remem.ledger import MemoryLedger
from remem.viz.ledger_cli import lifecycle

TEMPLATE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>ReMem memory dashboard</title>
<style>
:root{--bg:#fff;--fg:#1b1f24;--mut:#5b6470;--line:#e3e6ea;--card:#f6f8fa;
--ACTIVE:#2e7d32;--LOW_VALUE:#a67c00;--COMPRESSED:#c45d00;--ARCHIVED:#b3261e;--INVALIDATED:#6b6b6b}
@media (prefers-color-scheme:dark){:root{--bg:#0f1115;--fg:#e6e8eb;--mut:#9aa3ad;--line:#2a2f37;--card:#171a20;
--ACTIVE:#66bb6a;--LOW_VALUE:#e0b84d;--COMPRESSED:#ff9d4d;--ARCHIVED:#ef6b62;--INVALIDATED:#a0a0a0}}
body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,-apple-system,Segoe UI,sans-serif}
main{max-width:1180px;margin:0 auto;padding:20px 16px}h1{font-size:20px;margin:0 0 4px}.mut{color:var(--mut)}
.tiles{display:flex;flex-wrap:wrap;gap:10px;margin:14px 0}.tile{background:var(--card);border:1px solid var(--line);
border-radius:8px;padding:8px 12px;min-width:110px}.tile b{display:block;font-size:20px}
.bar{display:flex;gap:8px;flex-wrap:wrap;margin:10px 0}input,select{background:var(--bg);color:var(--fg);
border:1px solid var(--line);border-radius:6px;padding:6px 8px;font:inherit}input{flex:1;min-width:180px}
.wrap{overflow-x:auto}table{border-collapse:collapse;width:100%}th,td{text-align:left;padding:6px 8px;
border-bottom:1px solid var(--line);vertical-align:top}th{position:sticky;top:0;background:var(--bg);cursor:pointer}
tr.row{cursor:pointer}tr.row:hover{background:var(--card)}td.num{text-align:right;font-variant-numeric:tabular-nums}
.st{font-weight:600;font-size:12px}pre{white-space:pre-wrap;background:var(--card);border:1px solid var(--line);
border-radius:6px;padding:8px;margin:6px 0;font-size:12px}
</style></head><body><main>
<h1>ReMem memory dashboard</h1><div class="mut" id="src"></div>
<div class="tiles" id="tiles"></div>
<div class="bar"><input id="q" placeholder="Search memory text or id"><select id="st"><option value="">All states</option></select>
<select id="sort"><option value="updated">Recently updated</option><option value="credited">Most credited</option>
<option value="wasted">Most wasted</option><option value="shown">Most shown</option></select></div>
<div class="wrap"><table><thead><tr><th>ID</th><th>State</th><th>Memory</th><th class="num">Shown</th>
<th class="num">Credited</th><th class="num">Wasted</th><th class="num">Utility</th></tr></thead><tbody id="tb"></tbody></table></div>
</main><script>
const D = __DATA__;
const $ = id => document.getElementById(id);
$("src").textContent = D.source + " · " + D.records.length + " memories · ledger step " + D.step;
const states = ["ACTIVE","LOW_VALUE","COMPRESSED","ARCHIVED","INVALIDATED"];
$("tiles").innerHTML = states.map(s => `<div class="tile"><span class="st" style="color:var(--${s})">${s}</span><b>${D.counts[s]||0}</b></div>`).join("");
states.forEach(s => { const o = document.createElement("option"); o.value = s; o.textContent = s; $("st").appendChild(o); });
const esc = t => String(t ?? "").replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
let open = null;
function render(){
  const q = $("q").value.toLowerCase(), st = $("st").value, key = $("sort").value;
  let rows = D.records.filter(r => (!st || r.state === st) && (!q || r.text.toLowerCase().includes(q) || r.id.includes(q)));
  rows.sort((a,b) => (b[key] ?? 0) - (a[key] ?? 0));
  $("tb").innerHTML = rows.slice(0, 2000).map(r => {
    const main = `<tr class="row" data-id="${r.id}"><td>${esc(r.id)}</td><td class="st" style="color:var(--${r.state})">${r.state}</td>
      <td>${esc(r.text)}</td><td class="num">${r.shown}</td><td class="num">${r.credited.toFixed(1)}</td>
      <td class="num">${r.wasted}</td><td class="num">${r.utility == null ? "–" : r.utility.toFixed(2)}</td></tr>`;
    if (open !== r.id) return main;
    return main + `<tr><td colspan="7"><div><b>Lifecycle</b><pre>${esc(r.lifecycle)}</pre>
      <b>Why last retrieved</b><pre>${esc(r.reason || "–")}</pre>
      <b>Supersession</b><pre>superseded_by: ${esc(r.superseded_by || "–")}\nsupersedes: ${esc((r.supersedes||[]).join(", ") || "–")}</pre>
      <b>Recent events</b><pre>${r.events.map(e => `step ${e.step}  ${e.kind}  ${e.detail}`).map(esc).join("\n")}</pre></div></td></tr>`;
  }).join("");
}
$("tb").addEventListener("click", e => { const tr = e.target.closest("tr.row"); if (!tr) return; open = open === tr.dataset.id ? null : tr.dataset.id; render(); });
["q","st","sort"].forEach(id => $(id).addEventListener("input", render));
render();
</script></body></html>"""


def build_html(ledger: MemoryLedger, source: str = "") -> str:
    recs = []
    for r in ledger.records.values():
        recs.append({
            "id": r.fact_id, "text": r.original_text, "state": r.state.value, "shown": r.n_retrieved,
            "credited": r.n_credited, "wasted": r.n_wasted, "utility": r.last_utility, "updated": r.last_updated_step,
            "superseded_by": r.superseded_by, "supersedes": r.supersedes, "reason": r.last_reason,
            "lifecycle": lifecycle(r, ledger), "events": [e.to_dict() for e in r.events[-15:]],
        })
    data = {"source": source, "step": ledger.step, "counts": ledger.state_counts(), "records": recs}
    payload = json.dumps(data).replace("</", "<\\/")      # keep memory text from closing the script tag
    return TEMPLATE.replace("__DATA__", payload)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="remem.viz.dashboard")
    ap.add_argument("ledger", type=Path, help="path to ledger.json (a ReMem store directory also works)")
    ap.add_argument("-o", "--out", type=Path, default=Path("remem_dashboard.html"))
    a = ap.parse_args(argv)
    path = a.ledger / "ledger.json" if a.ledger.is_dir() else a.ledger
    a.out.write_text(build_html(MemoryLedger.load(path), html.escape(str(path))), encoding="utf-8")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
