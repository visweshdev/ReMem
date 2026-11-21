"""Regenerate the README figures (free, local): docs/img/demo.gif and docs/img/results.png.

    pip install pillow matplotlib
    python docs/make_figures.py

The GIF is a rendering of the real output of examples/basic_api.py (nothing is typed by hand). evidence.png reads its
numbers from results/real_knowledge_update/summary.json (real data, real LLM).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "img"

BG, FG, DIM, ACCENT, PROMPT = (22, 24, 29), (220, 223, 228), (130, 136, 148), (110, 200, 140), (120, 170, 255)


def _font(size: int):
    from PIL import ImageFont
    for name in ("consola.ttf", "DejaVuSansMono.ttf", "Menlo.ttc", "LiberationMono-Regular.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def demo_gif() -> Path:
    from PIL import Image, ImageDraw
    out = subprocess.run([sys.executable, str(ROOT / "examples" / "basic_api.py")], capture_output=True, text=True,
                         check=True, cwd=ROOT).stdout.rstrip().splitlines()
    cmd = "$ python examples/basic_api.py"
    font, pad, lh = _font(15), 18, 22
    cw = font.getbbox("M")[2]
    width = pad * 2 + cw * max(len(cmd), *(len(line) for line in out))
    height = pad * 2 + lh * (len(out) + 2)

    def frame(typed: str, n_lines: int):
        im = Image.new("RGB", (width, height), BG)
        d = ImageDraw.Draw(im)
        d.text((pad, pad), typed, font=font, fill=PROMPT)
        for i, line in enumerate(out[:n_lines]):
            colour = FG
            if line.strip().startswith(("INVALIDATED", "LOW_VALUE")):
                colour = DIM
            elif line.strip().startswith("ACTIVE") or line.startswith("3."):
                colour = ACCENT
            d.text((pad, pad + lh * (i + 2)), line, font=font, fill=colour)
        return im

    frames, durations = [], []
    for i in range(0, len(cmd) + 1, 2):
        frames.append(frame(cmd[:i], 0))
        durations.append(45)
    for n in range(1, len(out) + 1):
        frames.append(frame(cmd, n))
        durations.append(1300 if n <= 3 else 500)
    durations[-1] = 5000
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "demo.gif"
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=durations, loop=0, optimize=True)
    return path


def _load(name: str) -> dict:
    return json.loads((ROOT / "results" / name).read_text(encoding="utf-8"))


def _style(ax):
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.xaxis.grid(True, color="#e3e6ea", linewidth=0.8)
    ax.set_axisbelow(True)


def evidence_chart() -> Path:
    """Real data only: LongMemEval knowledge-update answer accuracy with a real LLM (paired difference vs plain RAG)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ku = _load("real_knowledge_update/summary.json")
    pa, summ = ku["paired_accuracy"], ku["summary"]
    rows = [("Extracted facts, similarity only", "facts_similarity"),
            ("Extracted facts + newest per key (rule)", "facts_newest_key"),
            ("Extracted facts + ReMem", "facts_remem")][::-1]
    fig, ax = plt.subplots(figsize=(8.4, 2.9), dpi=150)
    for i, (_lab, key) in enumerate(rows):
        d = pa[f"{key} - raw_similarity"]
        m, (lo, hi) = 100 * d["delta"], (100 * d["ci95"][0], 100 * d["ci95"][1])
        sig = lo > 0 or hi < 0
        col = ("#2f7d5b" if m > 0 else "#b5483b") if sig else "#8a929c"
        ax.plot([lo, hi], [i, i], color=col, linewidth=2.2, solid_capstyle="round")
        ax.plot(m, i, "o", color=col, markersize=7)
        ax.text(hi + 1.0, i, f"{m:+.1f} pp  ({100 * summ[key]['accuracy']:.0f}% correct)", va="center", fontsize=9,
                color=col)
    ax.axvline(0, color="#1d2228", linewidth=0.9)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows], fontsize=9)
    ax.set_xlim(-50, 50)
    base = 100 * summ["raw_similarity"]["accuracy"]
    ax.set_xlabel(f"Accuracy minus plain search ({base:.0f}% correct), points, 95% interval")
    fig.suptitle(f"REAL data: LongMemEval knowledge-update, {ku['n_questions']} questions, {ku['model']} answering",
                 x=0.02, ha="left", fontsize=11)
    _style(ax)
    fig.tight_layout()
    path = OUT / "evidence.png"
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    return path


if __name__ == "__main__":
    print(demo_gif())
    print(evidence_chart())
