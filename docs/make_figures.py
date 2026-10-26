"""Regenerate the README figures (free, local): docs/img/demo.gif and docs/img/results.png.

    pip install pillow matplotlib
    python docs/make_figures.py

The GIF is a rendering of the real output of examples/basic_api.py (nothing is typed by hand); the chart reads the
numbers from results/summary.json when present, otherwise uses the 15-seed values reported in the README.
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
        frames.append(frame(cmd[:i], 0)); durations.append(45)
    for n in range(1, len(out) + 1):
        frames.append(frame(cmd, n)); durations.append(1300 if n <= 3 else 500)
    durations[-1] = 5000
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "demo.gif"
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=durations, loop=0, optimize=True)
    return path


def results_chart() -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # mean task success over 15 test seeds, 7 core families, reader "first" (README section 10)
    data = [("Similarity only", 62.6), ("RMM-style reranker", 63.1), ("Static score", 65.9), ("ReMem", 76.4)]
    summary = ROOT / "results" / "summary.json"
    if summary.exists():
        try:
            s = json.loads(summary.read_text(encoding="utf-8"))
            names = {"A_vector": "Similarity only", "RMM_style": "RMM-style reranker", "B_static": "Static score",
                     "C_remem": "ReMem"}
            got = {names[k]: 100 * v["task_success"] for k, v in s.get("headline", {}).items() if k in names}
            if len(got) == 4:
                data = sorted(got.items(), key=lambda kv: kv[1])
        except (ValueError, KeyError, TypeError):
            pass
    labels, vals = zip(*data)
    fig, ax = plt.subplots(figsize=(7.2, 2.9), dpi=150)
    colours = ["#2f7d5b" if lab == "ReMem" else "#a9b0ba" for lab in labels]
    bars = ax.barh(labels, vals, color=colours, height=0.62)
    for b, v in zip(bars, vals):
        ax.text(v + 0.6, b.get_y() + b.get_height() / 2, f"{v:.1f}%", va="center", fontsize=10,
                color="#1d2228", fontweight="bold" if b.get_facecolor()[1] < 0.6 else "normal")
    ax.set_xlim(0, 100)
    ax.set_xlabel("Task success (%), synthetic benchmark, mean of 15 test seeds")
    ax.set_title("ReMem vs retrieval baselines (same memories, same agent)", loc="left", fontsize=11)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.xaxis.grid(True, color="#e3e6ea", linewidth=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "results.png"
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    return path


def _style(ax):
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.xaxis.grid(True, color="#e3e6ea", linewidth=0.8)
    ax.set_axisbelow(True)


def family_chart() -> Path:
    """Per-family task success, similarity only vs ReMem (README section 10, reader `first`, 15 test seeds)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fams = [("state change", 46.6, 82.0), ("contradiction", 52.6, 82.4), ("tight token budget", 41.0, 49.5),
            ("rare but critical", 74.9, 81.6), ("long horizon", 73.5, 79.9), ("distractors", 73.5, 79.3),
            ("useless-but-frequent", 76.0, 79.9), ("control: nothing recurs", 68.8, 68.7)]
    fams = fams[::-1]
    y = np.arange(len(fams))
    fig, ax = plt.subplots(figsize=(7.2, 4.0), dpi=150)
    h = 0.38
    ax.barh(y + h / 2, [f[1] for f in fams], height=h, color="#a9b0ba", label="Similarity only")
    ax.barh(y - h / 2, [f[2] for f in fams], height=h, color="#2f7d5b", label="ReMem")
    for i, (_, a, c) in enumerate(fams):
        ax.text(c + 0.8, i - h / 2, f"{c:.0f}", va="center", fontsize=8.5, color="#1d2228", fontweight="bold")
        ax.text(a + 0.8, i + h / 2, f"{a:.0f}", va="center", fontsize=8.5, color="#5b636e")
    ax.set_yticks(y, [f[0] for f in fams])
    ax.set_xlim(0, 100)
    ax.set_xlabel("Task success (%), mean of 15 test seeds")
    ax.set_title("Where the gain comes from: facts that change", loc="left", fontsize=11, pad=22)
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2, frameon=False, fontsize=9)
    _style(ax)
    fig.tight_layout()
    path = OUT / "families.png"
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    return path


def evidence_chart() -> Path:
    """ReMem minus similarity-only, across every evaluation setting, with 95% intervals where they exist."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [  # (label, delta in pp, half-width of 95% CI or None)
        ("Synthetic, simulated reader (hash embedder)", 13.8, 1.8),
        ("Synthetic, real embeddings (MiniLM)", 10.2, 1.4),
        ("Synthetic, timestamp-aware simulated reader", 5.6, 1.2),
        ("Real LLM (qwen2.5:3b), distractor tasks", 0.4, 4.7),
        ("Real LLM (qwen2.5:3b), contradiction tasks", -2.9, 23.3),
        ("LongMemEval retrieval (no repeated questions)", 1.3, None),
    ][::-1]
    fig, ax = plt.subplots(figsize=(8.4, 3.5), dpi=150)
    for i, (lab, d, ci) in enumerate(rows):
        sig = ci is not None and d - ci > 0
        col = "#2f7d5b" if sig else "#8a929c"
        if ci is not None:
            ax.plot([max(d - ci, -15), min(d + ci, 25)], [i, i], color=col, linewidth=2.2, solid_capstyle="round")
        ax.plot(d, i, "o", color=col, markersize=7)
        ax.text(min(d + (ci or 0), 25) + 0.8, i, f"{d:+.1f}", va="center", fontsize=9, color=col,
                fontweight="bold" if sig else "normal")
    ax.axvline(0, color="#1d2228", linewidth=0.9)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows], fontsize=9)
    ax.set_xlim(-15, 29)
    ax.set_xlabel("ReMem minus similarity-only, percentage points (bars: 95% interval)")
    fig.suptitle("Honest scorecard: strong in simulation, not yet with a real LLM", x=0.02, ha="left", fontsize=11)
    _style(ax)
    fig.tight_layout()
    path = OUT / "evidence.png"
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    return path


if __name__ == "__main__":
    print(demo_gif())
    print(results_chart())
    print(family_chart())
    print(evidence_chart())
