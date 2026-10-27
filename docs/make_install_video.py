"""Render docs/video/install_and_run.mp4: a captioned terminal walkthrough of installing and running ReMem.

    python docs/make_install_video.py            # needs pillow and ffmpeg on PATH

Every command output shown was captured from a real run (fresh `git clone` of the GitHub repo, `pip install -e .`,
the demo, a small script, the test suite) on Windows. Long pip output is shortened, and the video says so.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "video" / "install_and_run.mp4"
W, H, FPS = 1920, 1080, 15

BG = (14, 17, 22)
WIN = (22, 25, 31)
BAR = (34, 38, 46)
FG = (220, 223, 228)
DIM = (128, 135, 148)
GREEN = (110, 200, 140)
BLUE = (120, 170, 255)
YELLOW = (230, 200, 120)
RED = (235, 120, 110)
ACCENT = (47, 125, 91)


def font(name: str, size: int):
    try:
        return ImageFont.truetype(name, size)
    except OSError:
        return ImageFont.load_default()


MONO = font("consola.ttf", 20)
MONO_B = font("consolab.ttf", 20)
UI = font("segoeui.ttf", 30)
UI_B = font("segoeuib.ttf", 34)
UI_S = font("segoeui.ttf", 24)
TITLE = font("segoeuib.ttf", 92)
SUB = font("segoeui.ttf", 38)
LH = 30                                     # terminal line height
TX, TY, TW, TH = 70, 230, W - 140, 760      # terminal window box
MAX_LINES = (TH - 70) // LH

# ── script: (kind, payload) ───────────────────────────────────────────────────
# kinds: title, step (number, caption, detail), cmd (prompt, text), out (lines with colour), pause (seconds),
#        code (filename, lines), end
P0 = r"PS C:\Projects>"
P1 = r"PS C:\Projects\ReMem>"


def colour_demo(line: str):
    s = line.strip()
    if s.startswith(("INVALIDATED", "LOW_VALUE")):
        return DIM
    if s.startswith("ACTIVE") or line.startswith("3."):
        return GREEN
    return FG


SCRIPT = [
    ("title", None),
    ("step", (1, "Check that Python 3.10+ and Git are installed",
              "Get them free from python.org and git-scm.com if these commands fail")),
    ("cmd", (P0, "python --version")),
    ("out", [("Python 3.14.3", FG)]),
    ("cmd", (P0, "git --version")),
    ("out", [("git version 2.55.0.windows.5", FG)]),
    ("pause", 2.0),
    ("step", (2, "Download ReMem from GitHub", "This creates a ReMem folder with all the code")),
    ("cmd", (P0, "git clone https://github.com/visweshdev/ReMem.git")),
    ("out", [("Cloning into 'ReMem'...", FG)]),
    ("cmd", (P0, "cd ReMem")),
    ("pause", 1.5),
    ("step", (3, "Install it", "Only needs numpy, pyyaml and matplotlib; takes about a minute")),
    ("cmd", (P1, "pip install -e .")),
    ("out", [("Obtaining file:///C:/Projects/ReMem", FG),
             ("  Installing build dependencies ... done", FG),
             ("Collecting numpy>=1.26 (from remem==0.1.0)", FG),
             ("Collecting pyyaml>=6.0 (from remem==0.1.0)", FG),
             ("Collecting matplotlib>=3.7 (from remem==0.1.0)", FG),
             ("   ... (output shortened) ...", DIM),
             ("Successfully installed contourpy-1.4.0 cycler-0.12.1 fonttools-4.67.0 kiwisolver-1.5.1 "
              "matplotlib-3.11.2 numpy-2.5.4", GREEN),
             ("packaging-26.3 pillow-13.0.0 pyparsing-3.3.3 python-dateutil-3.0.0 pyyaml-6.0.3 remem-0.1.0 "
              "six-1.17.0 tzdata-2026.5", GREEN)]),
    ("pause", 2.5),
    ("step", (4, "Run the demo", "Watch the memory learn: the outdated fact is retired, the useless note demoted")),
    ("cmd", (P1, "python examples/basic_api.py")),
    ("out", "DEMO"),
    ("pause", 6.0),
    ("step", (5, "Use it in your own code", "Three calls: add facts, search before answering, feedback after")),
    ("code", ("my_app.py", [
        "from remem import ReMem",
        "",
        'mem = ReMem("my_memory")            # a folder; saved automatically',
        'mem.add("Our server is in east-1.", entity="server", attribute="region")',
        'mem.add("Our server moved to west-2.", entity="server", attribute="region")',
        "",
        'hits = mem.search("Where is our server?")',
        "print(hits.items[0].text)",
        "mem.feedback(hits.search_id, success=True, used_ids=[hits.items[0].id])",
        'print(mem.stats()["states"])',
    ])),
    ("pause", 5.0),
    ("cmd", (P1, "python my_app.py")),
    ("out", [("Our server moved to west-2.", GREEN),
             ("{'ACTIVE': 1, 'LOW_VALUE': 0, 'COMPRESSED': 0, 'ARCHIVED': 0, 'INVALIDATED': 1}", FG)]),
    ("pause", 4.0),
    ("step", (6, "Optional: check that everything works", "Runs the project's automatic tests")),
    ("cmd", (P1, "pip install pytest")),
    ("out", [("   ... (pip output hidden) ...", DIM)]),
    ("cmd", (P1, "python -m pytest tests -q")),
    ("out", [("....sss........................ss....................................... [ 84%]", GREEN),
             (".............                                                            [100%]", GREEN),
             ("80 passed, 5 skipped in 3.10s", GREEN),
             ("(the 5 skipped tests need optional extras such as the MCP server or FAISS)", DIM)]),
    ("pause", 3.5),
    ("end", None),
]


def demo_lines():
    lines = subprocess.run(["python", str(ROOT / "examples" / "basic_api.py")], capture_output=True, text=True,
                           check=True, cwd=ROOT).stdout.rstrip().splitlines()
    return [(ln, colour_demo(ln)) for ln in lines]


# ── drawing ───────────────────────────────────────────────────────────────────
def base(step: tuple | None, progress: float) -> Image.Image:
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)
    d.text((70, 44), "ReMem", font=UI_B, fill=(255, 255, 255))
    d.text((70 + d.textlength("ReMem  ", font=UI_B), 50), "install & run", font=UI, fill=DIM)
    if step:
        n, cap, det = step
        d.rounded_rectangle((70, 110, 70 + 58, 168), 14, fill=ACCENT)
        d.text((99, 139), str(n), font=UI_B, fill=(255, 255, 255), anchor="mm")
        d.text((150, 104), cap, font=UI_B, fill=(255, 255, 255))
        d.text((152, 150), det, font=UI_S, fill=DIM)
    d.rounded_rectangle((TX, TY, TX + TW, TY + TH), 16, fill=WIN)
    d.rounded_rectangle((TX, TY, TX + TW, TY + 46), 16, fill=BAR)
    d.rectangle((TX, TY + 30, TX + TW, TY + 46), fill=BAR)
    for i, c in enumerate([(237, 106, 94), (245, 191, 79), (98, 197, 84)]):
        d.ellipse((TX + 22 + i * 28, TY + 15, TX + 38 + i * 28, TY + 31), fill=c)
    d.text((TX + TW // 2, TY + 23), "Windows PowerShell", font=UI_S, fill=DIM, anchor="mm")
    d.rectangle((0, H - 8, int(W * progress), H), fill=ACCENT)
    d.text((W - 70, H - 46), "github.com/visweshdev/ReMem", font=UI_S, fill=DIM, anchor="ra")
    return im


def draw_terminal(im: Image.Image, lines: list, cursor: bool):
    d = ImageDraw.Draw(im)
    view = lines[-MAX_LINES:]
    y = TY + 66
    for i, ln in enumerate(view):
        x = TX + 28
        for text, col, bold in ln:
            d.text((x, y), text, font=MONO_B if bold else MONO, fill=col)
            x += d.textlength(text, font=MONO)
        if cursor and i == len(view) - 1:
            d.rectangle((x + 2, y + 2, x + 14, y + LH - 6), fill=FG)
        y += LH


def draw_code(im: Image.Image, filename: str, code: list[str]):
    d = ImageDraw.Draw(im)
    x0, y0, x1 = TX + 40, TY + 70, TX + TW - 40
    y1 = y0 + 70 + LH * len(code)
    d.rounded_rectangle((x0, y0, x1, y1), 12, fill=(30, 34, 42), outline=(60, 66, 78), width=2)
    d.text((x0 + 22, y0 + 14), filename + "  (create this file in the ReMem folder)", font=UI_S, fill=DIM)
    y = y0 + 58
    for line in code:
        code_part, _, comment = line.partition("#")
        d.text((x0 + 26, y), code_part, font=MONO, fill=YELLOW if code_part.startswith(("from", "print")) else FG)
        if comment:
            d.text((x0 + 26 + d.textlength(code_part, font=MONO), y), "#" + comment, font=MONO, fill=DIM)
        y += LH


def title_card(t: float, end: bool = False) -> Image.Image:
    im = Image.new("RGB", (W, H), (15, 47, 38))
    d = ImageDraw.Draw(im)
    if not end:
        d.text((W // 2, H // 2 - 90), "ReMem", font=TITLE, fill=(255, 255, 255), anchor="mm")
        d.text((W // 2, H // 2 + 10), "How to install and run it", font=SUB, fill=(214, 239, 228), anchor="mm")
        d.text((W // 2, H // 2 + 80), "Windows  ·  Python 3.10+  ·  free, runs on your computer", font=UI,
               fill=(159, 201, 182), anchor="mm")
    else:
        d.text((W // 2, 250), "That's it.", font=TITLE, fill=(255, 255, 255), anchor="mm")
        items = ["Use it as Claude's memory (MCP):   examples/claude_desktop_mcp.md",
                 "Use it with LangChain:   examples/langchain_retriever.py",
                 "Real embeddings, dashboard, experiments:   see the README",
                 "Source and docs:   github.com/visweshdev/ReMem"]
        for i, it in enumerate(items):
            d.text((W // 2, 430 + i * 70), it, font=UI, fill=(214, 239, 228), anchor="mm")
    return im


def frames(total: float = 1.0):
    """Yield (image, n_frames) pairs; `total` (seconds) scales the progress bar."""
    elapsed = 0.0
    lines: list = []
    step = None
    code = None

    def emit(img, secs):
        nonlocal elapsed
        elapsed += secs
        return img, max(1, round(secs * FPS))

    def screen(cursor=True):
        im = base(step, min(1.0, elapsed / total))
        if code:
            draw_code(im, *code)
        else:
            draw_terminal(im, lines, cursor)
        return im

    for kind, p in SCRIPT:
        if kind == "title":
            yield emit(title_card(0), 3.5)
        elif kind == "end":
            yield emit(title_card(0, end=True), 5.0)
        elif kind == "step":
            step, code = p, None
            lines.clear()                      # each step starts on a clean screen
            yield emit(screen(), 1.4)
        elif kind == "cmd":
            if code:
                code = None
                lines.append([("", FG, False)])
            prompt, text = p
            lines.append([(prompt + " ", BLUE, False), ("", FG, False)])
            for i in range(0, len(text) + 1, 2):
                lines[-1][1] = (text[:i], FG, True)
                yield emit(screen(), 1 / FPS)
            lines[-1][1] = (text, FG, True)
            yield emit(screen(), 0.5)
        elif kind == "out":
            out = demo_lines() if p == "DEMO" else p
            for text, col in out:
                lines.append([(text, col, False)])
                yield emit(screen(cursor=False), 0.35 if len(out) > 3 else 0.25)
        elif kind == "code":
            code = p
            yield emit(screen(), 0.1)
        elif kind == "pause":
            yield emit(screen(cursor=False), p)


def main() -> None:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise SystemExit("ffmpeg not found on PATH")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen([ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                             "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "slow",
                             "-crf", "23", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(OUT)],
                            stdin=subprocess.PIPE)
    total = sum(c for _, c in frames()) / FPS          # first pass: measure the length for the progress bar
    n = 0
    for img, count in frames(total):
        raw = img.tobytes()
        for _ in range(count):
            proc.stdin.write(raw)
        n += count
    proc.stdin.close()
    proc.wait()
    print(f"{OUT}  ({n / FPS:.0f} s, {OUT.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
