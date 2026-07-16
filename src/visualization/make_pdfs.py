"""Render project markdown documents to typeset PDFs.

Pipeline: markdown -> HTML (mistune, tables enabled) -> headless Chromium
print-to-PDF (bundled browser; fully offline). Figures embed from their
repo-relative paths; the image alt text becomes the printed caption.

Usage:
    python -m src.visualization.make_pdfs README.md reports/pdf/Beyond_AUC_paper.pdf \
        --title-note "pre-print"
"""
from __future__ import annotations

import argparse
import re
import subprocess
import tempfile
from pathlib import Path

import mistune

from src import config

CHROMIUM = "/opt/pw-browsers/chromium"

CSS = """
@page { size: Letter; margin: 21mm 19mm 23mm 19mm; }
:root { color-scheme: light; }
* { box-sizing: border-box; }
body {
  font-family: 'DejaVu Serif', Georgia, serif;
  font-size: 10.2pt; line-height: 1.62; color: #101010;
  margin: 0; padding: 0;
}
h1, h2, h3, h4 { font-family: 'DejaVu Sans', Helvetica, sans-serif;
  color: #0b0b0b; line-height: 1.25; page-break-after: avoid; }
h1 { font-size: 19pt; text-align: center; margin: 0 0 4pt; }
h2 { font-size: 13.5pt; margin: 22pt 0 6pt; border-bottom: 0.6pt solid #c9c8c0;
     padding-bottom: 3pt; }
h3 { font-size: 11pt; margin: 14pt 0 4pt; }
p { margin: 0 0 7pt; text-align: justify; }
.author { text-align: center; font-family: 'DejaVu Sans', sans-serif;
  font-size: 10.5pt; color: #33322f; margin-bottom: 2pt; }
.docnote { text-align: center; font-size: 8.8pt; color: #6f6e68;
  margin-bottom: 16pt; font-style: italic; }
strong { color: #0b0b0b; }
a { color: #1c5cab; text-decoration: none; word-break: break-all; }
code { font-family: 'DejaVu Sans Mono', monospace; font-size: 8.6pt;
  background: #f3f2ee; padding: 0.5pt 2.5pt; border-radius: 2px; }
pre { background: #f6f5f1; border: 0.6pt solid #e1e0d9; border-radius: 4px;
  padding: 8pt 10pt; font-size: 8.4pt; line-height: 1.45; overflow-x: hidden;
  white-space: pre-wrap; page-break-inside: avoid; }
pre code { background: none; padding: 0; }
table { border-collapse: collapse; margin: 10pt auto; font-size: 8.7pt;
  font-family: 'DejaVu Sans', sans-serif; page-break-inside: avoid;
  font-variant-numeric: tabular-nums; }
th { background: #f0efe9; border: 0.6pt solid #c9c8c0; padding: 3.5pt 6pt;
  text-align: left; font-weight: 600; }
td { border: 0.6pt solid #d9d8d0; padding: 3pt 6pt; }
tr:nth-child(even) td { background: #fafaf7; }
figure { margin: 12pt 0; text-align: center; page-break-inside: avoid; }
figure img { max-width: 100%; max-height: 8.4cm; }
figcaption { font-size: 8.6pt; color: #55544e; margin-top: 4pt;
  font-family: 'DejaVu Sans', sans-serif; }
blockquote { border-left: 3pt solid #86b6ef; background: #f2f7fd;
  margin: 9pt 0; padding: 6pt 10pt; page-break-inside: avoid; }
blockquote p { margin: 0 0 4pt; text-align: left; }
hr { border: none; border-top: 0.6pt solid #c9c8c0; margin: 14pt 0; }
li { margin-bottom: 3pt; }
.footer { position: fixed; bottom: -15mm; left: 0; right: 0;
  font-family: 'DejaVu Sans', sans-serif; font-size: 7.4pt; color: #8a8983;
  text-align: center; }
"""


def md_to_html(md_text: str, footer: str) -> str:
    render = mistune.create_markdown(plugins=["table", "strikethrough"])
    body = render(md_text)
    # images -> figure/figcaption using alt text as the caption
    def _fig(m):
        alt, src = m.group(1), m.group(2)
        return (f'<figure><img src="{src}" alt="{alt}"/>'
                f"<figcaption>{alt}</figcaption></figure>")
    body = re.sub(r'<img src="([^"]*)" alt="([^"]*)"\s*/?>',
                  lambda m: _fig(type("M", (), {"group": lambda s, i,
                                 g=(m.group(2), m.group(1)): g[i - 1]})()),
                  body)
    # first paragraph after h1 that starts with <strong> = author line
    body = re.sub(r"(</h1>\s*)<p>(<strong>.*?)</p>\s*<p><em>(.*?)</em></p>",
                  r'\1<p class="author">\2</p><p class="docnote">\3</p>',
                  body, count=1, flags=re.S)
    return (f"<!DOCTYPE html><html><head><meta charset='utf-8'>"
            f"<style>{CSS}</style></head><body>{body}</body></html>")


def html_to_pdf(html_path: Path, pdf_path: Path) -> None:
    subprocess.run(
        [CHROMIUM, "--headless=new", "--disable-gpu", "--no-sandbox",
         "--force-device-scale-factor=1", "--run-all-compositor-stages-before-draw",
         "--virtual-time-budget=20000",
         f"--print-to-pdf={pdf_path}", "--no-pdf-header-footer",
         html_path.as_uri()],
        check=True, capture_output=True, timeout=180)


def render(md_file: Path, pdf_file: Path, footer: str) -> Path:
    md_text = md_file.read_text()
    html = md_to_html(md_text, footer)
    # write the HTML next to the repo root so relative image paths resolve
    html_path = config.ROOT / f".pdf_build_{md_file.stem}.html"
    html_path.write_text(html)
    pdf_file.parent.mkdir(parents=True, exist_ok=True)
    html_to_pdf(html_path, pdf_file)
    html_path.unlink()
    return pdf_file


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("markdown")
    ap.add_argument("pdf")
    ap.add_argument("--footer", default="Beyond AUC - cost-sensitive, "
                    "calibrated CNP fraud detection under temporal drift")
    args = ap.parse_args()
    out = render(config.ROOT / args.markdown, config.ROOT / args.pdf, args.footer)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
