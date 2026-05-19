"""Convert RAPORT.md to standalone RAPORT.html (1:1 markdown content)."""

from __future__ import annotations

import argparse
from pathlib import Path

import markdown

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="pl">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Projekt 5 — Raport SAC CarRacing-v3</title>
<style>
:root {
  --text: #1a1a1a;
  --muted: #555;
  --border: #ddd;
  --code-bg: #f5f5f5;
  --link: #0b57d0;
}
* { box-sizing: border-box; }
body {
  font-family: "Segoe UI", system-ui, -apple-system, sans-serif;
  line-height: 1.6;
  color: var(--text);
  max-width: 920px;
  margin: 0 auto;
  padding: 2rem 1.5rem 4rem;
  background: #fff;
}
h1 { font-size: 1.75rem; border-bottom: 2px solid var(--border); padding-bottom: .4rem; }
h2 { font-size: 1.35rem; margin-top: 2rem; border-bottom: 1px solid var(--border); padding-bottom: .25rem; }
h3 { font-size: 1.15rem; margin-top: 1.5rem; }
h4 { font-size: 1.05rem; margin-top: 1.25rem; }
hr { border: none; border-top: 1px solid var(--border); margin: 2rem 0; }
p { margin: .75rem 0; }
ul, ol { margin: .5rem 0 .75rem 1.25rem; }
li { margin: .25rem 0; }
a { color: var(--link); }
code {
  font-family: Consolas, "Courier New", monospace;
  font-size: .9em;
  background: var(--code-bg);
  padding: .1em .35em;
  border-radius: 3px;
}
pre {
  background: var(--code-bg);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 1rem;
  overflow-x: auto;
}
pre code { background: none; padding: 0; }
table {
  border-collapse: collapse;
  width: 100%;
  margin: 1rem 0;
  font-size: .95rem;
}
th, td {
  border: 1px solid var(--border);
  padding: .45rem .65rem;
  text-align: left;
  vertical-align: top;
}
th { background: #f9f9f9; font-weight: 600; }
tr:nth-child(even) td { background: #fcfcfc; }
img { max-width: 100%; height: auto; display: block; margin: 1rem auto; border: 1px solid var(--border); border-radius: 4px; }
em { color: var(--muted); }
strong { font-weight: 600; }
@media print {
  body { max-width: none; padding: 1cm; }
  a { color: inherit; text-decoration: none; }
  pre { white-space: pre-wrap; word-break: break-word; }
}
</style>
</head>
<body>
{body}
</body>
</html>
"""


def convert(md_path: Path, html_path: Path) -> None:
    md_text = md_path.read_text(encoding="utf-8")
    body = markdown.markdown(
        md_text,
        extensions=["tables", "fenced_code", "sane_lists"],
    )
    html_path.write_text(HTML_TEMPLATE.replace("{body}", body), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert RAPORT.md → RAPORT.html")
    parser.add_argument("--input", type=Path, default=Path("RAPORT.md"))
    parser.add_argument("--output", type=Path, default=Path("RAPORT.html"))
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    convert(repo_root / args.input, repo_root / args.output)
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
