#!/usr/bin/env python3
"""Converts RESEARCH_PAPER.md and RESEARCH_NOTE.md into publication-quality HTML documents with KaTeX math rendering."""
from pathlib import Path

def convert_md_to_html(md_path_str: str, html_path_str: str, title: str):
    md_file = Path(md_path_str)
    if not md_file.exists():
        print(f"File not found: {md_path_str}")
        return
    
    md_content = md_file.read_text(encoding="utf-8")
    
    html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.css">
<style>
  @page {{
    size: letter;
    margin: 20mm;
  }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    line-height: 1.65;
    color: #1e293b;
    max-width: 960px;
    margin: 0 auto;
    padding: 30px 20px;
    background-color: #ffffff;
  }}
  h1 {{
    color: #0f172a;
    font-size: 26px;
    font-weight: 700;
    line-height: 1.3;
    border-bottom: 2px solid #3b82f6;
    padding-bottom: 12px;
    margin-top: 10px;
  }}
  h2 {{
    color: #1e3a8a;
    font-size: 19px;
    font-weight: 600;
    border-bottom: 1px solid #e2e8f0;
    padding-bottom: 6px;
    margin-top: 32px;
  }}
  h3 {{
    color: #1d4ed8;
    font-size: 15px;
    font-weight: 600;
    margin-top: 22px;
  }}
  h4 {{
    color: #334155;
    font-size: 14px;
    font-weight: 600;
    margin-top: 16px;
  }}
  p, li {{
    font-size: 14.5px;
  }}
  table {{
    border-collapse: collapse;
    width: 100%;
    margin: 20px 0;
    font-size: 13.5px;
  }}
  th, td {{
    border: 1px solid #cbd5e1;
    padding: 9px 12px;
    text-align: left;
  }}
  th {{
    background-color: #f1f5f9;
    font-weight: 600;
    color: #0f172a;
  }}
  tr:nth-child(even) {{
    background-color: #f8fafc;
  }}
  code {{
    background: #f1f5f9;
    color: #0f172a;
    padding: 2px 6px;
    border-radius: 4px;
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    font-size: 13px;
  }}
  pre {{
    background: #0f172a;
    color: #f8fafc;
    padding: 16px;
    border-radius: 8px;
    overflow-x: auto;
    font-size: 12.5px;
    line-height: 1.5;
  }}
  pre code {{
    background: transparent;
    color: inherit;
    padding: 0;
  }}
  blockquote {{
    border-left: 4px solid #3b82f6;
    margin: 16px 0;
    padding: 8px 18px;
    background-color: #eff6ff;
    color: #1e3a8a;
    border-radius: 0 6px 6px 0;
  }}
  hr {{
    border: none;
    border-top: 1px solid #e2e8f0;
    margin: 28px 0;
  }}
  .badge {{
    display: inline-block;
    padding: 3px 8px;
    background: #dbeafe;
    color: #1e40af;
    border-radius: 4px;
    font-size: 12px;
    font-weight: 600;
  }}
  @media print {{
    body {{
      max-width: 100%;
      padding: 0;
      color: #000;
    }}
    h1 {{ color: #000; border-bottom: 2px solid #000; }}
    h2, h3 {{ color: #000; }}
    pre, blockquote, table {{ page-break-inside: avoid; }}
    a {{ color: #000; text-decoration: underline; }}
  }}
</style>
<script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/contrib/auto-render.min.js"></script>
</head>
<body>
<div id="content"></div>
<script>
  const markdown = {repr(md_content)};
  document.getElementById('content').innerHTML = marked.parse(markdown);
  document.addEventListener("DOMContentLoaded", function() {{
    renderMathInElement(document.body, {{
      delimiters: [
        {{left: "$$", right: "$$", display: true}},
        {{left: "$", right: "$", display: false}},
        {{left: "\\\\(", right: "\\\\)", display: false}},
        {{left: "\\\\[", right: "\\\\]", display: true}}
      ],
      throwOnError: false
    }});
  }});
</script>
</body>
</html>
"""
    Path(html_path_str).write_text(html_template, encoding="utf-8")
    print(f"Generated: {html_path_str}")

if __name__ == "__main__":
    convert_md_to_html("RESEARCH_PAPER.md", "RESEARCH_PAPER.html", "The One Introduction Problem - Academic Research Paper")
    convert_md_to_html("RESEARCH_NOTE.md", "RESEARCH_NOTE.html", "Sequential Reciprocal Matching - Round 1 Research Note")
