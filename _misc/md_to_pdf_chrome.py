import markdown
import subprocess
import shutil
import os

md_path = r"C:\Users\301-4\.gemini\antigravity-ide\brain\7a6536d9-bcd6-4caa-ba4b-d3c75eaa3491\Integrated_Master_Report.md"
html_path = r"C:\team\chym_aki\Integrated_Master_Report.html"
pdf_path = r"C:\Users\301-4\.gemini\antigravity-ide\brain\7a6536d9-bcd6-4caa-ba4b-d3c75eaa3491\Integrated_Master_Report.pdf"

# 1. Read Markdown
with open(md_path, 'r', encoding='utf-8') as f:
    text = f.read()

# 2. Convert to HTML
html_content = markdown.markdown(text, extensions=['tables', 'fenced_code'])

# 3. Add styling with standard Malgun Gothic (Chrome will resolve it natively)
styled_html = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
    @page {{
        size: A4;
        margin: 20mm;
    }}
    body {{
        font-family: 'Malgun Gothic', '맑은 고딕', sans-serif;
        color: #000000;
        line-height: 1.6;
        font-size: 10.5pt;
        font-weight: 500;
    }}
    h1 {{
        font-size: 19pt;
        color: #000000;
        border-bottom: 2px solid #222222;
        padding-bottom: 5px;
        margin-top: 20px;
        margin-bottom: 15px;
        font-weight: bold;
    }}
    h2 {{
        font-size: 14.5pt;
        color: #111111;
        margin-top: 15px;
        margin-bottom: 10px;
        border-bottom: 1px solid #cccccc;
        padding-bottom: 3px;
        font-weight: bold;
    }}
    h3 {{
        font-size: 11.5pt;
        color: #111111;
        margin-top: 10px;
        margin-bottom: 5px;
        font-weight: bold;
    }}
    p {{
        margin-bottom: 10px;
    }}
    table {{
        width: 100%;
        border-collapse: collapse;
        margin-bottom: 15px;
        font-size: 9pt;
    }}
    th {{
        background-color: #f2f4f4;
        color: #2c3e50;
        font-weight: bold;
        border: 1px solid #bdc3c7;
        padding: 6px;
        text-align: left;
    }}
    td {{
        border: 1px solid #bdc3c7;
        padding: 6px;
        text-align: left;
    }}
    img {{
        max-width: 100%;
        height: auto;
        display: block;
        margin: 15px auto;
    }}
    blockquote {{
        background-color: #f8f9f9;
        border-left: 4px solid #3498db;
        padding: 8px 12px;
        margin: 10px 0;
        font-size: 9.5pt;
    }}
    pre, code {{
        font-family: Courier, monospace;
        background-color: #f4f6f7;
        font-size: 9pt;
    }}
</style>
</head>
<body>
    {html_content}
</body>
</html>
"""

with open(html_path, 'w', encoding='utf-8') as f:
    f.write(styled_html)

print("HTML generated successfully.")

# 4. Use Google Chrome in headless mode to convert HTML to PDF
chrome_path = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
if not os.path.exists(chrome_path):
    # Try alternate location
    chrome_path = r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"

print("Using Chrome at:", chrome_path)

cmd = [
    chrome_path,
    "--headless",
    "--disable-gpu",
    f"--print-to-pdf={pdf_path}",
    html_path
]

try:
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    print("Chrome execution output:", result.stdout)
    print("PDF generated successfully at:", pdf_path)
    # Copy to workspace
    shutil.copy2(pdf_path, r"C:\team\chym_aki\Integrated_Master_Report.pdf")
    print("Synced to workspace.")
except Exception as e:
    print("Error calling Chrome headless:", e)
