import markdown
from xhtml2pdf import pisa
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import os

md_path = r"C:\Users\301-4\.gemini\antigravity-ide\brain\7a6536d9-bcd6-4caa-ba4b-d3c75eaa3491\Integrated_Master_Report.md"
pdf_path = r"C:\Users\301-4\.gemini\antigravity-ide\brain\7a6536d9-bcd6-4caa-ba4b-d3c75eaa3491\Integrated_Master_Report.pdf"

# 1. Register Malgun Gothic directly in ReportLab using local copied files
font_path = r"C:\Users\301-4\.gemini\antigravity-ide\brain\7a6536d9-bcd6-4caa-ba4b-d3c75eaa3491\malgun.ttf"
font_bold_path = r"C:\Users\301-4\.gemini\antigravity-ide\brain\7a6536d9-bcd6-4caa-ba4b-d3c75eaa3491\malgunbd.ttf"

try:
    pdfmetrics.registerFont(TTFont('MalgunGothic', font_path))
    pdfmetrics.registerFont(TTFont('MalgunGothicBold', font_bold_path))
    print("Fonts registered successfully in ReportLab.")
except Exception as e:
    print("Font registration error:", e)

# 2. Read Markdown
with open(md_path, 'r', encoding='utf-8') as f:
    text = f.read()

# 3. Convert to HTML
html_content = markdown.markdown(text, extensions=['tables', 'fenced_code'])

# 4. Add styling (NO @font-face block, using pre-registered font family directly)
styled_html = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
    @page {{
        size: a4;
        margin: 2cm;
    }}
    body {{
        font-family: MalgunGothic;
        color: #2c3e50;
        line-height: 1.6;
        font-size: 10pt;
    }}
    h1 {{
        font-family: MalgunGothicBold;
        font-size: 18pt;
        color: #2c3e50;
        border-bottom: 2px solid #34495e;
        padding-bottom: 5px;
        margin-top: 20px;
        margin-bottom: 15px;
    }}
    h2 {{
        font-family: MalgunGothicBold;
        font-size: 14pt;
        color: #2980b9;
        margin-top: 15px;
        margin-bottom: 10px;
        border-bottom: 1px solid #ddecf8;
        padding-bottom: 3px;
    }}
    h3 {{
        font-family: MalgunGothicBold;
        font-size: 11pt;
        color: #34495e;
        margin-top: 10px;
        margin-bottom: 5px;
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
        font-family: MalgunGothicBold;
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

# 5. Generate PDF
with open(pdf_path, "wb") as pdf_file:
    pisa_status = pisa.CreatePDF(styled_html, dest=pdf_file)

if not pisa_status.err:
    print("PDF generated successfully at:", pdf_path)
    # Also sync to workspace
    import shutil
    shutil.copy2(pdf_path, r"C:\team\chym_aki\Integrated_Master_Report.pdf")
    print("Synced to workspace.")
else:
    print("Error generating PDF:", pisa_status.err)
