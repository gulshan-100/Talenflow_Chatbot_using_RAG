from PyPDF2 import PdfReader
import io

def get_text_from_pdf(pdf_content):
    try:
        text = ""
        pdf = PdfReader(io.BytesIO(pdf_content))
        for page in pdf.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return ""