from PyPDF2 import PdfReader

def get_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, 'rb') as file:
        pdf = PdfReader(file)
        for page in pdf.pages:
            text += page.extract_text()
    return text