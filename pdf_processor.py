# pdf_processor.py
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def process_pdf(file):
    # Extract text from PDF
    reader = PdfReader(file)
    text = "".join([page.extract_text() for page in reader.pages])
    
    # Handle Nepali text encoding
    text = text.encode('utf-8').decode('utf-8')
    
    # Split text into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    return splitter.split_text(text)