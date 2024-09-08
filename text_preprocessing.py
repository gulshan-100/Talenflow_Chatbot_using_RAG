from langchain.text_splitter import RecursiveCharacterTextSplitter

def text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50
    )
    chunks = splitter.split_text(text)
    return chunks