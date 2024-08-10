from langchain.text_splitter import RecursiveCharacterTextSplitter

def text_chunks(text, chunk_size=300, chunk_overlap=50):
    """
    Split the text into chunks for easier processing.

    Args:
        text (str): The text to be split.
        chunk_size (int): Size of each text chunk.
        chunk_overlap (int): Number of overlapping characters between chunks.

    Returns:
        list: List of text chunks.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_text(text)
    return chunks
