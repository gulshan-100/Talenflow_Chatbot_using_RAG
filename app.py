# Importing the Libraries
from PyPDF2 import PdfReader
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv, find_dotenv

# Load the environment variables
load_dotenv(find_dotenv())

# Load the base LLM and Embedding model
llm_model = ChatGoogleGenerativeAI(
    model='gemini-pro',
    temperature=0.0
)

embeddings_model = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001"
)

def get_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, 'rb') as file:
        pdf = PdfReader(file)
        for page in pdf.pages:
            text += page.extract_text()
    return text

def text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50
    )
    chunks = splitter.split_text(text)
    return chunks

def store_vector(chunks):
    vector_store = FAISS.from_texts(chunks, embeddings_model)
    vector_store.save_local("faiss_index")
    return vector_store

def conversation_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details to the user, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer

    Context: {context}
    Question: {question}

    Answer:
    """
    
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=['context', 'question']
    )
    
    chain = load_qa_chain(
        llm=llm_model,
        chain_type="stuff",
        prompt=prompt
    )
    return chain

def user_input(user_question, vector_store):
    docs = vector_store.similarity_search(user_question)
    
    chain = conversation_chain()
    
    answer = chain(
        {"input_documents": docs, "question": user_question},
    )
    
    print(answer['output_text'])

def main():
    pdf_path = "data.pdf"
    raw_text = get_text_from_pdf(pdf_path)
    chunks = text_chunks(raw_text)
    vector_store = store_vector(chunks)
    
    user_question = input("Enter your question (or 'quit' to exit): ")
    user_input(user_question, vector_store)

if __name__ == "__main__":
    main()