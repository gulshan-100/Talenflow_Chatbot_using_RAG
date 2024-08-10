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
    try:
        text = ""
        with open(pdf_path, 'rb') as file:
            pdf = PdfReader(file)
            for page in pdf.pages:
                text += page.extract_text()
        return text
    except Exception as e:
        print(f"Error reading the PDF document: {e}")
        return None

def text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50
    )
    chunks = splitter.split_text(text)
    return chunks

def store_vector(chunks):
    if os.path.exists("faiss_index"):
        vector_store = FAISS.load_local("faiss_index", embeddings_model)
    else:
        vector_store = FAISS.from_texts(chunks, embeddings_model)
        vector_store.save_local("faiss_index")
    return vector_store

def conversation_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details to the user. If the answer is not in
    the provided context just say, 'answer is not available in the context', don't provide the wrong answer.

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
    try:
        docs = vector_store.similarity_search(user_question)
        if not docs:
            print('No relevant information found in the documents')
            return 
        
        chain = conversation_chain()
    
        answer = chain(
            {"input_documents": docs, "question": user_question},
        )
    
        print(answer['output_text'])
    except Exception as e:
        print(f"Error processing the input: {e}")
        
def main():
    pdf_path = "data.pdf"
    raw_text = get_text_from_pdf(pdf_path)
    if not raw_text:
        print("Error reading the PDF Document")
    
    chunks = text_chunks(raw_text)
    vector_store = store_vector(chunks)
    
    while True:
        user_question = input("Enter your question (or write 'quit' to exit): ")
        if user_question.lower() == 'quit':
            break
        user_input(user_question, vector_store)

if __name__ == "__main__":
    main()