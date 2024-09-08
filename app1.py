import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from dotenv import load_dotenv, find_dotenv
from pdf_reader import get_text_from_pdf
from text_preprocessing import text_chunks
from vector_store import store_vector
from qa_chain import conversation_chain

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

def user_input(user_question, vector_store):
    docs = vector_store.similarity_search(user_question)
    
    chain = conversation_chain(llm_model)
    
    answer = chain(
        {"input_documents": docs, "question": user_question},
    )
    print(answer['output_text'])

def main():
    # Read the PDF file
    pdf_path = os.path.join("Data", "data.pdf")
    with open(pdf_path, "rb") as file:
        pdf_content = file.read()
    
    # Process the PDF content
    raw_text = get_text_from_pdf(pdf_content)
    chunks = text_chunks(raw_text)
    vector_store = store_vector(chunks, embeddings_model)
    
    print("Chatbot is ready. Type 'quit' to exit.")
    while True:
        user_question = input("Enter your question: ")
        if user_question.lower() == 'quit':
            break
        user_input(user_question, vector_store)

if __name__ == "__main__":
    main()