import os
from dotenv import load_dotenv, find_dotenv
from pdf_reader import get_text_from_pdf
from text_preprocessing import text_chunks
from vector_store import store_vector
from qa_chain import conversation_chain
from models import initialize_models

# Load the environment variables
load_dotenv(find_dotenv())

def user_input(user_question, vector_store, llm_model):
    docs = vector_store.similarity_search(user_question)
    
    chain = conversation_chain(llm_model)
    
    answer = chain(
        {"input_documents": docs, "question": user_question},
    )
    print(answer['output_text'])

def main():
    # Initialize models
    llm_model, embeddings_model = initialize_models()

    # Process the PDF
    pdf_path = os.path.join("Data", "data.pdf")
    raw_text = get_text_from_pdf(pdf_path)
    chunks = text_chunks(raw_text)
    vector_store = store_vector(chunks, embeddings_model)
    
    print("Chatbot is ready. Type 'quit' to exit.")
    while True:
        user_question = input("Enter your question: ")
        if user_question.lower() == 'quit':
            break
        user_input(user_question, vector_store, llm_model)

if __name__ == "__main__":
    main()