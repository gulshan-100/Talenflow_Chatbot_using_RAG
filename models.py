from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

def initialize_models():
    llm_model = ChatGoogleGenerativeAI(
        model='gemini-pro',
        temperature=0.0
    )

    embeddings_model = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001"
    )

    return llm_model, embeddings_model