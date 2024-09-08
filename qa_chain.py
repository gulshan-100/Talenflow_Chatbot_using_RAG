from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

def conversation_chain(llm_model):
    prompt_template = """
    Answer the question in full detail or in points from the provided context, make sure to provide all the details to the user, if the answer is not in
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