from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

def conversation_chain(llm_model):
    """
    Set up the question-answering chain with a specific LLM model.
    Args:
        llm_model: The language model to use for answering questions.
    Returns:
        Chain: Configured QA chain for processing questions.
    """
    prompt_template = """
    Answer the question in full detail or in points from the provided context. Make sure to provide all the details to the user.
    If the answer is not in the provided context, just say, "The answer is not available in the context." Don't provide an incorrect answer.

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