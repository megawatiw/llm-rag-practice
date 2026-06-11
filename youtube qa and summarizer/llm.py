from langchain.chains import LLMChain  # For creating chains of operations with LLMs
from langchain.prompts import PromptTemplate  # For defining prompt templates
from faiss import perform_similarity_search


def get_context(faiss_index, query, k=7):
    relevant_context = perform_similarity_search(faiss_index, query, k)
    return relevant_context


def generate_answer(query, faiss_index, qa_chain, k=7):
    relevant_context = get_context(faiss_index, query, k)
    answer = qa_chain.predict(context=relevant_context, question=query)
    return answer


def create_summary_prompt_template():
    summary_template = """
        <|begin_of_text|><|start_header_id|>system<|end_header_id|>
        You are an AI assistant tasked with summarizing YouTube video transcripts. Provide concise, informative summaries that capture the main points of the video content.

        Instructions:
        1. Summarize the transcript in a single concise paragraph.
        2. Ignore any timestamps in your summary.
        3. Focus on the spoken content (Text) of the video.

        Note: In the transcript, "Text" refers to the spoken words in the video, and "start" indicates the timestamp when that part begins in the video.<|eot_id|><|start_header_id|>user<|end_header_id|>
        Please summarize the following YouTube video transcript:

        {transcript}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """
    prompt_template = PromptTemplate(
        input_variables=['transcript'],
        template=summary_template
    )

    return prompt_template
    

def create_summary_chain(llm, prompt, verbose=True):
    return LLMChain(llm=llm, prompt=prompt, verbose=verbose)


def create_qa_prompt_template():
    qa_template = """
        You are an expert assistant providing detailed answers based on the following video content.

        Relevant Video Context: {context}

        Based on the above context, please answer the following question:
        Question: {question}
    """
    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template=qa_template
    )

    return prompt_template


def create_qa_chain(llm, prompt, verbose=True):
    return LLMChain(llm=llm, prompt=prompt, verbose=verbose)