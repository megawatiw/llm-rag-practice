from llm import *
from preprocess_data import *
from faiss import *
from config import *


def summarize_video(video_url):
    global fetched_transcript, processed_transcript 
    
    if video_url:
		fetched_transcript = get_transcript(video_url)
		processed_transcript = process_transcript(fetched_transcript)
	else:
		return "Please provide a valid YouTube URL."

    if processed_transcript:
        model_id, credentials, client, project_id = setup_credentials()
        params = define_parameters()
        llm = init_llm(model_id, credentials, project_id, params)

        summary_prompt = create_summary_prompt_template()
        summary_chain = create_summary_chain(llm, summary_prompt)

        summary = summary_chain.run(
            {
                "transcript": processed_transcript
            }
        )
        return summary
    else:
        return "No transcript available. Please fetch the transcript first."


def answer_question(video_url, question):
    global fetched_transcript, processed_transcript 
    
    if video_url:
		fetched_transcript = get_transcript(video_url)
		processed_transcript = process_transcript(fetched_transcript)
	else:
		return "Please provide a valid YouTube URL."

    if processed_transcript:
        model_id, credentials, client, project_id = setup_credentials()
        params = define_parameters()
        embedding_model_id = "ibm/slate-30m-english-rtrvr-v2"

        llm = init_llm(model_id, credentials, project_id, params)
        embedding_model = init_embedding_model(embedding_model_id, credentials, project_id)

        chunks = chunk_transcript(processed_transcript)
        faiss_index = create_faiss_index(chunks, embedding_model)

        qa_prompt = create_qa_prompt_template()
        qa_chain = create_qa_chain(llm, qa_prompt)

        answer = generate_answer(question, faiss_index, qa_chain)
        return answer
    else:
        return "No transcript available. Please fetch the transcript first."

