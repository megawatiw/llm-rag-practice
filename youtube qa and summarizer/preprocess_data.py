# Import necessary libraries for the YouTube bot

import re  #For extracting video id 
from youtube_transcript_api import YouTubeTranscriptApi  # For extracting transcripts from YouTube videos
from langchain.text_splitter import RecursiveCharacterTextSplitter  # For splitting text into manageable segments



def get_video_id(url):    
    # Regex pattern to match YouTube video URLs
    pattern = r'https:\/\/www\.youtube\.com\/watch\?v=([a-zA-Z0-9_-]{11})'
    match = re.search(pattern, url)
    return match.group(1) if match else None


def get_transcript(url):
    video_id = get_video_id(url)
    
    ytt_api = YouTubeTranscriptApi()
    transcripts = ytt_api.list(video_id)
    
    transcript = None
    for t in transcripts:
        if t.language_code == 'en':
            if t.is_generated:
                if transcript is None:
                    transcript = t.fetch()
            else:
                transcript = t.fetch()
                break
    
    return transcript if transcript else None


def process_transcript(transcript):
    txt = ""
    
    for item in transcript:
        try:
            cleaned_text = re.sub(r'^[A-Za-z0-9\'\s]', "", item.text)
            txt += f"Text: {cleaned_text} Start: {item.start}\n"
        except KeyError:
            pass
            
    return txt


def chunk_transcript(processed_transcript, chunk_size=200, chunk_overlap=20):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    chunks = text_splitter.split_text(processed_transcript)
    return chunks



