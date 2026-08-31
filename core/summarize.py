import os
from concurrent.futures import ThreadPoolExecutor

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter


def get_llm():
    return ChatMistralAI(
        model="mistral-small-latest",
        mistral_api_key=os.getenv("MISTRAL_API_KEY"),
        temperature=0.3,
    )


def split_transcript(transcript: str) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000,
        chunk_overlap=200,
    )
    return splitter.split_text(transcript)


def summarize(transcript: str):
    llm = get_llm()

    map_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Summarize this portion of a meeting transcript concisely."),
            ("human", "{text}"),
        ]
    )
    map_chain = map_prompt | llm | StrOutputParser()

    chunks = split_transcript(transcript)
    if not chunks:
        return "No speech detected in the audio."

    # Fast path: short transcript fits in one chunk — skip the reduce step.
    if len(chunks) == 1:
        return map_chain.invoke({"text": chunks[0]})

    with ThreadPoolExecutor(max_workers=4) as ex:
        chunk_summaries = list(ex.map(lambda c: map_chain.invoke({"text": c}), chunks))
    combined = "\n\n".join(chunk_summaries)

    combined_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an expert meeting summarizer. Combine these partial summaries "
                "into one final professional meeting summary in bullet points.",
            ),
            ("human", "{text}"),
        ]
    )
    combined_chain = combined_prompt | llm | StrOutputParser()

    return combined_chain.invoke({"text": combined})


def generate_title(transcript: str) -> str:
    llm = get_llm()

    title_chain = (
        ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Based on the meeting transcript, generate a short professional meeting title "
                    "(max 8 words). Only return the title, nothing else.",
                ),
                ("human", "{text}"),
            ]
        )
        | llm
        | StrOutputParser()
    )

    return title_chain.invoke({"text": transcript[:2000]})
