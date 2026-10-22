from typing import List
from langdetect import detect
from dotenv import load_dotenv
import os, torch
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from sentence_transformers import CrossEncoder
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_chroma.vectorstores import Chroma

load_dotenv()

GROQ_TRANSLATE_API_KEY = os.getenv("GROQ_TRANSLATE_API_KEY")
llm_translate = ChatGroq(api_key=GROQ_TRANSLATE_API_KEY, model="llama-3.3-70b-versatile")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
llm = ChatGroq(api_key=GROQ_API_KEY, model="llama-3.3-70b-versatile")

# Build Retrieval-QA Chain
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

vectordb = Chroma(persist_directory="vector_store", embedding_function=embeddings)
retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 20})

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cuda" if torch.cuda.is_available() else "cpu")

def rerank_docs(question: str, docs: List):
    pairs = [(question, d.page_content) for d in docs]
    scores = reranker.predict(pairs, show_progress_bar=False)
    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in ranked[:5]]

def build_context_text(docs: List, max_chars: int = 8000) -> str:
    context_parts = []
    current_len = 0

    for d in docs:
        text = d.page_content.strip()
        if not text:
            continue

        if current_len + len(text) > max_chars:
            remaining = max_chars - current_len
            if remaining > 0:
                context_parts.append(text[:remaining])
            break

        context_parts.append(text)
        current_len += len(text)

    return "\n\n".join(context_parts)

combine_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """RULES (MANDATORY):
        1. Answer ONLY if the exact information is explicitly present in the context.
        2. If the exact answer is NOT present, output exactly:
           "My database does not contain this information."
        3. Do NOT add related facts, examples, explanations, or partial matches.
        4. Do NOT provide alternative information.
        5. Do NOT mention documents, sources, rounds, events, or metadata unless explicitly asked AND present.
        6. Your response must be a single coherent answer, nothing extra.""".strip(),
    ),
    (
        "user",
        "Context:\n{context}\n\nQuestion:\n{input}",
    ),
])

# Chain composition: prompt → LLM → output parser
combine_docs_chain = (
    combine_prompt
    | llm
    | StrOutputParser()
)

def run_rag(question: str) -> str:
    docs = retriever.invoke(question)
    docs = rerank_docs(question, docs)

    context_text = build_context_text(docs)

    return combine_docs_chain.invoke({
        "context": context_text,
        "input": question,
    })

def detect_and_translate_question(user_input: str):
    prompt = (
        "You are a translator/detector. "
        "Detect the language of the following user question and then provide an English translation. "
        "Output exactly in the following two-line format (no extra explanation):\n"
        "LANGUAGE_CODE: <language name in small case> if its romanization of a language then 'Romanized <language name in small case>'\n"
        "TRANSLATION: <English translation of the question>\n\n"
        f"User question: {user_input}"
    )

    resp = llm_translate.invoke(prompt)
    result = resp.content.strip()

    lang_code = "english"
    translation = user_input

    try:
        for line in result.splitlines():
            if line.startswith("LANGUAGE_CODE:"):
                lang_code = line.split(":", 1)[1].strip()
            elif line.startswith("TRANSLATION:"):
                translation = line.split(":", 1)[1].strip()
    except Exception:
        pass

    return lang_code, translation

def translate_answer_back(answer: str, target_language_code: str):
    if target_language_code == "english":
        return answer

    prompt = (
        "You are a translator. Translate the following answer into the language "
        f"{target_language_code} if and only if it is an Indian Language or Romanized Indian Language, "
        "else output in English.\n"
        "IMPORTANT:\n"
        "- Output ONLY the translated text.\n"
        "- Do NOT include explanations, notes, brackets, or commentary.\n"
        "- If no translation is needed, repeat the answer verbatim.\n"
        f"{answer}"
    )

    resp = llm_translate.invoke(prompt)
    return resp.content.strip()

# Public API
def ask_user(user_input: str) -> str:
    lang_code, translated_question = detect_and_translate_question(user_input)

    answer = run_rag(translated_question)

    try:
        if lang_code != "english" and detect(answer) == "en":
            answer = translate_answer_back(answer, lang_code)
    except Exception:
        pass

    return answer

# Chat loop (Terminal-based)
if __name__ == "__main__":
    print("\nMultilingual Chatbot is ready!")
    print("Type your question (type 'exit' to quit)\n")

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        print(f"Bot: {ask_user(user_input)}\n")
