# song_finder_module.py

import os, getpass
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import Tool, initialize_agent
from langchain.memory import ConversationBufferMemory
import random
import pandas as pd
import re

# Load Model
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7)
embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Load dan bersihkan data
CSV_FILE_PATH = 'https://raw.githubusercontent.com/DUNEFWZ/Final-project-DSAI/refs/heads/main/spotify_songs.csv'
df = pd.read_csv(CSV_FILE_PATH)

# Adaptif Column Mapping

def match_column(possible_names, available_columns):
    for name in possible_names:
        for col in available_columns:
            if name.lower() in col.lower():
                return col
    return None

def validate_and_prepare_dataframe(df):
    name_col = match_column(["track_name", "song", "title"], df.columns)
    artist_col = match_column(["track_artist", "artist", "singer", "band"], df.columns)

    if not name_col or not artist_col:
        raise ValueError(
            "❌ Dataset tidak memiliki kolom yang sesuai untuk nama lagu dan artis."
        )

    df = df.rename(columns={name_col: "track_name", artist_col: "track_artist"})
    return df

df = validate_and_prepare_dataframe(df)

# Vectorstore Setup

# Combine text for vector search
try:
    df["combined_text"] = df.apply(lambda row: f"{row['track_name']} by {row['track_artist']}", axis=1)
except:
    raise ValueError("Gagal membentuk data gabungan untuk pencarian. Pastikan dataset berisi lagu.")

docs = df["combined_text"].tolist()
documents = [Document(page_content=text, metadata={"index": i}) for i, text in enumerate(docs)]
vectorstore = FAISS.from_documents(documents, embedding_model)

# Tools
import re

def detect_language(text: str) -> str:
    prompt = f"What language is this? Respond only with ISO code like 'id' or 'en'.\n\n{text}"
    return llm.invoke(prompt).content.strip().lower()

def translate_input(text: str) -> str:
    prompt = f"Translate this to English if it's not already:\n\n{text}"
    return llm.invoke(prompt).content.strip()

def translate_back(text: str, lang_code: str) -> str:
    prompt = f"""You are a translation assistant. Translate ONLY into the language with ISO code '{lang_code}'.
Only output one version. Do not provide multiple language translations.

Do not translate anything inside double curly braces like {{this}}.

Text:
{text}
"""
    return llm.invoke(prompt).content.strip()

def map_genre(query: str) -> str:
    prompt = f"From this user mood or query, infer potential music genre (pop, rock, acoustic, dance, sadcore, etc).\n\nQuery: {query}"
    return llm.invoke(prompt).content.strip().lower()

def randomize_results(results, k=3):
    shuffle(results)
    return results[:k]

def detect_requested_song_count(text: str) -> int:
    match = re.search(r"(\d+)\s*(lagu|songs?)", text.lower())
    if match:
        return int(match.group(1))
    return 3

chat_history = []
last_recommendation = ""
system_identity = "You are an emotionally-aware music recommender chatbot that responds with empathy, adapts to user's language, and explains song selections insightfully."

#Retrieve Songs

def retrieve_similar_songs(query: str, count: int = 3) -> str:
    results = vectorstore.similarity_search(query, k=10)
    shuffle(results)
    seen = set()
    selected = []

    for doc in results:
        if len(selected) >= count:
            break
        title = doc.page_content.strip().lower()
        if title not in seen:
            seen.add(title)
            selected.append(doc)

    global last_recommendation
    song_lines = []

    for doc in selected:
        raw_text = doc.page_content
        escaped_text = raw_text.replace("{", "{{").replace("}", "}}")
        display_text = raw_text.replace("{", "").replace("}", "")
        song_lines.append({
            "escaped": f"🎶 {escaped_text}",
            "display": f"🎶 {display_text}"
        })

    last_recommendation = "\n".join([line["escaped"] for line in song_lines])
    return "\n".join([line["display"] for line in song_lines])

# Explanation

def explain_recommendation(query: str, context: str = "") -> str:
    prompt = f"""
{system_identity}

Conversation history (if any):
{context}

User input: {query}
Recommended songs:
{last_recommendation}

Explain why these songs are suitable. Include breakdowns of lyrics, genre, and mood. Maintain user language style.
Avoid repeating the song list again.
"""
    return llm.invoke(prompt).content.strip()

def smart_rag_response(user_input: str) -> str:
    lang = detect_language(user_input)
    input_en = translate_input(user_input) if lang != "en" else user_input

    inferred_genre = map_genre(input_en)
    requested_count = detect_requested_song_count(user_input)
    songs = retrieve_similar_songs(f"{input_en}, genre: {inferred_genre}", count=requested_count)
    context = "\n".join([f"User: {u}\nBot: {b}" for u, b in chat_history[-2:]])
    explanation = explain_recommendation(input_en, context)

    full_response = f"\nHere are songs I picked for you:\n\n{songs}\n\n{explanation}"

    if lang != "en":
        full_response = translate_back(full_response, lang)
        full_response = re.split(r"\n?\*\*[a-z]{2}:\*\*", full_response)[0].strip()

    chat_history.append((user_input, full_response))
    return full_response


def clean_cli_text(text: str) -> str:
    # Hapus tanda bold markdown: **teks** atau *teks*
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"\*(.*?)\*", r"\1", text)

    # Hapus {{curly braces}} dan {single braces}
    text = re.sub(r"\{\{(.*?)\}\}", r"\1", text)
    text = re.sub(r"\{(.*?)\}", r"\1", text)
