# server/mcp_server_utils.py
import os
import re
from typing import List, Tuple
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from PyPDF2 import PdfReader
from groq import Groq
from dotenv import load_dotenv

load_dotenv()


FEW_SHOTS = [
    {"context": "Photosynthesis converts sunlight into chemical energy stored as sugars.",
     "question": "What is photosynthesis?",
     "answer": "Photosynthesis is the process by which green plants use sunlight to make food."},
    {"context": "Newton’s first law states an object stays at rest or uniform motion unless acted upon.",
     "question": "State Newton's first law.",
     "answer": "An object at rest stays at rest, and in motion stays in motion unless acted upon by a force."}
]

def load_documents_from_folder(folder_path=os.getenv("DOCUMENT_PATH", "server/data")):
    docs, titles = [], []
    pdf_files = [f for f in os.listdir(folder_path) if f.endswith(".pdf")]
    for file in pdf_files:
        path = os.path.join(folder_path, file)
        try:
            reader = PdfReader(path)
        except Exception:
            continue
        for page in reader.pages:
            text = page.extract_text()
            if not text:
                continue
            paragraphs = [re.sub(r"\s+", " ", p.strip()) for p in text.split("\n\n") if len(p.strip()) > 100]
            docs.extend(paragraphs)
            titles.extend([file] * len(paragraphs))
    return docs, titles

def build_index(docs):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(docs, normalize_embeddings=True, show_progress_bar=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(np.array(embeddings).astype("float32"))
    return model, index, embeddings

def compressed_few_shots(max_chars=500):
    shots = [f"Ctx:{ex['context'][:150]} | Q:{ex['question']} | A:{ex['answer']}" for ex in FEW_SHOTS]
    return " || ".join(shots)[:max_chars]

def summarize_chunks_bullets(chunks: List[str], max_bullets=4, client: Groq = None):
    if not chunks:
        return []
    if client is None:
        # fallback: return truncated bullets
        return [c[:120] + "..." for c in chunks[:max_bullets]]
    prompt = (
        "Summarize each excerpt into one short bullet (<=15 words):\n\n"
        + "\n".join([f"{i+1}. {c[:400]}" for i, c in enumerate(chunks[:max_bullets])])
    )
    try:
        resp = client.chat.completions.create(model="llama-3.1-8b-instant",
                                             messages=[{"role":"user","content":prompt}],
                                             temperature=0.0)
        bullets = [re.sub(r"^\s*\d+[\.\)]\s*", "", b.strip())
                   for b in resp.choices[0].message.content.splitlines() if b.strip()]
        return bullets[:max_bullets]
    except Exception:
        return [c[:120] + "..." for c in chunks[:max_bullets]]

def build_mcp_prompt(query: str, retrieved: List[str], bullets: List[str], memory_text: str = "", max_chars=2500):
    few_shots = (
        "Ctx:Photosynthesis converts sunlight into chemical energy stored as sugars. | Q:What is photosynthesis? | A:Photosynthesis is the process by which green plants use sunlight to make food."
        " || Ctx:Newton’s first law states an object stays at rest or uniform motion unless acted upon. | Q:State Newton's first law. | A:An object at rest stays at rest, and in motion stays in motion unless acted upon by a force."
    )
    context = "\n".join([f"- {b}" for b in bullets])
    prompt = (
        "SYSTEM: You are a concise high-school science tutor. Use only context facts below.\n\n"
        f"FEW-SHOTS:\n{few_shots}\n\n"
        f"CONTEXT BULLETS:\n{context}\n\n"
        f"QUESTION:\n{query}\n\n"
        f"MEMORY:\n{memory_text}\n\nANSWER:"
    )
    return prompt[:max_chars]