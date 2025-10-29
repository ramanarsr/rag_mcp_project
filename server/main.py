# server/main.py
import os
import re
import numpy as np
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import faiss
from PyPDF2 import PdfReader
from dotenv import load_dotenv
import logging

# Use your Groq client import (adjust if different)
from groq import Groq

from server.schemas import GenerateRequest, GenerateResponse, JudgeRequest, JudgeResponse
from server.mcp_server_utils import (
    load_documents_from_folder,
    build_index,
    build_mcp_prompt,
    summarize_chunks_bullets
)

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("Set GROQ_API_KEY in .env")

client = Groq(api_key=GROQ_API_KEY)

app = FastAPI(title="MCP RAG Server")

@app.get("/")
def root():
    return {"message": "MCP RAG backend is running!"}


# Globals (loaded at startup)
DOCS = []
TITLES = []
EMBEDDER = None
INDEX = None
ALL_EMBEDDINGS = None

@app.on_event("startup")
def startup():
    global DOCS, TITLES, EMBEDDER, INDEX, ALL_EMBEDDINGS
    data_folder = os.getenv("DOCUMENT_PATH", "server/data")
    DOCS, TITLES = load_documents_from_folder(data_folder)
    # build embedder + index
    EMBEDDER, INDEX, ALL_EMBEDDINGS = build_index(DOCS)
    logging.info(f"Loaded {len(DOCS)} paragraphs and built index.")

@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    try:
        # compute query embedding and retrieve
        q_embed = EMBEDDER.encode([req.query], normalize_embeddings=True)
        _, inds = INDEX.search(np.array(q_embed).astype("float32"), req.k or 5)
        retrieved = [DOCS[i] for i in inds[0] if i < len(DOCS)]
        # create bullets summary (optional/async)
        bullets = summarize_chunks_bullets(retrieved, client=client)
        prompt = build_mcp_prompt(req.query, retrieved, bullets=bullets, memory_text=(req.memory or ""))
        # call LLM via Groq client
        resp = client.chat.completions.create(
            model=req.model or "llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=req.temperature if req.temperature is not None else 0.2
        )
        answer = resp.choices[0].message.content.strip()
        return GenerateResponse(answer=answer, retrieved=retrieved)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/judge", response_model=JudgeResponse)
def judge(req: JudgeRequest):
    try:
        eval_prompt = f'''
You are a strict high school science teacher. Evaluate the student's answer to a science question.

### Question:
{req.query}

### Reference Answer:
{req.ref_ans}

### Student's Answer:
{req.gen_ans}

### Task:
Rate the student's answer from 1 to 5:
- 5 = Perfectly correct and complete
- 4 = Mostly correct, minor omissions
- 3 = Partially correct, missing key info or has small mistakes
- 2 = Mostly incorrect or vague
- 1 = Completely incorrect or irrelevant

Respond in this format:
Score: <number>
Reason: <your explanation>
'''
        resp = client.chat.completions.create(
            model=req.model or "llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": eval_prompt}],
            temperature=0.3
        )
        return JudgeResponse(judge_output=resp.choices[0].message.content.strip())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))