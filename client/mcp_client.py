# client/mcp_client.py
import os
import requests
from dotenv import load_dotenv

load_dotenv()
BASE = f"https://rag-mcp-project.onrender.com"

def generate(query: str, memory: str = "", k: int = 5, model: str = None, temperature: float = None):
    payload = {"query": query, "memory": memory, "k": k}
    if model:
        payload["model"] = model
    if temperature is not None:
        payload["temperature"] = temperature
    resp = requests.post(f"{BASE}/generate", json=payload, timeout=60)
    resp.raise_for_status()
    return resp.json()  # {answer:, retrieved:[]}

def judge(query: str, gen_ans: str, ref_ans: str, model: str = None):
    payload = {"query": query, "gen_ans": gen_ans, "ref_ans": ref_ans}
    if model:
        payload["model"] = model
    resp = requests.post(f"{BASE}/judge", json=payload, timeout=60)
    resp.raise_for_status()

    return resp.json()  # {judge_output: str}
