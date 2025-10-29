# server/schemas.py
from pydantic import BaseModel
from typing import List, Optional

class GenerateRequest(BaseModel):
    query: str
    memory: Optional[str] = None
    k: Optional[int] = 5
    model: Optional[str] = None
    temperature: Optional[float] = None

class GenerateResponse(BaseModel):
    answer: str
    retrieved: List[str]

class JudgeRequest(BaseModel):
    query: str
    gen_ans: str
    ref_ans: str
    model: Optional[str] = None

class JudgeResponse(BaseModel):
    judge_output: str