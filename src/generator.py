import re
import logging
import ollama
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a Sunrise AMC investor support assistant. 
Answer ONLY using the provided context. 
If the context does not contain the answer, respond exactly: "I don't have that information in my knowledge base."
Always cite the exact FAQ question number in brackets at the end. Example: [Q7]
Keep answers concise, professional, and investor-friendly."""

def parse_thinking(raw_text: str) -> Tuple[Optional[str], str]:
    """Extracts <think> or <think> blocks if present, returns (reasoning, clean_answer)"""
    pattern = r'(?:<think>|<think>)(.*?)(?:</think>|</think>)'
    match = re.search(pattern, raw_text, re.DOTALL | re.IGNORECASE)
    if match:
        reasoning = match.group(1).strip()
        clean = raw_text[:match.start()] + raw_text[match.end():]
        return reasoning, clean.strip()
    return None, raw_text.strip()

def validate_citation(text: str) -> bool:
    return bool(re.search(r'\[Q\d+\]', text, re.IGNORECASE))

def generate_answer(transcript: str, context: str, model_tag: str = "gemma4:e2b") -> dict:
    if not context or context.strip() == "No relevant context found in the knowledge base.":
        return {"answer": "I couldn't find relevant information in the FAQ to answer your question.", 
                "model_used": model_tag, "citation_verified": False, "thinking_chain": None}
        
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Context:\n{context}\n\nUser Query: {transcript}"}
    ]
    
    try:
        logger.info(f"Querying Ollama: {model_tag}")
        response = ollama.chat(model=model_tag, messages=messages, options={"temperature": 0.1})
        raw_text = response["message"]["content"]
        
        reasoning, clean_answer = parse_thinking(raw_text)
        has_citation = validate_citation(clean_answer)
        
        if not has_citation and reasoning:
            clean_answer += " [Citation not explicitly provided, but context was used]"
            
        return {
            "answer": clean_answer,
            "model_used": model_tag,
            "citation_verified": has_citation,
            "thinking_chain": reasoning
        }
        
    except Exception as e:
        logger.error(f"LLM generation failed: {e}")
        return {
            "answer": "Service temporarily unavailable. Please try again later.",
            "model_used": model_tag,
            "citation_verified": False,
            "thinking_chain": None
        }