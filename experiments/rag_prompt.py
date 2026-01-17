import re
from typing import List, Tuple, Union


# System instruction for RAG
SYSTEM_INSTRUCTION = """Answer the question using ONLY the provided documents. 

RULES:
- Extract the answer EXACTLY as it appears in the documents
- Answer must be 5 words or fewer
- If the answer is not in any document, respond: NO-RES

Respond with ONLY the answer, nothing else."""


def format_documents(documents: Union[str, List[str], List[dict]]) -> str:
    if isinstance(documents, str):
        # Already formatted
        return documents
    
    formatted_parts = []
    for i, doc in enumerate(documents, 1):
        if isinstance(doc, dict):
            title = doc.get('title', f'Document {i}')
            text = doc.get('text', str(doc))
            formatted_parts.append(f"[Document {i}] {title}\n{text}")
        else:
            formatted_parts.append(f"[Document {i}]\n{doc}")
    
    return "\n\n".join(formatted_parts)


def create_rag_prompt(question: str, documents: Union[str, List[str], List[dict]]) -> Tuple[str, str]:
    formatted_docs = format_documents(documents)
    
    user_prompt = f"""DOCUMENTS:
{formatted_docs}

QUESTION: {question}

Think through this step by step, then provide your final answer.
Remember: Your final answer must be extracted directly from the documents (max 5 tokens).
If the answer is not in the documents, respond with: NO-RES"""

    return SYSTEM_INSTRUCTION, user_prompt


def extract_final_answer(response: str) -> str:
    if not response:
        return ""
    
    response = response.strip()
    
    # Remove common prefixes the model might add anyway
    prefixes_to_remove = [
        r'^(the answer is:?\s*)',
        r'^(answer:?\s*)',
        r'^(response:?\s*)',
        r'^(\[answer\]:?\s*)',
    ]
    
    for pattern in prefixes_to_remove:
        response = re.sub(pattern, '', response, flags=re.IGNORECASE).strip()
    
    # Remove markdown bold markers if present
    response = re.sub(r'\*\*', '', response)
    
    # Take only the first line if multiple lines
    response = response.split('\n')[0].strip()
    
    # Remove trailing punctuation
    response = re.sub(r'[.,!?;:]+$', '', response)
    
    # Truncate to 5 words max
    words = response.split()
    if len(words) > 5:
        response = ' '.join(words[:5])
    
    return response.strip()


def normalize_answer(text: str) -> str:
    if text is None:
        return ""
    # Lowercase
    text = text.lower().strip()
    # Remove punctuation at the end
    text = re.sub(r'[.,!?;:]+$', '', text)
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text


def cover_exact_match(prediction: str, gold: str) -> bool:
    pred_norm = normalize_answer(prediction)
    gold_norm = normalize_answer(gold)
    
    if not gold_norm:
        return False
    
    return gold_norm in pred_norm
