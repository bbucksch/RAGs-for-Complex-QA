import os
import time
from typing import Optional, Tuple

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    import tiktoken
    _tiktoken_available = True
    _tokenizer = tiktoken.get_encoding("cl100k_base")
except ImportError:
    _tiktoken_available = False
    _tokenizer = None

try:
    from groq import Groq
    _groq_available = True
except ImportError:
    _groq_available = False
    Groq = None


def count_tokens(text: str) -> int:
    if _tiktoken_available and _tokenizer:
        return len(_tokenizer.encode(text))
    else:
        return max(1, len(text) // 4)


class GroqEngine:
    
    DEFAULT_MODEL = "llama-3.3-70b-versatile"
    DEFAULT_MAX_TOKENS = 30
    
    def __init__(self, data, model_name: str = None, temperature: float = 0.7, 
                 top_n: int = 1, max_tokens: int = None, paid_mode: bool = True):
        if not _groq_available:
            raise ImportError(
                "Groq package not installed. Install it with: pip install groq"
            )
        
        self.model_name = model_name or self.DEFAULT_MODEL
        self.temperature = temperature
        self.top_n = top_n
        self.max_tokens = max_tokens or self.DEFAULT_MAX_TOKENS
        self.data = data
        
        paid_key = os.environ.get("PAID_API_KEY", "")
        if not paid_key or paid_key == "your_paid_groq_api_key_here":
            raise ValueError(
                "No valid PAID_API_KEY found in .env file!\n"
                "Set PAID_API_KEY to your Groq API key."
            )
        
        print(f"\n   Using Groq API with paid key (no rate limits)")
        print(f"   Model: {self.model_name}")
        self.client = Groq(api_key=paid_key)
    
    def _make_request(self, messages: list, max_retries: int = 5) -> Tuple[str, Optional[dict]]:
        last_error = None
        
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    top_p=1.0,
                    n=self.top_n,
                    frequency_penalty=0.0,
                    presence_penalty=0.0,
                )
                
                # Extract usage information
                usage = None
                if hasattr(response, 'usage') and response.usage:
                    usage = {
                        'prompt_tokens': response.usage.prompt_tokens,
                        'completion_tokens': response.usage.completion_tokens,
                        'total_tokens': response.usage.total_tokens
                    }
                
                return response.choices[0].message.content, usage
                
            except Exception as e:
                last_error = e
                error_str = str(e).lower()
                
                # Server errors - wait and retry
                if any(code in str(e) for code in ["500", "502", "503", "504"]):
                    wait_time = min(1.0 * (attempt + 1), 5.0)
                    print(f"   Server error, retrying in {wait_time:.1f}s...")
                    time.sleep(wait_time)
                    continue
                
                # Rate limit - wait and retry
                if "rate" in error_str or "429" in str(e):
                    wait_time = 2.0 * (attempt + 1)
                    print(f"   Rate limit hit, waiting {wait_time:.1f}s...")
                    time.sleep(wait_time)
                    continue
                
                # Other errors - small backoff
                wait_time = 0.5 * (attempt + 1)
                time.sleep(wait_time)
        
        raise last_error or Exception("API request failed after max retries")
    
    def get_chat_completion(self, user_prompt: str, system_prompt: str, verbose: bool = False) -> str:
        if verbose:
            full_prompt = system_prompt + user_prompt
            input_tokens = count_tokens(full_prompt)
            print(f"   Input tokens: {input_tokens}")
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        response_text, usage = self._make_request(messages)
        
        return response_text
    
    def get_completion(self, prompt: str) -> str:
        return self.get_chat_completion(
            user_prompt=prompt,
            system_prompt="You are a helpful assistant. Answer the question accurately and concisely."
        )
    
    def answer_question(self, question: str, context: str) -> str:
        system_prompt = """You are a helpful assistant that answers questions based on the provided context.
Use only the information from the context to answer. If the answer is not in the context, say "I cannot find the answer in the provided context."
Be concise and accurate."""

        user_prompt = f"""Context:
{context}

Question: {question}

Answer:"""

        return self.get_chat_completion(user_prompt, system_prompt)
