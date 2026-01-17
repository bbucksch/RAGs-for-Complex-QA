import openai
from openai import OpenAI
import os

# Load environment variables from .env file if available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Initialize client with API key (will be None if not set)
_api_key = os.environ.get("OPENAI_KEY")
client = OpenAI(api_key=_api_key) if _api_key and _api_key != "your_openai_api_key_here" else None

class OpenAIEngine:
    def __init__(self, data,model_name: str, temperature: int, top_n: int):
        global client
        if client is None:
            # Try to initialize client if API key is now available
            _api_key = os.environ.get("OPENAI_KEY")
            if _api_key and _api_key != "your_openai_api_key_here":
                client = OpenAI(api_key=_api_key)
            else:
                raise ValueError("OPENAI_KEY environment variable not set. Please set it in your .env file.")
        self.model_name = model_name
        self.temperature = temperature
        self.top_n = top_n
        self.data = data
        self.client = client
    def get_completion(self, prompt: str) -> str:
        response = self.client.Completion.create(
            model="text-davinci-003",
            prompt="""You are a assistant that gives an answer along with the derivation of rationale in format Rationale: Answer:  to arrive at solution for questions mandatorily using information from both given table and text. In table columns are separated by | and rows by \n (newline). If you dont know the answer output UNKNOWN: \n
                      give best answer with accurate scale and precision for Question: """ + row["question"]+",Table: "+row["table"]+ "Text: "+row["text"]+"Output in format Rationale:, Answer:",
            temperature=self.temperature,
            max_tokens=2048,
            top_p=1.0,
            n = self.top_n,
            frequency_penalty=0.8,
            presence_penalty=0.6
            ) 
        return response['choices'][0]['text']     
    def get_chat_completion(self, user_prompt: str, system_prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role":"system","content":system_prompt } ,
    {"role":"user","content": user_prompt,
    }] ,
            temperature=self.temperature,
            max_tokens=1000,
            top_p=1.0,
            n=self.top_n,
            frequency_penalty=0.8,
            presence_penalty=0.6,
            )
        return response.choices[0].message.content