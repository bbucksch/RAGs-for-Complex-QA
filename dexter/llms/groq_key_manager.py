import os
import time
import threading
from collections import deque
from datetime import datetime, timedelta
from typing import Optional, List, Tuple
from dataclasses import dataclass, field

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    from groq import Groq
    _groq_available = True
except ImportError:
    _groq_available = False
    Groq = None


@dataclass
class APIKeyState:
    key: str
    index: int
    request_times: deque = field(default_factory=lambda: deque(maxlen=30))
    cooldown_until: Optional[datetime] = None
    short_cooldown_until: Optional[float] = None
    total_requests: int = 0
    total_errors: int = 0
    consecutive_errors: int = 0
    
    daily_requests: int = 0
    daily_tokens: int = 0
    daily_reset_date: Optional[str] = None
    
    DAILY_REQUEST_LIMIT: int = 14400
    DAILY_TOKEN_LIMIT: int = 500000
    
    def _check_daily_reset(self) -> None:
        today = datetime.now().strftime("%Y-%m-%d")
        if self.daily_reset_date != today:
            self.daily_requests = 0
            self.daily_tokens = 0
            self.daily_reset_date = today
            if self.is_on_daily_cooldown():
                self.cooldown_until = None
    
    def is_on_daily_cooldown(self) -> bool:
        if self.cooldown_until is None:
            return False
        return datetime.now() < self.cooldown_until
    
    def is_daily_limit_reached(self) -> bool:
        self._check_daily_reset()
        return (self.daily_requests >= self.DAILY_REQUEST_LIMIT or 
                self.daily_tokens >= self.DAILY_TOKEN_LIMIT)
    
    def is_on_short_cooldown(self) -> bool:
        if self.short_cooldown_until is None:
            return False
        return time.time() < self.short_cooldown_until
    
    def is_available(self) -> bool:
        self._check_daily_reset()
        return (not self.is_on_daily_cooldown() and 
                not self.is_on_short_cooldown() and 
                not self.is_daily_limit_reached())
    
    def set_daily_cooldown(self, minutes: int = 60):
        self.cooldown_until = datetime.now() + timedelta(minutes=minutes)
        self.short_cooldown_until = None
        
    def set_short_cooldown(self, seconds: float = 3.0):
        self.short_cooldown_until = time.time() + seconds
        self.consecutive_errors += 1
        if self.consecutive_errors >= 3:
            self.short_cooldown_until = time.time() + 10.0
        
    def clear_cooldowns(self):
        self.cooldown_until = None
        self.short_cooldown_until = None
        self.consecutive_errors = 0
    
    def record_success(self, tokens_used: int = 0):
        self._check_daily_reset()
        self.daily_tokens += tokens_used
        self.consecutive_errors = 0
    
    def record_request(self):
        self._check_daily_reset()
        self.request_times.append(time.time())
        self.total_requests += 1
        self.daily_requests += 1
    
    def record_error(self):
        self.total_errors += 1
    
    def get_daily_usage_summary(self) -> str:
        self._check_daily_reset()
        req_pct = (self.daily_requests / self.DAILY_REQUEST_LIMIT) * 100
        tok_pct = (self.daily_tokens / self.DAILY_TOKEN_LIMIT) * 100
        return f"Requests: {self.daily_requests}/{self.DAILY_REQUEST_LIMIT} ({req_pct:.1f}%) | Tokens: {self.daily_tokens}/{self.DAILY_TOKEN_LIMIT} ({tok_pct:.1f}%)"
    
    def get_cooldown_remaining(self) -> float:
        if self.is_on_daily_cooldown():
            return (self.cooldown_until - datetime.now()).total_seconds()
        if self.is_on_short_cooldown():
            return self.short_cooldown_until - time.time()
        return 0


class GroqKeyManager:
    
    def __init__(self, cooldown_minutes: int = 5, short_cooldown_seconds: float = 3.0):
        if not _groq_available:
            raise ImportError("Groq package not installed. Install with: pip install groq")
        
        self.cooldown_minutes = cooldown_minutes
        self.short_cooldown_seconds = short_cooldown_seconds
        self._lock = threading.Lock()
        self._current_index = 0
        
        self.key_states: List[APIKeyState] = []
        self._load_api_keys()
        
        if not self.key_states:
            raise ValueError(
                "No GROQ_API_KEY environment variables found.\n"
                "Add keys to your .env file:\n"
                "  GROQ_API_KEY_1=your_first_key\n"
                "  GROQ_API_KEY_2=your_second_key\n"
                "  etc."
            )
        
        self.clients: List[Groq] = [
            Groq(api_key=state.key, max_retries=0) for state in self.key_states
        ]
        
        print(f"   GroqKeyManager initialized with {len(self.key_states)} API keys")
        print(f"   Mode: FAST rotation (no waiting)")
        print(f"   Rate limit cooldown: {short_cooldown_seconds}s per key")
        print(f"   Daily limit cooldown: {cooldown_minutes} minutes")
    
    def _load_api_keys(self):
        i = 1
        while True:
            key = os.environ.get(f"GROQ_API_KEY_{i}")
            if key and key != "your_groq_api_key_here" and len(key) > 20:
                self.key_states.append(APIKeyState(key=key, index=i))
                i += 1
            else:
                break
        
        if not self.key_states:
            key = os.environ.get("GROQ_API_KEY")
            if key and key != "your_groq_api_key_here" and len(key) > 20:
                self.key_states.append(APIKeyState(key=key, index=0))
    
    def get_next_available_key(self) -> Tuple[int, bool]:
        with self._lock:
            n = len(self.key_states)
            
            for _ in range(n):
                idx = self._current_index
                self._current_index = (self._current_index + 1) % n
                
                if self.key_states[idx].is_available():
                    return idx, True
            
            min_wait = float('inf')
            best_idx = 0
            
            for i, state in enumerate(self.key_states):
                wait = state.get_cooldown_remaining()
                if wait < min_wait:
                    min_wait = wait
                    best_idx = i
            
            return best_idx, False
    
    def wait_for_key_if_needed(self, key_index: int) -> None:
        state = self.key_states[key_index]
        
        if state.is_on_daily_cooldown() or state.is_daily_limit_reached():
            return
        
        if state.is_on_short_cooldown():
            wait_time = min(state.get_cooldown_remaining(), 2.0)
            if wait_time > 0:
                time.sleep(wait_time)
    
    def record_success(self, key_index: int, tokens_used: int = 0):
        with self._lock:
            self.key_states[key_index].record_success(tokens_used)
            
            state = self.key_states[key_index]
            if state.is_daily_limit_reached():
                available = sum(1 for s in self.key_states if s.is_available())
                print(f"\n   Key {key_index + 1} reached DAILY limit!")
                print(f"   {state.get_daily_usage_summary()}")
                print(f"   Keys still available: {available}/{len(self.key_states)}")
    
    def record_request(self, key_index: int):
        with self._lock:
            self.key_states[key_index].record_request()
    
    def record_rate_limit_error(self, key_index: int):
        with self._lock:
            state = self.key_states[key_index]
            state.record_error()
            state.set_short_cooldown(self.short_cooldown_seconds)
    
    def record_daily_limit_error(self, key_index: int):
        with self._lock:
            state = self.key_states[key_index]
            state.record_error()
            state.set_daily_cooldown(self.cooldown_minutes)
            
            available = sum(1 for s in self.key_states if not s.is_on_daily_cooldown())
            cooldown_until = state.cooldown_until.strftime('%H:%M:%S')
            
            print(f"\n   Key {key_index + 1} hit DAILY limit! Cooldown until {cooldown_until}")
            print(f"   Keys still available: {available}/{len(self.key_states)}")
    
    def get_client(self, key_index: int) -> Groq:
        return self.clients[key_index]
    
    def get_status_summary(self) -> str:
        lines = ["API Key Status:"]
        
        for i, state in enumerate(self.key_states):
            state._check_daily_reset()
            
            if state.is_on_daily_cooldown():
                remaining = (state.cooldown_until - datetime.now()).total_seconds() / 60
                status = f"[COOLDOWN] ({remaining:.0f}m remaining)"
            elif state.is_daily_limit_reached():
                status = "[DAILY LIMIT] reached"
            elif state.is_on_short_cooldown():
                remaining = state.short_cooldown_until - time.time()
                status = f"[RATE LIMIT] ({remaining:.1f}s remaining)"
            else:
                status = "[AVAILABLE]"
            
            daily_info = state.get_daily_usage_summary()
            lines.append(f"  Key {i+1}: {status}")
            lines.append(f"         {daily_info}")
        
        return "\n".join(lines)
    
    def get_total_daily_usage(self) -> dict:
        total_requests = 0
        total_tokens = 0
        for state in self.key_states:
            state._check_daily_reset()
            total_requests += state.daily_requests
            total_tokens += state.daily_tokens
        
        max_requests = len(self.key_states) * APIKeyState.DAILY_REQUEST_LIMIT
        max_tokens = len(self.key_states) * APIKeyState.DAILY_TOKEN_LIMIT
        
        return {
            "total_requests": total_requests,
            "total_tokens": total_tokens,
            "max_requests": max_requests,
            "max_tokens": max_tokens,
            "requests_remaining": max_requests - total_requests,
            "tokens_remaining": max_tokens - total_tokens
        }


_key_manager: Optional[GroqKeyManager] = None


def get_key_manager(cooldown_minutes: int = 5, short_cooldown_seconds: float = 3.0) -> GroqKeyManager:
    global _key_manager
    
    if _key_manager is None:
        _key_manager = GroqKeyManager(cooldown_minutes, short_cooldown_seconds)
    
    return _key_manager


def reset_key_manager():
    global _key_manager
    _key_manager = None
