import os
import queue
import threading
import time
from typing import Optional, Any
import redis

class FallbackRedis:
    """Thread-safe in-memory Redis substitute when external Redis is unavailable."""
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._store = {}
                cls._instance._queues = {}
                cls._instance._store_lock = threading.Lock()
            return cls._instance

    def set(self, name: str, value: Any, ex: Optional[int] = None, **kwargs):
        with self._store_lock:
            self._store[name] = str(value)
        return True

    def get(self, name: str) -> Optional[str]:
        with self._store_lock:
            return self._store.get(name)

    def lpush(self, name: str, *values) -> int:
        with self._store_lock:
            if name not in self._queues:
                self._queues[name] = queue.Queue()
            q = self._queues[name]
        for v in values:
            q.put(str(v))
        return q.qsize()

    def brpop(self, keys, timeout: int = 0):
        if isinstance(keys, str):
            keys = [keys]
        deadline = time.time() + (timeout if timeout else 0)
        while True:
            for k in keys:
                with self._store_lock:
                    q = self._queues.get(k)
                if q:
                    try:
                        val = q.get_nowait()
                        return (k, val)
                    except queue.Empty:
                        pass
            if timeout and time.time() >= deadline:
                return None
            time.sleep(0.1)

    def ping(self) -> bool:
        return True

def get_redis_client():
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    if redis_url.startswith("memory://"):
        return FallbackRedis()
    try:
        client = redis.from_url(redis_url, decode_responses=True, socket_connect_timeout=1)
        client.ping()
        return client
    except Exception as e:
        print(f"[Storage] Redis not accessible at {redis_url} ({e}). Falling back to in-memory store.")
        return FallbackRedis()

