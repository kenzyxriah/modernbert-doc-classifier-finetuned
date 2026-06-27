import logging
from logging import Logger
import re, json
from redis import Redis

def setup_logger()-> Logger:
    # Create main logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # Console handler with timestamp, level
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter(f"%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Suppress noisy third-party loggers
    logging.getLogger("google_genai").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    # --- Uvicorn access log suppression ---
    access_logger = logging.getLogger("uvicorn.access")
    access_logger.setLevel(logging.INFO)

    # Optionally quiet the Uvicorn error logger
    logging.getLogger("uvicorn.error").setLevel(logging.INFO)

    return logger


logger = setup_logger()


redis_client = Redis(
    host='localhost',
    port= 6379,
    db=0,
    decode_responses=True)


def get_session(session_id: str) -> list:
    session_data = redis_client.get(f"session:{session_id}")

    if not session_data:
        return 0, None

    msgs = json.loads(session_data)

    return len(msgs), msgs


def set_session(session_id: str, history: list, ex: int = None):
    redis_client.set(f"session:{session_id}", json.dumps(history),
                     ex = ex) 

