from lib.ml.document_classifier.classifier import get_document_classifier
import hashlib
import json
import asyncio
from utils import redis_client
from datetime import datetime as dt
from zoneinfo import ZoneInfo
from lib.file_utils import upload, cap_doc_text

LABEL_DEFINITIONS = {
    "Public": "Public, safe for external marketing and anyone to read",
    "Internal": "Internal company announcements, team building, and safe employee information",
    "Confidential": "Confidential business data, financial reports, and executive-level secrets",
    "Restricted": "Highly restricted IT architecture, root passwords, and severe security risks"
}

classifier = get_document_classifier()

def classify_doc(doc_url: str):
    """
    Classifies a document text into predefined compliance categories.
    
    Args:
        doc_url (str): The document url to serve as unique identifier for caching, as well as fetch doc text.
        
    Returns:
        dict: A dictionary containing the winning 'category' and its 'confidence' score.
    """
    raw_key = f"compliance_{doc_url}"
    cache_key = hashlib.sha256(raw_key.encode('utf-8')).hexdigest()
    
    cached = redis_client.get(cache_key)
    if cached:
        return json.loads(cached)
            
    async def fetch_text():
        text_content, _ = await upload(doc_url)
        return await cap_doc_text(text_content, max_tokens=1500)

    text = asyncio.run(fetch_text())
    doc_class = classifier.predict(text, LABEL_DEFINITIONS)
    
    redis_client.set(cache_key, json.dumps(doc_class))
    
    return doc_class

async def update_category(doc_url: str, category: str) -> dict:
    """
    Manually overrides the compliance category for a document in the cache and stores details in MongoDB.

    Args:
        doc_url (str): The document url used to generate the cache key.
        category (str): The new compliance category to override with.

    Returns:
        dict: The updated document classification details.
    """

    raw_key = f"compliance_{doc_url}"
    cache_key = hashlib.sha256(raw_key.encode('utf-8')).hexdigest()
    
    doc_class = {
        "category": category,
        "confidence": 1.0,
        "updated": True
    }
    redis_client.set(cache_key, json.dumps(doc_class))
    return doc_class
