import os
from uuid import uuid4
from urllib.parse import urlparse
from utils import *
import httpx, asyncio, aiofiles
import tempfile
import html
import re
from dataclasses import dataclass
from typing import Any
import json

from fastapi import HTTPException
from httpx import RequestError
from aiofiles.tempfile import NamedTemporaryFile, SpooledTemporaryFile
from .parser import parse
import tiktoken

class FileUtilities:
    """
    A utility class for handling file operations like downloading, temporarily storing files,
    determining if a PDF is searchable, and estimating processing time.
    """

    @staticmethod
    async def download_file(url):
        """
        Download a file from a given URL.

        Args:
            url (str): The URL of the file to download.

        Returns:
            bytes: The content of the downloaded file.
            str: The filename extracted from the URL.

        Raises:
            HTTPException: If there is an error downloading the file.
        """
        # Download the file from the URL
        max_attempts = 3
        backoff_seconds = 0.5
        logger.info(f"Downloading file from URL: {url}")

        for attempt in range(1, max_attempts + 1):
            try:
                async with httpx.AsyncClient() as temp_client:
                        file_response = await temp_client.get(url, timeout=httpx.Timeout(60.0, connect=30.0))
                file_response.raise_for_status()
                file_bytes = file_response.content

                file_name = os.path.basename(urlparse(url).path)
                logger.info(f"File downloaded successfully: {file_name}")
                return file_bytes, file_name

            except RequestError as e:
                if attempt == max_attempts:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Error downloading file after {max_attempts} attempts: {str(e)}",
                    )
                wait = backoff_seconds * (2 ** (attempt - 1))
                logger.warning(f"Download attempt {attempt} failed . Retrying in {wait:.1f}s. Error: {e}")
                await asyncio.sleep(wait)

    @staticmethod
    async def store_file(file_bytes, file_name, temp_dir="../temp_file"):
        """
        Store the downloaded file bytes to disk using the original file name.

        Args:
            file_bytes (bytes): The content of the file to store.
            file_name (str): The original filename to use for storing the file.
            temp_dir (str): The directory to store the temporary file. Default is "../temp_file".
            max_temp_file_size (int): Maximum size for the temporary file in bytes. Default is 10MB.

        Returns:
            str: The path to the stored file.
        """

        # Generate the full file path
        file_path = os.path.join(temp_dir, file_name)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        async with aiofiles.open(file_path, "wb") as f:
            await f.write(file_bytes)


        return file_path


    @staticmethod
    async def download_and_store_file(url, temp_dir="../temp_file", max_temp_file_size=30 * 1024 * 1024):
        """
        Download a file from a given URL and store it temporarily.

        Args:
            url (str): The URL of the file to download.
            temp_dir (str): The directory to store the temporary file. Default is "../temp_file".
            max_temp_file_size (int): Maximum size for the temporary file in bytes. Default is 10MB.

        Returns:
            str: The path to the stored file.

        Raises:
            HTTPException: If there is an error downloading the file.
        """
        # Download the file
        file_bytes, file_name = await FileUtilities.download_file(url)
        if len(file_bytes) > max_temp_file_size:
            raise HTTPException(status_code=413, detail="File too large")

        # Store the file to disk
        base_, ext = os.path.splitext(file_name)
        unique_name = f"{base_}{uuid4().hex}{ext or '.pdf'}"
        return await FileUtilities.store_file(file_bytes, unique_name, temp_dir)

    @staticmethod
    async def cleanup_temp_file(file_path):
        """
        Clean up (delete) a temporary file if it exists.

        Args:
            file_path (str): The path to the temporary file to delete.
        """
        if os.path.exists(file_path):
            logger.info(f"Deleting temporary file: {file_path}...")
            os.remove(file_path)


async def upload(document_url: str, ex = None)-> tuple[str, bool]:
    """
    Download, parse, and cache a document by URL.
    Returns cached content when available and a flag indicating cache usage.

    Args:
        document_url (str): The URL of the document to download and parse.

    Returns:
        tuple[str, bool]: The parsed document content and a cache-hit flag.
    """
    if not document_url:
        raise ValueError("Cannot parse invalid document url")
    try:
        cached_data = redis_client.get(document_url)
        if cached_data:
            logger.info(f"Document extraction skipped. {document_url} previously extracted")
            return cached_data, True
        
        logger.info('\nPreparing to download upload pdf')
        temp_file_path =await FileUtilities.download_and_store_file(document_url)

        text = await parse(temp_file_path)
        redis_client.set(document_url, text, ex = ex)

        await FileUtilities.cleanup_temp_file(temp_file_path)
        return text, False

    except Exception as e:
        raise e

async def cap_doc_text(text: str, max_tokens: int, head_ratio: float = 0.6) -> str:
    """
    Truncate a long document to a head/tail window based on token count.
    Returns the original text if it is within the token limit.

    Args:
        text (str): The input document text.
        max_tokens (int): Maximum number of tokens to keep.
        head_ratio (float, optional): Fraction of tokens kept from the head. Defaults to 0.6.

    Returns:
        str: The capped document text.
    """
    enc = tiktoken.get_encoding("o200k_base")
    ids = enc.encode(text)
    if len(ids) <= max_tokens:
        return text
    
    logger.warning(f"Document text exceeds {max_tokens} tokens. Capping to most relevant sections.")
    head = int(max_tokens * head_ratio)
    tail = max_tokens - head
    return f"{enc.decode(ids[:head])}\n....\n{enc.decode(ids[-tail:])}"


