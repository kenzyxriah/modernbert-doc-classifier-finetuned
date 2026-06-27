import sys
import asyncio
import uvicorn
from utils import logger

if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

if __name__ == "__main__":
    logger.info('Starting server................................')
    uvicorn.run("main:app", host="127.0.0.1", port=8010, reload=False)
    logger.info('Started server at http://localhost:8010')
    