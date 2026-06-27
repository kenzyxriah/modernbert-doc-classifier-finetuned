from fastapi import Header, HTTPException
from decouple import config

def verify_api_key(x_api_key: str = Header(...)):
    """
    Authenticate the provided API key.

    :param api_key: API key provided by the user.
    :return: Boolean indicating if the API key is valid.
    """
    if x_api_key != config("X_API_KEY"):
        raise HTTPException(status_code=403, detail="Invalid API key")
    return x_api_key
