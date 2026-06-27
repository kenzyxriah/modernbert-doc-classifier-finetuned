from dotenv import load_dotenv
load_dotenv()

from dataclasses import dataclass
from openai import AsyncAzureOpenAI
from openai.types.chat import ChatCompletionMessage, ChatCompletionMessageToolCall
from pydantic import BaseModel
from typing import TypeVar, Type

from decouple import config
T = TypeVar("T", bound=BaseModel)

def format_message(msg):
    
    if isinstance(msg, list) and len(msg) > 0 and isinstance(msg[0], dict) and "role" in msg[0]:
        return msg
    
    if isinstance(msg, dict) and "role" in msg and "content" in msg:
        return [msg]
    
    if isinstance(msg, str):
        return [{"role": "user", "content": msg}]

    elif isinstance(msg, list) and all(isinstance(i, dict) and "type" in i for i in msg):
        return [{"role": "user", "content": msg}]

    elif isinstance(msg, list) and all(isinstance(i, str) for i in msg):
        return [{"role": "user", "content": [{"type": "text", "text": i} for i in msg]}]

    else:
        raise TypeError(f"Unsupported message type: {type(msg)}")

@dataclass
class Azure:
    azure_endpoint = config("AZURE_OPENAI_ENDPOINT")
    model: str = config("AZURE_OPENAI_DEPLOYMENT_NAME")
    azure_subscription_key = config("AZURE_OPENAI_API_KEY")

    client = AsyncAzureOpenAI(
        azure_endpoint=azure_endpoint,
        api_key=azure_subscription_key,
        api_version= config("AZURE_OPENAI_API_VERSION"),
    )

client = Azure.client
async def structured_output(
        system_prompt:str , user_msg: str | list[dict], response_format: Type[T], temperature=0.5, max_tokens=4096, **kwargs
    ):
        """
        Generate a structured response from the chat model using a specified response schema.

        Args:
            user_msg (str | list[dict]): The user input to send to the model. Can be a raw string or a list of message dicts in OpenAI chat format.
            response_format (Type[T]): A Pydantic model (or compatible schema class) defining the expected structure of the response.
            temperature (float, optional): Sampling temperature for response generation. Lower values make the output more deterministic. Defaults to 0.5.
            max_tokens (int, optional): Maximum number of tokens to generate in the response. Defaults to 4096.

            **kwargs: Additional keyword arguments to pass to the completion API.

                - top_p (float, optional): Top-p nucleus sampling parameter. Defaults to None.
                - extra_body (dict, optional): Additional request body fields for advanced API usage. Defaults to None.

        Raises:
            Exception: If the model refuses to comply with the prompt (`msg.refusal` is set).
            TypeError: If `user_msg` is not a string or list of dicts.

        Returns:
            dict: The structured response content as a standard Python dictionary, converted
            from the parsed schema instance.
        """
        use_groq = kwargs.pop("use_groq", True)
        if "tools" in kwargs or "tools_schema" in kwargs:
            raise ValueError("Cannot use tools with response_format")

        messages = []
        if isinstance(user_msg, str) or not (user_msg[0].get('role') or "").lower() == "system":
            messages.extend([{"role": "system", "content": system_prompt}])
        
        messages.extend(format_message(user_msg))
        try:
            completion = await client.beta.chat.completions.parse(
                model=Azure.model,
                temperature=temperature,
                max_tokens=max_tokens,
                messages=messages,
                response_format=response_format,
                **kwargs,
            )

            msg = completion.choices[0].message

            if msg.refusal:
                raise Exception(f"Model refusal: {msg.refusal}")

            parsed: T = msg.parsed
            return parsed.model_dump()
    
        except Exception as e:
            raise e
