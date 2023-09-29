from fastapi import FastAPI, BackgroundTasks
from langchain_handler import CallbackHandler
from logger import log_open_ai_chat_response, log_open_ai_completion_response, log_langchain_llm_response
import requests

app = FastAPI()

@app.post("/log/openai/chat/{prompt_slug}")
async def log_openai_chat(prompt_slug: str, log_data: dict, background_tasks: BackgroundTasks):
    # Add the task of logging the OpenAI chat response to the background tasks
    background_tasks.add_task(log_open_ai_chat_response, prompt_slug, **log_data)
    return {"message": "OpenAI chat response received and will be logged"}

@app.post("/log/openai/completion/{prompt_slug}")
async def log_openai_completion(prompt_slug: str, log_data: dict, background_tasks: BackgroundTasks):
    # Add the task of logging the OpenAI completion response to the background tasks
    background_tasks.add_task(log_open_ai_completion_response, prompt_slug, **log_data)
    return {"message": "OpenAI completion response received and will be logged"}

@app.post("/log/langchain/{prompt_slug}")
async def log_langchain(prompt_slug: str, log_data: dict, background_tasks: BackgroundTasks):
    # Add the task of logging the LangChain LLM response to the background tasks
    background_tasks.add_task(log_langchain_llm_response, prompt_slug, **log_data)
    return {"message": "LangChain LLM response received and will be logged"}