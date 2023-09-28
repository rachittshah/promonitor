from fastapi import FastAPI, BackgroundTasks
from .langchain_handler import CallbackHandler
import requests
from typing import Dict

app = FastAPI()

@app.post("/log/{logger_id}")
async def log_data(logger_id: str, background_tasks: BackgroundTasks, log_data: Dict[str, Dict[str, Any]]):
    # Add the task of forwarding the log data to the frontend to the background tasks
    background_tasks.add_task(forward_log_to_frontend, logger_id, log_data)
    return {"message": "Log data received and will be forwarded to the frontend"}

def forward_log_to_frontend(logger_id: str, log_data: Dict[str, Dict[str, Any]]):
    # This function will run in the background and forward the log data to the frontend
    frontend_url = "http://frontend-service-url/logs"  # Replace with your frontend service URL
    handler = CallbackHandler(logger_id)
    run_data = handler.runs.get(logger_id, {})
    response = requests.post(frontend_url, json={"logger_id": logger_id, "log_data": run_data})

    if response.status_code != 200:
        print(f"Failed to forward log data to frontend: {response.text}")