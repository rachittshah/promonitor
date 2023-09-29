import requests
from openai_logger import MODEL_COST_PER_1K_TOKENS, get_openai_token_cost_for_model, standardize_model_name

API_BASE_URL = "http://localhost:8000"

# Log the request and response from OpenAI chat completion to Magik API
#
# prompt_slug: name of the prompt for analytics and grouping
# messages: "messages" json array sent to OpenAI chat completion
# model: string id of the language model used (ex: gpt-3.5-turbo)
# chat_completion: "choices" json response from OpenAI chat completion
# response_time: response_time in milliseconds (optional)
# context: any structured json data you wish to associate with the prompt response (useful for analytics and testing)
def log_open_ai_chat_response(
    prompt_slug,
    messages,
    model,
    completion,
    response_time=None,
    context=None,
    environment=None,
    customer_id=None,
    customer_user_id=None,
    session_id=None,
    user_query=None,
    prompt_tokens= None,
    completion_tokens=None,
    total_tokens=None
):
    """
    Track the request and response.
    """
    # Calculate cost
    total_token = total_tokens if total_tokens is not None else 0
    model_name = standardize_model_name(model)
    cost = get_openai_token_cost_for_model(model_name, total_token)
    
    """
    Track the request and response.
    """
    payload = {
        "prompt_slug": prompt_slug,
        "prompt_messages": messages,
        "language_model_id": model,
        "completion": completion,
        "response_time": response_time,
        "context": context,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "environment": environment,
        "customer_id": str(customer_id),
        "customer_user_id": str(customer_user_id),
        "session_id": str(session_id),
        "user_query": str(user_query),
        "cost": cost,
    }
    # Remove None fields from the payload
    payload = {k: v for k, v in payload.items() if v is not None}
    requests.post(
        f"{API_BASE_URL}/log/openai/chat/{prompt_slug}",
        json=payload,
        # headers removed as per instructions
    )


# Log the request and response from OpenAI completion endpoint to Magik API
#
# prompt_slug: name of the prompt for analytics and grouping
# message: the prompt string sent to OpenAI completion endpoint
# model: string id of the language model used (ex: text-davinci-003)
# completion: json response from OpenAI completion endpoint
# response_time: response_time in milliseconds (optional)
# context: any structured json data you wish to associate with the prompt response (useful for analytics and testing)
def log_open_ai_completion_response(
    prompt_slug: str,
    prompt: str,
    model: str,
    completion,
    response_time=None,
    context=None,
    environment=None,
    customer_id=None,
    customer_user_id=None,
    session_id=None,
    user_query=None,
    prompt_tokens= None,
    completion_tokens=None,
    total_tokens=None
):
    """
    Track the request and response.
    """
    # Calculate cost
    total_token = total_tokens if total_tokens is not None else 0
    model_name = standardize_model_name(model)
    cost = get_openai_token_cost_for_model(model_name, total_token)
    
    payload = {
        "prompt_slug": prompt_slug,
        "prompt_text": prompt,
        "language_model_id": model,
        "completion": completion,
        "response_time": response_time,
        "context": context,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "environment": environment,
        "customer_id": str(customer_id),
        "customer_user_id": str(customer_user_id),
        "session_id": str(session_id),
        "user_query": str(user_query),
        "cost": cost,
    }
    # Remove None fields from the payload
    payload = {k: v for k, v in payload.items() if v is not None}
    requests.post(
        f"{API_BASE_URL}/log/openai/completion/{prompt_slug}", 
        json=payload
    )


# Log a generic llm response (not specific to any provider)
#
# prompt_slug: name of the prompt for analytics and grouping
# message: the prompt string used to generate this response
# llm_response: The string response you received from the LLM used
# response_time: response_time in milliseconds (optional)
# context: any structured json data you wish to associate with the prompt response (useful for analytics and testing)
def log_generic_response(
    prompt_slug: str,
    prompt: str,
    llm_response: str,
    prompt_tokens: None,
    completion_tokens: None,
    total_tokens: None,
    response_time=None,
    context=None,
    environment=None,
    customer_id=None,
    customer_user_id=None,
    session_id=None,
    user_query=None,
):
    payload = {
        "prompt_slug": prompt_slug,
        "prompt_text": prompt,
        "language_model_id": "generic",
        "prompt_response": llm_response,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "response_time": response_time,
        "context": context,
        "environment": environment,
        "customer_id": str(customer_id),
        "customer_user_id": str(customer_user_id),
        "session_id": str(session_id),
        "user_query": str(user_query),
    }
    # Remove None fields from the payload
    payload = {k: v for k, v in payload.items() if v is not None}
    requests.post(
        f"{API_BASE_URL}/api/v1/log/prompt/generic",
        json=payload
    )


def log_langchain_llm_response(
    prompt_slug,
    prompt_sent,
    prompt_response,
    model,
    response_time,
    prompt_tokens: None,
    completion_tokens: None,
    total_tokens: None,
    context=None,
    environment=None,
    customer_id=None,
    customer_user_id=None,
    session_id=None,
    user_query=None,
):
    """
    Track the request and response.
    """
    # Calculate cost
    total_token = total_tokens if total_tokens is not None else 0
    model_name = standardize_model_name(model)
    cost = get_openai_token_cost_for_model(model_name, total_token)
    
    """
    Track the request and response.
    """
    payload = {
        "prompt_slug": prompt_slug,
        "prompt_sent": prompt_sent,
        "language_model_id": model,
        "prompt_response": prompt_response,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "response_time": response_time,
        "context": context,
        "environment": environment,
        "customer_id": str(customer_id),
        "customer_user_id": str(customer_user_id),
        "session_id": str(session_id),
        "user_query": str(user_query),
    }
    # Remove None fields from the payload
    payload = {k: v for k, v in payload.items() if v is not None}
    requests.post(
        f"{API_BASE_URL}/log/langchain/{prompt_slug}",
        json=payload
    )
