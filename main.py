import os
import json
import time
import uuid
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
import httpx

load_dotenv()

AI_BACKEND_URL = os.getenv("AI_BACKEND_URL", "http://127.0.0.1:11434").rstrip("/")
EMOTION_ENABLED = os.getenv("EMOTION_ENABLED", "false").lower() in ("1", "true", "yes", "on")

app = FastAPI(title="Cathy AI Service", version="0.1.0")


@app.get("/health")
async def health():
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            await client.get(f"{AI_BACKEND_URL}/api/tags")
            backend_status = "up"
    except Exception:
        backend_status = "down"
    return {
        "ok": True,
        "emotion_enabled": EMOTION_ENABLED,
        "backend_status": backend_status
    }


@app.get("/models")
async def models():
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.get(f"{AI_BACKEND_URL}/api/tags")
            r.raise_for_status()
            data = r.json()
            return {"models": [m["name"] for m in data.get("models", [])]}
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=str(e))
    except httpx.RequestError as e:
        raise HTTPException(status_code=503, detail=f"Backend unavailable: {str(e)}")

@app.get("/models/raw")
async def models_raw():
    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.get(f"{AI_BACKEND_URL}/api/tags")
        r.raise_for_status()
        return r.json()

@app.post("/api/generate")
async def generate(request: Request):
    body = await request.json()
    stream = body.get("stream", True)

    try:
        if not stream:
            timeout = httpx.Timeout(connect=10.0, read=None, write=10.0, pool=10.0)
            async with httpx.AsyncClient(timeout=timeout) as client:
                r = await client.post(f"{AI_BACKEND_URL}/api/generate", json=body)
                r.raise_for_status()
                return r.json()

        timeout = httpx.Timeout(connect=10.0, read=None, write=10.0, pool=10.0)
        client = httpx.AsyncClient(timeout=timeout)

        async def iter_ndjson():
            try:
                async with client.stream("POST", f"{AI_BACKEND_URL}/api/generate", json=body) as r:
                    r.raise_for_status()
                    async for line in r.aiter_lines():
                        if line:
                            yield (line + "\n").encode("utf-8")
            finally:
                await client.aclose()

        return StreamingResponse(iter_ndjson(), media_type="application/x-ndjson")

    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except httpx.RequestError as e:
        raise HTTPException(status_code=503, detail=f"Backend unavailable: {str(e)}")


@app.post("/api/chat")
async def chat(request: Request):
    body = await request.json()
    stream = body.get("stream", True)

    try:
        if not stream:
            timeout = httpx.Timeout(connect=10.0, read=None, write=10.0, pool=10.0)
            async with httpx.AsyncClient(timeout=timeout) as client:
                r = await client.post(f"{AI_BACKEND_URL}/api/chat", json=body)
                r.raise_for_status()
                return r.json()

        timeout = httpx.Timeout(connect=10.0, read=None, write=10.0, pool=10.0)
        client = httpx.AsyncClient(timeout=timeout)

        async def iter_ndjson():
            try:
                async with client.stream("POST", f"{AI_BACKEND_URL}/api/chat", json=body) as r:
                    r.raise_for_status()
                    async for line in r.aiter_lines():
                        if line:
                            yield (line + "\n").encode("utf-8")
            finally:
                await client.aclose()

        return StreamingResponse(iter_ndjson(), media_type="application/x-ndjson")

    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except httpx.RequestError as e:
        raise HTTPException(status_code=503, detail=f"Backend unavailable: {str(e)}")

def _split_provider_model(model_id: str) -> tuple[str, str]:
    """Return (provider, backend_model). Defaults to Ollama for unprefixed models."""
    if ":" not in model_id:
        return "ollama", model_id

    provider, backend_model = model_id.split(":", 1)
    if provider == "ollama":
        return provider, backend_model

    # For now only Ollama is implemented behind /v1.
    # Later we can add llama_cpp:, vllm:, etc.
    raise HTTPException(status_code=400, detail=f"Unsupported model provider: {provider}")


def _ollama_options_from_openai(body: dict) -> dict:
    """Map common OpenAI-style generation options into Ollama options."""
    options = {}

    if body.get("temperature") is not None:
        options["temperature"] = body["temperature"]

    if body.get("top_p") is not None:
        options["top_p"] = body["top_p"]

    if body.get("max_tokens") is not None:
        options["num_predict"] = body["max_tokens"]

    if body.get("stop") is not None:
        options["stop"] = body["stop"]

    return options


@app.get("/v1/models")
async def openai_models():
    """OpenAI-compatible model list for Continue and other local clients."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.get(f"{AI_BACKEND_URL}/api/tags")
            r.raise_for_status()
            data = r.json()

        models = []
        for m in data.get("models", []):
            name = m.get("name") or m.get("model")
            if not name:
                continue

            models.append({
                "id": f"ollama:{name}",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "cathyAI",
            })

        return {
            "object": "list",
            "data": models,
        }

    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except httpx.RequestError as e:
        raise HTTPException(status_code=503, detail=f"Backend unavailable: {str(e)}")


@app.post("/v1/chat/completions")
async def openai_chat_completions(request: Request):
    """OpenAI-compatible chat completions endpoint backed by Ollama."""
    body = await request.json()

    requested_model = body.get("model")
    if not requested_model:
        raise HTTPException(status_code=400, detail="Missing required field: model")

    provider, backend_model = _split_provider_model(requested_model)
    if provider != "ollama":
        raise HTTPException(status_code=400, detail=f"Unsupported provider: {provider}")

    messages = body.get("messages")
    if not isinstance(messages, list):
        raise HTTPException(status_code=400, detail="Missing or invalid required field: messages")

    stream = bool(body.get("stream", False))
    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())

    ollama_body = {
        "model": backend_model,
        "messages": messages,
        "stream": stream,
    }

    options = _ollama_options_from_openai(body)
    if options:
        ollama_body["options"] = options

    timeout = httpx.Timeout(connect=10.0, read=None, write=10.0, pool=10.0)

    if not stream:
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                r = await client.post(f"{AI_BACKEND_URL}/api/chat", json=ollama_body)
                r.raise_for_status()
                data = r.json()

            msg = data.get("message") or {}
            content = msg.get("content") or ""
            reasoning = msg.get("thinking") or ""

            usage = {
                "prompt_tokens": data.get("prompt_eval_count", 0),
                "completion_tokens": data.get("eval_count", 0),
                "total_tokens": (data.get("prompt_eval_count", 0) or 0) + (data.get("eval_count", 0) or 0),
            }

            response_message = {
                "role": "assistant",
                "content": content,
            }

            # Some clients understand this; clients that do not should ignore it.
            if reasoning:
                response_message["reasoning_content"] = reasoning

            return {
                "id": completion_id,
                "object": "chat.completion",
                "created": created,
                "model": requested_model,
                "choices": [
                    {
                        "index": 0,
                        "message": response_message,
                        "finish_reason": data.get("done_reason", "stop"),
                    }
                ],
                "usage": usage,
            }

        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
        except httpx.RequestError as e:
            raise HTTPException(status_code=503, detail=f"Backend unavailable: {str(e)}")

    async def iter_openai_sse():
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                async with client.stream("POST", f"{AI_BACKEND_URL}/api/chat", json=ollama_body) as r:
                    r.raise_for_status()

                    async for line in r.aiter_lines():
                        if not line:
                            continue

                        try:
                            chunk = json.loads(line)
                        except json.JSONDecodeError:
                            continue

                        msg = chunk.get("message") or {}

                        thinking = msg.get("thinking")
                        if thinking:
                            event = {
                                "id": completion_id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": requested_model,
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": {
                                            "reasoning_content": thinking,
                                        },
                                        "finish_reason": None,
                                    }
                                ],
                            }
                            yield f"data: {json.dumps(event)}\n\n"

                        content = msg.get("content")
                        if content:
                            event = {
                                "id": completion_id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": requested_model,
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": {
                                            "content": content,
                                        },
                                        "finish_reason": None,
                                    }
                                ],
                            }
                            yield f"data: {json.dumps(event)}\n\n"

                        if chunk.get("done"):
                            final_event = {
                                "id": completion_id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": requested_model,
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": {},
                                        "finish_reason": chunk.get("done_reason", "stop"),
                                    }
                                ],
                            }
                            yield f"data: {json.dumps(final_event)}\n\n"
                            yield "data: [DONE]\n\n"
                            break

        except httpx.HTTPStatusError as e:
            error_event = {
                "error": {
                    "message": e.response.text,
                    "type": "backend_http_error",
                    "code": e.response.status_code,
                }
            }
            yield f"data: {json.dumps(error_event)}\n\n"
            yield "data: [DONE]\n\n"

        except httpx.RequestError as e:
            error_event = {
                "error": {
                    "message": f"Backend unavailable: {str(e)}",
                    "type": "backend_unavailable",
                }
            }
            yield f"data: {json.dumps(error_event)}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(iter_openai_sse(), media_type="text/event-stream")
