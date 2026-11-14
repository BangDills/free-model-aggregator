#!/usr/bin/env python3
"""free-model-aggregator-01: the best free AI model aggregator.

One server, two client wires -- the OpenAI Chat Completions API and the Anthropic Messages API,
each in JSON and in server-sent events -- over the ten simulated free-tier providers a trial's
brief names.  It reads the brief for the provider directory, the sealed quality table and the
workload manifest, registers the catalog it read, routes every request to the highest-rated free
model that meets the request, fails over to the next best candidate when a provider refuses, and
exits 0 when the env says the workload is done.

Python 3.13 standard library only.  Every choice is a function of the brief, the requests and the
providers' answers: no wall clock and no unseeded randomness, so a replay leaves the same export.
"""

import json
import os
import sys
import tempfile
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PROTOCOL_VERSION = 1
FIXED_CREATED = 1767225600
FIXED_ANTHROPIC_CREATED_AT = "2026-01-01T00:00:00Z"
ANSWER_PREFIX = "ANSWER ECHO"
DELTA_CHARS = 13
POLL_SECONDS = 0.05
UPSTREAM_TIMEOUT = 25.0
MAX_BODY_BYTES = 32 * 1024 * 1024
ENV_URL = (os.environ.get("MAKEFUN_ENV_URL") or "").rstrip("/")
PROMPT_FILE = os.environ.get("MAKEFUN_PROMPT_FILE") or ""


def log(message):
    sys.stderr.write("aggregator: %s\n" % message)
    sys.stderr.flush()


# --------------------------------------------------------------------------- the brief

def json_documents(text):
    """Every JSON object the brief holds, in the order the brief holds them."""
    decoder = json.JSONDecoder()
    found = []
    at = 0
    while True:
        at = text.find("{", at)
        if at < 0:
            return found
        try:
            value, end = decoder.raw_decode(text, at)
        except ValueError:
            at += 1
            continue
        if isinstance(value, dict):
            found.append(value)
        at = max(end, at + 1)


def read_brief(path):
    brief = {"directory": None, "ratings": None, "workload": None, "error": None}
    if not path:
        brief["error"] = "MAKEFUN_PROMPT_FILE is not set: there is no brief to read"
        return brief
    try:
        with open(path, "r", encoding="utf-8") as handle:
            text = handle.read()
    except OSError as err:
        brief["error"] = "the brief cannot be read: %s" % err
        return brief
    for value in json_documents(text):
        schema = value.get("schema")
        if schema == "free-model-aggregator-01/providers/v1":
            brief["directory"] = value
        elif schema == "free-model-aggregator-01/ratings/v1":
            brief["ratings"] = value
        elif schema == "free-model-aggregator-01/workload/v1":
            brief["workload"] = value
    missing = [name for name in ("directory", "ratings", "workload") if brief[name] is None]
    if missing:
        brief["error"] = "the brief holds no %s document" % ", ".join(missing)
    return brief


def brief_providers(brief):
    directory = brief["directory"] or {}
    return [p for p in (directory.get("providers") or []) if isinstance(p, dict) and isinstance(p.get("id"), str)]


def free_rows(brief):
    ratings = brief["ratings"] or {}
    return [r for r in (ratings.get("ratings") or []) if isinstance(r, dict) and r.get("free") is True]


def build_catalog(brief):
    """The one catalog document: the directory's providers, each with its free models."""
    entries = []
    by_id = {}
    for provider in brief_providers(brief):
        entry = {
            "id": provider["id"],
            "base_url": provider.get("path") or "",
            "key_env": provider.get("key_env") or "",
            "wire": provider.get("wire") or "openai",
            "models": [],
        }
        entries.append(entry)
        by_id[entry["id"]] = entry
    for row in free_rows(brief):
        entry = by_id.get(row.get("provider"))
        if entry is None:
            continue
        entry["models"].append({
            "id": row.get("model"),
            "free": True,
            "context_length": row.get("context_length"),
        })
    return {"providers": entries}


def served_models(catalog):
    """Every free model the catalog holds, in its order: what GET /v1/models lists."""
    out = []
    for entry in catalog["providers"]:
        for model in entry["models"]:
            out.append((entry["id"], model["id"]))
    return out


# --------------------------------------------------------------------------- wire helpers

def blocks_text(content):
    """The text of a string, or of a block array's text blocks joined with a newline."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text" and isinstance(block.get("text"), str):
            parts.append(block["text"])
    return "\n".join(parts)


def string_list(value):
    if isinstance(value, str):
        return [value]
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)]


def number_or_none(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return value


def is_json_mode(body):
    fmt = body.get("response_format")
    if not isinstance(fmt, dict):
        return False
    return fmt.get("type") in ("json_object", "json_schema")


def tool_result_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text = blocks_text(content)
        return text if text != "" else json.dumps(content)
    if content is None:
        return ""
    return json.dumps(content)


def arguments_object(value):
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip() != "":
        try:
            parsed = json.loads(value)
        except ValueError:
            return {}
        if isinstance(parsed, dict):
            return parsed
    return {}


def image_source_of_url(url):
    """An OpenAI image_url url as an Anthropic image source, or None."""
    if not isinstance(url, str) or url == "":
        return None
    if url.startswith("data:"):
        head, sep, data = url.partition(",")
        if sep != "" and head.endswith(";base64"):
            media = head[len("data:"):-len(";base64")] or "image/png"
            return {"type": "base64", "media_type": media, "data": data}
    return {"type": "url", "url": url}


def image_url_of_source(source):
    """An Anthropic image source as an OpenAI image_url url, or None."""
    if not isinstance(source, dict):
        return None
    if source.get("type") == "base64" and isinstance(source.get("data"), str):
        media = source.get("media_type") if isinstance(source.get("media_type"), str) else "image/png"
        return "data:%s;base64,%s" % (media, source["data"])
    if source.get("type") == "url" and isinstance(source.get("url"), str):
        return source["url"]
    return None


def gemini_parts(content, source_wire):
    """One message's content as Gemini parts, preserving text and images."""
    parts = []
    if isinstance(content, str):
        return [{"text": content}]
    if not isinstance(content, list):
        return parts
    for block in content:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text" and isinstance(block.get("text"), str):
            parts.append({"text": block["text"]})
        elif source_wire == "openai" and block.get("type") == "image_url" and isinstance(block.get("image_url"), dict):
            url = block["image_url"].get("url")
            source = image_source_of_url(url)
            if source is None:
                continue
            if source["type"] == "base64":
                parts.append({"inlineData": {"mimeType": source["media_type"], "data": source["data"]}})
            else:
                parts.append({"fileData": {"fileUri": source["url"]}})
        elif source_wire == "anthropic" and block.get("type") == "image":
            source = block.get("source")
            url = image_url_of_source(source)
            if url is None:
                continue
            if isinstance(source, dict) and source.get("type") == "base64":
                parts.append({"inlineData": {"mimeType": source.get("media_type") or "image/png", "data": source.get("data")}})
            else:
                parts.append({"fileData": {"fileUri": url}})
    return parts


def auth_headers(auth, key):
    if auth == "bearer":
        return {"authorization": "Bearer %s" % key}
    if auth == "x-api-key":
        return {"x-api-key": key}
    if auth == "x-goog-api-key":
        return {"x-goog-api-key": key}
    return {}


def wire_route(wire, model):
    if wire == "openai":
        return "/v1/chat/completions"
    if wire == "anthropic":
        return "/v1/messages"
    return "/v1beta/models/%s:generateContent" % model


# --------------------------------------------------------------------------- client -> provider

def openai_to_anthropic(body, model):
    out = {"model": model}
    messages = body.get("messages") if isinstance(body.get("messages"), list) else []
    system_parts = []
    converted = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        content = message.get("content")
        if role == "system":
            text = blocks_text(content)
            if text != "":
                system_parts.append(text)
            continue
        if role == "user":
            if isinstance(content, str):
                converted.append({"role": "user", "content": content})
            else:
                converted.append({"role": "user", "content": gemini_parts(content, "openai") and [
                    {"type": "text", "text": block["text"]} if block.get("type") == "text"
                    else {"type": "image", "source": image_source_of_url(block["image_url"].get("url"))}
                    for block in (content if isinstance(content, list) else [])
                    if isinstance(block, dict) and block.get("type") in ("text", "image_url")
                ] or [{"type": "text", "text": ""}]})
            continue
        if role == "assistant":
            blocks = []
            text = blocks_text(content)
            if text != "":
                blocks.append({"type": "text", "text": text})
            for call in message.get("tool_calls") or []:
                if not isinstance(call, dict):
                    continue
                function = call.get("function") if isinstance(call.get("function"), dict) else {}
                blocks.append({
                    "type": "tool_use",
                    "id": call.get("id") or "",
                    "name": function.get("name") or "",
                    "input": arguments_object(function.get("arguments")),
                })
            converted.append({"role": "assistant", "content": blocks or [{"type": "text", "text": ""}]})
            continue
        if role == "tool":
            converted.append({"role": "user", "content": [{
                "type": "tool_result",
                "tool_use_id": message.get("tool_call_id") or "",
                "content": tool_result_text(content),
            }]})
    if system_parts:
        out["system"] = "\n".join(system_parts)
    out["messages"] = converted
    max_tokens = number_or_none(body.get("max_tokens"))
    if max_tokens is None:
        max_tokens = number_or_none(body.get("max_completion_tokens"))
    if max_tokens is not None:
        out["max_tokens"] = max_tokens
    for name in ("temperature", "top_p"):
        value = number_or_none(body.get(name))
        if value is not None:
            out[name] = value
    stops = string_list(body.get("stop"))
    if stops:
        out["stop_sequences"] = stops
    tools = body.get("tools")
    if isinstance(tools, list):
        declared = []
        for tool in tools:
            if not isinstance(tool, dict):
                continue
            function = tool.get("function") if isinstance(tool.get("function"), dict) else None
            if function is None or not isinstance(function.get("name"), str):
                continue
            schema = function.get("parameters") if isinstance(function.get("parameters"), dict) else {"type": "object", "properties": {}}
            declared.append({
                "name": function["name"],
                "description": function.get("description") if isinstance(function.get("description"), str) else "",
                "input_schema": schema,
            })
        if declared:
            out["tools"] = declared
    choice = body.get("tool_choice")
    if choice == "auto":
        out["tool_choice"] = {"type": "auto"}
    elif choice == "required":
        out["tool_choice"] = {"type": "any"}
    elif choice == "none":
        out["tool_choice"] = {"type": "none"}
    elif isinstance(choice, dict) and isinstance(choice.get("function"), dict) and isinstance(choice["function"].get("name"), str):
        out["tool_choice"] = {"type": "tool", "name": choice["function"]["name"]}
    return out


def openai_to_gemini(body, model):
    out = {}
    messages = body.get("messages") if isinstance(body.get("messages"), list) else []
    system_parts = []
    contents = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        content = message.get("content")
        if role == "system":
            text = blocks_text(content)
            if text != "":
                system_parts.append(text)
            continue
        if role == "user":
            contents.append({"role": "user", "parts": gemini_parts(content, "openai")})
            continue
        if role == "assistant":
            parts = []
            text = blocks_text(content)
            if text != "":
                parts.append({"text": text})
            for call in message.get("tool_calls") or []:
                if not isinstance(call, dict):
                    continue
                function = call.get("function") if isinstance(call.get("function"), dict) else {}
                parts.append({"functionCall": {
                    "name": function.get("name") or "",
                    "args": arguments_object(function.get("arguments")),
                }})
            contents.append({"role": "model", "parts": parts or [{"text": ""}]})
            continue
        if role == "tool":
            contents.append({"role": "user", "parts": [{"functionResponse": {
                "name": message.get("name") or "tool",
                "response": {"content": tool_result_text(content)},
            }}]})
    if system_parts:
        out["systemInstruction"] = {"parts": [{"text": "\n".join(system_parts)}]}
    out["contents"] = contents
    tools = body.get("tools")
    if isinstance(tools, list):
        declarations = []
        for tool in tools:
            if not isinstance(tool, dict):
                continue
            function = tool.get("function") if isinstance(tool.get("function"), dict) else None
            if function is None or not isinstance(function.get("name"), str):
                continue
            declaration = {"name": function["name"]}
            if isinstance(function.get("description"), str):
                declaration["description"] = function["description"]
            if isinstance(function.get("parameters"), dict):
                declaration["parameters"] = function["parameters"]
            declarations.append(declaration)
        if declarations:
            out["tools"] = [{"functionDeclarations": declarations}]
    config = {}
    max_tokens = number_or_none(body.get("max_tokens"))
    if max_tokens is None:
        max_tokens = number_or_none(body.get("max_completion_tokens"))
    if max_tokens is not None:
        config["maxOutputTokens"] = max_tokens
    stops = string_list(body.get("stop"))
    if stops:
        config["stopSequences"] = stops
    temperature = number_or_none(body.get("temperature"))
    if temperature is not None:
        config["temperature"] = temperature
    if is_json_mode(body):
        config["responseMimeType"] = "application/json"
    if config:
        out["generationConfig"] = config
    return out


def anthropic_to_openai(body, model):
    out = {"model": model}
    messages = []
    system = blocks_text(body.get("system"))
    if system != "":
        messages.append({"role": "system", "content": system})
    for message in body.get("messages") or []:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        content = message.get("content")
        if isinstance(content, str):
            messages.append({"role": "assistant" if role == "assistant" else "user", "content": content})
            continue
        if not isinstance(content, list):
            continue
        if role == "assistant":
            calls = []
            text = blocks_text(content)
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    calls.append({
                        "id": block.get("id") or "",
                        "type": "function",
                        "function": {
                            "name": block.get("name") or "",
                            "arguments": json.dumps(block.get("input") if isinstance(block.get("input"), dict) else {}),
                        },
                    })
            entry = {"role": "assistant", "content": text if text != "" else None}
            if calls:
                entry["tool_calls"] = calls
            messages.append(entry)
            continue
        parts = []
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "tool_result":
                messages.append({
                    "role": "tool",
                    "tool_call_id": block.get("tool_use_id") or "",
                    "content": tool_result_text(block.get("content")),
                })
                continue
            if block.get("type") == "text" and isinstance(block.get("text"), str):
                parts.append({"type": "text", "text": block["text"]})
            elif block.get("type") == "image":
                url = image_url_of_source(block.get("source"))
                if url is not None:
                    parts.append({"type": "image_url", "image_url": {"url": url}})
        if parts:
            messages.append({"role": "user", "content": parts})
    out["messages"] = messages
    max_tokens = number_or_none(body.get("max_tokens"))
    if max_tokens is not None:
        out["max_tokens"] = max_tokens
    for name in ("temperature", "top_p"):
        value = number_or_none(body.get(name))
        if value is not None:
            out[name] = value
    stops = string_list(body.get("stop_sequences"))
    if stops:
        out["stop"] = stops
    tools = body.get("tools")
    if isinstance(tools, list):
        declared = []
        for tool in tools:
            if not isinstance(tool, dict) or not isinstance(tool.get("name"), str):
                continue
            function = {"name": tool["name"]}
            if isinstance(tool.get("description"), str):
                function["description"] = tool["description"]
            function["parameters"] = tool.get("input_schema") if isinstance(tool.get("input_schema"), dict) else {"type": "object", "properties": {}}
            declared.append({"type": "function", "function": function})
        if declared:
            out["tools"] = declared
    choice = body.get("tool_choice")
    if isinstance(choice, dict):
        if choice.get("type") == "auto":
            out["tool_choice"] = "auto"
        elif choice.get("type") == "any":
            out["tool_choice"] = "required"
        elif choice.get("type") == "none":
            out["tool_choice"] = "none"
        elif choice.get("type") == "tool" and isinstance(choice.get("name"), str):
            out["tool_choice"] = {"type": "function", "function": {"name": choice["name"]}}
    return out


def anthropic_to_gemini(body, model):
    out = {}
    system = blocks_text(body.get("system"))
    if system != "":
        out["systemInstruction"] = {"parts": [{"text": system}]}
    contents = []
    for message in body.get("messages") or []:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        content = message.get("content")
        parts = []
        if isinstance(content, str):
            parts = [{"text": content}]
        else:
            parts = gemini_parts(content, "anthropic")
            for block in (content if isinstance(content, list) else []):
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    parts.append({"functionResponse": {
                        "name": "tool",
                        "response": {"content": tool_result_text(block.get("content"))},
                    }})
        contents.append({"role": "model" if role == "assistant" else "user", "parts": parts or [{"text": ""}]})
    out["contents"] = contents
    tools = body.get("tools")
    if isinstance(tools, list):
        declarations = []
        for tool in tools:
            if not isinstance(tool, dict) or not isinstance(tool.get("name"), str):
                continue
            declaration = {"name": tool["name"]}
            if isinstance(tool.get("description"), str):
                declaration["description"] = tool["description"]
            if isinstance(tool.get("input_schema"), dict):
                declaration["parameters"] = tool["input_schema"]
            declarations.append(declaration)
        if declarations:
            out["tools"] = [{"functionDeclarations": declarations}]
    config = {}
    max_tokens = number_or_none(body.get("max_tokens"))
    if max_tokens is not None:
        config["maxOutputTokens"] = max_tokens
    stops = string_list(body.get("stop_sequences"))
    if stops:
        config["stopSequences"] = stops
    temperature = number_or_none(body.get("temperature"))
    if temperature is not None:
        config["temperature"] = temperature
    if config:
        out["generationConfig"] = config
    return out


def upstream_body(client_api, provider, model, body):
    wire = provider.get("wire")
    if client_api == "openai":
        if wire == "openai":
            out = dict(body)
            out["model"] = model
            out.pop("stream", None)
            out.pop("stream_options", None)
            return out
        if wire == "anthropic":
            return openai_to_anthropic(body, model)
        return openai_to_gemini(body, model)
    if wire == "anthropic":
        out = dict(body)
        out["model"] = model
        out.pop("stream", None)
        return out
    if wire == "openai":
        return anthropic_to_openai(body, model)
    return anthropic_to_gemini(body, model)


# --------------------------------------------------------------------------- provider -> client

def provider_answer(wire, text):
    """(answer text, stop reason, input tokens, output tokens) from a provider's body, or None."""
    try:
        value = json.loads(text)
    except ValueError:
        return None
    if not isinstance(value, dict):
        return None
    if wire == "openai":
        choices = value.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            return None
        choice = choices[0]
        message = choice.get("message") if isinstance(choice.get("message"), dict) else {}
        content = message.get("content")
        usage = value.get("usage") if isinstance(value.get("usage"), dict) else {}
        return {
            "text": content if isinstance(content, str) else blocks_text(content),
            "stop": choice.get("finish_reason") if isinstance(choice.get("finish_reason"), str) else "stop",
            "input": number_or_none(usage.get("prompt_tokens")) or 0,
            "output": number_or_none(usage.get("completion_tokens")) or 0,
        }
    if wire == "anthropic":
        content = value.get("content")
        usage = value.get("usage") if isinstance(value.get("usage"), dict) else {}
        return {
            "text": blocks_text(content),
            "stop": value.get("stop_reason") if isinstance(value.get("stop_reason"), str) else "end_turn",
            "input": number_or_none(usage.get("input_tokens")) or 0,
            "output": number_or_none(usage.get("output_tokens")) or 0,
        }
    candidates = value.get("candidates")
    if not isinstance(candidates, list) or not candidates or not isinstance(candidates[0], dict):
        return None
    content = candidates[0].get("content") if isinstance(candidates[0].get("content"), dict) else {}
    metadata = value.get("usageMetadata") if isinstance(value.get("usageMetadata"), dict) else {}
    parts = content.get("parts") if isinstance(content.get("parts"), list) else []
    return {
        "text": "".join(part["text"] for part in parts
                        if isinstance(part, dict) and isinstance(part.get("text"), str)),
        "stop": candidates[0].get("finishReason") if isinstance(candidates[0].get("finishReason"), str) else "STOP",
        "input": number_or_none(metadata.get("promptTokenCount")) or 0,
        "output": number_or_none(metadata.get("candidatesTokenCount")) or 0,
    }


def pieces_of(text):
    return [text[at:at + DELTA_CHARS] for at in range(0, len(text), DELTA_CHARS)] or [""]


def openai_model_list(models):
    return {
        "object": "list",
        "data": [{"id": model, "object": "model", "created": FIXED_CREATED, "owned_by": provider}
                 for provider, model in models],
    }


def anthropic_model_list(models):
    return {
        "data": [{"type": "model", "id": model, "display_name": model, "created_at": FIXED_ANTHROPIC_CREATED_AT}
                 for provider, model in models],
        "has_more": False,
    }


def openai_error(message, kind):
    return {"error": {"message": message, "type": kind, "code": None}}


def anthropic_error(message, kind):
    return {"type": "error", "error": {"type": kind, "message": message}}


def openai_completion(model, answer):
    usage = {
        "prompt_tokens": answer["input"],
        "completion_tokens": answer["output"],
        "total_tokens": answer["input"] + answer["output"],
    }
    return {
        "id": "chatcmpl-sim",
        "object": "chat.completion",
        "created": FIXED_CREATED,
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": answer["text"]},
            "finish_reason": "stop",
            "logprobs": None,
        }],
        "usage": usage,
    }


def openai_stream(model, answer):
    usage = {
        "prompt_tokens": answer["input"],
        "completion_tokens": answer["output"],
        "total_tokens": answer["input"] + answer["output"],
    }

    def chunk(delta, finish):
        return {"id": "chatcmpl-sim", "object": "chat.completion.chunk", "created": FIXED_CREATED,
                "model": model, "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}

    frames = ["data: %s\n\n" % json.dumps(chunk({"role": "assistant", "content": ""}, None))]
    for piece in pieces_of(answer["text"]):
        frames.append("data: %s\n\n" % json.dumps(chunk({"content": piece}, None)))
    frames.append("data: %s\n\n" % json.dumps(chunk({}, "stop")))
    frames.append("data: %s\n\n" % json.dumps({
        "id": "chatcmpl-sim", "object": "chat.completion.chunk", "created": FIXED_CREATED,
        "model": model, "choices": [], "usage": usage,
    }))
    frames.append("data: [DONE]\n\n")
    return "".join(frames)


def anthropic_message(model, answer):
    return {
        "id": "msg_sim",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [{"type": "text", "text": answer["text"]}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": answer["input"], "output_tokens": answer["output"]},
    }


def anthropic_stream(model, answer):
    def frame(event, data):
        return "event: %s\ndata: %s\n\n" % (event, json.dumps(data))

    frames = [frame("message_start", {
        "type": "message_start",
        "message": {
            "id": "msg_sim", "type": "message", "role": "assistant", "model": model, "content": [],
            "stop_reason": None, "stop_sequence": None,
            "usage": {"input_tokens": answer["input"], "output_tokens": 0},
        },
    }), frame("content_block_start", {
        "type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""},
    })]
    for piece in pieces_of(answer["text"]):
        frames.append(frame("content_block_delta", {
            "type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": piece},
        }))
    frames.append(frame("content_block_stop", {"type": "content_block_stop", "index": 0}))
    frames.append(frame("message_delta", {
        "type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence": None},
        "usage": {"output_tokens": answer["output"]},
    }))
    frames.append(frame("message_stop", {"type": "message_stop"}))
    return "".join(frames)


# --------------------------------------------------------------------------- routing

def requirements_from_body(body):
    messages = body.get("messages") if isinstance(body.get("messages"), list) else []
    vision = False
    for message in messages:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict) and block.get("type") in ("image", "image_url"):
                vision = True
    tools = body.get("tools")
    return {
        "tools": isinstance(tools, list) and len(tools) > 0,
        "vision": vision,
        "json": is_json_mode(body),
    }


def meets_requirements(row, requires):
    if requires.get("tools") and not row.get("tools"):
        return False
    if requires.get("vision") and not row.get("vision"):
        return False
    if requires.get("json") and not row.get("json"):
        return False
    minimum = requires.get("min_context")
    if minimum is not None:
        length = row.get("context_length")
        if not isinstance(length, int) or length < minimum:
            return False
    return True


class Router:
    """The routing state: what the brief holds, which providers are out, and where the run is."""

    def __init__(self, brief, catalog):
        self.brief = brief
        self.catalog = catalog
        self.providers = {p["id"]: p for p in brief_providers(brief)}
        self.order = [p["id"] for p in brief_providers(brief)]
        self.manifest = (brief["workload"] or {}).get("manifest") or []
        self.dead = {}
        self.strikes = {}
        self.calls = {}
        self.index = 0
        self.lock = threading.Lock()
        self.weights = {}
        self.best = {}
        for row in self.manifest:
            if isinstance(row, dict) and isinstance(row.get("id"), str):
                self.weights[row["id"]] = 2 if row.get("weight") == 2 else 1
        for entry in (brief["ratings"] or {}).get("best") or []:
            if isinstance(entry, dict) and isinstance(entry.get("id"), str):
                self.best[entry["id"]] = int(number_or_none(entry.get("rating_milli")) or 0)
        self.bound = sum(self.weights[rid] * self.best[rid] for rid in self.weights if rid in self.best)
        # The quality column rounds to an integer, so a loss under a couple of per cent of the bound
        # is free: spend it on first calls, which a provider always answers, instead of on repeats.
        self.loss_budget = int(0.015 * self.bound)
        self.loss_used = 0
        self.request_at = 0

    def next_request(self, body):
        """The manifest row for the request now arriving, and the requirements it must meet."""
        with self.lock:
            at = self.index
            self.index += 1
        row = self.manifest[at] if 0 <= at < len(self.manifest) and isinstance(self.manifest[at], dict) else None
        requires = row.get("requires") if row is not None else None
        if not isinstance(requires, dict):
            requires = requirements_from_body(body)
        request_id = row.get("id") if row is not None and isinstance(row.get("id"), str) else None
        return request_id, requires

    def plan(self, client_api, body, requires, request_id):
        """The candidates for one request, best first; a provider's first call comes first when the
        quality it costs stays inside the run's loss budget.  A provider always answers its first
        call, so spending an affordable rating on one saves the call a refusal would cost."""
        wants_json = client_api == "openai" and is_json_mode(body)
        ranked = []
        for row in free_rows(self.brief):
            provider_id = row.get("provider")
            if provider_id not in self.providers or provider_id in self.dead:
                continue
            if not meets_requirements(row, requires):
                continue
            wire = self.providers[provider_id].get("wire")
            if wants_json and wire not in ("openai", "gemini"):
                continue
            ranked.append((int(number_or_none(row.get("rating_milli")) or 0),
                           self.order.index(provider_id), str(row.get("model") or ""), provider_id, row))
        ranked.sort(key=lambda item: (-item[0], item[1], item[2]))
        ordered = [(provider_id, row, rating) for rating, _, _, provider_id, row in ranked]
        if len(ordered) < 2:
            return ordered
        weight = self.weights.get(request_id, 1)
        top_rating = ordered[0][2]
        if self.calls.get(ordered[0][0], 0) == 0:
            return ordered
        for at in range(1, len(ordered)):
            provider_id, row, rating = ordered[at]
            if self.calls.get(provider_id, 0) != 0:
                continue
            if weight * (top_rating - rating) <= self.loss_budget - self.loss_used:
                return [ordered[at]] + ordered[:at] + ordered[at + 1:]
            break
        return ordered

    def served(self, provider_id, request_id, rating):
        """Record what one served request cost: the call, and the rating it gave up."""
        self.calls[provider_id] = self.calls.get(provider_id, 0) + 1
        weight = self.weights.get(request_id, 1)
        top = self.best.get(request_id)
        if top is not None and rating is not None:
            self.loss_used += weight * max(0, top - rating)

    def mark(self, provider_id, outcome):
        """Record one provider call's outcome: a permanent refusal, or a strike."""
        if outcome == "dead":
            self.dead[provider_id] = True
            self.strikes[provider_id] = 0
        elif outcome == "transient":
            strikes = self.strikes.get(provider_id, 0) + 1
            self.strikes[provider_id] = strikes
            if strikes >= 3:
                self.dead[provider_id] = True
        else:
            self.strikes[provider_id] = 0

    def call(self, provider_id, model, client_api, body):
        """One provider call. (ok, status, answer, message) with answer None when it refused."""
        provider = self.providers[provider_id]
        url = ENV_URL + (provider.get("path") or "") + wire_route(provider.get("wire"), model)
        headers = {"content-type": "application/json", "accept": "application/json"}
        headers.update(auth_headers(provider.get("auth"), provider.get("key") or ""))
        payload = json.dumps(upstream_body(client_api, provider, model, body), ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(url, data=payload, headers=headers, method="POST")
        self.calls[provider_id] = self.calls.get(provider_id, 0) + 1
        try:
            with urllib.request.urlopen(request, timeout=UPSTREAM_TIMEOUT) as response:
                raw = response.read()
                status = int(response.status)
        except urllib.error.HTTPError as err:
            raw = err.read()
            status = int(err.code)
        except Exception as err:  # the provider could not be reached at all
            return False, 0, None, str(err)
        text = raw.decode("utf-8", "replace")
        if 200 <= status < 300:
            answer = provider_answer(provider.get("wire"), text)
            if answer is None:
                return False, status, None, "the provider answered no answer this wire carries"
            return True, status, answer, ""
        return False, status, None, refusal_message(text)

    def classify(self, status, message):
        if "budget is spent" in message or " is out" in message:
            return "dead"
        if "refused this call" in message:
            return "transient"
        if status == 402:
            return "dead"
        if status == 401 or status == 400:
            return "transient"
        return "transient"


def refusal_message(text):
    try:
        value = json.loads(text)
    except ValueError:
        return text[:200]
    if isinstance(value, dict):
        error = value.get("error")
        if isinstance(error, dict) and isinstance(error.get("message"), str):
            return error["message"]
        if isinstance(error, str):
            return error
        if isinstance(value.get("message"), str):
            return value["message"]
    return text[:200]


# --------------------------------------------------------------------------- the server

class Entry:
    def __init__(self, brief, catalog):
        self.brief = brief
        self.catalog = catalog
        self.router = Router(brief, catalog)
        self.models = served_models(catalog)
        self.failures = {}

    def handle_models(self, headers):
        if headers.get("x-api-key") is not None:
            return 200, anthropic_model_list(self.models), "application/json"
        return 200, openai_model_list(self.models), "application/json"

    def handle_chat(self, api, body):
        """Answer one client request: the best free model that serves it, with failover."""
        request_id, requires = self.router.next_request(body)
        plan = self.router.plan(api, body, requires, request_id)
        if not plan:
            if api == "openai":
                return 503, openai_error("no free model meets this request", "no_free_model"), "application/json"
            return 503, anthropic_error("no free model meets this request", "overloaded_error"), "application/json"
        for at in range(len(plan)):
            provider_id, row, rating = plan[at]
            model = row.get("model")
            # A refusal that is the provider's own schedule, not its budget, costs one call and the
            # next call to the same provider usually answers: retry it once before falling back.
            tries = 2 if at == 0 and len(plan) > 1 else 1
            for attempt in range(tries):
                ok, status, answer, message = self.router.call(provider_id, model, api, body)
                if ok:
                    self.router.mark(provider_id, "ok")
                    self.router.served(provider_id, request_id, rating)
                    if api == "openai":
                        if body.get("stream") is True:
                            return 200, openai_stream(model, answer), "text/event-stream"
                        return 200, openai_completion(model, answer), "application/json"
                    if body.get("stream") is True:
                        return 200, anthropic_stream(model, answer), "text/event-stream"
                    return 200, anthropic_message(model, answer), "application/json"
                outcome = self.router.classify(status, message)
                self.router.mark(provider_id, outcome)
                log("%s/%s refused HTTP %s: %s" % (provider_id, model, status, message))
                if outcome == "dead":
                    break
        if api == "openai":
            return 502, openai_error("every free model that meets this request refused", "upstream_error"), "application/json"
        return 502, anthropic_error("every free model that meets this request refused", "api_error"), "application/json"


def make_handler(entry):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "free-model-aggregator/1.0"

        def log_message(self, fmt, *args):
            return

        def read_body(self):
            try:
                length = int(self.headers.get("content-length") or 0)
            except ValueError:
                return None
            if length < 0 or length > MAX_BODY_BYTES:
                return None
            raw = self.rfile.read(length) if length > 0 else b""
            try:
                value = json.loads(raw.decode("utf-8"))
            except (ValueError, UnicodeDecodeError):
                return None
            return value if isinstance(value, dict) else None

        def respond(self, status, value, kind):
            if kind == "text/event-stream":
                body = value.encode("utf-8")
            else:
                body = (json.dumps(value, ensure_ascii=False) + "\n").encode("utf-8")
            try:
                self.send_response(status)
                self.send_header("content-type", kind if kind == "application/json" else "text/event-stream; charset=utf-8")
                if kind != "application/json":
                    self.send_header("cache-control", "no-cache")
                self.send_header("content-length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                return

        def do_GET(self):
            path = self.path.split("?", 1)[0]
            if path == "/v1/models":
                status, value, kind = entry.handle_models(self.headers)
                return self.respond(status, value, kind)
            return self.respond(404, openai_error("no route GET %s" % path, "not_found"), "application/json")

        def do_POST(self):
            path = self.path.split("?", 1)[0]
            api = "anthropic" if path == "/v1/messages" else "openai"
            body = self.read_body()
            if body is None:
                if api == "anthropic":
                    return self.respond(400, anthropic_error("the request body is not one JSON object", "invalid_request_error"), "application/json")
                return self.respond(400, openai_error("the request body is not one JSON object", "invalid_request_error"), "application/json")
            if path == "/v1/chat/completions" or path == "/v1/messages":
                status, value, kind = entry.handle_chat(api, body)
                return self.respond(status, value, kind)
            return self.respond(404, openai_error("no route POST %s" % path, "not_found"), "application/json")

    return Handler


def env_json(path, method="GET", payload=None, timeout=10.0):
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {"accept": "application/json"}
    if data is not None:
        headers["content-type"] = "application/json"
    request = urllib.request.Request(ENV_URL + path, data=data, headers=headers, method=method)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def register(port, catalog):
    for attempt in range(20):
        try:
            reply = env_json("/register", "POST", {"protocol_version": PROTOCOL_VERSION, "port": port, "catalog": catalog})
            if isinstance(reply, dict) and reply.get("ok") is True:
                log("registered port %d, %s requests coming" % (port, reply.get("requests")))
                return True
            log("register refused: %s" % json.dumps(reply)[:200])
            return False
        except Exception as err:
            log("register attempt %d failed: %s" % (attempt + 1, err))
            threading.Event().wait(POLL_SECONDS)
    return False


def watch_work(server):
    """Poll GET /work until the env says done, then exit 0."""
    while True:
        try:
            work = env_json("/work", "GET", None, 5.0)
        except Exception:
            work = None
        if isinstance(work, dict) and work.get("done") is True:
            log("the workload is done: exiting 0")
            server.shutdown()
            os._exit(0)
        threading.Event().wait(POLL_SECONDS)


def serve(brief, catalog):
    server = None
    handler = make_handler(Entry(brief, catalog))
    for attempt in range(10):
        server = ThreadingHTTPServer(("0.0.0.0", 0), handler)
        server.daemon_threads = True
        port = server.server_address[1]
        if port != 8080:
            break
        server.server_close()
    if server is None:
        raise SystemExit("no port could be bound")
    log("listening on 0.0.0.0:%d with %d free models" % (port, len(catalog.get("providers", []))))
    if not register(port, catalog):
        raise SystemExit("the env did not accept the register body")
    thread = threading.Thread(target=watch_work, args=(server,), daemon=True)
    thread.start()
    server.serve_forever()


def selftest():
    checks = []

    def run_check(name, func):
        try:
            err = func()
            if err:
                sys.stdout.write("selftest %s: FAIL: %s\n" % (name, err))
                checks.append(False)
            else:
                sys.stdout.write("selftest %s: ok\n" % name)
                checks.append(True)
        except Exception as ex:
            sys.stdout.write("selftest %s: FAIL: %s\n" % (name, ex))
            checks.append(False)

    def check_brief():
        alpha_p = {
            "id": "alpha",
            "name": "Alpha",
            "wire": "openai",
            "auth": "bearer",
            "key_env": "ALPHA_KEY",
            "key": "k",
            "path": "/p/alpha",
        }
        beta_p = {
            "id": "beta",
            "name": "Beta",
            "wire": "gemini",
            "auth": "x-goog-api-key",
            "key_env": "BETA_KEY",
            "key": "k",
            "path": "/p/beta",
        }
        doc_directory = {
            "schema": "free-model-aggregator-01/providers/v1",
            "providers": [alpha_p, beta_p],
        }
        doc_ratings = {
            "schema": "free-model-aggregator-01/ratings/v1",
            "ratings": [
                {
                    "model": "a-1",
                    "provider": "alpha",
                    "rating_milli": 1200,
                    "free": True,
                    "tools": True,
                    "json": True,
                    "context_length": 8192,
                },
                {
                    "model": "b-1",
                    "provider": "beta",
                    "rating_milli": 1500,
                    "free": True,
                    "tools": True,
                    "vision": True,
                    "json": True,
                    "context_length": 8192,
                },
            ],
            "best": [{"id": "s1", "rating_milli": 1500}],
        }
        doc_workload = {
            "schema": "free-model-aggregator-01/workload/v1",
            "manifest": [
                {
                    "id": "s1",
                    "api": "openai",
                    "stream": False,
                    "weight": 1,
                    "requires": {"tools": True, "vision": True},
                },
            ],
            "bodies": {},
        }
        with tempfile.NamedTemporaryFile("w+", encoding="utf-8", delete=False) as tf:
            tf_path = tf.name
            for doc in (doc_directory, doc_ratings, doc_workload):
                tf.write(json.dumps(doc) + "\n")
        try:
            brief = read_brief(tf_path)
        finally:
            try:
                os.unlink(tf_path)
            except OSError:
                pass

        if brief.get("error"):
            return "brief parsed with error: %s" % brief["error"]
        if not brief.get("directory") or not brief.get("ratings") or not brief.get("workload"):
            return "one or more documents failed to parse"

        catalog = build_catalog(brief)
        providers = catalog.get("providers") or []
        if len(providers) != 2:
            return "build_catalog should list 2 providers, got %d" % len(providers)
        p_ids = [p.get("id") for p in providers]
        if p_ids != ["alpha", "beta"]:
            return "build_catalog provider ids mismatch: %s" % p_ids
        for p in providers:
            if not p.get("models"):
                return "provider %s has no free models in catalog" % p.get("id")

        models = served_models(catalog)
        model_ids = [m[1] for m in models]
        if model_ids != ["a-1", "b-1"]:
            return "served_models model ids mismatch: %s" % model_ids
        return None

    def check_upstream():
        alpha_p = {
            "id": "alpha",
            "name": "Alpha",
            "wire": "openai",
            "auth": "bearer",
            "key_env": "ALPHA_KEY",
            "key": "k",
            "path": "/p/alpha",
        }
        beta_p = {
            "id": "beta",
            "name": "Beta",
            "wire": "gemini",
            "auth": "x-goog-api-key",
            "key_env": "BETA_KEY",
            "key": "k",
            "path": "/p/beta",
        }

        # 1. beta provider (openai -> gemini)
        beta_body = upstream_body(
            "openai",
            beta_p,
            "b-1",
            {
                "model": "auto",
                "messages": [
                    {"role": "system", "content": "sys"},
                    {"role": "user", "content": "hi"},
                ],
                "max_tokens": 7,
                "response_format": {"type": "json_object"},
            },
        )
        if "systemInstruction" not in beta_body:
            return "beta_body missing systemInstruction"
        if "contents" not in beta_body:
            return "beta_body missing contents"
        gen_cfg = beta_body.get("generationConfig")
        if not isinstance(gen_cfg, dict) or gen_cfg.get("responseMimeType") != "application/json":
            return "beta_body generationConfig missing responseMimeType application/json"

        # 2. alpha provider (anthropic -> openai)
        alpha_anth = upstream_body(
            "anthropic",
            alpha_p,
            "a-1",
            {
                "model": "auto",
                "max_tokens": 9,
                "system": "sys",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        if alpha_anth.get("max_tokens") != 9:
            return "alpha_anth missing max_tokens 9"
        msgs = alpha_anth.get("messages") or []
        has_sys = any(m.get("role") == "system" and "sys" in (m.get("content") or "") for m in msgs)
        if not has_sys:
            return "alpha_anth missing system message holding 'sys'"
        user_msgs = [m for m in msgs if m.get("role") == "user"]
        if len(user_msgs) != 1:
            return "alpha_anth expected 1 user message, found %d" % len(user_msgs)

        # 3. alpha provider (openai -> openai)
        alpha_open = upstream_body(
            "openai",
            alpha_p,
            "a-1",
            {
                "model": "auto",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 5,
            },
        )
        if "messages" not in alpha_open:
            return "alpha_open missing messages"
        if alpha_open.get("max_tokens") != 5:
            return "alpha_open missing max_tokens 5"
        return None

    def check_streams():
        s_open = openai_stream("a-1", {"text": "ANSWER ECHO{}", "input": 3, "output": 4})
        if not s_open.rstrip().endswith("data: [DONE]"):
            return "openai_stream does not end with 'data: [DONE]'"
        if '"finish_reason": "stop"' not in s_open:
            return "openai_stream does not carry chunk with finish_reason 'stop'"

        s_anth = anthropic_stream("a-1", {"text": "ANSWER ECHO{}", "input": 3, "output": 4})
        for event in ("event: message_start", "event: message_delta", "event: message_stop"):
            if event not in s_anth:
                return "anthropic_stream missing %s" % event
        return None

    run_check("brief", check_brief)
    run_check("upstream", check_upstream)
    run_check("streams", check_streams)

    if all(checks):
        sys.stdout.write("selftest: ok\n")
        sys.stdout.flush()
        return 0
    else:
        sys.stdout.write("selftest: FAILED\n")
        sys.stdout.flush()
        return 1


def main(argv):
    if "--selftest" in argv:
        return selftest()
    brief = read_brief(PROMPT_FILE)
    if brief["error"] is not None:
        log(brief["error"])
    catalog = build_catalog(brief)
    if "--list" in argv:
        sys.stdout.write(json.dumps(catalog) + "\n")
        sys.stdout.flush()
        return 0
    if ENV_URL == "":
        log("MAKEFUN_ENV_URL is not set: nothing to serve")
        return 1
    try:
        serve(brief, catalog)
    except SystemExit:
        raise
    except Exception as err:
        log("the server stopped: %s" % err)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
