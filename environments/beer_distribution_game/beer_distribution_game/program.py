# /// script
# requires-python = ">=3.11"
# dependencies = ["openai", "mcp"]
# ///
"""Minimal rolling-context chat program for the Beer Game harness."""

from __future__ import annotations

import argparse
import asyncio
from contextlib import AsyncExitStack
import json

from openai import AsyncOpenAI


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--system-prompt", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--mcp-config", required=True)
    parser.add_argument("--max-turns", required=True, type=int)
    return parser.parse_args()


async def connect_tools(stack: AsyncExitStack, config: dict):
    from mcp import ClientSession
    from mcp.client.streamable_http import (
        create_mcp_http_client,
        streamable_http_client,
    )

    dispatch = {}
    schemas = {}
    for spec in config["mcpServers"].values():
        http_client = await stack.enter_async_context(
            create_mcp_http_client(headers=spec.get("headers") or None)
        )
        read, write, *_ = await stack.enter_async_context(
            streamable_http_client(spec["url"], http_client=http_client)
        )
        session = await stack.enter_async_context(ClientSession(read, write))
        await session.initialize()
        for tool in (await session.list_tools()).tools:
            if tool.name in dispatch:
                raise RuntimeError(f"duplicate raw MCP tool name {tool.name!r}")
            dispatch[tool.name] = session
            schemas[tool.name] = tool
    required = {"place_order", "record_protocol_error"}
    if not required <= set(dispatch):
        raise RuntimeError(f"Beer tool server missing {sorted(required - set(dispatch))}")
    return dispatch, schemas


def result_text(result) -> str:
    texts = [block.text for block in result.content if block.type == "text"]
    if not texts:
        raise RuntimeError("Beer tool returned no text result")
    return "\n".join(texts)


async def call_tool(dispatch: dict, name: str, arguments: dict) -> str:
    result = await dispatch[name].call_tool(name, arguments)
    if getattr(result, "isError", False):
        raise RuntimeError(result_text(result))
    return result_text(result)


def parse_action(message) -> tuple[int | None, str | None]:
    calls = message.tool_calls or []
    if len(calls) != 1:
        return None, "missing_tool_call" if not calls else "multiple_tool_calls"
    call = calls[0]
    if call.function.name != "place_order":
        return None, "unknown_tool"
    try:
        arguments = json.loads(call.function.arguments or "{}")
    except json.JSONDecodeError:
        return None, "invalid_json"
    if not isinstance(arguments, dict):
        return None, "arguments_not_object"
    if set(arguments) != {"quantity"}:
        return None, "invalid_arguments"
    quantity = arguments["quantity"]
    if type(quantity) is not int or not 0 <= quantity <= 128:
        return None, "invalid_quantity"
    return quantity, None


async def main() -> None:
    args = parse_args()
    client = AsyncOpenAI(
        base_url=args.base_url,
        api_key=args.api_key,
        timeout=120.0,
        max_retries=2,
    )
    config = json.loads(args.mcp_config)
    async with AsyncExitStack() as stack:
        dispatch, schemas = await connect_tools(stack, config)
        place = schemas["place_order"]
        model_tools = [
            {
                "type": "function",
                "function": {
                    "name": "place_order",
                    "description": place.description or "Place this week's order.",
                    "parameters": place.inputSchema,
                },
            }
        ]
        current_prompt = args.prompt
        done = False
        for _ in range(args.max_turns):
            # Reconstruct context each week. The canonical observation already contains
            # the approved eight-record history; older chat/tool messages stay out.
            messages = [
                {"role": "system", "content": args.system_prompt},
                {"role": "user", "content": current_prompt},
            ]
            completion = await client.chat.completions.create(
                model=args.model,
                messages=messages,
                tools=model_tools,
                tool_choice="required",
                parallel_tool_calls=False,
            )
            message = completion.choices[0].message
            quantity, error = parse_action(message)
            if error is not None:
                current_prompt = await call_tool(
                    dispatch, "record_protocol_error", {"category": error}
                )
            else:
                current_prompt = await call_tool(
                    dispatch, "place_order", {"quantity": quantity}
                )
            payload = json.loads(current_prompt)
            if payload.get("done"):
                done = True
                break
        if not done:
            payload = json.loads(
                await call_tool(
                    dispatch, "record_protocol_error", {"category": "max_turns"}
                )
            )
            if not payload.get("done"):
                raise RuntimeError("harness turn cap reached before episode termination")


if __name__ == "__main__":
    asyncio.run(main())
