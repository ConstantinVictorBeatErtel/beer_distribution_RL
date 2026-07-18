#!/usr/bin/env python3
"""Make one minimal AkashML request to verify strict function calling."""

from __future__ import annotations

import argparse
import json
import os
import ssl
import sys
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import certifi


BASE_URL = "https://api.akashml.com/v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="deepseek-ai/DeepSeek-V4-Flash",
    )
    return parser.parse_args()


def request_json(path: str, key: str, payload: dict | None = None) -> dict:
    body = None if payload is None else json.dumps(payload).encode()
    request = Request(
        BASE_URL + path,
        data=body,
        method="GET" if payload is None else "POST",
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        },
    )
    ssl_context = ssl.create_default_context(cafile=certifi.where())
    with urlopen(request, timeout=120, context=ssl_context) as response:
        return json.load(response)


def main() -> int:
    args = parse_args()
    key = os.environ.get("AKASH_API_KEY")
    if not key:
        print("AKASH_API_KEY is not set", file=sys.stderr)
        return 2

    try:
        catalog = request_json("/models", key)
        models = {row["id"]: row for row in catalog.get("data", [])}
        if args.model not in models:
            print(f"model unavailable: {args.model}")
            print("available model IDs:")
            for model_id in sorted(models):
                print(f"  {model_id}")
            return 3

        response = request_json(
            "/chat/completions",
            key,
            {
                "model": args.model,
                "messages": [
                    {
                        "role": "system",
                        "content": "Call place_order exactly once. Do not answer in text.",
                    },
                    {
                        "role": "user",
                        "content": "Inventory is 12 and demand is 8. Place an order of 8.",
                    },
                ],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "place_order",
                            "description": "Place the weekly replenishment order.",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "quantity": {
                                        "type": "integer",
                                        "minimum": 0,
                                        "maximum": 128,
                                    }
                                },
                                "required": ["quantity"],
                                "additionalProperties": False,
                            },
                        },
                    }
                ],
                "tool_choice": "required",
                "temperature": 0,
                "max_tokens": 64,
            },
        )
    except HTTPError as error:
        print(f"Akash HTTP error: {error.code}", file=sys.stderr)
        return 4
    except (URLError, TimeoutError, json.JSONDecodeError) as error:
        print(f"Akash request failed: {type(error).__name__}", file=sys.stderr)
        return 5

    message = response["choices"][0]["message"]
    calls = message.get("tool_calls") or []
    valid = False
    quantity = None
    if len(calls) == 1 and calls[0].get("function", {}).get("name") == "place_order":
        try:
            arguments = json.loads(calls[0]["function"].get("arguments") or "{}")
        except json.JSONDecodeError:
            arguments = {}
        quantity = arguments.get("quantity")
        valid = (
            set(arguments) == {"quantity"}
            and type(quantity) is int
            and 0 <= quantity <= 128
        )

    model = models[args.model]
    print(f"model={args.model}")
    print(f"supported_features={model.get('supported_features', [])}")
    print(f"pricing={model.get('pricing', {})}")
    print(f"tool_call_valid={str(valid).lower()}")
    print(f"quantity={quantity if valid else 'invalid'}")
    usage = response.get("usage") or {}
    print(f"usage={json.dumps(usage, sort_keys=True)}")
    return 0 if valid else 6


if __name__ == "__main__":
    raise SystemExit(main())
