"""Bundled Verifiers v1 harness enforcing the benchmark context contract."""

from __future__ import annotations

import json
from pathlib import Path

import verifiers.v1 as vf

PROGRAM_SOURCE = (Path(__file__).resolve().parent / "program.py").read_text()


class BeerHarnessConfig(vf.HarnessConfig):
    pass


class BeerHarness(vf.Harness[BeerHarnessConfig]):
    APPENDS_SYSTEM_PROMPT = True
    SUPPORTS_MCP = True
    SUPPORTS_USER_SIM = False
    SUPPORTS_MESSAGE_PROMPT = False

    async def setup(self, runtime: vf.Runtime) -> None:
        await runtime.prepare_uv_script(PROGRAM_SOURCE, self.config.resolved_env)

    async def launch(
        self,
        ctx: vf.ModelContext,
        trace: vf.Trace,
        runtime: vf.Runtime,
        endpoint: str,
        secret: str,
        mcp_urls: dict[str, str],
    ) -> vf.ProgramResult:
        system_prompt, prompt = self.resolve_prompt(trace.task.data)
        if not system_prompt or not isinstance(prompt, str):
            raise ValueError("BeerHarness requires string system and user prompts")
        if not mcp_urls:
            raise ValueError("BeerHarness requires its task-scoped MCP toolset")
        horizon = int(trace.task.data.scenario["horizon"])
        args = [
            f"--base-url={endpoint}",
            f"--api-key={secret}",
            f"--model={ctx.model}",
            f"--system-prompt={system_prompt}",
            f"--prompt={prompt}",
            "--mcp-config="
            + json.dumps(
                {
                    "mcpServers": {
                        name: {"url": url} for name, url in mcp_urls.items()
                    }
                }
            ),
            f"--max-turns={horizon + 2}",
        ]
        program = await runtime.prepare_uv_script(
            PROGRAM_SOURCE, self.config.resolved_env
        )
        return await runtime.run_program(program + args, self.config.resolved_env)
