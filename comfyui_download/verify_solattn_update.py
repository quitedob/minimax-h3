"""Validate the updated SolAttn node with the production Turbo 8-step graph."""

import asyncio
import json
from pathlib import Path

import aiohttp
import psutil

import benchmark_h3_turbo as benchmark
from inspect_h3_turbo import inspect


RESULT = Path(__file__).with_name("solattn_26d816e_8step_validation.json")


async def main():
    if RESULT.exists():
        raise FileExistsError(f"Preserve existing validation: {RESULT}")
    template = json.loads(
        (benchmark.COMFY / "cloud_h3_sage3_solattn_easycache_prompt.json")
        .read_text(encoding="utf-8")
    )
    prompt = template["prompt"]
    assert prompt["sigmas"]["inputs"]["steps"] == 8
    assert prompt["solattn"]["inputs"]["int8_pv"] is False
    assert prompt["solattn"]["inputs"]["dense_blocks"] == ""
    prompt["solattn"]["inputs"]["verbose"] = True
    process = next(
        psutil.Process(connection.pid)
        for connection in psutil.net_connections(kind="tcp")
        if connection.status == psutil.CONN_LISTEN and connection.laddr.port == 8188
    )
    async with aiohttp.ClientSession() as session:
        report = await benchmark.run("solattn_26d816e", 42, template, process, session)
    report["upstream_revision"] = "26d816ebd4f1e43a2c6e4d4759be3137f10a7a73"
    report["measurement_note"] = "One post-restart validation, not a speedup benchmark."
    RESULT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if report["status"]["status_str"] != "success":
        raise RuntimeError(f"Generation failed: see {RESULT}")
    output = report["history"]["outputs"]["save"]["images"][0]
    path = benchmark.COMFY / "output" / output.get("subfolder", "") / output["filename"]
    report["media"] = inspect(path)
    RESULT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    media = report["media"]
    assert (media["width"], media["height"], media["decoded_frames"]) == (864, 480, 124)
    assert media["audio"]["finite"] and media["audio"]["rms"] > 0
    print(json.dumps({"result": str(RESULT), "media": media}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
