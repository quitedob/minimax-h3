"""Measure H3 Turbo cold/warm runs through the local ComfyUI API."""

import asyncio
import copy
import json
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

import psutil
import aiohttp

ROOT = Path(__file__).resolve().parent
COMFY = ROOT.parent / "ComfyUI_sage3_py312"
BASE = "http://127.0.0.1:8188"
RESULT = ROOT / "h3_turbo_v4_8step_benchmark.json"


def api(path, data=None):
    body = json.dumps(data).encode() if data is not None else None
    request = urllib.request.Request(BASE + path, data=body,
                                     headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        raise RuntimeError(error.read().decode()) from error


def resources(process):
    memory = psutil.virtual_memory()
    sample = {"time": time.time(), "process_rss_mib": process.memory_info().rss / 2**20,
              "system_used_mib": (memory.total - memory.available) / 2**20}
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used,utilization.gpu,temperature.gpu",
         "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5,
        creationflags=subprocess.CREATE_NO_WINDOW)
    if result.returncode == 0:
        used, util, temp = map(float, result.stdout.strip().splitlines()[0].split(","))
        sample.update(gpu_used_mib=used, gpu_util_percent=util, gpu_temp_c=temp)
    else:
        sample["gpu_error"] = result.stderr.strip()
    return sample


async def run(label, seed, template, process, session):
    payload = copy.deepcopy(template)
    payload["client_id"] = "h3-turbo-benchmark-" + label
    payload["prompt"]["noise"]["inputs"]["noise_seed"] = seed
    payload["prompt"]["save"]["inputs"]["filename_prefix"] = (
        "video/web_bench_h3_turbo_v4_8step_" + label)
    queue = api("/queue")
    if queue["queue_running"] or queue["queue_pending"]:
        raise RuntimeError("ComfyUI queue is busy; benchmark was not submitted")
    ws = await session.ws_connect("ws://127.0.0.1:8188/ws?clientId=" + payload["client_id"])
    start = time.time()
    pid = api("/prompt", payload)["prompt_id"]
    print(f"{label}: prompt_id={pid} seed={seed}", flush=True)
    report = {"label": label, "seed": seed, "prompt_id": pid, "submitted_at": start,
              "payload": payload, "events": [], "resources": []}
    next_resource = next_history = 0
    try:
        while time.time() - start < 1800:
            try:
                message = await asyncio.wait_for(ws.receive(), timeout=1)
            except asyncio.TimeoutError:
                message = None
            if message is not None and message.type == aiohttp.WSMsgType.TEXT:
                event = json.loads(message.data)
                data = event.get("data", {})
                if data.get("prompt_id") == pid:
                    report["events"].append({"elapsed_s": time.time() - start, **event})
                    if event["type"] == "executing":
                        print(f"{label} {time.time()-start:.1f}s node={data.get('node')}", flush=True)
                    elif event["type"] == "progress":
                        print(f"{label} {time.time()-start:.1f}s progress={data.get('value')}/{data.get('max')}", flush=True)
                    elif event["type"] in ("execution_success", "execution_error", "execution_interrupted"):
                        next_history = 0
            now = time.time()
            if now >= next_resource:
                report["resources"].append(resources(process))
                next_resource = now + 2
            if now >= next_history:
                history = api("/history/" + pid)
                next_history = now + 2
                if pid in history:
                    report["history"] = history[pid]
                    report["api_wall_s"] = time.time() - start
                    messages = history[pid]["status"]["messages"]
                    stamps = {kind: data.get("timestamp") for kind, data in messages}
                    if stamps.get("execution_success") and stamps.get("execution_start"):
                        report["server_wall_s"] = (stamps["execution_success"] - stamps["execution_start"]) / 1000
                    report["status"] = history[pid]["status"]
                    print(f"{label} DONE server={report.get('server_wall_s')}s status={report['status']['status_str']}", flush=True)
                    return report
        report["status"] = {"status_str": "timeout"}
        return report
    finally:
        await ws.close()


async def main():
    template = json.loads((COMFY / "cloud_h3_sage3_solattn_easycache_prompt.json").read_text(encoding="utf-8"))
    process = next(psutil.Process(c.pid) for c in psutil.net_connections(kind="tcp")
                   if c.status == psutil.CONN_LISTEN and c.laddr.port == 8188)
    report = {"created_at": time.time(), "server_pid": process.pid, "runs": []}
    async with aiohttp.ClientSession() as session:
        for label, seed in (("cold", 42), ("warm", 43)):
            result = await run(label, seed, template, process, session)
            report["runs"].append(result)
            RESULT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            if result["status"]["status_str"] != "success":
                raise RuntimeError(f"{label} failed; see {RESULT}")
            saved = result["history"]["outputs"]["save"]["images"][0]
            cond = result["payload"]["prompt"]["cond"]["inputs"]
            api("/h3/meta", {
                "filename": saved["filename"], "prompt": cond["prompt"],
                "steps": result["payload"]["prompt"]["sigmas"]["inputs"]["steps"],
                "sec": round(cond["length"] / 24),
                "res": f"{cond['width']}×{cond['height']}", "dur": result["server_wall_s"]})
    print(RESULT, flush=True)


if __name__ == "__main__":
    asyncio.run(main())
