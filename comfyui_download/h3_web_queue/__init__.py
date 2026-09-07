"""Serve a minimal single-input H3 video queue page over the existing ComfyUI server.

Registers GET /h3 on the running PromptServer so the same port/domain used for
ComfyUI also serves a small HTML frontend. The frontend submits to ComfyUI's own
POST /prompt, polls GET /history/{prompt_id} and serves the result via GET /view.
"""

import json
import os
from pathlib import Path

from aiohttp import web
from server import PromptServer
import folder_paths  # top-level module (see server.py:9), not comfy.folder_paths

NODE_ROOT = Path(__file__).resolve().parent
INDEX_HTML = NODE_ROOT / "web" / "index.html"

# The production workflow template lives next to this ComfyUI instance (sage3_py312).
# It is the source of truth for the node graph; the frontend copies it and swaps in
# the user's prompt. Fall back to the repo copy for convenience.
COMFY_ROOT = NODE_ROOT.parents[1]
TEMPLATE_CANDIDATES = [
    COMFY_ROOT / "cloud_h3_sage3_solattn_easycache_prompt.json",
    NODE_ROOT.parents[3] / "comfyui_download" / "cloud_h3_sage3_solattn_easycache_prompt.json",
]

# Placeholder replaced in index.html at serve time with the workflow JSON.
WORKFLOW_TOKEN = "__H3_WORKFLOW_JSON__"


def _load_workflow() -> dict:
    for path in TEMPLATE_CANDIDATES:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    raise FileNotFoundError("No H3 cloud workflow template found; looked in: "
                            + ", ".join(str(p) for p in TEMPLATE_CANDIDATES))


routes = PromptServer.instance.routes


def _server_page():
    if not INDEX_HTML.exists():
        return None
    workflow_text = json.dumps(_load_workflow(), ensure_ascii=False)
    return INDEX_HTML.read_text(encoding="utf-8").replace(WORKFLOW_TOKEN, workflow_text)


@routes.get("/video")
async def video_index(request):
    html = _server_page()
    if html is None:
        return web.Response(text="h3_web_queue/web/index.html not found", status=404,
                            content_type="text/plain")
    return web.Response(text=html, content_type="text/html")


@routes.get("/h3")
async def h3_index(request):
    html = _server_page()
    if html is None:
        return web.Response(text="h3_web_queue/web/index.html not found", status=404,
                            content_type="text/plain")
    return web.Response(text=html, content_type="text/html")


def _meta_path():
    return os.path.join(folder_paths.get_output_directory(), "webmeta.json")


def _load_meta():
    try:
        with open(_meta_path(), encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_meta(meta):
    with open(_meta_path(), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False)


@routes.post("/h3/meta")
async def h3_meta(request):
    """Stash per-job display metadata (prompt/params/用时) keyed by output filename.

    The page posts this when a job it submitted finishes, so /h3/files can show the
    prompt, parameters and generation duration next to each video. Survives restarts.
    """
    data = await request.json()
    fn = (data.get("filename") or "").strip()
    if not fn:
        return web.Response(status=400)
    meta = _load_meta()
    meta[fn] = {k: data.get(k) for k in ("prompt", "steps", "sec", "res", "dur")}
    _save_meta(meta)
    return web.json_response({"ok": True})


_AV_META_CACHE = {}  # path -> (mtime_ms, meta|None)


def _read_av_meta(path):
    """Recover prompt/params from the MP4's embedded ComfyUI metadata (pyav).

    SaveVideo writes the submitted graph as a `prompt` metadata string in the
    mp4's udta/meta box, so even videos generated before this page existed (and
    across restarts) keep their prompt & params on disk — no reliance on the
    in-memory /history.
    """
    try:
        import av
        with av.open(path, metadata_errors="ignore") as c:
            raw = c.metadata.get("prompt")
            if not raw:
                return None
            g = json.loads(raw)
        h3ds = (g.get("h3ds") or {}).get("inputs") or {}
        cond = (g.get("cond") or {}).get("inputs") or {}
        sig = (g.get("sigmas") or {}).get("inputs") or {}
        prompt = h3ds.get("user_text") or (cond.get("prompt") if isinstance(cond.get("prompt"), str) else None)
        length = cond.get("length")
        w, h_ = cond.get("width"), cond.get("height")
        return {
            "prompt": prompt,
            "steps": sig.get("steps"),
            "length": length,
            "sec": round(length / 24) if length else None,
            "res": ("%s×%s" % (w, h_)) if w and h_ else None,
        }
    except Exception:
        return None


def _prompt_meta(path, mtime_ms):
    cached = _AV_META_CACHE.get(path)
    if cached and cached[0] == mtime_ms:
        return cached[1]
    pm = _read_av_meta(path)
    _AV_META_CACHE[path] = (mtime_ms, pm)
    return pm


@routes.get("/h3/files")
async def h3_files(request):
    """List the web-queue video files on disk (persist across ComfyUI restarts).

    ComfyUI's /history is in-memory and is wiped on restart, so relying on it to
    surface finished videos is fragile. Listing the output directory instead means
    the page always shows every produced video. We only list the page's own outputs
    (filename_prefix `video/web_*`) so benchmark/other videos don't mix in. Prompt &
    params come from each mp4's embedded metadata; 用时 (generation duration) comes
    from the /h3/meta store (only known for jobs the page observed).
    """
    output_dir = folder_paths.get_output_directory()
    stored_meta = _load_meta()
    results = []
    if output_dir:
        for root, _dirs, names in os.walk(os.path.join(output_dir, "video")):
            for name in names:
                if not name.lower().endswith((".mp4", ".mov", ".webm")):
                    continue
                if not name.lower().startswith("web_"):  # 只列本页生成的视频
                    continue
                path = os.path.join(root, name)
                rel = os.path.relpath(path, output_dir)
                mtime_ms = int(os.path.getmtime(path) * 1000)
                pm = _prompt_meta(path, mtime_ms) or {}
                sm = stored_meta.get(name) or {}
                results.append({
                    "filename": name,
                    "subfolder": os.path.dirname(rel).replace("\\", "/"),
                    "type": "output",
                    "size": os.path.getsize(path),
                    "mtime_ms": mtime_ms,
                    "meta": {**pm, "dur": sm.get("dur")},
                })
    results.sort(key=lambda x: x["mtime_ms"], reverse=True)
    return web.json_response(results)


@routes.get("/h3/download")
async def h3_download(request):
    """Serve a saved output with Content-Disposition: attachment.

    ComfyUI's own /view returns a bare `filename="..."` header (treated as inline),
    which desktop Chrome would download but iOS/mobile Safari plays instead. Forcing
    `attachment` makes every browser's download manager save the file.
    """
    q = request.rel_url.query
    filename = os.path.basename((q.get("filename") or "").strip())
    subfolder = (q.get("subfolder") or "").strip()
    type_ = (q.get("type") or "output").strip() or "output"
    if not filename:
        return web.Response(status=400)
    output_dir = folder_paths.get_directory_by_type(type_)
    if not output_dir:
        return web.Response(status=400)
    full = os.path.abspath(os.path.join(output_dir, subfolder, filename))
    if not full.startswith(os.path.abspath(output_dir)):
        return web.Response(status=403)
    if not os.path.isfile(full):
        return web.Response(status=404)
    return web.FileResponse(
        full,
        headers={"Content-Disposition": 'attachment; filename="%s"' % filename},
    )


NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
