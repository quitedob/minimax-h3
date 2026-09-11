"""Serve the H3 text, reference-image and keyframe video queue page.

Registers GET /h3 on the running PromptServer so the same port/domain used for
ComfyUI also serves a small HTML frontend. The frontend submits to ComfyUI's own
POST /prompt, polls GET /history/{prompt_id} and serves the result via GET /view.
"""

import asyncio
import base64
import json
import logging
import os
import time
import urllib.error
import urllib.parse
import urllib.request
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


# --- DeepSeek prompt generation: .env + local skill + vision (self-contained).
# The h3_deepseek_prompt node loads under a private module name in ComfyUI, so we
# can't import it here; the helpers below mirror it so the page can generate the
# final prompt up-front and let the user review/edit it before submitting.
PROJECT_ROOT = NODE_ROOT.parents[2]
ENV_PATH = PROJECT_ROOT / ".env"
SKILL_PATH = PROJECT_ROOT / ".claude" / "skills" / "h3-prompt-writing"
DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-v4-flash"
DEFAULT_VISION_MODEL = "deepseek-v4-flash-vision-exp"


def _load_env(path):
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _skill_prompt():
    files = [SKILL_PATH / "SKILL.md",
             SKILL_PATH / "references" / "base-en.txt",
             SKILL_PATH / "references" / "ref-en.txt"]
    missing = [str(p) for p in files if not p.exists()]
    if missing:
        raise RuntimeError("H3 prompt skill files are missing: " + ", ".join(missing))
    return "\n\n".join("===== {} =====\n{}".format(p.as_posix(), p.read_text(encoding="utf-8"))
                       for p in files)


def _post_deepseek(payload, timeout=180):
    """POST a chat/completions payload to the DeepSeek endpoint and return content."""
    _load_env(ENV_PATH)
    api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    base_url = os.environ.get("DEEPSEEK_BASE_URL", DEFAULT_BASE_URL).strip() or DEFAULT_BASE_URL
    if not api_key or api_key == "your_deepseek_api_key_here":
        raise RuntimeError("Set DEEPSEEK_API_KEY in " + str(ENV_PATH))
    request = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": "Bearer " + api_key, "Content-Type": "application/json"},
        method="POST",
    )
    # The official domestic endpoint should not detour through the system proxy.
    # Custom API endpoints retain their existing proxy configuration.
    proxy_handler = (urllib.request.ProxyHandler({})
                     if urllib.parse.urlsplit(base_url).hostname == "api.deepseek.com"
                     else urllib.request.ProxyHandler())
    opener = urllib.request.build_opener(proxy_handler)
    started = time.perf_counter()
    try:
        with opener.open(request, timeout=timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"DeepSeek API HTTP {error.code}: {detail[:500]}") from error
    except urllib.error.URLError as error:
        raise RuntimeError(f"DeepSeek API connection failed: {error.reason}") from error
    choices = result.get("choices") or []
    if not choices or not choices[0].get("message", {}).get("content"):
        raise RuntimeError("DeepSeek API returned no prompt content")
    logging.info("H3 DeepSeek %s completed in %.2fs", payload["model"], time.perf_counter() - started)
    return choices[0]["message"]["content"].strip()


def _image_mime(raw):
    """Detect the actual image format from the bytes (DeepSeek decides by content,
    not by the declared MIME or filename). Returns a MIME string, or None if the
    file isn't one of the supported formats (JPEG/PNG/GIF/WebP).
    """
    if raw[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if raw[:2] == b"\xff\xd8":
        return "image/jpeg"
    if raw[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    if raw[:4] == b"RIFF" and raw[8:12] == b"WEBP":
        return "image/webp"
    return None


MAX_REFERENCE_IMAGES = 5
MAX_REFERENCE_BYTES = 32 * 1024 * 1024


def _describe_image(image_name):
    """Keep the existing single-image entry point."""
    return _describe_images([image_name])


def _describe_images(image_names):
    """Describe ordered references in one vision request; keep all labels on failure."""
    input_dir = folder_paths.get_input_directory()
    if not input_dir:
        return None, "no input directory"
    if not 1 <= len(image_names) <= MAX_REFERENCE_IMAGES:
        return None, "provide 1–5 reference images"

    content = [{"type": "text", "text":
        "Describe each reference image separately in the supplied order. "
        "Label its paragraph <Picture 1>, <Picture 2>, and so on. "
        "Keep subjects and visual details associated with the correct image."}]
    total_bytes = 0
    try:
        root = Path(input_dir).resolve()
        for index, image_name in enumerate(image_names, 1):
            label = f"<Picture {index}>"
            relative = Path(image_name)
            if relative.is_absolute():
                return None, f"{label}: image must be inside the input directory"
            path = (root / relative).resolve()
            if not path.is_relative_to(root) or not path.is_file():
                return None, f"{label}: image not found in input directory"
            with open(path, "rb") as f:
                raw = f.read(MAX_REFERENCE_BYTES - total_bytes + 1)
            total_bytes += len(raw)
            if total_bytes > MAX_REFERENCE_BYTES:
                return None, "reference images exceed the combined 32 MiB vision limit"
            mime = _image_mime(raw)
            if not mime:
                return None, f"{label}: unsupported image format (must be JPEG/PNG/GIF/WebP)"
            content.append({"type": "text", "text": label})
            content.append({"type": "image_url", "image_url": {
                "url": "data:%s;base64,%s" % (mime, base64.b64encode(raw).decode("ascii"))
            }})
    except (OSError, ValueError, RuntimeError) as exc:
        return None, "read reference images failed: " + str(exc)

    _load_env(ENV_PATH)
    api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not api_key or api_key == "your_deepseek_api_key_here":
        return None, "set DEEPSEEK_API_KEY"
    model = os.environ.get("DEEPSEEK_VISION_MODEL", DEFAULT_VISION_MODEL).strip() or DEFAULT_VISION_MODEL
    system = (
        "You are a visual analyst for video generation. Return a concise paragraph per supplied image, "
        "with its exact <Picture N> label, describing visible subjects, style, palette, lighting, framing "
        "and distinctive visual identity. Do not merge the numbered descriptions, invent unseen details, "
        "or assume the images are first/last frames or a chronological storyboard. "
        "Do not wrap your answer in Markdown fences."
    )
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": content},
        ],
        "temperature": 0.3,
        "thinking": {"type": "disabled"},
        "stream": False,
    }
    try:
        return _post_deepseek(payload), None
    except Exception as exc:
        return None, str(exc)

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


@routes.post("/h3/prompt")
async def h3_prompt(request):
    """Generate the final H3 prompt for the page.

    (1) Describe up to five uploaded images in one ordered vision request.
        (2) Feed the user's idea (+ those descriptions) through the
        local h3-prompt-writing skill via DeepSeek in the selected text, reference,
        or first/last-frame mode. Returns it for editing before submit.
        The vision step degrades gracefully (vlm_error) so a model hiccup never
        blocks prompt generation.
    """
    try:
        body = await request.json()
    except Exception:
        return web.json_response({"error": "invalid JSON body"}, status=400)

    if not isinstance(body, dict):
        return web.json_response({"error": "JSON body must be an object"}, status=400)
    user_text = body.get("user_text") or ""
    mode = body.get("mode") or "T2VA"
    if not isinstance(user_text, str) or not isinstance(mode, str):
        return web.json_response({"error": "user_text and mode must be strings"}, status=400)
    user_text, mode = user_text.strip(), mode.strip()
    if not user_text:
        return web.json_response({"error": "Enter a video idea"}, status=400)

    if "image_names" in body:
        image_names = body["image_names"]
    else:
        image_name = body.get("image_name") or ""
        if not isinstance(image_name, str):
            return web.json_response({"error": "image_name must be a string"}, status=400)
        image_names = [image_name] if image_name.strip() else []
    if (not isinstance(image_names, list) or len(image_names) > MAX_REFERENCE_IMAGES
            or any(not isinstance(name, str) or not name.strip() for name in image_names)):
        return web.json_response({"error": "参考图片应为最多 5 个非空文件名组成的数组"}, status=400)
    image_names = [name.strip() for name in image_names]
    counts = {"T2VA": (0, 0), "Ref2VA": (1, 5), "I2VA": (1, 1), "L2VA": (1, 1), "FL2VA": (2, 2)}
    if mode not in counts:
        return web.json_response({"error": "不支持的创作模式，请刷新页面后重试"}, status=400)
    minimum, maximum = counts[mode]
    if not minimum <= len(image_names) <= maximum:
        return web.json_response({"error": f"{mode} 的图片数量不正确：需要 {minimum}–{maximum} 张"}, status=400)
    frame_count = body.get("frame_count")
    if frame_count is not None and (type(frame_count) is not int
            or not 124 <= frame_count <= 362 or frame_count % 17 != 5):
        return web.json_response({"error": "视频帧数不符合当前 H3 时间网格"}, status=400)

    image_description, vlm_error = None, None
    if len(image_names) == 1:
        image_description, vlm_error = await asyncio.to_thread(_describe_image, image_names[0])
    elif image_names:
        image_description, vlm_error = await asyncio.to_thread(_describe_images, image_names)

    skill = _skill_prompt()
    idea = user_text
    if frame_count is not None:
        idea += (f"\n\nThe generated clip has {frame_count} frames at 24 fps, "
                 f"duration {frame_count / 24:.3f} seconds. The last frame is at "
                 f"{(frame_count - 1) / 24:.3f} seconds; use that timestamp for a supplied last-frame anchor.")
    if image_names:
        labels = ", ".join(f"<Picture {index}>" for index in range(1, len(image_names) + 1))
        idea += (
            f"\n\nThere are exactly {len(image_names)} supplied images, in this order: {labels}. "
            "The user's 图1 / 图片1 refers to <Picture 1>, and subsequent numbers follow the same order. "
            "Preserve this mapping. Do not introduce unprovided image, video, or audio assets."
        )
        if mode == "Ref2VA":
            idea += " These are visual references, not automatically first/last frames. Use the six Ref2VA sections."
        elif mode == "I2VA":
            idea += (" <Picture 1> is the FIRST frame at 0.00 seconds. Start from that image, then develop "
                     "the action forward. Use the I2VA alignment instruction and three core sections.")
        elif mode == "L2VA":
            idea += (" <Picture 1> is the LAST frame, not the opening. Infer a compatible starting state "
                     "and converge to it by the final frame. Use the L2VA alignment instruction and three core sections.")
        else:
            idea += (" <Picture 1> is the FIRST frame at 0.00 seconds; <Picture 2> is the LAST frame. "
                     "Describe a continuous observable path from the first state to the last. "
                     "Use the FL2VA alignment instruction and three core sections, not Ref2VA subject definitions.")
    if image_description:
        idea += "\n\nNumbered reference image descriptions:\n" + image_description
    elif image_names:
        idea += "\n\nVisual analysis is unavailable. Use only the user's stated image roles; do not invent image contents."

    _load_env(ENV_PATH)
    model = os.environ.get("DEEPSEEK_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL
    instruction = (
        f"Input mode: {mode}. Rewrite the user's idea into the complete MiniMax H3 prompt. "
        "Return only the final prompt, with the exact fields and section order required by the skill. "
        "Do not explain your work or wrap the result in Markdown fences.\n\n"
        f"User idea:\n{idea}"
    )
    try:
        # urllib is blocking; keep the ComfyUI HTTP/WebSocket loop responsive.
        h3_prompt = await asyncio.to_thread(_post_deepseek, {
            "model": model,
            "messages": [
                {"role": "system", "content": skill},
                {"role": "user", "content": instruction},
            ],
            "temperature": 0.7,
            "thinking": {"type": "disabled"},
            "stream": False,
        })
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=502)

    return web.json_response({
        "h3_prompt": h3_prompt,
        "image_description": image_description,
        "vlm_error": vlm_error,
        "mode": mode,
    })


NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
