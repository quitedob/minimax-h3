import json
import logging
import os
import urllib.error
import urllib.request
from pathlib import Path


NODE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = NODE_ROOT.parents[3]
ENV_PATH = PROJECT_ROOT / ".env"
SKILL_PATH = PROJECT_ROOT / ".claude" / "skills" / "h3-prompt-writing"
DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-v4-flash"


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
    files = [SKILL_PATH / "SKILL.md", SKILL_PATH / "references" / "base-en.txt", SKILL_PATH / "references" / "ref-en.txt"]
    missing = [str(path) for path in files if not path.exists()]
    if missing:
        raise RuntimeError("H3 prompt skill files are missing: " + ", ".join(missing))
    sections = []
    for path in files:
        sections.append(f"===== {path.relative_to(PROJECT_ROOT).as_posix()} =====\n{path.read_text(encoding='utf-8')}" )
    return "\n\n".join(sections)


def _request_deepseek(api_key, base_url, model, system_prompt, user_prompt, temperature):
    payload = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "stream": False,
    }).encode("utf-8")
    request = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            result = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"DeepSeek API HTTP {error.code}: {detail[:500]}") from error
    except urllib.error.URLError as error:
        raise RuntimeError(f"DeepSeek API connection failed: {error.reason}") from error
    choices = result.get("choices") or []
    if not choices or not choices[0].get("message", {}).get("content"):
        raise RuntimeError("DeepSeek API returned no prompt content")
    return choices[0]["message"]["content"].strip()


class H3DeepSeekPrompt:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "user_text": ("STRING", {"multiline": True, "default": "", "tooltip": "Your T2V or I2V idea. The complete H3 prompt-writing skill is sent with it."}),
            "mode": (["T2VA", "I2VA", "FL2VA", "L2VA", "Ref2VA"], {"default": "T2VA"}),
        }, "optional": {
            "model": ("STRING", {"default": DEFAULT_MODEL}),
            "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.05}),
        }}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("h3_prompt",)
    FUNCTION = "generate"
    CATEGORY = "H3/Prompt"
    DESCRIPTION = "Send a T2V/I2V idea to DeepSeek using the complete local H3 prompt-writing skill."

    def generate(self, user_text, mode, model=DEFAULT_MODEL, temperature=0.7):
        _load_env(ENV_PATH)
        api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
        if not api_key or api_key == "your_deepseek_api_key_here":
            raise RuntimeError(f"Set DEEPSEEK_API_KEY in {ENV_PATH}")
        if not user_text.strip():
            raise RuntimeError("Enter a video idea in user_text")
        system_prompt = _skill_prompt()
        instruction = (
            f"Input mode: {mode}. Rewrite the user's idea into the complete MiniMax H3 prompt. "
            "Return only the final prompt, with the exact fields and section order required by the skill. "
            "Do not explain your work or wrap the result in Markdown fences.\n\n"
            f"User idea:\n{user_text.strip()}"
        )
        result = _request_deepseek(
            api_key,
            os.environ.get("DEEPSEEK_BASE_URL", DEFAULT_BASE_URL),
            model.strip() or DEFAULT_MODEL,
            system_prompt,
            instruction,
            temperature,
        )
        logging.info("H3 DeepSeek prompt generated for %s", mode)
        return (result,)


NODE_CLASS_MAPPINGS = {"H3DeepSeekPrompt": H3DeepSeekPrompt}
NODE_DISPLAY_NAME_MAPPINGS = {"H3DeepSeekPrompt": "H3 DeepSeek Prompt"}
