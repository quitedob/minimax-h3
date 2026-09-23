"""Prompt API regressions; no GPU, credentials, or external API calls required.

Run with the ComfyUI venv: python -m unittest comfyui_download.test_h3_prompt -v
"""

import asyncio
import importlib.util
import json
import os
import sys
import tempfile
import threading
import types
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

from aiohttp import web


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WEB_SOURCE = PROJECT_ROOT / "comfyui_download/h3_web_queue/__init__.py"
NODE_SOURCE = PROJECT_ROOT / "ComfyUI_windows_portable/ComfyUI/custom_nodes/h3_deepseek_prompt/__init__.py"


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {
        "server": types.SimpleNamespace(PromptServer=types.SimpleNamespace(
            instance=types.SimpleNamespace(routes=web.RouteTableDef()))),
        "folder_paths": types.SimpleNamespace(get_input_directory=lambda: None),
    }):
        spec.loader.exec_module(module)
    return module


class PromptRouteTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.module = load_module(WEB_SOURCE, "test_h3_web_queue")
        self.module._load_env = Mock()
        self.module._skill_prompt = Mock(return_value="complete unchanged skill")
        self.request = types.SimpleNamespace(json=AsyncMock(return_value={
            "user_text": "A puppy runs on grass.", "mode": "T2VA",
        }))

    async def assert_nonblocking(self, operation):
        entered = threading.Event()
        release = threading.Event()
        worker_threads = []

        def delayed(*args):
            worker_threads.append(threading.get_ident())
            entered.set()
            release.wait(timeout=2)
            return ("puppy reference", None) if operation == "_describe_image" else "final prompt"

        self.module._post_deepseek = Mock(return_value="final prompt")
        if operation == "_describe_image":
            self.request.json.return_value["image_name"] = "puppy.png"
            self.request.json.return_value["mode"] = "Ref2VA"
        with patch.object(self.module, operation, side_effect=delayed):
            task = asyncio.create_task(self.module.h3_prompt(self.request))
            try:
                for _ in range(100):
                    if entered.is_set():
                        break
                    await asyncio.sleep(0.01)
                self.assertTrue(entered.is_set())
                self.assertNotIn(threading.get_ident(), worker_threads)
                # The event loop can run another request while the API is waiting.
                self.assertFalse(task.done())
            finally:
                release.set()
                response = await task
        self.assertEqual(response.status, 200)
        body = json.loads(response.body)
        self.assertEqual(set(body), {"h3_prompt", "image_description", "vlm_error", "mode"})
        return body

    async def test_text_request_does_not_block_event_loop(self):
        await self.assert_nonblocking("_post_deepseek")

    async def test_vision_request_does_not_block_event_loop(self):
        body = await self.assert_nonblocking("_describe_image")
        self.assertEqual(body["image_description"], "puppy reference")

    async def test_rewrite_disables_thinking_and_keeps_full_skill(self):
        self.module._post_deepseek = Mock(return_value="final prompt")
        await self.module.h3_prompt(self.request)
        payload = self.module._post_deepseek.call_args.args[0]
        self.assertEqual(payload["thinking"], {"type": "disabled"})
        self.assertEqual(payload["messages"][0]["content"], "complete unchanged skill")
        self.assertFalse(payload["stream"])

    async def test_vision_failure_still_generates_prompt(self):
        self.request.json.return_value["image_name"] = "missing.png"
        self.module._describe_image = Mock(return_value=(None, "image missing"))
        self.module._post_deepseek = Mock(return_value="text-only fallback")
        response = await self.module.h3_prompt(self.request)
        self.assertEqual(json.loads(response.body)["vlm_error"], "image missing")
        self.assertEqual(response.status, 200)

    async def test_upstream_failure_is_502_without_retry(self):
        self.module._post_deepseek = Mock(side_effect=RuntimeError("upstream unavailable"))
        response = await self.module.h3_prompt(self.request)
        self.assertEqual(response.status, 502)
        self.module._post_deepseek.assert_called_once()

    async def test_invalid_json_and_empty_input_do_not_call_api(self):
        self.module._post_deepseek = Mock()
        self.request.json.side_effect = ValueError("invalid")
        self.assertEqual((await self.module.h3_prompt(self.request)).status, 400)
        self.request.json.side_effect = None
        self.request.json.return_value = {"user_text": " "}
        self.assertEqual((await self.module.h3_prompt(self.request)).status, 400)
        self.module._post_deepseek.assert_not_called()


class DeepSeekTransportTests(unittest.TestCase):
    def setUp(self):
        self.web_module = load_module(WEB_SOURCE, "test_h3_web_transport")
        self.node_module = load_module(NODE_SOURCE, "test_h3_prompt_node")
        self.web_module._load_env = Mock()
        self.environment = patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-only-key"})
        self.environment.start()
        self.addCleanup(self.environment.stop)

    def call_transport(self, module, base_url):
        if module is self.web_module:
            with patch.dict(os.environ, {"DEEPSEEK_BASE_URL": base_url}):
                return module._post_deepseek({"model": "deepseek-v4-flash"})
        return module._request_deepseek("test-only-key", base_url, "deepseek-v4-flash",
                                        "skill", "idea", 0.7)

    def test_official_direct_custom_proxy_and_node_thinking(self):
        for module in (self.web_module, self.node_module):
            for base_url, direct in (("https://api.deepseek.com/v1", True),
                                     ("https://custom.example/v1", False)):
                with self.subTest(module=module.__name__, base_url=base_url), \
                     patch("urllib.request.ProxyHandler") as proxy_handler, \
                     patch("urllib.request.build_opener") as build_opener:
                    response = build_opener.return_value.open.return_value.__enter__.return_value
                    response.read.return_value = json.dumps({
                        "choices": [{"message": {"content": " final prompt "}}],
                    }).encode()
                    self.assertEqual(self.call_transport(module, base_url), "final prompt")
                    if direct:
                        proxy_handler.assert_called_once_with({})
                    else:
                        proxy_handler.assert_called_once_with()
                    call = build_opener.return_value.open.call_args
                    self.assertEqual(call.kwargs["timeout"], 180)
                    self.assertEqual(call.args[0].full_url, base_url + "/chat/completions")
                    if module is self.node_module:
                        self.assertEqual(json.loads(call.args[0].data)["thinking"], {"type": "disabled"})

    def test_empty_content_is_not_reported_as_success(self):
        for module in (self.web_module, self.node_module):
            with self.subTest(module=module.__name__), patch("urllib.request.build_opener") as opener:
                opener.return_value.open.return_value.__enter__.return_value.read.return_value = b'{"choices": []}'
                with self.assertRaisesRegex(RuntimeError, "no prompt content"):
                    self.call_transport(module, "https://api.deepseek.com")

    def test_connection_failure_preserves_error(self):
        for module in (self.web_module, self.node_module):
            with self.subTest(module=module.__name__), patch("urllib.request.build_opener") as opener:
                opener.return_value.open.side_effect = urllib.error.URLError("offline")
                with self.assertRaisesRegex(RuntimeError, "connection failed"):
                    self.call_transport(module, "https://api.deepseek.com")

    def test_vision_payload_disables_thinking(self):
        with tempfile.TemporaryDirectory() as input_dir:
            image_path = Path(input_dir) / "test.png"
            image_path.write_bytes(b"\x89PNG\r\n\x1a\n")
            self.web_module.folder_paths.get_input_directory = lambda: input_dir
            with patch.object(self.web_module, "_post_deepseek", return_value="description") as post:
                self.assertEqual(self.web_module._describe_image("test.png"), ("description", None))
            self.assertEqual(post.call_args.args[0]["thinking"], {"type": "disabled"})


if __name__ == "__main__":
    unittest.main()
