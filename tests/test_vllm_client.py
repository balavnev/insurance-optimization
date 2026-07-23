"""Tests the real VLLMOpenAIClient against a tiny local HTTP server standing
in for vLLM's OpenAI-compatible endpoint -- genuine request/response
handling over a real socket, entirely offline, no vendor cluster needed."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from offer_opt.llm.client import LLMUnavailable, VLLMOpenAIClient


def _make_server(response_content: str, status: int = 200, models_ok: bool = True):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass  # keep test output clean

        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length) or b"{}")
            self.server.last_request = body  # type: ignore[attr-defined]
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            payload = {"choices": [{"message": {"content": response_content}}]}
            self.wfile.write(json.dumps(payload).encode("utf-8"))

        def do_GET(self):
            self.send_response(200 if models_ok else 500)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"data": [{"id": "Qwen2.5-32B-Instruct"}]}).encode("utf-8"))

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.last_request = None
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def test_raises_immediately_with_no_base_url_configured(monkeypatch):
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    with pytest.raises(LLMUnavailable):
        VLLMOpenAIClient()


def test_raises_llm_unavailable_when_endpoint_unreachable():
    client = VLLMOpenAIClient(base_url="http://127.0.0.1:1", timeout_s=1.0)
    with pytest.raises(LLMUnavailable):
        client.complete_json(system="s", user="u", json_schema={})


def test_complete_json_parses_a_real_http_round_trip():
    server = _make_server(response_content='{"measure": "count", "confidence": "high"}')
    try:
        client = VLLMOpenAIClient(base_url=f"http://127.0.0.1:{server.server_port}", api_key="test-key")
        result = client.complete_json(system="classify", user="raw_type='foo'", json_schema={})
        assert result == {"measure": "count", "confidence": "high"}

        # Real request actually carried the right shape (model, messages,
        # auth header) -- not just "some request happened".
        req = server.last_request
        assert req["messages"][0]["role"] == "system"
        assert req["messages"][1]["content"] == "raw_type='foo'"
        assert req["temperature"] == 0.0
    finally:
        server.shutdown()


def test_complete_json_strips_think_blocks_from_a_real_response():
    """Proves the <think>-stripping in llm/parsing.py is actually wired into
    the real client's response handling, not just unit-tested in isolation."""
    server = _make_server(response_content='<think>reasoning...</think>\n{"measure": "cost"}')
    try:
        client = VLLMOpenAIClient(base_url=f"http://127.0.0.1:{server.server_port}")
        result = client.complete_json(system="s", user="u", json_schema={})
        assert result == {"measure": "cost"}
    finally:
        server.shutdown()


def test_complete_json_raises_llm_unavailable_on_unparseable_content():
    server = _make_server(response_content="not json at all and no braces either")
    try:
        client = VLLMOpenAIClient(base_url=f"http://127.0.0.1:{server.server_port}")
        with pytest.raises(LLMUnavailable):
            client.complete_json(system="s", user="u", json_schema={})
    finally:
        server.shutdown()


def test_health_check_true_when_models_endpoint_ok():
    server = _make_server(response_content="{}", models_ok=True)
    try:
        client = VLLMOpenAIClient(base_url=f"http://127.0.0.1:{server.server_port}")
        assert client.health_check() is True
    finally:
        server.shutdown()


def test_health_check_false_when_unreachable():
    client = VLLMOpenAIClient(base_url="http://127.0.0.1:1", timeout_s=1.0)
    assert client.health_check() is False
