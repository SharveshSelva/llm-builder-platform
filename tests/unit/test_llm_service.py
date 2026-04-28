import pytest
from unittest.mock import patch, MagicMock
from backend.services.llm import _pick_model, call_llm
from backend.config import get_settings

settings = get_settings()


class TestPickModel:
    def test_fast_returns_primary(self):
        assert _pick_model("fast") == settings.primary_model

    def test_smart_returns_smart(self):
        assert _pick_model("smart") == settings.smart_model

    def test_unknown_mode_returns_primary(self):
        assert _pick_model("turbo") == settings.primary_model


class TestCallLlm:
    def _mock_response(self, text="Test response", tokens=50):
        choice = MagicMock()
        choice.message.content = text
        usage = MagicMock()
        usage.total_tokens = tokens
        resp = MagicMock()
        resp.choices = [choice]
        resp.usage = usage
        return resp

    @pytest.mark.asyncio
    async def test_successful_call_returns_structured_output(self):
        mock_resp = self._mock_response("Hello world", tokens=30)
        with patch("backend.services.llm._get_client") as mock_client_fn:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_resp
            mock_client_fn.return_value = mock_client

            result = await call_llm("Say hello", mode="fast")

        assert result.raw_text == "Hello world"
        assert result.tokens_used == 30
        assert result.model_used == settings.primary_model
        assert result.latency_ms > 0

    @pytest.mark.asyncio
    async def test_smart_mode_uses_smart_model(self):
        mock_resp = self._mock_response("Smart response")
        with patch("backend.services.llm._get_client") as mock_client_fn:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_resp
            mock_client_fn.return_value = mock_client

            result = await call_llm("Explain quantum computing", mode="smart")

        assert result.model_used == settings.smart_model

    @pytest.mark.asyncio
    async def test_fallback_on_timeout(self):
        from groq import APITimeoutError
        primary_resp = self._mock_response("Fallback response")

        call_count = 0
        def side_effect(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise APITimeoutError(request=MagicMock())
            return primary_resp

        with patch("backend.services.llm._get_client") as mock_client_fn:
            mock_client = MagicMock()
            mock_client.chat.completions.create.side_effect = side_effect
            mock_client_fn.return_value = mock_client

            result = await call_llm("test", mode="fast")

        assert result.model_used == settings.fallback_model
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_json_mode_parses_dict(self):
        mock_resp = self._mock_response('{"name": "Alice", "score": 9}')
        with patch("backend.services.llm._get_client") as mock_client_fn:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_resp
            mock_client_fn.return_value = mock_client

            result = await call_llm(
                "Extract name and score",
                output_schema={"name": "str", "score": 0},
            )

        assert result.data == {"name": "Alice", "score": 9}
