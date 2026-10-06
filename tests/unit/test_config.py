from ai_research_agent.infrastructure.config import Settings


def test_settings_loads_openai_api_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    settings = Settings()

    assert settings.openai_api_key == "test-key"