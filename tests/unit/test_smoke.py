from ai_research_agent import hello


def test_hello_returns_greeting():
    result = hello()

    assert result == "Hello from ai-research-agent!"
