from ai_research_agent.application.agent.router import AgentRouter
from ai_research_agent.application.ask_question import AskQuestion
from ai_research_agent.domain.entities import Answer
from ai_research_agent.domain.ports import OrderNotFoundError, OrderStatus, ToolDecision
from ai_research_agent.infrastructure.order_api_adapter import FakeOrderRepository


class FakeToolCallingLLM:
    def __init__(self, decision: ToolDecision):
        self.decision = decision

    def decide_tool(self, question: str, tools: list[dict]) -> ToolDecision:
        return self.decision


class FakeAskQuestion:
    def __init__(self, answer: Answer):
        self.answer = answer
        self.received_questions = []

    def execute(self, question: str) -> Answer:
        self.received_questions.append(question)
        return self.answer


def test_routes_to_order_status_when_llm_chooses_that_tool():
    tool_calling_llm = FakeToolCallingLLM(
        decision=ToolDecision(tool_name="check_order_status", arguments={"order_id": "1001"})
    )
    ask_question = FakeAskQuestion(answer=Answer(text="não deveria chegar aqui", sources=[]))
    order_repository = FakeOrderRepository()

    router = AgentRouter(
        tool_calling_llm=tool_calling_llm,
        ask_question=ask_question,
        order_repository=order_repository,
    )

    answer = router.route("Cadê meu pedido 1001?")

    assert "shipped" in answer.text
    assert ask_question.received_questions == []


def test_routes_to_order_status_with_unknown_order():
    tool_calling_llm = FakeToolCallingLLM(
        decision=ToolDecision(tool_name="check_order_status", arguments={"order_id": "9999"})
    )
    ask_question = FakeAskQuestion(answer=Answer(text="não deveria chegar aqui", sources=[]))
    order_repository = FakeOrderRepository()

    router = AgentRouter(
        tool_calling_llm=tool_calling_llm,
        ask_question=ask_question,
        order_repository=order_repository,
    )

    answer = router.route("Cadê meu pedido 9999?")

    assert "não encontrei" in answer.text.lower()


def test_falls_back_to_rag_when_no_tool_is_chosen():
    tool_calling_llm = FakeToolCallingLLM(decision=ToolDecision(tool_name=None, arguments={}))
    expected_answer = Answer(text="Nossa política permite troca em até 30 dias.", sources=[])
    ask_question = FakeAskQuestion(answer=expected_answer)
    order_repository = FakeOrderRepository()

    router = AgentRouter(
        tool_calling_llm=tool_calling_llm,
        ask_question=ask_question,
        order_repository=order_repository,
    )

    answer = router.route("Qual a política de troca?")

    assert answer == expected_answer
    assert ask_question.received_questions == ["Qual a política de troca?"]