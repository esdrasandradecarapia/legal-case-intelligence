from ai_research_agent.application.agent.tools import TOOLS_SCHEMA
from ai_research_agent.application.ask_question import AskQuestion
from ai_research_agent.domain.entities import Answer
from ai_research_agent.domain.ports import OrderNotFoundError, OrderRepository, ToolCallingLLM

ASK_FOR_ORDER_ID_MESSAGE = "Claro! Pode me informar o número do seu pedido para eu consultar?"


class AgentRouter:
    def __init__(
        self,
        tool_calling_llm: ToolCallingLLM,
        ask_question: AskQuestion,
        order_repository: OrderRepository,
    ) -> None:
        self.tool_calling_llm = tool_calling_llm
        self.ask_question = ask_question
        self.order_repository = order_repository

    def route(self, question: str) -> Answer:
        decision = self.tool_calling_llm.decide_tool(question, TOOLS_SCHEMA)

        if decision.tool_name == "check_order_status":
            return self._handle_order_status(decision.arguments)

        return self.ask_question.execute(question)

    def _handle_order_status(self, arguments: dict) -> Answer:
        order_id = arguments.get("order_id", "")

        if not order_id.isdigit():
            return Answer(text=ASK_FOR_ORDER_ID_MESSAGE, sources=[])

        try:
            status = self.order_repository.get_order_status(order_id)
        except OrderNotFoundError:
            return Answer(
                text=f"Não encontrei nenhum pedido com o número {order_id}.",
                sources=[],
            )

        text = f"Seu pedido {status.order_id} está com status '{status.status}'"
        if status.status != "delivered":
            text += f", previsão de {status.eta_days} dia(s)."
        else:
            text += "."

        return Answer(text=text, sources=[])