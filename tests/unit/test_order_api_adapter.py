import pytest

from ai_research_agent.domain.ports import OrderNotFoundError
from ai_research_agent.infrastructure.order_api_adapter import FakeOrderRepository


def test_returns_status_for_known_order():
    repo = FakeOrderRepository()

    status = repo.get_order_status("1001")

    assert status.status == "shipped"
    assert status.eta_days == 3


def test_raises_for_unknown_order():
    repo = FakeOrderRepository()

    with pytest.raises(OrderNotFoundError):
        repo.get_order_status("9999")