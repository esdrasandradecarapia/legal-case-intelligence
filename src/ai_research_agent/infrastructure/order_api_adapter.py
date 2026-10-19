from ai_research_agent.domain.ports import OrderNotFoundError, OrderStatus

FAKE_ORDERS: dict[str, OrderStatus] = {
    "1001": OrderStatus(order_id="1001", status="shipped", eta_days=3),
    "1002": OrderStatus(order_id="1002", status="delivered", eta_days=0),
    "1003": OrderStatus(order_id="1003", status="delayed", eta_days=10),
}


class FakeOrderRepository:
    def get_order_status(self, order_id: str) -> OrderStatus:
        if order_id not in FAKE_ORDERS:
            raise OrderNotFoundError(order_id)
        return FAKE_ORDERS[order_id]