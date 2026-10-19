TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "check_order_status",
            "description": (
                "Get the status and delivery estimate for a specific customer order, "
                "given its order ID. Use this when the customer asks about a specific "
                "order, delivery, or tracking."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "The numeric order ID mentioned by the customer",
                    },
                },
                "required": ["order_id"],
            },
        },
    },
]