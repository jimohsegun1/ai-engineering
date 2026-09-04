"""A small inventory management module, used as sample input for
demonstrating code-aware (language-specific) text splitting."""


class InventoryItem:
    """Represents a single item tracked in the inventory."""

    def __init__(self, name: str, quantity: int, unit_price: float):
        self.name = name
        self.quantity = quantity
        self.unit_price = unit_price

    def total_value(self) -> float:
        """Return the total value of this item (quantity * unit price)."""
        return self.quantity * self.unit_price


class Inventory:
    """A simple in-memory collection of InventoryItem objects."""

    def __init__(self):
        self.items: dict[str, InventoryItem] = {}

    def add_item(self, name: str, quantity: int, unit_price: float) -> None:
        """Add a new item, or increase quantity if it already exists."""
        if name in self.items:
            self.items[name].quantity += quantity
        else:
            self.items[name] = InventoryItem(name, quantity, unit_price)

    def remove_item(self, name: str, quantity: int) -> None:
        """Remove a quantity of an existing item, raising if not enough stock."""
        if name not in self.items:
            raise KeyError(f"No such item: {name}")
        if self.items[name].quantity < quantity:
            raise ValueError(f"Not enough stock of {name} to remove {quantity}")
        self.items[name].quantity -= quantity

    def total_inventory_value(self) -> float:
        """Return the combined value of every item currently in stock."""
        return sum(item.total_value() for item in self.items.values())

    def low_stock_items(self, threshold: int = 5) -> list[str]:
        """Return the names of items whose quantity is at or below threshold."""
        return [name for name, item in self.items.items() if item.quantity <= threshold]


def restock_from_supplier_order(inventory: Inventory, order: dict[str, int], unit_prices: dict[str, float]) -> None:
    """Apply a supplier order (name -> quantity) to the given inventory."""
    for name, quantity in order.items():
        inventory.add_item(name, quantity, unit_prices.get(name, 0.0))


def generate_low_stock_report(inventory: Inventory, threshold: int = 5) -> str:
    """Build a human-readable report of items that need restocking."""
    low_stock = inventory.low_stock_items(threshold)
    if not low_stock:
        return "All items are sufficiently stocked."
    lines = [f"- {name} (qty: {inventory.items[name].quantity})" for name in low_stock]
    return "Low stock items:\n" + "\n".join(lines)
