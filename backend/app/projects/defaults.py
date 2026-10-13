"""Application defaults for new projects.

Millimeters are canonical. A project may store its own stock thickness.
That stored value is independent of this default, and this default does
not replace a thickness already saved on a project.
"""

from app.schemas.project import Stock

DEFAULT_STOCK_THICKNESS_MM = 38.10


def stock_for_create(stock: Stock) -> Stock:
    """Persist the application default when a new project omits thickness."""

    if stock.thickness_mm is not None:
        return stock
    return stock.model_copy(update={"thickness_mm": DEFAULT_STOCK_THICKNESS_MM})


def stock_for_update(requested: Stock, current: Stock) -> Stock:
    """Keep a saved project thickness when an update does not send one."""

    if requested.thickness_mm is not None:
        return requested
    return requested.model_copy(update={"thickness_mm": current.thickness_mm})
