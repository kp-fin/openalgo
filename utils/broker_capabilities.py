"""
Per-broker product-type capability gate.

Product-type validation (restx_api/schemas.py) is a single shared layer across
all broker plugins, but not every broker's mapping/transform_data.py actually
understands every product type -- an unrecognized product string typically
falls through that broker's own mapping to a silent MIS/intraday default
instead of erroring. This module gates a small set of broker-specific product
types (currently just MTF) before an order ever reaches that per-broker
mapping code, so an unsupported combination fails loudly instead of silently
placing the wrong kind of order.
"""

# Broker names whose plugin actually maps MTF (Margin Trading Facility) to a
# real broker-side product type. Add a broker here only once its own
# broker/<name>/mapping/transform_data.py has a real MTF branch, not just a
# read-back/display mapping.
MTF_SUPPORTED_BROKERS: set[str] = {"dhan"}


def validate_product_for_broker(broker: str, product: str) -> tuple[bool, str | None]:
    """Check whether `product` is actually supported by `broker`'s plugin.

    Returns (True, None) if fine, or (False, error_message) if this specific
    broker/product combination is known to be unsupported.
    """
    if product == "MTF" and broker not in MTF_SUPPORTED_BROKERS:
        return False, f"Product type MTF is not supported for broker '{broker}'."
    return True, None
