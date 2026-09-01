"""Default the OpenAlgo SDK ``strategy=`` tag for hosted Python strategies.

The strategy host injects ``STRATEGY_NAME`` into the subprocess environment.
Scripts that omit ``strategy`` (or pass an empty string) would otherwise land
in Analyzer/sandbox as untagged fills, so Strategy Portfolio cannot attribute
P&L. This wrapper fills the tag on SDK order methods before the request leaves
the hosted process.

An explicit non-empty ``strategy=`` argument is left unchanged so
``orderstatus`` / cancel flows that use a custom tag keep working.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

# SDK method names used by hosted strategies (current + snake_case aliases).
_ORDER_METHODS = (
    "placeorder",
    "place_order",
    "placesmartorder",
    "place_smart_order",
    "basketorder",
    "place_basket_order",
    "splitorder",
    "place_split_order",
    "optionsorder",
    "optionsmultiorder",
    "modifyorder",
    "cancelorder",
    "cancelallorder",
    "closeposition",
    "placegttorder",
    "modifygttorder",
    "cancelgttorder",
)


def hosted_strategy_name(explicit: str | None = None) -> str:
    """Return the hosted strategy display name from env (or ``explicit``)."""
    if explicit is not None and str(explicit).strip():
        return str(explicit).strip()
    return (os.environ.get("STRATEGY_NAME") or "").strip()


def apply_default_strategy_tag(kwargs: dict[str, Any], strategy_name: str) -> dict[str, Any]:
    """Fill ``strategy`` when missing or blank. Does not override a set tag."""
    if not strategy_name:
        return kwargs
    existing = kwargs.get("strategy")
    if existing is None or (isinstance(existing, str) and not existing.strip()):
        kwargs = dict(kwargs)
        kwargs["strategy"] = strategy_name
    return kwargs


def install_hosted_strategy_tag(strategy_name: str | None = None) -> bool:
    """Patch ``openalgo.api`` order methods in this process. Idempotent.

    Returns True if the SDK class was patched (or already patched).
    """
    name = hosted_strategy_name(strategy_name)
    if not name:
        return False
    try:
        import openalgo
    except ImportError:
        return False

    cls = getattr(openalgo, "api", None)
    if cls is None:
        return False
    if getattr(cls, "_openalgo_hosted_tag_installed", False):
        return True

    def _wrap(fn: Callable[..., Any]) -> Callable[..., Any]:
        def wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:
            return fn(self, *args, **apply_default_strategy_tag(kwargs, name))

        return wrapped

    for method_name in _ORDER_METHODS:
        orig = getattr(cls, method_name, None)
        if callable(orig):
            setattr(cls, method_name, _wrap(orig))
    cls._openalgo_hosted_tag_installed = True
    return True
