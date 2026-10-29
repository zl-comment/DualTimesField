"""Closed typed action vocabulary and state machine for the M1 contract.

The D arm may propose ONLY these four actions with bounded enum/ref parameters.
Anything else — unknown actions, URLs, shell/Python, free strings, MW/price/quantile
numbers, unlisted refs, out-of-order calls — is rejected BEFORE dispatch with zero
side effects.  Terminal states are absorbing.
"""
from __future__ import annotations

import enum


class M1ActionKind(enum.Enum):
    LIST = "LIST"
    FETCH_REF = "FETCH_REF"
    FINALIZE_TYPED = "FINALIZE_TYPED"
    FINALIZE_BASE = "FINALIZE_BASE"


class M1State(enum.Enum):
    INIT = "INIT"
    A_READY = "A_READY"
    LISTED = "LISTED"
    FETCHED = "FETCHED"
    TYPED_READY = "TYPED_READY"
    TERMINAL_TYPED = "TERMINAL_TYPED"
    TERMINAL_BASE_FALLBACK = "TERMINAL_BASE_FALLBACK"
    TERMINAL_SYSTEM_MISSING = "TERMINAL_SYSTEM_MISSING"
    TERMINAL_FAILED_NO_COMMIT = "TERMINAL_FAILED_NO_COMMIT"


TERMINAL_STATES = frozenset({
    M1State.TERMINAL_TYPED, M1State.TERMINAL_BASE_FALLBACK,
    M1State.TERMINAL_SYSTEM_MISSING, M1State.TERMINAL_FAILED_NO_COMMIT,
})

# The declared ref arity of every action: no ref, or one opaque v_ ref.
ACTION_REF_MODE = {
    M1ActionKind.LIST: "NONE",
    M1ActionKind.FETCH_REF: "V_REF",
    M1ActionKind.FINALIZE_TYPED: "NONE",
    M1ActionKind.FINALIZE_BASE: "NONE",
}


class ProposalRejected(Exception):
    """A structurally valid but non-executable proposal: zero side effects."""

    def __init__(self, reason_code: str = "ACTION_PRECONDITION", detail: str = ""):
        super().__init__(f"proposal rejected: {reason_code} {detail}")
        self.reason_code = reason_code
        self.detail = detail


def enabled_actions(state: M1State, *, list_calls: int, get_calls: int,
                    available_refs: tuple[str, ...] = (),
                    fetched_refs: tuple[str, ...] = (),
                    ) -> tuple[M1ActionKind, ...]:
    """The controller-side enabled-action set for the current machine state."""
    if state in TERMINAL_STATES or state is M1State.INIT:
        return ()
    if state is M1State.A_READY:
        return (M1ActionKind.LIST, M1ActionKind.FINALIZE_BASE)
    actions: list[M1ActionKind] = []
    if get_calls < 2 and any(ref not in fetched_refs for ref in available_refs):
        actions.append(M1ActionKind.FETCH_REF)
    if state in (M1State.FETCHED, M1State.TYPED_READY):
        actions.append(M1ActionKind.FINALIZE_TYPED)
    actions.append(M1ActionKind.FINALIZE_BASE)
    return tuple(actions)


def validate_proposal(state: M1State, kind: M1ActionKind, ref: str, *,
                      list_calls: int, get_calls: int,
                      available_refs: tuple[str, ...] = (),
                      fetched_refs: tuple[str, ...] = ()) -> None:
    """Controller-side precondition re-check; rejection leaves state untouched."""
    if state in TERMINAL_STATES:
        raise ProposalRejected(detail=f"terminal state {state.name} is absorbing")
    enabled = enabled_actions(state, list_calls=list_calls, get_calls=get_calls,
                              available_refs=available_refs,
                              fetched_refs=fetched_refs)
    if kind not in enabled:
        raise ProposalRejected(
            detail=f"{kind.name} is not enabled in {state.name} "
                   f"(enabled={[item.name for item in enabled]})")
    mode = ACTION_REF_MODE[kind]
    if mode == "NONE" and ref != "NONE":
        raise ProposalRejected(detail=f"{kind.name} requires ref NONE, got {ref!r}")
    if mode == "V_REF":
        if ref not in available_refs:
            raise ProposalRejected(detail=f"ref {ref!r} is not an enabled listed ref")
        if ref in fetched_refs:
            raise ProposalRejected(detail=f"ref {ref!r} was already fetched")
