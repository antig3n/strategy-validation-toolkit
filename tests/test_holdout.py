"""Persistent reservations, canonical parameters and concurrency."""

import json
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path
from threading import Event
from typing import Any

import pytest

from svt import HoldoutConsumedError, SealedHoldout, parameter_fingerprint
from svt.holdout import JSONValue


def test_persistent_single_use_even_if_parameters_change(tmp_path: Path) -> None:
    journal = tmp_path / "audit.sqlite3"
    holdout = SealedHoldout(journal, "2040-01-01", "2040-12-31")
    assert holdout.history() == ()
    assert holdout.evaluate({"window": 8}, lambda parameters: parameters["window"]) == 8
    reopened = SealedHoldout(journal, date(2040, 1, 1), date(2040, 12, 31))
    called = []
    with pytest.raises(HoldoutConsumedError):
        reopened.evaluate({"window": 9}, lambda _: called.append(True))
    assert not called
    record, = reopened.history()
    assert record.status == "completed"
    assert record.fingerprint == parameter_fingerprint({"window": 8})
    assert json.loads(record.parameters_json) == {"window": 8}
    assert record.finished_at is not None
    assert record.error_type is None


def test_fingerprint_canonicalization_and_type_distinctions() -> None:
    first = {"window": 8, "nested": {"b": 2, "a": [True, None]}}
    second = {"nested": {"a": [True, None], "b": 2}, "window": 8}
    assert parameter_fingerprint(first) == parameter_fingerprint(second)
    assert len(parameter_fingerprint(first)) == 64
    assert parameter_fingerprint({"x": 1}) != parameter_fingerprint({"x": 1.0})


@pytest.mark.parametrize("error_type", [RuntimeError, KeyboardInterrupt])
def test_failed_evaluation_stays_consumed(tmp_path: Path, error_type: type[BaseException]) -> None:
    holdout = SealedHoldout(tmp_path / "audit.sqlite3", "2040-01-01", "2040-12-31")

    def fail(_: object) -> None:
        raise error_type("synthetic failure")

    with pytest.raises(error_type):
        holdout.evaluate({}, fail)
    with pytest.raises(HoldoutConsumedError):
        holdout.evaluate({}, lambda _: None)
    record, = holdout.history()
    assert record.status == "failed"
    assert record.error_type == error_type.__name__


def test_overlap_refused_and_disjoint_period_allowed(tmp_path: Path) -> None:
    journal = tmp_path / "audit.sqlite3"
    first = SealedHoldout(journal, "2040-02-01", "2040-03-01")
    first.evaluate({}, lambda _: 1)
    for start, end in [("2040-03-01", "2040-04-01"), ("2040-02-15", "2040-02-20"),
                       ("2040-01-01", "2040-12-31")]:
        with pytest.raises(HoldoutConsumedError):
            SealedHoldout(journal, start, end).evaluate({}, lambda _: 2)
    assert SealedHoldout(journal, "2040-03-02", "2040-04-01").evaluate({}, lambda _: 3) == 3
    assert len(first.history()) == 2


def test_concurrent_attempt_is_refused_before_first_finishes(tmp_path: Path) -> None:
    journal = tmp_path / "audit.sqlite3"
    first = SealedHoldout(journal, "2040-01-01", "2040-12-31")
    second = SealedHoldout(journal, "2040-01-01", "2040-12-31")
    entered, release = Event(), Event()

    def evaluate(_: object) -> int:
        entered.set()
        assert release.wait(5)
        return 7

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(first.evaluate, {}, evaluate)
        try:
            assert entered.wait(5)
            assert second.history()[0].status == "running"
            with pytest.raises(HoldoutConsumedError):
                second.evaluate({}, lambda _: None)
        finally:
            release.set()
        assert future.result(timeout=5) == 7


@pytest.mark.parametrize("parameters", [
    {"x": float("nan")}, {"x": float("inf")}, {1: "value"},
    {"nested": {1: 2}}, {"x": object()}, {"x": (1, 2)},
])
def test_invalid_parameters_consume_nothing(tmp_path: Path, parameters: dict[str, Any]) -> None:
    holdout = SealedHoldout(tmp_path / "audit.sqlite3", "2040-01-01", "2040-12-31")
    with pytest.raises(ValueError):
        holdout.evaluate(parameters, lambda _: None)
    assert holdout.history() == ()
    assert holdout.evaluate({}, lambda _: 4) == 4


def test_callback_uses_a_deep_parameter_copy(tmp_path: Path) -> None:
    holdout = SealedHoldout(tmp_path / "audit.sqlite3", "2040-01-01", "2040-12-31")
    original = {"nested": [1, 2]}

    def mutate(parameters: Mapping[str, JSONValue]) -> None:
        nested = parameters["nested"]
        assert isinstance(nested, list)
        nested.append(3)

    holdout.evaluate(original, mutate)
    assert original == {"nested": [1, 2]}
    assert json.loads(holdout.history()[0].parameters_json) == original


def test_cyclic_parameters_consume_nothing(tmp_path: Path) -> None:
    holdout = SealedHoldout(tmp_path / "audit.sqlite3", "2040-01-01", "2040-12-31")
    cyclic: dict[str, JSONValue] = {}
    cyclic["self"] = cyclic
    with pytest.raises(ValueError, match="acyclic"):
        holdout.evaluate(cyclic, lambda _: None)
    assert holdout.history() == ()


@pytest.mark.parametrize("start, end", [
    ("2041-01-01", "2040-01-01"), ("invalid", "2040-01-01"),
    ("20400101", "2040-12-31"),
])
def test_invalid_intervals(tmp_path: Path, start: str, end: str) -> None:
    with pytest.raises(ValueError):
        SealedHoldout(tmp_path / "audit.sqlite3", start, end)
