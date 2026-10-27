"""Persistent, single-use evaluation of a declared final date interval."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Callable, Mapping
from contextlib import closing
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import TypeAlias, TypeVar

JSONValue: TypeAlias = None | bool | int | float | str | list["JSONValue"] | dict[str, "JSONValue"]
T = TypeVar("T")


class HoldoutConsumedError(RuntimeError):
    """The requested period overlaps an already reserved evaluation."""


@dataclass(frozen=True)
class HoldoutRecord:
    """An auditable reservation, including the original parameter snapshot."""

    start: date
    end: date
    fingerprint: str
    parameters_json: str
    status: str
    started_at: str
    finished_at: str | None
    error_type: str | None


def _date(value: date | str) -> date:
    """Normalize a date or canonical ISO date string."""
    if isinstance(value, datetime):
        raise ValueError("period boundaries must be dates, not timestamps")
    parsed = date.fromisoformat(value) if isinstance(value, str) else value
    if not isinstance(parsed, date) or (isinstance(value, str) and parsed.isoformat() != value):
        raise ValueError("period boundaries must be dates or canonical ISO dates")
    return parsed


def _validate_json(value: object) -> None:
    """Reject non-JSON objects and non-string keys before canonical encoding."""
    if value is None or type(value) in (bool, int, float, str):
        return
    if isinstance(value, list):
        for item in value:
            _validate_json(item)
        return
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        for item in value.values():
            _validate_json(item)
        return
    raise ValueError("parameters must contain JSON values with string object keys")


def parameter_fingerprint(parameters: Mapping[str, JSONValue]) -> str:
    """Hash canonical JSON parameters for an auditable, fixed evaluation plan.

    Motivation: Dwork et al. (2015), "The Reusable Holdout: Preserving
    Validity in Adaptive Data Analysis", Science 349. This implementation is
    a single-use guard, not that paper's reusable or private mechanism.
    Input must be a mapping of finite JSON values with string keys. Mapping
    order does not affect the SHA-256 digest. Numeric JSON types remain
    distinct (for example, 1 and 1.0). No random generator is involved.
    """
    return hashlib.sha256(_canonical_parameters(parameters).encode("utf-8")).hexdigest()


def _canonical_parameters(parameters: Mapping[str, JSONValue]) -> str:
    """Serialize a validated parameter snapshot reproducibly."""
    if not isinstance(parameters, Mapping):
        raise ValueError("parameters must be a mapping")
    snapshot = dict(parameters)
    try:
        _validate_json(snapshot)
        return json.dumps(snapshot, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except RecursionError as error:
        raise ValueError("parameters must contain acyclic JSON values") from error


class SealedHoldout:
    """Single-use date interval, guarded by a persistent SQLite journal.

    Motivation: Dwork et al. (2015), "The Reusable Holdout", Science 349;
    this is a deliberately stricter single-use workflow, with no privacy
    guarantee. All objects and processes sharing a journal also share the
    reservations. Inclusive overlaps are refused even when parameters change.
    The caller supplies the journal location and an evaluator that accesses
    the declared period. Keep the journal: this local guard cannot prevent
    deleting it, bypassing the object, or reading the underlying data directly.
    """

    def __init__(self, journal: str | Path, start: date | str, end: date | str) -> None:
        """Declare an interval; create its journal schema without consuming it.

        Follows the single-use holdout rationale in Dwork et al. (2015).
        The journal's parent directory must already exist. Inputs are dates
        or canonical ISO date strings, with start <= end. No data is loaded.
        """
        self._journal = Path(journal).resolve()
        self._start, self._end = _date(start), _date(end)
        if self._start > self._end:
            raise ValueError("start must not follow end")
        with closing(self._connect()) as connection, connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS evaluations ("
                "id INTEGER PRIMARY KEY, start TEXT NOT NULL, end TEXT NOT NULL, "
                "fingerprint TEXT NOT NULL, parameters_json TEXT NOT NULL, "
                "status TEXT NOT NULL, started_at TEXT NOT NULL, "
                "finished_at TEXT, error_type TEXT)"
            )

    def _connect(self) -> sqlite3.Connection:
        """Open the shared journal with a bounded transaction lock wait."""
        connection = sqlite3.connect(self._journal, timeout=10.0)
        connection.row_factory = sqlite3.Row
        return connection

    def evaluate(
        self, parameters: Mapping[str, JSONValue], evaluator: Callable[[Mapping[str, JSONValue]], T]
    ) -> T:
        """Reserve once, call evaluator with a parameter copy, and journal status.

        Single-use rationale: Dwork et al. (2015), "The Reusable Holdout".
        A committed SQLite write transaction reserves the inclusive interval
        before the callback can access data, serializing concurrent attempts.
        Subsequent overlapping requests raise HoldoutConsumedError before
        invoking their callback, regardless of parameters or prior success.
        Exceptions (including interrupted evaluations) consume the reservation
        and are re-raised; a process crash leaves a consumed running entry.
        Invalid parameters consume nothing. Callback results are returned
        unchanged and are not serialized into the journal. Evaluator-specific
        randomness should be supplied by the caller's callback or parameters.
        """
        if not callable(evaluator):
            raise ValueError("evaluator must be callable")
        encoded = _canonical_parameters(parameters)
        digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
        start, end = self._start.isoformat(), self._end.isoformat()
        with closing(self._connect()) as connection, connection:
            connection.execute("BEGIN IMMEDIATE")
            occupied = connection.execute(
                "SELECT 1 FROM evaluations WHERE start <= ? AND end >= ? LIMIT 1", (end, start)
            ).fetchone()
            if occupied is not None:
                raise HoldoutConsumedError("the holdout period has already been reserved")
            cursor = connection.execute(
                "INSERT INTO evaluations "
                "(start, end, fingerprint, parameters_json, status, started_at) "
                "VALUES (?, ?, ?, ?, 'running', ?)",
                (start, end, digest, encoded, datetime.now(timezone.utc).isoformat()),
            )
            identifier = cursor.lastrowid
        try:
            result = evaluator(json.loads(encoded))
        except BaseException as error:
            self._finish(identifier, "failed", type(error).__name__)
            raise
        self._finish(identifier, "completed", None)
        return result

    def _finish(self, identifier: int | None, status: str, error_type: str | None) -> None:
        """Record completion without clearing the original reservation."""
        with closing(self._connect()) as connection, connection:
            connection.execute(
                "UPDATE evaluations SET status = ?, finished_at = ?, error_type = ? WHERE id = ?",
                (status, datetime.now(timezone.utc).isoformat(), error_type, identifier),
            )

    def history(self) -> tuple[HoldoutRecord, ...]:
        """Read all journal reservations in creation order (Dwork et al., 2015).

        Returns immutable record objects, including canonical JSON parameters,
        fingerprints, status and UTC audit timestamps. Reading the journal
        never evaluates data or changes reservations.
        """
        with closing(self._connect()) as connection:
            rows = connection.execute("SELECT * FROM evaluations ORDER BY id").fetchall()
        return tuple(
            HoldoutRecord(
                date.fromisoformat(row["start"]), date.fromisoformat(row["end"]),
                row["fingerprint"], row["parameters_json"], row["status"],
                row["started_at"], row["finished_at"], row["error_type"],
            )
            for row in rows
        )
