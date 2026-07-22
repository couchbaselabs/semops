""" 
Every operator run emits one record. Point `sink` at a Couchbase collection
(or a file) to accumulate the numbers that decide which operators graduate
in-engine. Default sink is an in-memory list for tests/demos.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


@dataclass
class Telemetry:
    records: list[dict[str, Any]] = field(default_factory=list)
    sink: Optional[Callable[[dict[str, Any]], None]] = None
    clock: Callable[[], float] = time.time

    def log(self, operator: str, engine: str, params: dict[str, Any], stats: dict[str, Any],
            latency_ms: float) -> dict[str, Any]:
        rec = {
            "ts": self.clock(),
            "operator": operator,
            "engine": engine,
            "params": params,
            "stats": stats,
            "latency_ms": round(latency_ms, 2),
        }
        self.records.append(rec)
        if self.sink is not None:
            self.sink(rec)
        return rec
