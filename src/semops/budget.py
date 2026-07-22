"""Cost governance. A semantic query can cost real money (the KB cites a
10k x 10k join ~= $600k); every operator charges the guard before spending,
so a runaway query stops instead of draining a budget."""
from __future__ import annotations

import threading


class BudgetExceeded(RuntimeError):
    pass


class Budget:
    def __init__(self, limit_usd: float | None = None):
        self.limit_usd = limit_usd
        self.spent_usd = 0.0
        self._lock = threading.Lock()  # charged from worker threads

    def charge(self, usd: float) -> None:
        with self._lock:
            self.spent_usd += usd
            over = self.limit_usd is not None and self.spent_usd > self.limit_usd
            spent = self.spent_usd
        if over:
            raise BudgetExceeded(
                f"budget exceeded: spent ${spent:.4f} > limit ${self.limit_usd:.4f}"
            )

    def would_exceed(self, usd: float) -> bool:
        return self.limit_usd is not None and (self.spent_usd + usd) > self.limit_usd

    def remaining(self) -> float:
        return float("inf") if self.limit_usd is None else max(0.0, self.limit_usd - self.spent_usd)
