"""Single‑task data agent – inherits from :class:`BaseDataAgent`.

Implements the primary target creation logic for the original ``agent_data``
behaviour (classification via flat‑threshold or regression of the future
log‑return).
"""

from __future__ import annotations

from config.config import DataConfig

from .base_agent import BaseDataAgent


class SingleTaskDataAgent(BaseDataAgent):
    def __init__(self, cfg: DataConfig):
        super().__init__(cfg)

    def _create_primary_target(self, log_ret: float):
        if self.cfg.target_type == "classification":
            flat = self.cfg.flat_threshold
            if log_ret > flat:
                return 2  # up
            if log_ret < -flat:
                return 0  # down
            return 1  # flat
        # regression – raw log return
        return float(log_ret)
