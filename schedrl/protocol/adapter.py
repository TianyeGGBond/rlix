from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from schedrl.protocol.types import ActionResponse


class Adapter(ABC):
    @abstractmethod
    def resize_infer(self, dp_ranks_to_remove: List[int], dp_ranks_to_add: List[int]) -> ActionResponse:
        raise NotImplementedError
