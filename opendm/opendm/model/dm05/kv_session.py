"""Portable, episode-local DM05 inference with switchable historical KV updates."""
import torch
from .history_update import validate_update_mode
from .persistent_kv import PersistentKVBank
from .streaming_training import prepare_stream_step


class DM05KVSession:
    """One session per environment/episode; use a frozen eval-mode model.

    bank_factory must return a fresh empty bank, including a fresh scorer client
    when that client holds episode observations. Current RGB/text/state are still
    computed normally and are outside the historical KV budget.
    """
    def __init__(self, model, *, update_mode="window_refresh", bank_factory=None):
        self.model = model
        self.update_mode = update_mode
        self.bank_factory = bank_factory or (lambda: PersistentKVBank(model.config.ikv_history_capacity))
        self.reset()

    @property
    def update_mode(self):
        return self._update_mode

    @update_mode.setter
    def update_mode(self, value):
        self._update_mode = validate_update_mode(value)

    def reset(self):
        bank = self.bank_factory()
        if bank.generation != 0 or bank.metadata is not None or getattr(bank, "archive_metadata", None) is not None:
            raise ValueError("bank_factory must return a fresh empty episode bank")
        if bank.capacity != self.model.config.ikv_history_capacity:
            raise ValueError("bank capacity must match the model's reserved history slots")
        self.bank = bank

    @torch.no_grad()
    def inference_action(self, step, **inference_kwargs):
        if self.model.training:
            raise ValueError("DM05KVSession requires model.eval()")
        if "ikv_kv_request" in step or "ikv_kv_request" in inference_kwargs:
            raise ValueError("the session owns its KV request")
        data = dict(step)
        data.update(inference_kwargs)
        prepared = prepare_stream_step(self.model, data, self.bank, update_mode=self.update_mode)
        # Native inference commits the bank only after finite actions succeed.
        return self.model.inference_action(**prepared)
