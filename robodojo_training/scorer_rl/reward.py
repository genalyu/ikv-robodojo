"""Frozen native DM05 action FM with transactional archive counterfactuals."""
from copy import copy, deepcopy
import torch
from opendm.model.dm05.qwen_archive import QwenArchiveBank
from opendm.model.dm05.persistent_kv import splice_dm05_kv, commit_dm05
from opendm.model.dm05.dm05_utils import VLADynamicCache


class SelectionBank(QwenArchiveBank):
    """320 physical history slots, independently specified logical selection."""
    def __init__(self, task, selected=None):
        super().__init__(320, None, task, protect_current=False)
        self.selected = selected

    def plan(self, *args, **kwargs):
        capacity = self.capacity
        self.capacity = 2**31  # collect every candidate; no learned scoring here
        try:
            plan = super().plan(*args, **kwargs)
        finally:
            self.capacity = capacity
        ids = plan.metadata["ids"]
        selected = ids[-capacity:] if self.selected is None else torch.as_tensor(self.selected).long().cpu()
        if len(selected) > capacity or len(selected.unique()) != len(selected) or not torch.isin(selected, ids).all():
            raise ValueError("Invalid explicit archive subset")
        take = torch.where(torch.isin(ids, selected))[0]
        plan.metadata = {k: v[take] for k, v in plan.metadata.items()}
        plan.source, plan.fresh = plan.source[take], plan.fresh[take]
        return plan


def fork_bank(bank, selected):
    result = copy(bank)
    result.selected = selected
    # Plans allocate new tensors; shared archived payloads are read-only.
    result.layers = dict(bank.layers)
    result.archive_layers = dict(bank.archive_layers)
    return result


class FrozenFMReward:
    def __init__(self, model):
        self.model = model.eval().requires_grad_(False)
        self.bank = None

    def reset(self, task):
        self.bank = SelectionBank(task)

    @torch.no_grad()
    @torch.autocast("cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available())
    def prepare(self, sample, patch_ids, times, *, seed):
        m = self.model
        cache, hidden = m._compute_prefix_cache(
            **{k: sample.get(k) for k in ("input_ids", "attention_mask", "pixel_values", "token_type_ids",
                                         "history_pixel_values", "history_mask", "history_features")},
            cache_cls=VLADynamicCache, full_cache=True)
        del hidden
        patch_ids = torch.as_tensor(patch_ids).cpu().long()
        times = torch.as_tensor(times).cpu().float()
        old = self.bank.archive_metadata
        candidate_ids = patch_ids if old is None else torch.cat((old["ids"][~torch.isin(old["ids"],patch_ids)],patch_ids))
        candidate_ids = candidate_ids.sort().values
        action = sample["action"]
        device = action.device
        devices = [device.index] if device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed)
            noise = torch.randn_like(action)
            time = torch.distributions.Beta(1.5,1.).sample((len(action),)).to(action) * .999 + .001
        return dict(sample=sample, cache=cache, ids=patch_ids, times=times,
                    candidates=candidate_ids, noise=noise, time=time)

    def _splice(self, prepared, selected):
        sample = prepared["sample"]
        bank = fork_bank(self.bank, selected)
        cache = deepcopy(prepared["cache"])
        request = dict(bank=bank, ids=prepared["ids"], descriptors=prepared["ids"]//16,
                       times=prepared["times"])
        ids, transaction = splice_dm05_kv(self.model, cache, sample["input_ids"],
                                         sample["attention_mask"],sample["history_mask"],request)
        return ids, cache, transaction

    @torch.no_grad()
    @torch.autocast("cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available())
    def evaluate(self, prepared, selected):
        ids, cache, transaction = self._splice(prepared, selected)
        sample = prepared["sample"]
        loss = self.model.flow_matching_loss_from_prefix(ids, cache, sample["action"],sample["action_mask"],
                                                        noise=prepared["noise"],time=prepared["time"])
        if not torch.isfinite(loss):raise FloatingPointError("Nonfinite DM05 FM reward")
        # Never commit a candidate to the canonical roll-in bank.
        return float(loss)

    @torch.no_grad()
    def commit_observation(self, prepared):
        _, _, transaction = self._splice(prepared, None)
        commit_dm05(transaction)
        self.bank = transaction[0]

    @torch.no_grad()
    @torch.autocast("cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available())
    def native_loss(self, prepared):
        sample = prepared["sample"]
        return float(self.model.flow_matching_loss_from_prefix(sample["input_ids"],deepcopy(prepared["cache"]),
                     sample["action"],sample["action_mask"],noise=prepared["noise"],time=prepared["time"]))
