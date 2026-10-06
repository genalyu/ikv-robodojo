"""Bounded, irreversible online patch memory shared by inference and training."""
from __future__ import annotations
import torch
import torch.nn.functional as F
from opendm.model.dm05.ikv_rgb import history_motion_scores


class OnlineHistoryMemory:
    """Only survivors and one previous RGB frame persist; evicted patches cannot return."""
    def __init__(self, capacity=320, motion_only=False, motion_threshold=0.04):
        if capacity < 1:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self.motion_only = motion_only
        self.motion_threshold = motion_threshold
        self.features = self.dino = self.times = self.latest = None
        self.contact = self.motion = None
        self.previous_rgb = None
        self.last_time = None
        self.updates = 0

    @torch.no_grad()
    def update(self, features, dino, timestamp, rgb, contact_duration=None):
        if self.last_time is not None and timestamp <= self.last_time:
            raise ValueError("history timestamps must strictly increase")
        if features.ndim != 2 or len(features) != 16 or dino.ndim != 2 or len(dino) != 16:
            raise ValueError("one update requires 16 aligned visual and DINO patches")
        if not torch.isfinite(features).all() or not torch.isfinite(dino).all():
            raise ValueError("nonfinite history features")
        device = features.device
        dino = F.normalize(dino.to(device=device, dtype=torch.float32), dim=-1)
        time = torch.full((16,), float(timestamp), device=device)
        contact = torch.zeros_like(time) if contact_duration is None else torch.as_tensor(
            contact_duration, device=device, dtype=torch.float32).reshape(16)
        if not torch.isfinite(contact).all() or (contact < 0).any():
            raise ValueError("invalid contact duration")
        rgb = rgb.detach().cpu()
        if self.previous_rgb is None:
            motion = torch.ones_like(time)
        else:
            motion = history_motion_scores(torch.cat((self.previous_rgb, rgb)))[-1].to(device)
        values = [features.detach(), dino, time, time.clone(), contact, motion]
        names = ("features", "dino", "times", "latest", "contact", "motion")
        if self.features is not None:
            values = [torch.cat((getattr(self, name), value)) for name, value in zip(names, values)]
        features, descriptors, times, latest, contact, motion = values
        # Update class recency using newly observed real patches; no evicted index is kept.
        valid = descriptors.ne(0).any(-1)
        matches = (descriptors @ dino.T >= 0.9) & valid[:, None] & dino.ne(0).any(-1)[None, :]
        latest = torch.where(matches.any(1), torch.maximum(latest, time[0]), latest)
        score = (torch.exp(-(float(timestamp) - times).clamp_min(0) / 8)
                 + 1 - torch.exp(-contact / 8)
                 + valid * torch.exp(-(latest - times).clamp_min(0) / 8))
        selected = torch.argsort(score, descending=True, stable=True)[:self.capacity]
        selected = selected.sort().values  # temporal/spatial order, not score order
        for name, value in zip(names, (features, descriptors, times, latest, contact, motion)):
            setattr(self, name, value.index_select(0, selected).clone())
        self.previous_rgb = rgb.clone()
        self.last_time = float(timestamp)
        self.updates += 1

    def read(self):
        if self.features is None:
            return None
        eligible = self.motion > self.motion_threshold if self.motion_only else torch.ones_like(self.motion, dtype=torch.bool)
        return self.features[eligible]


def encode_policy_history(model, pixels):
    """Differentiable visual encoding. Inference callers use no_grad."""
    vlm = model.model.vlm.model
    pixels = pixels.to(device=next(vlm.vision_tower.parameters()).device,
                       dtype=next(vlm.vision_tower.parameters()).dtype)
    pieces = []
    for batch in pixels.split(8):
        features = vlm.get_image_features(batch, return_dict=True).pooler_output
        spatial = int(features.shape[1] ** 0.5)
        grid = features.reshape(-1, spatial, spatial, features.shape[-1]).permute(0, 3, 1, 2)
        pieces.append(F.adaptive_avg_pool2d(grid, (4, 4)).permute(0, 2, 3, 1).reshape(len(batch), 16, -1))
    return torch.cat(pieces)


def replay_history_indices(rgb, dino, capacity=320, motion_only=False, motion_threshold=0.04):
    """Replay each observed frame once; payloads are indices, never detached trainable features."""
    bank = OnlineHistoryMemory(capacity, motion_only, motion_threshold)
    for t in range(len(rgb)):
        indices = torch.arange(t * 16, (t + 1) * 16, device=dino.device).reshape(16, 1)
        bank.update(indices, dino[t], t, rgb[t:t + 1])
    result = bank.read()
    return torch.empty(0, dtype=torch.long, device=dino.device) if result is None else result[:, 0].long()


def pack_history_features(input_ids, token_type_ids, features_by_sample):
    """Install arbitrary surviving patches in fixed history slots; no frame alignment assumption."""
    from opendm.model.dm05.dm05_utils import HISTORY_PAD_TOKEN_ID
    ids = input_ids.clone()
    types = token_type_ids.clone()
    mask = torch.zeros_like(ids, dtype=torch.bool)
    for batch, features in enumerate(features_by_sample):
        slots = ((ids[batch] == 6) | (ids[batch] == HISTORY_PAD_TOKEN_ID)).nonzero().flatten()
        count = len(features)
        if count > len(slots):
            raise ValueError("retained history exceeds available prefix slots")
        ids[batch, slots] = HISTORY_PAD_TOKEN_ID
        types[batch, slots] = 0
        if count:
            chosen = slots[-count:]
            ids[batch, chosen] = 6
            types[batch, chosen] = 1
            mask[batch, chosen] = True
    return ids, types, mask, torch.cat(features_by_sample, dim=0)


def prepare_training_memory(model, pixels, rgb, counts):
    """Replay per sample, then encode only frames containing survivors with gradients."""
    from opendm.model.dm05.ikv_dino import encode_dino_grid
    config = model.config
    results = []
    offset = 0
    for count in counts.tolist():
        count = int(count)
        if count == 0:
            results.append(next(model.parameters()).new_empty((0, model.model.language_model.config.hidden_size)))
            continue
        sample_rgb = rgb[offset:offset + count]
        dino = encode_dino_grid(sample_rgb, checkpoint=config.ikv_dino_model_path,
                                grid=(4, 4), device=sample_rgb.device).flatten(1, 2)
        indices = replay_history_indices(sample_rgb, dino, config.ikv_history_capacity,
                                         config.ikv_motion_only, config.ikv_motion_threshold)
        frames, inverse = torch.unique(indices // 16, sorted=True, return_inverse=True)
        if len(indices):
            encoded = encode_policy_history(model, pixels[offset:offset + count].index_select(0, frames.to(pixels.device)))
            results.append(encoded[inverse.to(encoded.device), (indices % 16).to(encoded.device)])
        else:
            results.append(next(model.parameters()).new_empty((0, model.model.language_model.config.hidden_size)))
        offset += count
    if pixels is not None and offset != len(pixels):
        raise ValueError("history counts do not match image batch")
    return results
