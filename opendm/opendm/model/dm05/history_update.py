"""Choose refreshed history rows before visual encoding; never change KV ranking."""
import torch

UPDATE_MODES = ("window_refresh", "incremental")


def validate_update_mode(mode):
    if mode not in UPDATE_MODES:
        raise ValueError(f"Unknown KV update mode {mode!r}; choose {UPDATE_MODES}")
    return mode


def committed_watermark(bank):
    """Read only committed state, so failed preparations/forwards remain retryable."""
    watermark = int(getattr(bank, "watermark", -1))
    metadata = getattr(bank, "archive_metadata", None)
    if metadata is not None and len(metadata["ids"]):
        watermark = max(watermark, int(metadata["ids"].max()))
    return watermark


def select_history_update(step, bank, mode="window_refresh"):
    """Filter frame/patch-aligned tensors before computing features or prefix KV.

    IDs are episode frame_id*16+patch_id, never window-relative. The input can be
    the same bounded window for either mode. Incremental accepts every unseen
    frame in that window (including a backlog), not just its final frame.
    """
    validate_update_mode(mode)
    data = dict(step)
    ids = torch.as_tensor(data["history_patch_ids"])
    if ids.ndim != 1 or ids.dtype not in (torch.int32, torch.int64):
        raise ValueError("history_patch_ids must be a one-dimensional integer tensor")
    if len(ids) % 16 or len(ids) > bank.capacity:
        raise ValueError("history must contain complete 16-patch frames within the KV capacity")
    groups = ids.reshape(-1, 16)
    if len(groups):
        expected = (groups[:, :1] // 16) * 16 + torch.arange(16, device=ids.device)
        if (ids < 0).any() or not torch.equal(groups, expected) or (groups[1:, 0] <= groups[:-1, 0]).any():
            raise ValueError("history IDs must be complete frames in strictly increasing episode order")
    times = torch.as_tensor(data["history_times"])
    if times.shape != ids.shape or not torch.isfinite(times).all() or (times < 0).any():
        raise ValueError("history_times must be finite nonnegative seconds per patch")
    ft = times.reshape(-1, 16)
    if len(ft) and ((ft != ft[:, :1]).any() or (ft[1:, 0] <= ft[:-1, 0]).any()):
        raise ValueError("history timestamps must agree within frames and increase across frames")
    mark = committed_watermark(bank)
    if len(ids) and int(ids.max()) < mark:
        raise ValueError("stream steps must be chronological within one episode")
    frames = torch.arange(len(groups), device=ids.device)
    if mode == "incremental":
        if mark >= 0 and mark % 16 != 15:
            raise ValueError("incremental updates require a complete-frame committed watermark")
        frames = frames[groups[:, 0] > mark]
    patches = (frames[:, None] * 16 + torch.arange(16, device=ids.device)).flatten()
    # Motion is measured on the supplied observation sequence before subsetting.
    if data.get("history_motion") is None and data.get("history_rgb_values") is not None and len(ids):
        from .ikv_rgb import history_motion_scores
        data["history_motion"] = history_motion_scores(data["history_rgb_values"]).flatten()
    for name in ("history_patch_ids", "history_times", "history_descriptors", "history_features", "history_motion", "history_contact"):
        value = data.get(name)
        if value is not None:
            value = torch.as_tensor(value)
            if value.ndim == 0 or len(value) != len(ids):
                raise ValueError(f"{name} must have one row per history patch")
            data[name] = value.index_select(0, patches.to(value.device))
    for name in ("history_pixel_values", "history_rgb_values"):
        value = data.get(name)
        if value is not None:
            if len(value) != len(groups):
                raise ValueError(f"{name} must have one row per history frame")
            data[name] = value.index_select(0, frames.to(value.device))
    return data
