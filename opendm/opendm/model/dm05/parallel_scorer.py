"""Local parallel-scorer bridge for QwenArchiveBank; no service or machine paths.

The caller supplies a frozen encoder and RuntimeScorer from genalyu/ikv, and
creates a fresh client/bank per episode. Scores are raw logits, not percentages.
"""
import math
import torch


class ParallelScorerClient:
    def __init__(self, scorer, encoder, task, *, sensor_id=0):
        if not task or sensor_id < 0:
            raise ValueError("Task and nonnegative sensor ID are required")
        self.scorer, self.encoder, self.task = scorer, encoder, task
        self.revision = scorer.info["encoder_revision"]
        if encoder.revision != self.revision:
            raise ValueError("Encoder/checkpoint revision mismatch")
        if getattr(scorer.model, "training", False) or encoder.training:
            raise ValueError("Scorer and encoder must be in eval mode")
        self.sensor_id = int(sensor_id)
        self.frames = {}

    def register(self, frame_id, time_seconds, features):
        """Register immutable context-independent [64,D] features of real RGB."""
        if type(frame_id) is not int or frame_id < 0:
            raise ValueError("Frame IDs must be nonnegative integers")
        time_seconds = float(time_seconds)
        value = torch.as_tensor(features).detach().cpu().clone()
        if not math.isfinite(time_seconds) or time_seconds < 0:
            raise ValueError("Invalid observation time")
        if value.ndim != 2 or value.shape[0] != 64 or not torch.isfinite(value).all():
            raise ValueError("Expected finite 8x8 patch features")
        if self.frames:
            width = next(iter(self.frames.values()))[1].shape[1]
            if value.shape[1] != width:
                raise ValueError("Feature dimension changed")
        if frame_id in self.frames:
            old_time, old_value = self.frames[frame_id]
            if old_time != time_seconds or not torch.equal(old_value, value):
                raise ValueError("Frame identity changed; reset for a new episode")
            return
        self.frames[frame_id] = (time_seconds, value)

    @torch.inference_mode()
    def append(self, frame_ids, images, times_seconds):
        """Times are explicit seconds; no control-step/FPS inference is performed."""
        if not len(frame_ids) == len(images) == len(times_seconds):
            raise ValueError("Unaligned images, IDs and times")
        if len(set(frame_ids)) != len(frame_ids):
            raise ValueError("Duplicate input frame IDs")
        fresh = []
        for i, fid in enumerate(frame_ids):
            if fid in self.frames:
                if self.frames[fid][0] != float(times_seconds[i]):
                    raise ValueError("Frame timestamp changed")
            else:
                fresh.append(i)
        if not fresh:
            return
        encoded = self.encoder.images([images[i] for i in fresh])
        if encoded.ndim != 2 or len(encoded) != len(fresh) * 64:
            raise ValueError("Encoder must emit 64 patches per image")
        for i, value in zip(fresh, encoded.reshape(len(fresh), 64, -1)):
            self.register(frame_ids[i], times_seconds[i], value)

    @torch.inference_mode()
    def score(self, candidates, current_time):
        """Map 8x8 logits to native head-camera 4x4 tokens by 2x2 mean.

        All registered causal frames provide context. Register current head RGB
        explicitly if desired; it is never automatically inferred from history.
        Future observations are excluded and cannot be candidates.
        """
        current_time = float(current_time)
        if not math.isfinite(current_time) or current_time < 0:
            raise ValueError("Invalid query time")
        if not candidates:
            return {}
        if len({c['id'] for c in candidates}) != len(candidates):
            raise ValueError("Duplicate candidate IDs")
        frame_ids = sorted(f for f, (t, _) in self.frames.items() if t <= current_time)
        lookup = {f: i for i, f in enumerate(frame_ids)}
        for c in candidates:
            if c['frame'] not in lookup:
                raise ValueError("Missing or future candidate frame")
            if tuple(c.get('grid', (4, 4))) != (4, 4) or c.get('modality', 'visual') != 'visual':
                raise ValueError("DM05 bridge supports head-camera 4x4 visual KV only")
            if type(c['patch']) is not int or not 0 <= c['patch'] < 16:
                raise ValueError("Invalid native patch index")
            if 'time' in c and not math.isclose(float(c['time']), self.frames[c['frame']][0], abs_tol=1e-6):
                raise ValueError("Candidate/feature time mismatch")
        n = len(frame_ids) * 64
        xy = torch.stack(torch.meshgrid(torch.arange(8), torch.arange(8), indexing='ij'), -1).reshape(-1, 2).float()/7
        packet = dict(features=torch.cat([self.frames[f][1] for f in frame_ids]),
                      text=self.encoder.task(self.task).detach().cpu(),
                      times=torch.tensor([self.frames[f][0] for f in frame_ids]).repeat_interleave(64),
                      xy=xy.repeat(len(frame_ids), 1), modalities=torch.zeros(n, dtype=torch.long),
                      sensors=torch.full((n,), self.sensor_id, dtype=torch.long),
                      query_time=current_time, encoder_revision=self.revision,
                      ids=[f'{f}:{p}' for f in frame_ids for p in range(64)])
        logits = torch.as_tensor(self.scorer.score(packet, {0: n, 1: 0})['scores'])
        if logits.shape != (n,) or not torch.isfinite(logits).all():
            raise ValueError("Invalid scorer output")
        native = logits.reshape(len(frame_ids), 4, 2, 4, 2).mean((2, 4)).reshape(-1, 16)
        return {c['id']: float(native[lookup[c['frame']], c['patch']]) for c in candidates}
