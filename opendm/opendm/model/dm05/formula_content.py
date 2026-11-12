from array import array
import torch
import torch.nn.functional as F
class ContentHistory:
    """Bounded content prototypes; observation support, never cache residence."""
    def __init__(self, config, device):
        self.config, self.device = config, device
        self.features = torch.empty(0, 0, device=device)
        self.duration = torch.empty(0, device=device)
        self.last_time = torch.empty(0, device=device)
        self.clock = None
        self.decision_cache = None
        self.cache_hits = 0

    def snapshot(self):
        return (self.features.clone(), self.duration.clone(), self.last_time.clone(), self.clock)

    def restore(self, state):
        self.features, self.duration, self.last_time, self.clock = state

    @torch.no_grad()
    def observe(self, features, times, durations):
        if not features.shape[-1]:
            return
        if not (torch.isfinite(features).all() & torch.isfinite(times).all()):
            raise ValueError("content observations must be finite")
        if self.features.shape[-1] not in (0, features.shape[-1]):
            raise ValueError("DINO width changed within an episode")
        for t in torch.unique(times, sorted=True).tolist():
            if self.clock is not None and t <= self.clock:
                continue  # duplicate commit / grounded overlap
            frame = times == t
            values = F.normalize(features[frame].float(), dim=-1)
            spans = durations[frame].float()
            present = values.abs().sum(-1) > 0
            values, spans = values[present], spans[present]
            # Greedy deterministic matching is position independent. Prototypes
            # are fixed to avoid chaining distinct objects through centroid drift.
            seen = {}
            cached = (None if self.decision_cache is None
                      else self.decision_cache.get(float(t)))
            if cached is not None:
                if len(cached) != len(values):
                    raise ValueError("cached content decisions do not match this frame")
                self.cache_hits += 1
                for feature, span, code in zip(values, spans.tolist(), cached):
                    if code < 0:
                        continue
                    best, action = code >> 2, code & 3
                    if action == 1:  # append a new prototype
                        if best != len(self.features):
                            raise ValueError("cached prototype append order changed")
                        if not self.features.numel():
                            self.features = feature[None].clone()
                        else:
                            self.features = torch.cat((self.features, feature[None]))
                        self.duration = torch.cat((self.duration, spans.new_zeros(1)))
                        self.last_time = torch.cat((self.last_time, spans.new_tensor([t])))
                    elif action == 2:  # replace an evicted prototype
                        if best >= len(self.features):
                            raise ValueError("cached prototype replacement is out of range")
                        self.features[best] = feature
                        self.duration[best] = 0
                        self.last_time[best] = t
                    elif action != 0 or best >= len(self.features):
                        raise ValueError("invalid cached content decision")
                    seen[best] = max(seen.get(best, 0.), float(span))
            else:
                # The first visit records the exact greedy decisions. Subsequent
                # visits to this trajectory replay them without any similarity
                # kernels or per-feature device synchronizations.
                decisions = array("i") if self.decision_cache is not None else None
                last_times = self.last_time.tolist()
                for feature, span in zip(values, spans.tolist()):
                    similarities = self.features @ feature if self.features.numel() else feature.new_empty(0)
                    scores = similarities.cpu()
                    best = int(scores.argmax()) if len(scores) else -1
                    action = 0
                    if best < 0 or float(scores[best]) < self.config.content_threshold:
                        if len(self.features) >= self.config.content_capacity:
                            candidates = [i for i in range(len(self.features)) if i not in seen]
                            if not candidates:
                                if decisions is not None:
                                    decisions.append(-1)
                                continue
                            best = min(candidates, key=last_times.__getitem__)
                            last_times[best] = t
                            self.features[best] = feature
                            self.duration[best] = 0
                            self.last_time[best] = t
                            action = 2
                        else:
                            best = len(self.features)
                            last_times.append(t)
                            if not self.features.numel():
                                self.features = feature[None].clone()
                            else:
                                self.features = torch.cat((self.features, feature[None]))
                            self.duration = torch.cat((self.duration, spans.new_zeros(1)))
                            self.last_time = torch.cat((self.last_time, spans.new_tensor([t])))
                            action = 1
                    if decisions is not None:
                        decisions.append((best << 2) | action)
                    seen[best] = max(seen.get(best, 0.), float(span))
                if decisions is not None:
                    self.decision_cache[float(t)] = decisions
            if len(seen) > 1:
                # Each slot appears once: batched updates preserve one addition
                # per duration while avoiding several launches per prototype.
                ids = torch.tensor(list(seen), dtype=torch.long, device=self.device)
                increments = self.duration.new_tensor(list(seen.values()))
                self.duration.index_add_(0, ids, increments)
                self.last_time.index_fill_(0, ids, t)
            else:
                for group, span in seen.items():
                    self.duration[group] += span
                    self.last_time[group] = t
            self.clock = t

    def persistence(self, features):
        result = features.new_zeros(len(features))
        if not features.shape[-1] or not self.features.numel():
            return result
        for start in range(0, len(features), 256):
            block = features[start:start+256]
            similarity, group = (F.normalize(block.float(), dim=-1) @ self.features.T).max(-1)
            support = 1 - torch.exp(-self.duration[group] / self.config.persistence_scale)
            result[start:start+256] = support * (similarity >= self.config.content_threshold) * block.ne(0).any(-1)
        return result

    def class_recency(self, features, times):
        """Age a visual token against its class's last real observation."""
        result = torch.zeros(len(features), device=features.device, dtype=torch.float32)
        if not features.shape[-1] or not self.features.numel():
            return result
        if features.shape[-1] != self.features.shape[-1] or len(times) != len(features):
            raise ValueError("class recency requires aligned DINO features and times")
        for start in range(0, len(features), 256):
            block = features[start:start+256]
            similarity, group = (F.normalize(block.float(), dim=-1) @ self.features.T).max(-1)
            age = (self.last_time[group] - times[start:start+256]).clamp_min(0)
            value = torch.exp(-age / self.config.class_recency_scale)
            result[start:start+256] = value * (similarity >= self.config.content_threshold) * block.ne(0).any(-1)
        return result
