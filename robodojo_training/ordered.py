"""Chronological microbatches while Accelerate shards every world_size batches."""
import random

class InterleavedChunkSampler:
    def __init__(self, dataset, chunk_size, world_size, seed=0):
        if chunk_size < 1 or world_size < 1:
            raise ValueError('positive chunk_size and world_size required')
        self.dataset = dataset
        self.chunk_size = int(chunk_size)
        self.world_size = int(world_size)
        self.seed = int(seed)
        self.epoch = 0
        self.nchunks = len(dataset) // self.chunk_size
        self.nchunks -= self.nchunks % self.world_size
    def set_epoch(self, epoch):
        self.epoch = int(epoch)
    def __len__(self):
        return self.nchunks * self.chunk_size
    def __iter__(self):
        chunks = list(range(len(self.dataset) // self.chunk_size))
        random.Random(self.seed + self.epoch).shuffle(chunks)
        chunks = chunks[:self.nchunks]
        for start in range(0, len(chunks), self.world_size):
            group = chunks[start:start+self.world_size]
            for frame in range(self.chunk_size):
                for chunk in group:
                    yield chunk*self.chunk_size + frame
