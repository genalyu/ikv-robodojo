"""Episode-safe chronological windows for recurrent PI05 training."""
import bisect
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import jax

class EpisodeSequenceDataset:
    def __init__(self, dataset, metadata_path: str | Path, length: int = 4):
        self.dataset = dataset
        self.length = int(length)
        if self.length < 1:
            raise ValueError('sequence length must be positive')
        table = pq.read_table(metadata_path, columns=['episode_index','dataset_from_index','dataset_to_index'])
        rows = table.to_pylist()
        starts, ends = [], []
        sequence_count = 0
        expected = 0
        for row in rows:
            lo, hi = int(row['dataset_from_index']), int(row['dataset_to_index'])
            if lo != expected or hi <= lo:
                raise ValueError('episode offsets must be contiguous and nonempty')
            starts.append((lo,hi))
            sequence_count += (hi-lo+self.length-1)//self.length
            ends.append(sequence_count)
            expected = hi
        if expected != len(dataset):
            raise ValueError(f'episode metadata frames {expected} != dataset {len(dataset)}')
        self.episodes, self.ends = starts, ends
    def __len__(self):
        return self.ends[-1]
    def __getitem__(self,index):
        index = int(index)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        episode = bisect.bisect_right(self.ends,index)
        lo,hi = self.episodes[episode]
        previous = self.ends[episode-1] if episode else 0
        first = lo+(index-previous)*self.length
        indices = [min(first+t,hi-1) for t in range(self.length)]
        valid = np.asarray([first+t<hi for t in range(self.length)],dtype=np.bool_)
        samples = [self.dataset[i] for i in indices]
        stacked = jax.tree.map(lambda *xs: np.stack([np.asarray(x) for x in xs],axis=0),*samples)
        stacked['ikv_valid'] = valid
        return stacked
