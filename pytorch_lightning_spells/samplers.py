import numpy as np
from torch.utils.data.sampler import Sampler


class SortishSampler(Sampler):
    """Returns an iterator that traverses the the data in randomly ordered batches that are approximately the same size.
    The max key size batch is always returned in the first call because of pytorch cuda memory allocation sequencing.
    Without that max key returned first multiple buffers may be allocated when the first created isn't large enough
    to hold the next in the sequence.

    Taken from fastai library.
    """

    def __init__(self, data_source, key, bs, chunk_size=100):
        self.data_source, self.key, self.bs = data_source, key, bs
        self.chunk_size = chunk_size

    def __len__(self): return len(self.data_source)

    def __iter__(self):
        while True:
            idxs = np.random.permutation(len(self.data_source))
            sz = self.bs * self.chunk_size
            ck_idx = [idxs[i:i+sz] for i in range(0, len(idxs), sz)]
            sort_idx = np.concatenate(
                [sorted(s, key=self.key, reverse=True) for s in ck_idx])
            sz = self.bs
            ck_idx = [sort_idx[i:i+sz]for i in range(0, len(sort_idx), sz)]
            # find the chunk with the largest key,
            max_ck = np.argmax([self.key(ck[0]) for ck in ck_idx])
            # then make sure it goes first.
            if len(ck_idx[max_ck]) != self.bs:
                continue
            ck_idx[0], ck_idx[max_ck] = ck_idx[max_ck], ck_idx[0]
            sort_idx = np.concatenate(np.random.permutation([
                np.random.permutation(chunk.reshape(self.bs, -1)).reshape(-1)
                for chunk in ck_idx[1:-1]
            ]))
            sort_idx = np.concatenate((ck_idx[0], sort_idx, ck_idx[-1]))
            break
        return iter(sort_idx)


class SortSampler(Sampler):
    """
    Taken from fastai library.
    """

    def __init__(self, data_source, key):
        self.data_source, self.key = data_source, key

    def __len__(self):
        return len(self.data_source)

    def __iter__(self):
        return iter(sorted(
            range(len(self.data_source)),
            key=self.key, reverse=True
        ))
