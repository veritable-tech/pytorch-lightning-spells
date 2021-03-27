import numpy as np
from torch.utils.data.sampler import Sampler


class SortishSampler(Sampler):
    """Go through the text data by order of length with a bit of randomness.


    Returns an iterator that traverses the the data in randomly ordered batches that are approximately the same size.

    The data is first randomly shuffled and then put into a number of chunks. The data in each chunk is then sorted and
    sliced to get batches that are approximately the same size.

    The max key size batch is always returned in the first call because of pytorch cuda memory allocation sequencing.

    Without that max key returned first multiple buffers may be allocated when the first created isn't large enough
    to hold the next in the sequence.

    Taken from `Fast.ai <https://github.com/fastai/fastai1/blob/bcef12e95405655481bb309761f8c552b51b2bd2/fastai/text/data.py#L107>`_.

    Args:
        data_source (Iterable): The data you want to sample from.
        key (Callable): A function to get keys to sort. Input: the index number of the entry in data_source.
        bs (int): the batch size for the data loader
        chunk_size (int, optional): the number of batches one chunk contains. Defaults to 100.

    Example:
        >>> data_source = [[0], [0, 1]] * 100
        >>> sampler = SortishSampler(data_source, key=lambda idx: len(data_source[idx]), bs=2, chunk_size=2)
        >>> len(list(sampler))
        200
        >>> len(data_source[next(iter(sampler))]) # the largest/longest batch always goes first
        2
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
            # Sort inside the chunk (from longest to shortest)
            sort_idx = np.concatenate(
                [sorted(s, key=self.key, reverse=True) for s in ck_idx])
            sz = self.bs
            # Get batches
            ck_idx = [sort_idx[i:i+sz] for i in range(0, len(sort_idx), sz)]
            # find the batch with the largest key,
            max_ck = np.argmax([self.key(ck[0]) for ck in ck_idx])
            # then make sure it goes first.
            if len(ck_idx[max_ck]) != self.bs:
                # if not a full batch, reshuffle again
                continue
            # swap the batch
            ck_idx[0], ck_idx[max_ck] = ck_idx[max_ck], ck_idx[0]
            # Shuffle the list of batches
            sort_idx = np.concatenate(np.random.permutation([
                np.random.permutation(chunk.reshape(self.bs, -1)).reshape(-1)
                for chunk in ck_idx[1:-1]
            ]))
            sort_idx = np.concatenate((ck_idx[0], sort_idx, ck_idx[-1]))
            break
        return iter(sort_idx)


class SortSampler(Sampler):
    """Go through the text data by order of length (longest to shortest).

    Taken from `Fast.ai library <https://github.com/fastai/fastai1/blob/bcef12e95405655481bb309761f8c552b51b2bd2/fastai/text/data.py#L99>`_.

    Args:
        data_source (Iterable): The data you want to sample from.
        key (Callable): A function to get keys to sort. Input: the index number of the entry in data_source.

    Example:
        >>> data_source = [[0], [0, 1], [0, 1, 2, 3]]
        >>> sampler = SortSampler(data_source, key=lambda idx: len(data_source[idx]))
        >>> len(list(sampler))
        3
        >>> next(iter(sampler)) # the longest entry is the third one.
        2
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
