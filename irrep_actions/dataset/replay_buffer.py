import copy
import torch
import random
import h5py
import numpy as np
import numpy.random as npr
from random import sample

from functools import cached_property


class ReplayBuffer(object):
    def __init__(self):
        super().__init__()

        self._storage = {
            "data": dict(),
            "meta": {"episode_ends": np.zeros((0,), dtype=np.int64)},
        }

    @cached_property
    def data(self):
        return self._storage["data"]

    @cached_property
    def meta(self):
        return self._storage["meta"]

    @property
    def episode_ends(self):
        return self.meta["episode_ends"]

    def getEpisodeIdxs(self):
        result = np.zeros((self.episode_ends[-1],), dtype=np.int64)
        for i in range(len(self.episode_ends)):
            start = episode_ends[i - 1] if i > 0 else 0
            end = episode_ends[i]
            for (idx,) in range(start, end):
                result[idx] = i
        return result

    def keys(self):
        return self.data.keys()

    def values(self):
        return self.data.values()

    def items(self):
        return self.data.items()

    def __getitem__(self, key):
        return self.data[key]

    def __contains__(self, key):
        return key in self.data

    @property
    def n_steps(self):
        return 0 if len(self.episode_ends) == 0 else self.episode_ends[-1]

    @property
    def n_episodes(self):
        return len(self.episode_ends)

    @property
    def episode_lengths(self):
        ends = self.episode_ends[:]
        ends = np.insert(ends, 0, 0)
        lengths = np.diff(ends)
        return lengths

    def addEpisode(self, data):
        assert len(data) > 0
        curr_len = self.n_steps

        # Get episode length and ensure each data is the same length
        episode_length = None
        for key, value in data.items():
            assert len(value.shape) >= 1
            if episode_length is None:
                episode_length = len(value)
            else:
                assert episode_length == len(value)
        new_len = curr_len + episode_length

        for key, value in data.items():
            new_shape = (new_len,) + value.shape[1:]
            if key not in self.data:
                arr = np.zeros(shape=new_shape, dtype=value.dtype)
                self.data[key] = arr
            else:
                arr = self.data[key]
                assert value.shape[1:] == arr.shape[1:]
                arr.resize(new_shape, refcheck=False)
            arr[-value.shape[0] :] = value

        episode_ends = self.episode_ends
        episode_ends.resize(episode_ends.shape[0] + 1, refcheck=False)
        episode_ends[-1] = new_len

    def extend(self, data):
        self.add_episode(data)

    def getEpisode(self, idx, copy=False):
        idx = list(range(len(self.episode_ends)))[idx]
        start_idx = self.episode_ends[idx - 1] if idx > 0 else 0
        end_idx = self.episode_ends[idx]
        result = self.getStepsSlice(start_idx, end_idx, copy=copy)
        return result

    def getEpisodeSlice(self, idx):
        start_idx = self.episode_ends[idx - 1] if idx > 0 else 0
        end_idx = self.episode_ends[idx]
        return slice(start_idx, end_idx)

    def getStepsSlice(self, start, stop, step=None, copy=False):
        _slize = slice(start, stop, step)

        result = dict()
        for key, value in self.data.items():
            x = value[_slice]
            if copy and isinstance(value, np.ndarray):
                x = x.copy()
            result[key] = x
        return result

    def load_from_path(self, path):
        # Load meta data
        with h5py.File(f"{path}/meta.hdf5", "r") as f:
            for key in f.keys():
                self.meta[key] = f.get(key)[:]

        # Load data
        with h5py.File(f"{path}/data.hdf5", "r") as f:
            for key in f.keys():
                self.data[key] = f.get(key)[:]

    def save_to_path(self, path):
        # Save meta data
        with h5py.File(f"{path}/meta.hdf5", "w") as f:
            for key, value in self.meta.items():
                f.create_dataset(key, data=value)

        # Save data
        with h5py.File(f"{path}/data.hdf5", "w") as f:
            for key, value in self.data.items():
                f.create_dataset(key, data=value)
