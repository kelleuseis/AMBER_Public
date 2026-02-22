import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
import pytest
import torch.multiprocessing as mp

from dummy_data import create_dummy_dataset


def ddp_worker(rank, world_size, manager_dict, h5_path, csv_path):
    dist.init_process_group(
        backend="gloo", 
        init_method="tcp://127.0.0.1:29500", 
        world_size=world_size, rank=rank
    )

    dts = create_dummy_dataset(h5_path, csv_path)
    sampler = DistributedSampler(
        dts, num_replicas=world_size, rank=rank, shuffle=False
    )

    distidxs = list(sampler)
    sample_ids = [dts.sample_ids[i] for i in distidxs]

    manager_dict[rank] = sample_ids

    dist.destroy_process_group()


def test_ddp_sampler(dummy_hdf5, dummy_csv):
    manager = mp.Manager()
    manager_dict = manager.dict()

    mp.spawn(
        ddp_worker,
        args=(2, manager_dict, dummy_hdf5, dummy_csv),
        nprocs=2, join=True
    )

    rank0 = set(manager_dict[0])
    rank1 = set(manager_dict[1])

    comb_sample_ids = rank0.union(rank1)
    dts = create_dummy_dataset(dummy_hdf5, dummy_csv)
    
    assert comb_sample_ids == set(dts.sample_ids)