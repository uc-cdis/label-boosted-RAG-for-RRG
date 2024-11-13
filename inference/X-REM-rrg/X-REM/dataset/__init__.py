import torch
from dataset.caption_dataset import pretrain_dataset, re_eval_dataset, re_train_dataset
from dataset.itm_dataset import itm_dataset
from dataset.randaugment import RandomAugment
from torch.utils.data import DataLoader
from torchvision import transforms


def create_dataset(dataset, config):

    normalize = transforms.Normalize(
        (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
    )

    pretrain_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            normalize,
        ]
    )
    train_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            normalize,
        ]
    )
    test_transform = transforms.Compose(
        [
            normalize,
        ]
    )

    if dataset == "pretrain":
        dataset = pretrain_dataset(config["train_file"], pretrain_transform)
        return dataset

    elif dataset == "re":
        train_dataset = re_train_dataset(
            config["train_file"], train_transform, config["image_root"]
        )
        val_dataset = re_eval_dataset(
            config["val_file"], test_transform, config["image_root"]
        )
        test_dataset = re_eval_dataset(
            config["test_file"], test_transform, config["image_root"]
        )
        return train_dataset, val_dataset, test_dataset

    elif dataset == "itm":
        train_dataset = itm_dataset(config["train_file"], train_transform)
        val_dataset = itm_dataset(config["val_file"], test_transform)
        test_dataset = itm_dataset(config["test_file"], test_transform)
        return train_dataset, val_dataset, test_dataset


def create_sampler(datasets, shuffles, num_tasks, global_rank):
    samplers = []
    for dataset, shuffle in zip(datasets, shuffles):
        sampler = torch.utils.data.DistributedSampler(
            dataset, num_replicas=num_tasks, rank=global_rank, shuffle=shuffle
        )
        samplers.append(sampler)
    return samplers


def create_loader(datasets, samplers, batch_size, num_workers, is_trains, collate_fns):
    loaders = []
    for dataset, sampler, bs, n_worker, is_train, collate_fn in zip(
        datasets, samplers, batch_size, num_workers, is_trains, collate_fns
    ):
        if is_train:
            shuffle = sampler is None
            drop_last = True
        else:
            shuffle = False
            drop_last = False
        loader = DataLoader(
            dataset,
            batch_size=bs,
            num_workers=n_worker,
            pin_memory=True,
            sampler=sampler,
            shuffle=shuffle,
            collate_fn=collate_fn,
            drop_last=drop_last,
        )
        loaders.append(loader)
    return loaders
