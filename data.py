"""
DISCLAIMER: 
This code was previously part of Joris Heemskerk's prior
work for the Computer Vision course, and is being re-used here.
"""

import logging

from torch.utils.data import Dataset, DataLoader
from typing import Any


def to_dataloaders(
    datasets: list[Dataset],
    batch_sizes: list[int],
    shuffles: list[bool],
    logger: logging.Logger,
    **kwargs: dict[str, Any],
)-> list[DataLoader]:
    """
    Convert list of Dataset objects into DataLoaders.

    :param datasets: Datasets to convert.
    :type datasets: list[Dataset]
    :param batch_sizes: Batch size for the dataloaders.
    :type batch_sizes: list[int]
    :param shuffles: Shuffle the dataset order if True.
    :type shuffles: list[bool]
    :param logger: Logger to log to.
    :type logger: logging.Logger
    :param **kwargs: Extra keyword arguments to pass to all dataloaders.
    :type **kwargs: dict
    :returns: List of converted datasets as DataLoader objects.
    :rtype: list[DataLoader]
    """
    dataLoaders = []
    assert len(datasets) == len(batch_sizes) == len(shuffles), \
        "One of the provided arguments has the wrong length: " \
        f"{len(datasets)=}, {len(batch_sizes)=}, {len(shuffles)=}"
    
    for dataset, batch_size, shuffle in zip(datasets, batch_sizes, shuffles):
        logger.debug(
            f"Converting dataset of {len(dataset)} elements into "
            f"DataLoader with {len(dataset) // batch_size} partitions of "
            f"size {batch_size}."
        )
        dataLoaders.append(
            DataLoader(
                dataset=dataset, 
                batch_size=batch_size, 
                shuffle=shuffle, 
                **kwargs
            )
        )
    return dataLoaders
