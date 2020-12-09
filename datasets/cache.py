import numpy as np
import h5py
from PIL import Image
import os


class HDF5Cache:
    # TODO: make completely modular
    # creates
    def __init__(self, dataset_name, cache_size):
        self.cache_size = cache_size  # number images per cache
        self.cache_handler = None  # hdf5 open
        self.cache_name = None
        self.dataset_name = dataset_name  # make sure name is unique
        # self.cache_idx = 0 # holds the current index of the cache
        self.transforms = None

    def generate_cached_name(self, idx):
        # generates the name for the cache image idx would be located in
        # Ex: div2k_6_16.hdf5 corresponds to the 7th batch of 16 images (96 - 111)
        return f"{self.dataset_name}_{idx // self.cache_size}_{self.cache_size}.hdf5"

    def load_cache(self, idx):
        if self.cache_handler != None:
            self.cache_handler.close()
        filename = "./datasets/cached/" + self.generate_cached_name(idx)
        self.cache_handler = h5py.File(filename, "r")

    def cache_images(self, num_cache, images, labels):
        filename = "./datasets/cached/" + self.generate_cached_name(
            num_cache * self.cache_size
        )
        with h5py.File(filename, "w") as f:
            f.create_dataset("lr", np.shape(images), data=images)
            f.create_dataset("hr", np.shape(labels), data=labels)

    def already_cached(self, idx):
        return self.generate_cached_name(idx) in os.listdir("./datasets/cached")

    def get_item(self, idx, dataset):
        # breaks on 768
        # gets the image with index idx
        if not self.already_cached(idx):
            # self.cache_image(idx)
            dataset.cache_func(idx // self.cache_size)
        filename = "./datasets/cached/" + self.generate_cached_name(idx)
        # print(self.cache_handler)
        if self.cache_handler == None or self.cache_handler.filename != filename:
            self.load_cache(idx)
        return (
            self.cache_handler["lr"][idx % self.cache_size],
            self.cache_handler["hr"][idx % self.cache_size],
        )
