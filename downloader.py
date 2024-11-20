from concurrent.futures import as_completed, ThreadPoolExecutor
from operator import le
from itertools import accumulate
import os
import time

import pandas as pd
import requests
from tqdm import tqdm

MAX_THREADS = 20


class Downloader:
    def __init__(self, filename: str, path: str = "./data/images", mode: str = "train"):
        self.dataframe = pd.read_csv(filename, converters={"Number": str})
        self.dl_url = "https://assets.pokemon.com/assets/cms2/img/pokedex/full/{}.png"
        self.mode = mode
        self.path = path + "/" + self.mode

        os.makedirs(self.path, exist_ok=True)

    def download_image(self, dex_no: str, class_type: str):
        url = self.dl_url.format(dex_no)
        myfile = requests.get(url, allow_redirects=True)
        if myfile.ok:
            os.makedirs(f"{self.path}/{class_type.lower()}", exist_ok=True)
            open(f"{self.path}/{class_type.lower()}/{dex_no}.png", "wb").write(
                myfile.content
            )
        else:
            print(f"Unable to download for dex no. {dex_no}")

    def modify_data(self):
        # Create incremental counter for every duplicate Pokedex Number
        duplicate_counts = list(
            accumulate(
                self.dataframe["Number"].duplicated().astype(int).to_list(),
                lambda x, y: (x + y) * y,
            )
        )
        self.dataframe["Form No"] = duplicate_counts
        # Increment the counter by 1 for non zero values
        # Required for downloading form images
        self.dataframe["Form No"] = self.dataframe["Form No"] + (
            self.dataframe["Form No"] != 0
        )

        self.dataframe = self.dataframe[self.dataframe["Number"].duplicated()].copy()
        self.dataframe["Number"] = (
            self.dataframe["Number"] + "_f" + self.dataframe["Form No"].astype(str)
        )

    def download(self, threads: bool = True):
        if self.mode == "test":
            self.modify_data()
        dex_primary_type = self.dataframe[["Number", "Type 1"]].values.tolist()
        l = len(dex_primary_type)
        with tqdm(total=l) as pbar:
            if threads:
                with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
                    futures = (
                        executor.submit(self.download_image, d, t)
                        for d, t in dex_primary_type
                    )
                    for future in as_completed(futures):
                        future.result()
                        pbar.update(1)
            else:
                for d, t in dex_primary_type:
                    self.download_image(d, t)
                    pbar.update(1)
