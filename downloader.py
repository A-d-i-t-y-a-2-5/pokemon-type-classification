from concurrent.futures import as_completed, ThreadPoolExecutor
from operator import le
import os
import time

import pandas as pd
import requests
from tqdm import tqdm

MAX_THREADS = 20


class Downloader:
    def __init__(self, filename: str, path: str = "./data/images"):
        self.dataframe = pd.read_csv(filename, converters={"Number": str})
        self.dl_url = "https://assets.pokemon.com/assets/cms2/img/pokedex/full/{}.png"
        self.path = path

        os.makedirs(self.path, exist_ok=True)

    def download_image(self, dex_no: str):
        url = self.dl_url.format(dex_no)
        myfile = requests.get(url, allow_redirects=True)
        if myfile.ok:
            open(f"{self.path}/{dex_no}.png", "wb").write(myfile.content)
        else:
            print(f"Unable to download for dex no. {dex_no}")

    def download(self, threads: bool = True):
        dex_nos = self.dataframe["Number"].values
        l = len(dex_nos)
        with tqdm(total=l) as pbar:
            if threads:
                with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
                    futures = (
                        executor.submit(self.download_image, d)
                        for d in dex_nos
                    )
                    for future in as_completed(futures):
                        future.result()
                        pbar.update(1)
            else:
                for d in dex_nos:
                    self.download_image(d)
                    pbar.update(1)
