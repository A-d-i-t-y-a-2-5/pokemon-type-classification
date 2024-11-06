from downloader import Downloader
from training import Trainer

if __name__ == "__main__":
    # dl = Downloader("pokedex.csv")
    # dl.download(threads=True)
    t = Trainer("pokedex.csv")
    t.train()