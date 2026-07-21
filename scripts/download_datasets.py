#!/usr/bin/env python
from __future__ import annotations

import argparse
import gzip
import shutil
import tarfile
import zipfile
from pathlib import Path
from urllib.request import urlretrieve


DATASET_URLS = {
    "icews14": "https://raw.githubusercontent.com/soledad921/ATiSE/master/data/ICEWS14/train.txt",
}


def download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"exists {dest}")
        return
    print(f"downloading {url}")
    urlretrieve(url, dest)


def unpack(path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as zf:
            zf.extractall(out_dir)
    elif path.suffixes[-2:] == [".tar", ".gz"] or path.suffix == ".tgz":
        with tarfile.open(path) as tf:
            tf.extractall(out_dir)
    elif path.suffix == ".gz":
        with gzip.open(path, "rb") as src, open(out_dir / path.stem, "wb") as dst:
            shutil.copyfileobj(src, dst)
    else:
        shutil.copy2(path, out_dir / path.name)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Small dataset fetch helper. For publication-scale runs, prefer official dataset "
            "loaders and record exact URL/version in the experiment config."
        )
    )
    parser.add_argument("name", choices=sorted(DATASET_URLS))
    parser.add_argument("--out", default="data/raw")
    args = parser.parse_args()
    url = DATASET_URLS[args.name]
    dest = Path(args.out) / args.name / Path(url).name
    download_file(url, dest)
    print(dest)


if __name__ == "__main__":
    main()
