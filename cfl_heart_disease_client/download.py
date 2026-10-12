"""Fetch the four UCI Heart Disease files FLamby uses (about 40 KB in total).

FLamby's own script (fed_heart_disease/dataset_creation_scripts/download.py) pulls the
same four files from the UCI Machine Learning Repository:

    https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/
        processed.cleveland.data      (Cleveland Clinic Foundation)
        processed.hungarian.data      (Hungarian Institute of Cardiology, Budapest)
        processed.switzerland.data    (University Hospitals, Zurich and Basel)
        processed.va.data             (V.A. Medical Center, Long Beach)

Citation: Janosi, Steinbrunn, Pfisterer & Detrano, Heart Disease,
UCI Machine Learning Repository (1989). https://doi.org/10.24432/C52P4X

    python -m cfl_heart_disease_client.download            # -> heart/
    python -m cfl_heart_disease_client.download --out heart
"""
import argparse
import os
import urllib.request

BASE = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/"
FILES = ["processed.cleveland.data", "processed.hungarian.data",
         "processed.switzerland.data", "processed.va.data"]
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=os.path.join(ROOT, "heart"))
    a = p.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    for name in FILES:
        dst = os.path.join(a.out, name)
        if os.path.exists(dst):
            print("have   {} ({} bytes)".format(name, os.path.getsize(dst)))
            continue
        print("get    {} ...".format(name), end=" ", flush=True)
        urllib.request.urlretrieve(BASE + name, dst)
        print("{} bytes".format(os.path.getsize(dst)))
    total = sum(os.path.getsize(os.path.join(a.out, f)) for f in FILES
                if os.path.exists(os.path.join(a.out, f)))
    print("done: {}/{} files, {:.1f} KB in {}".format(
        sum(os.path.exists(os.path.join(a.out, f)) for f in FILES), len(FILES),
        total / 1024, a.out))


if __name__ == "__main__":
    main()
