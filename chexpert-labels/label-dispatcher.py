import argparse
import math
import os

import docker
import pandas as pd
from tqdm import tqdm, trange

import rrg

parser = argparse.ArgumentParser()
parser.add_argument("--n_jobs", type=int, required=True)
parser.add_argument("--data_path", required=True)
parser.add_argument("--output_path", required=True)
parser.add_argument("--image", default="chexpert-labeler:latest")
args = parser.parse_args()

os.makedirs(args.output_path)

ds = rrg.data.get_dataset(args.data_path)

train_findings = ds["train"]["findings"]
train_impressions = ds["train"]["impression"]
val_findings = ds["validation"]["findings"]
val_impressions = ds["validation"]["impression"]

train_notes = pd.Series(train_findings) + "\n" + pd.Series(train_impressions)
val_notes = pd.Series(val_findings) + "\n" + pd.Series(val_impressions)

val_frac = len(val_notes) / (len(train_notes) + len(val_notes))
val_jobs = max(int(args.n_jobs * val_frac), 1)
train_jobs = max(args.n_jobs - val_jobs, 1)

train_chunk_size = math.ceil(len(train_notes) / train_jobs)
val_chunk_size = math.ceil(len(val_notes) / val_jobs)

print(
    f"Assigning {train_jobs} jobs for {len(train_notes)} train notes ({train_chunk_size} notes per job)"
)
print(
    f"Assigning {val_jobs} jobs for {len(val_notes)} train notes ({val_chunk_size} notes per job)"
)

n_digits = max(len(str(len(train_notes))), len(str(len(val_notes))))

for i in range(0, len(train_notes), train_chunk_size):
    chunk_path = os.path.join(args.output_path, f"train_notes_{i:>0{n_digits}}.csv")
    train_notes.iloc[i : i + train_chunk_size].to_csv(
        chunk_path, index=False, header=False
    )
print(f"Wrote train note chunks to {args.output_path}")

for i in range(0, len(val_notes), val_chunk_size):
    chunk_path = os.path.join(args.output_path, f"val_notes_{i:>0{n_digits}}.csv")
    val_notes.iloc[i : i + val_chunk_size].to_csv(chunk_path, index=False, header=False)
print(f"Wrote val note chunks to {args.output_path}")

client = docker.from_env()
containers = []
for i in trange(0, len(train_notes), train_chunk_size, desc="Dispatching train jobs"):
    chunk_csv = f"train_notes_{i:>0{n_digits}}.csv"
    label_csv = chunk_csv.replace("notes", "labels")
    python_cmd = f"python label.py --reports_path /data/{chunk_csv} --output_path /data/{label_csv} --verbose"
    container = client.containers.run(
        args.image,
        python_cmd,
        detach=True,
        volumes=[f"{args.output_path}:/data"],
    )
    containers.append((container, label_csv))

for i in trange(0, len(val_notes), val_chunk_size, desc="Dispatching val jobs"):
    chunk_csv = f"val_notes_{i:>0{n_digits}}.csv"
    label_csv = chunk_csv.replace("notes", "labels")
    python_cmd = f"python label.py --reports_path /data/{chunk_csv} --output_path /data/{label_csv} --verbose"
    container = client.containers.run(
        args.image,
        python_cmd,
        detach=True,
        volumes=[f"{args.output_path}:/data"],
    )
    containers.append((container, label_csv))

for container, label_csv in tqdm(containers, desc="Waiting for containers to finish"):
    container.wait()
    chunk_log = os.path.join(
        args.output_path, label_csv.replace("labels", "logs").replace(".csv", ".log")
    )
    with open(chunk_log, "w") as f:
        f.write(container.logs().decode())
    container.remove()

train_label_csv = os.path.join(args.output_path, "train_labels.csv")
pd.concat(
    [
        pd.read_csv(os.path.join(args.output_path, label_csv))
        for _, label_csv in containers[:-val_jobs]
    ],
    ignore_index=True,
).to_csv(train_label_csv, index=False)
print(f"Wrote combined train labels to {train_label_csv}")

val_label_csv = os.path.join(args.output_path, "val_labels.csv")
pd.concat(
    [
        pd.read_csv(os.path.join(args.output_path, label_csv))
        for _, label_csv in containers[-val_jobs:]
    ],
    ignore_index=True,
).to_csv(val_label_csv, index=False)
print(f"Wrote combined val labels to {val_label_csv}")

for _, label_csv in containers:
    os.remove(os.path.join(args.output_path, label_csv))
    os.remove(os.path.join(args.output_path, label_csv.replace("labels", "notes")))
print("Cleaned up temporary files")
