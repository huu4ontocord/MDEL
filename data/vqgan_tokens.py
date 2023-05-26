# Adapted from https://huggingface.co/datasets/laion/laion_100m_vqgan_f8/blob/main/run_vqgan.py for vqgan_f16_16384

import os
import sys
import torch
import argparse
import traceback
import braceexpand
import warnings
import numpy as np
import pandas as pd
import webdataset as wds
import torch.multiprocessing as mp

from tqdm import tqdm
from dalle_pytorch.vae import VQGanVAE
from timeit import default_timer as timer

from huggingface_hub import hf_hub_download

warnings.filterwarnings("ignore", category=UserWarning)

VQGAN_VAE_PATH = "https://huggingface.co/boris/vqgan_f16_16384/blob/main/model.ckpt"
VQGAN_VAE_CONFIG_PATH = (
    "https://huggingface.co/boris/vqgan_f16_16384/blob/main/config.yaml"
)
ALLOWED_DATASETS = ["laion", "mmc4"]

EXPECTED_CHUNK_SIZE = 10000


def get_dataset(dataset_type, path, s3):
    if s3:
        path = f"pipe:aws s3 cp {path} -"

    if dataset_type == "laion":
        dataset = (
            wds.WebDataset(path)
            .decode(wds.imagehandler("torchrgb"))
            .to_tuple("jpg", "json")
        )
        return dataset
    elif dataset_type == "mmc4":

        def resize_image(sample):
            keys = ["png", "jpg", "jpeg"]
            for key in keys:
                if key in sample:
                    image = np.array(sample[key].resize((256, 256))).astype(np.float32)
                    image = image.transpose(2, 0, 1) / 255.0
                    sample["image"] = torch.from_numpy(image)
            return sample

        dataset = (
            wds.WebDataset(path)
            .decode("pil")
            .map(resize_image)
            .to_tuple("image", "__key__")
        )
        return dataset


def process_chunk(
    rank,
    world_size,
    vqgan_model_path,
    vqgan_config_path,
    paths,
    output_dir,
    num_workers,
    batch_size,
    s3,
    dataset_type,
):
    model = VQGanVAE(
        vqgan_model_path=vqgan_model_path, vqgan_config_path=vqgan_config_path
    ).to(device=rank)

    num_paths_per_chunk = int(np.ceil(len(paths) / world_size))

    worker_paths = paths[
        rank * num_paths_per_chunk : min(len(paths), (rank + 1) * num_paths_per_chunk)
    ]
    print (f"Rank: {rank} processing {len(worker_paths)} shards")
    for path in worker_paths:
        basename = os.path.basename(path)
        output_path = os.path.join(
            output_dir, os.path.splitext(basename)[0] + ".parquet"
        )

        try:
            dataset = get_dataset(dataset_type, path, s3)
            dataloader = torch.utils.data.DataLoader(
                dataset.batched(batch_size),
                batch_size=None,
                pin_memory=True,
                num_workers=num_workers,
            )
            rows = []
            embeddings = []
            for data, metas in tqdm(
                dataloader,
                total=int(np.ceil(EXPECTED_CHUNK_SIZE / batch_size)),
                desc=f"Rank : {rank}, Shard: {basename}",
                position=rank,
                leave=False,
            ):
                z = model.get_codebook_indices(data.to(rank))
                z = z.to(torch.int16)

                rows.extend(metas)
                embeddings.append(z)
            embeddings = torch.cat(embeddings, axis=0)

            df = pd.DataFrame(rows)
            embeddings_cpu = embeddings.cpu().numpy().reshape(len(df), -1)
            df["code"] = [item.tobytes() for item in embeddings_cpu]
            df.to_parquet(output_path, compression="brotli")
        except Exception:
            print(f"[-] Failed to process {basename}:", file=sys.stderr)
            traceback.print_exc()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--paths",
        type=str,
        help="/path/to/images/{0000..1111}.tar",
    )
    parser.add_argument(
        "-s3",
        action="store_true",
        help="Pass this flag if using s3 bucket",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        help="Output directory for *.parquet files with the code column",
    )
    parser.add_argument(
        "-nw",
        "--num_workers",
        type=int,
        help="Number of workers per gpu for the dataloader",
    )
    parser.add_argument(
        "-bs",
        "--batch_size",
        type=int,
        default=128,
        help="Batch size per gpu for the dataloader",
    )
    parser.add_argument(
        "-ng",
        "--num_gpus",
        type=int,
        default=8,
        help="Number of GPUs to use",
    )
    parser.add_argument(
        "-ds",
        "--dataset",
        type=str,
        default="laion",
        help="Type of dataset used. Can be 'laion' or 'mmc4'",
    )
    parser.add_argument(
        "-md",
        "--model_dir",
        type=str,
        help="Path to the directory to download hf models",
    )
    args = parser.parse_args()

    vqgan_model = hf_hub_download(
        repo_id="boris/vqgan_f16_16384", filename="model.ckpt", cache_dir=args.model_dir
    )
    vqgan_config = hf_hub_download(
        repo_id="boris/vqgan_f16_16384", filename="config.yaml", cache_dir=args.model_dir
    )

    paths = list(braceexpand.braceexpand(args.paths))

    start = timer()

    if args.dataset not in ALLOWED_DATASETS:
        raise ValueError(
            f"Dataset must be one of {ALLOWED_DATASETS}, got {args.dataset}"
        )

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if args.num_gpus > 1:
        mp.spawn(
            process_chunk,
            args=(
                args.num_gpus,
                vqgan_model,
                vqgan_config,
                paths,
                args.output_dir,
                args.num_workers,
                args.batch_size,
                args.s3,
                args.dataset,
            ),
            nprocs=args.num_gpus,
        )
    else:
        process_chunk(0, 1, vqgan_model, vqgan_config, paths, args.output_dir, args.num_workers, args.batch_size, args.s3, args.dataset)

    print(f"Processing {len(paths)} shards took {timer() - start} seconds")


if __name__ == "__main__":
    main()
