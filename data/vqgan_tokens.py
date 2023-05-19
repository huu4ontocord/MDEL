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

from tqdm import tqdm
import torch.multiprocessing as mp
from dalle_pytorch.vae import VQGanVAE
from huggingface_hub import hf_hub_download

warnings.filterwarnings("ignore", category=UserWarning)

VQGAN_VAE_PATH = "https://huggingface.co/boris/vqgan_f16_16384/blob/main/model.ckpt"
VQGAN_VAE_CONFIG_PATH = (
    "https://huggingface.co/boris/vqgan_f16_16384/blob/main/config.yaml"
)

EXPECTED_CHUNK_SIZE = 10000


def collate(batch):
    images, metas = zip(*batch)
    return torch.stack(images), metas


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
):
    model = VQGanVAE(
        vqgan_model_path=vqgan_model_path, vqgan_config_path=vqgan_config_path
    ).to(device=rank)

    num_paths_per_chunk = int(np.ceil(len(paths) / world_size))
    for path in tqdm(
        paths[
            rank
            * num_paths_per_chunk : min(len(paths), (rank + 1) * num_paths_per_chunk)
        ]
    ):
        basename = os.path.basename(path)
        output_path = os.path.join(
            output_dir, os.path.splitext(basename)[0] + ".parquet"
        )

        if s3:
            path = f"pipe:aws s3 cp {path} -"
        try:
            dataset = (
                wds.WebDataset(path)
                .decode(wds.imagehandler("torchrgb"))
                .to_tuple("jpg", "json")
            )

            dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=batch_size,
                collate_fn=collate,
                pin_memory=True,
                num_workers=num_workers,
            )
            rows = []
            embeddings = []
            for images, metas in tqdm(
                dataloader,
                total=int(np.ceil(EXPECTED_CHUNK_SIZE / batch_size)),
                desc=basename,
                leave=False,
            ):
                z = model.get_codebook_indices(images.to(rank))
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
        help="Number of workers for the dataloader",
    )
    parser.add_argument(
        "-bs",
        "--batch_size",
        type=int,
        default=128,
        help="Batch size for the dataloader",
    )
    parser.add_argument(
        "-ng",
        "--num_gpus",
        type=int,
        default=8,
        help="Number of GPUs to use",
    )
    args = parser.parse_args()

    vqgan_model = hf_hub_download(
        repo_id="boris/vqgan_f16_16384", filename="model.ckpt"
    )
    vqgan_config = hf_hub_download(
        repo_id="boris/vqgan_f16_16384", filename="config.yaml"
    )

    paths = braceexpand.braceexpand(args.paths)

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
        ),
        nprocs=args.num_gpus,
    )


if __name__ == "__main__":
    main()
