import argparse
import glob
import os

from huggingface_hub import HfApi

api = HfApi()


def upload_dataset(file_path, hf_repo, split):
    auth_token = os.getenv('HF_ACCESS_TOKEN')

    resolved_paths = glob.glob(file_path)

    for path in resolved_paths:
        name = path.split('/')[-1]
        print(f'Uploading file {name}')
        api.upload_file(path_or_fileobj=path,
                        repo_id=hf_repo,
                        repo_type="dataset",
                        path_in_repo=f"data/{split}/{name}",
                        use_auth_token=auth_token)


def parse_args():
    parser = argparse.ArgumentParser(description='Upload dataset to Hugging Face Hub')
    parser.add_argument('--file-path', type=str, required=True, help='Path to dataset file')
    parser.add_argument('--hf-repo', type=str, required=True, help='Path to dataset file')
    parser.add_argument('--split', type=str, required=True, help='Split')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    file_path = args.file_path
    hf_repo = args.hf_repo
    split = args.split
    upload_dataset(file_path, hf_repo, split)
