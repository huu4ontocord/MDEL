import argparse
import os

from huggingface_hub import HfApi

api = HfApi()


def upload_dataset(folder_path, hf_repo, split):
    auth_token = os.getenv('HF_ACCESS_TOKEN')

    print(f"Uploading {folder_path} to {hf_repo}...")
    
    api.upload_folder(folder_path=folder_path,
                    repo_id=hf_repo,
                    repo_type="dataset",
                    path_in_repo=f"data/{split}",
                    use_auth_token=auth_token)


def parse_args():
    parser = argparse.ArgumentParser(description='Upload dataset to Hugging Face Hub')
    parser.add_argument('--folder-path', type=str, required=True, help='Path to dataset file')
    parser.add_argument('--hf-repo', type=str, required=True, help='Path to dataset file')
    parser.add_argument('--split', type=str, required=True, help='Split')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    folder_path = args.folder_path
    hf_repo = args.hf_repo
    split = args.split
    upload_dataset(folder_path, hf_repo, split)
