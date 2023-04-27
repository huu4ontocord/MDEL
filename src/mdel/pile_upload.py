import argparse

from huggingface_hub import HfApi

api = HfApi()


def upload_dataset(file_path, hf_repo):
    name = file_path.split('/')[-1]
    api.upload_file(path_or_fileobj=file_path,
                    repo_id=hf_repo,
                    repo_type="dataset",
                    path_in_repo=f"data/{name}")


def parse_args():
    parser = argparse.ArgumentParser(description='Upload dataset to Hugging Face Hub')
    parser.add_argument('--file_path', type=str, required=True, help='Path to dataset file')
    parser.add_argument('--hf_repo', type=str, required=True, help='Path to dataset file')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    file_path = args.file_path
    hf_repo = args.hf_repo
    upload_dataset(file_path, hf_repo)
