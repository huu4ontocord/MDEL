import argparse
from datasets import load_dataset


def upload_dataset(file_path, hf_repo):
    dataset = load_dataset("json", data_files = file_path)
    dataset.push_to_hub(hf_repo)

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