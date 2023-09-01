import argparse
import itertools

import numpy as np

import pickle
from pathlib import Path

import torch

from tqdm.auto import tqdm

from torch.utils.data import DataLoader, IterableDataset

from kmeans_pytorch import KMeans as BalancedKMeans

from transformers import pipeline

from datasets import load_dataset

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt

from feature_extractor import FeatureExtractor
from memmap_utils import np_memmap


class ShardedIterator(IterableDataset):
    def __init__(self, iterable, num_shards, shard_id, fill_value=None):
        if shard_id < 0 or shard_id >= num_shards:
            raise ValueError('shard_id must be between 0 and num_shards')

        self._sharded_len = len(iterable) // num_shards
        if len(iterable) % num_shards > 0:
            self._sharded_len += 1

        self.itr = itertools.zip_longest(
            range(self._sharded_len),
            itertools.islice(iterable, shard_id, len(iterable), num_shards),
            fillvalue=fill_value
        )

    def __len__(self):
        return self._sharded_len

    def __iter__(self):
        return self

    def __next__(self):
        return next(self.itr)[1]



def load_model(path_to_model: Path):
    with open(path_to_model, 'rb') as file:
        output = pickle.load(file)

        file.close()

    return output


def extract_features(corpus, feature_extractor, batch_size=32, max_chars=256):
    corpus = [element[:max_chars] for element in corpus]
    batches = np.array_split(corpus, len(corpus) // batch_size, axis=0)

    features = []

    for batch in tqdm(batches):
        batch = list(batch) # batches is a list of numpy arrays

        features_current = feature_extractor(batch)
        features_current = np.max(features_current, axis=1)

        features.append(features_current)

    features = np.concatenate(features, axis=0)

    return features


def train_kmeans(features, n_clusters, path_to_kmeans, balanced=False, device='cpu'):
    kmeans = BalancedKMeans(n_clusters=n_clusters, device=device, balanced=balanced)
    
    batch_size = 512 # Hyperparameter
    batch_size = min(batch_size, len(features))
    
    batches = np.array_split(features, features.shape[0] // batch_size, axis=0)
    
    for idx, batch in tqdm(enumerate(batches)):
        kmeans.fit(torch.from_numpy(batch), iter_limit=20, online=True, iter_k=idx)

    with open(path_to_kmeans, 'wb+') as file:
        pickle.dump(kmeans, file)

        file.close()

    return kmeans


def main(n_clusters=16, balanced=False, output_dir=Path('cluster_output/'), shuffle_dataset=True, take_sample=None, embed_only=False, seed=42, visualize=False):
    dataset_name_train = 'JeanKaddour/minipile'
    content_column_train = 'text'
    
    subset_train = None # 'p3'
    split_train = 'train'
    
    dataset_train = load_dataset(dataset_name_train, subset_train, split=split_train, streaming=(take_sample is not None))
    
    if shuffle_dataset:
        dataset_train = dataset_train.shuffle(seed=seed)
    
    corpus = []
    
    for idx, element in enumerate(dataset_train):
        corpus.append(element[content_column_train])

        if take_sample:
            if idx >= take_sample:
                break

    if not output_dir.is_dir():
        output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using {device} device')

    feature_extractor_batch_size = 1

    feature_extractor_checkpoint = 'bigscience/bloom-560m' # 'sentence-transformers/LaBSE' # 'xlm-roberta-large'
    feature_extractor = FeatureExtractor(device=device) # pipeline('feature-extraction', framework='pt', model=feature_extractor_checkpoint)
    
    features = extract_features(corpus, feature_extractor, batch_size=feature_extractor_batch_size)
    
    memmap_file_path = 'output/embeddings.mmap' # TODO: Create a configs.py file

    np_memmap(memmap_file_path, data=features)

    if embed_only:
        return

    path_to_kmeans = output_dir / 'kmeans.pkl'
    kmeans = train_kmeans(features, n_clusters, path_to_kmeans, balanced=balanced, device=device)

    if visualize:
        tsne = TSNE(n_components=2)
        features_2d = tsne.fit_transform(features)

        plt.scatter(features_2d[:, 0], features_2d[:, 1], c=kmeans.predict(torch.from_numpy(features)).cpu())
        plt.show()

    return kmeans


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--num-clusters', required=True, type=int)
    parser.add_argument('--balanced', action='store_true')
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--eval-only', action='store_true')
    parser.add_argument('--shuffle-dataset', required=False, type=bool, default=True)
    parser.add_argument('--take-sample', required=False, type=int)
    parser.add_argument('--embed-only', required=False, type=bool, default=False)
    parser.add_argument('--visualize', action='store_true')

    args = parser.parse_args()

    if not args.eval_only:
        kmeans = main(
            n_clusters=args.num_clusters,
            balanced=args.balanced,
            output_dir=args.output_dir,
            take_sample=args.take_sample,
            shuffle_dataset=args.shuffle_dataset,
            embed_only=args.embed_only,
            visualize=args.visualize
        )

    path_to_kmeans = args.output_dir / 'kmeans.pkl'
    kmeans = load_model(path_to_kmeans)


# Usage

# python3 train_clusterer.py --num-clusters 4 --output-dir output/ --take-sample 128 --embed-only False --visualize
