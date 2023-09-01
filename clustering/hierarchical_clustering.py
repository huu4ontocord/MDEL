import math
import random

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import arange, argmax
from tqdm import tqdm
from collections import Counter

import uuid

import numpy as np
from fast_pytorch_kmeans import KMeans

from feature_extractor import FeatureExtractor
from memmap_utils import np_memmap, get_np_memmap_length


class ClusterAnalysis(nn.Module):
    def __init__(
        self,  
        mmap_file=None,
        embed_dim=128,
        dtype=np.float32,
    ):
        super().__init__()
        
        self.mmap_file = mmap_file
        self.embed_dim = embed_dim

        self.dtype = dtype

        self.clusters = {}
        self.span_to_cluster_label = {}


    @staticmethod
    def _cluster_one_batch(
        true_k,    
        spans, 
        clusters, 
        span_to_cluster_label, 
        level, 
        cluster_embeddings, 
        min_overlap_merge_cluster,
        device
    ):
        with torch.no_grad():  
            embeddings = torch.from_numpy(cluster_embeddings)

            km = KMeans(n_clusters=true_k, mode='cosine')
            km_labels = km.fit_predict(embeddings.to(device=device, dtype=torch.float32)).tolist()

            embeddings = None

        if not clusters:
            label_to_label = {}
            
            for span, label in zip(spans, km_labels):
                label = (label, level)
                
                if label not in label_to_label:
                    label_to_label[label] = (span[0], level)

                label = label_to_label[label]
                
                clusters[label] = clusters.get(label, []) +[ span]
                span_to_cluster_label[span] = label
                
            output = list(clusters.keys())
            
            return output

        tmp_cluster = {}
        
        for span, label in zip(spans, km_labels):
            tmp_cluster[label] = tmp_cluster.get(label, [])+[span]

        new_labels = []

        for a_cluster in tmp_cluster.values():        
            for span in a_cluster:
                need_labels = [span for span in a_cluster if span not in span_to_cluster_label or span_to_cluster_label[span][1] != level]
                cluster_labels = [span_to_cluster_label[span] for span in a_cluster if span in span_to_cluster_label and span_to_cluster_label[span][1] == level]
                
                if not need_labels: 
                    continue
                    
                if not cluster_labels:
                
                    label = (span[0], level)
                    
                else:
                    most_common = Counter(cluster_labels).most_common(1)[0]
                    
                    if most_common[1] < min_overlap_merge_cluster:
                        label = (span[0], level)
                        
                    else:
                        label = most_common[0]
                        
                new_labels.append(label)
                
                for span in need_labels:
                    clusters[label] = clusters.get(label, []) + [span]
                    span_to_cluster_label[span] = label
                    
        return new_labels


    def create_hiearchical_clusters(
        self,
        force_recluster_idxs=None,
        max_level=4,
        max_cluster_size=32, # Small value for debug purposes
        min_overlap_merge_cluster=2,
        prefered_leaf_node_size=None, 
        kmeans_batch_size=250000,
        use_tqdm=False,
        device='cuda:0'
    ):
        mmap_file = self.mmap_file
        embed_dim = self.embed_dim
        dtype = self.dtype
        
        mmap_len = get_np_memmap_length(mmap_file, [0, embed_dim], dtype=dtype)

        clusters = self.clusters
        span_to_cluster_label = self.span_to_cluster_label
        
        if force_recluster_idxs:
            force_recluster_idxs = set(force_recluster_idxs)
        else:
            force_recluster_idxs = ()
            
        already_clustered = set([span[0] for span in span_to_cluster_label if span[1] == 0 and span[0] not in force_recluster_idxs])
        
        idxs = []
        
        if force_recluster_idxs:  
            idxs = list(force_recluster_idxs)
            force_recluster_idxs = None
            
        idxs.extend([idx for idx in range(mmap_len) if idx not in already_clustered])

        if not idxs: 
            return
        
        already_clustered = list(already_clustered)
        
        if len(already_clustered) > int(0.5 * kmeans_batch_size):
            idxs.extend(random.sample(already_clustered, int(0.5 * kmeans_batch_size)))
        else:
            idxs.extend(already_clustered)
            
        already_clustered = None
        
        idxs.extend([span[0] for span in span_to_cluster_label if span[1] != 0])
        idxs = list(set(idxs))
        random.shuffle(idxs)
        
        if not prefered_leaf_node_size:
            prefered_leaf_node_size= int(max_cluster_size * 0.7)
            
        for level in range(max_level):
            all_spans = [(idx, level) for idx in idxs]
            len_spans = len(all_spans)
            
            step_size = int(0.7 * kmeans_batch_size)    
            num_times = max(3, math.ceil(len_spans / step_size))

            if use_tqdm:
                num_times_2 = tqdm.tqdm(range(num_times))
                
            else:
                num_times_2 = range(num_times)
                
            for times in num_times_2:
                max_rng = min(len_spans, step_size)
  
                spans = all_spans[:max_rng]
   
                not_already_clustered = [span for span in all_spans[:max_rng - step_size] if span not in span_to_cluster_label]
     
                if len(not_already_clustered) > int(0.5 * kmeans_batch_size):
                    spans.extend(random.sample(not_already_clustered, int(0.5 * kmeans_batch_size)))
                else:
                    spans.extend(not_already_clustered)

                if len(spans) == 0: break

                already_clustered = [span for span in all_spans[:max_rng - step_size] if span in span_to_cluster_label]
                        
                if len(already_clustered) > int(0.5 * kmeans_batch_size):                
                    spans.extend(random.sample(already_clustered, int(0.5 * kmeans_batch_size)))
                    
                else:
                    spans.extend(already_clustered)

                embedding_idxs = [span[0] for span in spans]
   
                if level == 0:                
                    true_k = int(len(embedding_idxs) / prefered_leaf_node_size)
                    
                else:
                    true_k = int(len(embedding_idxs ) / max_cluster_size)

                cluster_embeddings = np_memmap(mmap_file, shape=[mmap_len, embed_dim], idxs=embedding_idxs, dtype=dtype)

                new_labels = self._cluster_one_batch(true_k, spans, clusters, span_to_cluster_label, level, cluster_embeddings, min_overlap_merge_cluster, device)
                        
                if not new_labels: 
                    break
                        
                need_more = False
                        
                if times <= num_times - 2:
                    for label in new_labels:
                        if len(clusters[label]) < prefered_leaf_node_size:
                            del clusters[label]

                            need_more = True
                            
                if not need_more: 
                    break

            idxs = [val[0][0] for key, val in clusters.items() if key[1] == level]

            if len(idxs) < max_cluster_size:
                break


def main():
    cluster_analysis = ClusterAnalysis(
        mmap_file='output/embeddings.mmap',
        embed_dim=1024
    )

    cluster_analysis.create_hiearchical_clusters()

    print(list(cluster_analysis.clusters.keys()))


if __name__ == '__main__':
    main()
