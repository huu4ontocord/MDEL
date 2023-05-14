import io
import json
import os
import pandas as pd
from tqdm import tqdm
import argparse
import re
import numpy as np
from image_caption_to_jsonl import get_file_name_from_index

def convert_mmc4_shard(jsonl_filepath, embed_filepath, shard_index, image_pos="before"):
    # Pre-allocate np array
    embed_dict = np.load(embed_filepath, allow_pickle=True)
    num_rows = len(embed_dict)
    num_cols = embed_dict[list(embed_dict.keys())[0]].shape[0]
    embeddings = np.zeros((num_rows, num_cols))

    # Create map from image id to index
    image_id_to_index = {}

    # Load jsonl file
    jsonl_list = []
    with open(jsonl_filepath, 'r') as f:
        # Iterate over jsonl file
        for idx, line in tqdm(enumerate(f.readlines())):
            data = json.loads(line)

            text_list = data['text_list']

            # Get image id
            for image in data['image_info']:
                image_name = image['image_name']

                # Get image index if image id in map
                image_index = image_id_to_index.get(image_name, None)

                # If image id not in map, add to map and add to embeddings
                if image_index is None:
                    image_index = len(image_id_to_index)
                    image_id_to_index[image_name] = image_index
                    embeddings[len(image_id_to_index) - 1] = embed_dict[image_name]


                if image_pos == "before":
                    index_to_add = image['matched_text_index']
                elif image_pos == "after":
                    index_to_add = image['matched_text_index'] + 1
                elif image_pos == "random":
                    index_to_add = image['matched_text_index'] + np.random.randint(0, 2)

                text_list.insert(index_to_add, f"<|image|> <<{shard_index},{image_index},mmc4,i>> <|/image|>")

            text = " ".join(text_list)

            # Add to jsonl list
            data['text'] = text
            jsonl_list.append(data)

    # Return new jsonl file and embeddings array
    return jsonl_list, embeddings
    
# Main function
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
    parser.add_argument("--input_dir_embeddings", type=str, required=True, help="Path to the input directory")
    parser.add_argument("--input_dir_jsonl", type=str, required=True, help="Path to the input directory")
    parser.add_argument("--output_dir_embeddings", type=str, required=True, help="Path to the output directory")
    parser.add_argument("--output_dir_jsonl", type=str, required=True, help="Path to the output directory")
    parser.add_argument("--start_index", type=int, required=True, help="Start index of the dataset")
    parser.add_argument("--end_index", type=int, required=True, help="End index of the dataset")

    args = parser.parse_args()
    for i in range(args.start_index, args.end_index):
        index = i
        input_path_jsonl = os.path.join(args.input_dir_jsonl, get_file_name_from_index(index, args.dataset, args.input_dir_jsonl))
        input_path_embeddings = os.path.join(args.input_dir_embeddings, get_file_name_from_index(index, args.dataset, args.input_dir_embeddings))
        
        new_jsonl_list, embed = convert_mmc4_shard(input_path_jsonl, input_path_embeddings, index, image_pos="after")

        # Write jsonl list to output_dir_jsonl
        output_filename = input_path_jsonl.split('/')[-1].replace("parquet", "jsonl")
        with open(os.path.join(args.output_dir_jsonl, output_filename), 'w') as outfile:
            for entry in tqdm(new_jsonl_list):
                json.dump(entry, outfile)
                outfile.write('\n')
                
        # Write embeddings to output_dir_embeddings
        with open(os.path.join(args.output_dir_embeddings, output_filename.replace("jsonl", "npy")), 'wb') as f:
            np.save(f, embed)