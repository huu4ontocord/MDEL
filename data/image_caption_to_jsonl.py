import io
import json
import os
import pandas as pd
from tqdm import tqdm
import argparse
import re
import random

def get_file_name_from_index(index, dataset, dataset_path):
    if dataset == "laion_1b" or dataset == "laion_2b_en" or dataset == "laion_2b_multi":
        # List all files in the dataset_path
        files = os.listdir(dataset_path)
        
        # Get the prefix (dataset) and the file format (e.g., '.parquet' or '.npy')
        prefix, file_format = os.path.splitext(files[0])
        
        # Extract the index pattern from the first file (e.g., 0000)
        index_pattern = re.search(r'\d+', prefix).group()
        
        # Calculate the number of leading zeros
        leading_zeros = len(index_pattern)
        
        # Format the new index with the correct number of leading zeros
        new_index = str(index).zfill(leading_zeros)
        
        # Replace the index pattern with the new index
        new_file_name = prefix.replace(index_pattern, new_index) + file_format
        
        return new_file_name
    if dataset == "mmc4":
        # List all files in the dataset_path
        files = os.listdir(dataset_path)
        
        # Get the first file in the list
        first_file = files[0]
        
        # Extract the prefix and index pattern using regular expression
        match = re.match(r'(.+)_shard_(\d+)(_.+)', first_file)
        
        if match is None:
            raise ValueError("No matching files found in the dataset_path.")
        
        prefix, _, suffix = match.groups()
        
        # Replace the index pattern with the new index
        new_file_name = f"{prefix}_shard_{index}{suffix}"
        
        return new_file_name

TEXT_VARIANTS = [
("Describe this image", ""),
("The caption of the image", "is"),
("The image", "shows"),
("The image", "depicts"),
("This illustration", "represents"),
("The snapshot", "captures"),
("The scene", "consists of"),
("In the photo", "is"),
("This visual", "displays"),
("A picture of", "has"),
("The image", "features"),
("This graphic", "presents"),
("The image", "consists of"),
("The representation", "is of"),
("The photo", "captures"),
("This depiction", "reveals"),
("The scene", "shows"),
("The picture", "represents"),
("The image", "demonstrates"),
("In the illustration", "is"),
("This visual representation", "displays"),
("The photograph", "features"),
("The image", "presents"),
("This snapshot", "depicts"),
("The artwork", "shows"),
("The scene", "portrays"),
("This graphic", "represents"),
("This picture", "contains"),
("The image", "portrays"),
("In this visual", "is"),
("The illustration", "depicts"),
("This photo", "shows"),
("The image", "reveals"),
("The snapshot", "displays"),
("This picture", "presents"),
("The image", "illustrates"),
("This scene", "features"),
("The photograph", "represents"),
("The graphic", "depicts"),
("This illustration", "displays"),
("The picture", "demonstrates"),
("In the image", "is"),
("The visual", "presents"),
("This representation", "portrays"),
("The snapshot", "illustrates"),
("This photograph", "captures"),
("Can you describe what's in the image?", ""),
("What do you see in this picture?", ""),
("Tell me what's happening in the photo", ""),
("Explain the scene in the illustration", ""),
("What does this image represent?", ""),
("Provide a description of this snapshot", ""),
("What's going on in this graphic?", ""),
("Please give a brief summary of the photo", ""),
("What elements can you identify in this picture?", ""),
]

def get_text_variant(image_tag, caption):
    text_variant = random.choice(TEXT_VARIANTS)
    text = text_variant[0] + " " + image_tag + " " + text_variant[1] + " " + caption
    return text

def convert_parquet_to_jsonl(df, output_dir, output_filename, dataset_name):
    df['image_info'] = df.apply(lambda x: [{"image_name": x['key'], "raw_url": x['url'], "matched_text_index": 1, "width": x['width'], "height": x['height']}], axis=1)
    df["shard_index"] = index
    df["index"] = df.reset_index().index
    df["dataset"] = dataset_name
    df["modality"] = "i"
    df['text'] = df.apply(lambda x: get_text_variant('<|image|> <<'+str(x['shard_index']) +', '+ str(x['index']) +', '+x['dataset']+', '+x['modality']+'>> <|/image|>', x['caption']), axis=1)
    df = df[['text', 'image_info', 'shard_index', 'index', 'dataset', 'modality']]
    json_df = json.loads(df.to_json(orient="records"))
    with open(os.path.join(output_dir, output_filename), 'w') as outfile:
        for entry in tqdm(json_df):
            json.dump(entry, outfile)
            outfile.write('\n')

# Main function
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to the input directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to the output directory")
    parser.add_argument("--start_index", type=int, required=True, help="Start index of the dataset")
    parser.add_argument("--end_index", type=int, required=True, help="End index of the dataset")

    args = parser.parse_args()
    for i in range(args.start_index, args.end_index):
        index = i
        input_path = os.path.join(args.input_dir, get_file_name_from_index(index, args.dataset, args.input_dir))
        df = pd.read_parquet(input_path)
        output_filename = input_path.split('/')[-1].replace("parquet", "jsonl")
        convert_parquet_to_jsonl(df, args.output_dir, output_filename, "laion_5b")