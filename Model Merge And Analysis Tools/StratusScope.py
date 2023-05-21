import os
import subprocess
import json
import torch
from transformers import AutoModel, AutoConfig, logging
import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog
from colorama import init, Fore, Style

logging.set_verbosity_warning()
logging.set_verbosity_error()

def select_folder():
    Tk().withdraw()
    folder = filedialog.askdirectory()
    return folder

def clear_console():
    if os.name == "nt":  # For Windows
        subprocess.call("cls", shell=True)
    else:  # For Linux and macOS
        subprocess.call("clear", shell=True)

def load_sharded_layer(folder, target_layer):
    files = os.listdir(folder)
    model_files = sorted([f for f in files if f.startswith('pytorch_model-') and f.endswith('.bin')])

    layer_state_dict = {}
    for model_file in model_files:
        shard = torch.load(os.path.join(folder, model_file), map_location=torch.device('cpu'))
        for name, param in shard.items():
            layer_number = get_layer_number(name)
            if layer_number == target_layer:
                layer_state_dict[name] = param

    return layer_state_dict

def get_layer_number(name):
    parts = name.split('.')
    for part in parts:
        if part.isdigit():
            return int(part)
    return None

def get_total_layers(model_folder):
    files = os.listdir(model_folder)
    model_files = sorted([f for f in files if f.startswith('pytorch_model-') and f.endswith('.bin')])

    all_layers = set()

    for model_file in model_files:
        shard = torch.load(os.path.join(model_folder, model_file), map_location=torch.device('cpu'))
        for name in shard.keys():
            layer_number = get_layer_number(name)
            all_layers.add(layer_number)

    return len(all_layers)

# https://pytorch.org/docs/stable/tensors.html

def compare_layers(model1_folder, model2_folder):
    layer_diffs = []
    newline = '\n'
    num_layers = get_total_layers(model1_folder) -1
    print(f"Torch Version: {torch.__version__}")
    print(f"Total Layers Found: {num_layers}{newline}")

    for layer_number in range(num_layers):
        layer_diff = 0

        model1_layer = load_sharded_layer(model1_folder, layer_number)
        model2_layer = load_sharded_layer(model2_folder, layer_number)

        for n1, p1 in model1_layer.items():
            p2 = model2_layer[n1]

            print(f"{newline}{Fore.YELLOW}--------Found Tensor Pair--------{newline}")
            print(f"p1 = {p1}")
            print(f"p2 = {p2}")
            print(f"{newline}{Fore.GREEN}--------Casting p1 & p2 tensor pair to float32--------{newline}")
            p1 = p1.detach().to(torch.float32)
            print(f"p1 = {p1}")
            p2 = p2.detach().to(torch.float32)
            print(f"p2 = {p2}")
            
            if not (torch.isinf(p1).any() or torch.isinf(p2).any()):
                diff = torch.abs(p1 - p2).sum().item()
                layer_diff += diff

        print(f"{newline}{Fore.CYAN}----------- Layer {layer_number}: Aggregate Difference = {layer_diff} -----------{Style.RESET_ALL}{newline}")
        layer_diffs.append(layer_diff)

    return layer_diffs

def plot_layer_diff(layer_diffs, model1_name, model2_name):
    plt.figure(figsize=(20, 6))
    num_layers = len(layer_diffs)
    layer_indices = range(num_layers)
    plt.bar(layer_indices, layer_diffs)
    plt.xticks(layer_indices)
    plt.xlabel('Layer')
    plt.ylabel('Difference')
    plt.title(f"{model1_name} vs {model2_name} Layer Difference")
    plt.ylim(bottom=0)
    print("Script completed, close graph to unload models and return to commandline.")
    plt.show()

def main():
    print("Select model1 folder:")
    model1_folder = select_folder()
    model1_name = os.path.basename(model1_folder)
    print("Select model2 folder:")
    model2_folder = select_folder()
    model2_name = os.path.basename(model2_folder)

    print("Examining Models...")
    clear_console()
    layer_diffs = compare_layers(model1_folder, model2_folder)

    plot_layer_diff(layer_diffs, model1_name, model2_name)

    torch.cuda.empty_cache()
    import gc
    gc.collect()

if __name__ == "__main__":
    main()