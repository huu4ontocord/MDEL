import os
import shutil
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter import *
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM, LlamaConfig
import torch

#mixer output settings

fp16 = False                 #perform operations in fp16. Saves memory, but CPU inference will not be possible.
always_output_fp16 = True   #if true, will output fp16 even if operating in fp32
max_shard_size = "2000MiB"  #set output shard size
verbose_info = True        #will show model information when loading
force_cpu = True            #only use cpu

# Create a GUI for selecting the first and second models, and save path for merged model
root = tk.Tk()

# per https://www.youtube.com/watch?v=0WafQCaok6g
# 12:05 is where he shows how to put things like buttons inside the frame
# if you are reading this yes I used a youtube tutorial on how to make a scrollbar lol -Digitous
root.title('Block Merge Model Layers')
root.geometry('500x720')

# Create a Main Frame
main_frame = Frame(root)
main_frame.pack(fill=BOTH, expand=1)

# Create a Canvas
my_canvas = Canvas(main_frame)
my_canvas.pack(side=LEFT, fill=BOTH, expand=1)

# Add A Scrollbar To The Canvas
my_scrollbar = ttk.Scrollbar(main_frame, orient=VERTICAL, command=my_canvas.yview)
my_scrollbar.pack(side=RIGHT, fill=Y)

# Configure The Canvas
my_canvas.configure(yscrollcommand=my_scrollbar.set)
my_canvas.bind('<Configure>', lambda e:my_canvas.configure(scrollregion = my_canvas.bbox("all")))

# Create ANOTHER Frame INSIDE the Canvas
second_frame = Frame(my_canvas)

# Add that New frame To a Window In The Canvas
my_canvas.create_window((0,0), window=second_frame, anchor="nw")

# example w/button on how to put a thing in the window with the scrollbar, second_frame is the proper place.
#for thing in range(100):
#    Button(second_frame, text=f'Button {thing} Yo!').grid(row=thing, column=0, pady=10, padx=10)

# Ask user to select the first model
print("Opening file dialog, please select FIRST model directory...")
first_model_path = filedialog.askdirectory(title="Select the first model")

# Ask user to select the second model
print("Opening file dialog, please select SECOND model directory...")
second_model_path = filedialog.askdirectory(title="Select the second model")

# Ask user to select the save path for the merged model
print("Opening file dialog, please select OUTPUT model directory...")
merged_model_path = filedialog.askdirectory(title="Select where to save the merged model")

if not first_model_path or not second_model_path:
    print("\nYou must select two directories containing models to merge and one output directory. Exiting.")
    exit()

with torch.no_grad(): 
    if fp16:
        torch.set_default_dtype(torch.float16)
    else:
        torch.set_default_dtype(torch.float32)

    device = torch.device("cuda") if (torch.cuda.is_available() and not force_cpu) else torch.device("cpu")
    print(device)
    
    # Load the first and second models
    print("Loading Model 1...")
    first_model = AutoModelForCausalLM.from_pretrained(first_model_path)
    first_model = first_model.to(device)
    first_model.eval()
    print("Model 1 Loaded. Dtype: " + str(first_model.dtype))
    
    print("Loading Model 2...")
    second_model = AutoModelForCausalLM.from_pretrained(second_model_path)
    second_model = second_model.to(device)
    second_model.eval()
    print("Model 2 Loaded. Dtype: " + str(second_model.dtype))
    
    # Determine the number of layers in the first model
    num_layers = first_model.config.num_hidden_layers
    #num_layers = len(first_model.transformer.h)
    #model.transformer.h
    #num_layers = len(first_model.encoder.layer)

    # Create a GUI for selecting merge ratios for each layer
    class LayerSlider(tk.Frame):
        def __init__(self, parent, layer_num):
            super().__init__(parent)
            
            self.layer_num = layer_num
            
            # Create a label for the layer slider
            self.layer_label = tk.Label(self, text=f"Layer {layer_num}")
            self.layer_label.grid(row=0, column=0)
            
            # Create a slider for the merge ratio
            self.slider = tk.Scale(self, from_=0, to=1, resolution=0.01, orient=tk.HORIZONTAL, length=400)
            self.slider.grid(row=0, column=1)
            
    # Create a window with sliders for each layer
    layer_sliders = []
    for i in range(num_layers):
#anchor second_frame puts things inside the area with the scrollbar
#        layer_slider = LayerSlider(root, i) #modified below
        layer_slider = LayerSlider(second_frame, i)
        layer_slider.pack()
        layer_sliders.append(layer_slider)

# Create a "commit and merge" button
def merge_models():
    with torch.no_grad(): 
        # Read the merge ratios from the sliders
        merge_ratios = [layer_slider.slider.get() for layer_slider in layer_sliders]
        
        #    # Check that the merge ratios add up to 1
        #    if sum(merge_ratios) != 1:
        #        messagebox.showerror("Error", "Merge ratios must add up to 1")
        #        return
        
        # Merge the models using the merge ratios
        for i in range(num_layers):
            # Determine how much of each layer to use from each model
            first_ratio = merge_ratios[i]
            second_ratio = 1 - first_ratio

    # Merge the layer from the two models dependent on the model type

def merge_models():
    with torch.no_grad():
        # Read the merge ratios from the sliders
        merge_ratios = [layer_slider.slider.get() for layer_slider in layer_sliders]

        # Merge the models using the merge ratios
        for i in range(num_layers):
            # Determine how much of each layer to use from each model
            first_ratio = merge_ratios[i]
            second_ratio = 1 - first_ratio
# gpt-j
            # Merge the layer from the two models
            if hasattr(first_model, "transformer"):# and hasattr(first_model.transformer, "h"):
                merged_layer = (first_model.transformer.h[i].state_dict(), second_model.transformer.h[i].state_dict())
                for key in merged_layer[0].keys():
                    merged_layer[0][key] = first_ratio * merged_layer[0][key] + second_ratio * merged_layer[1][key]

                if verbose_info:
                    print("Merging tensor " + str(i))

                # Create the merged model by replacing the layers in the second model with the merged layers
                second_model.transformer.h[i].load_state_dict(merged_layer[0])
                if verbose_info:
                    print("Migrating tensor " + str(i))
# maybe BERT
            elif hasattr(first_model, "encoder"):#and hasattr(first_model.encoder, "layer"):
                merged_layer = (first_model.encoder.layer[i].state_dict(), second_model.encoder.layer[i].state_dict())
                for key in merged_layer[0].keys():
                    merged_layer[0][key] = first_ratio * merged_layer[0][key] + second_ratio * merged_layer[1][key]

                if verbose_info:
                    print("Merging tensor " + str(i))

                # Create the merged model by replacing the layers in the second model with the merged layers
                second_model.encoder.layer[i].load_state_dict(merged_layer[0])
                if verbose_info:
                    print("Migrating tensor " + str(i))
# opt
            elif hasattr(first_model.model, "decoder"):#and hasattr(first_model.decoder, "layers"):
                merged_layer = (first_model.model.decoder.layers[i].state_dict(), second_model.model.decoder.layers[i].state_dict())
                for key in merged_layer[0].keys():
                    merged_layer[0][key] = first_ratio * merged_layer[0][key] + second_ratio * merged_layer[1][key]

                if verbose_info:
                    print("Merging tensor " + str(i))

                # Create the merged model by replacing the layers in the second model with the merged layers
                second_model.model.decoder.layers[i].load_state_dict(merged_layer[0])
                if verbose_info:
                    print("Migrating tensor " + str(i))
# neox/pythia
            elif hasattr(first_model, "gpt_neox"):#and hasattr(first_model.decoder, "layers"):
                tokenizer = AutoTokenizer.from_pretrained(first_model_path, use_fast=True)
                merged_layer = (first_model.gpt_neox.layers[i].state_dict(), second_model.gpt_neox.layers[i].state_dict())
                for key in merged_layer[0].keys():
                    merged_layer[0][key] = first_ratio * merged_layer[0][key] + second_ratio * merged_layer[1][key]

                if verbose_info:
                    print("Merging tensor " + str(i))

                # Create the merged model by replacing the layers in the second model with the merged layers
                second_model.gpt_neox.layers[i].load_state_dict(merged_layer[0])
                if verbose_info:
                    print("Migrating tensor " + str(i))
# llama
            elif hasattr(first_model, "model"):#and hasattr(first_model.decoder, "layers"):
                merged_layer = (first_model.model.layers[i].state_dict(), second_model.model.layers[i].state_dict())
                for key in merged_layer[0].keys():
                    merged_layer[0][key] = first_ratio * merged_layer[0][key] + second_ratio * merged_layer[1][key]

                if verbose_info:
                    print("Merging tensor " + str(i))

                # Create the merged model by replacing the layers in the second model with the merged layers
                second_model.model.layers[i].load_state_dict(merged_layer[0])
                if verbose_info:
                    print("Migrating tensor " + str(i))
 
            else:
# model isn't supported
                raise ValueError("Unsupported model architecture")

#anchor got rid of the script generating a converted_model folder, simply adds / to the path now.
        if merged_model_path:
            print("Saving new model...")
            newsavedpath = merged_model_path + "/"
            #copies necessary files from the first selected model folder into the merged model folder
# Define a list of the files to copy
            files_to_copy = ["special_tokens_map.json", "tokenizer_config.json", "vocab.json", "merges.txt"]
# Copy each file to the new folder
            for filename in files_to_copy:
                src_path = os.path.join(first_model_path, filename)
                dst_path = os.path.join(merged_model_path, filename)
                try:
                    shutil.copy2(src_path, dst_path)
                except FileNotFoundError:
                    print("\nFile " + filename + " not found in" + first_model_path + ". Skipping.")
            if always_output_fp16 and not fp16:
                second_model.half()

            second_model.save_pretrained(newsavedpath, max_shard_size=max_shard_size)
            print("\nSaved to: " + newsavedpath)
        else:
            print("\nOutput model was not saved as no output path was selected.")

        # Close the GUI
        root.destroy()

#commit button on GUI
commit_button = tk.Button(root, text="Commit and Merge", command=merge_models)
commit_button.pack()

def handle_return(event):
    merge_models()

#allows pressing Enter (if the GUI is the current window) to commit and merge.
root.bind('<Return>', handle_return)

# Run the GUI
root.mainloop()