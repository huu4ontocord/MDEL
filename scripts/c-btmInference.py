from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from torch.nn import PairwiseDistance
import numpy

def load_models(model_names):
    """
    model_names: takes in list of model names and loads model and corresponding tokenizers
    returns separate lists of models and tokenizers to be accessed by other functions later
    """
    models = []
    tokenizers = []

    for model_name in model_names:
        model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Freeze the parameters of the loaded models if needed
        model.eval()
        for param in model.parameters():
            param.requires_grad = False

        models.append(model)
        tokenizers.append(tokenizer)

    return models, tokenizers

def generateNextTokenFromExpert(prompt, tokenizer, model):
    """
    takes in prompt, tokenizer and model which should be passed from topKFilter
    as topKFilter determines which domains are most relevant given the context
    returns the predicted token of the model and its corresponding probability
    prompt: the string prompt from input_file
    tokenizer: the tokenizer that's been trained on the most relevant domain
    model: the model that's been trained on the most relevant domain
    """
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    generated_tokens = []

    for _ in range(2):
        with torch.no_grad():
            output = model(input_ids)

        next_token_logits = output.logits[:, -1, :]
        next_token_probabilities = next_token_logits.softmax(dim=-1)

        # Sample the next token based on probabilities
        next_token = torch.multinomial(next_token_probabilities, num_samples=1)

        # Get the probability of the chosen token
        chosen_token_probability = next_token_probabilities[0, next_token.item()].item()

        generated_tokens.append(next_token.item())
        input_ids = torch.cat((input_ids, next_token), dim=-1)

    generated_text = tokenizer.decode(generated_tokens)
    return generated_text, chosen_token_probability




def findEnsembleWeights(embedder, prompt, T, clusterCenters):
  """
  takes in cluster centers and embedder using in clustering, and prompt, and T parameter
  calculates weighted logit between context and cluster center
  embedder: the embedder that was used to train the clustering model
  prompt: the string prompt from input_file
  T: temperature parameter for softmax
  clusterCenters: list of cluster centers
  """
  ensembleWeights = torch.tensor([])#[]
  pdist = torch.nn.PairwiseDistance(p=2)
  embeddedInput = embedder.encode(prompt)
  
  for domain, clusterCenter in enumerate(clusterCenters): # assuming modelList and clusterCenters have matching indices
    ensembleWeight = 0

    for token in embeddedInput[0]:

      token = torch.tensor(token)
      ensembleWeight += torch.exp(-1 * torch.pow(pdist(token, clusterCenter),2) / T)
    # Check if ensembleWeights is empty, if so, initialize it
    if ensembleWeights is None:
        ensembleWeights = ensembleWeight
    else:
        # Check if ensembleWeights and ensembleWeight have compatible shapes
        if ensembleWeights.ndim == 0:
            # If ensembleWeights is a scalar, convert it to a 1-dimensional tensor
            ensembleWeights = ensembleWeights.unsqueeze(0)
        if ensembleWeight.ndim == 0:
            # If ensembleWeight is a scalar, convert it to a 1-dimensional tensor
            ensembleWeight = ensembleWeight.unsqueeze(0)

    # Concatenate the tensors
    ensembleWeights = torch.cat((ensembleWeights, ensembleWeight))

  return ensembleWeights 



def topKFilter(ensembleWeights, k):
  """
  takes in ensemble weights and k parameter
  returns top k ensemble weights to determine most relevant domains given context
  ensembleWeights: list of ensemble weights as calculated in findEnsembleWeights
  k: number of top experts to choose
  """
  topK = torch.topk(ensembleWeights, k=k)
  indices = topK.indices
  topK = [float(p)/torch.sum(topK.values) for p in topK.values]

  return topK, indices

def findNextToken(embedder, models, tokenizers, prompt, k, T, clusterCenters):
  """
  takes in prompt, temperature T parameter and clusterCenters
  finds k most relevant domains given context
  returns most likely next token from predictions of most relevant domain experts
  embedder: the embedder that was used to train the clustering model
  models: list of models that were trained on most relevant domains
  tokenizers: list of tokenizers that were trained on most relevant domains
  prompt: the string prompt from input_file
  k: number of most relevant domains to choose from
  T: temperature parameter for softmax
  clusterCenters: list of cluster centers
  """
  ensembleWeights = findEnsembleWeights(embedder, prompt, T, clusterCenters)
  topKValues, topKIndices = topKFilter(ensembleWeights, k)
  expertsPredictedTokens = []
  for i, index in enumerate(topKIndices):
    predToken, predTokenProb = generateNextTokenFromExpert(
                                                           prompt,
                                                           tokenizers[index],
                                                           models[index]
                                                           )
    predTokenProb *= topKValues[i]
    expertsPredictedTokens.append((predToken, predTokenProb.numpy())) # convert to regular number for max

  return max(expertsPredictedTokens, key=lambda x: x[1])

def generateSequence(embedder, prompt, end_token, models, tokenizers, maxLength, k, T, clusterCenters):
  """
  takes in prompt, end_token which ideally is uniform across all tokenizers,
  parameter k, temperature T and cluster centers
  finds most likely token from most relevant domains based on prompt
  builds sequence until end token is generated or maxLength is reached
  embedder: the embedder that was used to train the clustering model
  models: list of models that were trained on most relevant domains
  tokenizers: list of tokenizers that were trained on most relevant domains
  prompt: the string prompt from input_file
  end_token: the token that ideally is uniform across all tokenizers
  k: number of most relevant domains to choose from
  T: temperature parameter for softmax
  clusterCenters: list of cluster centers
  """
  currToken, currSequence = None, prompt
  while (len(currSequence) < maxLength) and (currToken != end_token):
    currToken, currTokenProb = findNextToken(embedder, models, tokenizers, currSequence, k, T, clusterCenters)
    currSequence = currSequence + currToken

  return currSequence


def run_inference(embedder, model_names, input_file, output_file, end_token, maxLength, k, T, clusterCenters): #, models, tokenizers):
    """
    embedder: the embedder that was used to train the clustering model
    model_names: list of model names 
    input_file: input file name, contents are prompts
    output_file: where generated sequences are written to
    end_token: end token to signify termination of sequence
    maxLength: max length of generated sequence
    k: number of most relevant domains to choose from
    T: temperature parameter for softmax
    clusterCenters: list of cluster centers
    """
    models, tokenizers = load_models(model_names)

    with open(input_file, 'r') as file:
        input_data = file.readlines()
    print(input_data)
    results = []
    for prompt in input_data:
        results.append(generateSequence(embedder, prompt, end_token, models, tokenizers, maxLength, k, T, clusterCenters))

    with open(output_file, 'w') as file:
        for result in results:
            file.write(f"{result}\n")
