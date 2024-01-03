from clustering.feature_extractor import FeatureExtractor
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from torch.nn import PairwiseDistance
import numpy as np
import fire


def topKFilter(ensembleWeights, k):
  """
  Filters and normalizes top k ensemble weights.

  Parameters:
  - ensembleWeights: list of ensemble weights as calculated in findEnsembleWeights
  - k: number of top experts to choose

  Returns:
  Tuple with the normalized weights and corresponding indices as they pertain to the top k domains.
  """
  topK = torch.topk(ensembleWeights, k=k)
  indices = topK.indices

  if torch.sum(topK.values) == 0:
    return [1/k for i in range(k)], indices

  normalizedTopKValues = [float(p)/torch.sum(topK.values) for p in topK.values]

  return normalizedTopKValues, indices



def findTopKModelProbabilities(indices, query, models, tokenizers):
  """
  Find the probabilities of all tokens appearing in the next step for each model in the given list.

  Parameters:
    - indices: A list of indices representing the models to consider.
    - query: The input query for which to predict the next token probabilities.
    - models: A list of PyTorch models for predicting next token probabilities.
    - tokenizers: A list of tokenizers corresponding to the models.

  Returns:
    List of tensors, where each tensor contains the probabilities of the next token
    for the corresponding model in the input list.

  The function iterates over the specified models, tokenizes the input query, and
    calculates the probabilities of all tokens in the next step for each model. The results are
    collected into a list and returned.

  Note: The models and tokenizers in the input lists should correspond to each other.
    The length of indices, models, and tokenizers should be the same.
  """
  all_models_next_token_probs = []

  for index in range(len(indices)):

      input_ids = tokenizers[index].encode(query, return_tensors="pt")
      output = models[index](input_ids)
      next_token_probs = output.logits[:, -1, :].softmax(dim=-1)
      all_models_next_token_probs.append(next_token_probs)

  return all_models_next_token_probs



def findEnsembleWeights(embedder, query, clusterCenters, T, firstToken=False, prompt_data=None):
  """
  Finds ensemble weights based on distance of query and cluster centers.

  Parameters:
  - embedder: the embedder that was used to train the clustering model
  - query: the string prompt from input_file
  - clusterCenters: dictionary of cluster centers with the keys as domain{i}, as per the input_file
  - T: temperature parameter for softmax
  - firstToken: boolean to indicate if the first token is yet to be generated
  - prompt_data: the entire json object for the prompt

  Returns:
  list the distance between the queries and each of the cluster centers.
  """
  ensembleWeights = []
  pdist = torch.nn.PairwiseDistance(p=2)

  for i in range(len(clusterCenters)):

    if firstToken:

      distance = torch.tensor(prompt_data['meta'][f'domain_score{i+1}']) # this should be the L2 norm between the prompt and the cluster center
    
    else:

      cluster = clusterCenters[f'domain{i+1}'] 

      embedded_query = embedder(query) 

      if type(embedded_query) is not torch.Tensor:

        embedded_query = torch.Tensor(embedded_query)

      distance = torch.pow(pdist(embedded_query, cluster),2)

    ensembleWeights.append(torch.exp(-1 * distance / T).mean())

  return torch.tensor(ensembleWeights)


def findModelProbabilities(embedder, query, clusterCenters, models, tokenizers, T, k, firstToken, prompt_data):
  """
  Calculates the sum of the probabilities of the ensembled tokens using the top k most relevant domains.

  Parameters:
  - embedder: the embedder that was used to train the clustering model
  - query: the string prompt from input_file
  - models: list of models that were trained on most relevant domains
  - tokenizers: list of tokenizers that were trained on most relevant domains
  - T: temperature parameter for softmax
  - k: number of most relevant domains to choose from
  - firstToken: boolean to indicate if the first token is yet to be generated
  - prompt_data: the entire json object for the prompt

  Returns:
  the ensembled probabilities of all tokens at the next step.
  """
  modelProbs = 0
  ensembleWeights = findEnsembleWeights(embedder, query, clusterCenters, T, firstToken, prompt_data)
  weights, indices = topKFilter(ensembleWeights, k)
  modelWeights = findTopKModelProbabilities(indices, query, models, tokenizers)

  for i in range(k):
    index = indices[i]
    modelProbs += modelWeights[index] * weights[index]

  return modelProbs


def findNextToken(embedder, query, clusterCenters, models, tokenizers, T, k, firstToken, prompt_data):
  """
  Returns decoded next-step token with highest probability.

  Parameters:
  - embedder: the embedder that was used to train the clustering model
  - query: the string prompt from input_file
  - clusterCenters: dictionary of cluster centers with the keys as domain{i}, as per the input_file
  - models: list of models that were trained on most relevant domains
  - tokenizers: list of tokenizers that were trained on most relevant domains
  - T: temperature parameter for softmax
  - k: number of most relevant domains to choose from
  - firstToken: boolean to indicate if the first token is yet to be generated
  - prompt_data: the entire json object for the prompt
  """
  modelProbs = findModelProbabilities(embedder,query, clusterCenters, models, tokenizers, T, k, firstToken, prompt_data)
  return tokenizers[0].decode(np.argmax(modelProbs)) # doesn't matter which tokenizer bc they're all the same


def generateSequence(embedder, prompt_data, end_token, clusterCenters, models, tokenizers, T, k, maxLength):
  """
  Takes in prompt, end_token which ideally is uniform across all tokenizers, parameter k,
  temperature T and cluster centers and finds most likely token from most relevant domains based on prompt. Then, it
  builds sequence until end token is generated or maxLength is reached.

  Parameters:
  - embedder: the embedder that was used to train the clustering model
  - query: the string prompt from input_file
  - end_token: the token that ideally is uniform across all tokenizers
  - clusterCenters: dictionary of cluster centers with the keys as domain{i}, as per the input_file
  - models: list of models that were trained on most relevant domains
  - tokenizers: list of tokenizers that were trained on most relevant domains
  - T: temperature parameter for softmax
  - k: number of most relevant domains to choose from
  - maxLength: the maximum length the generated output can reach
  - firstToken: boolean to indicate if the first token is yet to be generated
  - prompt_data: the entire json object for the prompt

  Returns:
  Generated string sequence.
  """
  prompt = prompt_data['text']
  currToken, currSequence = None, prompt
  while (len(currSequence) < maxLength) and (currToken != end_token):

    if not currToken:
      currToken = findNextToken(
          embedder, currSequence, clusterCenters, models, tokenizers, T, k, firstToken=True, prompt_data=prompt_data)
      currSequence += currToken
      continue

    if currToken == end_token:
      break

    currToken = findNextToken(
        embedder, currSequence, clusterCenters, models, tokenizers, T, k, firstToken=False, prompt_data=None)
    currSequence += currToken

  return currSequence


def load_models(model_names):
    """
    Loads models and tokenizers as per the ids provided in model_names.

    Parameters:
    model_names: takes in list of model names and loads model and corresponding tokenizers

    Returns:
    Separate lists of models and tokenizers to be accessed by other functions later
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


def run_inference(embedder, model_names, 
                  input_file, output_file, end_token, maxLength, k, T, clusterCenters): 
    """
    Function that takes in an input file of prompts and writes generated outputs to an output file.
    Parameters:
    - embedder: the embedder that was used to train the clustering model
    - model_names: list of model names
    - input_file: input file name, contents are prompts
    - output_file: where generated sequences are written to
    - end_token: end token to signify termination of sequence
    - maxLength: max length of generated sequence
    - k: number of most relevant domains to choose from
    - T: temperature parameter for softmax
    - clusterCenters: dictionary of cluster centers with the keys as domain{i}, as per the input_file
    Returns:
    Should write generated output to output file.
    """
    models, tokenizers = load_models(model_names)

    with open(input_file, 'r') as file:
        input_data = file.readlines()

    results = []
    for query in input_data:
        results.append(generateSequence(
            embedder, query, end_token, clusterCenters, models, tokenizers, T, k, maxLength)
        )

    with open(output_file, 'w') as file:
        for result in results:
            file.write(f"{result}\n")



if __name__ == '__main__':
  fire.Fire(run_inference)
