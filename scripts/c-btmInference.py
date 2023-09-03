from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from torch.nn import PairwiseDistance
import numpy 

def load_models(model_names):
    """
    takes in list of model names and loads model and corresponding tokenizers
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
  """
  inputs = tokenizer(prompt)
  print(inputs['input_ids'])
  sizeOfInputs = len(inputs['input_ids']) 
  outputs = model(**inputs, max_new_tokens=1, 
                  return_dict_in_generate=True,
                  output_scores=True
                  )
  predToken = outputs['sequences'][0][sizeOfInputs] # pull out first token only
  predTokenProb = outputs['scores'][0][0]['predToken']

  return predToken, predTokenProb

def findEnsembleWeights(embedder, prompt, T, clusterCenters):
  """
  takes in embedding of context and cluster center of domain, and T parameter 
  calculates weighted logit between context and cluster center
  """
  ensembleWeights = []
  pdist = torch.nn.PairwiseDistance(p=2)
  for domain, clusterCenter in enumerate(clusterCenters): # assuming modelList and clusterCenters have matching indices
    embeddedInput = embedder.encode(prompt) 
    clusterCenter = torch.tensor(clusterCenter)
    embeddedInput = torch.tensor(embeddedInput) 
    ensembleWeights.append(torch.exp(-1 * torch.pow(pdist(embeddedInput, clusterCenter),2) / T))

  return torch.tensor(ensembleWeights) 



def topKFilter(ensembleWeights, k):
  """
  takes in ensemble weights and k parameter
  returns top k ensemble weights to determine most relevant domains given context
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
  """
  currToken, currSequence = None, prompt
  while currToken != end_token or len(currSequence) < maxLength:
    currToken = findNextToken(embedder, models, tokenizers, prompt, k, T, clusterCenters)
    currSequence = torch.cat((currSequence, currToken), dim=0)

  return generateSequence


def run_inference(embedder, model_names, input_file, output_file, end_token, maxLength, k, T, clusterCenters):

    models, tokenizers = load_models(model_names) # TODO: fails to recognize models even with credentials

    with open(input_file, 'r') as file:
        input_data = file.readlines()

    results = []
    for prompt in input_data:
        results.append(generateSequence(embedder, prompt, end_token, models, tokenizers, maxLength, k, T, clusterCenters))

    with open(output_file, 'w') as file:
        for result in results:
            file.write(f"{result}\n")
