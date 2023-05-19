from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from torch import torch.nn.PairwiseDistance
import numpy 

def load_models(model_names):
    """
    takes in list of model names and loads model and corresponding tokenizers
    returns separate lists of models and tokenizers to be accessed by other functions later
    """
    models = []
    tokenizers = []

    for model_name in model_names:
        model = AutoModel.from_pretrained(model_name)
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
  sizeOfInputs = inputs['input_ids'].size()[1]
  outputs = model(**inputs, max_new_tokens=1, 
                  return_dict_in_generate=True,
                  output_scores=True
                  )
  predToken = outputs['sequences'][0][sizeOfInputs] # pull out first token only
  predTokenProb = outputs['scores'][0][0]['predToken']
 
  return predToken, predTokenProb

def findEnsembleWeights(prompt, clusterCenters, T):
  """
  takes in embedding of context and cluster center of domain, and T parameter 
  calculates weighted logit between context and cluster center
  """
  ensembleWeights = []
  pdist = torch.nn.PairwiseDistance(p=2)

  for domain, clusterCenter in enumerate(clusterCenters): # assuming modelList and clusterCenters have matching indices
    tokenizer = tokenizers[domain]
    tokenizedInput = tokenizer(prompt)
    ensembleWeights.append(torch.exp(-1 * torch.pow(pdist(tokenizedInput, clusterCenter),2) / T))
  
  return ensembleWeights



def topKFilter(ensembleWeights, k):
  """
  takes in ensemble weights and k parameter
  returns top k ensemble weights to determine most relevant domains given context
  """
  topK = torch.topk(ensembleWeights, k=k)
  topK = [float(p)/torch.sum(topk.values) for p in topk.values]
  
  return topK

def findNextToken(prompt, k, T, clusterCenters):
  """
  takes in prompt, temperature T parameter and clusterCenters
  finds k most relevant domains given context
  returns most likely next token from predictions of most relevant domain experts
  """
  ensembleWeights = findEnsembleWeights(prompt, T, clusterCenters)
  topK = topKFilter(ensembleWeights, k)
  expertsPredictedTokens = []
  for index in topK.indices:
    predToken, predTokenProb = generateNextTokenFromExpert(prompt,
                                                           tokenizers[index],
                                                           models[index])
    predTokenProb *= topK.values[index]
    expertsPredictedTokens.append((predToken, predTokenProb.numpy())) # convert to regular number for max

  return max(expertsPredictedTokens, key=lambda x: x[1])

def generateSequence(prompt, end_token, maxLength, k, T, clusterCenters):
  """
  takes in prompt, end_token which ideally is uniform across all tokenizers, 
  parameter k, temperature T and cluster centers
  finds most likely token from most relevant domains based on prompt
  builds sequence until end token is generated or maxLength is reached
  """
  currToken, currSequence = None, prompt

  while currToken != end_token or len(currSequence) < maxLength:
    currToken = findNextToken(k, T, clusterCenters)
    currSequence = torch.cat((currSequence, currToken), dim=0)

  return generateSequence


def run_inference(model_names, input_file, output_file, end_token, maxLength, k, T, clusterCenters):
    
    models, tokenizers = load_models(model_names)

    with open(input_file, 'r') as file:
        input_data = file.readlines()

    results = []
    for prompt in input_data:
        results.append(generateSequence(prompt, end_token, maxLength, k, T, clusterCenters))

    with open(output_file, 'w') as file:
        for result in results:
            file.write(f"{result}\n")
