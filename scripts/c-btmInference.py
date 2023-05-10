import heapq
import torch


def findLogitsOfContextAndClusterCenter(contextEmbedding, clusterCenter, T):
  """
  Takes in embedding of context and cluster center of domain, and T parameter and 
  calculates weighted logit
  """
  pdist = torch.nn.PairwiseDistance(p=2)
  return torch.exp(-1 * torch.pow(pdist(contextEmbedding, clusterCenter),2) / T)


def filterTopK(contextEmbedding, clusterCenters, T, k):
  """
  Takes in embedding of context and cluster centers of domains, temperature parameter T
  and calculates top k filter - takes top k probabilities and normalizes to sum to 1
  """
  probabilities = []
  for center in clusterCenters:
    probabilities.append(findLogitsOfContextAndClusterCenter(contextEmbedding, center, T))
  probabilities = heapq.nlargest(k, probabilities)
  return [float(p)/torch.sum(probabilities) for p in probabilities]



def findCurrentTokenProbability(modelName, contextEmbedding, currentToken):
  """
  Takes in model name, context and current token and outputs probability of current token given context
  """
  outputs = modelName(contextEmbedding)
  return outputs.logits[0][0][currentToken]

def calculateCurrentTokenProbability(modelNames, contextEmbedding, currentToken):
  """
  Takes in list of expert models, context embedding, and current token and outputs formula 2 from c-btm paper
  """
  tokenProbability = 0
  for model in modelNames:
    nextTokenProb = findProbabilityOfNextWord(model, contextEmbedding, currentToken)
    ensembleWeights = filterTopK(contextEmbedding, clusterCenters, T, k)
    tokenProbability += (nextTokenProb * ensembleWeights)
  return tokenProbability
