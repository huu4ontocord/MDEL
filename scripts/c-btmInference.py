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
  """
  ensembleWeights = []
  pdist = torch.nn.PairwiseDistance(p=2)
  for domain, clusterCenter in enumerate(clusterCenters): # assuming modelList and clusterCenters have matching indices
    embeddedInput = embedder.encode(prompt)
    clusterCenter = torch.tensor(clusterCenter)
    embeddedInput = torch.tensor(embeddedInput) # TODO: do we need to tensor-ify them?
    ensembleWeights.append(torch.exp(-1 * torch.pow(pdist(embeddedInput, clusterCenter),2) / T))

  return torch.tensor(ensembleWeights) # TODO: return torch.tensor instead of list?



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
  while len(currSequence) < maxLength:# or currToken != end_token:
    currToken, currTokenProb = findNextToken(embedder, models, tokenizers, currSequence, k, T, clusterCenters)
    currSequence = currSequence + currToken

  return generateSequence


def run_inference(embedder, model_names, input_file, output_file, end_token, maxLength, k, T, clusterCenters, models, tokenizers):

    models, tokenizers = load_models(model_names) 

    with open(input_file, 'r') as file:
        input_data = file.readlines()

    results = []
    for prompt in input_data:
        results.append(generateSequence(embedder, prompt, end_token, models, tokenizers, maxLength, k, T, clusterCenters))

    with open(output_file, 'w') as file:
        for result in results:
            file.write(f"{result}\n")
