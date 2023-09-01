import requests
import logging

import trafilatura

from transformers import pipeline
from transformers import AutoTokenizer

import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

max_embedding_characters = 128 # This is a deliberately low value, as the current model is not intended for document embedding

feature_extractor_checkpoint = 'sentence-transformers/LaBSE'
tokenizer_checkpoint = 'gpt2'

feature_extractor = pipeline('feature-extraction', framework='pt', model=feature_extractor_checkpoint)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_checkpoint)

def fetch_and_parse(url):
    try:
        response = requests.get(url, timeout=10)

        response.raise_for_status()
    except (requests.HTTPError, requests.ConnectionError, requests.Timeout) as error:
        logging.error(f'Failed to fetch {url}: {error}')

        return None, None

    content = response.text

    markdown = trafilatura.extract(content, output_format='txt', include_formatting=True, \
                                       include_tables=True, include_images=True, no_fallback=True, include_links=True)

    return content, markdown

def embed(text):
    embedding = feature_extractor(text)

    return embedding

def tokenize(text):
    tokens = tokenizer.encode(text)

    return tokens

def process_url(url):
    content, markdown = fetch_and_parse(url)

    content_short = content[:max_embedding_characters]

    tokens = tokenize(content)
    embedding = embed(content_short)

    embedding = np.array(embedding)

    return content, markdown, tokens, embedding

def main():
    url = 'https://huggingface.co'

    content, markdown, tokens, embedding = process_url(url)
    
    for current in [content, markdown, embedding.shape]:
        print(f'{"-" * 32}\n{current}')
    
    print('-' * 32)

if __name__ == '__main__':
    main()
