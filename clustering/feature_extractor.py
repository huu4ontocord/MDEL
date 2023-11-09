import torch

from transformers import AutoTokenizer, AutoModelForCausalLM


class FeatureExtractor:
    def __init__(self, device='cpu', model_id='bigscience/bloom-560m', num_decoder_blocks=8):
        self.device = device       

        self.num_decoder_blocks = num_decoder_blocks
        self.model_id = model_id

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)

        self.model = AutoModelForCausalLM.from_pretrained(self.model_id)

        h = self.model.transformer.h[:num_decoder_blocks] # Note that this will change for different families of models
        self.model.transformer.h = h

        self.model = self.model.to(device)


    def encode(self, text):
        tokens = self.tokenizer(text, padding=True, return_tensors='pt').to(self.device)
        
        output = self.model(**tokens, output_hidden_states=True).hidden_states[-1]
        output = output.detach().cpu().numpy()

        return output


    def __call__(self, text):
        output = self.encode(text)

        return output


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using {device} device')

    feature_extractor = FeatureExtractor(device=device)

    output = feature_extractor('Hello world!')
    print(output)


if __name__ == '__main__':
    main()
