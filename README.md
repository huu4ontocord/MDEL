# MDEL

Multi-Domain Expert Layers

# Environment Setup

To set up the development environment, run `make setup_dev`. This will setup the
pre-commit hooks.

## Creating Expert Datasets

First, make sure you followed the Environment Setup guidelines.

To create an expert dataset using the Pile data, follow these steps:

1. Download the Pile shard 1 data: `./scripts/get_pile_shard1_data.sh`
2. To set the domain, edit the variable `SUBSET_NAME` in
   `scripts/create_domain_pile_mix.sh`. This should be set to a valid value of
   the Pile's variable `pile_set_name`. A list of valid values can be found below.
3. Run the above script to process the dataset
4. Authenticate into Hugginface:
   `export HF_ACCESS_TOKEN={YOUR HUGGINGFACE TOKEN}`
5. Set the dataset name in `scripts/upload_to_hf.sh`
6. Run the above script to upload the processed dataset to HuggingFace


vicuna dataset. built suite of tools. they struggle as well. Had idea of smmaller baby sitter models. With langchain hijack the agents thoughts and actions to see if it is on track. 

### Subsets

-   Pile-CC
    
-   PubMed Central
    
-   Books3†
    
-   OpenWebText2
    
-   ArXiv
    
-   Github
    
-   FreeLaw
    
-   Stack Exchange
    
-   USPTO Backgrounds
    
-   PubMed Abstracts
    
-   Gutenberg (PG-19)†
    
-   OpenSubtitles†
    
-   Wikipedia (en)†
    
-   DM Mathematics†
    
-   Ubuntu IRC
    
-   BookCorpus2
    
-   EuroParl†
    
-   HackerNews
    
-   YoutubeSubtitles
    
-   PhilPapers
    
-   NIH ExPorter
    
-   Enron Emails†
    
-   The Pile
(Gao et al., 2020, p. 3)

#### References
Gao, L., Biderman, S., Black, S., Golding, L., Hoppe, T., Foster, C., ... & Leahy, C. (2020).The 
pile: An 800gb dataset of diverse text for language modeling. *arXiv preprint arXiv:2101.00027*.
