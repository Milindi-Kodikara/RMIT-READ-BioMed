# Shots and hallucinations

In this project, we experiment with a range of prompting strategies for 
genetic information extraction to evaluate the performance, 
and find limitations of using generative technologies.

[Publication TBA]()

## Project overview
Organisation of information about genes, genetic variants, and associated diseases from vast
quantities of scientific literature texts through
automated information extraction (IE) strategies can facilitate progress in personalised
medicine.

We systematically evaluate the performance of
generative large language models (LLMs) on
the extraction of specialised genetic information, focusing on end-to-end IE encompassing
both named entity recognition and relation extraction. We experiment across multilingual 
datasets with a range of instruction strategies, including zero-shot and few-shot 
prompting along with providing an annotation guideline. Optimal results are obtained with
few-shot prompting. However, we also identify that generative LLMs failed to adhere to the 
instructions provided, leading to over-generation of entities and relations. 
We therefore carefully
examine the effect of learning paradigms on
the extent to which genetic entities are fabricated, and the limitations of exact matching to
determine performance of the model.

## Set up
1. Set up the data
2. Create train, dev (evaluation) and test datasets following the below format for each of the datasets.

    - For each dataset create a `<dataset_type>_text.tsv` file and a `<dataset_type>_gold_annotations.tsv`
    - `<dataset_type>_text.tsv` is a TSV file containing the columns `pmid` (PubMed ID of the paper), `filename` (File name of the paper the text is from), `text` (Text that NER task is acted on)
    - `<dataset_type>_gold_annotations.tsv` is a TSV file containing the ground truth/ gold annotations in order to do pairwise comparisons to evaluate the performance of this system. Contains the columns:
      - `pmid`: PubMed ID of the paper
      - `filename`: File name of the paper the text is from
      - `mark`: Annotation ID following the [BRAT format](https://brat.nlplab.org/standoff.html)
      - `label`: Entity label eg: `Disease`
      - `offset1`: Starting index of the span
      - `offset2`: Ending index of the span
      - `span`: Identified entity eg: `Síndrome de Gorlin`


2. [Install Jupyter notebook](https://jupyter.org/install) 


3. [Setting up Azure OpenAI model](https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/working-with-models?tabs=powershell#model-updates)


4. [Setting up connection to GPT-3.5 using Azure OpenAI service](https://learn.microsoft.com/en-us/azure/ai-services/openai/quickstart?tabs=command-line%2Cpython-new&pivots=programming-language-python)
   - In the Environment variables section, instead of doing what is outlined in the link, create a `.env` file in the root folder following the given template in `.env-template`.

5. \[Optional] Add custom prompts to the prompt library in `prompts.json`.
   
##### Note: The number of examples being added should not exceed the number of training texts available. 
    
## Run 
Run the Python program via IDE or via the terminal ` `.

## Evaluation
[Brat-Eval](https://github.com/READ-BioMed/brateval) is the tool we recommend to evaluate this IE pipeline.


## Contributors
Milindi Kodikara

Karin Verspoor

&copy; 2024 Copyright for this project by its contributors.

🧩 READ stands for Reading, Extraction, and Annotation of Documents!
