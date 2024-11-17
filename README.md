# RMIT READ-BioMed

In this project, we experiment with a range of prompting strategies for 
genetic information extraction to evaluate the performance, 
and find limitations of using generative technologies.

## List of Publications

[Milindi Kodikara and Karin M. Verspoor. 2024. Effectiveness of Cross-Linguistic Extraction of Genetic Information using
Generative Large Language Models. In Proceedings of the Iberian Languages Evaluation Forum (IberLEF 2024).](https://ceur-ws.org/Vol-3756/GenoVarDis2024_paper4.pdf)

Milindi Kodikara and Karin M. Verspoor. 2024 (to appear). Lesser the Shots, Higher the Hallucinations: Exploration of Genetic
Information Extraction using Generative Large Language Models. In Proceedings of the Australian Language Technology
Association (ALTA 2024).


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

### Set up
1. Download the datasets for IE tasks 
2. Create train, and test datasets following the below format for each of the datasets.

    - For each dataset create a `<dataset_type>_text.tsv` file and a `<dataset_type>_gold_annotations.tsv`
    - `<dataset_type>_text.tsv` is a TSV file containing the columns `pmid` (ID of the paper), `text` (Text from literature)
    - `<dataset_type>_gold_annotations.tsv` is a TSV file containing the ground truth/ gold annotations in order to do pairwise comparisons to evaluate the performance of this system. Contains the below columns.
      - For Named Entity Recognition (NER):
        - `pmid`: PubMed ID of the paper
        - `filename`: File name of the paper the text is from
        - `mark`: Annotation ID following the [BRAT format](https://brat.nlplab.org/standoff.html)
        - `label`: Entity label eg: `Disease`
        - `offset1`: Starting index of the span
        - `offset2`: Ending index of the span
        - `span`: Identified entity eg: `Síndrome de Gorlin`
        
      - For Relation Extraction (RE) or join NER and RE (NERRE):
        - `pmid`: PubMed ID of the paper
        - `filename`: File name of the paper the text is from
        - `mark1`: Annotation ID for first entity following the [BRAT format](https://brat.nlplab.org/standoff.html)
        - `label1`: First entity label eg: `Gene`
        - `offset1_start`: Starting index of the first span
        - `offset1_end`: Ending index of the first span
        - `span1`: First entity identified eg: `DUSP6`
        - `mark2`: Annotation ID for second entity following the [BRAT format](https://brat.nlplab.org/standoff.html)
        - `label2`: Second entity label eg: `Disease`
        - `offset2_start`: Starting index of the second span
        - `offset2_end`: Ending index of the second span
        - `span2`: Second entity identified eg: `Mood Disorders`
        - `relation_mark`: ID for the relation identified
        - `relation_type`: Relation type to annotate eg: `biomarker`
        
    - Alternatively: If the datasets are either one of [_GenoVarDis_](https://codalab.lisn.upsaclay.fr/competitions/17733), [_TBGA_](https://zenodo.org/records/5911097) or [_Variome_](https://bitbucket.org/readbiomed/variome-corpus-data/src/master/) the data can be cleaned and pre-processed once `CLEAN-DATA=true` in the `.env` file. 


2. [Install Jupyter notebook](https://jupyter.org/install) 


3. Set up models
   - Currently supported models are:
     - GPT-3.5 Turbo, model id: `gpt-35-turbo-16k`
     - Llama 3 70b Instruct, model id: `meta.llama3-70b-instruct-v1:0`
   - Using Azure OpenAI
     - [Setting up Azure OpenAI model](https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/working-with-models?tabs=powershell#model-updates)
     - [Setting up connection to GPT-3.5 Turbo using Azure OpenAI service](https://learn.microsoft.com/en-us/azure/ai-services/openai/quickstart?tabs=command-line%2Cpython-new&pivots=programming-language-python)
       
     Note: The environment variables should be set inside the `.env` file. 
     
   - Using [Amazon Bedrock](https://docs.aws.amazon.com/bedrock/latest/userguide/what-is-bedrock.html)
     - [Getting started with Amazon Bedrock](https://docs.aws.amazon.com/bedrock/latest/userguide/getting-started.html)
     - [Install AWS CLI](https://aws.amazon.com/cli/)
     - [Configure SSO for authentication](https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-sso.html)
       - `aws configure sso`
       - `aws sso login --profile <PROFILE-NAME>`

4. Duplicate the `.env-template` file as `.env` and populate according to the _task_ and _model_.

5. \[Optional] Add custom prompts to the matching prompt library file: `<task>_prompts.json`.
6. \[Optional] To add other models, update `models.py` file by creating the corresponding model class similar to class `GPTModel`.
   
##### Note: The number of examples being added should not exceed the number of training texts available. 
    
### Run 
Run the Python program via IDE `python main.py`.

### Evaluation
[Brat-Eval](https://github.com/READ-BioMed/brateval) is the tool we have used for evaluation. 

A summary of the datasets, 
extracted instances, hallucinated instances, visualisation of results, and performance details 
will be generated in `<RESULT-FOLDER-PATH>/results` once the program has finished running.

## Releases
[GenoVarDis 2024](https://github.com/Milindi-Kodikara/RMIT-READ-BioMed-Version-2.0/releases/tag/v1.0)

[ALTA 2024](https://github.com/Milindi-Kodikara/RMIT-READ-BioMed-Version-2.0/releases/tag/v2.0)

## Contributors
Milindi Kodikara

Karin Verspoor

&copy; 2024 Copyright for this project by its contributors.

🧩 READ stands for Reading, Extraction, and Annotation of Documents!
