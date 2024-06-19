# RMIT-READ-BioMed
The RMIT University system for NER of genetic entities in biomedical literature for the GenoVarDis shared task at IberLEF 2024.

## Project overview
This is a system developed for the [GenoVarDis](https://codalab.lisn.upsaclay.fr/competitions/17733) shared task at [IberLEF 2024](https://sites.google.com/view/iberlef-2024/home),
focusing on the task of Named Entity Recognition (NER) of genes, genetic variants, and associated diseases from
Spanish-language scientific literature texts.

The approach involves exploration of a general generative Large Language Model (LLM), GPT-3.5, for NER.

We explore the impact of providing English-language instructions with the Spanish-language target text (cross-
linguistic setting) as compared to a within-language setting where the instruction language matches the language
of the text. 

We further experiment with a range of instruction strategies, including zero-shot and few-shot
prompting under these two settings. Results indicate that the optimal results could be obtained with English-
language instructions under the few-shot learning paradigm, resulting in an F1-score of 0.5. While this approach
does not match the top results achieved for the shared task, our experiments provide insight into limitations
associated with simple prompting of LLMs in languages other than English.

## Set up
1. Create train, dev (evaluation) and test datasets following the below format for each of the datasets.

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
   - In the Environment variables section, instead of doing what is outlined in the link, create a `.env` file in the root folder following the given template:
    ```
      ENDPOINT=
      API-KEY=
      API-VERSION=
      DEPLOYMENT-NAME=

      # Eg: ./test_text.tsv
      TEXT-FILE-PATH=
      ANNOTATION-FILE-PATH=

      # DATA-TYPE arg takes in either test or dev
      DATA-TYPE=test

      TRAIN-DATA-TEXT-FILE-PATH=
      TRAIN-DATA-ANNOTATION-FILE-PATH=

      # GENERATE-BRAT-EVAL-ANNOTATIONS can be set to false if don't need BRAT annotations for Brat-Eval
      GENERATE-BRAT-EVAL-ANNOTATIONS=true
    ```


5. \[Optional] Add custom prompts to the prompt library in `prompts.json`.
   
##### Note: The number of examples being added should not exceed the number of training texts available. 
    
## Run 
Run Jupyter notebook via IDE or via local host.

## Evaluation
[Brat-Eval](https://github.com/READ-BioMed/brateval) is the tool we recommend to evaluate this IE pipeline. 

1. Create `helper.sh` script as follows, replace the <FOLDER_PATH>:
````
for filename in ../RMIT-READ-BioMed/results/temp/eval/*.ann; do
  BASE_NAME=$(basename "$filename")
  mv "../RMIT-READ-BioMed/results/temp/eval/$BASE_NAME" "../RMIT-READ-BioMed/results/brateval/eval/$BASE_NAME"
  mv "../RMIT-READ-BioMed/results/temp/gold/$BASE_NAME" "../RMIT-READ-BioMed/results/brateval/gold/$BASE_NAME"

  OUTPUT=$(mvn exec:java -Dexec.mainClass=au.com.nicta.csp.brateval.CompareEntities -Dexec.args="-e <FOLDER_PATH>/RMIT-READ-BioMed/results/brateval/eval -g <FOLDER_PATH>/RMIT-READ-BioMed/results/brateval/gold -s exact" | grep "all|")
  touch output.txt
  printf "%s | %s\n" "$BASE_NAME" "$OUTPUT" >> output.txt

  rm "../RMIT-READ-BioMed/results/brateval/eval/$BASE_NAME"
  rm "../RMIT-READ-BioMed/results/brateval/gold/$BASE_NAME"
done
````

2. Ensure that `RMIT-READ-BioMed`, `brateval` and `data` repos are in the same folder
3. Ensure that `GENERATE-BRAT-EVAL-ANNOTATIONS` in the `.env` file is set to `true`
4. Run the jupyter notebook -> Observe that `results` folder is populated
5. Run `helper.sh`
6. The evaluation results file `output.txt` will be created in the root folder of brateval

## Contributors
Milindi Kodikara

Karin Verspoor

&copy; 2024 Copyright for this project by its contributors.

P.S READ stands for Reading, Extraction, and Annotation of Documents 😏
