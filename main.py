import json
import logging
import os
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

pd.options.display.max_columns = None
pd.options.display.max_rows = None

from cleaners import get_cleaner
from models import get_model
from prompts import *
from result_cleaner import *
from evaluation import *

# Inputs
# Model
model_id = os.environ["MODEL-ID"]

# Task
task = os.environ["TASK"]
prompt_filepath = os.environ["PROMPT-FILEPATH"]

# Dataset
dataset_id = os.environ["DATASET-ID"]

text_filepath = os.environ["TEXT-FILEPATH"]
annotation_filepath = os.environ["ANNOTATION-FILEPATH"]

train_text_filepath = os.environ["TRAIN-TEXT-FILEPATH"]
train_annotation_filepath = os.environ["TRAIN-ANNOTATION-FILEPATH"]

# Evaluation
generate_brat_eval_annotations = os.environ["GENERATE-BRAT-EVAL-ANNOTATIONS"]
brat_eval_filepath = os.environ["BRAT-EVAL-FILEPATH"]
root_folder_filepath = os.environ["ROOT-FOLDER-FILEPATH"]
eval_log_filepath = os.environ["EVAL-FILEPATH"]

if __name__ == "__main__":
    # Initialisation
    logging.info("Initialising.")
    data_cleaner = get_cleaner(dataset_id)
    logging.info("Data cleaned.")

    prompts = load_prompts(prompt_filepath)

    logging.info("Prompts loaded.")

    model = get_model(model_id)

    # Clean datasets
    print('--------------CLEANING DATASETS--------------')
    train_text, train_gold_standard_data = data_cleaner(train_text_filepath, train_annotation_filepath)
    print(f"Training text data len: {len(train_text)}, Gold len: {len(train_gold_standard_data)}")
    print(f'training text head:\n{train_text.head()}\n')
    print(f'training gold head:\n{train_gold_standard_data.head()}\n')

    text, gold_standard_data = data_cleaner(text_filepath, annotation_filepath)
    print(f"Text data len: {len(text)}, Gold len: {len(gold_standard_data)}")
    print(f'text head:\n{text.head(1)}\n')
    print(f'gold head:\n{gold_standard_data.head()}\n')

    # Data + Prompts
    print('--------------EMBED PROMPTS--------------\n\n')
    embedded_prompts = embed_prompts(text, train_text, train_gold_standard_data, prompts, task)

    print('--------------RUN MODEL--------------\n\n')
    results = model.get_results(embedded_prompts)
    print(f'Length of results: {len(results)}')

    print('--------------POST PROCESSING--------------\n\n')
    cleaned_entities, hallucinations = result_cleaner(text, results)
    print(f"Cleaned entities len: {len(cleaned_entities)}\n{cleaned_entities.head()}\n\n")
    print(f"Hallucinated entities len: {len(hallucinations)}\n{hallucinations.head()}\n\n")

    # TODO: Clean up evaluation
    print('--------------EVALUATION--------------\n\n')
    evaluation_values = brat_eval(eval_log_filepath, generate_brat_eval_annotations, prompts, cleaned_entities,
                                  hallucinations, gold_standard_data, brat_eval_filepath, root_folder_filepath)

    for _, prompt in prompts.iterrows():
        prompt_id = prompt['prompt_id']
        hallucinated_results_subset = hallucinations[(hallucinations['prompt_id'] == prompt_id)]

        formatted_hallucinated_results = hallucinated_results_subset.loc[:, ['pmid', 'label', 'span', 'offset1',
                                                                             'offset2']]

        filename = f'results/hallucinations/{prompt_id}_hallucinations.tsv'
        formatted_hallucinated_results.to_csv(filename, sep='\t', index=False)

    # TODO: Analysis
