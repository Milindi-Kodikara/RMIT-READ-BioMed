import json
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
annotations = os.environ["ANNOTATIONS"]
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
    print('--------------Initialising--------------\n')
    data_cleaner = get_cleaner(dataset_id)
    prompts = load_prompts(prompt_filepath)
    model = get_model(model_id)

    # Clean datasets
    print('--------------CLEANING DATASETS--------------\n')
    train_text, train_gold_standard_data = data_cleaner(train_text_filepath, train_annotation_filepath)

    text, gold_standard_data = data_cleaner(text_filepath, annotation_filepath)

    # Data + Prompts
    print('--------------EMBED PROMPTS--------------\n\n')
    embedded_prompts = embed_prompts(text, train_text, train_gold_standard_data, prompts, task)

    print('--------------RUN MODEL--------------\n\n')
    results = model.get_results(embedded_prompts)
    print(f'Length of results: {len(results)}')

    print('--------------POST PROCESSING--------------\n\n')
    cleaned_entities, hallucinations = result_cleaner(text, results)

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
