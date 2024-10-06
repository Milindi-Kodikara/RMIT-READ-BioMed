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
from analysis import *

# Inputs
# Model
model_id = os.environ["MODEL-ID"]

# Task
task = os.environ["TASK"]
ner_annotations = os.environ["NER-ANNOTATIONS"]
re_annotations = os.environ["RE-ANNOTATIONS"]
cross_lang = os.environ["CROSS-LANG"]
if cross_lang == 'true':
    cross_lang = True
else:
    cross_lang = False

prompt_filepath = os.environ["PROMPT-FILEPATH"]

# Dataset
dataset_id = os.environ["DATASET-ID"]
clean_data = os.environ["CLEAN-DATA"]

text_filepath = os.environ["TEXT-FILEPATH"]
annotation_filepath = os.environ["ANNOTATION-FILEPATH"]

train_text_filepath = os.environ["TRAIN-TEXT-FILEPATH"]
train_annotation_filepath = os.environ["TRAIN-ANNOTATION-FILEPATH"]

# Evaluation
generate_brat_eval_annotations = os.environ["GENERATE-BRAT-EVAL-ANNOTATIONS"]
brat_eval_filepath = os.environ["BRAT-EVAL-FILEPATH"]
root_folder_filepath = os.environ["ROOT-FOLDER-FILEPATH"]
result_folder_path = os.environ["RESULT-FOLDER-PATH"]
note = os.environ["NOTE"]


def load_data_files(text_filepath, annotation_filepath):
    text_df = pd.read_csv(text_filepath, sep='\t', header=0)
    annotated_data_df = pd.read_csv(annotation_filepath, sep='\t', header=0)

    return text_df, annotated_data_df


if __name__ == "__main__":
    # Initialisation
    print('--------------INITIALISING--------------\n\n')
    text, gold_standard_data, train_text, train_gold_standard_data = (pd.DataFrame(), pd.DataFrame(), pd.DataFrame(),
                                                                      pd.DataFrame())

    if clean_data:
        print('--------------PRE-PROCESSING DATASETS--------------\n\n')
        data_cleaner = get_cleaner(dataset_id)
        # Clean datasets
        train_text, train_gold_standard_data = data_cleaner(train_text_filepath, train_annotation_filepath)
        text, gold_standard_data = data_cleaner(text_filepath, annotation_filepath)

    if not clean_data:
        train_text, train_gold_standard_data = load_data_files(train_text_filepath, train_annotation_filepath)
        text, gold_standard_data = load_data_files(text_filepath, annotation_filepath)

    text = text.head(2)
    prompts = load_prompts(prompt_filepath)
    model = get_model(model_id)

    print('--------------EMBED PROMPTS--------------\n\n')
    # Data + Prompts
    embedded_prompts = embed_prompts(text, train_text, train_gold_standard_data, prompts, task, cross_lang)

    print('--------------RUN MODEL--------------\n\n')
    results = model.get_results(embedded_prompts)

    print('--------------POST-PROCESSING--------------\n\n')
    cleaned_entities, hallucinations = result_cleaner(text, results, ner_annotations, re_annotations, task)

    print('--------------EVALUATION--------------\n\n')
    evaluation_values = evaluate(task, result_folder_path, generate_brat_eval_annotations, prompts, cleaned_entities,
                                 hallucinations, gold_standard_data, brat_eval_filepath, note)
    compute_dataset_details(result_folder_path, dataset_id, task, train_text, train_gold_standard_data, text,
                            gold_standard_data)

    print('--------------ANALYSIS--------------\n\n')
    analysis(root_folder_filepath, result_folder_path)

    print('--------------DONE!--------------\n')
    print(f'Result evaluation can be found in {result_folder_path}/results.')
