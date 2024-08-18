import sys
import pandas as pd
import os

from datetime import datetime
from pandas import DataFrame
from dotenv import load_dotenv

from cleaners import get_cleaner
from models import get_model

load_dotenv()

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


def load_prompts() -> DataFrame:
    return pd.read_json(prompt_filepath)


def format_example(row, lang):
    text_append = "Given an example text"
    output_append = "the output is"

    if lang == 'es':
        text_append = "Dado un texto de ejemplo"
        output_append = "la salida es"

    return f"{text_append} \"{row['text']}\", {output_append}: \"\nlabel\tspan\n{row['label-span']}\""


def create_examples(train_text: DataFrame, train_gold_standard_data: DataFrame) -> DataFrame:
    train_gold_standard_data['label-span'] = [f"{row['label']}\t{row['span']}" for _, row in
                                              train_gold_standard_data.iterrows()]

    split_train_gold_standard_data = train_gold_standard_data.loc[:, ['pmid', 'label-span']]

    split_train_gold_standard_data = split_train_gold_standard_data.groupby('pmid')['label-span'].apply('\n'.join)

    merged_training_data = pd.merge(train_text, split_train_gold_standard_data, on="pmid")

    merged_training_data['text-label-span-en'] = [format_example(row, 'en') for _, row in
                                                  merged_training_data.iterrows()]
    merged_training_data['text-label-span-es'] = [format_example(row, 'es') for _, row in
                                                  merged_training_data.iterrows()]

    examples_df = merged_training_data.loc[:, ['pmid', 'text-label-span-en', 'text-label-span-es']]

    return examples_df


def embed_prompts(
        text: DataFrame,
        train_text: DataFrame,
        train_gold_standard_data: DataFrame,
        prompts: DataFrame
) -> DataFrame:
    examples_df = create_examples(train_text, train_gold_standard_data)

    return pd.DataFrame()


class EvaluationData:
    # TODO
    pass


def brat_eval(results: list[str], gold_standard_data: str) -> EvaluationData:
    # TODO
    pass


if __name__ == "__main__":
    data_cleaner = get_cleaner(dataset_id)

    prompts = load_prompts()
    model = get_model(model_id)

    train_text, train_gold_standard_data = data_cleaner(train_text_filepath, train_annotation_filepath)
    text, gold_standard_data = data_cleaner(text_filepath, annotation_filepath)

    embedded_prompts = embed_prompts(text, train_text, train_gold_standard_data, prompts)

    results = model.generate_completions(embedded_prompts, task)

    evaluation = brat_eval(results, gold_standard_data)
