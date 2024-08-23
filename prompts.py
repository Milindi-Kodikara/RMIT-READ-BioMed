from typing import Any, LiteralString
import pandas as pd
from pandas import DataFrame


def load_prompts(prompt_filepath: str) -> DataFrame:
    return pd.read_json(prompt_filepath)


def format_example(row, cross_lang, task):
    text_append = "Given an example text"
    output_append = "the output is"

    if cross_lang:
        text_append = "Dado un texto de ejemplo"
        output_append = "la salida es"

    if task == 'RE':
        formatted_text_label = "gene\tdisease\trelation"
    else:
        formatted_text_label = "label\tspan"

    return f"{text_append} \"{row['text']}\", {output_append}: \"\n{formatted_text_label}\n{row['combination']}\""


# RE -> R#      relation_type Arg1:mark1 Arg2:mark2
# NER -> T#     label offset1 offset2       span
def create_examples(train_text: DataFrame, train_gold_standard_data: DataFrame, task: str,
                    cross_lang: bool) -> DataFrame:
    if task == 'RE':
        train_gold_standard_data['combination'] = [f"{row['span1']}\t{row['span2']}\t{row['relation_type']}"
                                                   for _, row in train_gold_standard_data.iterrows()]
    else:
        train_gold_standard_data['combination'] = [f"{row['label']}\t{row['span']}" for _, row in
                                                   train_gold_standard_data.iterrows()]

    split_train_gold_standard_data = train_gold_standard_data.loc[:, ['pmid', 'combination']]

    split_train_gold_standard_data = split_train_gold_standard_data.groupby('pmid')['combination'].apply('\n'.join)

    merged_training_data = pd.merge(train_text, split_train_gold_standard_data, on="pmid")

    merged_training_data['combination-en'] = [format_example(row, cross_lang, task) for _, row in
                                              merged_training_data.iterrows()]
    examples_df = merged_training_data.loc[:, ['pmid', 'combination-en']]

    if cross_lang:
        examples_df['combination-es'] = [format_example(row, cross_lang, task) for _, row in
                                         merged_training_data.iterrows()]
    return examples_df


def embed_data_in_prompts(row_data, examples_df, prompts):
    embedded_prompts = []
    pmid = row_data['pmid']
    data_text = row_data['text']

    for index, row_prompt in prompts.iterrows():
        instruction = row_prompt['instruction']
        guideline = row_prompt['guideline']
        no_of_examples = row_prompt['examples']

        if "es" in row_prompt['prompt_id']:
            examples = '\n'.join(examples_df.iloc[: no_of_examples]['combination-es'])
        else:
            examples = '\n'.join(examples_df.iloc[: no_of_examples]['combination-en'])

        expected_output = row_prompt['expected_output']
        prompt_text = row_prompt['text'].format(data_text)

        prompt_structure = [guideline, examples, instruction, expected_output, prompt_text]
        concatenated_prompt = '\n'.join(prompt_structure)

        embedded_prompt = {'prompt_id': row_prompt['prompt_id'], 'prompt': concatenated_prompt}

        embedded_prompts.append(embedded_prompt)

    return {'pmid': pmid, 'prompts': embedded_prompts}


def embed_prompts(
        text: DataFrame,
        train_text: DataFrame,
        train_gold_standard_data: DataFrame,
        prompts: DataFrame,
        task: str,
        cross_lang: bool
) -> list[dict[str, list[dict[str, LiteralString | Any]] | Any]]:
    examples_df = create_examples(train_text, train_gold_standard_data, task, cross_lang)
    embedded_prompts = [embed_data_in_prompts(row_data, examples_df, prompts) for index, row_data in text.iterrows()]

    return embedded_prompts
