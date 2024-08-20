from typing import List, Dict, Any, LiteralString
import pandas as pd
from pandas import DataFrame


def load_prompts(prompt_filepath: str) -> DataFrame:
    return pd.read_json(prompt_filepath)


def format_example(row, lang):
    text_append = "Given an example text"
    output_append = "the output is"

    if lang == 'es':
        text_append = "Dado un texto de ejemplo"
        output_append = "la salida es"

    return f"{text_append} \"{row['text']}\", {output_append}: \"\nlabel\tspan\n{row['label-span']}\""


# TODO: ES column based on dataset, maybe cross-ling input from user
def create_examples(train_text: DataFrame, train_gold_standard_data: DataFrame) -> DataFrame:
    train_gold_standard_data['label-span'] = [f"{row['label']}\t{row['span']}" for _, row in
                                              train_gold_standard_data.iterrows()]

    split_train_gold_standard_data = train_gold_standard_data.loc[:, ['pmid', 'label-span']]

    split_train_gold_standard_data = split_train_gold_standard_data.groupby('pmid')['label-span'].apply('\n'.join)

    print(f'Split train gold:\n{split_train_gold_standard_data.head()}\n')

    merged_training_data = pd.merge(train_text, split_train_gold_standard_data, on="pmid")

    merged_training_data['text-label-span-en'] = [format_example(row, 'en') for _, row in
                                                  merged_training_data.iterrows()]

    merged_training_data['text-label-span-es'] = [format_example(row, 'es') for _, row in
                                                  merged_training_data.iterrows()]

    examples_df = merged_training_data.loc[:, ['pmid', 'text-label-span-en', 'text-label-span-es']]

    print(f'examples:\n{examples_df.head()}\n')
    print(f'examples en:\n{examples_df.iloc[0]['text-label-span-en']}\n')
    print(f'examples es:\n{examples_df.iloc[0]['text-label-span-es']}\n')
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
            examples = '\n'.join(examples_df.iloc[: no_of_examples]['text-label-span-es'])
        else:
            examples = '\n'.join(examples_df.iloc[: no_of_examples]['text-label-span-en'])

        expected_output = row_prompt['expected_output']
        prompt_text = row_prompt['text'].format(data_text)

        prompt_structure = [guideline, examples, instruction, expected_output, prompt_text]
        concatenated_prompt = '\n\n'.join(prompt_structure)

        embedded_prompt = {'prompt_id': row_prompt['prompt_id'], 'prompt': concatenated_prompt}

        embedded_prompts.append(embedded_prompt)

    return {'pmid': pmid, 'prompts': embedded_prompts}


def embed_prompts(
        text: DataFrame,
        train_text: DataFrame,
        train_gold_standard_data: DataFrame,
        prompts: DataFrame,
        task: str
) -> list[dict[str, list[dict[str, LiteralString | Any]] | Any]]:
    examples_df = create_examples(train_text, train_gold_standard_data)

    embedded_prompts = [embed_data_in_prompts(row_data, examples_df, prompts) for index, row_data in text.iterrows()]

    print(f'embedded prompt prompt id: {embedded_prompts[0]['prompts'][0]['prompt_id']}\n')
    print(f'embedded prompt prompt:\n{embedded_prompts[0]['prompts'][0]['prompt']}\n')
    return embedded_prompts
