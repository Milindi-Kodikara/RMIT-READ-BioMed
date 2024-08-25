import csv
import os
import re
from datetime import datetime
import pandas as pd
import subprocess

eval_pattern = '(?P<prompt_id>.+.ann)(?:\|tp\|fp\|fn\|precision\|recall\|f1\\nall\|)(?P<values>[\w\W|]+)'


def update_evaluation_log(eval_log_filepath, new_eval_df):
    if os.path.isfile(eval_log_filepath):
        eval_log_df = pd.read_csv(eval_log_filepath, sep='\t', header=0)
    else:
        eval_log_df = pd.DataFrame(
            columns=['prompt_id', 'true_positive', 'false_positive', 'false_negative', 'precision',
                     'recall', 'f1', 'hallucination_count', 'entity_count', 'total_result_count', 'date', 'notes'])

    eval_log_df = pd.concat([eval_log_df, new_eval_df], ignore_index=True)

    eval_log_df.to_csv('eval_log.tsv', sep='\t', index=False, header=True)


def create_directory(directory):
    if not os.path.exists(directory):
        os.mkdir(directory)


def save_brat_output(brat, task, df_to_save=None, filename="./results/temp.tsv"):
    formatted_df_to_save = pd.DataFrame()

    if brat:
        if task == 'NER':
            df_to_save["label-offsets"] = df_to_save.apply(
                lambda df_row: f"{df_row['label']} {df_row['offset1']} {df_row['offset2']}", axis=1)

            formatted_df_to_save = df_to_save.loc[:, ['mark', 'label-offsets', 'span']]

        elif task == 'RE':
            # get the ner bits
            df_to_save["formatted_span1"] = df_to_save.apply(
                lambda
                    df_row: f"{df_row['mark1']}\t{df_row['label1']} {df_row['offset1_start']} {df_row['offset1_end']}\t{df_row['span1']}",
                axis=1)

            df_to_save["formatted_span2"] = df_to_save.apply(
                lambda
                    df_row: f"{df_row['mark2']}\t{df_row['label2']} {df_row['offset2_start']} {df_row['offset2_end']}\t{df_row['span2']}",
                axis=1)

            df_to_save["formatted_relation"] = df_to_save.apply(
                lambda
                    df_row: f"{df_row['relation_mark']}\t{df_row['relation_type']} Arg1:{df_row['mark1']} Arg2:{df_row['mark2']}",
                axis=1)

            formatted_df_to_save = pd.concat([df_to_save['formatted_span1'].rename('formatted'),
                                             df_to_save['formatted_span2'].rename('formatted'),
                                              df_to_save['formatted_relation'].rename('formatted')],
                                             ignore_index=True, axis=0)

        formatted_df_to_save.to_csv(filename, index=False, header=False)

    if not brat:
        df_to_save.to_csv(f"{filename}.tsv", sep='\t', index=False, header=True)


def brat_eval(task, eval_log_filepath, generate_brat_eval_annotations, prompts, cleaned_entities, hallucinations,
              gold_standard_data,
              brat_eval_filepath,
              root_folder_filepath):
    create_directory('./results')
    create_directory('./results/ordered_by_prompts')
    create_directory('./results/hallucinations')
    create_directory('./results/temp')
    create_directory('./results/brateval')
    create_directory('./results/temp/gold')
    create_directory('./results/brateval/gold')
    create_directory('./results/temp/eval')
    create_directory('./results/brateval/eval')

    if generate_brat_eval_annotations:
        for _, prompt in prompts.iterrows():
            prompt_id = prompt['prompt_id']
            gold_annotations_filename = f'results/temp/gold/{prompt_id}.ann'
            save_brat_output(True, task, gold_standard_data, gold_annotations_filename)

    for _, prompt in prompts.iterrows():
        prompt_id = prompt['prompt_id']
        results_subset = cleaned_entities[(cleaned_entities['prompt_id'] == prompt_id)]

        # Save results in BRAT format
        if generate_brat_eval_annotations:
            results_brat_filename = f'results/temp/eval/{prompt_id}.ann'
            save_brat_output(True, task, results_subset, results_brat_filename)

        # Save whole result output
        results_filename = f'results/ordered_by_prompts/{prompt_id}'
        save_brat_output(False, task, results_subset, results_filename)

    evaluation_script_output = subprocess.check_output(
        ['sh', './evaluation.sh', brat_eval_filepath, root_folder_filepath])
    evaluation_script_output_decoded = evaluation_script_output.decode("utf-8").split("::")

    evaluation_values = pd.DataFrame(
        columns=['prompt_id', 'true_positive', 'false_positive', 'false_negative', 'precision',
                 'recall', 'f1', 'hallucination_count', 'entity_count', 'total_result_count', 'date', 'notes'])

    len_cleaned_entities = len(cleaned_entities)
    len_hallucinated_entities = len(hallucinations)
    total_entities = len_cleaned_entities + len(hallucinations)
    date = datetime.today().strftime('%Y-%m-%d %H:%M:%S')

    for result in evaluation_script_output_decoded:
        stripped_result = result.strip()
        matches = re.search(eval_pattern, stripped_result)

        if matches:
            prompt_id = matches.group("prompt_id").strip()
            true_positive, false_positive, false_negative, precision, recall, f1 = matches.group(
                "values").strip().split("|")

            evaluation_values = pd.concat([evaluation_values, pd.DataFrame(
                [{'prompt_id': prompt_id, 'true_positive': true_positive, 'false_positive': false_positive,
                  'false_negative': false_negative, 'precision': precision,
                  'recall': recall, 'f1': f1, 'hallucination_count': len_hallucinated_entities,
                  'entity_count': len_cleaned_entities, 'total_result_count': total_entities, 'date': date,
                  'notes': ''},
                 ])], ignore_index=True)

    update_evaluation_log(eval_log_filepath, evaluation_values)

    return evaluation_values
