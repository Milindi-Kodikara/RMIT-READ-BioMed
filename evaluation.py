import csv
import math
import os
import re
from datetime import datetime

import numpy as np
import pandas as pd
import subprocess

eval_pattern_ner = r'(?P<prompt_id>.+)\.ann(?:\|tp\|fp\|fn\|precision\|recall\|f1\nall\|)(?P<values>[\w\W|]+)'
eval_pattern_re = r'(?P<prompt_id>[\w\_]+)\.annall(?P<values>((?:\|\w+:)(\d+\.?\d*)+)+)'
relation_pattern = r'(?:\|\w+:)(\d+\.?\d*)'


def update_evaluation_log(eval_log_filepath, new_eval_df):
    if os.path.isfile(eval_log_filepath):
        eval_log_df = pd.read_csv(eval_log_filepath, sep='\t', header=0)
    else:
        eval_log_df = pd.DataFrame(
            columns=['prompt_id', 'task', 'true_positive', 'false_positive', 'false_negative',
                     'false_positive_relations', 'false_negative_relations', 'precision',
                     'recall', 'f1', 'tuple_or_triplet_hallucinations_per_prompt',
                     'total_tuple_or_triplet_hallucinations',
                     'extracted_tuples_or_triplets_per_prompt',
                     'total_tuple_or_triplet_extractions', 'combined_total_extractions_and_hallucinations_per_prompt',
                     'combined_total_extractions_and_hallucinations', 'date', 'notes'])

    eval_log_df = pd.concat([eval_log_df, new_eval_df], ignore_index=True)

    eval_log_df.to_csv(eval_log_filepath, sep='\t', index=False, header=True)


def compute_dataset_details(dataset_id, task, train_text, train_gold_standard_data, test_text, test_gold_standard_data):
    train_gold_entity_count = 0
    train_gold_relation_count = 0
    test_gold_entity_count = 0
    test_gold_relation_count = 0

    average_train_text_size = np.mean(
        train_text['text'].apply(lambda x: len([words for words in x.split(" ") if isinstance(x, str)])))
    average_test_text_size = np.mean(
        test_text['text'].apply(lambda x: len([words for words in x.split(" ") if isinstance(x, str)])))
    average_text_size = math.ceil((average_test_text_size + average_train_text_size) / 2)

    gold_annotation_types_train, annotation_counts_train = [], []
    gold_relation_types_train, relation_counts_train = [], []

    gold_annotation_types_test, annotation_counts_test = [], []
    gold_relation_types_test, relation_counts_test = [], []

    if task == 'NER':
        train_gold_entity_count = len(train_gold_standard_data)
        test_gold_entity_count = len(test_gold_standard_data)

        train_label = train_gold_standard_data['label'].values.tolist()
        gold_annotation_types_train, annotation_counts_train = np.unique(train_label, return_counts=True)

        test_label = test_gold_standard_data['label'].values.tolist()
        gold_annotation_types_test, annotation_counts_test = np.unique(test_label, return_counts=True)
    elif task == 'RE' or task == 'NERRE':
        # multiplied by 2 because the entities depicting the relation are in tuples
        train_gold_entity_count = len(train_gold_standard_data) * 2
        test_gold_entity_count = len(test_gold_standard_data) * 2

        train_gold_relation_count = len(train_gold_standard_data)
        test_gold_relation_count = len(test_gold_standard_data)

        train_label1 = train_gold_standard_data['label1'].values.tolist()
        train_label2 = train_gold_standard_data['label2'].values.tolist()
        train_label_array = train_label1 + train_label2
        gold_annotation_types_train, annotation_counts_train = np.unique(train_label_array, return_counts=True)
        gold_relation_types_train, relation_counts_train = np.unique(train_gold_standard_data['relation_type'],
                                                                     return_counts=True)

        test_label1 = test_gold_standard_data['label1'].values.tolist()
        test_label2 = test_gold_standard_data['label2'].values.tolist()
        test_label_array = test_label1 + test_label2
        gold_annotation_types_test, annotation_counts_test = np.unique(test_label_array, return_counts=True)
        gold_relation_types_test, relation_counts_test = np.unique(test_gold_standard_data['relation_type'],
                                                                   return_counts=True)

    dataset_details_df = pd.DataFrame({'dataset_id': [dataset_id],
                                       'task': [task],
                                       'train_text_count': [len(train_text)],
                                       'train_gold_entity_count': [train_gold_entity_count],
                                       'train_gold_relation_count': [train_gold_relation_count],
                                       'test_text_count': [len(test_text)],
                                       'test_gold_entity_count': [test_gold_entity_count],
                                       'test_gold_relation_count': [test_gold_relation_count],
                                       'total_text_count': [(len(train_text) + len(test_text))],
                                       'total_gold_annotations_count': [
                                           train_gold_entity_count + train_gold_relation_count +
                                           test_gold_entity_count + test_gold_relation_count],
                                       'average_text_size': [average_text_size]})

    gold_annotations_type_count_df = pd.DataFrame(columns=['dataset_id', 'task', 'dataset_type', 'gold_annotation_type',
                                                           'count'])
    gold_annotations_type_count_df['dataset_id'] = [dataset_id, dataset_id, dataset_id, dataset_id]
    gold_annotations_type_count_df['task'] = [task, task, task, task]
    gold_annotations_type_count_df['dataset_type'] = ['train_entities', 'test_entities', 'train_relations',
                                                      'test_relations']
    gold_annotations_type_count_df['gold_annotation_type'] = [":".join(gold_annotation_types_train),
                                                              ":".join(gold_annotation_types_test),
                                                              ":".join(gold_relation_types_train),
                                                              ":".join(gold_relation_types_test)]
    gold_annotations_type_count_df['count'] = [':'.join(str(num) for num in annotation_counts_train),
                                               ':'.join(str(num) for num in annotation_counts_test),
                                               ':'.join(str(num) for num in relation_counts_train),
                                               ':'.join(str(num) for num in relation_counts_test)]
    save_dataset_details(dataset_details_df, gold_annotations_type_count_df)


def save_dataset_details(new_dataset_details_df, new_gold_annotations_type_count_df):
    dataset_details_filename = './results/dataset_details/dataset_details.tsv'
    gold_annotation_type_count_filename = f'./results/dataset_details/gold_annotation_type_count.tsv'

    if os.path.isfile(dataset_details_filename) and os.path.isfile(gold_annotation_type_count_filename):
        dataset_details_df = pd.read_csv(dataset_details_filename, sep='\t', header=0)
        gold_annotations_type_count_df = pd.read_csv(gold_annotation_type_count_filename, sep='\t', header=0)
    else:
        dataset_details_df = pd.DataFrame(
            columns=['dataset_id', 'task', 'train_text_count', 'train_gold_entity_count', 'train_gold_relation_count',
                     'test_text_count', 'test_gold_entity_count', 'test_gold_relation_count', 'total_text_count',
                     'total_gold_annotations_count', 'average_text_size'])
        gold_annotations_type_count_df = pd.DataFrame(columns=['dataset_id', 'task', 'dataset_type',
                                                               'gold_annotation_type', 'count'])

    dataset_details_df = pd.concat([dataset_details_df, new_dataset_details_df], ignore_index=True)
    gold_annotations_type_count_df = pd.concat([gold_annotations_type_count_df, new_gold_annotations_type_count_df],
                                               ignore_index=True)

    dataset_details_df.to_csv(dataset_details_filename, sep='\t', index=False, header=True)
    gold_annotations_type_count_df.to_csv(gold_annotation_type_count_filename, sep='\t', index=False, header=True)


def create_directory(directory):
    if not os.path.exists(directory):
        os.mkdir(directory)


def save_brat_output(brat, task, df_to_save=None, filename="./results/temp.tsv"):
    if brat:
        if task == 'NER':
            df_to_save["label-offsets"] = df_to_save.apply(
                lambda df_row: f"{df_row['label']} {df_row['offset1']} {df_row['offset2']}", axis=1)

            if 'mark' not in df_to_save.columns:
                df_to_save["mark"] = df_to_save.apply(lambda df_row: f"T{df_row.name + 1}", axis=1)

            formatted_df_to_save = df_to_save.loc[:, ['mark', 'label-offsets', 'span']]
            formatted_df_to_save.to_csv(filename, sep='\t', index=False, header=False)

        elif task == 'RE' or task == 'NERRE':
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

            np.savetxt(filename, formatted_df_to_save, fmt='%s')

    if not brat:
        df_to_save.to_csv(f"{filename}.tsv", sep='\t', index=False, header=True)


def save_hallucinations(prompts, hallucinations):
    for _, prompt in prompts.iterrows():
        prompt_id = prompt['prompt_id']
        hallucinated_results_subset = hallucinations[(hallucinations['prompt_id'] == prompt_id)]

        filename = f'results/hallucinations/{prompt_id}_hallucinations.tsv'
        hallucinated_results_subset.to_csv(filename, sep='\t', index=False)


def evaluate(task, eval_log_filepath, generate_brat_eval_annotations, prompts, cleaned_entities, hallucinations,
             gold_standard_data,
             brat_eval_filepath,
             root_folder_filepath, note):
    create_directory('./results')
    create_directory('./results/entities')
    create_directory('./results/dataset_details')
    create_directory('./results/ordered_by_prompts')
    create_directory('./results/hallucinations')
    create_directory('./results/temp')
    create_directory('./results/brateval')
    create_directory('./results/temp/gold')
    create_directory('./results/brateval/gold')
    create_directory('./results/temp/eval')
    create_directory('./results/brateval/eval')

    if generate_brat_eval_annotations:
        if task == 'NER':
            gold_standard_data = gold_standard_data.drop(['mark'], axis=1)

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
    results_filename = f'./results/entities/{task}_results'
    save_brat_output(False, task, cleaned_entities, results_filename)

    gold_filename = f'./results/entities/{task}_gold'
    save_brat_output(False, task, gold_standard_data, gold_filename)

    is_ner = 'true' if task == 'NER' else 'false'
    evaluation_script_output = subprocess.check_output(
        ['sh', './evaluation.sh', brat_eval_filepath, root_folder_filepath, is_ner])
    evaluation_script_output_decoded = evaluation_script_output.decode("utf-8").split("::")

    evaluation_values = pd.DataFrame(
        columns=['prompt_id', 'task', 'true_positive', 'false_positive', 'false_negative',
                 'false_positive_relations', 'false_negative_relations', 'precision',
                 'recall', 'f1', 'tuple_or_triplet_hallucinations_per_prompt', 'total_tuple_or_triplet_hallucinations',
                 'extracted_tuples_or_triplets_per_prompt',
                 'total_tuple_or_triplet_extractions', 'combined_total_extractions_and_hallucinations_per_prompt',
                 'combined_total_extractions_and_hallucinations', 'date', 'notes'])

    # for NER -> type, entity; RE and NERRE -> types, entities and relation
    total_tuple_or_triplet_extractions = len(cleaned_entities)
    total_tuple_or_triplet_hallucinations = len(hallucinations)
    combined_total_extractions_and_hallucinations = (total_tuple_or_triplet_extractions +
                                                     total_tuple_or_triplet_hallucinations)
    date = datetime.today().strftime('%Y-%m-%d %H:%M:%S')

    for result in evaluation_script_output_decoded:
        stripped_result = result.strip()
        matches = re.search(eval_pattern_ner, stripped_result) if task == 'NER' else re.search(eval_pattern_re,
                                                                                               stripped_result)
        if matches:
            prompt_id = matches.group("prompt_id").strip()
            false_positive_relations = ''
            false_negative_relations = ''

            if task == 'NER':
                true_positive, false_positive, false_negative, precision, recall, f1 = matches.group(
                    "values").strip().split("|")

            if not task == 'NER':
                relations = matches.group("values")

                (true_positive, false_positive, false_negative, precision,
                 recall, f1, false_positive_relations, false_negative_relations) = re.findall(relation_pattern,
                                                                                              relations)

            formatted_prompt_id = prompt_id.replace('.ann', '')
            extracted_tuples_or_triplets_per_prompt = len(cleaned_entities.loc[cleaned_entities['prompt_id'] ==
                                                                               formatted_prompt_id])
            tuple_or_triplet_hallucinations_per_prompt = len(hallucinations.loc[hallucinations['prompt_id'] ==
                                                                                formatted_prompt_id])
            combined_total_extractions_and_hallucinations_per_prompt = extracted_tuples_or_triplets_per_prompt + tuple_or_triplet_hallucinations_per_prompt

            evaluation_values = pd.concat([evaluation_values, pd.DataFrame(
                [{'prompt_id': prompt_id,
                  'task': task,
                  'true_positive': true_positive,
                  'false_positive': false_positive,
                  'false_negative': false_negative,
                  'false_positive_relations': false_positive_relations,
                  'false_negative_relations': false_negative_relations,
                  'precision': precision,
                  'recall': recall, 'f1': f1, 'tuple_or_triplet_hallucinations_per_prompt':
                      tuple_or_triplet_hallucinations_per_prompt,
                  'total_tuple_or_triplet_hallucinations': total_tuple_or_triplet_hallucinations,
                  'extracted_tuples_or_triplets_per_prompt': extracted_tuples_or_triplets_per_prompt,
                  'total_tuple_or_triplet_extractions': total_tuple_or_triplet_extractions,
                  'combined_total_extractions_and_hallucinations_per_prompt':
                      combined_total_extractions_and_hallucinations_per_prompt,
                  'combined_total_extractions_and_hallucinations': combined_total_extractions_and_hallucinations,
                  'date': date,
                  'notes': note
                  },
                 ])], ignore_index=True)

    update_evaluation_log(eval_log_filepath, evaluation_values)
    save_hallucinations(prompts, hallucinations)

    return evaluation_values
