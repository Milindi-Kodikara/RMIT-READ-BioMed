"""
All cleaners should have the same interface

(file input, task) -> source text[], gold standard data[]
which is the same as
Callable[[str, ...], (list[str], str)]
"""
import ast
from pathlib import Path
from typing import Callable
import pandas as pd
import re

from pandas import DataFrame

# The python types for a cleaner function.
type CleanerFnResponse = (DataFrame, DataFrame)
type CleanerFunction = Callable[[str, str], CleanerFnResponse]

pattern = '(?:[\d]{1,10}\|t\|)(?P<title>[\w\W]+)(?:\\n[\d]{1,20}\|a\|)(?P<abstract>[\w\W]+)'


def clean_genovardis_text(text):
    matches = re.search(pattern, text)
    reformatted_text = f'{matches.group("title")}\n{matches.group("abstract")}'
    return reformatted_text


def cleaner_genovardis(text_filepath: str, annotation_filepath: str = '') -> CleanerFnResponse:
    text_data = pd.read_csv(text_filepath, sep='\t', header=0)
    annotated_data = pd.read_csv(annotation_filepath, sep='\t', header=0)

    # TODO: Remove this maybe
    # text_data['text'] = [clean_genovardis_text(text) for text in text_data['text']]
    text_data = text_data.drop(columns=['filename'], axis=1)
    annotated_data = annotated_data.drop(columns=['filename'], axis=1)
    return text_data, annotated_data


def cleaner_variome(text_filepath: str, annotation_filepath: str = '') -> CleanerFnResponse:
    # return text, gold standard data
    parent_dir = Path(text_filepath)

    text_files = parent_dir.glob('*.txt')
    annotation_files = parent_dir.glob('*.ann')

    text_df = pd.DataFrame(columns=['pmid', 'text'])
    annotation_df = pd.DataFrame(columns=['pmid', 'mark1', 'label1', 'offset1_start', 'offset1_end', 'span1',
                                          'mark2', 'label2', 'offset2_start', 'offset2_end', 'span2',
                                          'relation_mark', 'relation_type'])

    for text_file in text_files:
        with open(text_file, 'r') as f:
            pmid = text_file.stem
            text = f.read()
            text_df.loc[len(text_df.index)] = [pmid, text]

    for annotation_file in annotation_files:
        with open(annotation_file, 'r') as f:
            pmid = annotation_file.stem
            data_list = f.read().split('\n')

            ner_data = pd.DataFrame(columns=['mark', 'label', 'offset1', 'offset2', 'span'])
            re_data = pd.DataFrame(columns=['relation_mark', 'relation_type', 'mark1', 'mark2'])
            for data_line in data_list:
                data_line_items = data_line.split('\t')
                mark = data_line_items[0]
                if mark.startswith('T'):
                    label_offsets = data_line_items[1]
                    label_offsets = label_offsets.split()
                    label = label_offsets[0]
                    offset1 = int(label_offsets[1])
                    offset2 = int(label_offsets[2])
                    span = data_line_items[2]
                    ner_data.loc[len(ner_data.index)] = [mark, label, offset1, offset2, span]
                elif mark.startswith('R'):
                    relation_triplet = data_line_items[1].split()
                    relation_type = relation_triplet[0]
                    mark1 = relation_triplet[1].replace('Arg1:', '')
                    mark2 = relation_triplet[2].replace('Arg2:', '')
                    re_data.loc[len(re_data.index)] = [mark, relation_type, mark1, mark2]

            for row in re_data.itertuples():
                relation_mark = 'R_' + str(len(annotation_df.index))
                relation_type = row.relation_type

                re_mark1 = row.mark1
                re_mark2 = row.mark2

                mark1 = 'T_1_' + str(len(annotation_df.index))
                label1 = ner_data.loc[ner_data['mark'] == re_mark1, 'label'].item()
                span1 = ner_data.loc[ner_data['mark'] == re_mark1, 'span'].item()
                offset1_start = ner_data.loc[ner_data['mark'] == re_mark1, 'offset1'].item()
                offset1_end = ner_data.loc[ner_data['mark'] == re_mark1, 'offset2'].item()

                mark2 = 'T_2_' + str(len(annotation_df.index))
                label2 = ner_data.loc[ner_data['mark'] == re_mark2, 'label'].item()
                span2 = ner_data.loc[ner_data['mark'] == re_mark2, 'span'].item()
                offset2_start = ner_data.loc[ner_data['mark'] == re_mark2, 'offset1'].item()
                offset2_end = ner_data.loc[ner_data['mark'] == re_mark2, 'offset2'].item()

                annotation_df.loc[len(annotation_df.index)] = [pmid, mark1, label1, offset1_start, offset1_end, span1,
                                                               mark2, label2, offset2_start, offset2_end, span2,
                                                               relation_mark, relation_type]

    return text_df, annotation_df


def cleaner_tbga(text_filepath: str, annotation_filepath: str = '') -> CleanerFnResponse:
    # return text, gold standard data
    with open(text_filepath, 'r') as f:
        test_data_list = f.read().split('\n')
        result = [ast.literal_eval('{%s}' % item[1:-1]) for item in test_data_list]
        df = pd.json_normalize(result)

        df['pmid'] = df.index
        # create text_df
        text = df.loc[:, ['pmid', 'text']]

        # create gold_df
        # pmid
        # mark1 label1 span1 offset1_start offset1_end
        # mark2 label2 span2 offset2_start offset2_end
        # relation_mark relation_type
        df['mark1'] = 'T' + df.index.astype(str) + '_gene'
        df['label1'] = 'Gene'

        df['mark2'] = 'T' + df.index.astype(str) + '_disease'
        df['label2'] = 'Disease'

        df["relation_mark"] = 'R' + df.index.astype(str)

        df = df.rename(columns={"relation": "relation_type", "h.name": "span1",
                                "h.pos": "span1_pos_len", "t.name": "span2", "t.pos": "span2_pos_len"})

        df = df.dropna()
        df['offset1_start'] = 0
        df['offset1_end'] = 0
        df['offset2_start'] = 0
        df['offset2_end'] = 0

        df['span1'] = [row.text[row.span1_pos_len[0]:(row.span1_pos_len[0] + row.span1_pos_len[1])] for row in
                       df.itertuples(index=False)]
        df['offset1_start'] = [row.span1_pos_len[0] for row in df.itertuples(index=False)]
        df['offset1_end'] = [(row.span1_pos_len[0] + row.span1_pos_len[1]) for row in df.itertuples(index=False)]

        df['span2'] = [row.text[row.span2_pos_len[0]:(row.span2_pos_len[0] + row.span2_pos_len[1])] for row in
                       df.itertuples(index=False)]
        df['offset2_start'] = [row.span2_pos_len[0] for row in df.itertuples(index=False)]
        df['offset2_end'] = [(row.span2_pos_len[0] + row.span2_pos_len[1]) for row in df.itertuples(index=False)]

        # NER -> T#     label offset1 offset2       span
        # RE -> R#      relation_type Arg1:mark1 Arg2:mark2
        annotated_data = df.loc[:, ['pmid', 'mark1', 'label1', 'offset1_start', 'offset1_end', 'span1',
                                    'mark2', 'label2', 'offset2_start', 'offset2_end', 'span2',
                                    'relation_mark', 'relation_type']]

    return text, annotated_data


DATA_CLEANERS: dict[str, CleanerFunction] = {
    "GenoVarDis": cleaner_genovardis,
    "Variome": cleaner_variome,
    "TBGA": cleaner_tbga,
}


def get_cleaner(data_id) -> CleanerFunction:
    return DATA_CLEANERS[data_id]
