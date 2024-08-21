"""
All cleaners should have the same interface

(file input, task) -> source text[], gold standard data[]
which is the same as
Callable[[str, ...], (list[str], str)]
"""
import ast
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


def cleaner_variome(file_name: str, task_type: ...) -> CleanerFnResponse:
    # return text, gold standard data
    # TODO
    return [], ""


def cleaner_tbga(text_filepath: str, annotation_filepath: str = '') -> CleanerFnResponse:
    # return text, gold standard data
    df = pd.DataFrame()
    with open(text_filepath, 'r') as f:
        test_data_list = f.read().split('\n')
        result = [ast.literal_eval('{%s}' % item[1:-1]) for item in test_data_list]
        df = pd.json_normalize(result)

        df['pmid'] = 0
        # create text_df
        text = df.loc[:, ['pmid', 'text']]

        # create gold_df
        # pmid
        # mark1 label1 span1 offset1_start offset1_end
        # mark2 label2 span2 offset2_start offset2_end
        # relation_mark relation_type
        df['mark1'] = 'T' + df.index.astype(str)
        df['label1'] = 'Gene'

        df['mark2'] = 'T' + (len(df) + df.index + 1).astype(str)
        df['label2'] = 'Disease'

        df["relation_mark"] = 'R' + df.index.astype(str)

        df = df.rename(columns={"relation": "relation_type", "h.name": "span1",
                                "h.pos": "span1_pos_len", "t.name": "span2", "t.pos": "span2_pos_len"})

        print(df.head().to_string())

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

        # NER -> T#     label offset1 offset2   span
        df_ner_gene = df.loc[:, ['pmid', 'mark1', 'label1', 'offset1_start', 'offset1_end', 'span1']]
        df_ner_gene = df_ner_gene.rename(
            columns={'mark1': 'mark', 'label1': 'label', 'span1': 'span', 'offset1_start': 'offset1',
                     'offset1_end': 'offset2'})

        df_ner_disease = df.loc[:, ['pmid', 'mark2', 'label2', 'offset2_start', 'offset2_end', 'span2']]
        df_ner_disease = df_ner_disease.rename(
            columns={'mark2': 'mark', 'label2': 'label', 'span2': 'span', 'offset2_start': 'offset1',
                     'offset2_end': 'offset2'})

        annotated_ner = pd.concat([df_ner_gene, df_ner_disease], ignore_index=True)

        annotated_re = df.loc[:, ['pmid', "relation_mark", "relation_type", 'mark1', 'mark2']]
        print(annotated_ner.head().to_string())
        print(annotated_ner.tail().to_string())
        print(annotated_re.head().to_string())

    return text, {'ner': annotated_ner, 're': annotated_re}


DATA_CLEANERS: dict[str, CleanerFunction] = {
    "GenoVarDis": cleaner_genovardis,
    "Variome": cleaner_variome,
    "TBGA": cleaner_tbga,
}


def get_cleaner(data_id) -> CleanerFunction:
    return DATA_CLEANERS[data_id]
