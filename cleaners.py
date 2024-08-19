"""
All cleaners should have the same interface

(file input, task) -> source text[], gold standard data[]
which is the same as
Callable[[str, ...], (list[str], str)]
"""
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


def cleaner_genovardis(text_filepath: str, annotation_filepath: str) -> CleanerFnResponse:
    text_data = pd.read_csv(text_filepath, sep='\t', header=0)
    annotated_data = pd.read_csv(annotation_filepath, sep='\t', header=0)

    text_data['text'] = [clean_genovardis_text(text) for text in text_data['text']]
    annotated_data.drop(columns=['filename'], axis=1)
    return text_data, annotated_data


def cleaner_variome(file_name: str, task_type: ...) -> CleanerFnResponse:
    # return text, gold standard data
    # TODO
    return [], ""


def cleaner_tbga(file_name: str, task_type: ...) -> CleanerFnResponse:
    # return text, gold standard data
    # TODO
    return [], ""


DATA_CLEANERS: dict[str, CleanerFunction] = {
    "GenoVarDis": cleaner_genovardis,
    "Variome": cleaner_variome,
    "TBGA": cleaner_tbga,
}


def get_cleaner(data_id) -> CleanerFunction:
    return DATA_CLEANERS[data_id]
