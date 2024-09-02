import re
import pandas as pd


def extract_tuple(tuple_string, annotation_pattern):
    stripped_tuple_string = tuple_string.strip()
    matches = re.search(annotation_pattern, stripped_tuple_string)

    if not matches:
        return

    label = matches.group("label").strip()
    span = matches.group("span").strip()

    return {'label': label, 'span': span}


def extract_triplet(triplet_string, annotation_pattern):
    stripped_triplet_string = triplet_string.strip()
    stripped_triplet_string = re.split(r'\s{2,}', stripped_triplet_string)
    reformatted_triplet_string = '\t'.join(stripped_triplet_string)

    matches = re.search(annotation_pattern, reformatted_triplet_string)

    if not matches:
        return

    span1 = matches.group("span1").strip()
    span2 = matches.group("span2").strip()
    relation_type = matches.group("relation_type").strip()

    return {'span1': span1, 'span2': span2, 'relation_type': relation_type}


def extract_quintuplet(quintuplet_string, annotation_pattern):
    stripped_triplet_string = quintuplet_string.strip()
    stripped_triplet_string = re.split(r'\s{2,}', stripped_triplet_string)
    reformatted_triplet_string = '\t'.join(stripped_triplet_string)

    matches = re.search(annotation_pattern, reformatted_triplet_string)

    if not matches:
        return

    label1 = matches.group("label1").strip()
    label2 = matches.group("label2").strip()

    span1 = matches.group("span1").strip()
    span2 = matches.group("span2").strip()
    relation_type = matches.group("relation_type").strip()

    return {'label1': label1, 'span1': span1, 'label2': label2, 'span2': span2, 'relation_type': relation_type}


def determine_extraction_func(task, annotation_pattern, extracted_list):
    if task == 'NER':
        return [
            extract_tuple(result_string, annotation_pattern) for result_string in
            extracted_list]

    if task == 'RE':
        return [extract_triplet(result_string, annotation_pattern) for result_string in
                extracted_list]

    if task == 'NERRE':
        return [
            extract_quintuplet(result_string, annotation_pattern) for result_string in
            extracted_list]


def extract_entities(results, task, ner_annotations, re_annotations):
    annotation_pattern = ''
    if task == 'NER':
        split_annotations = ner_annotations.replace(',', '|')
        annotation_pattern = '^(?P<label>' + split_annotations + ')\s+(?P<span>[\w\W]+)$'
        extracted_entities = pd.DataFrame(
            columns=['pmid', 'prompt_id', 'mark', 'label', 'offset_checked', 'offset1', 'offset2', 'span'])

    if task == 'RE':
        split_annotations = re_annotations.replace(',', '|')
        annotation_pattern = '^(?P<span1>.+)[\t](?P<span2>.+)[\t](?P<relation_type>' + split_annotations + ')$'
        extracted_entities = pd.DataFrame(
            columns=['pmid', 'prompt_id', 'offsets_checked', 'mark1', 'label1', 'offset1_start', 'offset1_end', 'span1',
                     'mark2', 'label2', 'offset2_start', 'offset2_end', 'span2',
                     'relation_mark', 'relation_type'])

    if task == 'NERRE':
        split_ner_annotations = ner_annotations.replace(',', '|')
        split_re_annotations = re_annotations.replace(',', '|')

        annotation_pattern = '^(?P<label1>' + split_ner_annotations + ')[\t](?P<span1>.+)[\t](?P<label2>' + split_ner_annotations + ')[\t](?P<span2>.+)[\t](?P<relation_type>' + split_re_annotations + ')$'
        extracted_entities = pd.DataFrame(
            columns=['pmid', 'prompt_id', 'offsets_checked', 'mark1', 'label1', 'offset1_start', 'offset1_end', 'span1',
                     'mark2', 'label2', 'offset2_start', 'offset2_end', 'span2',
                     'relation_mark', 'relation_type'])

    for result_dict in results:
        pmid = result_dict['pmid']
        prompt_id = result_dict['prompt_id']
        result_string = result_dict['result']

        if result_string:
            extracted_list = result_string.splitlines()
            extracted_items_list = determine_extraction_func(task, annotation_pattern, extracted_list)

            for extracted_item in extracted_items_list:
                if extracted_item:
                    df_row = {}
                    if task == 'NER':
                        df_row = {
                            "pmid": pmid,
                            "prompt_id": prompt_id,
                            "mark": 'T' + str(len(extracted_entities)),
                            "label": extracted_item['label'],
                            "offset_checked": False,
                            "offset1": '',
                            "offset2": '',
                            "span": extracted_item['span']
                        }
                    elif task == 'RE':
                        df_row = {
                            "pmid": pmid,
                            "prompt_id": prompt_id,
                            "offsets_checked": False,
                            "mark1": 'T' + str(len(extracted_entities)) + '_gene',
                            "label1": 'Gene',
                            "offset1_start": '',
                            "offset1_end": '',
                            "span1": extracted_item['span1'],
                            "mark2": 'T' + str(len(extracted_entities)) + '_disease',
                            "label2": 'Disease',
                            "offset2_start": '',
                            "offset2_end": '',
                            "span2": extracted_item['span2'],
                            "relation_mark": 'R' + str(len(extracted_entities)),
                            "relation_type": extracted_item['relation_type']
                        }
                    elif task == 'NERRE':
                        df_row = {
                            "pmid": pmid,
                            "prompt_id": prompt_id,
                            "offsets_checked": False,
                            "mark1": 'T_1_' + str(len(extracted_entities)),
                            "label1": extracted_item['label1'],
                            "offset1_start": '',
                            "offset1_end": '',
                            "span1": extracted_item['span1'],
                            "mark2": 'T_2_' + str(len(extracted_entities)),
                            "label2": extracted_item['label2'],
                            "offset2_start": '',
                            "offset2_end": '',
                            "span2": extracted_item['span2'],
                            "relation_mark": 'R' + str(len(extracted_entities)),
                            "relation_type": extracted_item['relation_type']
                        }
                    extracted_entities.loc[len(extracted_entities)] = df_row

    return extracted_entities


def mark_hallucinated_fabricated_spans(row_index, row_text, row, extracted_entity_df, task):
    if task == 'NER':
        span = row["span"]

        if not row['offset_checked'] and span not in row_text:
            extracted_entity_df.loc[row_index, 'offset_checked'] = True
            extracted_entity_df.loc[row_index, 'offset1'] = '-1'
            extracted_entity_df.loc[row_index, 'offset2'] = '-1'

    if task == 'RE' or task == 'NERRE':
        span1 = row['span1']
        span2 = row['span2']

        if type(span1) is not str:
            span1.astype(str)
        elif type(span2) is not str:
            span2.astype(str)

        if not row['offsets_checked'] and (not span1 in row_text or not span2 in row_text):
            extracted_entity_df.loc[row_index, 'offsets_checked'] = True

            extracted_entity_df.loc[row_index, 'offset1_start'] = '-1'
            extracted_entity_df.loc[row_index, 'offset1_end'] = '-1'

            extracted_entity_df.loc[row_index, 'offset2_start'] = '-1'
            extracted_entity_df.loc[row_index, 'offset2_end'] = '-1'
        elif not row['offsets_checked'] and span1 in row_text and span2 in row_text:
            extracted_entity_df.loc[row_index, 'offset1_start'] = str(row_text.find(span1))
            extracted_entity_df.loc[row_index, 'offset1_end'] = str(row_text.find(span1) + len(span1))

            extracted_entity_df.loc[row_index, 'offset2_start'] = str(row_text.find(span2))
            extracted_entity_df.loc[row_index, 'offset2_end'] = str(row_text.find(span2) + len(span2))

            extracted_entity_df.loc[row_index, 'offsets_checked'] = True


# TODO: Refactor this for RE, NERRE coz there are duplicated rows with same spans and relation coz of multiple
#  occurances of the same spans
# TODO: Check the offset calc in variomecorp
def mark_hallucinated_extra_spans(row_text, row, extracted_entity_df, prompt_id):
    if not row['offset_checked']:
        span = row['span']
        span_length = len(span)

        # span occurrences extracted
        matching_spans = extracted_entity_df[
            (extracted_entity_df['pmid'] == row['pmid']) & (extracted_entity_df['prompt_id'] == prompt_id) & (
                    extracted_entity_df['span'] == span)]

        # start indexes of the all the mentions of the span from the text
        span_start_indexes = [m.start() for m in re.finditer(re.escape(span), row_text)]
        # number of occurrences of the span in the text
        span_occurrences_in_text = len(span_start_indexes)

        span_count = 0

        for index, matched_span in matching_spans.iterrows():
            if not matched_span['offset_checked']:
                if span_count < span_occurrences_in_text:
                    extracted_entity_df.loc[index, 'offset1'] = str(span_start_indexes[span_count])
                    extracted_entity_df.loc[index, 'offset2'] = str(span_start_indexes[span_count] + span_length)
                    span_count = span_count + 1
                else:
                    # this means that the span is an extra
                    extracted_entity_df.loc[index, 'offset1'] = '-2'
                    extracted_entity_df.loc[index, 'offset2'] = '-2'

            extracted_entity_df.loc[index, 'offset_checked'] = True


def get_hallucinations(text_df, extracted_entity_df, task):
    # loop df, find each span, calculate the word length, find the indexes of each occurrence
    correct_entities = extracted_entity_df
    hallucinated_results = pd.DataFrame()

    for row_index, row in extracted_entity_df.iterrows():
        pmid = row['pmid']
        prompt_id = row['prompt_id']
        # find the text from the original_data with the pmid
        row_text = text_df.loc[text_df['pmid'] == pmid, 'text'].iloc[0]

        mark_hallucinated_fabricated_spans(row_index, row_text, row, extracted_entity_df, task)

        if task == 'NER':
            mark_hallucinated_extra_spans(row_text, row, extracted_entity_df, prompt_id)

            hallucinated_results = extracted_entity_df[
                ((extracted_entity_df['offset1'] == '-1') & (extracted_entity_df['offset2'] == '-1')) |
                ((extracted_entity_df['offset1'] == '-2') & (extracted_entity_df['offset2'] == '-2'))]

            correct_entities = extracted_entity_df[
                (extracted_entity_df['offset1'] != '-1') & (extracted_entity_df['offset2'] != '-1') &
                (extracted_entity_df['offset1'] != '-2') & (extracted_entity_df['offset2'] != '-2')]
        elif task == 'RE' or task == 'NERRE':
            hallucinated_results = extracted_entity_df[
                (extracted_entity_df['offset1_start'] == '-1') & (extracted_entity_df['offset2_start'] == '-1')]

            correct_entities = extracted_entity_df[
                (extracted_entity_df['offset1_start'] != '-1') & (extracted_entity_df['offset2_start'] != '-1')]

    return correct_entities, hallucinated_results


def result_cleaner(text, results, ner_annotations, re_annotations, task):
    extracted_entities = extract_entities(results, task, ner_annotations, re_annotations)
    return get_hallucinations(text, extracted_entities, task)
