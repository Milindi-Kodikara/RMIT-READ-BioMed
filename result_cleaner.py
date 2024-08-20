import re
import pandas as pd

label_entity_pattern = '^(?P<label>DNAMutation|SNP|DNAAllele|NucleotideChange-BaseChange|OtherMutation|Gene|Disease|Transcript)\s+(?P<span>[\w\W]+)$'


def extract_tuple(tuple_string):
    stripped_tuple_string = tuple_string.strip()
    matches = re.search(label_entity_pattern, stripped_tuple_string)

    if not matches:
        return

    label = matches.group("label").strip()
    span = matches.group("span").strip()

    return {'label': label, 'span': span}


def extract_entities(results):
    extracted_entities = pd.DataFrame(
        columns=['pmid', 'prompt_id', 'label', 'offset_checked', 'offset1', 'offset2', 'span'])

    for result_dict in results:
        pmid = result_dict['pmid']
        prompt_id = result_dict['prompt_id']
        result_string = result_dict['result']

        if result_string:
            extracted_list = result_string.splitlines()
            extracted_tuple_list = [extract_tuple(result_string) for result_string in extracted_list]

            for extracted_tuple in extracted_tuple_list:
                if extracted_tuple:
                    df_row = {
                        "pmid": pmid,
                        "prompt_id": prompt_id,
                        "label": extracted_tuple['label'],
                        "offset_checked": False,
                        "offset1": '',
                        "offset2": '',
                        "span": extracted_tuple['span']
                    }
                    extracted_entities.loc[len(extracted_entities)] = df_row

    return extracted_entities

# TODO: Fix hallucinations
def get_hallucinations(text_df, extracted_entity_df):
    # loop df, find each span, calculate the word length, find the indexes of each occurance
    for _, row in extracted_entity_df.iterrows():
        pmid = row['pmid']
        prompt_id = row['prompt_id']
        # find the text from the original_data with the pmid
        text = text_df.loc[text_df['pmid'] == pmid, 'text'].iloc[0]

        if not row['offset_checked'] and row['offset1'] == '':
            span = row['span']
            span_length = len(span)
            span_start_indexes = [m.start() for m in re.finditer(re.escape(span), text)]
            span_count = 0

            matching_spans = extracted_entity_df[
                (extracted_entity_df['pmid'] == pmid) & (extracted_entity_df['prompt_id'] == prompt_id) & (
                        extracted_entity_df['span'] == span) & (extracted_entity_df['offset1'] == '') & (
                        extracted_entity_df['offset_checked'] == False)]

            for index, matched_span in matching_spans.iterrows():
                if span_start_indexes and span_count < len(span_start_indexes):
                    extracted_entity_df.loc[index, 'offset1'] = str(span_start_indexes[span_count])
                    extracted_entity_df.loc[index, 'offset2'] = str(span_start_indexes[span_count] + span_length)

                    span_count = span_count + 1
                else:
                    # Add -1 to extra or missing ones
                    extracted_entity_df.loc[index, 'offset1'] = '-1'
                    extracted_entity_df.loc[index, 'offset2'] = '-1'

                extracted_entity_df.loc[index, 'offset_checked'] = True

    hallucinated_results = extracted_entity_df[
        (extracted_entity_df['offset1'] == '-1') & (extracted_entity_df['offset2'] == '-1')]

    return hallucinated_results


def result_cleaner(text, results):
    extracted_entities = extract_entities(results)
    print(f"Extracted entities len: {len(extracted_entities)}\n{extracted_entities.head()}\n\n")
    hallucinated_entities = get_hallucinations(text, extracted_entities)
    correct_entities = extracted_entities[
        (extracted_entities['offset1'] != '-1') & (extracted_entities['offset2'] != '-1')]

    return correct_entities, hallucinated_entities
