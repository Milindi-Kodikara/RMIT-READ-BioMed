import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor


def analysis(root_folder_path, result_folder_path):
    nb_template_path = f'{root_folder_path}/RMIT-READ-BioMed/analysis_template.ipynb'
    with open(nb_template_path) as ff:
        nb_in = nbf.read(ff, nbf.NO_CONVERT)

    ep = ExecutePreprocessor(timeout=600, kernel_name='python3')

    nb_out = ep.preprocess(nb_in)

    with open(f'{result_folder_path}/results/analysis.ipynb', 'w', encoding='utf-8') as f:
        nbf.write(nb_out[0], f)
