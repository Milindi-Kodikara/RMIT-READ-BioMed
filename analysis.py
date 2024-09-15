import subprocess

import nbformat as nbf


def generate_jupyter_notebook(nb, filepath):
    file_name = f'{filepath}/analysis.ipynb'
    with open(file_name, 'w') as f:
        nbf.write(nb, f)
# TODO: Automatically execute the nb


def add_cell(nb, cell_data, cell_type="code"):
    if cell_type == "mk":
        nb['cells'].append(nbf.v4.new_markdown_cell(cell_data))
        return nb
    elif cell_type == "code":
        nb['cells'].append(nbf.v4.new_code_cell(cell_data))
        return nb


def analysis(filepath):
    nb = nbf.v4.new_notebook()
    nb['cells'] = []

    intro = """#### RMIT READ-BioMed-Version-2.0\n### Milindi Kodikara"""
    nb = add_cell(nb, intro, "mk")

    imports = """import pandas as pd\nimport numpy as np\nimport matplotlib.pyplot as plt
    """
    nb = add_cell(nb, imports)

    setup = """evaluation_df = pd.read_csv('./eval_log.tsv', sep='\t', header=0)\nevaluation_df.head()"""
    nb = add_cell(nb, setup)

    generate_jupyter_notebook(nb, filepath)
