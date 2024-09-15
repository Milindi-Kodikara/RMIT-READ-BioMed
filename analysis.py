import subprocess

import nbformat as nbf


def generate_jupyter_notebook(nb, filepath):
    file_name = f'{filepath}.ipynb'
    with open(file_name, 'w') as f:
        nbf.write(nb, f)
# TODO: Automatically execute the nb


def analysis(cleaned_entities, hallucinations, evaluation_values, filepath):
    nb = nbf.v4.new_notebook()
    nb['cells'] = []

    intro = """#### RMIT READ-BioMed-Version-2.0\n### Milindi Kodikara"""
    nb['cells'].append(nbf.v4.new_markdown_cell(intro))

    imports = """import pandas as pd\nimport numpy as np\nimport matplotlib.pyplot as plt
    """
    nb['cells'].append(nbf.v4.new_code_cell(imports))

    code = """%pylab inline\nhist(normal(size=2000), bins=50);"""

    nb['cells'].append(nbf.v4.new_code_cell(code))

    generate_jupyter_notebook(nb, filepath)
