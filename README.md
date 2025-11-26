# **Analysis of the role of attention and prompt engineering in decoder-only transformers logical reasoning**

[![coursePage](https://img.shields.io/badge/University_of_Milan-Natural_Language_Processing-dae1e7?style=for-the-badge&logoColor=white&labelColor=004082)](https://www.unimi.it/en/education/degree-programme-courses/2025/natural-language-processing)

## Requirements

Pipenv is used to create a virtual environment and install all other required dependencies automatically.

```bash
pip install pipenv
```

Next, run the following commands in the project directory.

```bash
export PIPENV_VENV_IN_PROJECT=1
pipenv install
pipenv shell
```

Finally, execute the Jupyter Notebook file using the Python kernel in `.venv/bin/python`.


For externally managed Python installation:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install pipenv
pipenv install
pipenv shell
```