[![coursePage](https://img.shields.io/badge/University_of_Milan-Natural_Language_Processing-dae1e7?style=for-the-badge&logoColor=white&labelColor=004082)](https://www.unimi.it/it/corsi/insegnamenti-dei-corsi-di-laurea/2025/natural-language-processing-0)

# **Analysis of the role of attention and prompt engineering in logical reasoning with decoder-only transformers**

This project examines the ability of a large language model (LLM) to perform logical reasoning
by analyzing the attention mechanisms within a decoder-only transformer architecture.
The model is provided with instances composed by a theory and a logical proposition, and tasked with determining whether the proposition is true or false.
The central hypothesis is that, when the model produces the correct answer, it allocates greater attention
to the actual most relevant portions of the input, particularly the logical statements that support the reasoning process.
The second aspect is to evaluate the impact of different prompting techniques on the ability of the model
to classify logical propositions, as well as on the attention distribution.


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
