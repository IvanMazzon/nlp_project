from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, average_precision_score
from scipy.stats import entropy
import re
import os
import numpy as np
import pandas as pd
from functools import reduce
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from tqdm.notebook import tqdm
from prompt_utils import *

def print_metric(name, value): print(f"{name:>10}: {value:.3}")

def compute_performance(y_true, y_pred):
    assert len(y_true) == len(y_pred), f"{len(y_true)} != {len(y_pred)}"

    print("pos ground truth:", sum(y_true))
    print("neg ground truth:", len(y_true) - sum(y_true), "\n")
    
    print_metric("accuracy", accuracy_score(y_true, y_pred))
    print_metric("f1 score", f1_score(y_true, y_pred))
    print_metric("prec", precision_score(y_true, y_pred))
    print_metric("rec", recall_score(y_true, y_pred))
