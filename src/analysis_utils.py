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
import warnings
from sklearn.exceptions import UndefinedMetricWarning

def print_metric(name, value): print(f"{name:>10}: {value:.3}")

def compute_performance(y_true, y_pred):
    assert len(y_true) == len(y_pred), f"{len(y_true)} != {len(y_pred)}"

    print("pos ground truth:", sum(y_true))
    print("neg ground truth:", len(y_true) - sum(y_true), "\n")
    
    print_metric("accuracy", accuracy_score(y_true, y_pred))
    print_metric("f1 score", f1_score(y_true, y_pred))
    print_metric("prec", precision_score(y_true, y_pred))
    print_metric("rec", recall_score(y_true, y_pred))


def text_generation_and_attention_analysis(prompts, y_true, proofs, model, tokenizer, experiment_name, results_folder="results"):

    warnings.filterwarnings("error", category=UndefinedMetricWarning)

    y_pred = []
    try:
        mean_attention_per_group_last_layer = pd.DataFrame(columns=list(prompts[0][1].keys()) + ["correct_prediction"], dtype=np.float64)
    except: print(prompts)
    proofs_metrics = pd.DataFrame(columns=["auc", "auprc", "correct_prediction"], dtype=np.float64)

    generation_args = {
        "output_attentions" : True,
        "return_dict_in_generate": True,
        "max_new_tokens": 100,
        "do_sample": False,
        "pad_token_id": tokenizer.eos_token_id
    }

    k=0

    for i, (prompt, prompt_segmentation, theory_segmentation, tuples_and_rules) in tqdm(enumerate(prompts), total=len(prompts)):

        encoded = tokenizer(
            prompt,
            return_offsets_mapping=True,
            return_tensors="pt",
            padding=True
        )

        offset_mapping = encoded.pop("offset_mapping")
        encoded = encoded.to("cuda")

        with torch.no_grad():
            outputs = model.generate(**encoded, **generation_args)
        
        generated_text = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)[len(prompt):]
        
        # print(prompt)
        # print("----")
        # print(generated_text)

        res = re.findall(fr"<final>([\s\S]*?)</final>", generated_text)[0]

        try:
            y_pred.append(str_bool_to_bin(res))
            correct_prediction = int(y_true[i] == str_bool_to_bin(res))
        except: y_pred.append(False); correct_prediction = 0

        offsets = offset_mapping[0]
        groups_delimiters_by_characters = reduce(lambda acc, x : acc + [x[1]], prompt_segmentation.values(), [0])

        groups_intervals_by_token = get_groups_delimiter_intervals_by_tokens(groups_delimiters_by_characters, offsets)
        
        n_input_tokens = groups_intervals_by_token[-1][-1]

        prompt_length = encoded["input_ids"].shape[-1]
        new_token_ids = outputs.sequences[0][prompt_length:]
        tokens = tokenizer.convert_ids_to_tokens(new_token_ids)
        non_special_tokens = [t for t in tokens if not tokenizer.special_tokens_map.get(t) and t not in tokenizer.all_special_tokens]
        # print(f"Non special tokens generated: {len(non_special_tokens)}")

        means = []
        
        for k, attn_gen_token in enumerate(outputs.attentions):
            if k == 0: continue # attention map only referred to input tokens
            if k == len(non_special_tokens)+1: break

            means.append(np.mean([head[0][:n_input_tokens] for head in attn_gen_token[0][0].detach().to(torch.float32).cpu().numpy()], axis=0))

        mean_output_attention = np.mean(means, axis=0)

        groups_mean_attention = np.zeros(len(prompt_segmentation))

        for j, (start, end) in enumerate(groups_intervals_by_token):
            values = mean_output_attention[start:end]
            groups_mean_attention[j] = np.mean(values)

        mean_attention_per_group_last_layer = pd.concat([mean_attention_per_group_last_layer, pd.DataFrame([list(groups_mean_attention) + [correct_prediction]], columns= list(prompt_segmentation.keys()) + ["correct_prediction"])], ignore_index=True)
        
        theory_intervals_by_token = dict(zip(tuples_and_rules, get_groups_delimiter_intervals_by_tokens(theory_segmentation, offsets)))

        mean_attention_on_theory_components = dict(zip(tuples_and_rules, np.zeros(len(theory_intervals_by_token))))

        
        for key, (start, end) in theory_intervals_by_token.items():
            values = mean_output_attention[start:end]
            mean_attention_on_theory_components[key] = np.mean(values)
            
        
        actual_proof_terms = set(find_stmt_names(proofs[i]))
        stmt_labels = [1 if stmt_name in actual_proof_terms else 0 for stmt_name in tuples_and_rules]
        
        auc, auprc = np.nan, np.nan
        try:
            auc = roc_auc_score(stmt_labels, list(mean_attention_on_theory_components.values()))
        except: pass
        
        auprc = average_precision_score(stmt_labels, list(mean_attention_on_theory_components.values()))

        proofs_metrics = pd.concat([proofs_metrics, pd.DataFrame([[auc, auprc, correct_prediction]], columns=proofs_metrics.columns)], ignore_index=True)
        
        del outputs, encoded
        torch.cuda.empty_cache()

    compute_performance(y_true, y_pred)

    plt.figure(figsize=(6, 5))

    x_name, y_name = "groups of tokens", "average attention"
    long_df = mean_attention_per_group_last_layer.melt(var_name=x_name, value_name=y_name, id_vars="correct_prediction")

    sns.boxplot(x=x_name, y=y_name, data=long_df, hue="correct_prediction", hue_order=[1, 0], palette=["#1A85FF", "#D41159"], width=.5)

    handles, _ = plt.gca().get_legend_handles_labels()
    plt.legend(handles, ["correct prediction", "incorrect prediction"])
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    os.makedirs(f"{results_folder}/{experiment_name}", exist_ok=True)
    plt.savefig(f"{results_folder}/{experiment_name}/average_attentions.png", dpi=400, bbox_inches="tight")
    plt.show()

    ##### 

    plt.figure(figsize=(6, 5))

    x_name, y_name = "metrics", "value"
    long_df = proofs_metrics.melt(var_name=x_name, value_name=y_name, id_vars="correct_prediction")

    sns.boxplot(x=x_name, y=y_name, data=long_df, hue="correct_prediction", hue_order=[1, 0], palette=["#1A85FF", "#D41159"], width=.3)

    handles, _ = plt.gca().get_legend_handles_labels()
    plt.legend(handles, ["correct prediction", "incorrect prediction"])
    plt.ylim(0, 1)
    plt.tight_layout()
    os.makedirs(f"{results_folder}/{experiment_name}", exist_ok=True)
    plt.savefig(f"{results_folder}/{experiment_name}/proof_metrics.png", dpi=400, bbox_inches="tight")
    plt.show()
