from functools import reduce
import re

def get_groups_delimiters_by_tokens(groups_delimiters_by_characters, offsets):
    j = 0
    groups_delimiters_by_tokens = []
    for gi in groups_delimiters_by_characters:
        while offsets[j][1] < gi: j += 1

        if len(groups_delimiters_by_tokens) <= 1 or j != groups_delimiters_by_tokens[-1]:
            groups_delimiters_by_tokens.append(j)
    
    return groups_delimiters_by_tokens

def to_couples(l):
    if len(l) == 2: return [(l[0], l[1])]
    return [(l[0], l[1])] + to_couples(l[1:])

def get_groups_delimiter_intervals_by_tokens(groups_delimiters_by_characters, offsets):
    return to_couples(get_groups_delimiters_by_tokens(groups_delimiters_by_characters, offsets))

def str_bool_to_bin(b):
    if b not in ["True", "False"]: raise Exception(f"The input must be True of False; '{b}' received.")
    return int(b == "True")


def __build_base_prompt(current_elem, proof=True, examples=False):

    special_tags_instruction = \
    f"""
Produce two levels of output:
  1. Write the final answer (only one word: True or False) inside <final>...</final>.
  2. Write the proof{", formatted as in the example," if examples else ""} inside <proof>...</proof>.

When writing the reasoning, always choose the simplest valid proof. Avoid unnecessary steps or complex derivations.
If a single-step proof exists, use ONLY that step.\n""" if proof else \
    "Write the final answer (only one word: True or False) inside <final>...</final>."

    # simplest_proof_instruction = "When writing the reasoning, always choose the simplest valid proof. Avoid unnecessary steps or complex derivations."

    prompt_head = [
        ("preamble", "You will be shown a logic theory and a logic question." \
            # + ("Do not include punctuation, explanation, or any other text.\n" if not proof else "Include the proof of your answer, formatted as the example.\n")),
            + special_tags_instruction)
    ]

    prompt_tail = [
        ("instruction", "Now, evaluate the following."),
        ("theory_label", "Theory:"),
        ("theory", current_elem["context"]),
        ("question_label", "Question:"),
        ("question", f"{current_elem["question"]}\n"),
        ("answer_label", "Answer:\n")
    ]

    return prompt_head, prompt_tail


def build_prompt(current_elem, examples, proof=False):
    """Examples must be taken from the same theory."""
    
    add_index = lambda x, i: x + "_" + str(i+1)
    
    prompt_examples = [[
        (add_index("example_introduction", i), "Example:"),
        (add_index("example_theory_label", i), "Theory:"),
        (add_index("example_theory", i), e["context"]),
        (add_index("example_question_label", i), "Question:"),
        (add_index("example_question", i), e['question']),
        (add_index("example_answer_label", i), "Answer:"),
        (add_index("example_answer", i), "<final>" + e["label"] + "</final>"),
        (add_index("example_proof", i), "<reasoning>" + e["proof"] + "</reasoning>\n") if proof else (None, )
    ] for i, e in enumerate(examples)]

    prompt_head, prompt_tail = __build_base_prompt(current_elem, proof=proof, examples=True)

    if len(examples) > 0:
        prompt_segmentation = dict([*prompt_head, *sum(prompt_examples, [("examples_preamble", "Here is an example.\n" if len(examples) == 1 else "Here are some examples.\n")]), *prompt_tail])
        prompt_segmentation.pop(None, None) # remove proof entries when proof is false
    else:
        prompt_segmentation = dict([*prompt_head, *prompt_tail])

    # print(prompt_segmentation)

    prompt_text = "\n".join(prompt_segmentation.values())

    sep = ". "
    tuples_and_rules = [x for x in current_elem["context"].split(sep) if x != ""]

    # print(prompt_text)

    pos_theory_starts = prompt_text.index(current_elem["context"])
    theory_segmentation = reduce(lambda acc, x: acc + [acc[-1] + len(x) + len(sep)], tuples_and_rules, [pos_theory_starts])

    # print(current_elem["context"])
    # print(len(current_elem["context"]))

    # for i in range(len(theory_segmentation)-1):
    #    print(i, prompt_text[theory_segmentation[i] : theory_segmentation[i+1]])
    
    # print()

    i = 0
    for k, v in prompt_segmentation.items():
        prompt_segmentation[k] = (i, i + len(v))
        i += len(v)
    
    return prompt_text, prompt_segmentation, theory_segmentation, [find_stmt_name(x) for x in tuples_and_rules]


def find_stmt_name(text):
    return find_stmt_names(text)[0]

def find_stmt_names(text):
    return list(dict.fromkeys(re.findall(r"\b(?:t\d+|r\d+)\b", text)))


def create_prompts(dataset, examples=[], n_prompts=None, proof=True):
    prompts, y_true, proofs = [], [], []

    for i, e in enumerate(dataset):
        if n_prompts and i >= n_prompts: break
        y_true.append(str_bool_to_bin(e["label"]))
        proofs.append(e["proof"])
        # print(e["question"])
        # print(e["proof"])
        if len(examples) > 0:
            prompts.append(build_prompt(e, examples=examples, proof=proof))
        else:
            prompts.append(build_prompt(e, examples=examples, proof=proof))
    
    return prompts, y_true, proofs