from functools import reduce

def get_groups_delimiters_by_tokens(groups_delimiters_by_characters, offsets):
    j = 0
    groups_delimiters_by_tokens = []
    for gi in groups_delimiters_by_characters:
        while offsets[j][1] < gi: j += 1
        groups_delimiters_by_tokens.append(j)
    
    return groups_delimiters_by_tokens

def get_groups_delimiter_intervals_by_tokens(groups_delimiters_by_characters, offsets):

    groups_delimiters_by_tokens = get_groups_delimiters_by_tokens(groups_delimiters_by_characters, offsets)

    def to_couples(l):
        if len(l) == 2: return [(l[0], l[1])]
        return [(l[0], l[1])] + to_couples(l[1:])

    groups_intervals = to_couples(groups_delimiters_by_tokens)
    return groups_intervals



def build_prompt(elem):
    return "\n".join([
        "Given the following theory of logic rules, answer the question given the result as true or false. Do not generate other text.\n",
        elem["context"],
        elem["question"]
    ])


def __build_base_prompt(current_elem, proof=False):

    special_tags_instruction = \
    """
Produce two levels of output:
  1. Write the final answer (only one word: True or False) inside <final>...</final>.
  2. Write the proof or reasoning, formatted as in the example, inside <reasoning>...</reasoning>.

Proof selection rules:
- First, check if the conclusion follows directly from a single fact tX.
- If a single-step proof exists, use ONLY that step.
- If a single-step proof does NOT exist, use the shortest valid chain.
- Avoid unnecessary steps or complex derivations. Prefer the shortest chain of rules.\n"""

    # simplest_proof_instruction = "When writing the reasoning, always choose the simplest valid proof. Avoid unnecessary steps or complex derivations."

    prompt_head = [
        ("preamble", "You will be shown a logic theory and a logic question." \
            # + ("Do not include punctuation, explanation, or any other text.\n" if not proof else "Include the proof of your answer, formatted as the example.\n")),
            + special_tags_instruction)
    ]

    prompt_tail = [
        ("instruction", "Now, evaluate the following."),
        ("theory", f"Theory:\n{current_elem["context"]}"),
        ("question", f"Question:\n{current_elem["question"]}\n"),
        ("answer", f"Answer:\n")
    ]

    return prompt_head, prompt_tail


def build_few_shot_prompt(current_elem, examples, proof=False):
    """Examples must be taken from the same theory."""
    
    add_index = lambda x, i: x + "_" + str(i+1)
    
    prompt_examples = [[
        (add_index("example_introduction", i), f"Example:"),
        (add_index("example_theory", i), "Theory:\n" + e["context"]),
        (add_index("example_question", i), f"Question:\n{e['question']}"),
        (add_index("example_answer", i), f"Answer:"),
        (add_index("example_label", i), "<final>" + e["label"] + "</final>"),
        (add_index("example_proof", i), "<reasoning>" + e["proof"] + "</reasoning>\n") if proof else (None, )
    ] for i, e in enumerate(examples)]

    prompt_head, prompt_tail = __build_base_prompt(current_elem, proof)
    prompt_segmentation = dict([*prompt_head, *sum(prompt_examples, [("examples_preamble", "Here is an example.\n" if len(examples) == 1 else "Here are some examples.\n")]), *prompt_tail])
    prompt_segmentation.pop(None, None) # remove proof entries when proof is false

    # print(prompt_segmentation)

    prompt_text = "\n".join(prompt_segmentation.values())

    sep = ". "
    tuples_and_rules = current_elem["context"].split(sep)
    pos_theory_starts = prompt_text.index(current_elem["context"])
    theory_segmentation = reduce(lambda acc, x: acc + [acc[-1] + len(x) + len(sep)], tuples_and_rules, [pos_theory_starts])[:-1]

    # print(current_elem["context"])
    # print(len(current_elem["context"]))
    # print(theory_segmentation)

    # for i in range(len(theory_segmentation)-1):
    #    print(i, prompt_text[theory_segmentation[i] : theory_segmentation[i+1]])
    
    # print()

    i = 0
    for k, v in prompt_segmentation.items():
        prompt_segmentation[k] = (i, i + len(v))
        i += len(v)
    
    return prompt_text, prompt_segmentation, theory_segmentation