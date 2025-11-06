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


def build_one_shot_prompt(current_elem, examples, proof=False):
    """Examples must be taken from the same theory """
    
    add_index = lambda x, i: x + "_" + str(i+1)
    
    prompt_examples = [[
        (add_index("example_introduction", i), f"Example {i+1}:"),
        (add_index("example_theory", i), "Theory:\n" + e["context"]),
        (add_index("example_question", i), f"Question:\n{e['question']}"),
        (add_index("example_answer", i), f"Answer:"),
        (add_index("example_label", i), e["label"]),
        (add_index("example_proof", i), e["proof"] + "\n") if proof else (None, )
    ] for i, e in enumerate(examples)]

    

    prompt_head = [
        ("preamble", "You will be shown some examples of questions and answers based on a logic theory. Your task is to evaluate a question based on a new theory.\nRespond with only one word: either True or False. " \
             + ("Do not include punctuation, explanation, or any other text.\n" if not proof else "Include the proof of your answer, formatted as the example.\n")),
    ]

    prompt_tail = [
        ("instruction", "Now, evaluate the following."),
        ("theory", f"Theory:\n{current_elem["context"]}"),
        ("question", f"Question:\n{current_elem["question"]}"),
        ("answer", f"Answer:\n")
    ]

    prompt_segmentation = dict([*prompt_head, *sum(prompt_examples, []), *prompt_tail])

    prompt_segmentation.pop(None, None)

    print(prompt_segmentation)

    prompt_text = "\n".join(prompt_segmentation.values())

    sep = ". "
    tuples_and_rules = current_elem["context"].split(sep)
    pos_theory_starts = prompt_text.index(current_elem["context"])
    theory_segmentation = reduce(lambda acc, x: acc + [acc[-1] + len(x) + len(sep)], tuples_and_rules, [pos_theory_starts])[:-1]

    print(current_elem["context"])
    print(len(current_elem["context"]))
    print(theory_segmentation)

    for i in range(len(theory_segmentation)-1):
        print(i, prompt_text[theory_segmentation[i] : theory_segmentation[i+1]])
    
    print()

    i = 0
    for k, v in prompt_segmentation.items():
        prompt_segmentation[k] = (i, i + len(v))
        i += len(v)
    
    return prompt_text, prompt_segmentation, theory_segmentation