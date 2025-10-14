
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

def build_one_shot_prompt(current_elem, example_elem, proof=False):

    prompt_segmentation = {
        "preamble" :            "You will be shown a theory and a question. Your task is to evaluate the question based on the theory.\nRespond with only one word: either True or False. Do not include punctuation, explanation, or any other text.",
        "example_introduction": "Example:",
        "example_theory" :      "Theory:\n" + example_elem["context"],
        "example_question":     "Question:\n" + example_elem["question"],
        "example_label" :       example_elem["label"],
        "example_proof" :       (example_elem["proof"] if proof else "") + "\n",
        "instruction" :         "Now, evaluate the following.",
        "theory" :              "Theory:\n" + current_elem["context"],
        "question" :            "Question:\n" + current_elem["question"] + "\n"
    }

    prompt_text = "\n".join(prompt_segmentation.values())

    i = 0
    for k, v in prompt_segmentation.items():
        prompt_segmentation[k] = (i, i + len(v))
        i += len(v)
    
    return prompt_text, prompt_segmentation