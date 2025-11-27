from torch.utils.data import Dataset
import copy

import json
from functools import reduce

def read_jsonl(dataset_path: str):

    with open(dataset_path, "r", encoding="utf-8") as out:
        jsonl = list(out)

    return [json.loads(i) for i in jsonl]

class ProofWriterDataset(Dataset):

    def __init__(self, tokenizer, dataset_path, triple_label="triple", rule_label="rule"):
        self.tokenizer = tokenizer
        self.dataset_path = dataset_path
        self.triple_label = triple_label
        self.rule_label = rule_label

        (
            self.triples,
            self.rules,
            self.questions,
            self.labels,
            self.proofs,
            self.proofs_intermerdiates,
            self.depths,
        ) = self.__read_dataset_proof_generation_all()


    def __read_dataset_proof_generation_all(self, triples_key="triples", rules_key="rules", questions_key="questions"):
        data = read_jsonl(self.dataset_path)
        triples_list = []
        rules_list = []
        questions_list = []
        labels_list = []
        proofs_list = []
        proofs_intermediates_list = []
        depths_list = []

        proofs_intermediates_key = "proofsWithIntermediates"

        for i, e in enumerate(data):
            triples = {}
            rules = {}
            questions = []
            proofs = []
            proofs_intermediates = []
            labels = []
            depths = []

            for t, val in e[triples_key].items():
                key = self.triple_label + t.replace("triple", "")
                triples[key] = val["text"]

            for r, val in e[rules_key].items():
                key = self.rule_label + r.replace("rule", "")
                rules[key] = val["text"]

            for q in e[questions_key].values():
                
                if not isinstance(q["answer"], bool): continue

                questions.append(q["question"])
                if proofs_intermediates_key in q:
                    tmp_proof = []
                    for p in q[proofs_intermediates_key]:
                        str_proof = f"{p['representation']}"
                        if len(p["intermediates"]) > 0:
                            str_proof += " ; "
                            str_proof += "with "
                            for intr, val in p["intermediates"].items():
                                str_proof += f"{intr} = {val['text']}"
                        str_proof = str_proof.replace("triple", self.triple_label)
                        str_proof = str_proof.replace("rule", self.rule_label)
                        tmp_proof.append(str_proof)
                        break # temporary, taking only one proof

                labels.append(q["answer"])
                proofs.append(q["proofs"])
                proofs_intermediates.append(tmp_proof)
                depths.append(q["QDep"])

            for q, l, p, p_i, d in zip(questions, labels, proofs, proofs_intermediates, depths):
                triples_list.append(triples)
                rules_list.append(rules)
                questions_list.append(q)
                labels_list.append(l)
                proofs_list.append(p)
                proofs_intermediates_list.append(p_i)
                depths_list.append(d)

        # print(
        #     triples_list[0],
        #     rules_list[0],
        #     questions_list[0],
        #     labels_list[0],
        #     proofs_list[0],
        # )
        return (
            triples_list,
            rules_list,
            questions_list,
            labels_list,
            proofs_list,
            proofs_intermediates_list,
            depths_list,
        )

    def __getitem__(self, index):

        if type(index) == slice:
            res = []
            for i in range(*index.indices(len(self))):
                res.append(self[i])
            return res

        question = self.questions[index]

        f = lambda acc, e: acc + f"{e[0]}: {e[1]} "
        context = reduce(f, self.triples[index].items(), "") \
                + reduce(f, self.rules[index].items(), "")
        
        label = str(self.labels[index])
        proof = self.proofs_intermerdiates[index][0]

        """
        input_text = f"Question: {question}. Context: {context}"
        target_text = f"{label}. Proof: {proof}"

        input_encoding = self.tokenizer(
            input_text,
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors="pt"
        )

        target_encoding = self.tokenizer(
            target_text,
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors="pt"
        )

        labels = target_encoding["input_ids"].squeeze(0)
        labels[labels == self.tokenizer.pad_token_id] = -100  # Ignore padding tokens

        return {
            "input_ids": input_encoding["input_ids"].squeeze(0),
            "attention_mask": input_encoding["attention_mask"].squeeze(0),
            "labels": labels
        }
        """

        return {
            "context" : context,
            "question" : question,
            "label" : label,
            "proof" : proof
        }

    def __len__(self):
        return len(self.questions)
