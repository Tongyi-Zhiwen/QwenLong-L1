import os, csv, json
import argparse
import time
from tqdm import tqdm
from datasets import load_dataset
import re
from transformers import AutoTokenizer
import tiktoken
import torch.multiprocessing as mp
import string
from collections import Counter
from datasets import load_dataset, concatenate_datasets

MAX_INPUT_LEN=120000
MAX_OUTPUT_LEN=10000

template_0shot = """Please read the following text and answer the question below.

<text>
$DOC$
</text>

$Q$

Format your response as follows: "Therefore, the answer is (insert answer here)"."""


def extract_solution(solution_str):
    """Extracts the final answer from the model's response string.
    
    Args:
        solution_str: Raw response string from the language model
        
    Returns:
        Tuple containing (extracted_answer, processed_string)
    """
    # Extract final answer using XML-style tags
    if "</think>" not in solution_str:
        print("[Error] No valid answer tags found")
        return None, solution_str 
    
    final_answer = solution_str.split("</think>")[-1].strip()
    return final_answer, solution_str


def extract_answer(response):
    response = response.replace('*', '')

    if "the answer is" in response:
        ans = response.rsplit("the answer is", 1)[-1].strip().replace("<｜Assistant｜>", '').replace("<｜end▁of▁sentence｜>", '').strip().strip('.').strip()
    else:
        ans = None

    return ans


def normalize_answer(s):

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    ZERO_METRIC = (0, 0, 0)

    if normalized_prediction in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC
    if normalized_ground_truth in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return ZERO_METRIC
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall


def sub_exact_match_score(prediction, ground_truth):
    ground_truth = normalize_answer(ground_truth)
    prediction = normalize_answer(prediction) 
    return (ground_truth in prediction) or (prediction in ground_truth)

def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


def update_answer(metrics, prediction, gold):
    em = exact_match_score(prediction, gold)
    subem = sub_exact_match_score(prediction, gold)

    f1, prec, recall = f1_score(prediction, gold)
    metrics['sub_em'] += subem
    metrics['em'] += float(em)
    metrics['f1'] += f1
    metrics['prec'] += prec
    metrics['recall'] += recall
    metrics['total_num'] += 1
    return em, prec, recall


def calc_metrics(predictions, goldens):
    assert len(predictions) == len(goldens)
    metrics = {'f1': 0, 'prec': 0, 'recall': 0, 'em': 0, 'sub_em': 0, 'total_num': 0}
    for pred, gold in zip(predictions, goldens):
        update_answer(metrics, pred, gold)
    for k, _ in metrics.items():
        if k == 'total_num':
            continue
        metrics[k] = round((metrics[k]/metrics['total_num']), 2)
    return metrics


def get_pred(data, args, fout):
    model = args.model
    if "gpt" in model or "o1" in model or "o3" in model or "o4" in model or "gemini" in model or "claude" in model:
        tokenizer = tiktoken.encoding_for_model("gpt-4o-2024-08-06")
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    if args.api == "openai":
        from utils.openai_api import query_llm
    else:
        print(f"Invalid API: {args.api}")
        raise ValueError
    for item in tqdm(data):
        context = item["context"]
        template = template_0shot
        prompt = template.replace('$DOC$', context.strip()).replace('$Q$', item['input'].strip())
        output = query_llm(prompt, model, tokenizer, temperature=0.7, top_p=0.95, max_input_tokens=MAX_INPUT_LEN, max_new_tokens=MAX_OUTPUT_LEN)
        if output == '':
            continue
        response = output.strip()
        pred, _ = extract_solution(response)
        item['response'] = response
        item['answer'] = item["answers"][0]
        item['pred'] = extract_answer(pred) if pred else extract_answer(response)
        item['judge_f1'] = calc_metrics([item["pred"]], [item["answer"]])['f1'] if item["pred"] else 0
        item['judge_em'] = calc_metrics([item["pred"]], [item["answer"]])['em'] if item["pred"] else 0
        item['judge_sub_em'] = calc_metrics([item["pred"]], [item["answer"]])['sub_em'] if item["pred"] else 0
        fout.write(json.dumps(item, ensure_ascii=False) + '\n')
        fout.flush()
        print("="*40 + "New Item Start" + "="*40)
        print(item['response'])
        print("-"*80)
        print(item['pred'])
        print("-"*80)
        print(item['answer'])
        print("-"*80)
        print(item['judge_sub_em'])
        print("="*40 + "New Item End" + "="*40)


def main():
    os.makedirs(args.save_dir, exist_ok=True)
    print(args)
    out_file = os.path.join(args.save_dir, args.save_file + ".jsonl")

    dataset = concatenate_datasets([
            load_dataset("./datasets/longbench/LongBench", "2wikimqa", split="test"),
        ])
        
    print(f"original data len {len(dataset)}")
    # 通过深拷贝生成新数据集
    import copy
    dataset = [copy.deepcopy(item) for _ in range(args.sampling) for item in dataset]
    print(f"sampling data len {len(dataset)}")

    data_all = []
    for idx, item in enumerate(dataset):
        item["_id"] = idx  # 现在每个 item 是独立对象
        data_all.append(item)

    print(data_all[0]["_id"])
    print(data_all[-1]["_id"])

    # cache
    has_data = {}
    if os.path.exists(out_file):
        with open(out_file, encoding='utf-8') as f:
            has_data = {json.loads(line)["_id"]: 0 for line in f}
    fout = open(out_file, 'a', encoding='utf-8')
    data = []
    for item in data_all:
        if item["_id"] not in has_data:
            data.append(item)

    data_subsets = [data[i::args.n_proc] for i in range(args.n_proc)]
    processes = []
    for rank in range(args.n_proc):
        p = mp.Process(target=get_pred, args=(data_subsets[rank], args, fout))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", "-s", type=str, default="results/frames")
    parser.add_argument("--save_file", "-f", type=str, default="GLM-4-9B-Chat")
    parser.add_argument("--model", "-m", type=str, default="GLM-4-9B-Chat")
    parser.add_argument("--tokenizer", "-t", type=str, default="GLM-4-9B-Chat")
    parser.add_argument("--n_proc", "-n", type=int, default=16)
    parser.add_argument("--api", "-a", type=str, default="openai")
    parser.add_argument("--sampling", "-p", type=int, default=1)
    args = parser.parse_args()
    main()