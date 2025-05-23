import os, csv, json
import math
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
import numpy as np
from sympy import Rational

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


def round_up_to_decimal(number, decimals):
    factor = 10 ** decimals
    return math.ceil(number * factor) / factor


def is_number(string):
    pattern = r'^[-+]?(\d{1,3}(,\d{3})*|(\d+))(\.\d+)?$'
    match = re.match(pattern, string)
    return bool(match)


def is_scientific_number(string):
    pattern = r'^[-+]?\d+(\.\d+)?e[-]?\d+$'
    match = re.match(pattern, string)
    return bool(match)


def normalize(prediction: str):
    # Preprocessing the string [Stage 1]
    prediction = prediction.strip()
    prediction = prediction.rstrip('.')
    if not isinstance(prediction, str):
        prediction = str(prediction) if prediction is not None else '0'

    for money in ["£", "€", "¥", "million", "billion", "thousand", "US", "USD", "RMB"]:
        prediction = prediction.replace(money, '')
        
    # Replace special tokens
    if '=' in prediction:
        prediction = prediction.split('=')[-1].strip()
    if '≈' in prediction:
        prediction = prediction.split('≈')[-1].strip()
    if '`' in prediction:
        prediction = prediction.replace('`', '')
    if '%' in prediction:
        prediction = prediction.replace('%', '')
    if '$' in prediction:
        prediction = prediction.replace('$', '')
    if '°' in prediction:
        prediction = prediction.replace('°', '')

    # Detect the boolean keyword in the generation
    if prediction in ['true', 'yes', 'false', 'no']:
        if prediction == 'true' or prediction == 'yes':
            prediction = 'True'
        else:
            prediction = 'False'
    if 'True' in prediction or 'False' in prediction:
        prediction = 'True' if 'True' in prediction else 'False'

    # Detect the approximation keyword
    if 'approximately' in prediction:
        prediction = prediction.replace('approximately', '').strip()
    if ' or ' in prediction:
        prediction = prediction.split(' or ')[0]

    # Drop the units before and after the number
    if re.match(r'[-+]?(?:[\d,]*\.*\d+) [^0-9 ]+$', prediction):
        prediction = re.search(r'([-+]?(?:[\d,]*\.*\d+)) [^0-9 ]+$', prediction).group(1)
    if re.match(r'[^0-9 ]+ [-+]?(?:[\d,]*\.*\d+)$', prediction):
        prediction = re.search(r'[^0-9 ]+ ([-+]?(?:[\d,]*\.*\d+))$', prediction).group(1)
    if re.match(r'[-+]?(?:[\d,]*\.*\d+)[^\d]{1,2}$', prediction):
        prediction = re.search(r'([-+]?(?:[\d,]*\.*\d+))[^\d]{1,2}$', prediction).group(1)
    if re.match(r'[^-+\d]{1,2}(?:[\d,]*\.*\d+)$', prediction):
        prediction = re.search(r'[^-+\d]{1,2}((?:[\d,]*\.*\d+))$', prediction).group(1)

    # Preprocessing the number [Stage 1]
    if '10^' in prediction:
        prediction = re.sub(r'10\^(-?\d+)', r'math.pow(10, \1)', prediction)
    if ' x ' in prediction:
        prediction = prediction.replace(' x ', '*')
    if ' × ' in prediction:
        prediction = prediction.replace(' × ', '*')
    if is_number(prediction):
        prediction = prediction.replace(',', '')

    # Preprocessing the option [Stage 3]
    if '(a)' in prediction or '(b)' in prediction or '(c)' in prediction or '(d)' in prediction:
        prediction = '"' + re.search(r'\([a-d]\)', prediction).group(0) + '"'

    # If the prediction is empty, use dummy '0'
    if not prediction:
        prediction = '0'

    # Converting the string answer to a number/list/bool/option
    try:
        prediction = eval(prediction)
    except Exception:
        # TO CHECK
        prediction = 0 

    # Performing common type conversion
    if isinstance(prediction, (set, tuple)):
        prediction = list(prediction)
        if isinstance(prediction[0], complex):
            prediction = [tmp.real for tmp in prediction]
        elif isinstance(prediction[0], Rational):
            prediction = [float(tmp) for tmp in prediction]
    elif isinstance(prediction, np.ndarray):
        prediction = prediction.tolist()
    else:
        if isinstance(prediction, complex):
            prediction = prediction.real
        elif isinstance(prediction, Rational):
            prediction = float(prediction)

    return prediction


def extract_answer(response: str):
    """Parses the final answer from the model's response text.
    
    Args:
        response: Text extracted from the model's response
        
    Returns:
        The final answer as a numeric value (string), or None if not found
    """
    # Remove any asterisks or other unwanted characters
    response = response.replace('*', '')
    response = response.replace('(', '')
    response = response.replace(')', '')
    
    # Search for the pattern 'the answer is {final answer}.'
    match = re.search(r'the answer is (\=?\≈?\`?\%?\$?\°?\£?\€?\¥?-?[0-9\.,]+)', response, re.IGNORECASE)
    
    if match:
        # Remove commas from the matched number (if any)
        res = match.group(1).replace(',', '').rstrip('.')
        return res
    else:
        return None


def within_eps(pred: float, gt: float):
    eps = abs(gt) * 0.0015
    if pred >= gt - eps and pred <= gt + eps:
        return True
    else:
        return False


def compare_two_numbers(p, gt):
    if isinstance(p, int) or isinstance(p, float):
        pass
    elif isinstance(p, list) or isinstance(p, bool) or isinstance(p, str):
        return False
    elif isinstance(p, tuple) or isinstance(p, complex) or isinstance(p, dict):
        return False
    else:
        raise ValueError(p)

    v1, v2 = max(abs(gt), abs(p)), min(abs(gt), abs(p))
    if (v1 !=0 and v2 != 0) and int(math.log10(v1 / v2)) == math.log10(v1 / v2):
        return True

    if v2 <= v1 / 50 and within_eps(pred=v2*100, gt=v1):
        return True
    elif v2 <= v1 / 500 and within_eps(pred=v2*1000, gt=v1):
        return True
    elif v2 <= v1 / 50000 and within_eps(pred=v2*100000, gt=v1):
        return True

    if round_up_to_decimal(v1, 2) == round_up_to_decimal(v2, 2):
        return True

    return within_eps(pred=p, gt=gt)


def get_acc(prediction, gt, cot=True):
    print(f"get_acc({prediction}, {gt})")
    # try:
    if cot:
        prediction = normalize(prediction)
        gt = normalize(gt)
    else:
        prediction = float(prediction)
    print(f"after normalize pre = {prediction}")
    print(f"after normalize gt = {gt}")
    
    answer_type = type(gt).__name__
    print(f"answer_type::{answer_type}")
    assert answer_type in ["int", "float", "float64", "bool"], answer_type
    if isinstance(prediction, (str, int, float, bool)) or isinstance(prediction, list):
        # Comparing prediction against the reference
        if answer_type in ['bool']:
            acc = int(prediction == gt)
        elif answer_type == 'int':
            acc = int(compare_two_numbers(prediction, gt))
        elif answer_type == 'float' or answer_type == 'float64':
            acc = int(compare_two_numbers(prediction, gt))
        else:
            acc = 0
    else:
        acc = 0
        print("Error: ", prediction, type(prediction))
    return acc
    # except Exception as e:
    #     print(f"get_acc error::{e}")
    #     return 0
    

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
        context = '\n'.join(item['paragraphs'])
        template = template_0shot
        prompt = template.replace('$DOC$', context.strip()).replace('$Q$', item['question'].strip())
        output = query_llm(prompt, model, tokenizer, temperature=0.7, top_p=0.95, max_input_tokens=MAX_INPUT_LEN, max_new_tokens=MAX_OUTPUT_LEN)
        if output == '':
            continue
        response = output.strip()
        pred, _ = extract_solution(response)
        item['response'] = response
        item['pred'] = extract_answer(pred) if pred else extract_answer(response)
        item["answer"] = extract_answer(f"Therefore, the answer is {str(item['ground_truth'])}.")
        item['judge'] = get_acc(item['pred'], item["answer"]) if item['pred'] else 0
        fout.write(json.dumps(item, ensure_ascii=False) + '\n')
        fout.flush()
        print("="*40 + "New Item Start" + "="*40)
        print(item['response'])
        print("-"*80)
        print(item['pred'])
        print("-"*80)
        print(item['answer'])
        print("-"*80)
        print(item['judge'])
        print("="*40 + "New Item End" + "="*40)


def search_item(question, context, prompt_list):
    for prompt in prompt_list:
        if question.strip() in prompt and context.strip() in prompt:
            return True
    return False


def main():
    os.makedirs(args.save_dir, exist_ok=True)
    print(args)
    out_file = os.path.join(args.save_dir, args.save_file + ".jsonl")
    with open("./datasets/docmath/docmath_all_test_200.jsonl", "r", encoding="utf-8") as f:
        test_dataset_prompt_list = [json.loads(line)["prompt"][0]["content"] for line in f]
    test_dataset = []
    for file_name in ["complong", "compshort", "simplong", "simpshort"]:
        with open(f"./datasets/docmath/{file_name}_testmini.jsonl", "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                if search_item(item['question'], '\n'.join(item['paragraphs']), test_dataset_prompt_list):
                    item["domain"] = file_name + "_test"
                    test_dataset.append(item)
    dataset = test_dataset
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
    parser.add_argument("--save_dir", "-s", type=str, default="results/docmath")
    parser.add_argument("--save_file", "-f", type=str, default="GLM-4-9B-Chat")
    parser.add_argument("--model", "-m", type=str, default="GLM-4-9B-Chat")
    parser.add_argument("--tokenizer", "-t", type=str, default="GLM-4-9B-Chat")
    parser.add_argument("--n_proc", "-n", type=int, default=16)
    parser.add_argument("--api", "-a", type=str, default="openai")
    parser.add_argument("--sampling", "-p", type=int, default=1)
    args = parser.parse_args()
    main()