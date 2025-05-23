import os
import re
from typing import Dict, Tuple, Optional
import math
from sympy import Rational
import numpy as np
import sys
import re
import string
from collections import Counter, defaultdict
import pickle
from pathlib import Path
import jsonlines
import json
from tqdm import tqdm
import openai
import time

# For General ORM to verify correctness of LLM's solution. We disable this by default, as it doesn't help much.
GENERAL_ORM_PROMPT = """You are an expert in verifying if two answers are the same.
Your input is a problem and two answers, Answer 1 and Answer 2. You need to check if they are equivalent.
Your task is to determine if two answers are equivalent, without attempting to solve the original problem.
Compare the answers to verify they represent identical values or meaning, even when written in different forms or notations.

Your output must follow the following format:
1) Provide an explanation for why the answers are equivalent or not.
2) Then provide your final answer in the form of: [[YES]] or [[NO]]
"""

ORM_USER_TEMPLATE = """
Problem: {problem}
Answer 1: {answer_1}
Answer 2: {answer_2}
"""

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

def sub_em(prediction, ground_truth):
    ground_truth = normalize_answer(ground_truth)
    prediction = normalize_answer(prediction) 
    return (ground_truth in prediction) or (prediction in ground_truth)

def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))

def update_answer(metrics, prediction, gold):
    em = exact_match_score(prediction, gold)
    subem = sub_em(prediction, gold)

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

def extract_solution(solution_str: str) -> Tuple[Optional[str], str]:
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

def parse_model_answer(response: str) -> Optional[str]:
    """Parses the final answer from the model's response text.
    
    Args:
        response: Text extracted from the model's response
        
    Returns:
        The final answer as a numeric value (string), or None if not found
    """
    # Remove any asterisks or other unwanted characters
    response = response.replace('*', '')
    
    if "the answer is" in response:
        # Search for the pattern 'the answer is {final answer}.'
        # ans = response.rsplit("the answer is", 1)[-1].strip().strip('.')
        ans = response.rsplit("the answer is", 1)[-1].strip().replace("<｜Assistant｜>", '').replace("<｜end▁of▁sentence｜>", '').strip().strip('.').strip()
    else:
        ans = None

    return ans
    
def validate_response_structure(processed_str: str) -> bool:
    """Performs comprehensive validation of response structure.
    
    Args:
        processed_str: Processed response string from the model
        
    Returns:
        Boolean indicating whether all formatting requirements are met
    """
    print("\n[Structure Validation]")
    validation_passed = True

    # Check required tags
    tags = {
        'think_start': ('<think>', 1),
        'think_end': ('</think>', 1)
    }

    positions = {}
    for tag_name, (tag_str, expected_count) in tags.items():
        count = processed_str.count(tag_str)
        positions[tag_name] = pos = processed_str.find(tag_str)
        
        print(f"  {tag_str}: count={count}, position={pos}")
        
        if count != expected_count:
            print(f"  [Error] {tag_str} appears {count} times (expected {expected_count})")
            validation_passed = False

    # Verify tag order
    if (positions['think_start'] > positions['think_end']):
        print("  [Error] Incorrect tag order: Expected <think>...</think>")
        validation_passed = False
    else:
        print("  Tag sequence validation passed")

    return validation_passed

def call_oai_rm_llm(
    prompt: str,
    system_prompt: str,
    n: int = 1,
    temperature: float = 1.0,
    model_id: str = "gpt-4o",
    retry_count: int = 1000000000
) -> tuple[str, list[str]]:
    """Call OpenAI API with retry logic.

    Args:
        prompt: The text prompt to send to the model
        system_prompt: System instruction for the model
        n: Number of completions to generate
        temperature: Sampling temperature
        model_id: OpenAI model ID to use
        retry_count: Number of retries on rate limit errors

    Returns:
        Generated text(s) from the model
    """
    openai_api_key = "EMPTY"
    openai_api_base = f"http://{os.getenv('VERIFIER_HOST')}:{os.getenv('VERIFIER_PORT')}/v1"
    client = openai.OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )
    backoff = 1
    retry_count = int(retry_count)

    for _ in range(retry_count):
        try:
            response = client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                n=n,
            )
            break
        except Exception as exc:
            if "429" in str(exc):
                print("Retry due to rate limit: ", exc)
                time.sleep(backoff)
                backoff = min(backoff * 2, 64)  # Exponential backoff up to 64s
                continue
            print("Exception: ", exc)
            return []

    if n == 1:
        return response.choices[0].message.content
    return [choice.message.content for choice in response.choices]

def call_reward_model(problem: str, model_answer: str, ground_truth: str):
    start_index = problem.index("</text>")
    end_index = problem.index("Format your response as follows:")
    question = problem[start_index: end_index].replace("</text>", "").strip()
    orm_response = call_oai_rm_llm(
        system_prompt=GENERAL_ORM_PROMPT,
        prompt=ORM_USER_TEMPLATE.format(problem=question, answer_1=model_answer, answer_2=ground_truth),
        temperature=0.0,
        model_id=os.getenv("VERIFIER_PATH"),
        retry_count=5,
    )
    if "YES" in orm_response:
        return 1.0
    else:
        return 0.0

def compute_score(solution_str: str, 
                 ground_truth: Dict[str, str],
                 prompt_str: str,
                 format_reward: float = 0.0,
                 answer_reward: float = 1.0) :
    """Computes comprehensive score for model response.
    
    Args:
        solution_str: Raw model response string
        ground_truth: Dictionary containing ground truth data
        format_reward: Points awarded/deducted for format correctness
        answer_reward: Points awarded/deducted for answer correctness
        
    Returns:
        Total score (sum of format and answer rewards)
    """
    print("\n" + "="*80)
    print(" Processing New Sample ".center(80, '='))
    
    # Extract model answer
    answer_text, processed_str = extract_solution(solution_str)
    print(f"\n[Model Response]\n{processed_str}")
    # Validate answer content
    answer_score = 0
    if answer_text:
        pred_status = parse_model_answer(answer_text)
        gt_status = parse_model_answer(ground_truth)
        
        if pred_status:
            print(f"\n[Content Validation]")
            print(f"  Expected: {gt_status}")
            print(f"  Predicted: {pred_status}")
            metrics = calc_metrics([pred_status], [gt_status])
            metric = metrics['sub_em']
            if metric < 1.0 and os.getenv('LLM_JUDGE') == "Y":
                rm_metric = call_reward_model(prompt_str, pred_status, gt_status)
                print(f"  RM Score: {rm_metric}")
                metric = max(metric, rm_metric)
            answer_score = metric
            print(f"  Answer Score: {answer_score}")
        else:
            answer_score = 0.0
            print( "Fail to parse answer")
    else:
        print("\n[Content Validation] Skipped due to format errors or missing answer")
    print("\n" + "-"*80)
    print(f" Final Score ".center(80, '-'))
    print(f"  Answer: {answer_score}")
    print("="*80 + "\n")
    return {
        "score": answer_score,
        "acc": answer_score == 1.0,
        "pred": answer_text,
    }
