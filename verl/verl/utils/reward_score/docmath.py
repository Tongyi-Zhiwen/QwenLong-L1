import os
import re
from typing import Dict, Tuple, Optional
import math
from sympy import Rational
import numpy as np
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

    try:
        v1, v2 = max(abs(gt), abs(p)), min(abs(gt), abs(p))
        if (v1 !=0 and v2 != 0) and int(math.log10(v1) - math.log10(v2)) == (math.log10(v1) - math.log10(v2)):
            return True

        if v2 <= v1 / 50 and within_eps(pred=v2*100, gt=v1):
            return True
        elif v2 <= v1 / 500 and within_eps(pred=v2*1000, gt=v1):
            return True
        elif v2 <= v1 / 50000 and within_eps(pred=v2*100000, gt=v1):
            return True

        if round_up_to_decimal(v1, 3) == round_up_to_decimal(v2, 3):
            return True

        return within_eps(pred=p, gt=gt)
    except OverflowError:
        return False

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
            acc = get_acc(pred_status, gt_status)
            if acc < 1.0 and os.getenv('LLM_JUDGE') == "Y":
                rm_acc = call_reward_model(prompt_str, pred_status, gt_status)
                print(f"  RM Score: {rm_acc}")
                acc = max(acc, rm_acc)
            answer_score = acc
            print(f"  Answer Score: {answer_score}")
        else:
            answer_score = 0
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
