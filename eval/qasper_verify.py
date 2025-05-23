import os
import json
import openai
import time
import argparse
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


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
    client = openai.OpenAI(
        api_key=os.getenv('VERIFIER_API'),
        base_url=os.getenv('VERIFIER_URL'),
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
                seed=42
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


def call_reward_model(problem: str, model_answer: str, ground_truth: str, judge_model: str):
    question = problem
    orm_response = call_oai_rm_llm(
        system_prompt=GENERAL_ORM_PROMPT,
        prompt=ORM_USER_TEMPLATE.format(problem=question, answer_1=model_answer, answer_2=ground_truth),
        temperature=0.0,
        model_id=judge_model,
        retry_count=3
    )
    if "YES" in orm_response:
        return 1.0
    else:
        return 0.0


def process_single_item(item, judge_model):
    """处理单个数据项的函数"""
    question = item["input"]
    answer = item["answer"]
    pred = item["pred"]
    rm_score = call_reward_model(question, pred, answer, judge_model)
    
    item["judge_gpt"] = rm_score
    
    return item, pred, answer, rm_score


def main():
    os.makedirs(args.save_dir, exist_ok=True)
    print(args)
    in_file = os.path.join(args.save_dir, args.save_file + ".jsonl")
    out_file = os.path.join(args.save_dir, args.save_file + ".jsonl")

    # cache
    if os.path.exists(in_file):
        with open(in_file, encoding='utf-8') as f:
            result_dataset = [json.loads(line) for line in f]
    
    new_result_data = []
    max_workers = args.batch_size  # 可以根据需要调整并发数

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_item = {
            executor.submit(process_single_item, item, args.judge_model): item 
            for item in result_dataset
        }

        # 使用tqdm显示进度
        with tqdm(total=len(result_dataset)) as pbar:
            for future in as_completed(future_to_item):
                try:
                    item, pred, answer, rm_score = future.result()
                    new_result_data.append(item)
                    
                    # 打印处理结果
                    print("="*40 + "New Item" + "="*40)
                    print(f"Pred: {pred}")
                    print(f"Ans: {answer}")
                    print(f"RM Score: {rm_score}")
                    
                    pbar.update(1)
                except Exception as e:
                    print(f"处理出错: {str(e)}")
                    pbar.update(1)

    print(len(new_result_data))

    # print rm results
    rm_results = sum([max(item.get("judge_gpt", None), item.get("judge_sub_em", None)) for item in new_result_data]) / len(new_result_data)
    print(f"RM Results: {rm_results}")

    # 保存结果
    with open(out_file, "w", encoding='utf-8') as f:
        for item in new_result_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", "-s", type=str, default="results/qasper")
    parser.add_argument("--save_file", "-f", type=str, default="GLM-4-9B-Chat")
    parser.add_argument("--judge_model", "-j", type=str, default="deepseek-v3")
    parser.add_argument("--batch_size", "-b", type=int, default=1)
    args = parser.parse_args()
    main()