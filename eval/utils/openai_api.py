import os
from openai import OpenAI
import time

URL = f"http://{os.getenv('SERVE_HOST')}:{os.getenv('SERVE_PORT')}/v1"
API_KEY = "token-abc123"


def query_llm(prompt, model, tokenizer, temperature=0.7, top_p=0.95, max_input_tokens=120000, max_new_tokens=10000, stop=None):
    client = OpenAI(
        base_url=URL,
        api_key=API_KEY,
        timeout=1800
    )
    max_len = max_input_tokens
    input_ids = tokenizer.encode(prompt)
    if len(input_ids) > max_len:
        input_ids = input_ids[:max_len//2] + input_ids[-max_len//2:]
        prompt = tokenizer.decode(input_ids, skip_special_tokens=True)
    tries = 0
    while tries < 5:
        tries += 1
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_new_tokens,
            )
            return completion.choices[0].message.content
        except KeyboardInterrupt as e:
            raise e
        except Exception as e:
            print("Error Occurs: \"%s\"        Retry ..."%(str(e)))
            time.sleep(1)
    print("Max tries. Failed.")
    return ''