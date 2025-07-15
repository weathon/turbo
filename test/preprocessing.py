from pydantic import BaseModel
import tqdm
from openai import OpenAI
import pandas as pd

with open("prompts.csv", "r") as f:
    prompts = f.readlines()
    
client = OpenAI()

class Prompt(BaseModel):
    prompt: str
    missing_element: str

    
with open("test_prompt.jsonl", "w") as f:
    for prompt in tqdm.tqdm(prompts):
        response = client.responses.parse(
            model="gpt-4o",
            input="You will be given a sentence with negation in it, make it into two parts, the prompt and the missing element. Do not mention anything about the missing item in the prompt. The missing item should be a bag of word(s), it should not have 'no', or other negation in it. Do not rewrite the prompt to exclude the missing element longer. The sentence is: " + prompt.strip() + "\n Example: input: a dog with no ears. Output: prompt: a dog, missing_element: ear. The positive prompt should be a statement and does not have 'generate x', 'imagine x'",
            text_format=Prompt,
        )
        f.write(response.output_parsed.json() + "\n")
        f.flush()