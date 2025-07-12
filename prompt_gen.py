from openai import OpenAI
from pydantic import BaseModel
import dotenv
import tqdm

client = OpenAI()

class Prompt(BaseModel):
    prompt: str
    missing_element: str

with open("test_prompt.jsonl", "w") as f:
    for i in tqdm.tqdm(range(200)):
        response = client.responses.parse(
            model="o3",
            input="Write a detailed but not long prompt for generating an image of any object, creature, or scene that is missing a typically expected component or feature. The result should remain visually and physically coherent. It must not create a contradiction or impossibility. For example, acceptable prompts might be: 'a violin without strings', 'a spider with no legs', or 'a staircase with no steps'. Do not include self-contradictory cases like 'a crowded dining hall without people' or 'a lit candle with no flame'. The prompt should be physically plausible and clearly describe both the subject and the missing element. The response should be a descriptive text not a imperative sentences. Put the description and the missing element in the two sentences. The first sentence should be the description of the object, creature, or scene, and the second sentence (just a few words) should be the missing element (e.g. 'cars', 'strings'). The missing element should not be mentioned in the first sentence.",
            text_format=Prompt,
        )
        
        f.write(response.output_text + "\n")
        