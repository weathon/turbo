import os
import json
import base64
import io
from PIL import Image
from pydantic import BaseModel
from openai import OpenAI
import numpy as np
import random
import dotenv
# do not do that in first one? 

#          positive  negative
# ours     1.622222  1.800000
# vanilla  1.788889  1.288889

dotenv.load_dotenv()

client = OpenAI()
class Score(BaseModel):
    first_image_positive_and_quality: float
    first_image_negative: float
    second_image_positive_and_quality: float
    second_image_negative: float


def ask_gpt(image1: Image.Image, image2: Image.Image, pos: str, neg: str) -> list[Score]:
    if random.random() > 0.5:
      image1, image2 = image2, image1
      swapped = True
    else:
      swapped = False

    # Encode both images
    buf1 = io.BytesIO()
    image1 = image1.resize((448, 448))
    image1.save(buf1, format="PNG")
    b64_1 = base64.b64encode(buf1.getvalue()).decode("utf-8")

    buf2 = io.BytesIO()
    image2.save(buf2, format="PNG")
    image2 = image2.resize((448, 448))
    b64_2 = base64.b64encode(buf2.getvalue()).decode("utf-8")

    prompt = (
        f"You will get 2 images, you should rate them based on how well they follow the positive prompt and quality of the image ({pos}),"
        f"and how well they AVOID the negative prompt ({neg}), that means the more *unrelated* the negative prompt is to the image the higher score, only give 2 if the negative item is completely avoided without any artifacts, "
        f"For each item you can rate from 0.0-2.0, 0 means bad and 2 means good. "
        f"When the negative prompt is contradicted with positive prompt or quality following the negative prompt should not be a reason to decrease score for the positive and quality score. (such as negative prompt being 'car' while positive being 'New York street', showing no cars should not be a reason to decrease score for positive, even though it does not look like a New York street anymore) "
        f"The scoring is releative, so if image 1 is much better than image 2, image 1 should get a score higher than image 2. In this case, 1 or 1.5 means good but not as good as the other one that gets a 2. Your score should be as fine grained to 0.1"
    ) 

    completion = client.beta.chat.completions.parse(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_1}"}},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_2}"}},
            ]},
        ],
        response_format=Score,
        temperature=0.0,
    )

    answer = completion.choices[0].message.parsed

    if not swapped:
      answer = np.array(((answer.first_image_positive_and_quality, answer.second_image_positive_and_quality), (answer.first_image_negative, answer.second_image_negative)))
    else:
      answer = np.array(((answer.second_image_positive_and_quality, answer.first_image_positive_and_quality), (answer.second_image_negative, answer.first_image_negative)))
    return answer
