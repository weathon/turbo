from openai import OpenAI
import base64
import io
import dotenv

dotenv.load_dotenv()

client = OpenAI()
def mod(image):
  buffer = io.BytesIO()
  image.save(buffer, format="JPEG")
  data = base64.b64encode(buffer.getvalue()).decode("utf-8")

  response = client.moderations.create(
      model="omni-moderation-latest",
      input=[
          {
              "type": "image_url",
              "image_url": {
                  "url": f"data:image/jpeg;base64,{data}"
              }
          },
      ],
  )

  return response.results[0].category_scores.sexual