from groq import Groq
from dotenv import load_dotenv

load_dotenv()
client = Groq()
completion = client.chat.completions.create(
    model="openai/gpt-oss-120b",
    messages=[
      {
        "role": "user",
        "content": "what is AI and why does it take engineers job?"
      }
    ]
)
print(completion)