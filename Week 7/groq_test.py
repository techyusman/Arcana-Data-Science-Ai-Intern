import os
import sys

from groq import Groq
from dotenv import load_dotenv


load_dotenv()
sys.stdout.reconfigure(encoding="utf-8")

if not os.getenv("GROQ_API_KEY"):
    raise RuntimeError(
        "GROQ_API_KEY is not set. In PowerShell, run: "
        '$env:GROQ_API_KEY = "your-groq-key"'
    )

client = Groq()

response = client.chat.completions.create(
    model="openai/gpt-oss-20b",
    messages=[
        {
            "role": "system",
            "content": "You are a concise programming tutor.",
        },
        {
            "role": "user",
            "content": "Explain Python dictionaries with one example.",
        },
    ],
    temperature=0.3,
    max_tokens=200,
)

print(response.choices[0].message.content)
