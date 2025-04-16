import openai
import os, dotenv

dotenv.load_dotenv()
openai.api_key = os.environ['OPENAI_API_KEY']

client = openai.OpenAI()
# client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
# client = OpenAI(api_key="<PRIVATE_KEY>")

PROMPT = "What are the foundational models in Deep Learning"
completion = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            # Roles: System or User or Assistant
            "role": "user",
            "content": PROMPT,
        }
    ]
)

print(completion.choices[0].message.content)
# print(completion)