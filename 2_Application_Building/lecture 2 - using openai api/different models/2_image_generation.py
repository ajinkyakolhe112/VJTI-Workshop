import openai

client   = openai.OpenAI()

PROMPT   = "a cat dancing on a dog"
response = client.images.generate( model="dall-e-3", prompt= PROMPT,
  size="1024x1024", quality="standard", n=1,)

image_url = response.data[0].url

print(image_url)