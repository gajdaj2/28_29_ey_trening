from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()


def openai_test(prompt:str,system:str)->str:
  response = client.responses.create(
    model="gpt-4o",
    input=[
      {
        "role": "system",
        "content": [
          {
            "type": "input_text",
            "text": f"{system}"
          }
        ]
      },
      {
        "role": "user",
        "content": [
          {
            "type": "input_text",
            "text": f"{prompt}"
          }
        ]
      },
    ],
    temperature=1,
    max_output_tokens=2048,
    top_p=1,
  )

  return response.output[0].content[0].text


if __name__ == '__main__':
    print(openai_test("Kim jesteś","Pracujesz jako asystent"))
