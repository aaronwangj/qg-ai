from fastapi import FastAPI, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import openai
import os

openai.api_key = os.environ.get("OPENAI_KEY")

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def test():
    return {"message": "quantguide.io"}


@app.post("/ai")
def ai(file: UploadFile, prompt: str = Form(...)):
    try:
        contents = file.file.read()
        with open(file.filename, "wb") as f:
            f.write(contents)
    except Exception as e:
        return {"message": e}
    finally:
        file.file.close()

    transcription = get_transcription(file)

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {
                "role": "system",
                "content": "You are a professor. Grade the response out of 10 and explain your reasoning to the following question: " + prompt,
            },
            {
                "role": "user",
                "content": transcription,
            },
        ],
        temperature=0.8,
        max_tokens=1024,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
    )

    os.remove(file.filename)

    return response["choices"][0]["message"]["content"]


def get_transcription(file):
    with open(file.filename, "rb") as f:
        return openai.Audio.transcribe("whisper-1", f)["text"]

    # song = AudioSegment.from_wav(file.filename)
        
