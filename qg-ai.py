from fastapi import FastAPI, UploadFile, Form
import openai
import os

openai.api_key = "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"

app = FastAPI()


@app.post("/transcribe")
def transcribe(file: UploadFile):
    try:
        contents = file.file.read()
        with open(file.filename, "wb") as f:
            f.write(contents)
    except Exception as e:
        return {"message": e}
    finally:
        file.file.close()

    return get_transcription(file)


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
