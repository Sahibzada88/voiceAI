from fastapi import FastAPI
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import speech_recognition as sr
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

app = FastAPI()

# CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

TTS_MODEL = "playai-tts"
VOICE = "Arista-PlayAI"
RESPONSE_FORMAT = "wav"
message_history = [{"role": "system", "content": "You are a friendly assistant."}]
audio_counter = 0  # Counter to track responses

@app.post("/process")
async def process_speech():
    global audio_counter
    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    with mic as source:
        recognizer.adjust_for_ambient_noise(source, duration=0.5)
        print("🎙️ Listening...")
        audio = recognizer.listen(source)

    try:
        user_text = recognizer.recognize_google(audio)
        print("User said:", user_text)

        # LLM response
        message_history.append({"role": "user", "content": user_text})
        completion = client.chat.completions.create(
            messages=message_history,
            model="llama-3.3-70b-versatile"
        )
        reply = completion.choices[0].message.content
        message_history.append({"role": "assistant", "content": reply})

        # Generate TTS with unique filename
        filename = f"response{audio_counter}.wav"
        tts_response = client.audio.speech.create(
            model=TTS_MODEL,
            voice=VOICE,
            input=reply,
            response_format=RESPONSE_FORMAT
        )
        tts_response.write_to_file(filename)
        audio_counter += 1

        return {
            "user_text": user_text,
            "response_text": reply,
            "audio_url": f"/audio/{filename}"
        }

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/audio/{filename}")
async def get_audio(filename: str):
    file_path = filename
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type="audio/wav")
    else:
        return JSONResponse(content={"error": "File not found."}, status_code=404)
