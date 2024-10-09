import io
import torch
import logging

from fastapi import FastAPI, UploadFile, File

from ray import serve

from transformers import pipeline

logger = logging.getLogger("ray.serve")

app = FastAPI()


@serve.deployment(name="STT")
@serve.ingress(app)
class SpeechToText:
    def __init__(
        self,
    ):
        logger.info(f"Loading OpenAI Whisper model.")
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self._asr = pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-small",
            chunk_length_s=30,
            device=device,
        )

    @app.post("/")
    async def transcribe(self, file: UploadFile = File(...)):
        audio = await file.read()
        buffer = io.BytesIO(audio)
        transcription = self._asr(buffer, batch_size=8)["text"]
        return {"text": transcription}


app = SpeechToText.options(route_prefix="/transcribe").bind()
