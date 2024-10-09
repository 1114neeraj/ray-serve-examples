import base64
import torch
import logging

from fastapi import FastAPI

from ray import serve
from starlette.requests import Request
from starlette.responses import Response

from transformers import pipeline
from datasets import load_dataset


logger = logging.getLogger("ray.serve")

app = FastAPI()


@serve.deployment(name="TTS")
@serve.ingress(app)
class TextToSpeech:
    def __init__(
        self,
    ):
        logger.info(f"Loading Microsoft SpeechT5 TTS model.")
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self._synthesiser = pipeline("text-to-speech", "microsoft/speecht5_tts")
        self._speaker_embedding = torch.tensor(
            load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")[7306][
                "xvector"
            ]
        ).unsqueeze(0)

    @app.post("/")
    async def synthesise(self, request: Request):
        request = await request.json()
        speech = self._synthesiser(
            request["text"],
            forward_params={"speaker_embeddings": self._speaker_embedding},
        )
        audio = base64.b64decode(speech["audio"])
        response = Response(
            content=audio,
            media_type="audio/wav",
            headers={"Content-Disposition": f'attachment; filename="output.wav"'},
        )
        return response


app = TextToSpeech.options(route_prefix="/synthesise").bind()
