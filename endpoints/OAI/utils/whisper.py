import pathlib
from fastapi import Request
import torch
from loguru import logger
from common.tabby_config import config

from transformers.pipelines import pipeline

from endpoints.OAI.types.whisper import UsageInfo, WhisperObject, WhisperRequest, WhisperResponse





transcriber = pipeline(
    task="automatic-speech-recognition",
    model="nyrahealth/CrisperWhisper",
    #model="D:\models\whisper-large-v3-turbo",
    torch_dtype=torch.bfloat16,
    device=0,
)


#import nemo.collections.asr as nemo_asr







#asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name="nvidia/parakeet-tdt-0.6b-v2")












async def get_whisper(data: WhisperRequest, request: Request) -> dict:

    logger.info(f"Received whisper request {request.state.id}")

    text = transcriber(data.input, generate_kwargs={"language": "en", "task":"transcribe"})
    #text = asr_model.transcribe([data.input])

    response = WhisperResponse(
        data=text.get("text"),
        model="a snail",
        usage=UsageInfo(prompt_tokens=0, total_tokens=0),
    )

    logger.info(f"Finished embeddings request {request.state.id}")

    return response