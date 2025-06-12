import pathlib
from fastapi import Request
from peft import PeftModel
import torch
from loguru import logger
from common.tabby_config import config

from transformers import WhisperFeatureExtractor, WhisperForCausalLM, WhisperForConditionalGeneration, WhisperProcessor, WhisperTokenizer
#from unsloth import FastModel
from transformers.pipelines import pipeline

from endpoints.OAI.types.whisper import UsageInfo, WhisperObject, WhisperRequest, WhisperResponse




#whisper_model = WhisperForConditionalGeneration.from_pretrained("")

#model, tokenizer = FastModel.from_pretrained(
#    model_name = "D:\models\whisper-vc-merged",
#    dtype = None, # Leave as None for auto detection
#    load_in_4bit = False, # Set to True to do 4bit quantization which reduces memory
#    auto_model = WhisperForConditionalGeneration,
#    whisper_language = "English",
#    whisper_task = "transcribe",
#    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
#)
#FastModel.for_inference(model)
#model.eval()

#model, tokenizer = WhisperForConditionalGeneration.from_pretrained("D:\models\whisper-vc-merged")
#model, tokenizer = WhisperForConditionalGeneration.from_pretrained("D:\models\whisper-vc-merged")

#processor = WhisperProcessor.from_pretrained("D:\models\whisper-vc-merged")


#transcriber = pipeline(
#    "automatic-speech-recognition",
#    model=model,
#    tokenizer=tokenizer,
#    feature_extractor=model.tokenizer.feature_extractor,
#    processor=tokenizer,
#    return_language=True,
#    torch_dtype=torch.bfloat16  # Remove the device parameter
#)



model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3", low_cpu_mem_usage=True)
peft_model = PeftModel.from_pretrained(model, "D:\models\whisper-large-v3-lora")
featureextractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-large-v3")
merged_model = peft_model.merge_and_unload()

transcriber = pipeline(
    #model="openai/whisper-large-v3",
    model=merged_model,
    tokenizer="openai/whisper-large-v3",
    task="automatic-speech-recognition",
    feature_extractor=featureextractor, 
    torch_dtype=torch.bfloat16,
    device=0,
)

#transcriber = pipeline(
#    model="D:\models\whisper-vc-merged",
#
#    task="automatic-speech-recognition", 
#    torch_dtype=torch.bfloat16,
#    return_language=True,
#    #device=0,
#)

#from transformers import WhisperProcessor
#processor = WhisperProcessor.from_pretrained("unsloth/whisper-large-v3", task="automatic-speech-recognition")
#transcriber = pipeline(
#    "automatic-speech-recognition",
#    model="unsloth/whisper-large-v3",
#    tokenizer=processor.tokenizer,
#    feature_extractor=processor.feature_extractor,
#    processor=processor,
#    return_language=True,
#    torch_dtype=torch.float16  # Remove the device parameter
#)


#transcriber.model.load_adapter(peft_model_id = "D:\models\whisper-vc-lora")

    #model="nyrahealth/CrisperWhisper",
    #model="D:\models\whisper-large-v3-turbo",


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