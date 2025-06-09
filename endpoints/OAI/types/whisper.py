from typing import List, Optional, Union

from pydantic import BaseModel, Field


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    total_tokens: int = 0
    completion_tokens: Optional[int] = 0


class WhisperRequest(BaseModel):
    input: Union[str, List[str]] = Field(
        ..., description="Audio to transcribe. base64 encoded WAV file, 16kHz"
    )
    model: Optional[str] = Field(
        None,
        description="Name of the whisper model to use. "
        "If not provided, the default model will be used.",
    )


class WhisperObject(BaseModel):
    text: str = Field(
        ..., description="the text."
    )



class WhisperResponse(BaseModel):
    #object: str = Field("list", description="Type of the response object.")
    data: str = Field(..., description="The transcribed text.")
    
    model: str = Field(..., description="Name of the whisper model used.")
    usage: UsageInfo = Field(..., description="Information about token usage.")
