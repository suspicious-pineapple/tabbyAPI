"""Tokenization types"""

from pydantic import BaseModel, Field, AliasChoices
from typing import List, Union, Dict, Optional

from endpoints.OAI.types.chat_completion import ChatCompletionMessage


class CommonTokenRequest(BaseModel):
    """Represents a common tokenization request."""

    add_bos_token: bool = True
    encode_special_tokens: bool = True
    decode_special_tokens: bool = True

    def get_params(self):
        """Get the parameters for tokenization."""
        return {
            "add_bos_token": self.add_bos_token,
            "encode_special_tokens": self.encode_special_tokens,
            "decode_special_tokens": self.decode_special_tokens,
        }


class TokenEncodeRequest(CommonTokenRequest):
    """Represents a tokenization request."""

    text: Union[str, List[ChatCompletionMessage]]
    template_vars: Optional[dict] = Field(
        default={},
        validation_alias=AliasChoices("template_vars", "chat_template_kwargs"),
        description="Aliases: chat_template_kwargs",
    )

class TokenEncodeResponse(BaseModel):
    """Represents a tokenization response."""

    tokens: List[int]
    length: int


class TokenDecodeRequest(CommonTokenRequest):
    """ " Represents a detokenization request."""

    tokens: List[int]


class TokenDecodeResponse(BaseModel):
    """Represents a detokenization response."""

    text: str


class TokenCountResponse(BaseModel):
    """Represents a token count response."""

    length: int
