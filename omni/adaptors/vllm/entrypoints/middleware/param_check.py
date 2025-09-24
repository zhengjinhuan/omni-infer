import json
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Type, Union
from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from http import HTTPStatus
from vllm.entrypoints.openai.protocol import ErrorResponse


class BaseValidator(ABC):
    def __init__(self,
                 param_name: str,
                 min_val:Union[float, int, None] = None,
                 max_val:Union[float, int, None] = None,
                 type_: Optional[Type] = None,
                 error_msg: Optional[str] = None,
                 subfield: Optional[list[str]] = None
                 ):
        self.param_name = param_name
        self.error_msg = error_msg or f"{param_name} is not supported."
        self.min_val = min_val
        self.max_val = max_val
        self.type_ = type_
        self.subfield = subfield

        @abstractmethod
        def validate(self, value: Any) -> Optional[str]:
            pass


class SupportedValidator(BaseValidator):
    def check_subfield_dict(self, value):
        for param_name, _ in value.items():
            if param_name not in self.subfield:
                return f"{self.param_name}:{param_name} is not supported."
        return None
    
    def check_subfield_list(self, value):
        for val in value:
            if isinstance(val, Dict):
                for param_name, _ in val.items():
                    if param_name not in self.subfield:
                        return f"{self.param_name}:{param_name} is not supported."
        return None

    def validate(self, value: Any) -> Optional[str]:
        # The value must be included within the subfield.
        if isinstance(value, Dict):
            return self.check_subfield_dict(value)
        if isinstance(value, list):
            return self.check_subfield_list(value)
        return None
    

class UnsupportedValidator(BaseValidator):
    def validate(self, value: Any) -> Optional[str]:
        return self.error_msg


class RangeValidator(BaseValidator):
    def validate(self, value: Any) -> Optional[str]:
        if not isinstance(value, self.type_):
            return f"{self.param_name} must be of type {self.type_.__name__}"
        if not (self.min_val <= value <= self.max_val):
            return f"{self.param_name} must between {self.min_val} and {self.max_val}, got {value}."
        return None


VALIDATORS: Dict[str, BaseValidator] = {
    "model": SupportedValidator("model"),
    "messages": SupportedValidator("messages", subfield=["role", "content"]),
    "stream": SupportedValidator("stream"),
    "stream_options": SupportedValidator("stream_options", subfield=["include_usage"]),
    "top_k": SupportedValidator("top_k"),
    "top_p": SupportedValidator("top_p"),
    "temperature": SupportedValidator("temperature"),
    "stop": SupportedValidator("stop"),
    "max_tokens": SupportedValidator("max_tokens"),
    "tool_choice": SupportedValidator("tool_choice"),
    "tools": SupportedValidator("tools"),
    "frequency_penalty": SupportedValidator("frequency_penalty"),
    "presence_penalty": SupportedValidator("presence_penalty"),
    "n": SupportedValidator("n"),
    "length_penalty": SupportedValidator("length_penalty"),
    "repetition_penalty": SupportedValidator("repetition_penalty"),
    "chat_template_kwargs": SupportedValidator("chat_template_kwargs", subfield=["thinking", "enable_thinking"]),
}


class ValidateSamplingParams(BaseHTTPMiddleware):
    def create_error_response(self, status_code, error):
        return JSONResponse(
            status_code=status_code,
            content=ErrorResponse(
                message=str(error),
                type="BadRequestError",
                code=status_code.value
            ).model_dump()
        )
    
    async def dispatch(self, request: Request, call_next):
        if request.method == "POST" and request.url.path in ("/v1/completions", "/v1/chat/completions"):
            body = await request.body()
            if not body:
                return await call_next(request)
            
            try:
                json_load = json.loads(body.decode("utf-8"))
            except json.JSONDecodeError:
                return await call_next(request)
            
            status_code = HTTPStatus.BAD_REQUEST
            for param_name, value in json_load.items():
                validator = VALIDATORS.get(param_name)
                if not validator:
                    return self.create_error_response(status_code, f"{param_name} is not supported.")
                error = validator.validate(value)
                if error is not None:
                    return self.create_error_response(status_code, error)
                
        return await call_next(request)