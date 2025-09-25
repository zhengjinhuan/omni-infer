import json
from abc import ABC, abstractmethod
from typing import Any, Optional, Type, Union
from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from http import HTTPStatus
from vllm.entrypoints.openai.protocol import ErrorResponse


class BaseValidator(ABC):
    def __init__(self,
                 param_name: str,
                 error_msg: Optional[str] = None
                 ):
        self.param_name = param_name
        self.error_msg = error_msg or f"{param_name} is not supported."

    @abstractmethod
    def validate(self, value: Any) -> Optional[str]:
        pass


class SupportedValidator(BaseValidator):
    def __init__(
            self,
            param_name: str,
            error_msg: Optional[str] = None,
            subfield: Union[list, None] = None
    ):
        super.__init__(param_name, error_msg)
        self.subfield = subfield

    def check_field(self, value: Any, curr_field: str = None):
        if isinstance(value, dict):
            return self.check_subfield_dict(value, curr_field)
        if isinstance(value, list):
            return self.check_subfield_list(value, curr_field)
        return None

    def check_subfield_dict(self, value: Any, curr_field: str):
        for param_name, val in value.items():
            curr_subfield = f"{curr_field}.{param_name}"
            if curr_subfield not in self.subfield:
                return f"{curr_subfield} is not supported."
            if isinstance(val, dict) or isinstance(val, list):
                if check_result := self.check_field(val, curr_subfield):
                    return check_result
        return None
    
    def check_subfield_list(self, value: Any, curr_field: str):
        for val in value:
            if isinstance(val, dict):
                return self.check_subfield_dict(val, curr_field)
        return None

    def validate(self, value: Any) -> Optional[str]:
        # The value must be included within the subfield.
        return self.check_field(value, self.param_name)
    

class UnsupportedValidator(BaseValidator):
    def validate(self, value: Any) -> Optional[str]:
        return self.error_msg


class RangeValidator(BaseValidator):
    def __init__(self,
                 param_name: str,
                 error_msg: Optional[str] = None,
                 min_val: Union[float, int, None] = None,
                 max_val: Union[float, int, None] = None,
                 type_: Optional[Type] = None
                 ):
        super.__init__(param_name, error_msg)
        self.min_val = min_val
        self.max_val = max_val
        self.type_ = type_
        
    def validate(self, value: Any) -> Optional[str]:
        if not isinstance(value, self.type_):
            return f"{self.param_name} must be of type {self.type_.__name__}"
        if not (self.min_val <= value <= self.max_val):
            return f"{self.param_name} must between {self.min_val} and {self.max_val}, got {value}."
        return None

class ValueValidator(SupportedValidator):
    def __init__(
            self,
            param_name: str,
            error_msg: Optional[str] = None,
            subfield: Union[list, None] = None,
            target_value: Any = None
    ):
        super.__init__(param_name, error_msg, subfield)
        self.target_value = target_value

    def validate(self, value):
        if error := super().validate(value):
            return error
        if value not in self.target_value:
            return self.error_msg or f"{self.param_name} only support the value in {self.target_value}."
        return None
    

VALIDATORS: dict[str, BaseValidator] = {
    "model": SupportedValidator("model"),
    "messages": SupportedValidator("messages", subfield=["name", "role", "tool_call_id", "content", "prefix",
                                                         "refusal", "partial", "tool_calls", "tool_calls.type",
                                                         "tool_calls.id", "tool_calls.function", 
                                                         "tool_calls.function.arguments",
                                                         "tool_calls.function.name"
                                                        ]),
    "stream": SupportedValidator("stream"),
    "stream_options": SupportedValidator("stream_options", subfield=["include_usage", "chunk_include_usage"]),
    "tool_choice": ValueValidator("tool_choice", target_value=["auto"]),
    "tools": SupportedValidator("tools", subfield=["type", "function", "function.description", "function.name",
                                                   "function.parameters", "function.strict"]),
    "chat_template": SupportedValidator("chat_template"),
    "chat_template_kwargs": SupportedValidator("chat_template_kwargs"),
    "top_p": SupportedValidator("top_p"),
    "frequency_penalty": SupportedValidator("frequency_penalty"),
    "presence_penalty": SupportedValidator("presence_penalty"),
    "temperature": SupportedValidator("temperature"),
    "seed": SupportedValidator("seed"),
    "stop": SupportedValidator("stop"),
    "logit_bias": SupportedValidator("logit_bias"),
    "max_tokens": SupportedValidator("max_tokens"),
    "add_generation_prompt": SupportedValidator("add_generation_prompt"),
    "add_special_tokens": SupportedValidator("add_special_tokens"),
    "allowed_token_ids": SupportedValidator("allowed_token_ids"),
    "bad_words": SupportedValidator("bad_words"),
    "continue_final_message": SupportedValidator("continue_final_message"),
    "detokenize": SupportedValidator("detokenize"),
    "echo": SupportedValidator("echo"),
    "ignore_eos": SupportedValidator("ignore_eos"),
    "include_stop_str_in_output": SupportedValidator("include_stop_str_in_output"),
    "length_penalty": SupportedValidator("length_penalty"),
    "max_completion_tokens": SupportedValidator("max_completion_tokens"),
    "min_p": SupportedValidator("min_p"),
    "min_tokens": SupportedValidator("min_tokens"),
    "prompt": SupportedValidator("prompt"),
    "repetition_penalty": SupportedValidator("repetition_penalty"),
    "request_id": SupportedValidator("request_id"),
    "skip_special_tokens": SupportedValidator("skip_special_tokens"),
    "spaces_between_special_tokens": SupportedValidator("spaces_between_special_tokens"),
    "stop_token_ids": SupportedValidator("stop_token_ids"),
    "enable_thinking": SupportedValidator("enable_thinking")
}

CUSTOM_VALIDATORS: dict[str, BaseValidator] = {
    "continue_final_message": ValueValidator("add_generation_prompt", target_value=[False],
                                             error_msg="continue_final_message can only be passed when " \
                                                       "add_generation_prompt is False."),
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
                if error := validator.validate(value):
                    return self.create_error_response(status_code, error)
                if validator := CUSTOM_VALIDATORS.get(param_name):
                    if error := validator.validate(json_load[validator.param_name]):
                        return self.create_error_response(status_code, error)
                
        return await call_next(request)