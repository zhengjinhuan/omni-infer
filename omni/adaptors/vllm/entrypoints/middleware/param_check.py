import json
import os
from abc import ABC, abstractmethod
from typing import Any, Optional, Type, Union
from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from http import HTTPStatus
from vllm.entrypoints.openai.protocol import ErrorResponse

TYPE_MAPPING = {
    "int": int,
    "float": float,
    "str": str,
    "bool": bool,
    "list": list,
    "dict": dict
}

class BaseValidator(ABC):
    def __init__(self,
                 param_name: str,
                 error_msg: Optional[str] = None
                 ):
        self.param_name = param_name
        self.error_msg = error_msg

    @abstractmethod
    def validate(self, value: Any) -> Optional[str]:
        pass


class SupportedValidator(BaseValidator):
    def __init__(
            self,
            param_name: str,
            error_msg: Optional[str] = None,
            subfield: list = [],
            skip_subfield_check: list = []
    ):
        super().__init__(param_name, error_msg)
        self.subfield = subfield
        self.skip_subfield_check = skip_subfield_check

    def check_field(self, value: Any, curr_field: str = ""):
        if not curr_field in self.skip_subfield_check:
            if isinstance(value, dict):
                return self.check_subfield_dict(value, curr_field)
            if isinstance(value, list):
                return self.check_subfield_list(value, curr_field)
        return None

    def check_subfield_dict(self, value: Any, curr_field: str):
        for param_name, val in value.items():
            curr_subfield = f"{curr_field}.{param_name}"
            if curr_subfield not in self.subfield:
                return self.error_msg or f"`{curr_subfield}` is not supported."
            if isinstance(val, (dict, list)):
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

class RangeValidator(BaseValidator):
    def __init__(self,
                 param_name: str,
                 error_msg: Optional[str] = None,
                 min_val: Union[float, int, None] = None,
                 max_val: Union[float, int, None] = None,
                 type_: Union[tuple[Type, ...], None] = None
                 ):
        super().__init__(param_name, error_msg)
        self.min_val = min_val
        self.max_val = max_val
        self.type_ = type_
        
    def validate(self, value: Any) -> Optional[str]:
        if self.type_ and not isinstance(value, self.type_):
            return (self.error_msg or 
                    f"The type of `{self.param_name}` must belong to {[i.__name__ for i in self.type_]}, "
                    f"but got {type(value).__name__!r}")
        if not (self.min_val <= value <= self.max_val):
            return (self.error_msg or f"`{self.param_name}` must between {self.min_val} and {self.max_val}, "
                    f"but got {value}.")
        return None

class ValueValidator(SupportedValidator):
    def __init__(
            self,
            param_name: str,
            error_msg: Optional[str] = None,
            subfield: list = [],
            target_value: list = []
    ):
        super().__init__(param_name, error_msg, subfield)
        self.target_value = target_value
        self.error_msg = self.error_msg or f"`{self.param_name}` only support the value in {self.target_value}"

    def validate(self, value):
        if error := super().validate(value):
            return error
        if value not in self.target_value:
            return self.error_msg
        return None

def create_validator(param_name: str, config: dict[str, Any]) -> Optional[BaseValidator]:
    validator_type = config.get("validator_type")
    
    if validator_type == "supported":
        return SupportedValidator(
            param_name=config.get("param_name", param_name),
            error_msg=config.get("error_msg"),
            subfield=config.get("subfield", []),
            skip_subfield_check=config.get("skip_subfield_check", [])
        )
    
    elif validator_type == "value":
        return ValueValidator(
            param_name=config.get("param_name", param_name),
            error_msg=config.get("error_msg"),
            subfield=config.get("subfield", []),
            target_value=config.get("value", [])
        )
    
    elif validator_type == "range":
        type_str = config.get("type_", [])
        if type_str and any(type_ not in TYPE_MAPPING for type_ in type_str):
            raise ValueError(f"Only supported type: {TYPE_MAPPING.keys()}")

        return RangeValidator(
            param_name=config.get("param_name", param_name),
            min_val=config.get("min_val"),
            max_val=config.get("max_val"),
            type_=list(map(TYPE_MAPPING.get, type_str))
        )
    
    else:
        raise ValueError(f"Unknown validator type: {validator_type}")

def load_validators_from_json(config_path: str) -> tuple[dict[str, BaseValidator], dict[str, BaseValidator]]:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    validators = {}
    custom_validators = {}
    
    # load validators
    for param_name, validator_config in config.get("validators", {}).items():
        validator = create_validator(param_name, validator_config)
        if validator:
            validators[param_name] = validator
    
    # load custom_validators
    for param_name, validator_config in config.get("custom_validators", {}).items():
        validator = create_validator(param_name, validator_config)
        if validator:
            custom_validators[param_name] = validator
    
    return validators, custom_validators

VALIDATORS, CUSTOM_VALIDATORS = load_validators_from_json(os.getenv("VALIDATORS_CONFIG_PATH", ""))

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
                    if validator.param_name not in json_load.keys():
                        return self.create_error_response(status_code, validator.error_msg)
                    elif error := validator.validate(json_load[validator.param_name]):
                        return self.create_error_response(status_code, error)
                
        return await call_next(request)