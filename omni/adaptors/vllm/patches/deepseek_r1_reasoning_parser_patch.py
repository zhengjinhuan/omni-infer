from typing import Optional

from vllm.entrypoints.openai.protocal import (ChatCompletionRequest,
                                              DeltaMessage)


def patch_extract_reasoning_content_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
) -> Union[DeltaMessage, None]:
    """
    Extract reasoning content from a delta message.
    Handles streaming output where previous + delta = current.
    Uses token IDs for faster processing.
    For text <think>abc</think>xyz:
    - 'abc' goes to reasoning_content
    - 'xyz' goes to content
    """
    # Skip single special tokens
    if len(delta_token_ids) == 1 and (delta_token_ids[0] in [
        self.start_token_id, self.end_token_id
    ]):
        return None

    # Check if <think> is present in previous or delta.
    # Keep compatibility with models that don't generate <think> tokens.
    if self.start_token_id in previous_token_ids:
        if self.end_token_id in delta_token_ids:
            # <think> in previous, </think> in delta,
            # extract reasoning content
            end_index = delta_text.find(self.end_token)
            reasoning_content = delta_text[:end_index]
            content = delta_text[end_index + len(self.end_token):]
            return DeltaMessage(
                reasoning_content=reasoning_content,
                content=content if content else None,
            )
        elif self.end_token_id in previous_token_ids:
            # <think> in previous, </think> in previous,
            # reasoning content continues
            return DeltaMessage(content=delta_text)
        else:
            # <think> in previous, no </think> in previous or delta,
            # reasoning content continues
            return DeltaMessage(reasoning_content=delta_text)
    elif self.start_token_id in delta_token_ids:
        if self.end_token_id in delta_token_ids:
            # <think> in delta, </think> in delta, extract reasoning content
            start_index = delta_text.find(self.start_token)
            end_index = delta_text.find(self.end_token)
            reasoning_content = delta_text[start_index +
                                           len(self.start_token):end_index]
            content = delta_text[end_index + len(self.end_token):]
            return DeltaMessage(
                reasoning_content=reasoning_content,
                content=content if content else None,
            )
        else:
            # <think> in delta, no </think> in delta,
            # reasoning content continues
            # <think> in delta, </think> in delta, extract reasoning content
            # Adapt: 适配delta tokens不出一个tokens的情况，Reasoning content需要去除start_token
            start_index = delta_text.find(self.start_token)
            reasoning_content = delta_text[start_index +
                                           len(self.start_token)]
            return DeltaMessage(reasoning_content=reasoning_content)
    else:
        # No <think> in previous or delta, also need to check for </think>.
        # Because the model may have generated </think> without <think>
        # Ref https://huggingface.co/deepseek-ai/DeepSeek-R1/commit/8a58a132790c9935686eb97f042afa8013451c9f
        if self.end_token_id in delta_token_ids:
            # </think> in delta with more tokens,
            # extract reasoning content and content
            end_index = delta_text.find(self.end_token)
            reasoning_content = delta_text[:end_index]
            content = delta_text[end_index + len(self.end_token):]
            return DeltaMessage(
                reasoning_content=reasoning_content,
                content=content if content else None,
            )
        elif self.end_token_id in previous_token_ids:
            # </think> in previous, thinking content ends
            return DeltaMessage(content=delta_text)
        else:
            # no </think> in previous or delta, reasoning content continues
            return DeltaMessage(reasoning_content=delta_text)


def patch_extract_reasoning_content(
        self, model_output: str, request: ChatCompletionRequest
) -> tuple[Optional[str], Optional[str]]:
    """
    Extract reasoning content from the model output.

    For text <think>abc</think>xyz:
    - 'abc' goes to reasoning_content
    - 'xyz' goes to content

    Returns:
        tuple[Optional[str], Optional[str]]: reasoning content and content
    """

    # Adapt: 判断是否包含start token or end token, 兼容Qwen3系列可关闭reasoning特性
    if (self.start_token not in model_output
            and self.end_token not in model_output):
        return None, model_output

    # Check if the start token is present in the model output, remove it
    # if it is present.
    model_output_parts = model_output.partition(self.start_token)
    model_output = model_output_parts[2] if model_output_parts[
        1] else model_output_parts[0]

    # DeepSeek R1 doesn't generate <think> now.
    # Thus we assume the reasoning content is always at the start.
    # Ref https://huggingface.co/deepseek-ai/DeepSeek-R1/commit/8a58a132790c9935686eb97f042afa8013451c9f
    if self.end_token not in model_output:
        return model_output, None
    else:
        reasoning_content, _, content = model_output.partition(
            self.end_token)
        # If the end token is not found, return the model output as is.
        # It should not happen since we already checked for the presence
        # of the end token.
        # If generation stops right after end-of-think, return null content
        final_content = content or None
        return reasoning_content, final_content


def patch_deepseek_r1_reasoning_parser():
    from vllm.reasoning.deepseek_r1_reasoning_parser import DeepSeekR1ReasoningParser

    DeepSeekR1ReasoningParser.extract_reasoning_content_streaming = patch_extract_reasoning_content_streaming
    DeepSeekR1ReasoningParser.extract_reasoning_content = patch_extract_reasoning_content