import inspect
import os
import logging
from argparse import RawTextHelpFormatter
from dataclasses import asdict, dataclass

import torch
import torch_npu

from vllm import LLM, SamplingParams
from vllm.engine.arg_utils import EngineArgs
from vllm.utils import FlexibleArgumentParser

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

BATCH_SIZE_DEFAULT = 1
PROMPT_LEN_DEFAULT = 256
OUTPUT_LEN_DEFAULT = 5
# Profile of the third token in the decode collection
PROFILER_DECODE_STEP = 3
PROFILE_DIR_DEFAULT = ".torch_profile"


class layerwise_profile(torch_npu.profiler.profile):

    def __init__(
            self,
            torch_profiler_dir: str,
            record_shapes: bool = False,
            profile_memory: bool = False,
            with_stack: bool = False,
            with_modules: bool = False,
            profiler_level: str = torch_npu.profiler.ProfilerLevel.Level1,
    ):
        self.torch_profiler_trace_dir = torch_profiler_dir
        print(
            "Profiling enabled. Traces will be saved to: %s",
            self.torch_profiler_trace_dir,
        )
        experimental_config = torch_npu.profiler._ExperimentalConfig(
            profiler_level=profiler_level,
            aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
        )
        super().__init__(
            activities=[
                torch_npu.profiler.ProfilerActivity.CPU,
                torch_npu.profiler.ProfilerActivity.NPU,
            ],
            record_shapes=record_shapes,
            profile_memory=profile_memory,
            with_stack=with_stack,
            with_modules=with_modules,
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
                self.torch_profiler_trace_dir
            ),
            experimental_config=experimental_config,
        )

    def __enter__(self):
        return super().__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        super().__exit__(exc_type, exc_val, exc_tb)


@dataclass
class ProfileContext:
    engine_args: EngineArgs
    prompt_len: list
    output_len: list
    batch_size: list
    save_path: str
    temperature: float
    top_p: float
    top_k: float
    repetition_penalty: float


def get_dtype(dtype: str):
    if dtype == "torch.float":
        return torch.float
    else:
        return dtype


def run_profile(
        context: ProfileContext,
        is_decode: bool,
        is_prefill: bool,
        with_stack: bool,
        with_modules: bool,
        record_shapes: bool,
        profile_memory: bool,
        profiler_level: str,
):
    logger.info("Run profile with:")
    for key, value in asdict(context).items():
        logger.info(f"{key} = {value}")

    batch_size_list = context.batch_size
    prompt_len_list = context.prompt_len
    output_len_list = context.output_len
    save_path = context.save_path
    if len({len(batch_size_list), len(prompt_len_list), len(output_len_list)}) != 1:
        logger.error(f"The array length of batch_size, prompt_len, and output_len are different!")
        return

    # Create LLM
    llm = LLM(**asdict(context.engine_args))

    scheduler_config = llm.llm_engine.scheduler_config
    max_model_len = llm.llm_engine.model_config.max_model_len
    max_num_batched_tokens = scheduler_config.max_num_batched_tokens
    max_num_seqs = scheduler_config.max_num_seqs
    if profiler_level == "level0":
        profiler_level = torch_npu.profiler.ProfilerLevel.Level0
    elif profiler_level == "level2":
        profiler_level = torch_npu.profiler.ProfilerLevel.Level2
    else:
        profiler_level = torch_npu.profiler.ProfilerLevel.Level1

    def add_requests():
        for i in range(batch_size):
            prompt_token_ids = torch.randint(
                llm.llm_engine.model_config.get_vocab_size(), size=(prompt_len,)
            ).tolist()

            llm.llm_engine.add_request(
                request_id=f"seq{i}",
                prompt={"prompt_token_ids": prompt_token_ids},
                params=sampling_params,
            )

    def profile_decode_steps(
            llm_engine,
            save_path,
            prompt_len,
            batch_size,
            output_len,
            record_shapes,
            with_stack,
            with_modules,
            profile_memory,
            profiler_level,
            args
    ):
        for decode_step_num in range(args.output_len - 1):
            if decode_step_num >= PROFILER_DECODE_STEP:
                decode_prof_path = os.path.join(
                    save_path, f"decode_input{prompt_len}_bs{batch_size}"
                )
                with layerwise_profile(
                        decode_prof_path,
                        record_shapes=record_shapes,
                        with_stack=with_stack,
                        with_modules=with_modules,
                        profile_memory=profile_memory,
                        profiler_level=profiler_level,
                ) as prof:
                    llm_engine.step()
                    prof.step()
                    torch.npu.synchronize()
                break
            else:
                llm_engine.step()

    def abort_requests():
        for i in range(batch_size):
            llm.llm_engine.abort_request(f"seq{i}")

    logger.info(
        f"llm.llm_engine.model_config.max_model_len: {llm.llm_engine.model_config.max_model_len}"
    )

    for batch_size in batch_size_list:
        for i, prompt_len in enumerate(prompt_len_list):
            output_len = output_len_list[i]

            # Create sampling params
            sampling_params = SamplingParams(
                temperature=context.temperature,
                top_p=context.top_p,
                top_k=context.top_k,
                repetition_penalty=context.repetition_penalty,
                max_tokens=output_len,
                ignore_eos=True,
            )

            logger.info(f"profile running with batch_size: {batch_size}, "
                        f"prompt_len: {prompt_len}, output_len: {output_len}")
            if output_len < 5:
                logger.error("output-len must be greater than or equal to 5")
                continue
            if batch_size >= max_num_seqs:
                logger.error(
                    f"ERROR: chosen batch_size ({batch_size}) is larger than "
                    f"max_num_seqs ({max_num_seqs}) and therefore cannot be run in a "
                    f"single profile step, please choose a smaller batch size"
                )
                continue
            if prompt_len + output_len > llm.llm_engine.model_config.max_model_len:
                logger.error(
                    f"ERROR: chosen prompt_len + output_len ({prompt_len} + "
                    f"{output_len} = {prompt_len + output_len}) is larger than the "
                    f"model's max_model_len ({max_model_len}), please choose a smaller "
                    f"prompt_len or output_len, or increase --max-model-len"
                )
                continue

            # Warm up run
            logger.info("Warm up run ...")
            add_requests()
            llm.llm_engine.step()  # Prefill
            llm.llm_engine.step()  # Decode
            abort_requests()

            logger.info("Profile run ...")
            add_requests()

            if is_prefill:
                prefill_prof_path = os.path.join(
                    save_path, f"prefill_input{prompt_len}_bs{batch_size}"
                )
                with layerwise_profile(
                        prefill_prof_path,
                        record_shapes=record_shapes,
                        with_stack=with_stack,
                        with_modules=with_modules,
                        profile_memory=profile_memory,
                        profiler_level=profiler_level,
                ) as prof:
                    llm.llm_engine.step()  # First step is prefill
                    prof.step()
                    torch.npu.synchronize()
            else:
                llm.llm_engine.step()  # First step is prefill

            if batch_size * prompt_len > max_num_batched_tokens:
                for _ in range(batch_size * prompt_len // max_num_batched_tokens):
                    llm.llm_engine.step()
            if is_decode:
                profile_decode_steps(
                    llm.llm_engine,
                    save_path,
                    prompt_len,
                    batch_size,
                    output_len,
                    record_shapes,
                    with_stack,
                    with_modules,
                    profile_memory,
                    profiler_level,
                    args
                )

            abort_requests()


if __name__ == "__main__":
    parser = FlexibleArgumentParser(
        description="Profile a model", formatter_class=RawTextHelpFormatter
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0,
        help="Control the randomness of sampling",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.8,
        help="Controls the cumulative probability of the top token considered",
    )
    parser.add_argument(
        "--top-k",
        type=float,
        default=20,
        help="Controls the number of top tokens to be considered",
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1,
        help="Penalty value for the new token, based on its presence in the prompt and generated text",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default=PROFILE_DIR_DEFAULT,
        help="path to save profile results" "- default={PROFILE_DIR_DEFAULT}",
    )
    parser.add_argument(
        "--prompt-len",
        nargs='+',
        type=int,
        default=[PROMPT_LEN_DEFAULT],
        help=f"Length of the random prompt to use when profiling, all batched "
             f"requests use the same prompt_len, default={PROMPT_LEN_DEFAULT}",
    )
    parser.add_argument(
        "--batch-size",
        nargs='+',
        type=int,
        default=[BATCH_SIZE_DEFAULT],
        help=f"Number of requests to run as a single batch, "
             f"default={BATCH_SIZE_DEFAULT}",
    )
    parser.add_argument(
        "--output-len",
        nargs='+',
        type=int,
        default=[OUTPUT_LEN_DEFAULT],
        help="Number of llm steps to run (includes prefill and decode), it should >=5, "
             f"default={OUTPUT_LEN_DEFAULT}",
    )
    parser.add_argument(
        "--decode",
        action="store_true",
        help="Indicates whether to collect decode profiling data",
    )
    parser.add_argument(
        "--prefill",
        action="store_true",
        help="Indicates whether to collect prefill profiling data.",
    )
    parser.add_argument(
        "--with-stack",
        action="store_true",
        help="Indicates whether to record the operator call stack.",
    )
    parser.add_argument(
        "--with-modules",
        action="store_true",
        help="record module hierarchy (including function names)",
    )
    parser.add_argument(
        "--record-shapes",
        action="store_true",
        help="Indicates whether to record InputShapes and InputTypes of an operator.",
    )
    parser.add_argument(
        "--profile-memory",
        action="store_true",
        help="Indicates whether to record the memory usage of the operator.",
    )
    parser.add_argument(
        "--profiler-level",
        type=str,
        default="level1",
        choices=["level0", "level1", "level2"],
        help="Collected Level \n"
             "level0: Collects upper-layer application data, lower-layer NPU data (Ascend Headware data), and information about operators executed on the NPU.\n"
             "level1: On the basis of Level0, collect Ascend CL data at the CANN layer and AI Core performance indicators executed on the NPU, and enable aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization.\n"
             "level2: Collect GE and Runtime data, HCCL and AI CPU data at the CANN layer, and enable aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization on the basis of Level1.",
    )

    EngineArgs.add_cli_args(parser)

    args = parser.parse_args()

    context = ProfileContext(
        engine_args=EngineArgs.from_cli_args(args),
        **{
            k: v
            for k, v in vars(args).items()
            if k in inspect.signature(ProfileContext).parameters
        },
    )
    run_profile(
        context,
        args.decode,
        args.prefill,
        args.with_stack,
        args.with_modules,
        args.record_shapes,
        args.profile_memory,
        args.profiler_level,
    )
