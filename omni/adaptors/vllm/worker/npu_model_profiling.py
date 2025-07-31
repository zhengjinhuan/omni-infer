import os
import torch
import torch_npu

def run_model_with_profiling(model, input_ids, positions, intermediate_tensors, model_kwargs):
    prof_save_path = os.environ.get("PROFILING_SAVE_PATH", "./")
    experimental_config = torch_npu.profiler._ExperimentalConfig(
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
        aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization)
    with torch_npu.profiler.profile(
            activities=[
                torch_npu.profiler.ProfilerActivity.NPU,
                torch_npu.profiler.ProfilerActivity.CPU],
            with_stack=False,
            record_shapes=False,
            profile_memory=False,
            experimental_config=experimental_config,
            schedule=torch_npu.profiler.schedule(wait=0, warmup=0, active=4, repeat=1, skip_first=1),
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
                prof_save_path + "_generate")) as prof:
        for _ in range(6):
            torch.npu.synchronize()
            hidden_states = model(
                input_ids=input_ids,
                positions=positions,
                intermediate_tensors=intermediate_tensors,
                inputs_embeds=None,
                **model_kwargs,
            )
            torch.npu.synchronize()
            prof.step()

    return hidden_states