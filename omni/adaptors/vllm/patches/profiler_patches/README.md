## Function-Level Profiling Patches

This folder provides a **non-intrusive way to apply function-level profiling**.

### Supported Profilers

1. **Marker** - Simply add something before/after target functions.
2. **Timer** – Basic time measurement for target functions.
3. **VizTracer** – Execution trace visualization using VizTracer.
4. **Torch-NPU** – Profiling via `torch_npu.profiler`.

### Enable Profiling

Set the corresponding environment variable to a YAML config file:

* `PROFILING_NAMELIST`

### Usage
* export PROFILING_NAMELIST=/path/to/namelist.yml
* Example yaml configs are in the [`assets/`](./assets) folder.