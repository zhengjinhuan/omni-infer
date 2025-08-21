package metrics_collector

import (
	"sync"
)

var (
	staticMap map[string][string]
	initOnce  sync.Once
)

// 需要从RouterServer中获取的指标名称和vLLM原生的指标名称对照
func initStaticMap() {
	staticMap = map[string]string{
		"rs:time_to_first_token_sec":  "vllm:time_to_first_token_seconds"
		"rs:time_per_output_token_ms": "vllm:time_per_output_token_seconds"
		"rs:request_total_time_secs":  "vllm:e2e_request_latency_seconds"
	}
}

func GetStaticMap() map[string]string {
	initOnce.Do(initStaticMap)
	return staticMap
}

var vLLMLabels = []string{"model_name", "engine"}
var CollectLabels = []string{"role", "instance", "engine", "model_name"}

func getMetricsLabels(isAgg bool) []string {
	if isAgg { 
		return vLLMLabels
	}
	return CollectLabels
}
