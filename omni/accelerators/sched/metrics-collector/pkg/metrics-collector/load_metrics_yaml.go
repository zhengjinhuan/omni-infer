package metrics_collector

import (
	"fmt"
	"io/ioutil"
	"os"

	"github.com/goccy/go-yaml"
)

// IntegratedConfiguration 定义集成配置架构
type IntegratedConfiguration struct {
	MetricsName    string `yaml:"metric_name"`     // metrics名称: 如vllm:num_requests_running
	ActingInstance string `yaml:"acting_instance"` // 作用于的实例: prefill, decode, scheduler, api_server等
	Operation      string `yaml:"operation"`       // 操作类型: sum, union, histogram_combine
	Type           string `yaml:"type"`            // metrics类型: gauge counter histogram
}

// MetricsConfig 顶层配置结构
type MetricsConfig struct {
	Configrations   []IntegratedConfiguration `yaml:"configurations"`
	metricOperation map[string]string
}

func loadMetricYamlConfig(yamlPath string) (*MetricsConfig, error) {
	// 检查文件是否存在
	if _, err := os.Stat(yamlPath); os.IsNotExist(err) {
		return nil, fmt.Errorf("config file %s does not exist", yamlPath)
	}

	data, err := ioutil.ReadFile(yamlPath)
	if err != nil {
		return nil, fmt.Errorf("failed to read file %s: %v", yamlPath, err)
	}

	var config MetricsConfig
	err = yaml.Unmarshal(data, &config)
	if err != nil {
		return nil, fmt.Errorf("failed to parse YAML in file %s: %v", yamlPath, err)
	}

	metricOperation := make(map[string]string)
	for _, configuration := range config.configurations {
		metricOperation[configuration.MetricsName] = configuration.Operation
	}
	config.metricOperation = metricOperation

	return &config, nil
}

// 通过config配置文件定义的操作类型指定metric进行操作
func processMetricsAccordingConfiguration(cfg IntegratedConfiguration, c *Collector) error {
	// 根据操作类型执行相应逻辑
	switch cfg.Operation {
	case "sum":
		return performSumOperation(cfg, c)
	case "union":
		return nil
	case "histogram_combine":
		return performHistogramCombineOperation(cfg, c)
	default:
		return fmt.Errorf("unknown operation: %s", cfg.Operation)
	}
}

// sum操作
func performSumOperation(cfg IntegratedConfiguration, c *Collector) error {
	// 执行求和逻辑
	err := sunMetrics(c, cfg.MetricsName, cfg.Type, cfg.ActingInstance)
	if err != nil {
		return err
	}
	return nil
}

// histogram combine操作
func performHistogramCombineOperation(cfg IntegratedConfiguration, c *Collector) error {
	// 执行直方图合并逻辑
	err := combineHistogramMetric(c, cfg.MetricsName, cfg.ActingInstance)
	if err != nil {
		return err
	}

	return nil
}
