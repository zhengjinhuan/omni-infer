package metrics_collector

import (
	"fmt"
	"github.com/prometheus/client_golang/prometheus"
	dto "github.com/prometheus/client_model/go"
	"reflect"
	"strings"
)

// 对目标metrics数据执行histogram combine操作
func combineHistogramMetrics(c *Collector, MetricsName string, role string) error {
	SampleCount := uint64(0)
	SampleSum := 0.0
	buckets := make[map[float64]uint64]
	var labels = prometheus.Labels{}

	for _, instance := range c.instances {
		// 实例身份需要为目标身份才会被统计， 注： api_server包括所有的P、D实例
		if strings.ToLower(instance.Role) == role || (role == "api_server" && strings.ToLower(instance.Role) != "scheduler") {
			metrics := c.histogramMetrics[MetricsName]
			allLabels := instance.getInstanceLabelsForMetrics(MetricsName)
			for _, label := range allLabels {
				if len(labels) == 0 {
					labels = filterCustomLabels(label)
					labels["engine"] = "All"
				}
				histogramData := metrics.GetHistogramData(label)
				SampleCount += histogramData.SampleCount
				SampleSum += histogramData.SampleSum
				for upperBound, count := range histogramData.Buckets {
					buckets[upperBound] += count
				}
			}
		}
	}
	if len(labels) > 0 {
		// 获取到目标数据写入普罗
		c.aggregateHistogramMetrics[MetricsName].SetHistogramData(labels, &HistogramData{
			SampleCount: sampleCount,
			SampleSum:   sampleSum,
			Buckets:     buckets,
		})
	}
	return nil
}

// 对目标metrics执行sum操作
func sumMetrics(c *Collector, MetricsName string, metricsType string, role string) error {
	if metricsType == "Gauge" {
		err := sumGaugeMetrics(c, MetricsName, role)
		if err != nil {
			return err
		}
	} else if metricsType == "Counter" {
		err := sunCounterMetrics(c, MetricsName, role)
		if err != nil {
			return err
		}
	} else {
		return fmt.Errorf("metrics type %s is not supported", metricsType)
	}
	return nil
}
