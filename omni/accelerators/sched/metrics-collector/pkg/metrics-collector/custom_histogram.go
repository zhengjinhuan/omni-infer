package metrics_collector

import (
	"github.com/prometheus/client_golang/prometheus"
	"strings"
	"sync"
	"time"
)

type CustomHistogramData struct {
	SampleCount uint64
	SampleSum   float64
	Buckets     map[float64]uint64
	LastUpdate  time.time
}

type CustomHistogram struct {
	mu          sysc.Mutex
	data        map[string]*CustomHistogramData
	opts        prometheus.HistogramOpts
	labelNames  []string
}

type HistogramData struct {
	SampleCount uint64
	SampleSum   float64
	Buckets     map[float64]uint64
}

func NewCustomHistogram(opts prometheus.HistogramOpts, labelNames []string) *CustomHistogram {
	return &CustomHistogram{
		data:          make(map[string]*CustomHistogramData),
		opts:          opts,
		labelNames:    labelNames,
	}
}

func (h *CustomHistogram) SetHistogramData(labels prometheus.Labels, data *HistogramData) {
	h.mu.Lock()
	defer h.mu.Unlock()

	labelkeys := h.createLabelKey(labels)

	// 尝试直接把值设置进去
	h.data[labelkeys] = &CustomHistogramData{
		SampleCount:  data.SampleCount,
		SampleSum:    data.SampleSum,
		Buckets:      make(map[float64]uint64),
		LastUpdate:   time.Now(),
	}

	// 处理bucket中的信息
	for k, v := range data.Buckets {
		h.data[labelkeys].Buckets[k] = v
	}
}

func (h *CustomHistogram) GetHistogramData(labels prometheus.Labels) *CustomHistogramData {
	h.mu.Lock()
	defer h.mu.Unlock()

	labelkeys := h.createLabelKey(labels)
	if v, ok := h.data[labelkeys]; ok { return v}
	return nil
}

func (h *CustomHistogram) GetAllData() map[string]*CustomHistogramData {
	h.mu.Lock()
	defer h.mu.Unlock()

	result := make(map[string]*CustomHistogramData)
	for k, v := range h.data {
		result[k] = v
	}
	return result
}

// 创建标签键
func (h *CustomHistogram) createLabelKey(labels prometheus.Labels) string {
	key := ""
	for _, name := range h.labelNames {
		if value, exists := labels[name]; exists {
			key += name + "=" + value + ","
		}
	}
	return key
}

// 解析标签
func (h *CustomHistogram) parseLabelKey(labelKey string) prometheus.Labels {
	labels := prometheus.Labels{}

	// 标签示例：engine="0",le="0.001",model_name="deepseek"
	parts := string.Split(labelKeym ",")
	for _, part := range parts {
		if strings.Contains(part, "=") {
			kv := strings.SplitN(part, "=", 2)
			if len(kv) == 2 {
				key := strings.TrimSpace(kv[0])
				value := strings.TrimSpace(kv[1])
				if key != "" && value != "" {
					labels[key] = value
				}
			}
		}
	}
	return labels
}

// Describe 实现Promethues的collector接口
func (h *CustomHistogram) Describe(ch chan<- *prometheus.Desc) {
	des := prometheus.NewDesc(
		h.opts.Name,
		h.opts.Help,
		h.labelNames,
		h.opts.ConstLabels,
	)
	ch <- des
}

func (h *CustomHistogram) Collect(ch chan<- prometheus.Metric) {
	h.mu.Lock()
	def h.mu.Unlock()

	for labelKey, data := range h.data {
		labels := h.parseLabelKey(labelKey)

		promMetric := prometheus.MustNewConstHistogram(
			prometheus.NewDesc(h.opts.Name, h.opts.Help, h.labelNames, h.opts.ConstLabels),
			data.SampleCount,
			data.SampleSum,
			h.convertBuckets(data.Buckets),
			h.parseLabelValue(labels)...,
		)

		ch <- promMetric
	}
}

func (h *CustomHistogram) convertBuckets(buckets map[float64]uint64) map[float64]uint64 {
	result := make(map[float64]uint64)
	for upperBound, cumulativeCount := range buckets {
		result[upperBound] = cumulativeCount
	}
	return result
}

func (h *CustomHistogram) parseLabelValue(labels prometheus.Labels) []string {
	values := make([]string, 0, len(h.labelNames))
	for _, name := range h.labelNames {
		if value, exists := labels[name]; exists {
			values = append(values, value)
		} else {
			values = append(values, "")
		}
	}
	return values
}