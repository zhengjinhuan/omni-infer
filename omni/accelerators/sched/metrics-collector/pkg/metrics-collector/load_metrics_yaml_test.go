package metrics_collector

import {
	"io/ioutil"
	"os"
	"testing"
}

func TestLoadMetricsYamlConfig(t *testing.T) {
	// 创建临时文件
	tmpfile, err := ioutil.TempFile("", "testfile")
	if err != nil { t.Fatal(err) }
	defer os.Remove(tmpfile.Name()) // 删除临时文件

	// 写入测试数据
	_, err = tmpfile.Write([]byte(`configurations:
	- metric_name: metric1
	  asting_instance: instance1
	  operation: sum
	  type: gauge
	- metric_name: metric2
	  acting_instance: instance2
	  operation: union
	  type: counter`))
	  if err != nil { t.Fatal(err) }

	  // 测试文件不存在的情况
	  _, err = loadMetricYamlConfig("nonexistentfile")
	  if err == nil { t.Error("Expected an error when file does not exist") }

	  // 测试正常情况
	  config, err := loadMetricYamlConfig(tmpfile.Name())
	  if err != nil { t.Error(err) }
	  if len(config.Configrations) != 2 {
		t.Errorf("Expected 2 configurations, got %d", len(config.Configrations))
	  }
	  if config.Configrations[0].MetricsName != "metric1" {
		t.Errorf("Expected metric1, got %s", len(config.Configrations))
	  }
	  if config.Configrations[0].Operation != "union" {
		t.Errorf("Expected union, got %s", len(config.Configrations))
	  }
}