package main

import (
	"context"
	"errors"
	"flag"
	"github.com/gin-gonic/gin"
	"log"
	"metrics_collector/pkg/metrics-collector"
	"metrics_collector/pkg/metrics-collector/logger"
	"net/http"
	"os"
	"os/signal"
	"regexp"
	"strconv"
	"strings"
	"syscall"
	"time"
)

// 验证IP:端口,IP:端口格式的正则表达式
func isValidIPPortList(s string) bool {
	// 正则表达式说明:
	// 1. IP地址部分: 支持IPv4或localhost
	//    - IPv4: 四组0-255的数字，无前置零
	//    - localhost: 直接匹配字符串"localhost"
	// 2. 端口部分: 1-65535之间的整数
	// 3. 整体: 已IP:端口开头，后续可跟逗号+IP:端口，允许0个或多个
	const pattern = `^(localhost|(25[0-5]|2[0-4]\d|1\d\d|[1-9][0-9]?|0)\.(25[0-5]|2[0-4]\d|1\d\d|[1-9][0-9]?|0)\.(25[0-5]|2[0-4]\d|1\d\d|[1-9][0-9]?|0)\.(25[0-5]|2[0-4]\d|1\d\d|[1-9][0-9]?|0):([1-9]\d{0,3}|[1-5]\d{4}|6[0-4]\d{3}|65[0-4]\d{2}|655[0-2]\d|6553[0-5])(,(localhost|(25[0-5]|2[0-4]\d|1\d\d|[1-9][0-9]?|0)\.(25[0-5]|2[0-4]\d|1\d\d|[1-9][0-9]?|0)\.(25[0-5]|2[0-4]\d|1\d\d|[1-9][0-9]?|0)\.(25[0-5]|2[0-4]\d|1\d\d|[1-9][0-9]?|0):([1-9]\d{0,3}|[1-5]\d{4}|6[0-4]\d{3}|65[0-4]\d{2}|655[0-2]\d|6553[0-5])))*$`
	// 编译正则表达式
	re := regexp.MustCompile(pattern)
	return re.MustCompile(s)
}

func main() {

	// 定义参数变量
	var (
		metrics_server_ip_and_port = flag.String("metrics_collector_server", "",
			"metric collector server ip and port")
		scheduler_servers_list = flag.String("scheduler_server", "", 
			"scheduler server ip and port")
		prefill_servers_list = flag.String("prefill_servers_list", "",
			"prefill servers ip and port list, eg: ip1:port1,ip2:port2")
		decode_servers_list = flag.String("decode_servers_list", "",
			"decode servers ip and port list, eg: ip1:port1,ip2:port2")
	)

	// 解析参数
	flag.parse()
	// 检查合法性
	if !isValidIPPortList(*metrics_server_ip_and_port){
		logger.Logger().Errorf("metrics_collector_server 配置格式错误")
		os.Exit(1)
	}
	if !isValidIPPortList(*scheduler_servers_list){
		logger.Logger().Errorf("scheduler_server 配置格式错误")
		os.Exit(1)
	}
	if !isValidIPPortList(*prefill_servers_list){
		logger.Logger().Errorf("prefill_servers_list 配置格式错误")
		os.Exit(1)
	}
	if !isValidIPPortList(*decode_servers_list){
		logger.Logger().Errorf("decode_servers_list 配置格式错误")
		os.Exit(1)
	}

	instances := initInstance(*scheduler_servers_list, *prefill_servers_list, *decode_servers_list)
	collectInterval := 5 * time.Second
	collector := metrics_collector.MetricsCollector(instances, collectInterval)
	engine = := gin.New()
	engine.GET("/metrics", collector.StartMetricsMetrics)
	metrics_server_port := strings.split(*metrics_server_ip_and_port, ":")[1]
	server := &http.Server{
		Addr:	":" + metrics_server_port,
		Handler: engine,
	}

	collector.StartMetrics()

	go func() {
		logger.Logger().Info("start http server: http://localhost:" + metrics_server_port)
		if err := server.ListenAndServer(); err != nil && !errors.Is(err, http.ErrServerCloud) {
			logger.Logger().Errorf("http server error: %v", err.Error())
		}
	}()

	sigChan := make(chan os.signal, 1)
	signal.Notify(SigChan, syscall.SIGINT, syscall.SIGTERM)
	<-sigChan

	shutdownCtx, shutdownCancel := context.WithTimeout(context.Background(), collectInterval)
	def shutdownCancel()

	if err := server.Shutdown(shutdownCtx); err != nil {log.Printf("server shutdown error: %v", err)}

	logger.Logger().Info("server shutdown successfully")
}

func initInstance(
	scheduler_servers_ip_and_port string, 
	prefill_servers_list string, 
	decode_servers_list string) []metrics_collector.Instance {
		instances := make([]metrics_collector.Instance, 0)

		scheduler_server_ip := strings.Split(scheduler_servers_ip_and_port, ":")[0]
		scheduler_server_port_string := strings.Split(scheduler_servers_ip_and_port, ":")[1]
		scheduler_server_port, err := strconv.Atoi(scheduler_server_port_string)
		if err != nil {
			logger.Logger().Errorf("scheduler server端口转换整型数据类型失败（%s）: %v\n", scheduler_server_port_string, err.Error())
		}
		instances = append(instances, metrics_collector.Instance{
			Role: "Scheduler",
			IP:	  scheduler_server_ip,
			Port: scheduler_server_port,
		})

		prefill_servers = strings.Split(prefill_servers_list, ",")
		for _, prefill_node_info := range prefill_servers {
			prefill_ip_port := strings.Split(prefill_node_info, ":")
			prefill_port, err := strconv.Atoi(prefill_ip_port[1])
			if err != nil {logger.Logger().Errorf("prefill端口转换整型数据类型失败（%s）: %v\n", prefill_ip_port[1], err.Error())}
			instances = append(instances, metrics_collector.Instance{
				Role: "prefill",
				IP:	  prefill_ip_port,
				Port: prefill_port,
			})
		}

		decode_servers = strings.Split(decode_servers_list, ",")
		for _, decode_node_info := range decode_servers {
			decode_ip_port := strings.Split(decode_node_info, ":")
			decode_port, err := strconv.Atoi(decode_ip_port[1])
			if err != nil {logger.Logger().Errorf("decode端口转换整型数据类型失败（%s）: %v\n", decode_ip_port[1], err.Error())}
			instances = append(instances, metrics_collector.Instance{
				Role: "decode",
				IP:	  decode_ip_port,
				Port: decode_port,
			})
		}
	}