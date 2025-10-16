# 🚀 Awesome Benchmarking

A curated collection of **benchmarking resources, performance studies, and scalability tests** across cloud platforms, databases, orchestration tools, AI models, and distributed systems.

> 🧩 Maintained by [Apoorw Anand](https://github.com/apoorvanand)  
> 💡 Focused on real-world metrics — latency, throughput, resource utilization, and resiliency.

---

## ☁️ Cloud Platforms

| Platform | Resource | Description |
|-----------|-----------|-------------|
| **AWS EC2 / EKS** | [AWS Compute Optimizer Benchmarks](https://aws.amazon.com/blogs/compute/benchmarking-ec2-instances/) | Instance performance and scaling comparison |
| **Azure VMSS** | [Azure VM Scale Sets Performance Guide](https://learn.microsoft.com/en-us/azure/virtual-machine-scale-sets/overview) | Scaling, elasticity, and autoscaling performance |
| **GCP GKE / Dataflow** | [GCP Performance Testing](https://cloud.google.com/architecture/performance-testing-on-gcp) | Benchmarking pipelines and compute efficiency |

---

## 🗄️ Databases & Key-Value Stores

| Database | Benchmark | Description |
|-----------|------------|-------------|
| **PostgreSQL** | [pgbench](https://www.postgresql.org/docs/current/pgbench.html) | Native transaction benchmarking |
| **MySQL** | [SysBench MySQL Suite](https://dev.mysql.com/doc/refman/8.0/en/benchmark-suite.html) | I/O and query throughput testing |
| **Redis** | [redis-benchmark CLI](https://redis.io/docs/interact/benchmarks/) | Latency and throughput analysis |
| **Cassandra** | [NoSQLBench](https://github.com/nosqlbench/nosqlbench) | Modern NoSQL benchmarking framework |
| **etcd** | [etcd Benchmark Tool](https://etcd.io/docs/v3.5/dev-guide/benchmark/) | Leader election, write latency, and consensus overhead benchmarks |

---

## ⚙️ Orchestration & Cloud-Native Systems

| System | Resource | Description |
|---------|-----------|-------------|
| **Kubernetes** | [k8s Scalability Benchmarks](https://github.com/kubernetes/perf-tests) | Official Kubernetes scalability test suite |
| **Airflow** | [Airflow Benchmarking](https://airflow.apache.org/docs/apache-airflow/stable/benchmarks.html) | Scheduler and DAG latency metrics |
| **Service Mesh** | [Istio Performance Benchmarks](https://istio.io/latest/docs/ops/deployment/performance-and-scalability/) | Proxy overhead and mesh scaling studies |

---

## 🧩 Data Processing Frameworks

| Framework | Resource | Description |
|------------|-----------|-------------|
| **Apache Spark** | [Spark SQL Perf (TPC-DS)](https://github.com/databricks/spark-sql-perf) | Standard Spark query benchmark |
| **Apache Flink** | [Flink Benchmarks](https://github.com/ververica/flink-benchmarks) | Streaming and batch performance |
| **Hadoop** | [HiBench](https://github.com/Intel-bigdata/HiBench) | Big Data benchmark suite from Intel |

---

## 🔍 Observability Tools

| Tool | Resource | Description |
|------|-----------|-------------|
| **Grafana** | [Grafana Load Testing Guide](https://grafana.com/docs/grafana/latest/setup-grafana/load-testing/) | UI rendering and concurrent dashboard testing |
| **Prometheus** | [PromBench](https://github.com/prometheus/prombench) | Performance test setup for Prometheus scalability |

---

## 🧪 Load Testing Tools

| Tool | Resource | Description |
|------|-----------|-------------|
| **JMeter** | [JMeter Best Practices](https://jmeter.apache.org/usermanual/best-practices.html) | General-purpose load and stress testing |
| **Gatling** | [Gatling OSS Benchmarks](https://gatling.io/open-source/) | Simulation-based HTTP performance testing |
| **Locust** | [Locust Distributed Benchmarks](https://docs.locust.io/en/stable/running-distributed.html) | Scalable load generation and orchestration |

---

## 🧠 AI / ML Benchmarks

### 🧩 LLM Model Performance

| Model / Framework | Benchmark | Description |
|-------------------|------------|-------------|
| **OpenAI GPT Series** | [MLPerf Inference (NLP)](https://mlcommons.org/en/inference-nlp/) | Standardized NLP model benchmark |
| **LLaMA / Mistral / Falcon** | [HuggingFace LLM Benchmarks](https://huggingface.co/blog/benchmarking-transformers) | Latency and throughput comparison across models |
| **Gemma / Qwen / Mixtral** | [AI Benchmark Leaderboard](https://browse.ai-benchmarks.com/) | Inference cost and performance trade-offs |

### ⚙️ Inference & Serving Frameworks

| Framework | Benchmark | Description |
|------------|------------|-------------|
| **TensorRT / Triton** | [NVIDIA Triton Benchmarks](https://developer.nvidia.com/nvidia-triton-inference-server) | GPU inference performance for production |
| **ONNX Runtime** | [ONNX Performance Tuning](https://onnxruntime.ai/docs/performance/) | Cross-platform inference optimization |
| **vLLM / FastChat** | [vLLM Performance Docs](https://vllm.ai/) | High-throughput, memory-efficient serving benchmark |

### 💻 GPU / Hardware Benchmarks

| Platform | Resource | Description |
|-----------|-----------|-------------|
| **NVIDIA CUDA / Tensor Cores** | [MLPerf Training Benchmarks](https://mlcommons.org/en/training-overview/) | GPU-based distributed training benchmarks |
| **AMD ROCm** | [ROCm Performance Tests](https://rocmdocs.amd.com/en/latest/performance.html) | Benchmarking deep learning performance on AMD GPUs |
| **Intel Gaudi / Habana** | [Habana AI Benchmark Suite](https://habana.ai/training-benchmarks/) | AI workload performance on Gaudi accelerators |

---

## 🧵 Messaging & Streaming Systems

| System | Benchmark | Description |
|---------|------------|-------------|
| **Kafka** | [Kafka Official Performance Guide](https://kafka.apache.org/documentation/#design_performance) | Producer, consumer, and replication benchmarks |
| **RabbitMQ** | [RabbitMQ Perf Test](https://github.com/rabbitmq/rabbitmq-perf-test) | Native throughput and latency testing |
| **NATS** | [NATS Benchmarks](https://github.com/nats-io/nats-bench) | Pub/sub message latency and fan-out benchmarks |

---

## 🧰 Contributing
Pull requests welcome! Please ensure:
- Links are **official**, **recent**, and **quantitative**
- Add section summaries and short descriptions for clarity

---

## 🪴 License
[MIT](LICENSE)
