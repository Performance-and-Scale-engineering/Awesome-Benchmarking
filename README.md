# 🚀 Awesome Benchmarking

A curated collection of **benchmarking resources, performance studies, and scalability tests** across cloud platforms, databases, orchestration tools, AI models, and distributed systems.

> 🧩 Maintained by [Apoorw Anand](https://github.com/apoorvanand) [Performance and Scale Engineering](https://github.com/Performance-and-Scale-engineering)  
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

| Tool | Benchmarking Resource | Description |
|------|------------------------|-------------|
| **JMeter** | [JMeter Best Practices](https://jmeter.apache.org/usermanual/best-practices.html) | General-purpose load and stress testing |
| **Gatling** | [Gatling OSS Benchmarks](https://gatling.io/open-source/) | Simulation-based HTTP performance testing |
| **Locust** | [Locust Distributed Benchmarks](https://docs.locust.io/en/stable/running-distributed.html) | Scalable load generation and orchestration |

---

## 🧠 AI / ML Benchmarks (Fixed Version)

### 🧩 Core MLCommons Benchmarks

| Benchmark | Resource | Description |
|------------|-----------|-------------|
| **MLPerf Training** | [MLCommons Training](https://mlcommons.org/en/training-overview/) | Deep learning training benchmark across vision, NLP, and recommendation workloads |
| **MLPerf Inference** | [MLCommons Inference](https://mlcommons.org/en/inference-overview/) | Inference latency and throughput benchmarking suite |
| **MLPerf HPC** | [MLCommons HPC](https://mlcommons.org/en/hpc-overview/) | Evaluates large-scale AI model performance on HPC systems |
| **MLPerf Tiny** | [MLCommons Tiny](https://mlcommons.org/en/tiny-overview/) | Lightweight benchmark for edge AI and microcontrollers |
| **MLPerf Storage** | [MLCommons Storage](https://mlcommons.org/en/storage-overview/) | Benchmarks I/O and data pipelines for ML workloads |
| **MLPerf Edge** | [MLCommons Edge](https://mlcommons.org/en/edge-overview/) | Measures on-device inference on edge and mobile hardware |

---

### 🧮 Model & Framework-Specific Benchmarks

| Framework / Model | Benchmark | Description |
|-------------------|------------|--------------|
| **Hugging Face Transformers** | [Benchmarking Transformers](https://huggingface.co/blog/benchmarking-transformers) | Comparison of BERT, GPT, and T5 model performance |
| **vLLM** | [vLLM Benchmarks](https://vllm.ai/benchmark/) | High-throughput inference benchmark for LLM serving |
| **ONNX Runtime** | [ONNX Performance Guide](https://onnxruntime.ai/docs/performance/) | Cross-platform inference optimization and benchmarking |
| **NVIDIA Triton / TensorRT** | [Triton Inference Server Benchmarks](https://developer.nvidia.com/nvidia-triton-inference-server) | GPU-accelerated inference benchmarking suite |
| **DeepSpeed** | [DeepSpeed Performance Docs](https://github.com/microsoft/DeepSpeed/tree/master/docs) | Parallel training and inference scaling performance |
| **PyTorch** | [PyTorch Performance Guide](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html) | Distributed training and inference performance tips |

---

### 💻 Hardware / Accelerator Benchmarks

| Hardware | Benchmark | Description |
|-----------|------------|-------------|
| **NVIDIA GPUs** | [MLPerf Results Portal](https://mlcommons.org/en/inference-results/) | Official MLPerf benchmark results for NVIDIA GPUs |
| **AMD ROCm** | [ROCm Performance Reports](https://rocmdocs.amd.com/en/latest/performance.html) | Deep learning performance data for AMD GPUs |
| **Intel Gaudi / Habana** | [Habana AI Benchmarks](https://habana.ai/training-benchmarks/) | Training and inference performance on Gaudi processors |

---

### 🌍 Community & Aggregated Dashboards

| Source | Benchmark | Description |
|---------|------------|-------------|
| **MLCommons Results Portal** | [MLCommons Results](https://mlcommons.org/en/results/) | Official MLPerf submissions (training, inference, HPC, edge) |
| **Papers With Code – Benchmarks** | [Papers With Code Leaderboards](https://paperswithcode.com/sota) | Real-time model accuracy and performance leaderboards |
| **AI-Benchmarks Portal** | [AI Benchmark Leaderboard](https://browse.ai-benchmarks.com/) | Independent repository of LLM inference cost and latency metrics |

---

## 🧰 Contributing

Pull requests welcome! Please ensure:
- Links are **official**, **recent**, and **quantitative**
- Add section summaries and short descriptions for clarity

---

## 🪴 License
[MIT](LICENSE)
