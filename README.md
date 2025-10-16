# 🚀 Awesome Benchmarking

A curated collection of **benchmarking resources, performance studies, and scalability tests** across cloud platforms, databases, orchestration tools, AI models, and distributed systems.

> 🧩 Maintained by [Performance and Scale Engineering](https://github.com/Performance-and-Scale-engineering)  
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

## 🧠 AI / ML Benchmarks (Final MLCommons Version)

### 🧩 MLCommons Benchmark Suites

| Benchmark | URL | Description |
|------------|-----|-------------|
| **AILuminate Benchmark** | [https://mlcommons.org/benchmarks/ailuminate/](https://mlcommons.org/benchmarks/ailuminate/) | Research initiative focused on illuminating AI performance and transparency. |
| **AlgoPerf: Training Algorithms** | [https://mlcommons.org/benchmarks/algorithms/](https://mlcommons.org/benchmarks/algorithms/) | Benchmarks for algorithmic efficiency and training scalability. |
| **MLPerf Automotive** | [https://mlcommons.org/benchmarks/mlperf-automotive/](https://mlcommons.org/benchmarks/mlperf-automotive/) | Evaluates autonomous driving and in-vehicle AI workloads. |
| **MLPerf Client** | [https://mlcommons.org/benchmarks/client/](https://mlcommons.org/benchmarks/client/) | Benchmark for local client-side machine learning inference. |
| **MLPerf Inference: Datacenter** | [https://mlcommons.org/benchmarks/inference-datacenter/](https://mlcommons.org/benchmarks/inference-datacenter/) | Standard suite for datacenter-scale inference performance. |
| **MLPerf Inference: Edge** | [https://mlcommons.org/benchmarks/inference-edge/](https://mlcommons.org/benchmarks/inference-edge/) | Evaluates inference workloads on embedded and IoT edge devices. |
| **MLPerf Training: HPC** | [https://mlcommons.org/benchmarks/training-hpc/](https://mlcommons.org/benchmarks/training-hpc/) | High-performance computing benchmark for distributed AI training. |
| **MLPerf Inference: Mobile** | [https://mlcommons.org/benchmarks/inference-mobile/](https://mlcommons.org/benchmarks/inference-mobile/) | Mobile device inference benchmarking suite. |
| **MLPerf Storage** | [https://mlcommons.org/benchmarks/storage/](https://mlcommons.org/benchmarks/storage/) | Storage and data I/O performance benchmarks for ML pipelines. |
| **MLPerf Inference: Tiny** | [https://mlcommons.org/benchmarks/inference-tiny/](https://mlcommons.org/benchmarks/inference-tiny/) | Benchmarking ultra-low-power and microcontroller ML inference. |
| **MLPerf Training** | [https://mlcommons.org/benchmarks/training/](https://mlcommons.org/benchmarks/training/) | Industry standard for training performance across model classes. |

---

## 🧰 Contributing

Pull requests welcome! Please ensure:
- Links are **official**, **recent**, and **quantitative**
- Add section summaries and short descriptions for clarity

---

## 🪴 License
[MIT](LICENSE)
