---

## INFO XXXX: High-Performance Computing for AI

---

### Course Prerequisites

**Strong programming skills** are required.

### Course Description

High-Performance Computing (HPC) is crucial for training and deploying large-scale AI models with billions of parameters. As models and data grow exponentially, **efficient parallelism strategies** (data, model, pipeline) across **heterogeneous architectures** (GPUs, TPUs, custom ASICs) are critical for tractable training and inference. HPC techniques also ensure **deterministic scalability, fault tolerance, and throughput** for real-time AI services. Optimized memory usage, communication efficiency (e.g., NCCL, RDMA), and compute orchestration (e.g., Ray, Kubernetes) directly impact the feasibility of deploying foundation models at scale. Without HPC, enterprises cannot achieve the performance, cost-efficiency, or reliability needed to operationalize state-of-the-art AI in mission-critical workloads.

---

### Required Textbook(s)

* **Efficient Processing of Deep Neural Networks** by Vivienne Sze et al. (MIT Press)
    * Covers kernel optimization, low-level systems design, and model–hardware interaction.
* **Programming Models for Parallel Computing** by Pavan Balaji et al. (MIT Press)
    * Covers parallel programming models in depth, crucial for understanding distributed training, inference scaling, and serverless architecture design.

---

### Course Topics

* **Distributed Training Architectures and Parallelism Techniques**
    * Model, data, tensor, and pipeline parallelism, with performance tradeoffs and implementation strategies.
    * *Example Tools:* Megatron-LM, DeepSpeed, FSDP, Horovod, PyTorch/XLA
* **Memory Optimization and Efficient Attention Mechanisms**
    * Efficient memory layout, fused kernels, and reduced precision techniques for scalable transformer training.
    * *Example Tools:* FlashAttention, xFormers, Triton, HuggingFace Accelerate
* **LLM Inference Optimization and Serverless Systems**
    * Fast weight loading, paged attention, and serverless model serving for low-latency inference.
    * *Example Tools:* VLLM, FasterTransformer, ggml, NVIDIA TensorRT-LLM, AWS Lambda, Modal
* **GPU Programming, Kernel Fusion, and Custom Ops**
    * Writing high-performance kernels, optimizing for GPU utilization, and hardware-aware code generation.
    * *Example Tools:* CUDA, Triton, TVM, ROCm
* **Large-Scale Data Pipelines for Foundation Model Training**
    * Streaming terabyte-scale datasets, sharding, prefetching, and on-the-fly transformations at scale.
    * *Example Tools:* WebDataset, Apache Arrow, NVIDIA DALI, Petastorm, HuggingFace Datasets
* **Real-Time Inference Clusters and Continuous Batching Systems**
    * Designing scalable, autoscaling systems that support dynamic request batching and structured output handling.
    * *Example Tools:* Ray Serve, HuggingFace Text Generation Inference, Inferless, KServe
* **Online and Continual Learning Systems for LLMs**
    * Efficient fine-tuning in production, online evaluation, and drift handling in evolving environments.
    * *Example Tools:* LoRA, QLoRA, AdapterHub, PEFT, streaming parameter updates
* **Deep Reinforcement Learning at Scale**
    * Parallelized actor-learner architectures, distributed replay buffers, and RL-specific compute patterns.
    * *Example Tools:* RLlib, CleanRL, OpenAI Gymnasium, TorchRL, DeepMind Launchpad
* **Cluster Management, Scheduling, and Infrastructure as Code**
    * Managing heterogeneous compute resources, autoscaling, GPU scheduling, and reproducible training pipelines.
    * *Example Tools:* Kubernetes, Slurm, Terraform, Ray, Airflow, Metaflow
* **Profiling, Debugging, and Benchmarking Large AI Workloads**
    * Diagnosing bottlenecks, optimizing hardware utilization, and regression testing across model versions.
    * *Example Tools:* Nsight Systems, PyTorch Profiler, Perf, MemoryEye, DeepSpeed Profiler

---

### Student Learning Outcomes

By the end of this course, you will be able to **build and deploy practical, state-of-the-art HPC systems for AI for enterprises.**

---

### Course Activities

* **Five programming projects** focused on building components for high-performance systems for training and inference of large models.
* There will be no quizzes or exams.

---

### Grade Breakdown

* **Programming projects constitute 100% of the final grade.**