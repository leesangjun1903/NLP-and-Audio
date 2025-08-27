> ## Building Large Language Models and Lightweight Language Models, Language Models Algorithms, Metrics, RNN, LSTM Language Models. 

# Lecture
Deep Learning Bible - 8. Large Language Models : https://wikidocs.net/book/14965

# PaLM + RLHF - Pytorch
https://github.com/lucidrains/PaLM-rlhf-pytorch/tree/main?tab=readme-ov-file

## LangChain
- 랭체인(LangChain) 입문부터 응용까지 : https://wikidocs.net/book/14473  
- <랭체인LangChain 노트> - LangChain 한국어 튜토리얼 :  https://wikidocs.net/book/14314  , https://github.com/teddylee777/langchain-kr?tab=readme-ov-file\
- RAG From Scratch : https://github.com/langchain-ai/rag-from-scratch

## papers
### Main Architecture : Transformers
- Transformer : Attention Is All You Need
: https://github.com/huggingface/transformers/tree/main

- (Survey) A Survey of Resource-efficient LLM and Multimodal Foundation Models :  https://wikidocs.net/237419
- (Survey) Parameter-Efficient Fine-Tuning for Large Models : A Comprehensive Survey
- (Survey) Parameter-Efficient Fine-Tuning Methods for Pretrained Language Models: A Critical Review and Assessment
- (Survey) A Survey of Large Language Models : https://github.com/RUCAIBox/LLMSurvey?tab=readme-ov-file#table-of-contents , https://wikidocs.net/237619, https://arxiv.org/abs/2303.18223
- Large Language Models: A Survey(2024) : https://velog.io/@sohtks/Paper-Review-Large-Language-Models-A-Survey, https://arxiv.org/abs/2402.06196
- (Survey) A Survey on Large Language Model based Autonomous Agents
- (Survey) Towards Reasoning in Large Language Models: A Survey

- AlphaGeometry : Solving Olympiad Geometry without Human Demonstrations
- AlphaGeometry2 : Gold-medalist Performance in Solving Olympiad Geometry with AlphaGeometry2
- LoRA: Low-Rank Adaptation of Large Language Models | LLM, Fine-tuning, Mathematical Reasoning
- LongLoRA: Efficient Fine-tuning of Long-Context Large Language Models | Instruction Following, Fine-tuning, Question answering
- iVQA : Inverse Visual Question Answering: A New Benchmark and VQA Diagnosis Tool | Question answering, Reinforcement Learning
- GPT-1 : Improving Language Understanding by Generative Pre-Training
- GPT-2: Language Models are Unsupervised Multitask Learners
- GPT-3 : Language Models are Few-Shot Learners
- GPT-4 Technical Report
- DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models
- PAS: Data-Efficient Plug-and-Play Prompt Augmentation System
- DeepSeek-V3 Technical Report
- DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning
- Transformer-Squared: Self-Adaptive LLMs
- Titans: Learning to Memorize at Test Time
- PolySketchFormer: Fast Transformers via Sketching Polynomial Kernels
- Parallelizing Linear Transformers with the Delta Rule over Sequence Length
- DFloat11 : 70% Size, 100% Accuracy: Lossless LLM Compression for Efficient GPU Inference via Dynamic-Length Float
- Representation Learning Using Multi-Task Deep Neural Networks for Semantic Classification and Information Retrieval | Quety classification
- Mitigating Language-Dependent Ethnic Bias in BERT
- Learning to Summarize from Human Feedback
- PaLM: Scaling Language Modeling with Pathways
- A Length-Extrapolatable Transformer
- Understanding R1-Zero-Like Training: A Critical Perspective
- Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?
- RoBERTa: A Robustly Optimized BERT Pretraining Approach
- DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter
- Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer
- mT5: A Massively Multilingual Pre-trained Text-to-Text Transformer
- PanGu-α: Large-scale Autoregressive Pretrained Chinese Language Models with Auto-parallel Computation
- CPM-2: Large-scale Cost-effective Pre-trained Language Models
- BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension
- RoFormer: Enhanced Transformer with Rotary Position Embedding
- Direct Preference Optimization: Your Language Model is Secretly a Reward Model
- KTO: Model Alignment as Prospect Theoretic Optimization
- Noise Contrastive Alignment of Language Models with Explicit Rewards
- SpinQuant: LLM quantization with learned rotations
- CBQ: Cross-Block Quantization for Large Language Models
- LeanQuant: Accurate and Scalable Large Language Model Quantization with Loss-error-aware Grid
- KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization
- QBB: Quantization with Binary Bases for LLMs
- ZipCache: Accurate and Efficient KV Cache Quantization with Salient Token Identification
- DuQuant: Distributing Outliers via Dual Transformation Makes Stronger Quantized LLMs
- KV Cache is 1 Bit Per Channel: Efficient Large Language Model Inference with Coupled Quantization
- Evaluating Quantized Large Language Models
- SqueezeLLM: Dense-and-Sparse Quantization
- KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache
- AQLM : Extreme Compression of Large Language Models via Additive Quantization
- BiLLM: Pushing the Limit of Post-Training Quantization for LLMs
- OmniQuant: Omnidirectionally Calibrated Quantization for Large Language Models
- LoftQ: LoRA-Fine-Tuning-Aware Quantization for Large Language Models
- SpQR: Near-Lossless LLM Weight Compression
- QA-LoRA: Quantization-Aware Low-Rank Adaptation of Large Language Models
- LUT-GEMM: Quantized Matrix Multiplication based on LUTs for Efficient Inference in Large-Scale Generative Language Models
- IR-QLoRA : Accurate LoRA-Finetuning Quantization of LLMs via Information Retention
- QuIP: 2-Bit Quantization of Large Language Models With Guarantees
- Memory-Efficient Fine-Tuning of Compressed Large Language Models via sub-4-bit Integer Quantization
- Atom: Low-bit Quantization for Efficient and Accurate LLM Serving
- AWQ: Activation-aware Weight Quantization for On-Device LLM Compression and Acceleration
- LLM-QAT: Data-Free Quantization Aware Training for Large Language Models
- Outlier Suppression+: Accurate quantization of large language models by equivalent and effective shifting and scaling
- RPTQ: Reorder-based Post-training Quantization for Large Language Models
- The Case for 4-Bit Precision: k-Bit Inference Scaling Laws
- SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models
- GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers
- BiBERT: Accurate Fully Binarized BERT
- Outlier Suppression: Pushing the Limit of Low-bit Transformer Language Models
- LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale
- ZeroQuant: Eﬃcient and Aﬀordable Post-Training Quantization for Large-Scale Transformers
- Compression of Generative Pre-trained Language Models via Quantization
- BinaryBERT: Pushing the Limit of BERT Quantization
- Understanding and Overcoming the Challenges of Efficient Transformer Quantization
- TernaryBERT: Distillation-aware Ultra-low Bit BERT
- GOBO: Quantizing Attention-Based NLP Models for Low Latency and Energy Efficient Inference
- Q-BERT: Hessian Based Ultra Low Precision Quantization of BERT
- Efficient 8-Bit Quantization of Transformer Neural Machine Language Translation Model
- Q8BERT: Quantized 8Bit BERT
- Few-Shot Parameter-Efficient Fine-Tuning is Better and Cheaper than In-Context Learning
- ALBERT: A Lite BERT for Self-supervised Learning of Language Representations
- UNILM: Unified Language Model Pre-training for Natural Language Understanding and Generation
- ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators
- GLM-130B: An Open Bilingual Pre-trained Model
- ST-MoE: Designing Stable and Transferable Sparse Expert Models
- OPT: Open Pre-trained Transformer Language Models
- BLOOM: A 176B-Parameter Open-Access Multilingual Language Model
- GLaM: Efficient Scaling of Language Models with Mixture-of-Experts
- Using DeepSpeed and Megatron to Train Megatron-Turing NLG 530B, A Large-Scale Generative Language Model
- Scaling Language Models: Methods, Analysis & Insights from Training Gopher
- Chinchilla : Training Compute-Optimal Large Language Models
- LaMDA: Language Models for Dialog Applications
- LLaMA: Open and Efficient Foundation Language Models
- BloombergGPT: A Large Language Model for Finance
- GPT-NeoX-20B: An Open-Source Autoregressive Language Model
- PaLM 2 Technical Report
- Llama 2: Open Foundation and Fine-Tuned Chat Models
- Model Card and Evaluations for Claude Models
- Is ChatGPT a General-Purpose Natural Language Processing Task Solver?
- Benchmarking Large Language Models for News Summarization
- News Summarization and Evaluation in the Era of GPT-3
- Is ChatGPT A Good Translator? Yes With GPT-4 As The Engine
- Can ChatGPT Understand Too? A Comparative Study on ChatGPT and Fine-tuned BERT
- Atlas: Few-shot Learning with Retrieval Augmented Language Models
- Solving math word problems with process- and outcome-based feedback
- Large Language Models Encode Clinical Knowledge
- Scaling Laws for Neural Language Models
- Emergent Abilities of Large Language Models
- Is GPT-3 a Good Data Annotator?
- Want To Reduce Labeling Cost? GPT-3 Can Help
- GPT3Mix: Leveraging Large-scale Language Models for Text Augmentation
- ChatGPT Outperforms Crowd-Workers for Text-Annotation Tasks
- G-EVAL: NLG Evaluation Using GPT-4 with Better Human Alignment
- GPTScore: Evaluate as You Desire
- Is ChatGPT a Good NLG Evaluator? A Preliminary Study

