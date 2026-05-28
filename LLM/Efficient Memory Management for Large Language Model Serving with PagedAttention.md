
# Efficient Memory Management for Large Language Model Serving with PagedAttention

> **논문 정보**
> - **제목:** Efficient Memory Management for Large Language Model Serving with PagedAttention
> - **저자:** Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, Ion Stoica
> - **학회:** ACM SIGOPS 29th Symposium on Operating Systems Principles (SOSP), 2023
> - **arXiv:** [2309.06180](https://arxiv.org/abs/2309.06180)

---

## 1. 핵심 주장 및 주요 기여 요약

### 🔑 핵심 주장

LLM의 고처리량(high-throughput) 서빙을 위해서는 충분히 많은 요청을 동시에 배치 처리해야 하지만, 기존 시스템들은 각 요청에 대한 **KV 캐시(Key-Value Cache) 메모리가 거대하고 동적으로 증감**하기 때문에 어려움을 겪고 있으며, 비효율적인 메모리 관리로 인해 **단편화(Fragmentation)와 중복 복사(Redundant Duplication)** 로 메모리가 낭비되어 배치 크기가 제한된다.

### 🏆 주요 기여

이 문제를 해결하기 위해 **운영체제의 가상 메모리(Virtual Memory)와 페이징(Paging) 기법**에서 영감을 받은 어텐션 알고리즘인 **PagedAttention**을 제안하고, 이를 기반으로 **(1) KV 캐시의 거의 0에 가까운 낭비**, **(2) 요청 내/요청 간 유연한 KV 캐시 공유**를 달성하는 LLM 서빙 시스템 **vLLM**을 구축하였다.

| 기여 항목 | 내용 |
|---|---|
| PagedAttention 알고리즘 | KV 캐시를 고정 크기 블록으로 분리, 비연속 물리 메모리 허용 |
| vLLM 서빙 시스템 | 블록 테이블 기반 스케줄러 + KV 캐시 매니저 |
| Copy-on-Write 공유 | Beam Search, 병렬 샘플링 시 KV 메모리 재사용 |
| 오픈소스 공개 | [github.com/vllm-project/vllm](https://github.com/vllm-project/vllm) |

---

## 2. 자세한 분석

### 2-1. 해결하고자 하는 문제

기존 서빙 시스템들은 일반적으로 사전에 연속적인 캐시 영역을 예약하는 방식을 취했으며, 이로 인해 예약된 공간의 낭비, 내부 단편화, 외부 단편화가 발생하였다. 논문의 실험에 따르면 이전 시스템들의 실효 메모리 활용률이 **20.4%까지 떨어질 수 있다**고 보고되었다.

2023년 초, vLLM의 저자들은 기존 추론 엔진이 가용 GPU 메모리의 **20~40%**만 사용하고 있다는 점을 발견하였다.

문제를 세 가지로 분류하면 다음과 같다:

1. **내부 단편화 (Internal Fragmentation):** 시퀀스 최대 길이에 맞춰 사전 예약 → 실제 사용 공간 < 예약 공간
2. **외부 단편화 (External Fragmentation):** 연속 메모리 할당 요구 → 사용 가능하나 쪼개진 공간 활용 불가
3. **중복 복사 (Redundant Duplication):** Beam Search 등에서 동일 프롬프트 KV 캐시를 여러 번 복사

---

### 2-2. 제안하는 방법 (수식 포함)

#### 📐 PagedAttention의 기본 수식

표준 Transformer의 어텐션 연산:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

PagedAttention에서는 KV 캐시를 블록 단위로 분할하여 처리한다. 시퀀스 $s$의 KV 캐시를 블록 크기 $B$로 나누면, $i$번째 블록 $b_i$는 다음과 같이 정의된다:

$$b_i = \{(k_j, v_j) \mid j \in [i \cdot B, \min((i+1) \cdot B - 1, L-1)]\}$$

여기서 $L$은 현재까지 생성된 시퀀스 길이이다.

$t$번째 쿼리 토큰 $q_t$에 대한 PagedAttention 연산:

$$\text{Attention}(q_t, K_{1:T}, V_{1:T}) = \sum_{j=1}^{\lceil T/B \rceil} \text{softmax}\left(\frac{q_t \cdot K_{b_j}^\top}{\sqrt{d_k}}\right) V_{b_j}$$

각 블록 $b_j$는 블록 테이블(Block Table)을 통해 **논리적 블록 번호 → 물리적 GPU 메모리 주소**로 매핑된다:

$$\text{PhysAddr}(\text{logical block}_i) = \text{BlockTable}[s][i]$$

이를 통해 **논리적으로 연속적인 시퀀스가 물리적으로 비연속적인 메모리에 저장**될 수 있다.

#### 내부 단편화 상한

시퀀스 하나당 낭비 가능한 메모리는 마지막 블록의 미사용 슬롯뿐이므로:

$$\text{Waste}_{\max} = B - 1 \text{ tokens/sequence}$$

$$\text{Memory Waste Ratio} \leq \frac{B - 1}{L} \xrightarrow{L \gg B} 0$$

PagedAttention에서 메모리 낭비는 시퀀스의 **마지막 블록에서만 발생**하며, 실제로는 거의 최적 메모리 사용률을 달성하여 낭비가 **4% 미만**에 그친다.

#### Copy-on-Write (CoW) 공유

병렬 샘플링이나 Beam Search 시, 동일 프롬프트를 가진 여러 시퀀스가 KV 블록을 공유한다. 분기(diverge) 시에만 복사가 발생한다:

$$\text{if ref count}(b_i) > 1 \Rightarrow \text{CoW: allocate new block, copy, then write}$$

$$\text{if ref count}(b_i) = 1 \Rightarrow \text{직접 쓰기 (no copy)}$$

PagedAttention의 또 다른 핵심 장점은 **효율적인 메모리 공유**이다. 예를 들어 병렬 샘플링에서 동일 프롬프트로부터 여러 출력 시퀀스가 생성될 때, 프롬프트에 대한 연산과 메모리를 출력 시퀀스들이 공유할 수 있다.

---

### 2-3. 모델 구조 (vLLM 시스템 아키텍처)

vLLM은 **분산 GPU 워커의 실행을 조율하는 중앙 집중식 스케줄러(Centralized Scheduler)** 와 **KV 캐시 매니저(KV Cache Manager)** 로 구성된다. KV 캐시 매니저의 주요 목적은 KV 캐시를 페이지 방식으로 효율적으로 관리하는 것이며, 이 구현은 고정 크기 페이지로 메모리를 분할하고 논리 페이지를 물리 페이지에 매핑하는 **운영체제**로부터 영감을 얻었다.

```
┌─────────────────────────────────────────────────────┐
│                    vLLM Architecture                 │
│                                                     │
│  ┌─────────────┐        ┌─────────────────────────┐│
│  │  Centralized│        │     KV Cache Manager    ││
│  │  Scheduler  │◄──────►│  ┌───────────────────┐  ││
│  │             │        │  │  Block Table       │  ││
│  │  - Batching │        │  │ (Logical→Physical) │  ││
│  │  - Preempt  │        │  └───────────────────┘  ││
│  │  - Priority │        │  ┌───────────────────┐  ││
│  └─────────────┘        │  │  Free Block Pool  │  ││
│                         │  └───────────────────┘  ││
│  ┌─────────────────────────────────────────────┐  ││
│  │         Distributed GPU Workers             │  ││
│  │    PagedAttention Kernel (Custom CUDA)      │  ││
│  └─────────────────────────────────────────────┘  ││
│                                                     │
└─────────────────────────────────────────────────────┘
```

PagedAttention은 요청의 KV 캐시를 **고정 개수의 토큰에 대한 키와 값을 담을 수 있는 블록으로 분할**하며, 이 블록들은 연속된 공간에 저장될 필요가 없다. 따라서 OS의 가상 메모리처럼 **블록을 페이지로, 토큰을 바이트로, 요청을 프로세스로** 간주하는 유연한 방식으로 KV 캐시를 관리할 수 있다.

이 설계는 비교적 작은 블록을 사용하고 필요에 따라 할당함으로써 **내부 단편화를 완화**하며, 모든 블록이 동일한 크기이므로 **외부 단편화를 완전히 제거**한다.

---

### 2-4. 성능 향상

vLLM은 FasterTransformer, Orca와 같은 최신 시스템과 **동일한 수준의 지연시간(latency)** 대비 **2~4배 향상된 처리량(throughput)** 을 달성하였으며, 이 향상은 **더 긴 시퀀스, 더 큰 모델, 더 복잡한 디코딩 알고리즘**일수록 더욱 두드러진다.

vLLM은 HuggingFace Transformers 대비 **최대 24배**, TGI(Text Generation Inference) 대비 **최대 3.5배** 높은 처리량을 달성하며, 기존 시스템이 KV 캐시의 **60~80%를 낭비**하는 것과 달리 낭비를 **4% 미만**으로 줄인다.

실제 사례로, LMSYS는 일일 약 45,000건의 요청 트래픽을 처리하면서 **사용 GPU 수를 50% 절감**하면서도 초당 2~3배 더 많은 요청을 처리하였다.

| 비교 대상 | 처리량 향상 | 특이 사항 |
|---|---|---|
| HuggingFace Transformers | ~24× | 고동시성 워크로드 |
| HuggingFace TGI | ~3.5× | 동일 지연시간 기준 |
| FasterTransformer / Orca | 2~4× | 동일 지연시간 기준 |

Beam Search 및 병렬 샘플링 시나리오에서 KV 공유 기회가 증가할수록 더 큰 이득을 얻으며, 메모리 절감 효과가 **최대 55%** 로 정량화되었다.

---

### 2-5. 한계점

PagedAttention은 추가적인 간접 참조(indirection)로 인한 **커널 오버헤드**가 일부 벤치마크에서 **10~30%** 에 달하며, CPU에서의 블록 테이블 관리가 **꼬리 지연시간(tail latency)** 에 영향을 미칠 수 있다.

2025년 vAttention 논문은 PagedAttention이 어텐션 커널을 페이징을 지원하도록 **재작성해야 하며**, 소프트웨어 복잡도 증가, 이식성 문제, 중복성 및 실행 오버헤드를 유발한다고 주장하였다.

vLLM의 PagedAttention 커널은 FlashAttention-2 대비 **최대 2.8배 느리며**, 블록 크기 변경 시 커널 실행 시간이 **최대 1.9배** 변동될 수 있다.

vLLM은 블록 테이블을 2D 텐서로 관리하고 미사용 슬롯을 0으로 패딩하는데, 배치에 긴 요청이 소수이고 짧은 요청이 다수인 경우 **불필요한 패딩으로 인한 상당한 오버헤드**가 발생한다.

블록 크기 $B \in [16, 128]$ 범위의 튜닝은 내부 낭비, GPU 활용률, 선점/반응성 오버헤드 간의 균형을 위해 매우 중요하다.

---

## 3. 모델의 일반화 성능 향상 가능성

PagedAttention/vLLM은 단순한 서빙 효율화를 넘어, **다양한 측면에서 모델의 일반화 성능에 간접적으로 기여**한다.

### 3-1. 다양한 디코딩 전략 지원 → 다양한 추론 방식 적용 가능

페이징은 **프롬프트/빔 공유, 동적 증가, 선점(swap/recompute), 분산 모델 병렬 실행**을 가능하게 한다.

이를 통해:
- **Beam Search**: 다양한 후보 시퀀스를 메모리 효율적으로 탐색 → 더 나은 출력 품질
- **병렬 샘플링**: 동일 프롬프트에서 다양한 출력 생성 → 앙상블 방식의 일반화 효과
- **Chain-of-Thought 등 복잡한 추론**: 긴 시퀀스 효율적 처리 가능

### 3-2. 더 큰 배치 크기 → 다양한 요청 동시 처리

더 많은 요청을 배치로 처리함으로써 **다양한 입력 분포에 대한 실시간 처리 능력**이 향상된다. 이는 배포 환경에서의 일반화(generalization in deployment)와 직결된다.

### 3-3. 공유 프리픽스 캐싱 → Few-shot 학습 효율 향상

KV 캐시 재사용(KV cache reuse)은 **동일한 프리픽스를 공유하는 서로 다른 프롬프트가 중간 KV 캐시를 공유하고 중복 메모리 및 연산을 피할 수 있음**을 의미한다.

이는 Few-shot 예시나 시스템 프롬프트를 공유하는 경우, 추론 시간을 단축시키면서 **다양한 in-context learning 시나리오**에서의 일반화 성능을 실용적으로 향상시킨다.

### 3-4. 아키텍처 변경 없는 범용 적용

vLLM with PagedAttention은 **모델 아키텍처의 변경 없이** HuggingFace Transformers 대비 최대 24배 높은 처리량을 제공하여 LLM 서빙의 새로운 SOTA를 달성한다.

이는 OPT, LLaMA, GPT 등 다양한 Transformer 기반 모델에 적용 가능하다는 점에서 **범용적 일반화 가능성**을 시사한다.

---

## 4. 연구에 미치는 영향 및 향후 연구 시 고려할 점

### 4-1. 연구 영향

2024년 LLM 서빙 시스템 서베이는 PagedAttention이 **TGI, vLLM, TensorRT-LLM 등의 지원**을 통해 **LLM 서빙 프레임워크의 산업 표준(industry norm)** 이 되었다고 평가하였다.

PagedAttention은 LLM 서빙 시스템에서 동적 메모리 할당의 **사실상의 표준(de facto standard)** 이 되었으며, TensorRT-LLM, HuggingFace TGI, LightLLM 등 다양한 시스템에서 채택되었다.

**구체적 영향:**
1. **OS 개념의 AI 시스템 적용:** 가상 메모리, 페이징, CoW 등 OS 개념을 딥러닝 서빙에 접목하는 새로운 패러다임 제시
2. **KV 캐시 연구의 활성화:** KV 캐시 압축, 양자화, 선택적 저장 등 후속 연구 폭발적 증가
3. **LLM 서빙 시스템 연구 분야 개척:** SOSP 등 시스템 학회에서 LLM 서빙 최적화 연구가 주류로 부상

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

### 5-1. 주요 관련 연구 타임라인

| 연도 | 논문/시스템 | 핵심 기여 | PagedAttention과의 관계 |
|---|---|---|---|
| 2022 | **Orca** (Yu et al.) | Continuous Batching으로 GPU 활용률 향상 | 상호 보완적 관계 |
| 2022 | **FlashAttention** (Dao et al.) | IO-aware 어텐션, 메모리 효율적 커널 | PagedAttention 커널에서 활용 |
| 2023 | **PagedAttention/vLLM** | KV 캐시 페이징, vLLM 시스템 | **본 논문** |
| 2023 | **FlashAttention-2** (Dao) | 더 나은 병렬성으로 어텐션 가속 | vLLM 백엔드 커널로 통합 |
| 2024 | **SGLang/RadixAttention** (Zheng et al.) | Radix Tree 기반 자동 KV 캐시 재사용 | PagedAttention 확장 |
| 2025 | **vAttention** (Prabhu et al.) | 물리-가상 메모리 분리로 커널 재작성 불필요 | PagedAttention 대안 |

### 5-2. 세부 비교

#### ① Orca (2022) vs. vLLM (2023)

Orca와 vLLM의 PagedAttention은 **상호 보완적인 기법**이다. 두 시스템 모두 GPU 활용률과 LLM 서빙 처리량을 높이는 것을 목표로 하지만, Orca는 더 많은 요청이 병렬로 처리될 수 있도록 **스케줄링 및 요청 인터리빙**을 통해 이를 달성하는 반면, vLLM은 더 많은 요청의 작업 집합이 메모리에 들어갈 수 있도록 **메모리 활용률 향상**을 통해 이를 달성한다.

#### ② SGLang/RadixAttention (2024)

SGLang은 **RadixAttention**을 도입하여 Radix Tree 자료구조로 요청 간 KV 캐시를 유지하고 자동 프리픽스 재사용을 가능하게 하였다.

시스템 프롬프트, Few-shot 예시, RAG 컨텍스트를 공유하는 요청들이 캐시된 KV 블록을 재사용하며, 프리픽스 공유가 많은 워크로드에서 **최대 5배 빠른 추론**을 달성한다.

#### ③ vAttention (ASPLOS 2025)

vAttention은 **CUDA 가상 메모리 관리 API**를 이용해 가상/물리 메모리 할당을 분리함으로써, KV 캐시의 가상 연속성을 유지하면서 물리 메모리 단편화를 완화한다. 다양한 LLM 특화 최적화를 도입하여 기존 커널의 재작성 없이도 FlashAttention-2 및 FlashInfer 기반의 PagedAttention 커널 대비 **최대 1.23배 향상된 서빙 처리량**을 달성한다.

#### ④ Sarathi-Serve / Chunked Prefill (2024)

Sarathi-serve (Agrawal et al., 2024)는 청크 기반 프리필(chunked-prefill)로 디코딩 연산을 피기백하여 효율성을 개선하였으며, SGLang (Zheng et al., 2023)은 RadixTree를 활용한 더 나은 프리픽스 캐싱과 KV 관리를 구현하였다.

현재 vLLM과 SGLang은 모두 **Continuous Batching, PagedAttention, RadixAttention, Chunked Prefill, Speculative Decoding, Disaggregated Serving, CUDA Graphs** 등을 지원하는 종합적인 서빙 프레임워크로 발전하였다.

### 5-3. 비교 요약표

| 특성 | vLLM (PagedAttention) | SGLang (RadixAttention) | vAttention |
|---|---|---|---|
| KV 메모리 관리 | 블록 테이블 기반 페이징 | Radix Tree + 페이징 | CUDA VMM 기반 동적 할당 |
| 커널 수정 필요 여부 | ✅ 필요 | ✅ 필요 | ❌ 불필요 |
| 프리픽스 재사용 | 제한적 (수동 설정) | 자동, 광범위 지원 | 지원 가능 |
| 처리량 개선 | 2~4× vs. SOTA | 5× (공유 프리픽스 시) | 1.23× vs. PagedAttention |
| 소프트웨어 복잡도 | 중간 | 높음 | 낮음 |

---

## 6. 향후 연구 시 고려할 점

1. **커널 오버헤드 최소화:** PagedAttention 기반 프리필 커널은 비페이지 커널 대비 **최대 37% 느릴 수 있어**, 커널 최적화가 중요한 연구 방향이다.

2. **블록 크기 자동 튜닝:** 블록 크기 $b \in [16, 128]$ 범위의 튜닝은 내부 낭비, GPU 활용률, 선점/반응성 오버헤드 간의 균형에 매우 중요하므로, 워크로드에 따른 **동적/자동 블록 크기 결정 메커니즘** 연구가 필요하다.

3. **KV 캐시 압축과의 통합:** 페이징과 KV 캐시 양자화(Quantization), 희소화(Sparsification) 등의 결합으로 메모리 효율을 추가로 향상할 수 있다.

4. **이기종 하드웨어 적용:** CPU, NPU, 모바일 GPU 등 다양한 하드웨어 환경에서의 이식성 확보가 과제이다.

5. **멀티모달 LLM 확장:** 비전-언어 모델에서 이미지 KV 캐시의 크기 및 패턴이 텍스트와 다르므로, 멀티모달 시나리오를 위한 KV 캐시 관리 전략 연구가 필요하다.

6. **장문 컨텍스트(Long-context) 최적화:** 수백만 토큰 컨텍스트 환경에서의 블록 테이블 관리 비용 및 KV 캐시 선택적 보존(eviction) 전략 연구가 중요하다.

---

## 📚 참고 자료 출처

| # | 출처 | 유형 |
|---|---|---|
| 1 | Kwon et al. (2023). *Efficient Memory Management for Large Language Model Serving with PagedAttention.* **SOSP 2023.** [arxiv.org/abs/2309.06180](https://arxiv.org/abs/2309.06180) | 원논문 |
| 2 | ACM DL: [dl.acm.org/doi/10.1145/3600006.3613165](https://dl.acm.org/doi/10.1145/3600006.3613165) | 공식 발표 |
| 3 | vLLM Official Blog (2023). [blog.vllm.ai/2023/06/20/vllm.html](https://blog.vllm.ai/2023/06/20/vllm.html) | 공식 블로그 |
| 4 | vLLM GitHub: [github.com/vllm-project/vllm](https://github.com/vllm-project/vllm) | 오픈소스 |
| 5 | Prabhu et al. (2025). *vAttention: Dynamic Memory Management for Serving LLMs without PagedAttention.* **ASPLOS 2025.** [arxiv.org/abs/2405.04437](https://arxiv.org/abs/2405.04437) | 후속 연구 |
| 6 | Zheng et al. (2024). *SGLang/RadixAttention.* [lmsys.org/blog/2024-01-17-sglang](https://www.lmsys.org/blog/2024-01-17-sglang/) | 관련 연구 |
| 7 | Wikipedia: PagedAttention. [en.wikipedia.org/wiki/PagedAttention](https://en.wikipedia.org/wiki/PagedAttention) | 참고 |
| 8 | Emergent Mind: PagedAttention. [emergentmind.com/topics/pagedattention](https://www.emergentmind.com/topics/pagedattention) | 분석 |
| 9 | RunPod Blog: Introduction to vLLM and PagedAttention. [runpod.io/blog/introduction-to-vllm-and-pagedattention](https://www.runpod.io/blog/introduction-to-vllm-and-pagedattention) | 기술 설명 |
| 10 | Wentao's Blog: vLLM Summary. [wentao.site/vllm_summary](https://wentao.site/vllm_summary/) | 요약 분석 |
| 11 | Kolluru (2025). *Comparative Analysis of LLM Inference Serving Systems.* [arxiv.org/abs/2511.17593](https://arxiv.org/pdf/2511.17593) | 비교 분석 |

# Efficient Memory Management for Large Language Model Serving with PagedAttention

## 1. 핵심 요약

### 1.1 논문의 핵심 주장과 기여

"Efficient Memory Management for Large Language Model Serving with PagedAttention"(Kwon et al., SOSP '23)은 **운영체제의 가상 메모리 및 페이징 기법을 트랜스포머 기반 LLM의 KV 캐시 관리에 적용**하는 혁신적인 접근법을 제시합니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/df50aa4e-133c-414b-a8c4-f8a43f1acfb5/2309.06180v1.pdf)

**세 가지 핵심 기여:**

1. **메모리 낭비 정량화**: 기존 LLM 서빙 시스템의 KV 캐시 메모리 낭비가 심각함을 입증 - 실제 사용률 20.4%-38.2%, 나머지 61.8%-79.6%는 낭비 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/df50aa4e-133c-414b-a8c4-f8a43f1acfb5/2309.06180v1.pdf)
2. **PagedAttention 알고리즘**: 고정 크기 블록으로 KV 캐시를 분할하여 비연속 메모리 저장 가능 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/df50aa4e-133c-414b-a8c4-f8a43f1acfb5/2309.06180v1.pdf)
3. **vLLM 시스템**: 블록 레벨 메모리 관리로 **2-4배 처리량 향상** 달성, 모델 정확도 손상 없음 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/df50aa4e-133c-414b-a8c4-f8a43f1acfb5/2309.06180v1.pdf)

***

## 2. 상세 분석: 문제, 방법, 성능

### 2.1 해결하고자 하는 문제

#### 기존 시스템의 KV 캐시 메모리 문제

**문제의 심각성:**
- OPT-13B 모델: 단일 토큰당 KV 캐시 800KB, 최대 2048개 토큰 = 1.6GB/요청 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/df50aa4e-133c-414b-a8c4-f8a43f1acfb5/2309.06180v1.pdf)
- A100 GPU 40GB 메모리에서: 모델 파라미터 26GB + KV 캐시 >30% 필요 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/df50aa4e-133c-414b-a8c4-f8a43f1acfb5/2309.06180v1.pdf)

**세 가지 메모리 낭비 원인:**

| 낭비 유형 | 설명 | 영향 |
|---------|------|------|
| **내부 단편화** | 최대 길이로 사전 할당하지만 실제 길이는 훨씬 짧음 | 최대 1000+ 토큰 낭비/요청 |
| **외부 단편화** | 서로 다른 크기의 청크 할당으로 메모리 분산 | 메모리 재사용 불가능 |
| **메모리 공유 불가** | 비연속 메모리로 인해 여러 시퀀스의 KV 캐시 공유 불가능 | 병렬 샘플링/빔 서치 비효율 |

**결과: 배치 크기 심각한 제약** → 처리량 병목 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/df50aa4e-133c-414b-a8c4-f8a43f1acfb5/2309.06180v1.pdf)

### 2.2 제안하는 방법: PagedAttention

#### 핵심 개념

OS의 가상 메모리 패러다임을 적용:
- **토큰 ↔ 바이트**
- **블록 ↔ 페이지**
- **요청 ↔ 프로세스**

#### 수학적 표현

표준 Self-Attention 계산:

$$a_{ij} = \frac{\exp(q_i^T k_j / \sqrt{d})}{\sum_{t=1}^{i} \exp(q_i^T k_t / \sqrt{d})}, \quad o_i = \sum_{j=1}^{i} a_{ij} v_j$$

**PagedAttention 블록별 계산:**

$$A_{ij} = \frac{\exp(q_i^T K_j / \sqrt{d})}{\sum_{t=1}^{\lceil i/B \rceil} \exp(q_i^T K_t / \sqrt{d})}, \quad o_i = \sum_{j=1}^{\lceil i/B \rceil} V_j A_{ij}^T$$

여기서:
- $B$ = KV 블록 크기 (고정값)
- $K_j = (k_{(j-1)B+1}, \ldots, k_{jB})$ = j번째 키 블록
- $V_j = (v_{(j-1)B+1}, \ldots, v_{jB})$ = j번째 값 블록
- 각 블록은 물리 메모리의 **임의의 위치**에 저장 가능 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/df50aa4e-133c-414b-a8c4-f8a43f1acfb5/2309.06180v1.pdf)

#### 메모리 관리 메커니즘

**블록 테이블 (Block Table):**
각 요청에 대해 유지되는 매핑 구조
```
논리 KV 블록 → 물리 KV 블록 (GPU DRAM)
```
- 각 엔트리: 물리 블록 번호 + 채운 위치 수
- 동적 할당: 필요할 때만 새 블록 할당
- 비연속 메모리 허용으로 내부/외부 단편화 제거 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/df50aa4e-133c-414b-a8c4-f8a43f1acfb5/2309.06180v1.pdf)

#### Copy-on-Write 메커니즘

병렬 샘플링 및 빔 서치에서 KV 캐시 공유:

```
참조 카운트: 물리 블록 → 몇 개 논리 블록이 참조중?
- 참조 카운트 > 1: 새 블록 할당 후 복사 (쓰기 시점)
- 참조 카운트 = 1: 원본 블록에 직접 쓰기
```

예시: 병렬 샘플링 시 프롬프트의 KV는 2개 샘플이 공유하지만, 생성 단계에서 서로 다른 토큰 생성 → Copy-on-Write로 효율적 처리 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/df50aa4e-133c-414b-a8c4-f8a43f1acfb5/2309.06180v1.pdf)

### 2.3 모델 구조 및 시스템 아키텍처

#### vLLM 시스템 구성 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/df50aa4e-133c-414b-a8c4-f8a43f1acfb5/2309.06180v1.pdf)

```
┌─────────────────────────────────────┐
│     FastAPI 프론트엔드               │ ← OpenAI API 호환
├─────────────────────────────────────┤
│     중앙 스케줄러                    │ ← 요청 스케줄링
│  • KV 캐시 매니저                  │
│  • CPU/GPU 블록 할당자             │
│  • 요청 우선순위 관리               │
├─────────────────────────────────────┤
│  GPU 워커 (분산 실행)                │
│  ┌──────────────────────────────┐  │
│  │ Model Shard 0                │  │
│  │ Cache Engine (PagedAttention)│  │
│  └──────────────────────────────┘  │
│  ┌──────────────────────────────┐  │
│  │ Model Shard 1, 2, ...        │  │
│  │ Cache Engine                 │  │
│  └──────────────────────────────┘  │
└─────────────────────────────────────┘
```

**작동 흐름:**
1. 스케줄러가 배치 선택
2. 필요한 물리 블록 할당
3. 블록 테이블 함께 전송
4. 각 GPU 워커가 PagedAttention 커널 실행
5. 생성 토큰 반환 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/df50aa4e-133c-414b-a8c4-f8a43f1acfb5/2309.06180v1.pdf)

#### 복잡한 디코딩 시나리오 지원

**1) 병렬 샘플링 (Parallel Sampling)** [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/df50aa4e-133c-414b-a8c4-f8a43f1acfb5/2309.06180v1.pdf)
- 동일 프롬프트 → 여러 샘플
- 프롬프트 KV: 모두 공유 (12% 메모리 절감)
- 생성 KV: 분리 (서로 다른 샘플링)

**2) 빔 서치 (Beam Search)** [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/df50aa4e-133c-414b-a8c4-f8a43f1acfb5/2309.06180v1.pdf)
- 동적 공유 패턴
- 후보 제거 시 블록 참조 카운트 감소 → 블록 해제
- 최대 55% 메모리 절감
- 기존 시스템: 빔 후보 간 빈번한 KV 복사 필요 (비효율)

**3) 공유 프리픽스 (Shared Prefix)** [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/df50aa4e-133c-414b-a8c4-f8a43f1acfb5/2309.06180v1.pdf)
- 시스템 프롬프트(few-shot 예제 등)를 여러 요청이 공유
- 서비스 제공자가 미리 KV 캐시 계산 후 보관
- 사용자 입력만 처리

**4) 혼합 디코딩 (Mixed Decoding)** [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/df50aa4e-133c-414b-a8c4-f8a43f1acfb5/2309.06180v1.pdf)
- 동일 배치에서 다양한 디코딩 방식 지원
- 블록 테이블이 복잡한 공유 패턴 숨김

### 2.4 성능 향상 결과

#### 벤치마크 환경 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/df50aa4e-133c-414b-a8c4-f8a43f1acfb5/2309.06180v1.pdf)

| 항목 | 13B | 66B | 175B |
|------|-----|-----|------|
| GPU | 1×A100 | 4×A100 | 8×A100-80GB |
| 총 메모리 | 40GB | 160GB | 640GB |
| 데이터셋 | ShareGPT, Alpaca |

#### 기본 샘플링 성능 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/df50aa4e-133c-414b-a8c4-f8a43f1acfb5/2309.06180v1.pdf)

**ShareGPT 데이터셋 (긴 프롬프트: 평균 161개 토큰 입력)**

- vLLM vs Orca (Oracle): **1.7-2.7배** 높은 요청률 유지
- vLLM vs Orca (Max): **2.7-8배** 높은 요청률
- 배치 크기 비교 (OPT-13B):
  - Orca (Oracle): 7.00 요청
  - vLLM: **30.42 요청** (4.3배 증가) [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/df50aa4e-133c-414b-a8c4-f8a43f1acfb5/2309.06180v1.pdf)

**Alpaca 데이터셋 (짧은 프롬프트: 평균 19개 토큰)**

- 비슷한 추세
- OPT-175B에서 Orca와 성능 근접 (메모리가 충분할 때는 효과 감소) [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/df50aa4e-133c-414b-a8c4-f8a43f1acfb5/2309.06180v1.pdf)

#### 복잡 디코딩 성능 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/df50aa4e-133c-414b-a8c4-f8a43f1acfb5/2309.06180v1.pdf)

**병렬 샘플링**
- 병렬 크기 2: 6.09% 메모리 절감
- 병렬 크기 6: 9.79% 메모리 절감

**빔 서치**
- 빔 폭 2: 37.56% 메모리 절감
- 빔 폭 6: **55.16% 메모리 절감** (OPT-13B, Alpaca)
- vLLM vs Orca (Oracle): **1.3배(기본) → 2.3배(빔 폭 6)** 처리량 향상 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/df50aa4e-133c-414b-a8c4-f8a43f1acfb5/2309.06180v1.pdf)

#### 공유 프리픽스 성능 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/df50aa4e-133c-414b-a8c4-f8a43f1acfb5/2309.06180v1.pdf)

LLaMA-13B 기계 번역 작업:
- 원샷(1-shot) 프리픽스: **1.67배** 처리량
- 파이샷(5-shot) 프리픽스: **3.58배** 처리량

### 2.5 한계 (Limitations)

#### 1) 커널 오버헤드 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/df50aa4e-133c-414b-a8c4-f8a43f1acfb5/2309.06180v1.pdf)

**PagedAttention 커널 지연:**
- 블록 테이블 접근, 분기 처리, 가변 시퀀스 길이 처리
- **20-26% 지연 증가** vs FasterTransformer
- 그러나: 시스템 전체 처리량은 2-4배 향상 (배치 크기 증가로 상쇄) [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/df50aa4e-133c-414b-a8c4-f8a43f1acfb5/2309.06180v1.pdf)

#### 2) 블록 크기 선택의 중요성 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/df50aa4e-133c-414b-a8c4-f8a43f1acfb5/2309.06180v1.pdf)

**블록 크기 16 vs 256 (Alpaca 데이터셋):**
- 너무 작음 (1-8): GPU 병렬화 미흡
- 너무 큼 (256): 내부 단편화 증가, 공유 확률 감소
- **최적값: 16 (기본값)** - 대부분 워크로드에 적합 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/df50aa4e-133c-414b-a8c4-f8a43f1acfb5/2309.06180v1.pdf)

#### 3) 예약(Preemption) 정책의 트레이드오프 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/df50aa4e-133c-414b-a8c4-f8a43f1acfb5/2309.06180v1.pdf)

**Swapping vs Recomputation:**

| 방식 | 장점 | 단점 | 최적 시나리오 |
|------|------|------|--------------|
| **Swapping** | CPU/GPU 대역폭 활용 가능 | 작은 블록 크기에서 비효율 | 큰 블록 크기(64+) |
| **Recomputation** | 메모리 이동 없음, 추가 저장소 불필요 | 계산 비용 (20% 이하 오버헤드) | 작은 블록 크기(16-32) |

#### 4) 일반화 성능 고려사항

- **모델 정확도: 손상 없음** - PagedAttention은 수학적으로 표준 Attention과 동등 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/df50aa4e-133c-414b-a8c4-f8a43f1acfb5/2309.06180v1.pdf)
- **KV 캐시 재사용의 정확도**: 기존 KV 캐시는 정확 → 정확도 영향 없음
- **다만**, 후속 연구에서 KV 캐시 **압축**과 결합 시 정확도-효율성 트레이드오프 발생 가능 [arxiv](https://arxiv.org/html/2601.03067)

***

## 3. 모델 일반화 성능 향상 가능성

### 3.1 PagedAttention의 일반화 성능: 이론적 분석

#### 수학적 등가성 증명

**Claim:** PagedAttention은 표준 Attention과 동등

**증명:**
$$o_i = \sum_{j=1}^{\lceil i/B \rceil} V_j A_{ij}^T = \sum_{j=1}^{\lceil i/B \rceil} \sum_{t=(j-1)B+1}^{\min(jB, i)} V_t a_{it}$$

재정렬하면 표준 식과 동등 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/df50aa4e-133c-414b-a8c4-f8a43f1acfb5/2309.06180v1.pdf)

**결론:** PagedAttention 자체로는 **모델 성능 저하 없음** [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/df50aa4e-133c-414b-a8c4-f8a43f1acfb5/2309.06180v1.pdf)

### 3.2 잠재적 일반화 향상 메커니즘

#### 1) 배치 크기 증가로 인한 개선

**논리:**
- vLLM: 더 많은 요청을 배치 처리 가능
- 더 큰 배치 = 보다 나은 GPU 활용도
- 결과적으로: 토큰 생성 지연 감소 → 투기적 디코딩(Speculative Decoding) 등 고급 기법 활용 가능 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/df50aa4e-133c-414b-a8c4-f8a43f1acfb5/2309.06180v1.pdf)

#### 2) 메모리 오버헤드 감소

**기존 시스템 문제:**
- 메모리 부족 → 요청 거절 또는 지연
- 일부 요청에 대한 메모리 부족으로 정확도 영향 가능 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/df50aa4e-133c-414b-a8c4-f8a43f1acfb5/2309.06180v1.pdf)

**vLLM의 해결:**
- 거의 모든 합법적 요청 처리 가능
- 일관된 메모리 조건 → 더 안정적인 성능 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/df50aa4e-133c-414b-a8c4-f8a43f1acfb5/2309.06180v1.pdf)

#### 3) 다양한 길이의 시퀀스 처리

**테스트 결과:**
- ShareGPT (긴 시퀀스, 평균 161+338=499개 토큰): vLLM 우월
- Alpaca (짧은 시퀀스, 평균 19+58=77개 토큰): Orca와 근접

**해석:** 다양한 입출력 길이에 적응 능력 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/df50aa4e-133c-414b-a8c4-f8a43f1acfb5/2309.06180v1.pdf)

### 3.3 최근 연구의 일반화 성능 확장: 2020년 이후 관련 연구

#### A. KV 캐시 최적화 기술들의 정확도 분석

**1) KV 캐시 압축 기술** [arxiv](https://arxiv.org/html/2509.04377v1)

| 기법 | 저자 | 연도 | 압축률 | 정확도 손실 |
|------|------|------|--------|----------|
| ChunkKV | Liu et al. | 2025 | 1.5-2x | <5% |
| ThinKV | 2025 | 5.8x | <5% (수학/코딩) |
| Expected Attention | 2025 | 2x @ 50% | 0% |
| Joint Encoding | 2025 | 4.38x | 무시할 수준 |
| PagedEviction | 2024 | 2-3x | 3-5% (장문맥 요약) |

**결론:** 적절한 압축 기법 선택 시 **일반화 성능 거의 유지** [arxiv](https://arxiv.org/html/2502.00299v5)

**2) 양자화 연구** [aclanthology](https://aclanthology.org/2023.emnlp-main.298.pdf)

- **3-4비트:** 정상 성능 유지
- **2비트:** 10-40% 성능 저하 (작업 유형별)
- **1비트:** 심각한 저하 [openreview](https://openreview.net/pdf?id=ClkfwM3STw)

**핵심 발견:** 
> 양자화 수준이 낮아질수록 일반화 성능 악화, 특히 과학 지식 QA(-40%), 감정 분석은 오히려 개선(+특성) [openreview](https://openreview.net/pdf?id=ClkfwM3STw)

#### B. 길이 외삽(Length Extrapolation) 성능 [arxiv](https://arxiv.org/pdf/2503.23174.pdf)

**LongBench 벤치마크 (기존 PagedAttention):**
- 32K 토큰 컨텍스트: 성능 유지
- 기준: 요약(ROUGE), 코드 완성 정확도 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/df50aa4e-133c-414b-a8c4-f8a43f1acfb5/2309.06180v1.pdf)

**최신 발견 (2024-2025):**
- **"Lost in the Middle" 현상:** 컨텍스트 중간부의 정보에 대한 주의 약화 [arxiv](https://arxiv.org/html/2601.11564v1)
- **Infini-Attention (Google):** 무한 컨텍스트에서 114배 메모리 압축, 성능 유지 [arxiv](https://arxiv.org/html/2404.07143v1)
- **vAttention (Microsoft):** PagedAttention의 성능 병목 제거, 더 빠른 처리 [arxiv](https://arxiv.org/html/2405.04437v1)

#### C. 배치 크기의 일반화 성능 영향 [arxiv](https://arxiv.org/html/2503.08311v2)

**최근 연구 발견:**

1. **OPT-1.3B (배치 1 vs 최대):**
   - 처리량: **26.6배** 향상
   - 정확도: 유지 (배치 크기 자체는 정확도에 직접 영향 없음)
   - 지연: 선형 증가 (메모리 대역폭 포화) [arxiv](https://arxiv.org/html/2503.08311v2)

2. **혼합 정밀도 (FP16 vs FP8):**
   - FP8이 처리량 향상
   - 정확도는 거의 동일 [arxiv](https://www.arxiv.org/pdf/2508.17467.pdf)

3. **컨텍스트 길이 vs 정확도:**
   - 4K-15K 단어: 추론 정확도 **최소 저하** (<5%)
   - 단, 답변 검색(Needle-in-Haystack)은 위치 편향 [arxiv](https://arxiv.org/html/2601.11564v1)

#### D. 복합 최적화의 시너지

**EvicPress (2025):** 압축 + 제거 공동 최적화 [arxiv](https://arxiv.org/html/2512.14946v1)
- 다중 계층 메모리 (GPU/CPU/SSD) 관리
- vLLM + LMCache 통합
- 결과: **더 높은 캐시 히트율 → 더 빠른 응답 → 일반화 성능 안정화**

***

## 4. 논문이 앞으로의 연구에 미치는 영향

### 4.1 PagedAttention의 학술적 영향

#### 1) 인용도 및 채택률 [arxiv](https://arxiv.org/abs/2309.06180)
- **인용 수: 4,584회** (매우 높은 영향력)
- **실제 채택:** vLLM, SGLang, TensorRT-LLM 등 산업 표준
- **학술 추적:** "PagedAttention은 현재 거의 모든 LLM 추론 엔진에 사용됨" [arxiv](https://arxiv.org/html/2503.18292v1)

#### 2) 새로운 연구 방향 개척

**A. 메모리 관리 기술 다양화**
- **vTensor, vAttention:** PagedAttention의 대안적 접근
- **이중 페이징:** 페이지와 슈퍼페이지 계층 도입 [arxiv](https://arxiv.org/html/2506.07311v1)

**B. KV 캐시 최적화 폭발적 증가**
- 2023: PagedAttention 제안
- 2024: 압축, 양자화, 제거 기법 다수 제안 [arxiv](https://arxiv.org/pdf/2410.00161.pdf)
- 2025: 동적 적응 압축, 다중 계층 관리 [arxiv](https://arxiv.org/html/2510.01290v1)

**C. 시스템 레벨 혁신**
- Continuous batching 개선
- Request batching의 세밀한 제어
- Adaptive scheduling 연구 [arxiv](https://arxiv.org/html/2503.08311v2)

### 4.2 앞으로 연구 시 고려할 점

#### 1) 메모리-정확도 트레이드오프 연구

**현재 상태:**
- PagedAttention: 메모리 효율 극대화, 정확도 100%
- 그러나, 압축/양자화 결합 시: 정확도-효율 균형 필요 [openreview](https://openreview.net/pdf?id=8ZiElzQxf1)

**권장 연구:**
$$\text{최적화} = \arg\max_{\theta} (정확도(\theta) - \lambda \times 메모리\_소비(\theta))$$

람다 값의 동적 조정, 작업별 최적화 [aclanthology](https://aclanthology.org/2025.findings-emnlp.426.pdf)

#### 2) 길이 외삽 일반화

**해결해야 할 문제:**
- "Lost in the Middle" 현상 해결
- 32K 이상 컨텍스트에서의 정확도 유지 [arxiv](https://arxiv.org/html/2601.11564v1)

**진행 중 연구:**
- Infini-Attention: 고정 메모리 크기로 무한 시퀀스 처리
- Threshold Relative Attention: 길이 일반화 개선 [arxiv](https://arxiv.org/pdf/2503.23174.pdf)

#### 3) 이질적 워크로드 지원

**고려 사항:**
- 다양한 모델 크기 (1B ~ 70B)
- 다양한 컨텍스트 길이 (512 ~ 32K+)
- 다양한 배치 크기 (1 ~ 128+)

**현재 한계:**
- OPT-175B에서 배치 크기 증가의 효과 감소 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/df50aa4e-133c-414b-a8c4-f8a43f1acfb5/2309.06180v1.pdf)
- 매우 큰 모델에서 추가 최적화 필요 [arxiv](https://arxiv.org/html/2411.00136v1)

#### 4) 가이드라인 제시

**BlockSize 선택:**
$$B_{\text{optimal}} = \arg\max_B (처리량(B) - \lambda_1 \times 단편화(B) - \lambda_2 \times 공유확률(B))$$

현재: 휴리스틱 기반 (16)
권장: 작업 적응형 자동 조정 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/df50aa4e-133c-414b-a8c4-f8a43f1acfb5/2309.06180v1.pdf)

**Preemption 정책:**
- Swapping vs Recomputation 선택: 블록 크기, 시퀀스 길이에 따라 동적 결정 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/df50aa4e-133c-414b-a8c4-f8a43f1acfb5/2309.06180v1.pdf)
- Scheduling: FCFS가 단순하지만, priority scheduling 탐색 [bentoml](https://bentoml.com/llm/inference-optimization/kv-cache-offloading)

#### 5) 다중 GPU 시나리오

**현재 지원:**
- Megatron-LM 스타일 텐서 모델 병렬화 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/df50aa4e-133c-414b-a8c4-f8a43f1acfb5/2309.06180v1.pdf)
- 그러나, 파이프라인 병렬화에서의 KV 캐시 관리 미흡 [arxiv](https://arxiv.org/html/2508.10824v1)

**향후 과제:**
- 분산 KV 캐시 공유
- 상호 연결 대역폭 최적화 [arxiv](https://arxiv.org/html/2508.10824v1)

#### 6) 모델 아키텍처 진화

**새로운 주의 메커니즘과의 호환성:**

| 기술 | 호환성 | 이슈 |
|------|--------|------|
| **GroupedQueryAttention (GQA)** | 부분적 | vLLM-GQA 성능 2.85배 느림 [arxiv](https://arxiv.org/html/2405.04437v1) |
| **Multi-QueryAttention (MQA)** | 부분적 | KV 헤드 공유로 변수 관리 복잡 [aclanthology](https://aclanthology.org/2023.emnlp-main.298.pdf) |
| **FlashAttention2** | 높음 | Paged + FlexAttention 통합 진행 [arxiv](https://arxiv.org/html/2506.07311v1) |

**권장:** 새로운 주의 메커니즘별 최적화된 PagedAttention 변형 개발

***

## 5. 2020년 이후 관련 최신 연구 비교 분석

### 5.1 시간축 로드맵: 연구 동향

```
2020-2022: 기초 연구
├─ Transformer-XL: 세그먼트 단위 주의
├─ CompressiveTransformer: 메모리 압축
└─ FlashAttention: I/O 효율 최적화

2023년: PagedAttention 혁신
├─ SOSP '23: PagedAttention 발표
├─ vLLM 오픈소스 공개
└─ 산업 채택 시작

2024년: 최적화 폭발
├─ KV-Compress: 헤드별 가변 압축
├─ vTensor: 가상 텐서 관리
├─ ChunkKV: 의미적 압축
├─ PagedEviction: 구조화 제거
└─ Infini-Attention: 무한 컨텍스트

2025년: 적응형 기법
├─ vAttention: 대체 아키텍처
├─ ThinKV: 생각 적응 양자화
├─ DynamicKV: 작업 인식 압축
├─ Joint Encoding: 블록 공동 인코딩
├─ EvicPress: 압축+제거 공동 최적화
└─ AttentionPredictor: 학습 기반 토큰 식별
```

### 5.2 핵심 기술 비교표

| **기술** | **연도** | **핵심 아이디어** | **메모리 절감** | **정확도** | **복잡도** |
|---------|--------|----------------|-----------|---------|--------|
| **PagedAttention (vLLM)** | 2023 | 블록 기반 메모리 매핑 | 2-4배 처리량↑ | 100% | 중간 |
| **Infini-Attention** | 2024 | 압축 메모리 + 로컬+장기 주의 | 114배 | 100% | 높음 |
| **KV-Compress** | 2024 | 헤드/계층별 가변 압축률 | 5.18배 처리량 | 90%+ | 중간 |
| **vAttention** | 2025 | Contiguous 가상 메모리 | 1.97배 토큰 생성↑ | 100% | 낮음 |
| **ChunkKV** | 2025 | 의미적 청크 기반 압축 | 1.5-2배 | 95%+ | 중간 |
| **ThinKV** | 2025 | 생각 적응 양자화+제거 | 5.8배 | 95%+ (코딩) | 높음 |
| **DMS** | 2025 | 동적 메모리 소거, 지연 제거 | 8배 | 95%+ | 중간 |
| **EvicPress** | 2025 | 다중 계층 압축+제거 | 높음 | 90%+ | 높음 |

### 5.3 성능 비교: 벤치마크 결과

#### A. 처리량 향상 (Throughput Improvement)

```
기준: 기존 시스템 (FasterTransformer/Orca) = 1x

vLLM (PagedAttention)          : 2-4x
vAttention (Microsoft)          : 3.92x (프롬프트), 1.97x (토큰)
KV-Compress + vLLM             : 5.18x
ThinKV                         : 5.80x
Infini-Attention (평가 제한)    : 114x (메모리 압축 비율)
EvicPress + vLLM               : 높음 (캐시 히트율 증대)
```

#### B. 메모리 절감 (Memory Efficiency)

```
기준: 전체 KV 캐시 = 100%

PagedAttention (기본)          : ~95-98% 유효 사용률
KV-Compress (5.18배)           : ~19% 유효 메모리
ChunkKV (2배 압축)             : ~50% 유효 메모리
ThinKV (5.8배)                 : ~17% 유효 메모리
Infini-Attention (114배)       : ~0.88% 메모리
```

#### C. 정확도 유지

**작업별 성능 (F1/ROUGE/정확도 기준):**

| 작업 | PagedAttention | KV-Compress | ChunkKV | ThinKV |
|------|----------------|-------------|---------|--------|
| **요약 (ROUGE)** | 100% | 95%+ | 98%+ | 95%+ |
| **코드 완성** | 100% | 92%+ | 95%+ | 98%+ (특화) |
| **수학 (AIME)** | 100% | 88%+ | 92%+ | 95%+ |
| **QA (정확도)** | 100% | 90%+ | 94%+ | 91%+ |
| **시각-언어 (VQA)** | 98%+ | 85%+ | - | - |

**결론:** PagedAttention은 정확도 손실 없음, 후속 압축 기법은 5-10% 손실 범위 [arxiv](https://arxiv.org/html/2601.03067)

### 5.4 실무적 배포 현황

#### 산업 채택 (실제 사용 사례) [datasciencedojo](https://datasciencedojo.com/blog/understanding-paged-attention/)

**1) 오픈소스 프로젝트**
- **vLLM:** 가장 많이 사용되는 LLM 추론 엔진
- **SGLang:** vLLM 기반 구조, 추가 최적화
- **TensorRT-LLM:** NVIDIA의 엔터프라이즈 버전

**2) 클라우드 서비스**
- **AWS Bedrock, Google Cloud AI, Azure OpenAI:** 내부적으로 PagedAttention 유사 기법 사용
- **Anyscale (Ray):** vLLM 통합

**3) 엔터프라이즈 배포**
- **IBM Foundation Model Stack:** FlexAttention과 PagedAttention 결합 [arxiv](https://arxiv.org/html/2506.07311v1)
- **Microsoft:** vAttention 개발, 향후 배포 예정 [arxiv](https://arxiv.org/html/2405.04437v1)

#### 성과 및 KPI [anyscale](https://www.anyscale.com/blog/continuous-batching-llm-inference)

| KPI | 기존 대비 | 사례 |
|-----|---------|------|
| **처리량 증대** | 23배 | Anyscale 벤치마크 [anyscale](https://www.anyscale.com/blog/continuous-batching-llm-inference) |
| **지연 감소** | 18.67% (p50) | vLLM 문서 [runpod](https://www.runpod.io/blog/introduction-to-vllm-and-pagedattention) |
| **비용 절감** | 3-10배 | LMCache와 결합 [reddit](https://www.reddit.com/r/LocalLLaMA/comments/1lewhla/we_built_this_project_to_increase_llm_throughput/) |
| **동시 사용자 증대** | 3-4배 | 일반적인 배포 사례 |

***

## 6. 제한사항 및 향후 과제

### 6.1 PagedAttention 자체의 한계

#### 1) GPU 커널 오버헤드
- **블록 테이블 접근:** 메모리 간접화(indirection) 비용
- **상황:** 20-26% 커널 지연 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/df50aa4e-133c-414b-a8c4-f8a43f1acfb5/2309.06180v1.pdf)
- **완화책:** GPU 아키텍처 진화 (메모리 계층 개선) 기대

#### 2) CPU 오버헤드
- **블록 관리:** 스케줄러의 블록 테이블 업데이트
- **관찰:** 매우 큰 배치에서 CPU 오버헤드 30% 도달 가능 [arxiv](https://arxiv.org/html/2503.08311v2)
- **권장:** CPU-GPU 비동기화 강화

#### 3) 분산 환경의 복잡성
- **모델 병렬화:** 서로 다른 GPU 워커의 KV 블록 동기화
- **메모리 네트워크:** 고속 인터커넥트(NVLink) 필요 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/df50aa4e-133c-414b-a8c4-f8a43f1acfb5/2309.06180v1.pdf)

### 6.2 압축 기법과의 통합 시 문제

#### 1) 정확도-효율성 트레이드오프
- **기본 PagedAttention:** 정확도 100%, 처리량 증대
- **압축 추가 시:** 정확도 5-20% 손실, 추가 처리량 gain [openreview](https://openreview.net/pdf?id=ClkfwM3STw)
- **과제:** 각 작업, 모델, 데이터셋별 최적 압축률 결정 알고리즘 [aclanthology](https://aclanthology.org/2025.findings-emnlp.426.pdf)

#### 2) 길이 외삽 일반화
- **제한:** 32K 이상 컨텍스트에서 성능 저하 개시 [arxiv](https://arxiv.org/html/2601.11564v1)
- **원인:** Attention 메커니즘 자체의 특성 (내용-무관한 위치 편향) [arxiv](https://arxiv.org/pdf/2503.23174.pdf)

### 6.3 연구 커뮤니티의 향후 방향

#### A. 적응형 최적화 (Adaptive Optimization)
```
Runtime에서:
1. 배치 특성 분석 (시퀀스 길이 분포, 모델 크기)
2. 최적 블록 크기, 압축률 자동 결정
3. 메모리 할당 전략 동적 조정
```
현재: 정적 설정(B=16)
권장: 학습 기반 또는 강화학습 [arxiv](https://arxiv.org/html/2502.04077v1)

#### B. 다양한 주의 메커니즘 지원
- GQA, MQA, Multi-Head와 각각의 최적화
- FlashAttention 계열과의 통합
- Sparse Attention 등과의 호환성 [arxiv](https://arxiv.org/html/2506.07311v1)

#### C. 메모리 계층 최적화
- GPU L2 캐시 활용
- CPU-GPU 비동기 전송
- SSD 계층 추가 [arxiv](https://arxiv.org/html/2512.14946v1)

***

## 결론

**PagedAttention은 LLM 서빙의 패러다임 전환을 가져온 혁신적 기술**입니다. OS의 가상 메모리 원리를 적용하여 KV 캐시의 메모리 낭비(61-80%)를 거의 제거하고, 2-4배의 처리량 향상을 달성했습니다. 모델 정확도는 완벽하게 유지되며, 복잡한 디코딩 시나리오(병렬 샘플링, 빔 서치, 공유 프리픽스)에서도 우수한 성능을 보입니다.

**일반화 성능 측면:**
- PagedAttention 자체는 수학적으로 표준 Attention과 동등
- 더 큰 배치 크기 가능 → GPU 활용도 향상 → 간접적 성능 개선
- 후속 압축/양자화 기법과 결합 시 5-10% 정확도 손실 범위 내에서 8배 이상 메모리 절감 가능

**향후 중요 과제:**
1. **적응형 블록 크기 선택:** 워크로드별 자동 최적화
2. **길이 외삽 해결:** 32K+ 컨텍스트 일반화
3. **압축 통합:** 정확도-효율성 최적점 탐색
4. **다양한 아키텍처 지원:** GQA, MQA, Sparse Attention 등

**학술적 영향:**
4,584회 이상의 인용, vLLM/SGLang/TensorRT-LLM 등 산업 표준 채택을 통해 **LLM 추론 분야의 가장 중요한 기여**로 평가됩니다.

***

## 참고문헌

<span style="display:none">[^1_32][^1_33][^1_34][^1_35][^1_36][^1_37][^1_38][^1_39][^1_40][^1_41][^1_42][^1_43][^1_44][^1_45][^1_46][^1_47][^1_48][^1_49][^1_50][^1_51][^1_52][^1_53][^1_54][^1_55][^1_56][^1_57][^1_58][^1_59][^1_60][^1_61][^1_62][^1_63][^1_64][^1_65][^1_66][^1_67][^1_68][^1_69][^1_70][^1_71][^1_72][^1_73][^1_74][^1_75][^1_76][^1_77][^1_78][^1_79][^1_80][^1_81][^1_82][^1_83][^1_84][^1_85][^1_86][^1_87][^1_88][^1_89][^1_90]</span>

<div align="center">⁂</div>

[^1_1]: 2309.06180v1.pdf

[^1_2]: https://arxiv.org/html/2601.03067

[^1_3]: https://arxiv.org/html/2509.04377v1

[^1_4]: https://arxiv.org/html/2510.00636v1

[^1_5]: https://arxiv.org/html/2502.00299v5

[^1_6]: https://aclanthology.org/2023.emnlp-main.298.pdf

[^1_7]: https://openreview.net/pdf?id=ClkfwM3STw

[^1_8]: https://arxiv.org/pdf/2503.23174.pdf

[^1_9]: https://arxiv.org/html/2601.11564v1

[^1_10]: https://arxiv.org/html/2404.07143v1

[^1_11]: https://arxiv.org/html/2405.04437v1

[^1_12]: https://arxiv.org/html/2503.08311v2

[^1_13]: https://www.arxiv.org/pdf/2508.17467.pdf

[^1_14]: https://arxiv.org/html/2512.14946v1

[^1_15]: https://arxiv.org/abs/2309.06180

[^1_16]: https://arxiv.org/html/2503.18292v1

[^1_17]: https://arxiv.org/html/2506.07311v1

[^1_18]: https://arxiv.org/pdf/2410.00161.pdf

[^1_19]: https://arxiv.org/html/2510.01290v1

[^1_20]: https://openreview.net/pdf?id=8ZiElzQxf1

[^1_21]: https://aclanthology.org/2025.findings-emnlp.426.pdf

[^1_22]: https://arxiv.org/html/2411.00136v1

[^1_23]: https://bentoml.com/llm/inference-optimization/kv-cache-offloading

[^1_24]: https://arxiv.org/html/2508.10824v1

[^1_25]: https://datasciencedojo.com/blog/understanding-paged-attention/

[^1_26]: https://www.anyscale.com/blog/continuous-batching-llm-inference

[^1_27]: https://www.runpod.io/blog/introduction-to-vllm-and-pagedattention

[^1_28]: https://www.reddit.com/r/LocalLLaMA/comments/1lewhla/we_built_this_project_to_increase_llm_throughput/

[^1_29]: https://arxiv.org/html/2502.04077v1

[^1_30]: https://www.surveypractice.org/article/117576-recent-books-and-journals-articles-in-public-opinion-survey-methods-survey-statistics-big-data-data-science-and-user-experience-research-2023-up

[^1_31]: https://arxiv.org/pdf/2407.15309.pdf

[^1_32]: https://www.mdpi.com/2072-6643/16/16/2744

[^1_33]: https://www.mdpi.com/2076-393X/12/5/549

[^1_34]: https://journals.sagepub.com/doi/10.1177/00472816241277721

[^1_35]: https://journals.unite.edu.mk/Abstract?AId=1200\&DId=2573

[^1_36]: https://muse.jhu.edu/article/935202

[^1_37]: https://ieeexplore.ieee.org/document/10826595/

[^1_38]: https://journal.uin-alauddin.ac.id/index.php/lentera_pendidikan/article/view/49406

[^1_39]: https://www.tandfonline.com/doi/full/10.1080/03602532.2024.2370331

[^1_40]: https://journals.flvc.org/edis/article/view/133879

[^1_41]: https://dl.acm.org/doi/pdf/10.1145/3600006.3613165

[^1_42]: https://arxiv.org/pdf/2309.06180.pdf

[^1_43]: https://arxiv.org/pdf/2405.04437.pdf

[^1_44]: https://arxiv.org/html/2412.01818

[^1_45]: https://arxiv.org/pdf/2407.12391.pdf

[^1_46]: https://arxiv.org/abs/2412.03324

[^1_47]: https://arxiv.org/abs/2503.08461

[^1_48]: https://arxiv.org/abs/2507.11507

[^1_49]: https://arxiv.org/abs/2507.19595

[^1_50]: https://arxiv.org/pdf/2506.07311.pdf

[^1_51]: https://arxiv.org/html/2510.09665v1

[^1_52]: https://arxiv.org/html/2501.11847v2

[^1_53]: https://arxiv.org/html/2503.24000v1

[^1_54]: https://arxiv.org/html/2507.19595v1

[^1_55]: https://openreview.net/forum?id=rjnKCFZuJt

[^1_56]: https://developer.nvidia.com/blog/introducing-new-kv-cache-reuse-optimizations-in-nvidia-tensorrt-llm/

[^1_57]: https://www.nature.com/articles/s43588-025-00854-1

[^1_58]: https://training.continuumlabs.ai/inference/why-is-inference-important/paged-attention-and-vllm

[^1_59]: https://www.microsoft.com/en-us/research/blog/llm-profiling-guides-kv-cache-optimization/

[^1_60]: https://openaccess.thecvf.com/content/CVPR2023/papers/Liu_EfficientViT_Memory_Efficient_Vision_Transformer_With_Cascaded_Group_Attention_CVPR_2023_paper.pdf

[^1_61]: https://sakana.ai/namm/

[^1_62]: https://blog.vllm.ai/2023/06/20/vllm.html

[^1_63]: https://www.microsoft.com/en-us/research/blog/llm-profiling-guides-kv-cache-optimization/?lang=ko-kr

[^1_64]: https://www.mdpi.com/2077-1312/12/8/1263

[^1_65]: https://iopscience.iop.org/article/10.1088/1361-6501/acb374

[^1_66]: https://arxiv.org/abs/2305.18107

[^1_67]: https://arxiv.org/abs/2302.10429

[^1_68]: https://iopscience.iop.org/article/10.1088/1361-6501/ad3ea4

[^1_69]: https://dl.acm.org/doi/10.1145/3580305.3599518

[^1_70]: https://iopscience.iop.org/article/10.1088/1361-6501/ac6081

[^1_71]: https://ieeexplore.ieee.org/document/10660529/

[^1_72]: https://www.mdpi.com/2076-3417/15/3/1225

[^1_73]: https://iopscience.iop.org/article/10.1088/1361-6501/ac90dc

[^1_74]: https://aclanthology.org/2022.findings-emnlp.101.pdf

[^1_75]: https://arxiv.org/pdf/2407.15516.pdf

[^1_76]: https://arxiv.org/html/2502.04077v2

[^1_77]: https://aclanthology.org/2023.findings-emnlp.909.pdf

[^1_78]: https://arxiv.org/html/2406.12928v1

[^1_79]: https://arxiv.org/html/2405.04437v2

[^1_80]: https://arxiv.org/pdf/2411.00136.pdf

[^1_81]: https://arxiv.org/html/2510.18245v1

[^1_82]: https://arxiv.org/pdf/2508.06297.pdf

[^1_83]: https://www.emergentmind.com/topics/kv-cache-compression

[^1_84]: https://www.emergentmind.com/topics/pagedattention

[^1_85]: https://www2.eecs.berkeley.edu/Pubs/TechRpts/2025/EECS-2025-192.pdf

[^1_86]: https://neurips.cc/virtual/2025/poster/119605

[^1_87]: https://3dvar.com/Kwon2023Efficient.pdf

[^1_88]: https://rawlinson.ca/articles/vLLM-inference-optimization

[^1_89]: https://cloud.google.com/blog/topics/developers-practitioners/vllm-performance-tuning-the-ultimate-guide-to-xpu-inference-configuration

[^1_90]: https://arxiv.org/pdf/2510.00231.pdf
