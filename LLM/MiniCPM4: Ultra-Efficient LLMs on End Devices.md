
# MiniCPM4: Ultra-Efficient LLMs on End Devices 

> **참고 논문 및 출처**
> - **주 논문**: MiniCPM Team (Chaojun Xiao et al.), "MiniCPM4: Ultra-Efficient LLMs on End Devices", arXiv:2506.07900, June 2025. ([arxiv.org/abs/2506.07900](https://arxiv.org/abs/2506.07900))
> - MarkTechPost (June 2025): [marktechpost.com](https://www.marktechpost.com/2025/06/16/openbmb-releases-minicpm4/)
> - Neurohive (June 2025): [neurohive.io](https://neurohive.io/en/state-of-the-art/minicpm4-open-local-model-achieves-qwen3-8b-performance-with-7x-inference-acceleration/)
> - EmergentMind: [emergentmind.com/topics/minicpm4](https://www.emergentmind.com/topics/minicpm4)
> - OpenBMB GitHub: [github.com/OpenBMB/MiniCPM](https://github.com/OpenBMB/MiniCPM)
> - HuggingFace Model Card: [huggingface.co/openbmb/MiniCPM4-8B](https://huggingface.co/openbmb/MiniCPM4-8B)
> - ADS Abstract: [ui.adsabs.harvard.edu](https://ui.adsabs.harvard.edu/abs/2025arXiv250607900M/abstract)

---

## 1. 📌 핵심 주장 및 주요 기여 (요약)

MiniCPM4는 엔드-사이드(end-side) 기기에서의 효율적인 배포를 목적으로 설계된 LLM으로, **모델 아키텍처, 학습 데이터, 학습 알고리즘, 추론 시스템**의 네 가지 핵심 차원에서 체계적인 혁신을 통해 효율성을 달성합니다.

MiniCPM4는 Qwen3-8B 대비 22%의 학습 데이터만으로 동등한 성능을 달성하면서, 엔드-사이드 기기에서 128K 길이 문서 처리 속도를 7배 향상시킵니다.

### 🔑 주요 기여 4가지

| 차원 | 핵심 기여 |
|------|-----------|
| 모델 아키텍처 | InfLLM v2 (학습 가능한 희소 어텐션) |
| 학습 데이터 | UltraClean + UltraChat v2 |
| 학습 알고리즘 | ModelTunnel v2 + BitCPM + Chunk-wise Rollout |
| 추론 시스템 | CPM.cu (양자화 + 투기적 샘플링) |

다양한 온디바이스 요구사항을 충족하기 위해 MiniCPM4는 **0.5B**와 **8B** 두 가지 파라미터 버전으로 제공됩니다.

---

## 2. 🔍 상세 분석

### 2-1. 해결하고자 하는 문제

모델의 규모가 계속 확대됨에 따라 계산 자원에 대한 요구가 기하급수적으로 증가하여, 이러한 모델들이 주로 클라우드 서버에 배포되는 상황이 발생합니다.

LLM의 장문(long-context) 처리와 심층 추론 능력에 대한 수요가 증가하면서, 자기-어텐션(self-attention) 메커니즘의 계산 및 메모리 요구량이 엔드-사이드 기기에서의 효율적인 처리를 위한 중요한 도전 과제가 되었습니다.

표준 어텐션 메커니즘의 경우, 128K 토큰 처리 시 160억 건 이상의 관계 계산이 필요합니다.

---

### 2-2. 제안하는 방법 (수식 포함)

#### ① InfLLM v2 — 학습 가능한 희소 어텐션

논문은 희소 어텐션 아키텍처를 제안하여 효율적인 장문 컨텍스트 처리를 가능하게 합니다. 이 구조는 기존의 동적 희소 어텐션 아키텍처 InfLLM을 기반으로 InfLLM v2를 도입합니다.

InfLLM v2는 KV 캐시를 블록 $B_j$로 분할하며, 각 쿼리 토큰 $q_i$에 대해 관련성 점수를 기반으로 관련 블록의 부분 집합을 동적으로 선택합니다.

희소 어텐션의 핵심 수식은 다음과 같이 표현할 수 있습니다:

$$\text{SparseAttention}(Q, K, V) = \text{softmax}\left(\frac{Q K_{\mathcal{S}}^T}{\sqrt{d_k}}\right) V_{\mathcal{S}}$$

여기서 $\mathcal{S}$는 선택된 블록의 인덱스 집합이며, 전체 KV 캐시가 아닌 Top- $k$ 관련 블록만 사용합니다.

각 쿼리 그룹은 어텐션 계산을 위한 key-value 블록의 일부를 선택하며, 초기 토큰과 슬라이딩 윈도우 내의 지역 토큰은 항상 선택됩니다.

InfLLM v2는 81%의 어텐션 희소성(attention sparsity)으로 전체 어텐션 메커니즘과 동등한 장문 컨텍스트 처리 능력을 달성합니다.

InfLLM v2는 128K 장문 처리 시 각 토큰이 전체 토큰의 5% 미만의 토큰과만 관련성을 계산하면, 장문에 대한 계산 오버헤드를 크게 줄입니다.

**블록 선택 메커니즘 (Contextual Block Selection)**:

$$\text{score}(q_i, B_j) = \frac{1}{|B_j|} \sum_{k \in B_j} q_i \cdot k_k^T$$

$$\mathcal{S}_i = \text{Top-}K\{j : \text{score}(q_i, B_j)\} \cup \mathcal{S}_{\text{initial}} \cup \mathcal{S}_{\text{local}}$$

시맨틱 커널(Semantic Kernel)은 겹치는 텍스트 세그먼트(커널당 32토큰, 스트라이드 16)를 사용하며, 동적 블록 선택 시 각 쿼리 토큰은 상위 6개의 관련 블록만을 처리합니다.

---

#### ② UltraClean — 고품질 사전학습 데이터 필터링

UltraClean은 검증 비용을 1,200 GPU-시간에서 110 GPU-시간으로 줄이는 필터링 전략이며, 사전학습된 1B 모델을 활용해 2단계 어닐링 프로세스로 데이터 품질을 평가합니다. FastText 분류기는 15조 토큰을 1,000 CPU-시간에 처리하며 이는 전통적인 방법의 6,000 GPU-시간 대비 획기적인 개선입니다.

UltraClean은 손실을 개선하는 "포지티브 시드(positive seed)" 감지와 FastText 분류기를 활용하여 사전학습 데이터를 선택하며, UltraChat v2는 멀티턴·다중 능력 SFT(지도학습 파인튜닝) 데이터를 제공합니다.

UltraFineWeb은 FineWeb 대비 영어 3.61%, 중국어 1.98% 성능 향상을 보였으며, ARC-C 35.67%, ARC-E 70.62%, MMLU 32.24%를 달성하여 기존 데이터셋을 초과했습니다.

---

#### ③ ModelTunnel v2 — 효율적 사전학습 전략 탐색

ModelTunnel v2는 스케일링 법칙과 ScalingBench 지표에 기반한 자동화된 하이퍼파라미터 탐색 시스템입니다. ScalingBench는 손실과 다운스트림 정확도 간의 시그모이드 매핑을 모델링하여 신뢰할 수 있는 구성 선택을 가능하게 합니다. 최대 업데이트 파라미터화(μP) 패러다임은 소규모 프록시 모델에서 최적화된 하이퍼파라미터를 풀-스케일 모델로 전이할 수 있게 해줍니다.

스케일링 법칙 기반 성능 예측 수식:

$$\mathcal{L}(N, D) = \frac{A}{N^{\alpha}} + \frac{B}{D^{\beta}} + \mathcal{L}_{\infty}$$

여기서 $N$은 파라미터 수, $D$는 학습 토큰 수, $\alpha, \beta$는 스케일링 지수입니다.

---

#### ④ BitCPM — 3진(Ternary) 양자화 LLM

BitCPM4는 3진 양자화 인식 변형으로, 모델 가중치를 $\{-1, 0, +1\}$로 전환하는 지속적 QAT(Quantization-Aware Training)을 2단계에 걸쳐 수행합니다.

BitCPM은 배포 시 계산 요구량을 줄이고, 토큰 예산을 크게 늘리지 않으면서 온디바이스 운용을 가능하게 합니다.

양자화 수식:

$$W_q = \text{round}\left(\text{clip}\left(\frac{W}{\Delta}, -1, 1\right)\right), \quad \Delta = \frac{\max(|W|)}{1}$$

BitCPM4는 극도로 제한된 하드웨어에 적합한 3진 LLM을 구현합니다.

---

#### ⑤ CPM.cu — 통합 추론 시스템

CPM.cu는 블록 단위 희소 어텐션(InfLLM v2)과 P-GPTQ(Prefix-Aware GPTQ) 양자화를 통합하며, FR-Spec 투기적 디코딩은 토큰 빈도의 롱테일 특성을 활용하여 어휘의 상위 25%만으로 소프트맥스 계산을 수행합니다.

**Chunk-wise Rollout (강화학습)**:

강화학습 알고리즘은 청크 단위 롤아웃(chunk-wise rollout)과 클리핑/KL 목적을 사용하는 그룹 RPO를 통해 효율적인 샘플링을 달성합니다.

RL 파인튜닝은 매우 긴 궤적을 길이 제한 청크로 분할하여 병렬화하며, 불완전한 롤아웃의 로그 확률은 캐시되어 이후 집계됩니다. 이는 GPU 부하를 균형 있게 하고 활용 저하를 방지합니다.

---

### 2-3. 모델 구조

MiniCPM4는 대부분의 오픈소스 LLM과 마찬가지로 Transformer를 기본 아키텍처로 채택합니다.

- **기본 아키텍처**: Transformer (Vaswani et al., 2017) 기반
- **어텐션 메커니즘**: InfLLM v2 (희소 어텐션, prefilling + decoding 모두 가속)
- **학습 목표**: Multi-Token Prediction (MTP) + FP8 혼합 정밀도 학습

DeepSeek(2024)에서 영감을 받아 멀티-토큰 예측 학습 목표(Gloeckle et al., 2024)와 FP-8 혼합 정밀도 학습 프레임워크를 구현했습니다. 멀티-토큰 예측은 보다 집중적인 지도 신호를 도입하고 투기적 샘플링에서 더 높은 수용 길이를 달성하도록 합니다.

| 구분 | MiniCPM4-0.5B | MiniCPM4-8B |
|------|--------------|-------------|
| 파라미터 수 | 0.5B | 8B |
| 학습 토큰 수 | 1T | 8T |
| 컨텍스트 길이 | 32K → 128K (YaRN) | 32K → 128K (YaRN) |
| 어텐션 | InfLLM v2 | InfLLM v2 |

MiniCPM4는 32K 장문 텍스트로 사전학습되며, YaRN을 통해 길이 확장을 달성합니다.

---

### 2-4. 성능 향상

MiniCPM4-8B는 학습 토큰을 8조 대 36조로 4.5배 줄이면서도 Qwen3-8B와 동등한 성능(81.13 vs 80.55)을 달성합니다.

MiniCPM4의 0.5B 및 8B 파라미터 변형 모두 MMLU, CMMLU, CEval, BBH, GSM8K, MATH500, MBPP, HumanEval 등 다양한 벤치마크에서 유사 크기의 오픈소스 모델을 능가합니다.

MiniCPM4-0.5B는 Qwen3-0.6B 대비 평균 점수 52.99 vs 44.93을 기록했습니다.

**추론 속도**:

Jetson AGX Orin과 RTX 4090 두 가지 대표적인 엔드-사이드 칩에서 MiniCPM4는 장문 텍스트 처리 작업에서 동일 크기 모델보다 빠른 처리 속도를 보이며, 텍스트 길이가 증가할수록 속도 이점이 더욱 두드러집니다. Jetson AGX Orin에서 Qwen3-8B 대비 약 7배의 디코딩 속도 향상을 달성합니다.

InfLLM v2는 블록 수준 어텐션을 통해 어텐션 계산 비용을 60% 절감합니다.

---

### 2-5. 한계점

논문에서 명시적으로 언급된 한계점과 연구 커뮤니티에서 제기된 사항은 다음과 같습니다:

1. **희소 어텐션의 태생적 한계**: 어텐션이 희소할 경우 일부 복잡한 추론 과제에서 전체 어텐션 대비 성능 저하가 발생할 수 있습니다.

2. **멀티모달 미지원**: MiniCPM4는 순수 텍스트 모델로, 멀티모달(이미지·오디오) 처리는 별도의 MiniCPM-V 시리즈에서 다룹니다.

3. **3진 양자화의 정밀도 손실**: 극도로 자원 제한된 기기를 위해 MiniCPM4를 3진 버전 BitCPM4로 적응시키며 유망한 결과를 보이나, 정밀도와 효율성 간의 트레이드오프는 여전히 존재합니다.

4. **도메인 특화 평가 부족**: 다양한 일반 벤치마크에서 우수한 성능을 보이지만, 의료·법률 등 전문 도메인에서의 평가는 제한적입니다.

5. **하드웨어 의존성**: CPM.cu는 NVIDIA CUDA 기반으로 설계되어, 다양한 엣지 칩(ARM, NPU 등) 범용 지원에는 추가 작업이 필요합니다.

---

## 3. 🌐 모델의 일반화 성능 향상 가능성

### 3-1. 장문 컨텍스트 일반화

희소 어텐션은 대규모 컨텍스트 윈도우에서 강건한 외삽(extrapolation)을 제공합니다. MiniCPM4는 5%의 활성 어텐션 희소성만으로 128K 토큰 Needle-in-a-Haystack 작업에서 100% 정확도를 달성하며, 이는 탁월한 장거리 의존성 추적 능력을 시사합니다.

MiniCPM4는 기본 32K에서 외삽하여 128K 토큰 처리 시 RULER-NIAH 테스트에서 100% 정확도를 달성합니다.

### 3-2. 데이터 효율성을 통한 일반화

UltraChat v2는 지식 집약적, 추론 집약적, 명령어 따르기, 장문 컨텍스트, 도구 사용 시나리오를 포함하는 지도 파인튜닝 데이터셋을 제공하여 다양한 도메인에 대한 일반화 능력을 지원합니다.

### 3-3. 하이브리드 추론 모드를 통한 일반화

MiniCPM4.1이라는 하이브리드 추론 모델을 구축하여 심층 추론 모드와 비추론 모드 모두에서 사용할 수 있습니다. 이는 다양한 태스크 복잡도에 대한 유연한 일반화를 가능하게 합니다.

### 3-4. 다양한 응용으로의 일반화

MiniCPM4는 신뢰할 수 있는 서베이 논문 생성과 모델 컨텍스트 프로토콜을 통한 도구 사용을 포함한 다양한 응용 분야를 성공적으로 지원하며, 광범위한 활용 가능성을 명확히 보여줍니다.

MiniCPM4-Survey는 MiniCPM4-8B를 기반으로 구축되어, 훨씬 더 큰 모델과 경쟁적인 성능을 유지하면서 신뢰할 수 있는 장문 서베이 논문을 생성할 수 있습니다.

### 3-5. 일반화 성능의 수학적 기반

MiniCPM4의 일반화 성능 향상은 다음과 같은 수학적 메커니즘에 기반합니다:

**μP (Maximal Update Parameterization)를 통한 학습 안정화:**

$$\eta_l \propto \frac{1}{\sqrt{d_{\text{model}}}} \cdot \eta_{\text{base}}$$

이를 통해 소규모 프록시 모델에서 탐색된 하이퍼파라미터가 대규모 모델로 안정적으로 전이됩니다.

**ScalingBench의 시그모이드 매핑:**

$$\text{Acc}(\mathcal{L}) = \frac{1}{1 + e^{-k(\mathcal{L} - \mathcal{L}_0)}}$$

여기서 $\mathcal{L}$은 검증 손실, $\mathcal{L}_0$는 전이점, $k$는 기울기 파라미터입니다. 이 매핑은 손실에서 벤치마크 정확도로의 신뢰할 수 있는 예측을 가능하게 합니다.

---

## 4. 📊 2020년 이후 관련 최신 연구 비교 분석

| 모델 | 연도 | 파라미터 | 핵심 효율화 기법 | 학습 토큰 | 엣지 최적화 |
|------|------|---------|----------------|---------|------------|
| GPT-3 (Brown et al.) | 2020 | 175B | Dense Attention | ~300B | ❌ |
| LLaMA-2 (Meta) | 2023 | 7B–70B | GQA, RoPE | 2T | ❌ |
| Phi-2 (Microsoft) | 2023 | 2.7B | 고품질 데이터 | 1.4T | △ |
| Qwen2.5 (Alibaba) | 2024 | 0.5B–72B | GQA, Long-context | ~18T | △ |
| Qwen3-8B (Alibaba) | 2025 | 8B | Hybrid Thinking | 36T | △ |
| **MiniCPM4-8B** | **2025** | **8B** | **Sparse Attention + QAT + SpecDec** | **8T** | **✅** |

멀티-토큰 예측은 더 집중적인 지도 신호를 도입하고 투기적 샘플링에서 더 높은 수용 길이를 달성하게 하며, FP-8 혼합 정밀도 학습은 InfLLM v2가 전체 어텐션 메커니즘과 동등한 장문 컨텍스트 처리 능력을 효율적인 지식 학습을 통해 달성하게 합니다.

**MiniCPM4 vs Qwen3-8B 핵심 비교:**
MiniCPM4는 Qwen3-8B 대비 학습 토큰 수 8T vs 36T로 대폭 절감하면서, InfLLM v2가 블록 수준 어텐션을 통해 어텐션 계산 비용을 60% 줄였습니다.

---

## 5. 🔮 향후 연구에 미치는 영향 및 고려 사항

### 5-1. 향후 연구에 미치는 영향

1. **엣지 AI 패러다임 전환**: MiniCPM4는 대표적인 엔드-사이드 칩에서 5배 이상의 생성 속도 가속을 달성하여, 클라우드 의존도를 줄이고 프라이버시 보호 온디바이스 AI를 가능하게 하는 새로운 연구 방향을 제시합니다.

2. **학습 가능한 희소 어텐션의 주류화**: MiniCPM4.1-8B는 학습 가능한 희소 어텐션을 탑재한 최초의 오픈소스 추론 LLM으로, 향후 대규모 모델에서의 희소 어텐션 설계 연구를 가속화할 것입니다.

3. **데이터 효율적 학습**: MiniCPM4는 Qwen3-8B 대비 22%의 데이터만으로 동등한 성능을 달성함으로써, 데이터 중심(data-centric) AI 연구의 중요성을 재조명합니다.

4. **하이브리드 추론 모드**: 심층 추론 모드와 비추론 모드를 모두 사용할 수 있는 하이브리드 추론 모델 MiniCPM4.1의 구축은 계산 자원 적응형 추론 시스템 설계에 새로운 방향을 제시합니다.

### 5-2. 앞으로 연구 시 고려할 점

**① 희소 어텐션의 이론적 보장 강화**
- 현재 InfLLM v2의 블록 선택 방법은 경험적으로 검증되었으나, 정보 손실에 대한 이론적 상한(upper bound) 분석이 필요합니다.
- 어텐션 희소성 $s$와 태스크 성능 사이의 정량적 관계 규명:

$$\text{Perf}(s) \geq \text{Perf}(\text{Full}) - \epsilon(s), \quad \epsilon(s) \to 0 \text{ as } s \to 0$$

**② 도메인 특화 일반화 연구**
- LLM은 옵션 길이, 문제 유형, 명사 대체 등 다양한 변형에서 일반화에 어려움을 보입니다. 예를 들어 Qwen 2.5 1.5B의 MMLU 점수는 답변 선택지 길이 변경만으로 89에서 36으로 떨어집니다. MiniCPM4도 이러한 포맷 민감도를 체계적으로 평가할 필요가 있습니다.

**③ 3진 양자화의 성능-효율 트레이드오프 연구**
- BitCPM의 $\{-1, 0, +1\}$ 가중치 양자화가 다양한 벤치마크에서 어느 정도의 성능 손실을 야기하는지 체계적 분석이 필요합니다.

**④ 멀티모달 확장**
- 현재 MiniCPM4는 텍스트 전용이므로, InfLLM v2의 희소 어텐션을 비전-언어 모델(VLM)로 확장하는 연구가 필요합니다.

**⑤ 강화학습 안정성**
- 강화학습 알고리즘에서 청크 단위 롤아웃과 그룹 RPO를 활용하지만, 장기 추론 체인에서의 보상 해킹(reward hacking) 및 분포 드리프트(distribution drift) 문제에 대한 추가 연구가 필요합니다.

**⑥ 하드웨어 포터빌리티**
- CPM.cu의 CUDA 최적화를 ARM NPU, Apple Silicon, Qualcomm Hexagon 등 다양한 엣지 칩으로 확장하는 연구가 필요합니다.

**⑦ 연속 학습(Continual Learning)과의 결합**
- 엣지 기기에서 모델을 지속적으로 업데이트할 때 희소 어텐션 구조의 안정성 유지 방법 연구가 필요합니다.

---

> ⚠️ **정확도 안내**: 본 답변은 공개된 arXiv 논문 초록, HuggingFace 모델 카드, 공식 GitHub, 기술 리뷰 사이트를 기반으로 작성되었습니다. 논문 내부의 일부 세부 수식(특히 InfLLM v2의 상세 블록 선택 수식)은 공개된 정보를 바탕으로 재구성되었으며, 전체 논문 PDF의 정확한 수식과 다소 차이가 있을 수 있습니다. 완전한 수식 검증을 위해 [arXiv 원문 PDF](https://arxiv.org/pdf/2506.07900)를 직접 참조하시기 바랍니다.
