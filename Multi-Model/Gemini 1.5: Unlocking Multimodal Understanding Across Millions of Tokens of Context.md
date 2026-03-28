# Gemini 1.5: Unlocking Multimodal Understanding Across Millions of Tokens of Context

---

## 1. 핵심 주장 및 주요 기여 (Executive Summary)

Gemini 1.5 제품군은 차세대 고효율(compute-efficient) 멀티모달 모델로, 수백만 토큰의 컨텍스트에서 세밀한 정보를 회상(recall)하고 추론(reasoning)하는 능력을 갖추고 있으며, 복수의 긴 문서, 수 시간의 비디오와 오디오를 처리할 수 있습니다.

**주요 기여 요약:**

| 기여 항목 | 내용 |
|---|---|
| **모델 제품군** | Gemini 1.5 Pro (업데이트 버전)와 Gemini 1.5 Flash (경량·고효율 버전) 두 가지 모델을 도입 |
| **컨텍스트 윈도우** | 최대 10M 토큰까지 near-perfect retrieval (>99%)과 next-token prediction 개선을 달성하며, Claude 2.1 (200k), GPT-4 Turbo (128k) 대비 세대적 도약(generational leap) |
| **멀티모달 SOTA** | 장문 문서 QA, 장문 비디오 QA, 장문 ASR에서 SOTA를 달성하고, 광범위한 벤치마크에서 Gemini 1.0 Ultra의 SOTA 성능을 동등하거나 능가 |
| **In-Context Learning** | 세계적으로 200명 미만의 화자를 가진 Kalamang 언어의 문법 매뉴얼이 주어졌을 때, 동일한 자료에서 학습한 인간과 유사한 수준으로 영어→Kalamang 번역을 수행 |
| **실용적 영향** | 10가지 직업 범주에서 전문가와 협업 시 26~75%의 시간 절감 효과를 달성 |

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

기존 LLM은 컨텍스트 윈도우가 제한적(예: GPT-4 Turbo 128k, Claude 200k)이어서 대규모 멀티모달 입력(장시간 비디오, 대량 문서, 긴 오디오)을 한 번에 처리하기 어려웠습니다.

장문 컨텍스트 능력 개선을 위한 기존 접근법은 새로운 아키텍처(Zaheer et al., 2020; Ainslie et al., 2023; Gu and Dao, 2023), 학습 후 수정(Press et al., 2021; Xiong et al., 2023), 검색 증강 모델(Guu et al., 2020; Izacard et al., 2022), 메모리 증강 모델(Wu et al., 2022; Bulatov et al., 2023) 등 여러 범주로 나뉘었습니다.

그러나 기존 평가들은 점점 더 발전하는 대형 멀티모달 모델의 능력을 따라잡지 못하며, 대부분 개별 모달리티에 초점을 맞추거나 짧은 컨텍스트 길이에 제한됩니다. 따라서 실제 환경의 장문 혼합 모달리티 사용 사례를 대변하는 벤치마크에 대한 요구가 커지고 있습니다.

### 2.2 제안하는 방법: Sparse Mixture-of-Experts (MoE) 아키텍처

#### 2.2.1 아키텍처 개요

Gemini 1.5는 새로운 Mixture-of-Experts 아키텍처와 학습·서빙 인프라의 주요 발전을 통합한 고성능 멀티모달 모델 제품군입니다.

MoE 모델은 학습된 라우팅 함수(learned routing function)를 사용하여 입력을 모델 파라미터의 부분집합으로 전달하여 처리합니다. 이러한 조건부 계산(conditional computation) 방식을 통해, 모델의 전체 파라미터 수를 늘리면서도 주어진 입력에 대해 활성화되는 파라미터 수는 일정하게 유지할 수 있습니다.

#### 2.2.2 수식적 표현

**MoE 게이팅 메커니즘** — Sparse MoE의 핵심 구성 요소인 **라우터(Gating Network)**는 다음과 같이 정의됩니다:

**1단계: 라우팅 로짓 계산**

입력 토큰 표현 $\mathbf{x} \in \mathbb{R}^{d}$에 대해, $N$명의 전문가(expert)에 대한 점수를 계산합니다:

$$\mathbf{h}(\mathbf{x}) = \mathbf{W}_g \cdot \mathbf{x}, \quad \mathbf{W}_g \in \mathbb{R}^{N \times d}$$

여기서 $\mathbf{W}_g$는 게이팅 네트워크의 학습 가능한 파라미터입니다.

**2단계: Noisy Top- $k$ Gating (Shazeer et al., 2017)**

학습 안정성과 부하 균형(load balancing)을 위해 노이즈를 추가한 뒤 Top- $k$ 선택을 수행합니다:

$$\tilde{\mathbf{h}}(\mathbf{x}) = \mathbf{h}(\mathbf{x}) + \text{StandardNormal}() \cdot \text{Softplus}(\mathbf{W}_{\text{noise}} \cdot \mathbf{x})$$

$$\text{TopK}(\tilde{\mathbf{h}}(\mathbf{x}), k)_i = \begin{cases} \tilde{h}_i(\mathbf{x}) & \text{if } \tilde{h}_i(\mathbf{x}) \text{ is in the top } k \text{ elements} \\ -\infty & \text{otherwise} \end{cases}$$

**3단계: 게이팅 확률 및 최종 출력**

$$G(\mathbf{x}) = \text{Softmax}(\text{TopK}(\tilde{\mathbf{h}}(\mathbf{x}), k))$$

$$\mathbf{y} = \sum_{i=1}^{N} G(\mathbf{x})_i \cdot E_i(\mathbf{x})$$

여기서 $E_i(\mathbf{x})$는 $i$번째 전문가 네트워크(일반적으로 FFN)의 출력이며, 상위 $k$개의 전문가만 실제로 활성화됩니다.

**4단계: 부하 균형 보조 손실(Auxiliary Load Balancing Loss)**

전문가 간 균등한 토큰 분배를 위해 아래 보조 손실이 추가됩니다:

$$\mathcal{L}_{\text{aux}} = \alpha \cdot N \sum_{i=1}^{N} f_i \cdot P_i$$

여기서 $f_i$는 전문가 $i$에 라우팅된 토큰의 비율, $P_i$는 전문가 $i$에 대한 평균 라우팅 확률, $\alpha$는 균형 계수입니다.

> **참고:** 모델 크기, 아키텍처 실험 세부사항, 또는 전문가 수에 대한 상세 정보는 공개되지 않았지만, 모델은 인컨텍스트 기억(memorization)과 일반화(generalization)에서 우수한 성과를 보입니다.

#### 2.2.3 모델 구조 상세

| 구성 요소 | 설명 |
|---|---|
| **기반 아키텍처** | Gemini 1.5 Pro는 Sparse Mixture-of-Experts (MoE) Transformer 아키텍처, Gemini 1.5 Flash는 Pro로부터 증류(distill)된 Dense Transformer 모델 |
| **학습 인프라** | Google의 TPUv4 가속기 4096칩 포드(pod) 다수에서, 여러 데이터 센터에 분산하여 학습 |
| **학습 데이터** | 웹 문서, 코드를 포함한 다양한 도메인의 데이터에서 소싱되었으며, 이미지·오디오·비디오 콘텐츠를 포함 |
| **학습 파이프라인** | 사전학습(pre-training), 미세조정(fine-tuning), 지시 조정(instruction tuning)의 다단계 학습 과정과 RLHF(인간 피드백 기반 강화학습)를 사용하여 인간 선호 및 윤리 표준에 모델 행동을 정렬 |
| **컨텍스트 처리 능력** | 최대 10M 토큰 입력을 성능 저하 없이 처리 가능하며, 이는 약 107시간의 오디오, "전쟁과 평화"(1440쪽) 전체의 10배 이상, Flax 코드베이스 전체(41,070 라인), 또는 초당 1프레임 기준 10.5시간의 비디오에 해당 |
| **멀티모달 입력** | 네이티브 멀티모달(natively multimodal) 모델로서, 서로 다른 모달리티의 데이터를 인터리빙(interleaving)하여 동일 입력 시퀀스에서 오디오·시각·텍스트·코드 입력을 혼합 지원 |

### 2.3 성능 향상

#### Needle-in-a-Haystack (NIAH) 결과

합성 "needle-in-a-haystack" 태스크에서 Gemini 1.5 Pro는 텍스트, 비디오, 오디오 등 모든 모달리티에서 수백만 토큰의 "haystack"에 대해 near-perfect (>99%)의 "needle" recall을 달성하며, 텍스트 모달리티에서 10M 토큰까지 확장해도 이 recall 성능을 유지합니다.

#### 실제 벤치마크 결과

장문 문서 질의응답이나 장문 비디오 질의응답과 같이, 컨텍스트의 여러 부분에서 검색 및 추론을 요구하는 현실적 멀티모달 장문 벤치마크에서도, 외부 검색 방법으로 증강된 경쟁 모델을 포함하여 모든 모달리티에서 Gemini 1.5 Pro가 최고 성능을 달성했습니다.

#### 핵심 역량(Core Capabilities) 비교

아키텍처, 데이터, 최적화 및 시스템에 걸친 거의 전체 모델 스택의 개선을 통해, Gemini 1.5 Pro는 Gemini 1.0 Ultra와 동등한 품질을 달성하면서도, 학습 연산량(training compute)을 현저히 줄이고 서빙 효율성도 크게 향상시켰습니다.

다국어 수학(MGSM) 데이터셋에서 Gemini 1.0 Ultra 대비 약 +10%의 상당한 성능 향상을 보였으며, 이러한 개선은 특정 자원 그룹에 국한되지 않고 다양한 자원 수준의 언어에서 균등하게 향상되었습니다.

### 2.4 한계

1. **아키텍처 세부사항 미공개:** 모델 크기, 전문가 수, 아키텍처 실험 등에 대한 상세 정보가 공개되지 않음
2. **평가 체계의 한계:** 기존 평가 체계는 빠르게 발전하는 대형 멀티모달 모델의 능력을 충분히 반영하지 못하며, 실제 환경의 장문 혼합 모달리티 사용 사례를 대변하는 벤치마크가 필요합니다. 특히 장문 혼합 모달리티 시퀀스에서의 추론 능력에 대한 정량적 평가가 핵심 과제입니다.
3. **안전성 리스크:** 장문 컨텍스트 이해 능력이 잠재적 이점을 강화하는 동시에, Gemini 1.0 기술 보고서에서 언급된 일부 위험을 악화시킬 수 있으며, 새로운 부정적 효과가 발생할 가능성이 있습니다.
4. **연산 비용 문제:** 수백만 토큰의 컨텍스트 처리에는 여전히 상당한 연산 자원이 필요하며, 지연 시간(latency) 최적화가 지속적으로 필요합니다.

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 MoE 기반의 일반화 메커니즘

Sparse MoE 아키텍처는 일반화 성능 향상에 핵심적 기여를 합니다:

$$\text{Generalization} \propto \frac{\text{Total Parameters (Capacity)}}{\text{Activated Parameters (per input)}}$$

MoE는 전문가들에 걸쳐 훨씬 더 많은 파라미터를 가질 수 있지만 토큰당 일부 파라미터만 활성화합니다. 이것이 라우팅이 건강할 때(healthy routing) MoE 모델이 비교 가능하거나 더 낮은 토큰당 FLOPs에서 dense baseline을 동등하거나 초과할 수 있는 이유입니다.

### 3.2 In-Context Learning (ICL)을 통한 제로샷 일반화

Gemini 1.5는 긴 지시사항에 대한 제로샷 일반화에 탁월하며, 3시간의 비디오 분석, 22시간의 오디오를 near-perfect recall로 처리할 수 있습니다.

가장 인상적인 일반화 사례는 **Machine Translation from One Book (MTOB)** 벤치마크에서의 성과입니다:

Gemini 1.5 Pro는 인상적인 "in-context learning" 능력을 보여주며, 긴 프롬프트에서 새로운 기술을 학습할 수 있습니다. 이 능력은 MTOB 벤치마크에서 테스트되었으며, Kalamang 언어의 문법 매뉴얼이 주어졌을 때 동일한 자료로 학습한 인간과 유사한 수준의 번역 능력을 보였습니다.

### 3.3 다국어 일반화

이러한 개선은 특정 자원 그룹에 국한되지 않으며, Gemini 1.5 Pro는 다양한 자원 수준의 언어에서 동등하게 성능을 향상시킵니다. 중·저자원 언어에서는 1.0 Ultra와 1.5 Pro 사이의 격차가 더 벌어집니다.

### 3.4 장문 컨텍스트에서의 일반화 유지

성능 저하(degradation) 없는 장문 처리는 일반화의 핵심 지표입니다:

$$\text{NLL}(L) \leq \text{NLL}(L_0), \quad \forall L \leq 10M$$

Gemini 1.5 Pro는 기존 언어 모델의 컨텍스트 길이를 한 자릿수 이상 확장합니다. 수백만 토큰으로 확장 시, 예측 성능의 지속적 개선, 합성 검색 태스크에서의 near-perfect recall (>99%), 그리고 전체 긴 문서로부터의 in-context learning과 같은 놀라운 새로운 능력이 나타납니다.

---

## 4. 향후 연구에 미치는 영향과 고려사항

### 4.1 연구에 미치는 영향

1. **장문 컨텍스트의 패러다임 전환:** 기존 RAG(Retrieval-Augmented Generation) 접근법과의 경계를 재정의합니다. 수백만 토큰을 처리하는 능력은 이전에는 불가능했던 실용적 응용을 가능하게 합니다.

2. **MoE의 산업 표준화:** MoE를 사용할 것인지의 문제가 아니라, 프로덕션 규모에서 라우팅, 부하 균형, 추론 서빙을 어떻게 최적화할 것인가가 핵심 질문이 되었습니다.

3. **증류(Distillation) 방법론의 발전:** Gemini 2.5 시리즈에서도 Flash 크기 이하의 소형 모델에 증류를 사용하며, 이는 Gemini 1.5 시리즈에서 확립된 방법입니다.

4. **멀티모달 평가 체계의 혁신 필요:** 기존 평가 체계는 점점 더 발전하는 모델 역량에 한계를 보이며, 개별 모달리티에 초점을 맞추거나 짧은 컨텍스트 길이에 제한됩니다. 따라서 장문 혼합 모달리티 시퀀스에서의 추론 능력 정량 평가가 핵심 과제입니다.

### 4.2 향후 연구 시 고려사항

| 고려사항 | 세부 내용 |
|---|---|
| **MoE 학습 불안정성** | MoE 모델은 학습 불안정성(training instabilities)으로 알려져 있습니다 (Chowdhery et al., 2022; Fedus et al., 2021 등). 라우팅 안정성 개선이 필수적입니다. |
| **라우팅 전략 최적화** | All-to-all 통신과 불균형한 부하 분배는 이론적 이득을 무효화할 수 있습니다. |
| **재현성(Reproducibility)** | 구체적 아키텍처 세부사항이 공개되지 않아, 독립적 연구 재현이 어렵습니다. |
| **안전성 평가** | 장문 컨텍스트의 오용 가능성 및 새로운 위협 벡터에 대한 연구가 필요합니다. |
| **하이브리드 아키텍처** | Attention + SSM 등 이종 아키텍처 결합의 가능성 탐색이 필요합니다. |

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

| 연구 | 연도 | 아키텍처 | 컨텍스트 길이 | 핵심 차별점 |
|---|---|---|---|---|
| **Gemini 1.5 Pro** | 2024 | Sparse MoE Transformer | **10M 토큰** | 멀티모달, near-perfect recall |
| **GPT-4 Turbo** (OpenAI) | 2023 | Dense Transformer | 128K 토큰 | 범용 성능 SOTA |
| **Claude 3.0** (Anthropic) | 2024 | Dense Transformer | 200K 토큰 | 긴 컨텍스트 안정성 |
| **Mamba** (Gu & Dao) | 2023 | Selective SSM | 이론상 무제한 | 선택 메커니즘을 도입한 구조화된 상태공간 모델로, 시퀀스 길이에 선형적으로 확장되며 attention-free 아키텍처에서 SOTA 결과를 달성 |
| **Mixtral 8x7B** (Mistral) | 2023 | Sparse MoE Transformer | 32K 토큰 | LLaMA 2 70B와 비등한 성능을 보이면서 추론 연산 비용은 약 1/6 수준 |
| **MoE-Mamba** | 2024 | MoE + Mamba (SSM) | 확장 가능 | Mamba와 Mixture of Experts 레이어를 결합한 MoE-Mamba 모델 |
| **Gemini 2.5 Pro** | 2025 | Sparse MoE Transformer | 1M+ 토큰 | LMArena 점수가 Gemini 1.5 Pro보다 120포인트 이상 높은 현저한 개선 |
| **LongMamba** | 2024 | 개선된 SSM | 확장 컨텍스트 | SSM(Mamba 모델 등)이 장문 컨텍스트 이해 태스크에서 Transformer보다 일반적으로 성능이 낮다는 최근 연구 결과를 해결하고자 함 |

### 핵심 아키텍처 비교: Attention vs. SSM vs. Sparse MoE

**Transformer Self-Attention의 계산 복잡도:**

$$O(\text{Attention}) = O(L^2 \cdot d)$$

여기서 $L$은 시퀀스 길이, $d$는 모델 차원입니다.

**Mamba (SSM)의 계산 복잡도:**

$$O(\text{Mamba}) = O(L \cdot d \cdot N)$$

여기서 $N$은 상태 차원으로, $L$에 대해 선형 복잡도를 가집니다.

**Sparse MoE (Gemini 1.5)의 계산 복잡도:**

$$O(\text{MoE}) = O(L^2 \cdot d_{\text{attn}} + L \cdot k \cdot d_{\text{expert}})$$

여기서 $k \ll N_{\text{experts}}$이며, attention 부분의 $L^2$는 효율적 attention 기법(Flash Attention 등)과 아키텍처 혁신으로 완화됩니다.

### SSM vs. Transformer 관점에서의 Gemini 1.5 의의

SSM의 주목할 만한 단점은 긴 입력 시퀀스 복사, in-context learning, induction heads 등 특정 시퀀스 처리 태스크에 필수적인 핵심 능력에서의 타협입니다. 이에 반해 Gemini 1.5는 Transformer 기반이면서 MoE를 통해 효율성을 확보하는 전략을 취했으며, 이는 장문 컨텍스트에서의 **in-context learning**과 **정밀 검색 능력**을 모두 유지하는 데 효과적이었음을 실증적으로 보여줍니다.

---

## 참고자료 출처

1. **Gemini Team, Google (2024).** *"Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context."* arXiv:2403.05530. [https://arxiv.org/abs/2403.05530](https://arxiv.org/abs/2403.05530)
2. **Google DeepMind Technical Report (PDF).** [https://storage.googleapis.com/deepmind-media/gemini/gemini\_v1\_5\_report.pdf](https://storage.googleapis.com/deepmind-media/gemini/gemini_v1_5_report.pdf)
3. **Google Blog (2024).** *"Introducing Gemini 1.5, Google's next-generation AI model."* [https://blog.google/technology/ai/google-gemini-next-generation-model-february-2024/](https://blog.google/technology/ai/google-gemini-next-generation-model-february-2024/)
4. **Gradient Flow (2024).** *"Gemini 1.5 Technical Report: Key Reveals and Insights."* [https://gradientflow.com/gemini-1-5-technical-report/](https://gradientflow.com/gemini-1-5-technical-report/)
5. **Gu, A. & Dao, T. (2023).** *"Mamba: Linear-Time Sequence Modeling with Selective State Spaces."* arXiv:2312.00752. [https://arxiv.org/abs/2312.00752](https://arxiv.org/abs/2312.00752)
6. **Patro, B. & Agneeswaran, V. (2025).** *"Mamba-360: Survey of state space models as transformer alternative for long sequence modelling."* Engineering Applications of AI. [https://www.sciencedirect.com/science/article/abs/pii/S0952197625012801](https://www.sciencedirect.com/science/article/abs/pii/S0952197625012801)
7. **Piątkowski et al. (2024).** *"MoE-Mamba: Efficient Selective State Space Models with Mixture of Experts."* [https://llm-random.github.io/posts/moe\_mamba/](https://llm-random.github.io/posts/moe_mamba/)
8. **Encord Blog (2024).** *"Gemini 1.5: Google's Generative AI Model with Mixture of Experts Architecture."* [https://encord.com/blog/google-gemini-1-5-generative-ai-model-with-mixture-of-experts/](https://encord.com/blog/google-gemini-1-5-generative-ai-model-with-mixture-of-experts/)
9. **Prompting Guide.** *"Gemini 1.5 Pro."* [https://www.promptingguide.ai/models/gemini-pro](https://www.promptingguide.ai/models/gemini-pro)
10. **LessWrong (2024).** *"The Gemini 1.5 Report."* [https://www.lesswrong.com/posts/seM8aQ7Yy6m3i4QPx/the-gemini-1-5-report](https://www.lesswrong.com/posts/seM8aQ7Yy6m3i4QPx/the-gemini-1-5-report)
11. **Google DeepMind (2025).** *"Gemini 2.5: Pushing the Frontier with Advanced Reasoning, Multimodality, Long Context, and Next Generation Agentic Capabilities."* [https://storage.googleapis.com/deepmind-media/gemini/gemini\_v2\_5\_report.pdf](https://storage.googleapis.com/deepmind-media/gemini/gemini_v2_5_report.pdf)
12. **Shazeer, N. et al. (2017).** *"Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer."* ICLR 2017.
13. **Fedus, W. et al. (2021).** *"Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity."* JMLR.
14. **Skywork AI (2024).** *"Mixture-of-Experts (MoE) in LLMs: Architecture, Routing, and Gemini."* [https://skywork.ai/blog/mixture-of-experts-moe-llms-architecture-routing-gemini/](https://skywork.ai/blog/mixture-of-experts-moe-llms-architecture-routing-gemini/)
15. **OpenReview.** *Gemini 1.5 Paper Discussion.* [https://openreview.net/forum?id=UUd7I7ZIYn](https://openreview.net/forum?id=UUd7I7ZIYn)

---

> **주의:** Gemini 1.5 논문은 모델의 정확한 전문가 수, 전체 파라미터 규모, 라우팅 전략의 세부사항(Top- $k$의 $k$ 값 등)을 명시적으로 공개하지 않았습니다. 위 수식들은 Gemini 1.5가 근간으로 하는 Sparse MoE 프레임워크(Shazeer et al., 2017; Fedus et al., 2021; Lepikhin et al., 2020)의 일반적 정형화를 기반으로 작성된 것입니다. 모델 내부의 구체적 변형이나 추가 혁신에 대해서는 확정적으로 기술하지 않았습니다.
