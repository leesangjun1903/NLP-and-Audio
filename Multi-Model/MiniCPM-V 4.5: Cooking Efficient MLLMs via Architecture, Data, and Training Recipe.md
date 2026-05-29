
# MiniCPM-V 4.5: Cooking Efficient MLLMs via Architecture, Data, and Training Recipe 

> **논문 정보**
> - **제목:** MiniCPM-V 4.5: Cooking Efficient MLLMs via Architecture, Data, and Training Recipe
> - **저자:** Tianyu Yu 외 32인 (OpenBMB / Tsinghua University)
> - **arXiv ID:** [2509.18154](https://arxiv.org/abs/2509.18154)
> - **제출일:** 2025년 9월 16일
> - **코드/모델:** [GitHub - OpenBMB/MiniCPM-V](https://github.com/OpenBMB/MiniCPM-V), [HuggingFace - openbmb/MiniCPM-V-4_5](https://huggingface.co/openbmb/MiniCPM-V-4_5)

---

## 1. 핵심 주장 및 주요 기여 요약

Multimodal Large Language Models(MLLMs)은 AI 개발의 최전선에서 빠르게 발전하고 있으나, 훈련 및 추론 효율성이 MLLMs의 접근성과 확장성을 제한하는 핵심 병목으로 부상하고 있습니다.

이 논문의 핵심 주장은 **"고성능과 고효율은 상충하지 않는다"**는 것입니다.

이 문제를 해결하기 위해 본 논문은 8B 파라미터 모델인 MiniCPM-V 4.5를 제시하며, 세 가지 핵심 개선 사항을 도입합니다: (1) 이미지와 비디오에 대해 고도로 압축된 인코딩을 위한 통합 3D-Resampler 모델 아키텍처, (2) 복잡한 데이터 엔지니어링 없이 문서 지식과 텍스트 인식을 위한 통합 학습 패러다임, (3) 단답형 및 장문 추론 모드 모두에 능숙한 하이브리드 강화학습 전략입니다.

OpenCompass 평가의 포괄적인 실험 결과에 따르면 MiniCPM-V 4.5는 GPT-4o-latest 같은 널리 사용되는 상용 모델과, Qwen2.5-VL 72B 같은 훨씬 더 큰 오픈소스 모델을 능가합니다.

### 주요 기여 요약표

| 기여 축 | 내용 |
|---|---|
| **Architecture** | Unified 3D-Resampler (공간-시간 통합 압축) |
| **Data** | Unified Learning Paradigm (외부 파서 없이 문서 직접 학습) |
| **Training** | Hybrid RL Strategy (단답/장문 추론 공동 최적화) |

---

## 2. 해결하고자 하는 문제, 제안 방법, 모델 구조, 성능 및 한계

### 2-1. 해결하고자 하는 문제

현재 MLLMs이 직면한 효율성 과제는 크게 세 가지입니다: 고해상도 이미지와 비디오를 위한 대규모 시각 토큰 시퀀스 문제, 문서 지식을 위한 불안정한 외부 파서 의존 문제, 그리고 다양한 추론 모드에 대한 비최적 강화학습 전략 문제입니다.

구체적으로, 기존 비디오 처리의 토큰 비용 문제는 심각합니다:

예를 들어, 해상도 448×448에서 2fps로 촬영된 6초짜리 저해상도 비디오를 처리하는 데 Qwen2.5-VL은 1,536 토큰이, InternVL3는 3,072 토큰이 필요합니다. 이처럼 긴 시각 토큰 시퀀스는 GPU 메모리와 연산 속도 측면에서 훈련 및 추론 비용을 급격히 높입니다.

두 번째로, 문서를 훈련 형식으로 변환하는 데 필요한 외부 PDF 파서가 구조적 오류를 종종 유발한다는 데이터 문제가 있습니다. 세 번째로, 복잡한 추론을 위한 강화학습 기반 훈련 방법이 단순한 작업에도 지나치게 장황한 출력을 생성하여 비효율을 초래합니다.

---

### 2-2. 제안 방법 (수식 포함)

#### (A) Unified 3D-Resampler

이미지와 비디오 인코딩 효율성 병목을 해결하기 위해, 2D-Resampler를 확장하여 비디오의 공간-시간 정보를 공동 압축하는 3D-Resampler를 제안합니다.

핵심 혁신은 2D-Resampler를 시간 차원까지 처리할 수 있도록 확장하여, 연속된 비디오 프레임 간 중복성을 활용한 6× 시간 압축을 달성하는 것입니다. 이미지 처리의 경우 LLaVA-UHD 분할 전략을 채택하여, 2D 공간 위치 임베딩이 있는 학습 가능한 쿼리를 통해 크로스 어텐션으로 고정 길이 시퀀스를 생성합니다.

2D-Resampler에서 3D-Resampler로의 확장 핵심 개념을 수식으로 표현하면 다음과 같습니다:

**2D-Resampler (이미지):**

$$
\mathbf{Q}_{2D} \in \mathbb{R}^{N_q \times d}, \quad \text{where } N_q \ll H \times W
$$

$$
\mathbf{Z}_{img} = \text{CrossAttn}(\mathbf{Q}_{2D},\; \mathbf{K}_{img},\; \mathbf{V}_{img})
$$

**3D-Resampler (비디오):** $T$개 프레임을 그룹 $G$로 묶어 공동 압축:

$$
\mathbf{Q}_{3D} \in \mathbb{R}^{N_q \times d}, \quad \text{Group size: } G = 6 \text{ frames}
$$

$$
\mathbf{Z}_{vid} = \text{CrossAttn}\bigl(\mathbf{Q}_{3D},\; \mathbf{K}_{vid}^{(1:G)},\; \mathbf{V}_{vid}^{(1:G)}\bigr)
$$

$$
\text{Compression ratio} = \frac{H \times W \times G}{N_q} = \frac{448 \times 448 \times 6}{64} \approx 96\times
$$

MiniCPM-V 4.5의 3D-Resampler는 최대 6개의 연속 비디오 프레임을 단 64개의 토큰(MiniCPM-V 시리즈에서 단일 이미지에 사용되는 것과 동일한 토큰 수)으로 그룹화하여 공동 압축함으로써, 비디오 토큰에 대해 96× 압축률을 달성합니다.

이 모듈은 이미지 및 비디오 특징을 압축된 토큰 시퀀스로 효율적으로 압축하며(이미지에 대해 최대 16× 압축률, 비디오에 대해 추가 6×), 이후 LLM 디코더에서 처리됩니다.

#### (B) Unified Learning Paradigm (통합 학습 패러다임)

본 논문은 문서 지식과 텍스트 인식을 위한 통합 학습 패러다임을 제안하는데, 텍스트 영역을 동적으로 손상(dynamic corruption)시켜 외부 파서 없이도 문서 이미지에서 직접 학습할 수 있게 합니다.

이 통합 접근 방식은 보다 효율적이고 탄력적인 학습 과정을 만들어냅니다. 문서의 시각적·텍스트 구조에서 직접 학습함으로써 복잡한 문서 파싱 파이프라인 구축을 피하고, 불안정한 파서가 유발하는 잠재적 노이즈를 방지합니다.

입력 데이터 전략을 수식으로 나타내면:

$$
x_{corrupted} = \text{Corrupt}(x_{doc},\; m),\quad m \sim \text{Bernoulli}(p)
$$

$$
\mathcal{L}_{unified} = \mathbb{E}_{(x_{doc}, y)}\bigl[-\log P_\theta(y \mid x_{corrupted})\bigr]
$$

여기서 $m$은 텍스트 영역 마스킹 여부를 결정하는 이진 마스크이며, 모델은 손상된 입력 $x_{corrupted}$로부터 정답 $y$를 복원하도록 학습됩니다.

#### (C) Hybrid Reinforcement Learning Strategy (하이브리드 강화학습)

MiniCPM-V 4.5는 일상적인 효율적 사용을 위한 **fast thinking** 모드와 복잡한 작업을 위한 **deep thinking** 모드, 두 가지 전환 가능한 추론 모드를 제공합니다. 새로운 하이브리드 강화학습 방법을 통해 두 모드를 공동으로 최적화하여, deep thinking 모드의 성능을 저하시키지 않으면서 fast thinking 모드의 성능을 크게 향상시킵니다.

RLPR과 RLAIF-V를 통합하여, 광범위한 멀티모달 데이터에서 견고한 추론 능력을 일반화하면서 환각(hallucination)을 효과적으로 줄입니다.

Hybrid RL의 보상 함수는 규칙 기반 + 확률 기반 보상의 결합으로 표현됩니다:

```math
r(y, y^*) = \lambda_1 \cdot r_{rule}(y, y^*) + \lambda_2 \cdot r_{prob}(y, y^*)
```

$$
\mathcal{J}(\theta) = \mathbb{E}_{\pi_\theta}\bigl[r(y, y^*)\bigr] - \beta \cdot D_{KL}\bigl(\pi_\theta \| \pi_{ref}\bigr)
$$

여기서 $r_{rule}$은 정답 검증 기반 규칙 보상, $r_{prob}$은 확률 기반 보상(RLAIF-V), $\beta$는 KL 발산 규제 계수입니다.

---

### 2-3. 모델 구조

모델은 Qwen3-8B와 SigLIP2-400M을 기반으로 구축되었으며, 총 8B 파라미터를 가집니다.

아키텍처는 세 가지 주요 모듈로 구성됩니다: 유연한 고해상도 이미지 분할이 가능한 경량 시각 인코더, 압축 특징 인코딩을 위한 통합 3D-Resampler, 그리고 텍스트 생성을 위한 LLM 디코더.

**사전 훈련(Pre-training) 3단계 전략:**

사전 훈련 전략은 기초 능력을 체계적으로 구축하는 점진적 3단계 방식을 채택합니다. 1단계는 다른 구성 요소를 고정한 채 이미지-캡션 정렬로 2D-Resampler를 워밍업하는 데 초점을 맞춥니다. 2단계는 OCR-rich 데이터를 사용해 지각 능력을 향상시키기 위해 시각 인코더를 언프리즈(unfreeze)합니다. 3단계는 텍스트 코퍼스, 인터리빙 샘플, 비디오를 포함한 최고 품질 데이터로 모든 파라미터를 종단간(end-to-end)으로 훈련합니다.

---

### 2-4. 성능 향상

특히 주목할 만한 점은, 이 강력한 성능이 놀라운 효율성과 함께 달성된다는 것입니다. 예를 들어, 널리 사용되는 VideoMME 벤치마크에서 MiniCPM-V 4.5는 30B 미만 모델 중 최고 성능을 달성하면서도, Qwen2.5-VL 7B 대비 GPU 메모리는 46.7%만 사용하고 추론 시간은 8.7%만 사용합니다.

MiniCPM-V 4.5는 8개의 인기 벤치마크를 종합 평가한 OpenCompass에서 평균 77.0점을 달성하며, 8B 파라미터만으로 GPT-4o-latest, Gemini-2.0 Pro 같은 상용 모델과 Qwen2.5-VL 72B 같은 강력한 오픈소스 모델을 능가하여 30B 미만 파라미터에서 가장 성능이 뛰어난 MLLM이 됩니다.

하이브리드 사후 훈련 전략을 기반으로 MiniCPM-V 4.5는 단답형 및 장문 추론 모드 모두에서 뛰어나며, 동시대의 thinking 모델들보다 OpenCompass 평가에서 우수한 성능을 보이면서도 추론 시간은 42.9%~68.2%만 사용합니다.

OpenCompass에서는 GLM-4.1V 대비 42.9%의 시간만으로 평가를 완료하며, VideoMME에서는 추론 시간을 약 10배 단축하면서 가장 적은 GPU 메모리(28G)를 사용합니다.

---

### 2-5. 한계

논문에서 직접 언급된 한계는 다음과 같습니다:

예를 들어, 비디오 OCR 기능이 어느 정도 관찰되었지만, 이를 위해 특별히 설계된 훈련을 수행하지 않았습니다. 이는 모델이 의도하지 않은 능력을 갖추기도 하지만, 특정 도메인에서는 명시적 훈련이 없으면 성능이 불안정할 수 있음을 시사합니다.

또한 다음과 같은 한계도 유추할 수 있습니다:
- **극단적 압축의 손실 가능성:** 96× 압축률은 미세한 시각적 세부 정보 손실을 초래할 수 있음
- **장문 추론의 효율성:** deep thinking 모드에서는 여전히 상당한 추론 시간이 소요됨
- **문서 파싱의 일반화:** 동적 손상 기반 학습이 모든 문서 유형에 대해 동일하게 효과적이지 않을 수 있음

---

## 3. 모델의 일반화 성능 향상 가능성

이 논문의 일반화 향상과 관련된 핵심 요소들은 다음과 같습니다.

### 3-1. 통합 아키텍처를 통한 모달리티 간 지식 전이

통합 설계는 동일한 가중치로 이미지와 비디오 모두를 처리할 수 있게 하여 모델 복잡성을 줄이고, **모달리티 간 지식 전이를 촉진**합니다.

3D-Resampler로의 확장은 경량 SFT 단계를 통해 효율적으로 달성되며, 이미지에서 비디오로의 효율적인 지식 전이를 가능하게 합니다. 예를 들어, 특별한 훈련 없이도 MiniCPM-V 4.5에서 합리적인 수준의 비디오 OCR 능력이 관찰됩니다.

이는 일반화 능력의 핵심적 증거로, 이미지 도메인에서 학습된 OCR 지식이 비디오 도메인으로 자연스럽게 전이되었음을 보여줍니다.

### 3-2. RLPR + RLAIF-V를 통한 추론 일반화

RLPR과 RLAIF-V를 통합함으로써, **광범위한 멀티모달 데이터에서 견고한 추론 능력을 일반화**하면서 환각을 효과적으로 줄입니다.

하이브리드 사후 훈련 전략은 훈련 및 추론 효율성을 향상시킬 뿐만 아니라, **단답형과 장문 추론 모드 간의 일반화를 촉진**합니다.

수식으로 일반화의 메커니즘을 표현하면:

$$
\mathcal{L}_{generalize} = \underbrace{\mathcal{L}_{fast}(\theta)}_{\text{단답형 모드}} + \alpha \cdot \underbrace{\mathcal{L}_{deep}(\theta)}_{\text{장문 추론 모드}} + \beta \cdot D_{KL}(\pi_\theta \| \pi_{ref})
$$

여기서 두 모드를 동시에 최적화함으로써, 모델이 태스크 복잡도에 따라 적절한 추론 깊이를 선택하는 일반화 능력을 갖추게 됩니다.

### 3-3. 통합 문서 학습 패러다임의 일반화 기여

통합 학습 패러다임이 외부 파서에 비해 문서 지식과 텍스트 인식 능력 모두를 향상시키는지가 핵심 연구 질문 중 하나입니다.

아키텍처 혁신, 통합 학습 패러다임, 하이브리드 훈련 전략의 세 가지 접근 방식의 결합은 고성능과 효율성이 상호 배타적인 목표가 아님을 증명합니다. 통합 3D-Resampler는 이해 능력을 희생하지 않고 극적인 토큰 압축을 가능하게 합니다. 문서 직접 학습은 불안정한 외부 의존성을 제거하면서 지식 습득을 향상시킵니다.

---

## 4. 앞으로의 연구에 미치는 영향과 고려할 점

### 4-1. 앞으로의 연구에 미치는 영향

#### (1) 효율적 MLLM 설계 패러다임의 전환

MiniCPM-V 4.5는 MLLM의 접근성과 확장성을 제한해 온 효율성 병목을 해결하는 유망한 방향을 제시합니다. 고성능과 효율성이 상충하지 않는다는 것을 입증한 이 연구는 향후 효율적 MLLM 설계의 기준점이 될 것입니다.

- **3D-Resampler 아키텍처**는 비디오 이해 모델의 기본 구성 요소로 채택될 가능성이 높음
- **동적 손상 기반 통합 학습**은 데이터 증강과 멀티태스크 학습의 새로운 방향을 제시
- **하이브리드 RL**은 추론 모드 제어 가능한 LLM 훈련의 표준 방법론으로 자리잡을 가능성

#### (2) 온디바이스(On-Device) AI로의 경로 제시

대부분의 MLLM은 모바일, 오프라인, 에너지 민감, 프라이버시 보호 시나리오와 같은 애플리케이션 범위를 크게 제한하는 고성능 클라우드 서버에 배포되어야 합니다. MiniCPM-V 시리즈는 엣지 디바이스에 배포 가능한 효율적인 MLLM을 제시합니다.

#### (3) 후속 연구 계보의 촉진

실제로 이후 MiniCPM-o 4.5(실시간 전이중 옴니모달 상호작용)와 MiniCPM-V 4.6(4x/16x 혼합 시각 토큰 압축)으로의 발전이 이어지고 있으며, 이 논문의 성과가 빠른 후속 연구로 이어지고 있음을 확인할 수 있습니다.

---

### 4-2. 앞으로 연구 시 고려할 점

#### (1) 압축-성능 트레이드오프의 정밀 분석

96× 압축이 달성되었지만, 아래의 수식처럼 정보 손실 $\mathcal{I}_{loss}$와 압축률 $r$ 사이의 관계를 더 정밀하게 분석해야 합니다:

$$
\mathcal{I}_{loss}(r) = H(X) - I(X; Z_r), \quad Z_r = \text{Resampler}_r(X)
$$

$$
\text{Optimal } r^* = \arg\max_{r} \; \text{Performance}(r) \;\text{s.t.}\; \text{Latency}(r) \leq T_{budget}
$$

태스크 유형(세밀한 시각적 분석 vs. 고수준 이해)에 따라 최적 압축률이 달라질 수 있습니다.

#### (2) 하이브리드 RL의 안정성 및 수렴 보장

$$
\text{Reward hacking risk: } \max_\theta \mathbb{E}[r] \text{ without } D_{KL} \text{ constraint} \rightarrow \text{degenerate policy}
$$

$\beta$ (KL 발산 규제 계수)의 스케줄링 전략과 두 추론 모드 간 보상 균형 $(\lambda_1, \lambda_2)$에 대한 더 깊은 연구가 필요합니다.

#### (3) 도메인 특화 일반화 검증

공간-시간 압축을 통한 높은 시각 압축률 달성과 통합 아키텍처를 통한 최소 추가 훈련으로의 효율적 적응이 가능하지만, 의료 이미징, 위성 영상, 산업 검사 등 분포 외(Out-of-Distribution) 도메인에서의 일반화 성능은 별도로 검증되어야 합니다.

#### (4) 공정한 벤치마크 비교

VideoMME와 OpenCompass 평가 모두 8×A100 GPU를 사용한 추론으로 평가되었습니다. 실제 엣지 디바이스(예: 스마트폰, 임베디드 시스템)에서의 성능 측정을 포함한 보다 공정한 비교 기준 마련이 필요합니다.

#### (5) 2020년 이후 관련 최신 연구 비교 분석

| 모델/연구 | 연도 | 파라미터 | 주요 특징 | MiniCPM-V 4.5와의 관계 |
|---|---|---|---|---|
| **LLaVA** (Liu et al.) | 2023 | 7~13B | 시각-언어 명령 튜닝 | 기반 패러다임, 토큰 효율성 미흡 |
| **LLaVA-UHD** | 2024 | 7B | 고해상도 이미지 분할 | MiniCPM-V 4.5가 이미지 파티셔닝 전략 채택 |
| **Qwen-VL / Qwen2.5-VL** | 2023/2025 | 7~72B | 강력한 OCR 및 다국어 | 72B 모델을 8B로 능가 |
| **MiniCPM-V (이전 버전)** | 2024 | ~8B | 엣지 배포 효율성 | 4.5의 직접 전신 |
| **DeepSeek-R1** | 2025 | 多 | RL 기반 추론 능력 강화 | Hybrid RL 전략의 영감 |
| **GPT-4o** | 2024/2025 | 비공개 | 상용 멀티모달 SOTA | 이 논문이 8B로 능가 |
| **InternVL3** | 2025 | 多 | 비디오 이해 강화 | 비디오 토큰 처리에서 비효율 (3,072 토큰 vs. 64 토큰) |

---

## 결론

MiniCPM-V 4.5는 **"작지만 강한(Efficient but Powerful)"** MLLM의 새로운 기준을 제시한 연구입니다. MiniCPM-V 4.5는 아키텍처, 데이터, 훈련 방법을 통해 훈련과 추론 모두에서 높은 효율성을 달성합니다. 통합 3D-Resampler를 통해 고프레임 및 장문 비디오 이해에서 강력한 성능을 달성하고, 통합 학습 패러다임을 통해 문서 이미지에서 직접 학습하며, 하이브리드 사후 훈련 전략을 통해 단답형과 장문 추론 모드 간의 일반화를 촉진합니다.

전반적으로 MiniCPM-V 4.5는 MLLM 개발의 효율성 병목을 해결하는 유망한 방향을 제시합니다.

---

## 📚 참고 자료 및 출처

| # | 자료 | URL |
|---|---|---|
| 1 | **MiniCPM-V 4.5 (arXiv 원문)** | https://arxiv.org/abs/2509.18154 |
| 2 | **arXiv HTML 전문** | https://arxiv.org/html/2509.18154v1 |
| 3 | **arXiv PDF 원문** | https://arxiv.org/pdf/2509.18154 |
| 4 | **HuggingFace 모델 카드** | https://huggingface.co/openbmb/MiniCPM-V-4_5 |
| 5 | **GitHub 공식 저장소** | https://github.com/OpenBMB/MiniCPM-V |
| 6 | **ChatPaper 요약** | https://www.chatpaper.ai/dashboard/paper/1e618724-abf1-4ad8-9002-a42484e122a3 |
| 7 | **AIModels.fyi 분석** | https://www.aimodels.fyi/papers/arxiv/minicpm-v-45-cooking-efficient-mllms-via |
| 8 | **Liner.com 리뷰** | https://liner.com/review/minicpmv-45-cooking-efficient-mllms-via-architecture-data-and-training |
| 9 | **MiniCPM-V (이전 버전, arXiv)** | https://arxiv.org/abs/2408.01800 |
| 10 | **HyperAI 논문 페이지** | https://hyper.ai/en/papers/2509.18154 |

> ⚠️ **정확도 주의 사항:** 3D-Resampler 및 Hybrid RL의 내부 수식 일부는 공개된 arXiv 원문과 HuggingFace 모델 카드를 기반으로 재구성한 것이며, 논문의 정확한 수식 표기와 다소 다를 수 있습니다. 완전한 수식 검증을 위해서는 arXiv 원문(PDF)을 직접 참고하시기를 권장합니다.
