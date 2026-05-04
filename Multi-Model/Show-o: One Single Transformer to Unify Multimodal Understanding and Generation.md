
# Show-o: One Single Transformer to Unify Multimodal Understanding and Generation

---

## 1. 핵심 주장 및 주요 기여 요약

### 1.1 핵심 주장

Show-o는 멀티모달 이해(Understanding)와 생성(Generation)을 통합하는 단일 트랜스포머로, 완전한 자기회귀(autoregressive) 모델과 달리, 자기회귀 및 (이산적) 확산(discrete diffusion) 모델링을 통합하여 다양한 모달리티의 입력과 출력을 적응적으로 처리한다.

기존의 이해 모델(LLaVA)도 트랜스포머 구조이고, 선도적 생성 모델(Stable Diffusion 3)도 트랜스포머이다. 이 사실이 핵심 연구 질문을 자극한다: **"하나의 단일 트랜스포머가 멀티모달 이해와 생성 모두를 처리할 수 있는가?"**

### 1.2 주요 기여

논문의 주요 기여는 다음과 같이 요약된다:
1. 단일 트랜스포머를 이용한 멀티모달 이해 및 생성 통합 모델(Show-o) 제시
2. 하나의 트랜스포머 내에서 자기회귀와 (이산적) 확산 모델링을 혁신적으로 통합하여 텍스트와 이미지를 서로 다른 방식으로 처리하는 다용성 입증
3. 이해/생성 개별 기준 모델과 동등하거나 더 많은 파라미터를 가진 모델에 비해 동등하거나 더 나은 성능 달성

Show-o는 ICLR 및 NeurIPS 2025에 채택된 논문이다.

---

## 2. 해결하고자 하는 문제, 제안 방법(수식), 모델 구조, 성능 및 한계

### 2.1 해결하고자 하는 문제

기존의 연구들은 이해와 생성을 결합한 통합 시스템을 구성하려 시도했지만, 두 도메인을 독립적으로 처리하고 이해와 생성 각각을 담당하는 개별 모델을 포함하는 방식이 대부분이었다. 예를 들어, NExT-GPT는 멀티모달 이해를 위해 기반 언어 모델을 사용하지만, 이미지 생성을 위해 별도의 사전 학습된 확산 모델이 필요하다.

개발 목표는 자기회귀 및 확산 모델링을 동시에 포함하는 멀티모달 이해·생성 통합 모델이며, 이를 위해 해결해야 할 핵심 과제들은 다음과 같다: ①모델의 입/출력 공간 정의 방법, ②다양한 모달리티의 입력 데이터 통합 방법, ③하나의 트랜스포머에 자기회귀와 확산 모델링을 함께 포함하는 방법, ④통합 모델의 효과적 학습 방법. Show-o는 이 모든 도전 과제에 대한 패키지 솔루션으로 제시된다.

### 2.2 제안 방법 및 수식

#### 2.2.1 이산 토큰 공간 (Discrete Token Space)

연속 확산 대신, MaskGIT에서 사용된 마스크 토큰 예측을 단순화된 이산적 확산 모델링으로 채택하여, 하나의 단일 트랜스포머 내에서 이산 토큰 예측이라는 보다 통합된 학습 목적을 가능하게 한다.

Show-o는 사전 학습된 LLM 위에 구축되었으며, 이산 공간에서 통합 학습을 수행하고 이산 텍스트 및 이미지 토큰을 포함하는 통합 어휘(unified vocabulary)를 유지한다.

**텍스트 토큰화:** 기존 LLM 토크나이저 활용

$$x_{\text{text}} \rightarrow \{v_1, v_2, \ldots, v_n\} \in \mathcal{V}_{\text{text}}$$

**이미지 토큰화 (MAGVIT-v2 기반):**

$$x_{\text{image}} \rightarrow \{u_1, u_2, \ldots, u_m\} \in \mathcal{V}_{\text{image}}$$

통합 어휘: $\mathcal{V} = \mathcal{V}\_{\text{text}} \cup \mathcal{V}_{\text{image}}$

#### 2.2.2 Omni-Attention 메커니즘

Show-o는 시퀀스 내 텍스트 토큰 $v$를 **인과적 어텐션(causal attention)**으로 처리하며, 이미지 토큰 $u$는 **전체 어텐션(full attention)**으로 처리하여 각 토큰이 다른 모든 토큰과 종합적으로 상호작용할 수 있도록 한다.

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V$$

여기서 마스크 $M$은:

$$M_{ij} = \begin{cases} 0 & \text{(text-to-text: 이전 토큰만 참조, causal)} \\ 0 & \text{(image-to-all: 모든 토큰 참조, full)} \\ -\infty & \text{(미래 텍스트 토큰 마스킹)} \end{cases}$$

#### 2.2.3 학습 목적 함수 (Training Objectives)

두 가지 학습 목적 함수를 사용한다: ①**Next Token Prediction (NTP)** — 자기회귀 모델링용, ②**Mask Token Prediction (MTP)** — 이산 확산 모델링용.

**① NTP (자기회귀 언어 모델링 / 텍스트 생성 및 이해):**

$$\mathcal{L}_{\text{NTP}} = -\sum_{t=1}^{T} \log p_\theta\left(v_t \mid v_{<t}, u_{1:m}\right)$$

**② MTP (마스크 토큰 예측 / 이미지 생성):**

이산 확산 과정에서, 시간 스텝 $t$에서 이미지 토큰을 마스킹:

$$q(u^t \mid u^0) = \prod_{i} \left[(1 - \gamma(t)) \cdot \mathbf{1}[u_i^t = u_i^0] + \gamma(t) \cdot \mathbf{1}[u_i^t = \texttt{[MASK]}]\right]$$

모델은 이를 복원하는 역방향 과정 학습:

$$\mathcal{L}_{\text{MTP}} = -\mathbb{E}_{t, u^0, u^t} \left[\sum_{i: u_i^t = [\text{MASK}]} \log p_\theta(u_i^0 \mid u^t, v_{1:n}, t)\right]$$

**통합 목적 함수:**

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{NTP}} + \lambda \cdot \mathcal{L}_{\text{MTP}}$$

여기서 $\lambda$는 두 목적 함수 간 균형을 조정하는 하이퍼파라미터.

#### 2.2.4 통합 프롬프팅 전략 (Unified Prompting Strategy)

Show-o는 텍스트 조건 정보를 내재적으로 인코딩하여 추가적인 텍스트 인코더를 제거한다. 다양한 입력 데이터와 태스크 변형을 수용하기 위해 텍스트 토크나이저와 이미지 토크나이저를 사용하여 이산 토큰으로 인코딩하고, 통합 프롬프팅 전략을 통해 구조화된 시퀀스로 처리한다.

입력 시퀀스 형식:

$$\mathbf{S} = [\texttt{[SOT]}, v_1, \ldots, v_n, \texttt{[SOI]}, u_1, \ldots, u_m, \texttt{[EOI]}, \ldots]$$

### 2.3 모델 구조

기본 Show-o는 사전 학습된 MAGVIT-v2를 채택하여 입력 이미지를 이산 토큰으로 토큰화하고, 이 토큰들은 임베딩 레이어를 통해 멀티모달 이해를 위한 입력 임베딩으로 변환된다.

입력 데이터는 모달리티에 관계없이 토큰화된 후 형식화된 입력 시퀀스로 구성된다. Show-o는 텍스트 토큰은 인과적 어텐션(causal attention)으로 자기회귀적으로 처리하고, 이미지 토큰은 전체 어텐션(full attention)을 통한 (이산적) 디노이징 확산 모델링으로 처리한 후 원하는 출력을 생성한다.

전체 모델 구조 개요:

```
입력 (텍스트/이미지/혼합)
        ↓
   통합 토크나이저 (Unified Tokenizer)
   - 텍스트: LLM 토크나이저
   - 이미지: MAGVIT-v2
        ↓
   통합 임베딩 레이어 (Unified Vocabulary)
        ↓
   단일 트랜스포머 (Phi-1.5 기반, ~1.3B 파라미터)
   - Omni-Attention (텍스트: Causal, 이미지: Full)
        ↓
   ┌──────────────────┬───────────────────────┐
   ↓ (텍스트 출력)         ↓ (이미지 출력)
자기회귀 디코딩 (NTP)   이산 확산 디코딩 (MTP/MaskGIT 스타일)
   ↓                        ↓
VQA, 캡션 등           텍스트→이미지, 인페인팅 등
```

Show-o의 시퀀스 포맷과 Omni-Attention을 따라, 비디오 토큰으로 파인튜닝함으로써 비디오 이해 및 생성으로도 확장하기 편리한 구조를 갖는다.

### 2.4 성능 향상

Show-o는 단일 트랜스포머 내에서 멀티모달 이해 및 생성을 통합하는 프레임워크로, 훨씬 더 작은 모델 크기로도 InstructBLIP, Qwen-VL-Chat, mPLUG-Owl2와 같은 이해 전용 모델과 비교하여 POPE, MME, Flickr30k, VQAv2 벤치마크에서 경쟁력 있는 성능을 달성하고 GQA 벤치마크에서는 더 우수한 성능을 보인다. 또한 NExT-GPT-13B, Chameleon-34B와 같이 훨씬 더 많은 파라미터를 가진 통합 모델과 비교해서도 Flickr30k에서 준수한 성능을, VQAv2에서 훨씬 더 좋은 성능을 달성한다.

| 태스크 | 평가 지표 | Show-o (1.3B) | 비교 모델 |
|--------|-----------|--------------|-----------|
| VQAv2 | Accuracy | ↑ | NExT-GPT-13B, Chameleon-34B 대비 우수 |
| GQA | Accuracy | ↑ | InstructBLIP, Qwen-VL-Chat 대비 우수 |
| Text-to-Image | GenEval | 경쟁적 | DALL-E 2, SDXL 대비 |
| Inpainting/Extrapolation | 정성적 | 가능 | 통합 모델로서 최초 수준 |

### 2.5 한계 (Limitations)

논문 Figure 11에서 Show-o의 멀티모달 이해 및 생성에 대한 실패 사례가 보고되고 있다.

논문과 관련 문헌에서 확인되는 주요 한계:

1. **이산 토큰의 해상도 한계**: MAGVIT-v2 기반 이산 이미지 토큰화는 연속 확산 기반 모델 대비 이미지 품질에서 격차가 있을 수 있음
2. **모달리티 간 간섭 (Gradient Interference)**: 이질적인 상태 공간과 손상 과정이 목적 함수 불일치 및 그래디언트 간섭을 유발할 수 있어 불안정한 최적화와 차선의 성능으로 이어질 수 있다.
3. **모델 규모 제한**: 1.3B 파라미터 규모로 대형 전문 모델 대비 복잡한 추론에서 약점
4. **추가 텍스트 인코더 부재**: 이점이 될 수 있지만, 텍스트 조건화의 표현력 제약 가능

---

## 3. 일반화 성능 향상 가능성

Show-o의 일반화 성능 향상 가능성은 여러 측면에서 논의될 수 있다.

### 3.1 통합 학습의 교차 모달 일반화 효과

Show-o는 멀티모달 이해와 생성을 종합적으로 처리하기 위해 기존의 발전된 기술들을 유연하게 통합하는 통합 모델이다. 단일 트랜스포머 내에서 두 학습 목적이 공유 파라미터를 통해 최적화됨으로써, **이해에서 학습된 표현이 생성 품질 향상에 기여하고, 생성을 위한 이산 확산 학습이 이해의 시각적 토큰 표현을 풍부하게** 만드는 양방향 정규화 효과가 발생한다.

수식으로 표현하면, 공유 파라미터 $\theta$에 대한 다중 태스크 학습:

$$\theta^* = \arg\min_\theta \left[ \mathcal{L}_{\text{NTP}}(\theta; \mathcal{D}_{\text{understanding}}) + \lambda \cdot \mathcal{L}_{\text{MTP}}(\theta; \mathcal{D}_{\text{generation}}) \right]$$

이 구조에서 각 태스크가 다른 태스크에 대한 암묵적 정규화(implicit regularization) 역할을 하여 **과적합 방지 및 일반화 향상**이 기대된다.

### 3.2 비디오·혼합 모달리티로의 확장성

이산 토큰으로 8 FPS 비디오 텐서($3 \times 17 \times 256 \times 256$)를 $5 \times 16 \times 16$으로 압축하여, Show-o의 시퀀스 포맷과 Omni-Attention 구조를 그대로 활용해 비디오 이해 및 생성으로의 파인튜닝이 편리하다.

### 3.3 사전 학습 LLM 지식의 전이

MaskGIT과 유사한 단순화된 이산 디노이징 확산을 활용하여 이산 이미지 토큰을 모델링하며, Show-o는 텍스트 조건 정보를 내재적으로 인코딩함으로써 추가적인 텍스트 인코더를 제거한다. 이는 사전 학습된 LLM의 언어 이해 능력이 이미지 생성의 조건 정보로 직접 활용됨을 의미하며, **다양한 프롬프트 스타일에 대한 일반화 성능 향상**에 기여한다.

### 3.4 Omni-Attention의 일반화 기여

Omni-Attention은 텍스트 토큰이 이전 이미지 토큰 전체에 접근하고, 이미지 토큰이 선행 텍스트 토큰 전체와 상호작용하도록 허용하여:

$$\text{Attention Score}(u_i, v_j) = \frac{\exp(q_{u_i}^T k_{v_j} / \sqrt{d})}{\sum_{k} \exp(q_{u_i}^T k_k / \sqrt{d})}$$

이러한 교차 모달 어텐션은 텍스트-이미지 정렬(alignment)을 강화하고, 특히 **비전-언어 이해 태스크에서의 도메인 외 일반화(OOD generalization)**를 향상시킬 잠재력을 갖는다.

### 3.5 인페인팅/확장 작업의 자연스러운 일반화

이미지와 질문이 함께 주어지면 Show-o는 자기회귀적으로 답변을 생성하며, 텍스트 토큰만 제공되면 이산 디노이징 확산 방식으로 이미지를 생성한다. 이산 확산의 마스킹 특성은 추가 학습 없이도 인페인팅 등의 새로운 태스크로 자연스럽게 일반화 가능하다.

---

## 4. 앞으로의 연구에 미치는 영향과 연구 시 고려할 점

### 4.1 앞으로의 연구에 미치는 영향

#### ① 단일 통합 기반 모델 패러다임 확립

다양한 벤치마크에서 이해 또는 생성을 위해 특화된 동등하거나 더 많은 파라미터를 가진 기존 개별 모델들과 동등하거나 우수한 성능을 입증하며, 차세대 기반 모델로서의 잠재력을 부각시킨다.

Show-o는 이후 연구에서 **통합 멀티모달 모델의 기준선(baseline)**이 되었으며, Janus, JanusFlow, BAGEL 등 후속 연구들이 Show-o의 한계를 개선하는 방향으로 발전하고 있다.

#### ② 이산 확산의 LLM 프레임워크 통합 가능성 증명

개선된 네이티브 통합 멀티모달 모델은 자기회귀 모델링과 플로우 매칭(flow matching)을 활용하여 텍스트, 이미지, 비디오를 포함한 다양한 모달리티에 걸쳐 광범위한 멀티모달 이해 및 생성 태스크에서 다용성을 입증한다.

#### ③ 관련 후속 연구들

Janus는 멀티모달 이해와 생성을 통합하는 자기회귀 프레임워크로, 시각적 인코딩을 별도의 경로로 분리하면서도 단일 통합 트랜스포머 구조를 활용한다. JanusFlow는 자기회귀 언어 모델과 정류 흐름(rectified flow)을 통합하는 미니멀리스트 아키텍처를 도입한다.

#### ④ 비디오 및 Any-to-Any 모달리티 확장 트리거

이미지, 텍스트, 오디오, 행동을 이해하고 생성할 수 있는 최초의 자기회귀 멀티모달 모델 연구들이 Show-o의 영향 하에 발전하고 있다.

### 4.2 2020년 이후 관련 최신 연구 비교 분석

| 모델 | 연도 | 아키텍처 | 이해 | 생성 | 핵심 차별점 |
|------|------|----------|------|------|------------|
| DALL-E 2 | 2022 | 확산 모델 | ✗ | ✓ | 이미지 생성 전용 |
| LLaVA | 2023 | AR Transformer | ✓ | ✗ | 이해 전용 |
| Chameleon | 2024 | AR Transformer | ✓ | ✓ | 모든 모달 AR 처리 |
| NExT-GPT | 2023 | LLM + 확산 | ✓ | ✓ | 별도 확산 모델 필요 |
| **Show-o** | **2024** | **AR + Discrete Diffusion** | **✓** | **✓** | **단일 트랜스포머 통합** |
| Janus | 2024 | AR Transformer | ✓ | ✓ | 이해/생성 인코더 분리 |
| JanusFlow | 2024 | AR + Rectified Flow | ✓ | ✓ | 플로우 매칭 통합 |
| BAGEL | 2025 | 통합 기반 모델 | ✓ | ✓ | 오픈소스, 강화된 추론 |

BAGEL은 멀티모달 이해와 생성을 기본적으로 지원하는 오픈소스 기반 모델로, 오픈소스 통합 모델들 중 표준 벤치마크에서 멀티모달 생성 및 이해 모두에서 크게 앞서며 고급 멀티모달 추론 능력을 보여준다.

Show-o는 Chameleon과 달리, 자기회귀 모델링 대신 이산 확산 과정을 시각 생성에 활용한다는 핵심 차별점이 있다.

### 4.3 앞으로 연구 시 고려할 점

#### ① 모달리티 간 최적화 충돌 해결

두 분기가 이질적인 상태 공간과 손상 과정에서 작동하기 때문에, 단일 밀집 모델에서 두 모달리티를 함께 학습하는 것은 목적 함수 불일치 및 그래디언트 간섭을 유발해 학습 충돌과 차선의 성능으로 이어질 수 있다. 이 문제를 해결하기 위한 **모달리티별 학습률 스케줄링**, **MoE(Mixture of Experts) 활용** 등이 중요 연구 방향이다.

$$\nabla_\theta \mathcal{L}_{\text{total}} = \nabla_\theta \mathcal{L}_{\text{NTP}} + \lambda \nabla_\theta \mathcal{L}_{\text{MTP}}$$

두 그래디언트의 방향이 충돌할 경우(cosine similarity < 0), 가중치 조정이나 그래디언트 수술(gradient surgery) 기법이 필요하다.

#### ② 이산 vs. 연속 표현의 트레이드오프

이산 이미지 토큰화는 LLM 프레임워크와의 통합을 용이하게 하지만, 연속 확산 모델 대비 이미지 품질에서 한계가 있다. **이산 + 연속 하이브리드** 방식 (예: Transfusion, LlaDA-o)의 가능성을 고려해야 한다.

#### ③ 스케일업 전략 및 데이터 균형

데이터셋 규모와 이미지 해상도가 이산 이미지 토큰의 멀티모달 이해 능력에 미치는 영향에 대한 체계적인 탐구가 필요하다.

#### ④ 비디오·오디오 등 다중 모달리티 확장

이미지, 텍스트, 오디오, 행동을 모두 이해하고 생성하는 Any-to-Any 멀티모달 모델의 개발은 Show-o의 이산 확산 통합 패러다임을 더 많은 모달리티로 확장하는 자연스러운 방향이다.

#### ⑤ 추론 효율성 및 병렬 디코딩 활용

이산 확산의 MaskGIT 스타일 병렬 디코딩은 자기회귀 방식 대비 **추론 속도 향상**의 잠재력을 가진다:

$$\text{AR 복잡도}: O(n) \text{ 순차 단계} \quad \text{vs.} \quad \text{MaskGIT}: O(\log n) \text{ 병렬 단계}$$

이 이점을 극대화하는 효율적 추론 알고리즘 개발이 필요하다.

---

## 📚 참고자료 및 출처

| # | 출처 | URL |
|---|------|-----|
| 1 | **Show-o 논문 (arXiv)** — Xie et al., 2024 | https://arxiv.org/abs/2408.12528 |
| 2 | **Show-o ICLR 2025 공식 논문 (PDF)** | https://proceedings.iclr.cc/paper_files/paper/2025/file/45f0d179ef7e10eb7366550cd4e574ae-Paper-Conference.pdf |
| 3 | **Show-o GitHub 공식 저장소** | https://github.com/showlab/Show-o |
| 4 | **Show-o 프로젝트 페이지** | https://showlab.github.io/Show-o/ |
| 5 | **Show-o Hugging Face 논문 페이지** | https://huggingface.co/papers/2408.12528 |
| 6 | **Show-o Semantic Scholar** | https://www.semanticscholar.org/paper/Show-o:... |
| 7 | **Show-o 기술 보고서 (PDF)** | https://showlab.github.io/Show-o/assets/show-o.pdf |
| 8 | **Awesome Unified Multimodal Models (GitHub)** | https://github.com/showlab/Awesome-Unified-Multimodal-Models |
| 9 | **LlaDA-o (arXiv, 2026)** — 후속 통합 확산 모델 비교 | https://arxiv.org/html/2603.01068 |
| 10 | **Omni-Diffusion (arXiv, 2026)** — 비교 후속 연구 | https://arxiv.org/pdf/2603.06577 |
| 11 | **The Evolution of Multimodal Model Architectures (arXiv)** | https://arxiv.org/html/2405.17927v1 |
| 12 | **GENEVAL Benchmark (NeurIPS 2023)** | https://proceedings.neurips.cc/paper_files/paper/2023/... |
| 13 | **OpenReview (Show-o)** | https://openreview.net/forum?id=o6Ynz6OIQ6 |

---

> ⚠️ **정확도 안내**: 수식의 세부 표기(예: $\gamma(t)$의 구체적 스케줄, $\lambda$ 값 등)는 논문 본문에서 명시적으로 공개된 내용을 기반으로 하되, 일부 세부 표기는 논문의 맥락에서 표준적인 MaskGIT/이산 확산 표기법에 따라 재구성하였습니다. 완전한 수식 세부 사항은 공식 논문 PDF를 참조하시기 바랍니다.
