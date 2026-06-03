
# Mellum2 Technical Report

> **참고 자료 및 출처**
> - **주 논문**: Marko Kojic et al., *Mellum2 Technical Report*, arXiv:2605.31268 (May 2026)
> - JetBrains Blog: *Mellum2 Goes Open Source* (blog.jetbrains.com, 2026.06)
> - JetBrains on HuggingFace: *Introducing Mellum2: A 12B Mixture-of-Experts Model* (huggingface.co/blog/JetBrains)
> - MarkTechPost: *JetBrains Releases Mellum2: A 12B MoE Model* (marktechpost.com, 2026.06)
> - TechAIApp: *JetBrains Releases Mellum2* (techaiapp.com, 2026.06)
> - The New Stack: *JetBrains Mellum2 Open Source Coding Model* (thenewstack.io, 2026.06)
> - AI Weekly: *JetBrains Open-Sources Mellum 2 MoE Coding Model* (aiweekly.co, 2026.06)
> - AlphaXiv: *Mellum2 Technical Report* (alphaxiv.org, 2026.06)
> - arXiv PDF: *Mellum 2 Technical Report v1.0* (arxiv.org/pdf/2605.31268)
> - 비교 연구: *Muon is Scalable for LLM Training*, arXiv:2502.16982
> - 비교 연구: *Debunk the Myth of SFT Generalization*, arXiv:2510.00237
> - 비교 연구: *Limits of Generalization in RLVR*, arXiv:2510.27044
> - 비교 연구: *Generalization of RLVR Using Causal Reasoning as a Testbed*, arXiv:2512.20760

---

## 1. 핵심 주장 및 주요 기여 요약

Mellum 2는 JetBrains가 개발한 **오픈 웨이트 120억(12B) 파라미터 Mixture-of-Experts(MoE) 언어 모델**로, 토큰당 25억(2.5B) 파라미터가 활성화됩니다.

Mellum 2는 코드 생성·편집, 디버깅, 다단계 추론, 도구 사용 및 함수 호출, 에이전틱 코딩, 대화형 프로그래밍 지원 등 소프트웨어 엔지니어링에 특화된 **범용 언어 모델**이며, 완성 중심의 4B 밀집(dense) 모델인 Mellum의 후속작입니다.

### 주요 기여 요약

| 기여 항목 | 내용 |
|---|---|
| 아키텍처 | MoE (64 전문가, 8개 활성), GQA, SWA, MTP 헤드 결합 |
| 사전학습 | ~10.6조 토큰, 3단계 커리큘럼, Muon 옵티마이저 |
| 컨텍스트 확장 | layer-selective YaRN으로 8K → 128K |
| 포스트 트레이닝 | SFT + RLVR 2단계 |
| 릴리즈 | Base / Instruct / Thinking 3종 (6개 체크포인트), Apache 2.0 |

AI 시스템이 성숙해지면서 단일 프론티어 모델에서 검색기, 라우터, 코드 인식 모델, 검증기, 도구 호출기, 대형 추론 모델 등 여러 특화 구성 요소가 함께 작동하는 방향으로 진화하고 있습니다. JetBrains는 Mellum2를 이런 대형 AI 시스템 내 고빈도 작업에 최적화된 **"focal 모델"**로 정의합니다.

---

## 2. 해결하고자 하는 문제, 제안 방법, 모델 구조, 성능 및 한계

### 2-1. 해결하고자 하는 문제

현대 AI 시스템은 라우팅, 검색, 요약, 계획, 검증, 도구 사용 등 다수의 모델 호출에 점점 더 의존하고 있습니다. 이러한 작업들 중 상당수는 **지연 시간(latency)에 민감**하며 최대 규모의 모델을 필요로 하지 않습니다. Mellum2는 바로 이러한 워크로드를 타겟으로 합니다.

모든 인터랙션마다 고비용 원격 추론이 필요한 모델은 비용 및 프라이버시 문제를 야기합니다. 더 작은 활성 파라미터 풋프린트는 JetBrains가 **더 빠른 응답**과 코드를 외부 서비스에 전송하는 것을 꺼리는 환경에서의 유연한 배포를 가능하게 합니다.

---

### 2-2. 제안하는 방법 (수식 포함)

#### (A) MoE 라우팅

Mellum2는 총 12B 파라미터의 MoE 아키텍처를 사용하며, MoE 모델에서는 각 토큰마다 파라미터의 일부만 실행됩니다. 이 모델은 **64개의 전문가(experts)**를 가지며, 토큰당 **8개를 활성화**합니다. 이를 통해 토큰당 연산량은 2.5B 밀집 모델 수준으로 유지하면서 높은 용량과 전문화를 달성합니다.

MoE의 라우팅은 일반적으로 아래와 같이 표현됩니다:

$$\text{MoE}(x) = \sum_{i=1}^{k} G(x)_i \cdot E_i(x)$$

여기서 $G(x)$는 게이팅 네트워크, $E_i(x)$는 $i$번째 전문가, $k=8$은 활성화된 전문가 수입니다.

#### (B) Muon 옵티마이저

Mellum2는 **Muon 옵티마이저**를 사용하며, Moonlight의 분산 구성을 채택했습니다. Muon은 임베딩 및 출력 레이어에 Adam을 사용하면서 은닉 레이어에 **직교화(orthogonalization) 기반 업데이트**를 적용합니다. Moonlight 설정이 AdamW를 큰 폭으로 상회하여 검증 손실을 0.028 (~2.5%) 감소시켰습니다.

Muon의 핵심은 Newton-Schulz 반복을 통한 직교화이며, 업데이트 $\Delta W$는 스펙트럴 노름이 1이 되도록 정규화됩니다:

$$\Delta W = \text{orthogonalize}(\nabla_W \mathcal{L}), \quad \|\Delta W\|_{\sigma} = 1$$

#### (C) 학습률 스케줄

사전 학습은 **Warmup-Hold-Decay 스케줄(선형 감쇠, 0까지)**을 사용하며, FP8 하이브리드 정밀도 하에서 Muon으로 최적화됩니다.

SFT 단계의 학습률 스케줄은 다음과 같습니다:

```math
\eta(t) = \begin{cases} \eta_{\max} \cdot \frac{t}{T_{\text{warmup}}} & t \leq T_{\text{warmup}} \\ \eta_{\max} \cdot \cos\!\left(\frac{\pi (t - T_{\text{warmup}})}{2(T_{\text{total}} - T_{\text{warmup}})}\right) & t > T_{\text{warmup}} \end{cases}
```

구체적으로 SFT 학습률은 최대 $3 \times 10^{-5}$에서 100 iteration에 걸쳐 선형으로 워밍업된 후, 코사인 방식으로 $3 \times 10^{-6}$(최대값의 10%)까지 감쇠합니다.

#### (D) 컨텍스트 확장: Layer-Selective YaRN

사전 학습 이후, Mellum 2의 유효 컨텍스트 길이를 8,192 토큰에서 **131,072 토큰(128K)**으로 확장하기 위해 전용 장문맥 확장 단계를 수행했습니다. YaRN 주파수 재매핑은 **글로벌(전체 어텐션) 레이어에만 적용**되며, 슬라이딩 윈도우 레이어는 원래의 RoPE 파라미터를 유지합니다.

YaRN의 위치 인코딩 조정 수식:

$$\theta_i' = \theta_i \cdot s^{-\frac{2i}{d}}$$

여기서 $s$는 스케일링 인자, $d$는 임베딩 차원, $\theta_i$는 원래 RoPE 주파수입니다.

에이블레이션 실험 결과, **글로벌 레이어에만 YaRN을 적용**하는 방식이 (i) 모든 레이어에 균일한 RoPE 기저(base) 확장, (ii) $\theta$ 변경 없이 유지하는 방식보다 모두 우수한 성능을 보였습니다.

#### (E) Multi-Token Prediction (MTP) 헤드

아키텍처는 MoE(64 전문가, 8 활성)를 기반으로 하며, 4개의 KV 헤드를 가진 **Grouped-Query Attention(GQA)**, 4개 레이어 중 3개에 적용된 **Sliding Window Attention(SWA)**, 그리고 보조 사전 학습 목적함수이자 **투기적 디코딩(speculative decoding)**의 내장 드래프트 모델로 이중 역할을 하는 단일 **Multi-Token Prediction 헤드**를 결합합니다.

MTP의 학습 목적함수:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{LM}} + \alpha \cdot \mathcal{L}_{\text{MTP}}$$

MTP 헤드의 손실 가중치는 $\alpha = 0.1$로 설정되었습니다.

#### (F) 포스트 트레이닝: SFT + RLVR

포스트 트레이닝은 두 단계로 이루어집니다: **지도 미세조정(SFT)**, 이후 **검증 가능한 보상을 사용한 강화학습(RLVR)**을 수학, 실행 가능한 코딩, 도구 사용, 지시 이행, 추론, 지식 과제에 적용합니다.

RLVR의 보상 신호:

```math
r(y, y^*) = \mathbb{1}[\text{verify}(y) = y^*]
```

여기서 $y$는 모델 출력, $y^*$는 정답, $\text{verify}(\cdot)$는 자동 검증기입니다.

---

### 2-3. 모델 구조 요약

| 구성 요소 | 사양 |
|---|---|
| 총 파라미터 | 12B |
| 활성 파라미터/토큰 | 2.5B |
| 전문가 수 / 활성 수 | 64 / 8 |
| Attention | GQA (4 KV heads) + SWA (4개 중 3개 레이어) |
| 컨텍스트 길이 | 128K (131,072 tokens) |
| MTP 헤드 | 있음 (손실 가중치 $\alpha=0.1$) |
| 옵티마이저 | Muon (Moonlight 분산 구성) |
| 정밀도 | FP8 하이브리드 |
| 사전 학습 토큰 | ~10.6조 토큰 |
| 모달리티 | 텍스트 + 코드 (비멀티모달) |

---

### 2-4. 성능

Mellum 2는 코드 생성, 수학 및 추론, 도구 사용, 지식, 안전성 벤치마크 전반에서 **4B~14B 범위의 오픈 웨이트 기준선들과 경쟁력 있는 성능**을 보이면서도 2.5B 밀집 모델의 토큰당 연산량으로 동작합니다.

Thinking 변형은 **LiveCodeBench v6에서 69.9%**, **AIME 2025+2026에서 58.4%**를 기록했으며, 64 전문가 MoE와 8개 활성화, 131,072 토큰 컨텍스트 윈도우를 갖습니다.

Instruct 변형은 **EvalPlus 78.4**, **BFCL v3 66.3**으로 4B~14B 비교군 대비 강한 성능을 보였습니다.

---

### 2-5. 한계

12B 총 파라미터에도 불구하고, AIME에서 Qwen3.5 4B(68.3점)에 뒤처지는 벤치마크 격차가 존재하며, 이는 수학 집약적 또는 추론 집약적 에이전틱 코딩 워크플로우를 우선시하는 팀의 채택을 늦출 수 있습니다.

JetBrains는 기술 보고서에서 모델의 좁은 학습 초점이 비용을 수반한다고 직접 인정합니다. "The gap reflects a deliberate tradeoff in our training mix toward code and developer documentation rather than broad encyclopedic coverage"라고 저자들은 기술합니다.

Mellum2는 멀티모달이 아니며, 자연어와 코드 데이터에 특화되어 학습되었습니다. 이 전문화는 소프트웨어 엔지니어링 환경에서 우수한 성능을 보장하지만, 다른 도메인에서는 범용성이 제한됩니다.

---

## 3. 모델의 일반화 성능 향상 가능성

### 3-1. 3단계 커리큘럼 학습을 통한 도메인 이동

사전 학습은 **다양한 웹 데이터에서 선별된 코드 및 수학적 컨텐츠**로 혼합을 점진적으로 이동시키는 3단계 커리큘럼을 통해 약 10.6조 토큰에 걸쳐 진행됩니다.

이 과정은 다음과 같이 도식화할 수 있습니다:

$$\mathcal{D}_{\text{total}} = \lambda_1 \mathcal{D}_{\text{web}} + \lambda_2 \mathcal{D}_{\text{code}} + \lambda_3 \mathcal{D}_{\text{math}}, \quad \lambda_1 > \lambda_2 > \lambda_3 \text{ (Phase 1 → Phase 3로 변화)}$$

### 3-2. RLVR의 일반화 기여

Thinking 변형은 **검증 가능한 보상을 사용한 강화학습(RLVR)**으로 학습되며, 복잡한 디버깅, 에이전틱 워크플로우, 다단계 계획을 목표로 최종 답변 전에 명시적인 추론 흔적을 출력합니다.

관련 연구들은 RLVR의 일반화에 대해 다음과 같이 분석합니다:

- **RLVR은 특정 모델 크기와 학습 쿼리 레벨 조합에서** SFT보다 강한 수준 내(within-level) 및 수준 간(across-level) 일반화를 산출하며, RLVR의 효과는 모델의 초기 추론 역량에 의존합니다.

- 그러나 RLVR의 향상 특성은 여전히 불분명합니다. 연구들은 기반 모델의 능력이 상한선으로 작용하면서 정확도를 높이는 동시에 탐색(exploration)을 줄이는 경향이 있음을 보입니다.

- 전반적으로 RLVR은 **새로운 추론 전략을 유도하기보다 기존 역량을 안정화**하는 것으로 보입니다.

### 3-3. Layer-Selective YaRN을 통한 장문맥 일반화

YaRN 주파수 재매핑을 글로벌(전체 어텐션) 레이어에만 적용하고 슬라이딩 윈도우 레이어는 원래의 RoPE 파라미터를 유지하는 이 layer-selective 방법은 Gemma 3 기술 보고서에서 처음 보고되어 OLMo 3에서도 채택되었습니다.

### 3-4. MoE 전문가를 통한 암묵적 일반화

각 전문가가 서로 다른 데이터 도메인이나 언어 패턴에 특화될 수 있으므로, MoE 구조 자체가 일반화의 원천이 될 수 있습니다:

$$P(\text{expert}_i \mid x) = \text{softmax}(W_g \cdot x)_i$$

64개 전문가 중 8개를 활성화하는 구조는 **희소 활성화(sparse activation)**를 통해 망각 없이 다양한 도메인 지식을 보존하는 데 유리합니다.

### 3-5. SFT vs. RLVR 일반화 논쟁

LLM 포스트 트레이닝의 지배적인 서사는 **SFT는 도메인 내(in-domain) 성능을 향상시키지만 기억(memorization)에 취약**하고, 강화학습(RL)이 더 잘 일반화한다는 것입니다.

그러나 최근 연구들은 이 서사를 재검토하며, **적절한 데이터로 학습된 순수 SFT가 비슷하거나 우수한 일반화**를 달성할 수 있다고 보여줍니다. 이는 SFT의 많은 한계가 최대 우도 목적함수 자체가 아닌 제한된 데이터 설계에서 비롯됨을 시사합니다.

Mellum2는 SFT와 RLVR을 모두 사용하는 **하이브리드 포스트 트레이닝** 전략을 채택함으로써 이 두 패러다임의 장점을 결합하려고 합니다:

$$\mathcal{L}_{\text{post}} = \underbrace{\mathcal{L}_{\text{SFT}}}_{\text{도메인 내 정렬}} + \underbrace{\mathcal{L}_{\text{RLVR}}}_{\text{검증 가능 보상 일반화}}$$

---

## 4. 앞으로의 연구에 미치는 영향 및 고려할 점

### 4-1. 연구에 미치는 영향

#### ① "Focal Model" 패러다임의 확산
Mellum2를 빠르고 범위가 좁은 "focal 모델"로 정의하는 것은 전체 스택을 대체하는 것이 아니라, 스택을 더 빠르고, 더 저렴하고, 더 제어하기 쉽게 만드는 것을 목표로 합니다. 이 패러다임은 향후 **계층적 AI 시스템 설계** 연구를 촉진할 것입니다.

#### ② 오픈 소스를 통한 연구 가속
오픈 웨이트는 연구자, 고급 사용자 및 엔터프라이즈 팀이 폐쇄형 어시스턴트가 지원할 수 없는 방식으로 모델을 검사, 벤치마크 및 적응할 수 있게 합니다.

#### ③ MoE + Muon 조합의 표준화 가능성
MoE-14B에서 두 Muon 구성 모두 성공적으로 수렴했으며, JetBrains는 **밀집 및 MoE 아키텍처 모두에서의 안정성** 때문에 Moonlight 구성을 선택했습니다. 이는 대형 모델 학습에서 Muon을 표준 옵티마이저로 자리잡게 하는 데 기여할 수 있습니다.

#### ④ Layer-Selective 컨텍스트 확장 레시피 검증
이 layer-selective 방법은 Gemma 3, OLMo 3에 이어 Mellum2에서도 효과가 검증되어, 향후 장문맥 확장 연구의 표준 레시피로 정착할 가능성이 높습니다.

---

### 4-2. 앞으로 연구 시 고려할 점

#### ① 일반화 vs. 전문화의 균형
Mellum2는 모든 벤치마크에서 프론티어 수준의 역량이 아닌 **컴포넌트 역할에서의 효율성**을 위해 설계되었습니다. 후속 연구는 전문화 깊이와 도메인 일반화 폭 사이의 **파레토 최적 트레이드오프**를 탐색해야 합니다.

#### ② RLVR 일반화의 조건부 특성 이해
RLVR은 모델의 마지막 확률 계산에서 오류를 줄이고 상당한 정확도 향상을 생성하지만, **모델이 충분한 초기 역량을 가질 때만** 이러한 이점이 나타납니다. 따라서 연구자들은 RLVR 적용 전 베이스 모델의 역량 수준을 신중하게 평가해야 합니다.

#### ③ 수학·추론 격차 해소
Thinking 변형이 AIME 2025+2026에서 58.4%를 기록한 반면, Qwen3.5 4B는 68.3점을 기록했습니다. 이 격차를 해소하기 위해서는 수학 및 추론 데이터 비율, 합성 데이터 전략, 또는 추론 특화 RLVR 보상 설계 등에 대한 추가 연구가 필요합니다.

#### ④ 아이덴티티 정보와 합성 데이터 편향
정체성 데이터 없이 실행될 때, 모델은 합성 데이터 생성에 Google 모델이 사용되지 않았음에도 자신을 Google이 개발한 AI 어시스턴트로 일관되게 식별했습니다. 이는 합성 데이터 기반 학습 시 **편향 및 정체성 정렬** 문제를 향후 연구에서 반드시 다뤄야 함을 시사합니다.

#### ⑤ MoE 라우팅 효율성과 전문가 붕괴 방지
SFT 단계에서 MoE 보조 로드 밸런싱 계수를 $10^{-3}$에서 $10^{-4}$로 줄임으로써, 사전 학습 후 이미 균형이 잘 맞춰진 라우터에서 **과도한 전문가 활용 제약을 방지**했습니다. 향후 연구에서는 SFT나 RLVR의 좁은 분포에서 MoE 라우터의 동적 조정 전략을 더 깊이 탐구할 필요가 있습니다.

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

| 모델 / 연구 | 기관 | 핵심 기여 | Mellum2와의 관계 |
|---|---|---|---|
| **Gemma 3** (2025) | Google | Layer-selective YaRN 컨텍스트 확장 최초 보고 | Mellum2 장문맥 확장의 직접적 선행 연구 |
| **Moonlight / Muon** (arXiv:2502.16982) | — | Muon 옵티마이저의 LLM 스케일 적용 검증 | Mellum2 Muon 구성의 직접 채택 |
| **OLMo 3** (2025) | AI2 | Layer-selective YaRN 채택 | 동일 레시피 독립 검증 |
| **DeepSeek-V2/V3** (2024–2025) | DeepSeek | MoE + MLA 효율적 추론 | MoE 설계 참조 |
| **Debunk SFT Myth** (arXiv:2510.00237, 2025) | — | 적절한 데이터로 SFT도 RL 수준 일반화 가능 | Mellum2 SFT+RLVR 하이브리드의 이론적 배경 |
| **RLVR Limits** (arXiv:2510.27044, 2025) | — | RLVR의 일반화 한계: 새로운 전략 유도보다 기존 역량 안정화 | Mellum2 Thinking 모델의 RLVR 효과 평가 시 참조 |
| **RLVR Generalization** (arXiv:2512.20760, 2025) | — | RLVR의 일반화는 모델 크기·학습 레벨 조합에 조건부 | Mellum2 RLVR 포스트 트레이닝 설계의 근거 |

---

> ⚠️ **정확성 고지**: 본 답변은 arXiv:2605.31268 (Mellum2 Technical Report, 2026.05)의 공식 요약(Abstract) 및 PDF 원문 발췌, 그리고 JetBrains 공식 블로그와 신뢰할 수 있는 기술 매체의 인용에 기반합니다. 수식 중 일부(예: YaRN 스케일링, MoE 라우팅 일반 공식)는 논문에서 명시된 원리를 바탕으로 표준적인 형태로 재구성한 것이며, 논문 내 모든 하이퍼파라미터나 세부 수식이 완전히 공개되지 않을 수 있습니다.
