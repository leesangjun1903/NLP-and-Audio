# Plan, Watch, Recover: A Benchmark and Architectures for Proactive Procedural Assistance

# 1. 핵심 주장과 주요 기여 (요약)

이 논문은 **절차적 과제(procedural task)** 수행 중 사용자에게 실시간으로 단계별 안내를 제공하되, **언제(when) 개입**하고 **어떻게(how) 코칭**할지를 자율적으로 결정하는 **능동적(proactive) 멀티모달 어시스턴트 시스템**을 제안합니다.

그러나 기존 연구는 현실적 조건—특히 사용자가 예상 단계 순서에서 이탈(deviation)하는 일반적인 경우—을 반영하는 대규모 크로스 도메인 벤치마크의 부재로 인해 발전이 제한되어 왔습니다.

**4대 기여(Contributions)**:


1. **EgoProactive**: 명시적 Out-of-Plan(OOP) 어노테이션과 복구(recovery) 단계를 포함하는 대규모 웨어러블-에고센트릭 데이터셋 공개
2. **Pro²Bench**: 기존 5개 벤치마크(Ego4D, EPIC-KITCHENS, EgoExo4D, HoloAssist, HowTo100M)를 통합 proactive-guidance 스키마로 재구성
3. **Decoupled Planner–Interaction Architecture**: 절차적 상태(procedural state), 시각 단서(visual cues), 복구 주입(recovery injection)에 특화된 분리형 아키텍처
4. **모델 간 전이 가능한 포스트 트레이닝 레시피**: Llama 4 및 Qwen-3.6-VL에 대한 교차 백본 복제로 검증


실험 결과, 학습된 Llama-4 시스템은 강력한 독점 모델 베이스라인(Claude Opus 4.6, Gemini 3.1 Pro, GPT 5.2)과 오픈 웨이트 베이스라인(Qwen3 VL 235B) 모두를 6개 전체 데이터셋에서 능가했습니다.

---

# 2. 세부 분석

## 2.1 해결하고자 하는 문제

이 논문은 능동적 어시스턴트 시스템 배포를 가로막는 **3가지 핵심 장벽(Barriers)** 을 식별합니다:
- **(B1)** 사용자 이탈을 다루는 벤치마크 부재: 기존 HoloAssist, ProAssist 등은 표준적(canonical) 실행만 가정하지만, 실제 사용자는 단계를 건너뛰고, 재배열하고, 대체하고, 실수하고, 복구하는(OOP) 행동을 보임
- **(B2)** 실시간 상호작용과 장기 계획의 상충되는 연산 예산: 프레임 단위 스트리밍 판단 vs. 드문 결정 시점에서의 심층 추론 간 트레이드오프
- **(B3)** 포스트 트레이닝 성과의 모델 특수성: 기존 레시피가 단일 아키텍처에 묶여 있음


## 2.2 제안하는 방법 및 모델 구조

### (A) 벤치마크: EgoProactive + Pro²Bench

**Pro²Bench**는 새로 수집된 EgoProactive 데이터셋과 5개 기존 공개 코퍼스를 단일 어노테이션 스키마로 통합합니다. EgoProactive는 소비자용 스마트 글래스로 촬영되었으며, OOP 이탈에 대한 복구 지도(recovery supervision)를 고유하게 제공합니다.

EgoProactive는 4개 활동 도메인(요리, 공예, DIY, 튜토리얼)에 걸친 700개 비디오 녹화로 구성되며, OOP 시나리오·실수·이탈과 쌍을 이루는 복구 발화(recovery utterance)를 포함합니다.

### (B) 아키텍처: Decoupled Planner–Interaction Architecture (PWR)

**Plan, Watch, Recover (PWR)** 는 시스템·벤치마크·레시피를 아우르며, 3가지 장벽 모두를 해결합니다.

PWR 아키텍처의 핵심 설계 원리는 **monolithic 모델을 분리(decouple)** 하는 것입니다:

| 모듈 | 역할 | 연산 특성 |
|------|------|----------|
| **Planner** | 절차적 상태 추적, 장기 계획 수립, OOP 감지, 복구 계획 생성 | 드문 결정 시점에서 깊은 추론 |
| **Interaction Module** | 스트리밍 비디오에서 실시간 시각 단서 처리, 사용자에게 적시 개입 결정 | 프레임 단위 빠른 판단 |

이 분리 구조는 (B2)에서 지적한 실시간 처리와 장기 계획 간의 연산 예산 충돌 문제를 구조적으로 해결합니다.

### (C) 포스트 트레이닝 레시피 및 수식

논문은 교차 백본(cross-backbone) 전이 가능한 포스트 트레이닝 레시피를 제안합니다. 검색된 정보에 기반하여 추론할 수 있는 핵심 평가 메트릭은 다음과 같습니다:

**G-Mean (Geometric Mean)** — 개입의 정밀도(Precision)와 재현율(Recall)의 기하 평균으로, 능동적 개입 품질을 측정합니다:

$$G\text{-}Mean = \sqrt{Precision \times Recall}$$

이는 개입 시점 결정(when to intervene)과 개입 내용 품질(how to coach)을 동시에 평가하는 데 사용됩니다.

전체 학습 목표는 다음과 같이 표현될 수 있습니다:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{planner}} + \lambda \cdot \mathcal{L}_{\text{interaction}}$$

여기서 $\mathcal{L}\_{\text{planner}}$는 절차적 상태 추적·OOP 감지·복구 계획에 대한 손실, $\mathcal{L}_{\text{interaction}}$은 실시간 개입 결정에 대한 손실이며, $\lambda$는 두 모듈 간 가중치 균형 파라미터입니다.

> ⚠️ **주의**: 위 수식은 논문의 아키텍처 설명에서 논리적으로 추론한 것이며, 논문 원문의 정확한 수식 표기와 다를 수 있습니다. 원문 PDF를 통해 정확한 수식을 확인하시기 바랍니다.

## 2.3 성능 향상

Pro²Bench만으로 모놀리식 모델을 파인튜닝하면, 평균 G-Mean이 Llama 4에서 +0.22, Qwen3.6에서 +0.14 향상되었으며, HowTo 분할은 0.21에서 0.69로 크게 상승했습니다.

분리형 아키텍처는 별도의 추가 이득을 제공하며, 트레이닝 후에만 그 효과가 나타납니다. 동일한 monolithic-to-duplex 분리가 제로샷에서는 변화가 없었지만, 두 모듈 모두 파인튜닝된 후에는 Qwen3.6이 0.45에서 0.83 G-Mean으로 상승했습니다.

모놀리식 파인튜닝(0.45)과 제로샷 분리(0.49) 각각은 전체 PWR(0.83)에 비해 약 0.35 부족하여, 데이터도 아키텍처도 단독으로는 불충분함을 보여줍니다.

최종 시스템은 Qwen3.6에서 0.83, Llama 4에서 0.76 평균 G-Mean을 달성하여, 두 오픈 웨이트 백본 모두 프론티어 제로샷 수준 이하에서 시작했음에도 모든 베이스라인을 초과했습니다.

## 2.4 한계

검색된 정보에 기반한 주요 한계점은 다음과 같습니다:

1. **OOP 시나리오의 범위**: EgoProactive가 4개 도메인(요리, 공예, DIY, 튜토리얼)을 다루지만, 의료·산업·안전 등 고위험 도메인으로의 확장은 검증되지 않았습니다.
2. **실시간 배포 제약**: 분리형 아키텍처가 연산 예산 문제를 완화하지만, 엣지 디바이스(스마트 글래스)에서의 실제 실시간 추론 지연(latency)에 대한 구체적 분석이 필요합니다.
3. **포스트 트레이닝 레시피의 범용성**: 기존 레시피가 모델에 특수화되어 있다는 문제를 해결하고자 했지만, 검증된 백본은 Llama 4와 Qwen-3.6-VL 두 가지입니다.

---

# 3. 모델의 일반화 성능 향상 가능성

이 논문에서 **일반화(generalization)** 와 관련된 핵심 설계 및 결과는 다음과 같습니다:

### 3.1 교차 백본 일반화 (Cross-Backbone Generalization)

포스트 트레이닝 레시피가 모델 패밀리 간 전이됨을 Llama 4와 Qwen-3.6-VL에 대한 교차 백본 복제로 검증했습니다. 이는 (B3) 장벽을 직접 해결한 것으로, 특정 모델에 종속되지 않는 범용적 학습 레시피의 가능성을 입증합니다.

### 3.2 교차 도메인 일반화 (Cross-Domain Generalization)

저자들은 "Robustness across backbones, unseen domains and unscripted deviations"를 강조합니다.

- **6개 이질적 데이터셋** 통합(Pro²Bench)은 모델이 단일 도메인에 과적합하는 것을 방지
- **EgoProactive의 4개 도메인**(요리, 공예, DIY, 튜토리얼)과 5개 기존 벤치마크의 다양한 시나리오 결합
- HowTo100M 같은 대규모 instructional 비디오 코퍼스 포함으로 long-tail 분포 학습

### 3.3 OOP(Out-of-Plan) 강건성

실제 사용자는 단계를 건너뛰고, 재배열하고, 대체하고, 실수하고, 복구하는 행동을 보이며, 기존 데이터셋은 이를 다루지 않았습니다. PWR 시스템은 이러한 **비표준적 실행 경로**에서도 강건하게 복구 안내를 생성할 수 있도록 설계되었으며, 이것이 핵심적인 일반화 차원입니다.

### 3.4 데이터와 아키텍처의 상보적 관계

데이터와 아키텍처는 상보적 요인이며, 모놀리식 파인튜닝(0.45)도 제로샷 분리(0.49)도 단독으로는 전체 PWR(0.83)에 크게 미달합니다. 이는 일반화 성능 향상이 **데이터 다양성** + **구조적 분리** + **통합 학습 레시피**의 삼위일체에서 비롯됨을 시사합니다.

### 3.5 향후 일반화 개선 방향

| 방향 | 접근법 | 기대 효과 |
|------|--------|----------|
| **도메인 확장** | 의료, 제조, 안전 등 고위험 도메인 추가 | 도메인 불변 표현 학습 |
| **백본 다양화** | 3개 이상 모델 패밀리 검증 | 레시피 범용성 확인 |
| **OOP 유형 확장** | 더 복잡한 이탈 패턴(다중 연쇄 오류 등) | 복구 플래닝 강건성 |
| **Few-shot 적응** | 새 절차에 대한 소량 데이터 적응 | 빠른 도메인 전이 |

---

# 4. 향후 연구에 미치는 영향과 고려 사항

## 4.1 연구 커뮤니티에 미치는 영향

1. **벤치마크 표준화**: Pro²Bench는 6개 이질적 데이터셋을 통합 스키마로 재구성함으로써, 향후 능동적 어시스턴트 연구의 **표준 평가 프레임워크**로 자리잡을 가능성이 높습니다.

2. **OOP를 기본 전제로**: 이 논문은 사용자 이탈을 "예외"가 아닌 "기본 가정"으로 전환시킵니다. 향후 절차적 보조 연구는 반드시 OOP 시나리오를 포함해야 하는 방향으로 패러다임이 전환될 것입니다.

3. **모놀리식 → 분리형 아키텍처 패러다임**: 실시간 처리와 장기 추론의 명시적 분리는 로보틱스, AR/VR, 산업 자동화 등 다양한 분야로 확산될 수 있는 설계 원칙입니다.

4. **교차 백본 레시피**: 모델 특수적이지 않은 포스트 트레이닝 방법론은 빠르게 진화하는 LLM/VLM 생태계에서 매우 중요한 실용적 가치를 제공합니다.

## 4.2 향후 연구 시 고려할 점

| 고려 사항 | 세부 내용 |
|-----------|----------|
| **안전성(Safety)** | 고위험 절차(의료 수술, 화학 실험)에서 잘못된 복구 안내의 위험성 평가 필요 |
| **지연 시간(Latency)** | 엣지 디바이스에서의 실시간 추론 가능성과 품질 간 트레이드오프 정량화 |
| **사용자 연구** | 자동 메트릭(G-Mean)과 실제 사용자 만족도·과제 완수율 간의 상관관계 검증 |
| **개인화** | 사용자별 숙련도·선호도에 따른 개입 전략 적응 |
| **다국어·다문화** | 절차적 지식의 문화적 차이 반영 |
| **프라이버시** | 웨어러블 에고센트릭 데이터의 개인정보 보호 |

---

# 5. 2020년 이후 관련 최신 연구 비교 분석

아래 표는 2020년 이후 능동적/절차적 보조(Proactive Procedural Assistance) 분야의 주요 연구를 비교합니다:

| 연구 | 연도 | 핵심 접근 | OOP 처리 | 실시간 | 벤치마크 | 주요 차별점 |
|------|------|----------|----------|--------|----------|-----------|
| **HoloAssist** | 2023 | 혼합현실 기반 원격 보조 | ✗ (canonical) | △ | 자체 | MR 환경 특화 |
| **ProAssist** (Zhang, 2025) | 2025 | 스트리밍 에고센트릭 비디오 기반 최초 E2E 시스템 | ✗ | ✓ | 자체 | "대화 품질은 인식이 아닌 절차적 추론에 병목" |
| **ProAct-75** (Zhu et al., 2026) | 2026 | 태스크 그래프 기반 구조적 의사결정, 엔트로피 기반 휴리스틱 탐색 | △ (병렬 실행) | △ | ProAct-75 (75 tasks, 91K annotations) | 병렬 스레드 독립 실행 가능 |
| **Pro²Assist** (Xu et al., 2026) | 2026 | AR 글래스 기반 멀티스케일 시간적 동적, 단계 인식 추론 | △ | ✓ | 자체 | 절차적 행동 이해 정확도 21%↑, 타이밍 정확도 2.29× |
| **PrISM-Q&A** (Arakawa, 2024) | 2024 | 오디오+IMU 기반 웨어러블 LLM 보조 | △ | ✓ | 자체 | 스마트워치 센서 활용 |
| **VLWM** (2025) | 2025 | Vision-Language World Model 기반 절차적 계획 | ✗ | ✗ | COIN, VPA | WorldPrediction 45% SOTA |
| **PWR (본 논문)** | 2026 | 분리형 플래너-인터랙션, 교차 백본 레시피, OOP 복구 | **✓ (명시적)** | ✓ | **Pro²Bench (6개 통합)** | 0.83 G-Mean (Qwen3.6), 프론티어 모델 전체 능가 |

### 핵심 비교 분석

1. **OOP 처리**: PWR은 OOP를 **명시적 어노테이션과 복구 메커니즘**으로 다루는 **최초의** 대규모 벤치마크를 제공합니다. 다른 연구들은 표준 실행을 가정하거나 제한적으로만 이탈을 처리합니다.

2. **벤치마크 규모와 통합성**: Pro²Bench의 6개 데이터셋 통합은 기존 어떤 단일 벤치마크보다 포괄적이며, 교차 도메인 일반화 평가를 가능케 합니다.

3. **아키텍처 혁신**: ProAssist가 모놀리식 스트리밍을 사용하는 반면, PWR은 **기능적 분리**를 통해 실시간성과 추론 깊이를 동시에 확보합니다.

4. **교차 백본 전이**: 다른 연구들이 특정 모델에 특화된 반면, PWR은 동일 레시피가 Llama 4와 Qwen-3.6-VL 모두에서 효과적임을 입증하여, 빠르게 변화하는 LLM/VLM 생태계에서 실용적 우위를 확보합니다.

---

# 참고 자료 출처

1. **[주 논문]** Kundu, K. et al., "Plan, Watch, Recover: A Benchmark and Architectures for Proactive Procedural Assistance," arXiv:2606.04970, June 2026. — https://arxiv.org/abs/2606.04970 / https://arxiv.org/html/2606.04970
2. **[Pro²Assist]** Xu, L. et al., "Pro²Assist: Continuous Step-Aware Proactive Assistance with Multimodal Egocentric Perception for Long-Horizon Procedural Tasks," arXiv:2605.04227, May 2026. — https://arxiv.org/abs/2605.04227
3. **[ProAct]** Zhu, X. et al., "ProAct: A Benchmark and Multimodal Framework for Structure-Aware Proactive Response," arXiv:2602.03430, February 2026. — https://arxiv.org/pdf/2602.03430
4. **[VLWM]** "Planning with Reasoning using Vision Language World Model," arXiv:2509.02722, September 2025. — https://arxiv.org/html/2509.02722v1
5. **[PrISM-Q&A]** Arakawa, R., "PrISM-Q&A: Step-Aware Question Answering with LLMs Enabled by Multimodal Procedure Tracking," IMWUT 2024.
6. **[Proactive Conversational Assistant]** "Proactive Conversational Assistant for a Procedural Manual Task based on Audio and IMU," arXiv:2602.15707, February 2026. — https://arxiv.org/html/2602.15707v1

> **면책 조항**: 본 분석의 수식 부분(특히 $\mathcal{L}_{\text{total}}$)은 논문의 아키텍처 설명에서 논리적으로 추론한 것입니다. 정확한 수식·하이퍼파라미터·실험 설정은 원문 PDF(https://arxiv.org/pdf/2606.04970)를 직접 참조하시기 바랍니다.
