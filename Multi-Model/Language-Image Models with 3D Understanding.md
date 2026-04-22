
# Language-Image Models with 3D Understanding

> **논문 정보**
> - **제목**: Language-Image Models with 3D Understanding
> - **저자**: Jang Hyun Cho, Boris Ivanovic, Yulong Cao, Edward Schmerling, Yue Wang, Xinshuo Weng, Boyi Li, Yurong You, Philipp Krähenbühl, Yan Wang, Marco Pavone
> - **소속**: University of Texas at Austin, NVIDIA
> - **발표**: ICLR 2025
> - **arXiv**: [arXiv:2405.03685](https://arxiv.org/abs/2405.03685)
> - **프로젝트 페이지**: https://janghyuncho.github.io/Cube-LLM/

---

## 1. 🎯 핵심 주장 및 주요 기여 요약

Multi-modal Large Language Models(MLLMs)은 다양한 2D 비전 및 언어 태스크에서 뛰어난 성능을 보여 왔다. 이 논문은 MLLMs의 지각 능력을 3차원 공간에서의 이미지 grounding 및 추론으로 확장하는 것을 목표로 한다.

핵심 주장은 다음 세 가지로 요약된다:

| # | 핵심 주장 |
|---|-----------|
| 1 | **3D 특화 아키텍처 없이도 데이터 스케일링만으로 강력한 3D 지각 능력을 획득 가능** |
| 2 | **통합된 2D+3D 사전학습 데이터셋(LV3D)과 멀티턴 QA 포맷의 표준화** |
| 3 | **Chain-of-Thought(CoT) 기반 2D→3D 일반화 유도** |

이를 위해 다수의 기존 2D/3D 인식 데이터셋을 멀티턴 질의응답 형식으로 통합한 대규모 사전학습 데이터셋 **LV3D**를 구축하고, 이를 기반으로 새로운 MLLM인 **Cube-LLM**을 사전학습시켰다. 논문은 3D 특화 아키텍처 설계나 학습 목표 없이 순수 데이터 스케일링만으로 강력한 3D 지각 능력을 달성할 수 있음을 보인다.

---

## 2. 🔬 상세 분석: 문제 정의 · 방법 · 모델 구조 · 성능 · 한계

### 2-1. 해결하고자 하는 문제

이 연구의 목표는 2D와 3D 공간 모두에서 추론 가능한 MLLM 학습 프레임워크 개발이다. 논문은 3D 특화 아키텍처나 학습 목표 없이 순수 데이터 스케일링으로 이를 달성 가능함을 증명하고, **"어떤 태스크들이 2D→3D 일반화를 유도하는가"** 라는 핵심 질문에 집중한다.

기존 방법들의 문제점:
- 3D 특화 아키텍처(포인트 클라우드 인코더, 깊이 센서 등)에 의존
- 2D와 3D 태스크의 분리 학습 → 통합 추론 불가
- LiDAR 등 특수 센서 없이는 3D grounding이 어려움

---

### 2-2. 제안 방법 (수식 포함)

#### ① 데이터 표준화 (Data Standardization)

단일 2D+3D MLLM을 모든 가용 데이터 소스로부터 학습시키기 위해, 다양한 2D/3D grounding 태스크를 하나로 표준화하고, 모든 태스크를 **next token prediction**으로 정식화하며, 3D 추론을 멀티턴 대화로 형식화한다.

모든 태스크는 아래의 언어 모델 손실 함수로 학습된다:

$$\mathcal{L} = -\sum_{t=1}^{T} \log P_\theta(y_t \mid y_{<t}, \mathbf{x}_{img}, \mathbf{x}_{text})$$

여기서:
- $y_t$: $t$번째 출력 토큰 (텍스트 또는 박스 좌표)
- $\mathbf{x}_{img}$: 입력 이미지
- $\mathbf{x}_{text}$: 입력 텍스트 프롬프트
- $\theta$: 모델 파라미터

#### ② Task Scaling (태스크 스케일링)

3D 박스 레이블을 더 쉬운 하위 태스크(2D point, depth, size, orientation 등)로 분해하는 데이터 증강을 수행한다. 이를 통해 다양한 입출력 형식에 적응하도록 학습하고, 2D와 3D의 내재적 구조를 연결한다. 핵심적으로, 한 객체에 대해 쉬운 태스크(2D box)에서 어려운 태스크(3D box)로 이어지는 "단계적 QA 쌍"을 구성하여, MLLMs의 자기회귀적 특성을 통해 2D→3D 일반화를 직접 유도한다.

3D 바운딩 박스 표현은 다음과 같이 정의된다:

$$B_{3D} = (x_c, y_c, z_c, w, h, l, \theta)$$

여기서:
- $(x_c, y_c, z_c)$: 3D 공간에서의 중심 좌표
- $(w, h, l)$: 너비, 높이, 길이 (3D 크기)
- $\theta$: 회전각 (yaw angle)

BEV(Bird's Eye View) 공간에서의 IoU 기반 평균 정밀도(AP)는:

$$\text{AP}_{BEV} = \frac{1}{|R|} \sum_{r \in R} \max_{\hat{B}: \text{IoU}_{BEV}(\hat{B}, B^*) \geq r} \text{Precision}(\hat{B})$$

#### ③ Visual Chain-of-Thought (시각적 사고 사슬)

Cube-LLM은 자체 2D 예측 결과를 프롬프트로 활용하여 3D 추론 성능을 **자기 향상(self-improve)** 할 수 있다. 이 시각적 Chain-of-Thought 추론은 LLMs의 잘 알려진 동작 방식(Wei et al., 2022)을 닮았다.

CoT 추론 과정을 수식으로 나타내면:

$$P(B_{3D} \mid I, q) = \sum_{B_{2D}} P(B_{3D} \mid I, q, B_{2D}) \cdot P(B_{2D} \mid I, q)$$

즉, 모델은 먼저 2D 박스 $B_{2D}$를 예측하고, 이를 중간 추론 단계로 활용하여 최종 3D 박스 $B_{3D}$를 예측한다.

---

### 2-3. 모델 구조 (Cube-LLM)

Cube-LLM은 **LLaVA-1.5 아키텍처** 기반의 MLLM이다. 일반적인 아키텍처는 유지하되 두 가지 핵심 변경을 적용한다: (1) CLIP 비주얼 인코더를 **DINOv2**로 교체하고, (2) Vicuna-7B 언어 모델을 파인튜닝하되 비주얼 인코더는 동결한다. DINOv2는 텍스트 정렬 인코더가 아님에도 표준 VLM 벤치마크에서 성능 저하가 최소화되면서 3D 관련 태스크에서 현저한 개선을 보인다.

```
┌──────────────────────────────────────────────────────┐
│                    Cube-LLM 구조                      │
├─────────────┬──────────────────────┬─────────────────┤
│ 이미지 입력  │   DINOv2             │  Visual Encoder │
│             │   (CLIP 대체)         │  (Frozen)       │
├─────────────┼──────────────────────┤                 │
│ 텍스트 입력  │   MLP Projector      │  Alignment      │
│ (QA 형식)   │                      │  Layer          │
├─────────────┴──────────────────────┼─────────────────┤
│        Vicuna-7B LLM               │  Fine-tuned     │
│  (Next Token Prediction)           │                 │
├────────────────────────────────────┼─────────────────┤
│  출력: 텍스트 / 2D Box / 3D Box    │  Multi-format   │
└──────────────────────────────────────────────────────┘
```

학습 데이터(LV3D):

실내/실외의 다양한 2D/3D 비전 데이터셋을 조합하고 레이블을 일관된 포맷으로 표준화한 뒤 사용한다.

LV3D 전체 데이터셋은 약 **9.6M 이미지**를 포함한다.

---

### 2-4. 성능 향상

**Talk2Car 데이터셋**에서 Cube-LLM은 베이스라인 대비 BEV AP에서 **71.4 vs 50.1(+21.3)**, 3D AP에서 **64.1 vs 45.4(+18.7)**를 달성한다. **DriveLM 데이터셋**에서는 BEV AP가 **66.0 vs 33.2**로 baseline 대비 두 배 가까운 성능 향상을 보인다. 복잡한 드라이빙 추론 벤치마크에서도 **50.1 vs 32.4(+17.7)**를 기록한다.

일반 MLLM 벤치마크에서도 refCOCO/+/g에서 평균 **87.0점**, VQAv2, GQA, SQA, POPE 등 다양한 시각 질의응답 벤치마크에서 경쟁력 있는 성능을 보인다.

| 벤치마크 | Cube-LLM | 이전 SOTA | 향상폭 |
|---------|---------|---------|--------|
| Talk2Car (BEV AP) | 71.4 | 50.1 | **+21.3** |
| Talk2Car (3D AP) | 64.1 | 45.4 | **+18.7** |
| DriveLM (전체 점수) | 50.1 | 32.4 | **+17.7** |
| refCOCO/+/g (avg.) | 87.0 | - | SOTA |

---

### 2-5. 한계점

Cube-LLM은 여러 한계를 가진다. 첫째, **resampling 방법을 사용하지 않아** 비전 토큰 수를 줄이지 못하며, 이로 인해 입력 해상도 증가에 제약이 생긴다.

또한, 두 유사한 객체가 인접한 경우(예: 은색 세단 vs 흰색 세단) 속성 인식에 어려움을 겪는다.

벤치마크가 주로 야외 장면 및 자율주행 시나리오에 집중되어 있어, 실내 장면 이해나 3D 물체 감지와 같은 광범위한 3D 지각 태스크에 대한 검증이 부족하다.

또한 Cube-LLM의 계산 및 메모리 요구 사항에 대한 논의가 부족하여 실제 배포 환경에서의 고려가 미흡하다.

---

## 3. 🌐 모델의 일반화 성능 향상 가능성

논문은 3D 특화 아키텍처 설계나 학습 목표 없이 **순수 데이터 스케일링**으로 목표를 달성할 수 있음을 증명하며, 핵심 질문인 **"어떤 태스크가 2D→3D 일반화를 유도하는가"** 에 집중한다.

### 3-1. 데이터 다양성을 통한 일반화

LV3D는 실내/실외의 2D 및 3D 비전 데이터셋의 다양한 컬렉션을 결합하고 레이블을 표준화하며, 이를 MLLM 학습을 위한 일련의 질의응답 쌍으로 블렌딩한다. 이러한 **다중 도메인 혼합 학습**은 특정 데이터셋이나 장면 유형에 대한 과적합을 방지하고 일반화 성능을 높이는 핵심 메커니즘이다.

### 3-2. Visual Chain-of-Thought를 통한 2D→3D 일반화

모델은 쉬운 태스크(2D box)에서 어려운 태스크(3D box)로 이어지는 단계적 QA 쌍을 학습함으로써, **MLLMs의 자기회귀적 특성을 통한 2D→3D 일반화**가 직접적으로 유도된다.

이는 2D 시각 정보가 풍부한 인터넷 데이터로 학습된 기반 모델의 지식을 3D 공간으로 전이(transfer)하는 핵심 메커니즘이다:

$$P(B^{3D} \mid I, q) \approx P(B^{3D} \mid I, q, \hat{B}^{2D}) \cdot P(\hat{B}^{2D} \mid I, q)$$

### 3-3. 데이터 스케일링 법칙 (Scaling Law)

Cube-LLM은 실내/실외 장면 grounding 및 자율주행 QA와 같은 복잡한 추론 태스크 모두에서 2D/3D 데이터 스케일링에 따른 **현저한 성능 향상**을 보인다.

이를 스케일링 법칙으로 표현하면:

$$\text{Performance}(\mathcal{D}) \propto \log |\mathcal{D}|$$

여기서 $|\mathcal{D}|$는 학습 데이터 크기를 나타내며, 이는 특화된 3D 아키텍처 없이도 데이터 양만으로 성능이 향상됨을 의미한다.

### 3-4. 전문가 모델과의 협력적 일반화 (Specialist Prompting)

Cube-LLM은 다양한 입출력 형식과 질문에 적응하는 **instruction following 능력**을 갖추고 있으며, LiDAR 등 추가 모달리티를 가진 전문가 모델의 예측을 질문에 추가하는 방식으로 어떠한 전문가 모델과도 연동될 수 있다.

---

## 4. 🔭 앞으로의 연구에 미치는 영향 및 고려 사항

### 4-1. 연구에 미치는 영향

**① 데이터 중심 패러다임의 확립**

이 연구는 대형 언어 모델이 2D와 3D 모두에서 강력한 멀티모달 지각 및 추론 시스템으로 기능할 수 있음을 강조한다. 3D 특화 아키텍처 설계보다 **고품질 데이터 큐레이션**이 더 중요하다는 패러다임 전환을 이끈다.

**② 자율주행·로보틱스 분야로의 파급**

Cube-LLM은 3D grounded reasoning에서 Talk2Car 데이터셋 AP-BEV 기준 21.3점, DriveLM에서 복잡한 추론 17.7점의 상당한 성능 향상을 달성했다. 이는 자율주행의 언어-공간 통합 이해 연구를 가속화한다.

**③ 후속 연구들의 기반**

LocateAnything3D와 같은 후속 연구들은 Cube-LLM의 전체 LV3D 데이터셋(약 9.6M 이미지) 학습 결과와 비교하며, 훨씬 작은 1.7M 이미지 데이터셋으로도 모든 데이터셋과 지표에서 성능을 크게 능가하는 Chain-of-Sight 방법론을 제안했다. 이는 Cube-LLM이 후속 연구의 **표준 베이스라인**으로 자리잡았음을 보여준다.

---

### 4-2. 향후 연구 시 고려할 점

#### ① 벤치마크 다양화

일부 3D 태스크는 2D VLM만으로도 쉽게 해결될 수 있어 진정한 3D 능력을 평가하지 못하는 **"2D-Cheating"** 현상이 존재한다. 따라서 진정한 3D 공간 이해를 측정하는 새로운 벤치마크 설계가 필요하다.

#### ② 실내 시나리오로의 확장

현재 벤치마크가 주로 야외/자율주행 시나리오에 집중되어 있어, 실내 장면 이해나 3D 물체 감지 등 더 광범위한 태스크에서의 성능을 검증할 필요가 있다.

#### ③ 효율적 토큰 압축

Cube-LLM은 비전 토큰 수를 줄이는 resampling 방법을 채용하지 않아 입력 해상도 증가에 제약이 있다. 향후 연구는 토큰 압축 기법(예: Q-Former, Perceiver)을 결합하여 고해상도 입력을 효율적으로 처리하는 방향을 고려해야 한다.

#### ④ 구조화된 3D 장면 표현

S2-MLLM과 같은 후속 연구에서는 피드포워드 재구성을 통한 공간 가이던스와 구조 강화 모듈을 통합하여, 추론 시 명시적 포인트 클라우드 재구성 없이도 3D 장면을 이해하고 잠재적 공간 추론을 수행하는 방향이 제안된다.

#### ⑤ 멀티뷰 및 비디오 확장

단일 이미지에서의 3D 이해를 넘어, 멀티뷰 이미지나 비디오 시퀀스를 활용한 시간적·공간적 3D 추론으로 확장하는 연구가 필요하다.

---

## 5. 📊 2020년 이후 관련 최신 연구 비교 분석

| 연구명 | 연도 | 입력 형태 | 핵심 방법 | 3D 특화 아키텍처 |
|-------|------|----------|----------|----------------|
| **3D-LLM** (Hong et al.) | 2023 | 포인트 클라우드 + 이미지 | 3D 특징을 LLM에 주입 | ✅ 필요 |
| **PointLLM** (Xu et al.) | 2023 | 포인트 클라우드 | 포인트 인코더 + LLM 정렬 | ✅ 필요 |
| **SpatialVLM** (Chen et al.) | 2024 | 단일 이미지 | 공간 추론 데이터 생성 | ❌ 불필요 |
| **Cube-LLM** (Cho et al.) | 2024 | 단일 이미지 | LV3D + CoT + 데이터 스케일링 | ❌ 불필요 |
| **LLMI3D** (Yang et al.) | 2024 | 단일 이미지 | 기하학적 투영 기반 3D 추론 | ✅ 부분 |
| **LocateAnything3D** | 2025 | 단일 이미지 | Chain-of-Sight + 소규모 데이터 | ❌ 불필요 |
| **S2-MLLM** | 2025 | 멀티뷰 RGB-D | 구조 강화 모듈 + 피드포워드 재구성 | ✅ 필요 |

**PointLLM**은 색상이 있는 포인트 클라우드를 이해할 수 있는 MLLM으로, 모호한 깊이, 가림, 시점 의존성에 대한 걱정 없이 객체 유형, 기하학적 구조 및 외관을 지각한다. PointLLM은 포인트 클라우드라는 명시적 3D 입력이 필요하여 단일 이미지만으로 3D를 이해하는 Cube-LLM과 대조적이다.

**LocateAnything3D**는 Chain-of-Sight 방식으로 학습하며, Cube-LLM보다 적은 데이터임에도 세 데이터셋과 두 지표 모두에서 Cube-LLM을 크게 능가한다. 이는 Cube-LLM의 **단순 데이터 스케일링 전략의 한계**를 시사하며, 데이터 질과 구조가 양보다 중요할 수 있음을 보여준다.

현재 3D LLMs의 초기 발전 단계를 감안할 때, 이들을 평가하기 위한 전용 대표 벤치마크가 아직 존재하지 않는다. 이는 Cube-LLM을 포함한 모든 3D MLLM 연구의 **공통 과제**다.

---

## 📚 참고 자료 및 출처

1. **논문 원문 (arXiv)**: [arXiv:2405.03685](https://arxiv.org/abs/2405.03685) — "Language-Image Models with 3D Understanding", Cho et al., 2024
2. **ICLR 2025 OpenReview**: [openreview.net/forum?id=yaQbTAD2JJ](https://openreview.net/forum?id=yaQbTAD2JJ)
3. **프로젝트 페이지**: [janghyuncho.github.io/Cube-LLM](https://janghyuncho.github.io/Cube-LLM/)
4. **NVIDIA AVG 연구 페이지**: [research.nvidia.com/labs/avg/publication/cho.ivanovic.etal.iclr2025](https://research.nvidia.com/labs/avg/publication/cho.ivanovic.etal.iclr2025/)
5. **AI Models FYI 분석 페이지**: [aimodels.fyi/papers/arxiv/language-image-models-3d-understanding](https://www.aimodels.fyi/papers/arxiv/language-image-models-3d-understanding)
6. **LocateAnything3D (후속 연구)**: [arXiv:2511.20648](https://arxiv.org/html/2511.20648)
7. **PointLLM (비교 연구)**: [arXiv:2308.16911](https://arxiv.org/html/2308.16911)
8. **S2-MLLM (비교 연구)**: [arXiv:2512.01223](https://arxiv.org/html/2512.01223)
9. **Revisiting 3D LLM Benchmarks**: [arXiv:2502.08503](https://arxiv.org/html/2502.08503v3)
10. **Semantic Scholar 논문 페이지**: [semanticscholar.org — Language-Image Models with 3D Understanding](https://www.semanticscholar.org/paper/Language-Image-Models-with-3D-Understanding-Cho-Ivanovic/5de2dd75319f5bdfd4ba68ce61eaaaaa532dfc8d)
