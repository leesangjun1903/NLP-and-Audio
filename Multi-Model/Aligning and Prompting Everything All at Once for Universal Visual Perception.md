
# APE: Aligning and Prompting Everything All at Once for Universal Visual Perception

> **논문 정보**
> - **제목:** Aligning and Prompting Everything All at Once for Universal Visual Perception
> - **저자:** Yunhang Shen, Chaoyou Fu, Peixian Chen, Mengdan Zhang, Ke Li, Xing Sun, Yunsheng Wu, Shaohui Lin, Rongrong Ji
> - **학회:** CVPR 2024
> - **arXiv:** [2312.02153](https://arxiv.org/abs/2312.02153)
> - **코드:** [github.com/shenyunhang/APE](https://github.com/shenyunhang/APE)

---

## 1. 핵심 주장 및 주요 기여 요약

### 1.1 핵심 주장

APE는 이미지 내 모든 것을 한 번에 정렬하고 프롬프팅하는 범용 시각 인식 모델로, Detection(탐지), Segmentation(분할), Grounding(접지)을 **인스턴스 수준의 문장-객체 매칭 패러다임(instance-level sentence-object matching paradigm)**으로 수행한다.

즉, 기존의 개별 task 전용 모델이나 무거운 교차 모달(cross-modality) 상호작용 기반 모델의 한계를 극복하고, **단일 가중치 세트(one-suit of weights)**로 수백 개의 데이터셋에서 범용 시각 인식을 수행한다는 것이 핵심 주장이다.

### 1.2 주요 기여 (Contributions)

APE의 핵심 기여는 다음과 같다: APE는 이미지 내에서 모든 것을 한 번에 정렬·프롬프팅하는 범용 시각 인식 모델로, 대규모 데이터에서 학습되어 **task별 fine-tuning 없이** SotA 성능을 제공한다.

구체적으로는:

1. **Detection-Grounding 통합:** 언어 기반 grounding을 open-vocabulary detection으로 재공식화하여, 교차 모달 융합의 효과를 유지하면서 수천 개의 카테고리 어휘와 영역 설명으로 모델 프롬프팅을 효율적으로 확장한다.

2. **Thing-Stuff 균등화:** 픽셀 수준 태스크의 입도(granularity) 격차를 해소하기 위해, APE는 어떤 고립된 영역도 개별 인스턴스로 간주하는 **proxy instance learning**을 통해 semantic과 panoptic segmentation을 동등하게 취급한다.

3. **대규모 정렬:** APE는 자연스럽고 도전적인 특성을 가진 광범위한 데이터에서 시각·언어 표현을 한 번에 정렬하며, task별 fine-tuning 없이 160개 이상의 데이터셋에서 최신 모델과 동등하거나 그 이상의 성능을 달성했다.

---

## 2. 상세 분석: 문제 정의 → 방법론 → 구조 → 성능

### 2.1 해결하고자 하는 문제

**문제 1 — Object-Word Alignment의 한계:**

인스턴스 수준 태스크를 object-word alignment로 처리하는 지배적 패러다임은 무거운 교차 모달 상호작용을 발생시키며, 이는 객체 탐지 및 visual grounding의 프롬프팅에 비효율적이다.

**문제 2 — Things vs. Stuff의 Granularity Gap:**

픽셀 수준 태스크에 집중하는 또 다른 연구 흐름은 things(전경 객체)와 stuff(배경 클래스)의 큰 annotation 격차를 갖고 있으며, 전경 객체 분할과 배경 클래스 분할 사이의 상호 간섭 문제를 겪는다.

**문제 3 — 기존 모델의 확장성 한계:**

UNINEXT, GroundingDINO 등은 깊은 region-word fusion을 통해 visual grounding 문제로 공식화하기 때문에, 많은 수의 카테고리와 grounding phrase를 동시에 처리하는 것이 불가능하다.

---

### 2.2 제안하는 방법론 (수식 포함)

#### (A) Region-Sentence Alignment (영역-문장 정렬)

기존 MDETR 등은 단어 수준의 token matching인 **Object-Word Alignment**을 사용했지만, APE는 이를 **문장 수준의 매칭**으로 업그레이드한다.

MDETR은 각 매칭된 객체에 대해 텍스트 프롬프트에서 토큰 범위를 예측하는 단어-영역 정렬(word-region alignment)을 처음 제안했다. APE는 주어진 이미지와 대응 텍스트 프롬프트에 대해 각각 비전·텍스트 임베딩을 추출하고, 단어 수준 텍스트 임베딩 $P$를 문장 수준 임베딩 $\bar{P}$로 집계한다.

수식으로 표현하면:

$$
\bar{P} = \text{Aggregate}(P_1, P_2, \ldots, P_N) \quad \text{(Word-level → Sentence-level)}
$$

교차 모달 인코더는 두 모달리티의 정보를 융합하여 텍스트 쿼리로 객체 쿼리 $O$를 조건화하고 텍스트 임베딩 $\hat{P}$를 업데이트한다. 트랜스포머 디코더는 객체 쿼리 $O$로부터 최종 객체 임베딩 $\hat{O}$를 생성하며, 시각-언어 정렬 모듈이 영역과 프롬프트의 올바른 쌍을 예측한다.

$$
\hat{V}_{set}, \hat{P}_{set} = \text{GatedFusion}(V, \bar{P}_{set})
$$

$$
\hat{O} = \text{TransformerDecoder}(O, \hat{V}_{set}, \hat{P}_{set})
$$

#### (B) Gated Cross-modality Interaction (게이트 교차 모달 상호작용)

계산 복잡도와 메모리 사용량을 줄이기 위해 단어 수준 표현을 문장 수준 프롬프트 임베딩으로 압축하는 **Gated Cross-modality Interaction**을 제안한다. 기존의 deep fusion은 일반적으로 수천 개의 어휘를 학습해야 하는 open-vocabulary detection에서 많은 입력 요소에 대한 multi-head attention을 수반하며, APE는 서로 다른 유형의 프롬프트들이 시각-언어 융합을 하는 방식을 제한하기 위해 게이트 교차 모달 상호작용을 추가로 제안한다.

게이트 기반 융합의 핵심 이점:

게이트 융합은 두 가지 장점을 지닌다: (1) 학습 및 추론 중 단일 순전파(single forward)로 수천 개의 detection 카테고리를 모델링하고 수백 개의 grounding 문장을 융합할 수 있다. (2) 이전 연구는 deep fusion 모듈로 detection 데이터를 학습하면 새로운 카테고리에 대한 zero-shot 일반화가 저하될 수 있음을 보였는데, 게이트 상호작용은 detection 태스크에 대한 fusion을 명시적으로 금지함으로써 이러한 퇴보를 방지한다.

수식으로:

$$
g = \sigma(W_g \cdot [V; \bar{P}]) \quad \text{(게이트 벡터)}
$$

$$
\hat{V} = g \odot \text{Attention}(V, \bar{P}) + (1-g) \odot V
$$

여기서 $\sigma$는 sigmoid 함수, $\odot$는 element-wise 곱이다.

#### (C) Image-Centric Grounding Format

기존 방법들은 $\{I, T, B\}$ 형태의 grounding 샘플을 구성했는데, 여기서 $I$는 이미지, $T$는 인스턴스를 설명하는 phrase, $B$는 대응 bounding box이다. 이 region-centric 형식은 각 샘플에서 단일 인스턴스만 학습하기 때문에 비효율적이다. APE는 학습·추론 시 단일 순전파에서 여러 phrase prompt를 처리할 수 있으므로, grounding 샘플을 image-centric 형식으로 수집한다.

$$
\{I, (T_1, B_1), (T_2, B_2), \ldots, (T_n, B_n)\}
$$

이 새로운 image-centric 형식은 기존 region-centric 형식 대비 학습 반복 횟수를 **92배 가속**한다.

#### (D) Thing-Stuff Equalizing (TSE)

픽셀 수준 태스크의 granularity 불일치를 해소하기 위한 핵심 기법으로, 수식으로 정의하면:

$$
\mathcal{L}_{TSE} = \mathcal{L}_{instance}(\hat{O}, G_{things}) + \mathcal{L}_{instance}(\hat{O}, G_{stuff})
$$

여기서 $G_{things}$와 $G_{stuff}$ 모두 동일한 instance 학습 파이프라인으로 처리된다.

실험 결과, APE의 통합(unification)이 PQ와 mIoU 측면에서 segmentation 성능을 유의미하게 향상시킴을 보여준다.

---

### 2.3 모델 구조

APE는 이미지 특징 추출을 위한 **비전 백본(vision backbone)**, 텍스트 특징 추출을 위한 **언어 모델(language model)**, 교차 모달 deep fusion을 갖춘 **트랜스포머 인코더**, 그리고 **트랜스포머 디코더**로 구성된다.

구조 다이어그램 요약:

```
입력: 이미지 I + 텍스트 프롬프트 집합 {T₁, T₂, ..., Tₙ}
         ↓                    ↓
   Vision Backbone        Language Model
   (EVA-02 ViT-L)       (EVA-CLIP BERT-like)
         ↓                    ↓
    Visual Feature V     Word Embedding P
                         ↓
                  Sentence-level Agg. P̄
                         ↓
         ↓         Gated Cross-Modal Encoder
              (Detection: no fusion / Grounding: fusion)
                         ↓
              Transformer Decoder
                         ↓
         Scores, Boxes, Masks (Detection + Segmentation + Grounding)
```

APE는 주어진 이미지와 프롬프트 세트에 대해 **점수(scores), 박스(boxes), 마스크(masks)** 집합을 출력한다.

또한 APE는 다양한 크기로 제공된다:
- **APE-Ti**: APE-Ti는 이미지 백본의 크기를 ViT-Ti로 축소한 경량 버전이다.
- **APE-L (Large)**: EVA-02 ViT-L 백본 기반의 최고 성능 버전

---

### 2.4 성능 향상

160개 이상의 데이터셋에 걸친 광범위한 실험에서, 단 하나의 가중치 세트만으로 APE는 최신 모델들을 능가하거나 동등한 수준의 성능을 달성했다.

GLIP, OWL, UNINEXT 등 기존 방법들이 Object365를 학습에 포함했음에도 불구하고 강력한 성능을 제공하지 못한 반면, APE는 그들보다 현저히 우수하여 APE가 모든 학습된 개념을 잘 기억함을 증명한다.

또한 100개 데이터셋으로 구성된 Roboflow와 35개 데이터셋으로 구성된 ODinW를 도입하여 실세계 시나리오에서의 일반화 가능성을 평가한 결과, APE는 두 벤치마크에서 모두 새로운 SotA를 달성하며 다양한 야생 환경의 대규모 개념을 처리할 수 있음을 검증했다.

APE는 단일 순전파에서 수천 개의 things, stuff, 문장을 프롬프팅하고 다양한 segmentation 태스크를 수행하는 것을 지원한다.

---

### 2.5 한계점

논문에서 직접 언급되거나 구조적으로 추론되는 한계는 다음과 같다:

1. **Fine-grained 정보 손실:** 문장 수준 텍스트 임베딩은 세밀한(fine-grained) 정보를 손실할 수 있으나, region-sentence fusion은 object-word fusion과 비교 가능한 성능을 여전히 달성한다.

2. **계산 비용:** 이미지 특징과 대규모 어휘 간의 상호작용은 여전히 계산 비용이 매우 크다.

3. **History Embedding Bank의 필요성:** fine-grained 정보 손실을 보완하기 위해 제안된 **History Embedding Bank**는 추가적인 메모리 관리 비용을 요구한다.

4. **비디오/시간적 태스크 부재:** APE는 정적 이미지 처리에 집중되어 있어 비디오 이해, 시간적 추론 등의 동적 태스크로의 직접 확장이 제한적이다.

---

## 3. 모델의 일반화 성능 향상 가능성

APE의 일반화 성능은 여러 설계 선택에 의해 강력하게 뒷받침된다.

### 3.1 Task-Specific Fine-tuning 없는 광범위한 데이터 정렬

APE는 자연스럽고 도전적인 특성을 가진 광범위한 데이터에서 시각·언어 표현을 **task별 fine-tuning 없이** 한 번에 정렬한다.

이는 단일 모델이 도메인 특화 fine-tuning 없이도 다양한 분야의 데이터셋에 즉시 적용될 수 있음을 의미한다.

### 3.2 Zero-Shot 일반화 보존

기존 연구는 deep fusion 모듈로 detection 데이터를 학습하면 새로운 카테고리에 대한 zero-shot 일반화가 저하될 수 있음을 보였는데, **Gated Interaction**은 detection 태스크에 대한 fusion을 명시적으로 금지함으로써 이러한 퇴보를 방지한다.

수식으로 표현하면 detection 시:

$$
g_{det} = 0 \quad \Rightarrow \quad \hat{V}_{det} = V \quad \text{(fusion 비활성화)}
$$

grounding 시:

$$
g_{grd} = \sigma(W_g \cdot [V; \bar{P}]) \quad \Rightarrow \quad \hat{V}_{grd} = g_{grd} \odot \text{Attn}(V, \bar{P}) + (1-g_{grd}) \odot V
$$

### 3.3 Open-Vocabulary 확장성

APE는 교차 모달 융합의 효과를 유지하면서 수천 개의 카테고리 어휘와 영역 설명으로 모델 프롬프팅을 효율적으로 확장하는, detection과 grounding의 수렴을 달성한다.

이는 학습 시 보지 못한 카테고리(novel categories)에 대한 zero-shot/open-vocabulary 일반화 능력을 크게 향상시킨다.

### 3.4 대규모·다양성 데이터 학습

APE는 수천 개의 어휘나 언어 설명을 동시에 사용하여 한 번에 모든 것을 탐지하고 분할한다.

APE는 instance segmentation과 semantic segmentation 모두에서 전경 객체와 배경 stuff를 모두 지원한다.

이러한 **다양한 도메인 커버리지**는 실세계 데이터 분포 시프트(distribution shift)에 대한 강건성을 높인다.

---

## 4. 2020년 이후 관련 최신 연구 비교 분석

| 모델 | 연도 | 핵심 방법 | 탐지 | 분할 | Grounding | 범용성 | 주요 한계 |
|---|---|---|---|---|---|---|---|
| **CLIP** | 2021 | 대조 학습 (Contrastive Learning) | ✗ | ✗ | △ | 분류 중심 | 위치 예측 불가 |
| **GLIP** | 2022 | Object-Word Alignment | ✓ | ✗ | ✓ | △ | 세분화 미지원, 느린 추론 |
| **SAM** | 2023 | Promptable Segmentation | ✗ | ✓ | ✗ | △ | 의미론적 이해 부재 |
| **X-Decoder** | 2023 | 통합 Decoder | ✓ | ✓ | ✓ | ✓ | Granularity Gap |
| **GroundingDINO** | 2023 | DINO + 언어 인코더 융합 | ✓ | ✗ | ✓ | △ | 대규모 카테고리 처리 한계 |
| **OpenSeeD** | 2023 | 개방 어휘 탐지+분할 | ✓ | ✓ | ✗ | △ | 단일 도메인 한계 |
| **APE (본 논문)** | 2024 | Sentence-Object Matching + Gated Fusion + TSE | ✓ | ✓ | ✓ | ✓✓ | Fine-grained 정보 손실, 비디오 미지원 |
| **DINO-X** | 2024 | 다중 헤드 + 대규모 그라운딩 데이터 | ✓ | ✓ | ✓ | ✓✓ | 매우 큰 모델 크기 |

### 4.1 GLIP vs. APE

GLIP은 object-word 수준의 정렬에 의존하여 수천 개의 카테고리를 동시에 처리하는 데 한계가 있다.

$$
\text{GLIP: } \mathcal{L} = \mathcal{L}_{loc} + \mathcal{L}_{align}^{word}
$$

반면 APE는 sentence-level embedding으로 전환하여 효율성을 크게 향상시켰다.

$$
\text{APE: } \mathcal{L} = \mathcal{L}_{loc} + \mathcal{L}_{align}^{sentence} + \mathcal{L}_{seg}^{TSE}
$$

### 4.2 SAM vs. APE

SAM, X-Decoder, OpenSeeD, SEEM과 같은 분할 중심 모델들은 일반적으로 전경 객체와 배경 stuff 사이의 granularity 불일치 문제를 겪는다.

APE는 이를 **Thing-Stuff Equalizing(TSE)** 기법으로 해결하며, SAM과 달리 텍스트 프롬프트 기반의 의미론적 이해 능력도 갖춘다.

### 4.3 GroundingDINO vs. APE

GroundingDINO는 텍스트 백본, 이미지 백본, feature enhancer, 언어 기반 쿼리 선택, 교차 모달 디코더를 포함한다.

두 모델 모두 transformer 기반이지만, APE는 detection-grounding 통합 및 TSE를 통해 **픽셀 수준 태스크까지 통합**하는 더 넓은 범용성을 제공한다.

### 4.4 DINO-X와의 관계

객체 탐지는 점차 폐쇄 집합 탐지(closed-set detection)에서 GroundingDINO, GLIP 등과 같이 사용자 제공 프롬프트에 대응하는 객체를 식별할 수 있는 개방 집합 탐지(open-set detection)로 발전했다.

APE는 이 흐름에서 **단일 모델 범용성**을 극대화하는 방향의 선구자적 역할을 했으며, DINO-X 등 후속 연구에 영향을 미쳤다.

---

## 5. 미래 연구에 미치는 영향 및 고려할 점

### 5.1 미래 연구에 미치는 영향

**① 단일 범용 모델 패러다임의 강화**

APE는 넓은 데이터에서 시각·언어 표현을 한 번에 정렬하여 task별 fine-tuning 없이 다양한 언어-시각 태스크를 수행할 수 있음을 증명한 범용 시각 인식 모델로서 이후 연구들에 직접 활용되고 있다.

**② Image-Centric 학습 패러다임 확산**

Image-centric 포맷($\{I, (T_1,B_1), \ldots, (T_n, B_n)\}$)은 학습 효율을 92배 향상시키는 핵심 설계로, 다중 태스크 학습의 새로운 표준을 제시했다.

**③ Gated Modality Fusion의 파급 효과**

Gated Cross-modality Interaction의 개념, 즉 **태스크에 따라 융합 게이트를 제어**하는 아이디어는 멀티모달 시스템에서 zero-shot 일반화 유지와 fusion 효율화를 동시에 달성하는 방법론적 기반이 되고 있다.

**④ Frozen Backbone으로서의 활용**

일부 연구는 APE를 frozen backbone으로 사용하여 2D ground-truth 특징을 추출하는 데 활용하고 있다.

이는 APE가 단순한 탐지 모델을 넘어 **범용 특징 추출기(universal feature extractor)**로서도 활용됨을 의미한다.

---

### 5.2 향후 연구 시 고려할 점

#### ① 비디오/시간적 일반화 확장

현재 APE는 정적 이미지에 초점이 맞춰져 있다. SAM 2는 비디오 추적 및 소형 객체 분할이 개선된 기능을 추가했다. 이처럼 APE의 패러다임을 비디오로 확장하기 위해서는 **시간적 일관성(temporal consistency)** 학습이 중요한 연구 과제이다.

#### ② Fine-grained 정보 손실 보완

문장 수준 집계에서 발생하는 세밀한 정보 손실은 다음과 같이 정형화될 수 있다:

$$
\bar{P} = \frac{1}{N}\sum_{i=1}^{N} P_i \quad (\text{정보 손실 발생})
$$

이를 해결하기 위해 **어텐션 기반 가중 집계**나 **계층적 문장 임베딩** 기법 도입을 고려할 수 있다:

$$
\bar{P} = \sum_{i=1}^{N} \alpha_i P_i, \quad \alpha_i = \text{softmax}(W_\alpha P_i)
$$

#### ③ 데이터 효율성과 편향(Bias) 문제

대부분의 연구자들이 대규모 데이터 사전 학습을 활용하는 foundation 모델에 집중하면서, 학습 중 데이터 누출(data leakage) 문제에 주의를 기울이는 연구자가 적어졌으며, 성능 향상이 데이터 누출에 의한 것일 가능성이 있다.

따라서 **엄격한 zero-shot 평가 프로토콜** 수립이 중요한 고려 사항이다.

#### ④ 경량화 및 엣지 배포

APE-Ti는 단 6M 파라미터의 백본만을 사용하는 체크포인트를 공개했다. 하지만 실세계 모바일/엣지 환경에서의 실시간 범용 인식을 위한 **양자화(quantization), 프루닝(pruning), 지식 증류(knowledge distillation)** 등의 경량화 연구가 추가로 필요하다.

#### ⑤ 멀티모달 LLM과의 통합

open-set 탐지 모델은 동적 환경에서의 로봇 적응성 향상, 자율주행 차량의 새로운 객체 인식, 멀티모달 대형 언어 모델(MLLM)의 지각 능력 향상 등 수많은 실용적 응용을 가지고 있다.

APE와 같은 범용 시각 인식 모델이 LLM의 시각 인식 모듈로 통합되는 연구는 매우 유망한 방향이다.

---

## 📚 참고 자료 및 출처

| # | 출처 | URL |
|---|---|---|
| 1 | **arXiv 논문 원문** (Shen et al., 2023) | https://arxiv.org/abs/2312.02153 |
| 2 | **CVPR 2024 Open Access 논문 PDF** | https://openaccess.thecvf.com/content/CVPR2024/papers/Shen_Aligning_and_Prompting_Everything_All_at_Once_for_Universal_Visual_CVPR_2024_paper.pdf |
| 3 | **GitHub 공식 코드 저장소** (shenyunhang/APE) | https://github.com/shenyunhang/APE |
| 4 | **CVPR 2024 포스터 페이지** | https://cvpr.thecvf.com/virtual/2024/poster/29477 |
| 5 | **IEEE Xplore 논문 페이지** | https://ieeexplore.ieee.org/document/10658173 |
| 6 | **East China Normal University 순수 연구 DB** | https://pure.ecnu.edu.cn/en/publications/aligning-and-prompting... |
| 7 | **ResearchGate 논문 페이지** | https://www.researchgate.net/publication/384235422 |
| 8 | **BulletPapers AI 요약** | https://www.bulletpapers.ai/paper/ce73a802-237f-b369-5037-cd8480509347 |
| 9 | **DINO-X arXiv** (비교 연구) | https://arxiv.org/html/2411.14347v1 |
| 10 | **GroundingDINO GitHub** (비교 연구) | https://github.com/IDEA-Research/GroundingDINO |
| 11 | **OV-VG ScienceDirect** (비교 분석) | https://www.sciencedirect.com/science/article/pii/S0925231224005095 |

> **⚠️ 주의사항:** 본 분석에서 수식 중 일부(게이트 벡터 상세 수식, TSE Loss 분해 등)는 논문의 공개된 arXiv 원문 및 CVPR PDF에 기반하되, 수식의 정확한 기호 정의는 논문 내 Appendix 또는 세부 섹션에서 확인하시기를 권장합니다. 논문 PDF 원문 접근이 가능하다면 Section 3 (Method)를 직접 참조하시면 가장 정확한 수식을 확인할 수 있습니다.
