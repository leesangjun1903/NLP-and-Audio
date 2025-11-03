# Bottom-Up and Top-Down Attention for Image Captioning and Visual Question Answering

### 1. 핵심 주장 및 주요 기여

이 논문은 **물체 및 시각적으로 두드러진 영역 수준에서 주의 메커니즘을 작동시키는 결합형 상향식-하향식 주의 모델**을 제안합니다. 기존의 상향식 주의 메커니즘은 CNN의 균일한 격자형 특성 맵에 작동하지만, 이 논문의 핵심 주장은 **인간의 시각계와 유사하게, 물체와 두드러진 영역이 주의의 자연스러운 기초**라는 점입니다.[1]

**주요 기여:**
- 이미지 캡셔닝 태스크에서 MSCOCO 테스트 서버 기준 새로운 최고 성능 달성 (CIDEr 117.9, SPICE 21.5, BLEU-4 36.9)[1]
- 2017 VQA Challenge에서 1위 획득 (VQA v2.0에서 70.3% 정확도)[1]
- 두 가지 상이한 시각-언어 태스크에서 방법론의 광범위한 적용 가능성 입증[1]

---

### 2. 논문이 해결하는 문제

**주요 문제점:**

1. **영역 선택의 임의성**: 기존 주의 메커니즘은 이미지 내용과 관계없이 균등한 크기와 모양의 수용 영역(receptive field)을 사용합니다. 이는 서로 다른 스케일의 물체 감지를 어렵게 만듭니다.[1]

2. **특징 결합 문제**: 한 물체의 서로 다른 특징들이 공간적으로 분산되어 있으면, 같은 물체의 정보를 동시에 처리하지 못합니다. 인간 시각계의 "특징 결합 문제(feature binding problem)"에 해당합니다.[1]

3. **세밀한 분석 부족**: 상향식 영역 선택이 명시적으로 다루어지지 않아, 모델의 해석 가능성이 낮습니다.

***

### 3. 제안 방법 (수식 포함)

#### 3.1 상향식 주의 모델 (Bottom-Up Attention)

**Faster R-CNN 기반 구현:**

Faster R-CNN을 ResNet-101과 함께 사용하여 시각적으로 두드러진 영역을 감지합니다. 각 이미지로부터 최대 100개의 영역 제안을 추출하며, 신뢰도 임계값(0.2) 이상인 모든 영역을 선택합니다.[1]

각 영역 $$i$$에 대해, 특성 벡터 $$v_i \in \mathbb{R}^{2048}$$를 평균 풀링된 컨볼루셔널 특성으로 정의합니다.[1]

#### 3.2 캡셔닝 모델의 상향식-하향식 결합

**상향식 주의 LSTM:**

입력 벡터는 다음과 같이 구성됩니다:[1]

$$
x_t^1 = [h_{t-1}^2, \bar{v}, W_e \Pi_t]
$$

여기서:
- $$h_{t-1}^2$$: 이전 언어 LSTM의 출력
- $$\bar{v} = \frac{1}{k}\sum_i v_i$$: 평균 풀링된 이미지 특성
- $$W_e \Pi_t$$: 이전 생성 단어의 임베딩

**주의 가중치 계산:**

LSTM 출력 $$h_t^1$$이 주어질 때, 각 영역 $$i$$에 대한 비정규화 주의 가중치는:[1]

$$
a_{i,t} = w_a^T \tanh(W_{va}v_i + W_{ha}h_t^1)
$$

정규화된 주의 가중치:[1]

$$
\alpha_t = \text{softmax}(a_t)
$$

참석된 이미지 특성 (가중 평균):[1]

$$
\hat{v}_t = \sum_{i=1}^k \alpha_{i,t}v_i
$$

**언어 LSTM:**

언어 모델 LSTM의 입력:[1]

$$
x_t^2 = [\hat{v}_t, h_t^1]
$$

조건부 확률 분포:[1]

$$
p(y_t | y_{1:t-1}) = \text{softmax}(W_p h_t^2 + b_p)
$$

전체 시퀀스 확률:[1]

$$
p(y_{1:T}) = \prod_{t=1}^T p(y_t | y_{1:t-1})
$$

**훈련 목적:**

교차 엔트로피 손실:[1]

$$
L_{XE}(\theta) = -\sum_{t=1}^T \log(p_\theta(y_t^* | y_{1:t-1}^*))
$$

CIDEr 최적화 (Self-Critical Sequence Training):[1]

$$
L_R(\theta) = -\mathbb{E}_{y_{1:T} \sim p_\theta}[r(y_{1:T})]
$$

여기서 $$r$$은 평가 메트릭입니다.

#### 3.3 VQA 모델

**게이트 하이퍼볼릭 탄젠트 활성화:**

$$
\tilde{y} = \tanh(Wx + b)
$$

$$
g = \sigma(W'x + b')
$$

$$
y = \tilde{y} \odot g
$$

여기서 $$\odot$$는 요소별 곱(Hadamard product)입니다.[1]

**주의 계산:**

$$
a_i = w_a^T f_a([v_i, q])
$$

최종 응답 분포:[1]

$$
h = f_q(q) \odot f_v(\hat{v})
$$

$$
p(y) = \sigma(W_o f_o(h))
$$

***

### 4. 모델 구조

#### 캡셔닝 모델 구조

```
입력 이미지 
    ↓
[Faster R-CNN (ResNet-101)]
    ↓
k개의 영역 특성 V = {v₁, ..., vₖ}
    ↓
┌─────────────────────────────────┐
│   상향식-하향식 주의 메커니즘    │
├─────────────────────────────────┤
│  LSTM 계층 1: 상향식 주의 LSTM  │ → α_{i,t} 계산
│  LSTM 계층 2: 언어 모델 LSTM    │ → 단어 생성
└─────────────────────────────────┘
    ↓
자유로운 형태의 캡션 생성
```

#### VQA 모델 구조

```
질문 & 이미지
    ↓
┌──────────────┬──────────────────┐
│ GRU 인코더   │ Faster R-CNN 특성│
│ (질문 표현)  │ (이미지 영역)    │
└──────┬───────┴────────┬─────────┘
       │                │
       └────┬───────────┘
            ↓
    [주의 계산 모듈]
    가중 합 특성 계산
            ↓
    [다중 모달 결합]
    h = f_q(q) ⊙ f_v(v̂)
            ↓
    [답변 예측]
    p(y) = σ(W_o f_o(h))
```

***

### 5. 성능 향상

#### 이미지 캡셔닝 결과[1]

| 메트릭 | ResNet 기준선 | Up-Down (제안) | 상대적 향상 |
|--------|--------------|---------------|-----------|
| BLEU-4 | 33.4 | 36.2 | 8% |
| METEOR | 26.1 | 27.0 | 3% |
| ROUGE-L | 54.4 | 56.4 | 4% |
| CIDEr | 105.4 | 113.5 | 8% |
| SPICE | 19.2 | 20.3 | 6% |

**SPICE 세부 분석 (CIDEr 최적화 후):**[1]

| 평가 항목 | ResNet | Up-Down | 향상도 |
|---------|--------|---------|--------|
| Objects | 37.0 | 39.1 | +2.1 |
| Attributes | 9.2 | 10.0 | +0.8 |
| Relations | 6.1 | 6.5 | +0.4 |
| Color | 10.6 | 11.4 | +0.8 |
| Count | 12.0 | 18.4 | +6.4 |

상향식 주의는 **물체 인식, 특성 감지, 관계 파악 능력을 모두 향상**시킵니다.[1]

#### VQA 결과[1]

| 질문 유형 | ResNet 14×14 | Up-Down | 향상도 |
|---------|------------|---------|--------|
| Yes/No | 76.6% | 80.3% | 3.7% |
| Number | 36.2% | 42.8% | 6.6% |
| Other | 49.5% | 55.8% | 6.3% |
| 전체 정확도 | 57.9% | 63.2% | 5.3% |

**특히 "Number" 질문에서 14% 상대적 향상** - 정확한 물체 위치 파악의 중요성을 나타냅니다.[1]

***

### 6. 한계 및 제약

#### 6.1 구조적 한계

1. **단일 패스 주의**: 논문이 제시한 모델은 간단한 한 번의 주의 메커니즘만 사용합니다. 더 복잡한 스택형, 다중 헤드, 양방향 주의 체계는 추가 개선을 가능하게 할 수 있습니다.[1]

2. **고정된 영역 수**: 최대 100개의 영역으로 제한되어 있습니다. 일부 실험에서는 단순히 상위 36개 특성만 선택해도 거의 동일한 성능을 얻을 수 있음이 발견되었습니다.[1]

3. **특성 고정**: 캡셔닝 및 VQA 모델 훈련 시 Faster R-CNN의 이미지 특성이 고정되어 있어, 엔드-투-엔드 미세 조정이 불가능합니다.[1]

#### 6.2 데이터 및 도메인 한계

1. **도메인 편향**: 모델은 Visual Genome과 MSCOCO라는 특정 도메인의 데이터로 훈련됩니다. 완전히 다른 도메인 (예: 의료 영상, 위성 영상)에서의 일반화 성능은 명시되지 않습니다.

2. **객체 감지 성능에 종속**: 상향식 주의의 품질은 Faster R-CNN의 객체 감지 정확도에 직접 영향을 받습니다. 드물거나 작은 물체는 제대로 감지되지 않을 수 있습니다.

3. **Visual Genome 데이터 정제**: 2,000개의 물체 클래스와 500개의 속성 클래스에서 시작하여 1,600개와 400개로 수동 정제되었습니다. 추상적 클래스와 명확한 지역화가 어려운 클래스들이 제거되었습니다.[1]

#### 6.3 정성적 한계

1. **실패 사례**: 그림 9의 마지막 예시에서 개가 뛰어다니는 포즈를 누워있는 것으로 잘못 인식하는 경우가 있습니다. 이는 부정확한 영역 자르기로 인해 개의 머리와 발이 누락되었기 때문입니다.[1]

2. **읽기/계수 능력 제한**: VQA 모델의 단순한 구조로 인해 이미지 내 텍스트 읽기나 정확한 물체 계수 능력이 제한적입니다.[1]

---

### 7. 일반화 성능 향상 가능성

#### 7.1 현재 연구 기반의 도메인 일반화 문제

최신 연구는 **비전-언어 모델의 도메인 일반화**를 중점적으로 다루고 있습니다. 특히:[2]

**교차 도메인 이미지 캡셔닝 (Domain Generalization for Image Captioning, DGIC)** - Ren et al. (2023)은 훈련 도메인과 다른 미지의 테스트 도메인에서의 일반화 능력을 연구합니다. MSCOCO (일반 도메인), Vizwiz (보조 도메인), Flickr30k (소셜 도메인), CUB-200 (조류 도메인), Oxford-102 (꽃 도메인)를 포함하는 벤치마크를 제시했습니다.[3]

#### 7.2 일반화 성능 개선 방향

1. **Vision Transformer (ViT) 기반 대체**: 최신 연구에 따르면 **다중 헤드 주의와 상향식 주의의 조합**이 이미지 캡셔닝에서 최고 성능을 달성합니다. ViT는 CNN보다 분포 이동에 더 강건한 특성을 보입니다.[4][5]

2. **비전-언어 모델 활용**: CLIP, BLIP 등의 대규모 사전 훈련 모델을 활용한 프롬프트 학습이 **도메인 시프트 상황에서 우수한 일반화**를 보여줍니다. 특히:[6]
   - **프롬프트 기반 미세 조정**: 원래 특성 공간 보존으로 OOD 일반화 유지[7]
   - **메타 학습 기반 접근**: 한 도메인의 프롬프트가 다른 도메인에서도 성능을 유지하도록 제약[6]

3. **약한 감독(Weak Supervision) 기반 개선**: WeaQA는 인간 주석 없이 이미지 캡션에서 합성 Q-A 쌍을 생성하여, **고비용 객체 경계 상자 주석 의존성을 감소**시킵니다. 이는 저자원 도메인에서의 일반화를 개선합니다.[8]

4. **멀티모달 특성 정렬**: 텍스트 설명을 활용한 **도메인 불변 표현 학습**이 도메인 시프트를 완화합니다.[9][10]

5. **테스트 시간 적응**: **TPS (Test-Time Prototype Shifting)**는 테스트 데이터에 대한 프로토타입을 동적으로 조정하여 도메인 갭을 메웁니다. 메모리와 계산 효율성이 우수합니다.[11]

#### 7.3 본 논문의 일반화 제약 극복 방안

**논문 내 관찰사항:**
- 기존 Faster R-CNN 기반 상향식 주의는 **훈련 도메인에 특화**되어 있습니다.
- ResNet 기반선은 더 많은 합성곱 계층(2배)을 사용하지만 성능이 낮으므로, **특성 추출의 질보다 선택 메커니즘의 구조가 중요**함을 시사합니다.

**개선 전략:**
1. 시각적으로 두드러진 영역 제안을 보다 **일반화 가능한 특성**으로 대체 (예: 더 큰 이미지 해상도, 다중 스케일 특성)
2. Faster R-CNN 대신 **미분 가능한 영역 선택 메커니즘** 도입
3. **교차 도메인 데이터로 사전 훈련**된 특성 사용
4. 언어 모델과 시각 인코더를 **독립적으로 훈련**하여 모달 간 정렬 개선

---

### 8. 미래 연구에 미치는 영향

#### 8.1 이론적 기여

1. **특징 결합 문제의 구체화**: 인간 시각계의 기본 원리(특징 결합 문제)를 컴퓨터 비전 모델에 명시적으로 적용한 첫 연구입니다. 이는 이후 많은 시각-언어 연구의 이론적 기초가 되었습니다.[12][1]

2. **객체 수준 주의의 정당화**: 균일한 격자 기반 주의가 아닌 **객체 중심 주의**가 보다 자연스럽고 해석 가능함을 실증적으로 입증했습니다.[1]

#### 8.2 방법론적 영향

1. **Bottom-up 특성 사전 계산의 표준화**: 이 논문 이후, 이미지 캡셔닝 및 VQA 연구에서 **상향식 주의 특성을 사전 계산된 특성으로 사용**하는 것이 사실상의 표준이 되었습니다. GitHub 리포지토리에서 배포된 사전 계산 특성이 널리 사용되고 있습니다.[13][4]

2. **두 가지 태스크에서의 통일된 접근**: 동일한 상향식 특성이 이미지 캡셔닝과 VQA 모두에서 작동함을 보여 **멀티태스크 학습의 가능성**을 시사합니다.[1]

3. **주의 메커니즘의 재해석**: 이후 연구에서 **다중 헤드 주의, 계층적 주의, 트랜스포머 기반 주의**가 이 개념을 기반으로 발전했습니다.[4]

#### 8.3 학술적 영향 (인용도)

이 논문은 **6,240회 이상 인용**되었으며, 다음 분야에 광범위한 영향을 미쳤습니다:[14]
- 시각-언어 사전 훈련 (Vision-Language Pretraining)
- 이미지 캡셔닝 및 VQA의 벤치마크 설정
- 트랜스포머 기반 시각-언어 모델 개발

***

### 9. 앞으로 연구 시 고려할 점 (최신 연구 기반)

#### 9.1 도메인 일반화 중시

**권장사항:** 단일 도메인(MSCOCO) 성능 최적화에서 벗어나, **다중 도메인 일반화 능력**을 평가해야 합니다.[3]

- DGIC 벤치마크와 같은 교차 도메인 평가 필수
- 의료, 원격 감지, 미술 등 특수 도메인에서의 성능 검증

#### 9.2 약한 감독 및 저자원 시나리오

**권장사항:** 객체 경계 상자 주석의 고비용을 고려하여 **약한 감독 학습** 방법론 개발:[8]
- 이미지 캡션에서 자동 생성된 Q-A 쌍 활용
- 공간-피라미드 이미지 패치를 경계 상자 대체로 사용

#### 9.3 비전 트랜스포머 활용

**권장사항:** ResNet-101 기반 Faster R-CNN 대신 **ViT 기반 영역 선택**을 고려:[5][4]
- 더 나은 분포 이동 강건성
- 멀티헤드 주의로의 자연스러운 확장

#### 9.4 멀티모달 대형 언어 모델 연계

**권장사항:** LLaMA, GPT 등의 대형 언어 모델과 결합하여:[9]
- 보다 풍부한 의미 정보 캡처
- 복잡한 추론 능력 향상
- 저자원 도메인에서의 지식 전이 개선

#### 9.5 해석 가능성 강화

**권장사항:** 주의 맵 시각화 외에도:[1]
- 개념 수준의 해석 (예: 특정 속성이나 관계가 왜 선택되었는가)
- 어트리뷰션 기반 방법으로 모델 의사결정 추적

#### 9.6 효율성 개선

**권장사항:** 실시간 응용 고려:
- 영역 수 감소 (상위 36개 사용 가능성 확인)[1]
- 경량 백본 모델로의 지식 증류[15]

***

### 10. 결론

"Bottom-Up and Top-Down Attention for Image Captioning and Visual Question Answering"는 **객체 수준의 주의 메커니즘**이 시각-언어 이해 태스크에 얼마나 중요한지를 선도적으로 입증한 획기적 논문입니다. 2017년 발표 이후, 이 방법론은 이미지 캡셔닝 및 VQA 연구의 기본 기초가 되었습니다.

그러나 **도메인 일반화, 계산 효율성, 개방 집합 인식(open-set recognition)** 측면에서는 제약이 있습니다. 최신 연구 동향은 Vision Transformer, 비전-언어 모델, 프롬프트 학습을 활용하여 이러한 한계를 극복하고 있습니다. 향후 연구자들은 이 논문의 핵심 직관(객체 중심 주의)을 유지하면서, **다양한 도메인과 저자원 환경에서의 일반화 능력**을 추가로 강화해야 합니다.

---

### 참고문헌

 Anderson, P., He, X., Buehler, C., Teney, D., Johnson, M., Gould, S., & Zhang, L. (2018). Bottom-up and top-down attention for image captioning and visual question answering. In *Proceedings of IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)* (6077-6086).[1]

 Attention-based transformer models survey (2024). Retrieved from arXiv:2506.05399.[16]

 WeaQA: Weak Supervision via Captions (2021). Retrieved from ACL Anthology 2021.findings-acl.302.[8]

 Neural attention for image captioning review (2021). Retrieved from arXiv:2111.15015.[4]

 Anderson, P. Bottom-up attention GitHub repository. Retrieved from https://github.com/peteanderson80/bottom-up-attention[13]

 Domain shift in vision-language models. Retrieved from IEEE 11094684 (2024).[17]

 VLLaVO: Mitigating Visual Gap (2024). Retrieved from arXiv.[9]

 Cross-modal alignment for vision-language (2024). Retrieved from arXiv.[10]

 Test-Time Prototype Shifting (2024). Retrieved from arXiv.[11]

 OGEN: Overcoming Pitfalls (2024). Retrieved from arXiv:2401.15914.[7]

 Learning Domain Invariant Prompt (2023). Retrieved from arXiv:2212.04196.[6]

 Generalizing Vision-Language Models (2024). Retrieved from arXiv.[12]

 Vision-language generalization survey (2023). Retrieved from arXiv.[2]

 Crossing the Gap: Domain Generalization for Image Captioning (2023). Retrieved from CVPR:2023.[3]

 Leveraging Vision-Language Models (2024). Retrieved from GitHub.[15]

 Vision Transformers Robustness (2022). Retrieved from22). Retrieved from AAAI.[48]

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/e11dca74-7939-4f92-a790-4540d01ada3a/1707.07998v3.pdf)
[2](https://arxiv.org/html/2506.18504v1)
[3](https://openaccess.thecvf.com/content/CVPR2023/papers/Ren_Crossing_the_Gap_Domain_Generalization_for_Image_Captioning_CVPR_2023_paper.pdf)
[4](https://arxiv.org/pdf/2111.15015.pdf)
[5](https://github.com/sayakpaul/robustness-vit)
[6](https://arxiv.org/pdf/2212.04196.pdf)
[7](https://arxiv.org/abs/2401.15914)
[8](https://aclanthology.org/2021.findings-acl.302.pdf)
[9](https://www.semanticscholar.org/paper/135521c4432193178966a5ec24343f1a1599d541)
[10](https://ieeexplore.ieee.org/document/10448156/)
[11](https://ieeexplore.ieee.org/document/10943562/)
[12](http://arxiv.org/pdf/2311.17091.pdf)
[13](https://github.com/peteanderson80/bottom-up-attention)
[14](https://arxiv.org/abs/1707.07998)
[15](https://github.com/val-iisc/VL2V-ADiP)
[16](https://ieeexplore.ieee.org/document/8578734/)
[17](https://ieeexplore.ieee.org/document/11094684/)
[18](https://www.semanticscholar.org/paper/a79b694bd4ef51207787da1948ed473903b751ef)
[19](https://openresearch-repository.anu.edu.au/handle/1885/164018)
[20](https://www.semanticscholar.org/paper/71fa26ab5fe121f5dea3232ea8346e520939d676)
[21](https://www.semanticscholar.org/paper/741ceba059bddef78dbff9cc414d9eebac09bdc5)
[22](https://arxiv.org/pdf/1707.07998.pdf)
[23](http://arxiv.org/pdf/2012.02356.pdf)
[24](https://arxiv.org/pdf/2109.05014.pdf)
[25](https://www.aclweb.org/anthology/P19-1348.pdf)
[26](http://arxiv.org/pdf/2404.08589.pdf)
[27](https://arxiv.org/pdf/2301.07389.pdf)
[28](https://arxiv.org/html/2506.05399v1)
[29](https://openaccess.thecvf.com/content/CVPR2024/papers/Danish_Improving_Single_Domain-Generalized_Object_Detection_A_Focus_on_Diversification_and_CVPR_2024_paper.pdf)
[30](https://researchers.mq.edu.au/en/publications/bottom-up-and-top-down-attention-for-image-captioning-and-visual-)
[31](https://ieeexplore.ieee.org/document/10914740/)
[32](https://pmc.ncbi.nlm.nih.gov/articles/PMC8123487/)
[33](https://openaccess.thecvf.com/content_cvpr_2018/papers/Anderson_Bottom-Up_and_Top-Down_CVPR_2018_paper.pdf)
[34](https://www.sciencedirect.com/science/article/abs/pii/S1574013725000425)
[35](https://arxiv.org/pdf/1807.05511.pdf)
[36](https://www.semanticscholar.org/paper/Bottom-Up-and-Top-Down-Attention-for-Image-and-VQA-Anderson-He/a79b694bd4ef51207787da1948ed473903b751ef)
[37](https://arxiv.org/abs/2411.16407)
[38](https://ieeexplore.ieee.org/document/10657688/)
[39](https://arxiv.org/abs/2407.01842)
[40](https://ieeexplore.ieee.org/document/10943680/)
[41](https://arxiv.org/abs/2402.09816)
[42](https://arxiv.org/abs/2407.19795)
[43](http://arxiv.org/pdf/2404.18758.pdf)
[44](http://arxiv.org/pdf/2411.04892.pdf)
[45](http://arxiv.org/pdf/2311.12327.pdf)
[46](https://arxiv.org/html/2503.18483v1)
[47](http://arxiv.org/pdf/2406.12638.pdf)
[48](https://openaccess.thecvf.com/content/CVPR2023/papers/Li_Trade-Off_Between_Robustness_and_Accuracy_of_Vision_Transformers_CVPR_2023_paper.pdf)
[49](https://arxiv.org/pdf/2308.02862.pdf)
[50](https://arxiv.org/abs/2411.04892)
[51](https://www.nature.com/articles/s41598-025-91802-6)
[52](https://dl.acm.org/doi/10.1145/3715136)
[53](https://openaccess.thecvf.com/content/CVPR2024/html/Chen_PracticalDG_Perturbation_Distillation_on_Vision-Language_Models_for_Hybrid_Domain_Generalization_CVPR_2024_paper.html)
