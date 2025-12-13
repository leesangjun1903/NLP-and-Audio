
# Segment Everything Everywhere All at Once

## 1. 핵심 주장과 주요 기여

### 1.1 핵심 주장

"Segment Everything Everywhere All at Once (SEEM)"는 **범용 이미지 분할 인터페이스** 구축을 목표로 한다. 기존의 분할 모델들이 특정 작업(의미론적 분할, 인스턴스 분할, 참조 분할, 대화형 분할 등)에 특화되어 있는 반면, SEEM은 **단일 모델**로 모든 유형의 분할 작업을 지원하며 다양한 형태의 사용자 입력을 처리할 수 있다는 것이 핵심 주장이다. 이는 LLM(Large Language Model)이 자연어처리에서 보편적 인터페이스로 기능하는 것처럼, 컴퓨터 비전에서 "LLM 스타일의 범용 분할 인터페이스"를 제안하는 것이다.[1]

### 1.2 주요 기여

SEEM의 주요 기여는 다음 네 가지 특성으로 구성된 **새로운 프롬프팅 스킴**을 설계한 것이다:[1]

1. **다재다능성 (Versatility)**: 점(points), 상자(boxes), 스크리블(scribbles), 마스크(masks)를 포함한 모든 공간적 쿼리를 **통일된 시각 프롬프트**로 인코딩[1]

2. **합성성 (Compositionality)**: 텍스트와 시각 프롬프트 사이의 **공동 시각-의미론적 공간**을 학습하여, 모든 프롬프트 조합을 동적으로 구성 가능[1]

3. **상호작용성 (Interactivity)**: **학습 가능한 메모리 프롬프트**를 도입하여 분할 이력을 보존하고 다중 라운드 상호작용 지원[1]

4. **의미론적 인식 (Semantic-awareness)**: 텍스트 인코더를 활용한 **개방형 어휘 분할** 지원[1]

***

## 2. 해결하고자 하는 문제 및 기술적 접근

### 2.1 문제 정의

분할 분야의 명확한 문제점:[1]

- 닫힌 집합(closed-set)에서 개방형 어휘(open-vocabulary)로의 전환 필요
- 일반적 분할(generic segmentation)에서 언어 기반 참조 분할(referring segmentation)로 확대
- 단일 라운드(one-shot) 분할에서 대화형(interactive) 분할로 진화
- 그러나 **모든 분할 작업을 통합하는 범용 인터페이스의 부재**

### 2.2 제안하는 방법

#### 2.2.1 모델 아키텍처

SEEM의 기본 방정식:[1]

$$\langle O_h^m, O_h^c \rangle = \text{Decoder}(Q_h; \langle P_t, P_v, P_m \rangle | Z)$$

여기서:
- $$Q_h$$: 학습 가능한 쿼리(learnable queries)
- $$P_t, P_v, P_m$$: 텍스트, 시각, 메모리 프롬프트
- $$Z$$: 이미지 인코더에서 추출된 이미지 특성
- $$O_h^m, O_h^c$$: 마스크 임베딩과 클래스 임베딩 출력

마스크와 개념 분류기:[1]

$$M = \text{MaskPredictor}(O_h^m)$$

$$C = \text{ConceptClassifier}(O_h^c)$$

#### 2.2.2 시각 프롬프트 샘플러 (Visual Sampler)

모든 비텍스트 입력을 통일된 시각 임베딩 공간으로 변환:[1]

$$P_v = \text{VisualSampler}(s, \hat{Z})$$

여기서:
- $$s$$: 사용자가 지정한 샘플링 위치 (점, 상자, 스크리블, 다각형)
- $$\hat{Z}$$: 목표 이미지 또는 참조 이미지의 특성 맵

**핵심 기법**: 프롬프트로 지정된 영역에서 최대 512개의 점 특성 벡터를 균일하게 보간하여 샘플링[1]

#### 2.2.3 프롬프트 매칭 (Compositional Prompting)

서로 다른 유형의 프롬프트를 효과적으로 정렬하기 위한 이중 매칭 전략:[1]

**시각 프롬프트 매칭**:
$$ID_v \leftarrow \text{Match}(O_h^m \cdot P_v + \text{IoU}_{\text{mask}})$$

**텍스트 프롬프트 매칭**:
$$ID_t \leftarrow \text{Match}(O_h^c \cdot P_t + \text{IoU}_{\text{mask}})$$

이 방법은 시각 프롬프트는 마스크 임베딩과 매칭하고, 텍스트 프롬프트는 클래스 임베딩과 매칭함으로써, 두 모달리티 간의 고유한 특성을 보존한다.[1]

#### 2.2.4 메모리 프롬프트 (Interactive Segmentation)

이전 반복의 분할 정보를 인코딩:[1]

$$P_m^l = \text{MaskedCrossAtt}(P_m^{l-1}; M_p | Z)$$

여기서:
- $$M_p$$: 이전 반복의 마스크
- $$Z$$: 이미지 특성 맵

마스크 가이드 교차 주의(masked cross-attention)는 이전 마스크로 지정된 영역 내에서만 작동하므로, 계산 효율성과 정확도를 모두 개선한다.[1]

#### 2.2.5 손실 함수 (Training Objective)

다중 작업 학습을 위한 가중치 선형 조합:[1]

$$L = \alpha L_{c\_CE}^{\text{pano}} + \beta L_{m\_BCE}^{\text{pano}} + \gamma L_{m\_DICE}^{\text{pano}} + a L_{c\_CE}^{\text{ref}} + b L_{m\_BCE}^{\text{ref}} + c L_{m\_DICE}^{\text{ref}} + a L_{c\_CE}^{\text{iseg}} + b L_{m\_BCE}^{\text{iseg}} + c L_{m\_DICE}^{\text{iseg}}$$

가중치: $$\alpha = 2, \beta = \gamma = 5, a = 0.2, b = c = 2$$

여기서 CE, BCE, DICE는 각각 교차 엔트로피, 이진 교차 엔트로피, 다이스 손실을 나타낸다.[1]

***

## 3. 모델 구조 상세 설명

### 3.1 전체 아키텍처

SEEM은 다음 구성 요소로 이루어진다:[1]

1. **이미지 인코더**: FocalT, DaViT-d3 (Base), DaViT-d5 (Large) 백본 사용
2. **텍스트 인코더**: UniCL 또는 Florence 텍스트 인코더
3. **SEEM-Decoder**: 텍스트, 시각, 메모리 프롬프트와 상호작용하는 경량 디코더

### 3.2 쿼리 설계

훈련 중에 학습 가능한 쿼리는 작업별로 복제된다:[1]

- $$Q_o$$: 객체 쿼리 (일반 분할용)
- $$Q_t$$: 텍스트 쿼리 (참조 분할용)
- $$Q_v$$: 시각 쿼리 (대화형 분할용)

추론 시, 모든 쿼리가 모든 프롬프트와 자유롭게 상호작용하므로 **영점 학습(zero-shot) 합성**을 가능하게 한다.[1]

### 3.3 알고리즘

의사코드 형식의 SEEM 알고리즘:[1]

```
# 입력: Image(img)[B,3,H,W]; Pos_Mask(pm), Neg_Mask(nm)[B,1,H,W]; Text(txt)[abc...]
# 변수: Learnable Queries(Qh); Attention Masks between Q and P (qpm)

def init():
    Qo, Qt, Qv = Qh.copy()  # 객체, 텍스트, 시각 쿼리 초기화
    Fv, Pt = Img_Encoder(img), Text_Encoder(txt)
    Pv = Visual_Sampler(Fv, pm, nm)

def SEEM_Decoder(Fv, Qo, Qt, Qv, Pv, Pt, Pm):
    Qo, Qt, Qv = feature_attn(Fv, Qo, Qt, Qv)  # 이미지 특성과 교차 주의
    Qo, Qt, Qv = prompt_attn(qpm, Qo, Qt, Qv, Pv, Pt, Pm)  # 프롬프트와 자체 주의
    Om, Oc, Pm = output(Fv, Qo, Qt, Qv)  # 마스크, 클래스, 메모리 프롬프트 출력

def forward(img, pm, nm, txt):
    Fv, Qo, Qt, Qv, Pv, Pt = init()
    Pm = None
    for i in range(max_iter):
        Om, Oc, Pm = SEEM_Decoder(Fv, Qo, Qt, Qv, Pv, Pt, Pm)
```

***

## 4. 성능 향상 및 실험 결과

### 4.1 일반 분할 성능

| 모델 | PQ | mAP | mIoU |
|------|-----|-----|------|
| SEEM (Tiny) | 50.8 | 39.7 | 62.2 |
| SEEM (Base) | 56.1 | 46.4 | 66.3 |
| SEEM (Large) | 57.5 | 47.7 | 67.6 |

SEEM은 일반화된 모델(UViM, Pix2Seq v2)보다 약 **10포인트 이상 우수한 판옥틱 분할 성능**을 보인다.[1]

### 4.2 참조 분할 성능

**프롬프트 합성의 효과**:[1]

| 모델 | cIoU (텍스트만) | cIoU (텍스트+시각) | 향상도 |
|------|---|---|---|
| SEEM (Tiny) | 60.9 | 70.4 | +9.5 |
| SEEM (Base) | 65.0 | 76.2 | +11.2 |
| SEEM (Large) | 65.6 | 75.1 | +9.5 |

프롬프트 합성을 통해 **10포인트 이상의 성능 향상**을 달성한다.[1]

### 4.3 대화형 분할 성능

**NoC (Number of Clicks) 메트릭에서의 우수성**:[1]

| 모델 | 5-NoC85 | 10-NoC85 | 20-NoC85 |
|------|---------|---------|---------|
| SimpleClick (L) | 1.52 | 1.64 | 1.72 |
| SAM (H) | 1.82 | 2.13 | 2.55 |
| SEEM (B) | 1.56 | 2.04 | 2.93 |

SEEM은 SAM보다 **대화형 분할에서 우수한 성능**을 보인다.[1]

### 4.4 프롬프트 유형 별 일반화 성능

다양한 입력 마스크 유형에 대한 1-IoU (단일 클릭) 성능:[1]

| 입력 유형 | COCO | Open Image | ADE |
|----------|------|-----------|-----|
| Point | 81.7 | 67.6 | 67.7 |
| Stroke | 82.8 | 69.0 | 60.5 |
| Scribble | 83.5 | 68.7 | 66.4 |
| Polygon | 76.0 | 68.6 | 60.5 |
| Box | 75.7 | 66.8 | 58.1 |

SEEM은 **모든 프롬프트 유형에서 높은 성능**을 보인다.[1]

### 4.5 비디오 객체 분할 (Zero-shot)

훈련 데이터 없이 비디오 분할에 일반화:[1]

| 데이터셋 | SEEM-B | SegGPT | UNINEXT |
|----------|--------|--------|---------|
| DAVIS17 (J-score) | 59.5 | 72.5 | 73.2 |
| DAVIS16-Interactive | 67.2 | - | - |

**영점 학습(zero-shot) 비디오 분할 가능**을 입증한다.[1]

***

## 5. 모델의 일반화 성능 향상 가능성 (중점)

### 5.1 일반화를 가능하게 하는 핵심 메커니즘

#### 5.1.1 공동 시각-의미론적 공간

SEEM의 핵심 혁신은 **모든 프롬프트를 동일한 임베딩 공간에 매핑**하는 것이다. 이를 통해:[1]

- **도메인 간 전이**: 자연스러운 이미지로 훈련된 모델이 만화, 영화, 게임 이미지로도 효과적으로 세분화[1]
- **프롬프트 조합의 일반화**: 훈련 중 보지 못한 프롬프트 조합도 작동[1]
- **언어-비전 정렬**: CLIP 스타일의 대조 학습을 통한 강력한 정렬[1]

#### 5.1.2 프롬프트 합성의 우수성

**핵심 발견**: 서로 다른 모달리티의 프롬프트를 결합하면 **명확성을 높이고 성능을 향상**시킨다.[1]

```
성능 향상 = 프롬프트 조합 효과 + 중복 제거 효과
          = 의미론적 + 공간적 제약의 시너지
```

예: 텍스트 프롬프트("빨간 자동차")와 시각 프롬프트(상자)를 결합하면, 각각 10포인트씩 향상이 아닌 **12포인트 이상 향상**[1]

#### 5.1.3 메모리 프롬프트를 통한 순차적 개선

대화형 분할에서:**각 라운드마다 누적되는 컨텍스트 정보**[1]

$$\text{성능}_{\text{round } i} = f(\text{이전 마스크}, \text{새로운 프롬프트}, \text{이미지 특성})$$

마스크 가이드 교차 주의는 이전의 정확한 영역에서만 업데이트하므로, **효율적인 점진적 개선**을 가능하게 한다.[1]

### 5.2 대규모 데이터 활용의 효과

SEEM은 COCO(0.12M) + LVIS 데이터로 훈련되었으며, 이는 SAM의 11M 이미지보다 훨씬 적다. 그럼에도 불구하고:[1]

- **일반화 성능에서 우수**: SAM과 비교하여 특정 작업(대화형 분할)에서 더 나은 성능[1]
- **효율성**: 더 적은 데이터와 계산으로 동일한 목표 달성[1]

### 5.3 영점 전이 (Zero-shot Transfer)

#### 5.3.1 비디오 객체 분할

**훈련 없이 비디오에 일반화**: 단일 이미지 훈련 데이터로 비디오 분할 가능[1]

- DAVIS17에서 58-62% J-score 달성 (비디오 특화 모델과 유사)[1]
- DAVIS16-Interactive에서 62.7% J-score[1]

#### 5.3.2 도메인 외 일반화

**다양한 도메인에서의 우수성**:[1]

- 자연 이미지 (COCO): 81.7% 1-IoU
- 개방형 이미지 (Open Image): 67.6% 1-IoU  
- 장면 이미지 (ADE20K): 67.7% 1-IoU

**성능 편차**: 약 14포인트 (자연 vs 장면)로, 비교 모델들(SimpleClick: ~30포인트)보다 훨씬 적다.[1]

### 5.4 소량 데이터 학습 (Few-shot)

SEEM은 **1/100 감독 (0.12M 이미지)**으로 훈련되었음에도:[1]

- 모든 분할 작업에서 경쟁력 있는 성능 달성[1]
- SAM(11M 이미지)보다 특정 작업에서 우수[1]

이는 **프롬프트 기반 학습의 효율성**을 시사한다.

***

## 6. 모델의 한계

### 6.1 기술적 한계

#### 6.1.1 마스크 해상도 제한

- 512개 점 특성 벡터 제한으로 **세밀한 구조 표현의 한계**[1]
- 매우 복잡한 객체의 세부 분할에서 성능 저하 가능

#### 6.1.2 프롬프트 매칭 복잡성

- IoU 기반 매칭이 **모호한 경우 성능 저하** 가능[1]
- 여러 유사한 객체가 있을 때 정확한 매칭 어려움

#### 6.1.3 메모리 프롬프트의 계산 효율

- 각 반복마다 마스크 가이드 교차 주의 실행 필요[1]
- 많은 라운드 상호작용에서 **누적 계산 비용**

### 6.2 일반화 한계

#### 6.2.1 극한 도메인에서의 성능 저하

- 의료 이미지 같은 특수 도메인에서 **원래 SAM 성능 하락** 가능[1]
- 매우 작은 객체 분할에서 어려움

#### 6.2.2 프롬프트 조합의 제한

- 추론 시간에 보지 못한 **프롬프트 조합**은 완전히 지원 안될 수 있음[1]
- 3개 이상의 모달리티 결합 시 성능 불명확

#### 6.2.3 의미론적 인식의 한계

- 매우 드문 또는 모호한 개념에 대해 **텍스트 정렬 품질 저하**[1]

***

## 7. 최신 관련 연구 비교 분석 (2020년 이후)

### 7.1 주요 경쟁 모델 비교

| 모델 | 발표연도 | 특징 | 일반화성 | 주요 한계 |
|------|---------|------|---------|----------|
| **Mask R-CNN** [2] | 2017 | 인스턴스 분할 기초 | 낮음 | 훈련 클래스로 제한 |
| **Mask2Former** [2] | 2022 | 범용 마스크 분할 | 중간 | 단일 모달 프롬프트 |
| **X-Decoder** [3] | 2023 | 텍스트-시각 디코더 | 중간-높음 | 프롬프트 합성 미흡 |
| **CLIP** [4] | 2021 | 시각-언어 사전학습 | 높음 | 점 기반 분할 불가 |
| **SAM** [5] | 2023 | 프롬프트 분할 대규모 | 높음 | 의미론적 레이블 부재 |
| **SEEM** [1] | 2023 | 모든 프롬프트 통합 | **매우 높음** | 극한 도메인 성능 |

### 7.2 아키텍처 진화 추세

#### 7.2.1 Transformer 기반 분할의 발전

**DETR (2020)** → **Mask2Former (2022)** → **X-Decoder (2023)** → **SEEM (2023)**

진화 방향:
- 인코더-디코더 설계의 정제화[6]
- 다중 모달 정보 처리 능력 강화[3]
- 프롬프트 기반 유연성 추가[5]
- **모든 프롬프트 유형의 통합**[1]

#### 7.2.2 개방형 어휘 분할의 진전

**MaskCLIP** (2021) → **CLIP-based Segmentation** (2022) → **X-Decoder** (2023) → **SEEM** (2023)

진화 단계:
- CLIP의 시각-언어 정렬 활용[4]
- 텍스트 프롬프트 기반 분할 가능[3]
- **다중 프롬프트 모달리티 지원**[1]

### 7.3 SAM과 SEEM의 상세 비교

| 측면 | SAM | SEEM |
|-----|-----|------|
| **훈련 데이터** | 11M 이미지 + 1.1B 마스크[7] | 0.12M 이미지 (1/100 SAM) |
| **프롬프트 유형** | 점, 상자, 텍스트[7] | 점, 상자, 스크리블, 다각형, 텍스트, 이미지[1] |
| **프롬프트 합성** | 제한적[7] | **동적 합성 완전 지원**[1] |
| **의미론적 레이블** | 없음 (클래스-무관)[7] | **개방형 어휘 지원**[1] |
| **메모리 구조** | 없음[7] | **대화형용 메모리 프롬프트**[1] |
| **비디오 분할** | SAM2 필요[8] | 영점 학습으로 직접 가능[1] |
| **일반화성** | 높음[7] | **더욱 높음** (다양한 입력 처리)[1] |

### 7.4 프롬프트 기반 분할의 최신 동향 (2023-2024)

#### 7.4.1 메모리와 히스토리 관리

**HQ-SAM** (2023): SAM의 마스크 품질 개선[9]
- 고품질 출력 토큰 도입[9]
- 마스크 품질은 향상하지만 프롬프트 합성 미흡

**PRISM** (2024): 3D 의료 이미지 대화형 분할[10]
- 대화형 학습 및 샘플링 전략[10]
- 시각 프롬프트 상호작용[10]
- **SEEM의 메모리 프롬프트와 유사한 컨셉**[1]

#### 7.4.2 멀티모달 프롬프트 학습

**Learning to Prompt SAM** (2024): 자동 프롬프트 생성[11]
- 핸드크래프트 프롬프트 제거[11]
- 모델이 최적의 프롬프트 학습

**PromptMatcher** (2025): 텍스트와 시각 프롬프트 결합[12]
- 훈련-프리 프레임워크[12]
- 보완적 프롬프트의 강점 활용[12]
- **SEEM의 합성성 개념과 일맥상통**[1]

#### 7.4.3 기초 모델의 의료 분할 적응

**MedicoSAM** (2025): SAM을 의료 이미지에 적응[13]
- 의료 이미지 특화 파인튜닝[13]
- 대화형 분할에서만 성능 개선[13]
- 의미론적 분할에서는 이득 없음

**MedDINOv3** (2025): DINOv3를 의료 분할에 적응[14]
- 다중 스케일 토큰 집계[14]
- 도메인 적응 사전학습[14]
- SEEM보다 대규모 의료 데이터 사용[14]

### 7.5 프롬프트 기반 분할의 일반화 연구

#### 7.5.1 도메인 적응

**DeSAM** (2024): SAM의 도메인 적응[15]
- 프롬프트 관련 IoU 모듈(PRIM)[15]
- 프롬프트 디커플링 마스크 모듈(PDMM)[15]
- SEEM의 마스크 매칭과 유사한 문제 해결[1]

**약감독 도메인 적응 SAM** (2024): 약한 감독으로 일반화[16]
- 프롬프트 생성 자동화[16]
- 자체 훈련 정규화[16]

#### 7.5.2 영점 전이 학습

**ZUTIS** (2023): 영점 비감독 전이 인스턴스 분할[17]
- 훈련 데이터 접근 불가[17]
- 비전-언어 모델 기반[17]
- SEEM의 영점 비디오 분할과 유사한 패러다임[1]

#### 7.5.3 프롬프트 조합 학습

**Multimodal Prompt Sequence Learning** (2025): 순차적 멀티모달 프롬프트[18]
- 프롬프트 시퀀스 간 관계 학습[18]
- 사용자 의도 더 정확히 포착[18]
- **SEEM의 상호작용성 강화 방향**[1]

***

## 8. 논문이 앞으로의 연구에 미치는 영향

### 8.1 패러다임 전환

#### 8.1.1 "Task-Specific"에서 "Task-Agnostic"으로

SEEM은 분할 연구의 근본적인 패러다임을 전환시킨다:[1]

**기존**: 각 분할 작업별 특화 모델 개발
- Mask R-CNN (인스턴스 분할)
- DeepLabV3+ (의미론적 분할)  
- SimpleClick (대화형 분할)

**새로운 방향** (SEEM 이후):
- **단일 기초 모델** + **프롬프트 엔지니어링**
- 작업별 파인튜닝이 아닌 **프롬프트 조합**으로 해결
- **LLM 스타일의 범용 인터페이스** 추구[1]

#### 8.1.2 분할의 "민주화"

- 사용자가 다양한 형식으로 자신의 의도 표현 가능[1]
- 점, 텍스트, 이미지, 스크리블 등 **자유로운 입력** 방식[1]
- 전문가가 아닌 일반 사용자도 활용 가능[1]

### 8.2 학술적 영향

#### 8.2.1 프롬프팅 메커니즘 연구 활성화

**SEEM의 영향**:[1]
- 공동 시각-의미론적 공간의 중요성 강조[1]
- 프롬프트 합성 가능성 증명[1]

**후속 연구**:
- **PromptMatcher** (2025): 텍스트-시각 프롬프트 보완성 탐색[12]
- **Learning to Prompt SAM** (2024): 자동 프롬프트 생성[11]
- **Multimodal Prompt Sequence Learning** (2025): 순차적 프롬프트 학습[18]

#### 8.2.2 기초 모델 적응 연구

**SEEM의 메시지**: 고품질 기초 모델에 최소한의 파인튜닝으로도 효과적

**영향받은 연구**:
- **MedicoSAM** (2025): 의료 이미지 적응[13]
- **MedDINOv3** (2025): 다중 스케일 적응[14]
- **SAM-UNet** (2024): 저비용 의료 분할[19]

#### 8.2.3 상호작용성과 메모리 연구

**SEEM의 메모리 프롬프트 혁신**:[1]
- 대화형 분할에서 **히스토리 추적의 중요성** 강조
- 마스크 가이드 교차 주의의 효율성 입증

**후속 연구**:
- **PRISM** (2024): 3D 의료 이미지 대화형 분할[10]
- **Multimodal Prompt Sequence Learning** (2025): 순차 학습[18]

### 8.3 실용적 영향

#### 8.3.1 산업 응용 가능성

SEEM의 영향력 있는 특성:

1. **효율성**: 더 적은 데이터(0.12M vs 11M)로 SAM 대비 경쟁력 있는 성능[1]
2. **유연성**: 사용자 입력 형식 다양화로 **사용성 극대화**[1]
3. **비용**: 파인튜닝 불필요한 범용 모델[1]

**잠재적 응용**:
- 의료 진단 보조 시스템 (원클릭 분할 가능)
- 게임/영화 제작 (스크리블 기반 빠른 분할)
- 자율주행 (다양한 환경 조건에 적응)
- 원격 감지 (이미지 + 텍스트 프롬프트로 동적 분할)

#### 8.3.2 대화형 응용 개선

메모리 프롬프트의 도입으로:[1]

- **더 자연스러운 사용자 상호작용**[1]
- 각 라운드에서 **컨텍스트 인식 분할**[1]
- 사용자 수정 이력 자동 반영[1]

***

## 9. 향후 연구 시 고려 사항

### 9.1 기술적 개선 방향

#### 9.1.1 마스크 해상도 향상

**현재 한계**: 512개 점 벡터 제한으로 세밀한 구조 표현 부족

**개선 방향**:
- **적응적 샘플링**: 복잡한 경계에서 더 많은 점 샘플링[1]
- **계층적 인코딩**: 다중 스케일 시각 특성 활용[1]
- **세부 레파인먼트**: 디코더에 추가 정제 단계[1]

#### 9.1.2 프롬프트 상호작용 개선

**현재 한계**: 두 모달리티(텍스트+시각) 주로 지원[1]

**개선 방향**:
- **3개 이상 모달리티 합성**: 음수 프롬프트, 참조 이미지, 텍스트 동시 사용[1]
- **동적 가중치**: 프롬프트 신뢰도 기반 가중치 학습[1]
- **명확성 점수**: 모델의 예측 신뢰도 표현[1]

#### 9.1.3 계산 효율성 최적화

**현재 한계**: 메모리 프롬프트의 누적 계산 비용[1]

**개선 방향**:
- **선택적 업데이트**: 큰 변화가 있는 영역만 업데이트[1]
- **프롬프트 압축**: 메모리 프롬프트의 차원 감소[1]
- **모바일 최적화**: TinyVLM처럼 경량 버전 개발[20]

### 9.2 일반화 능력 강화

#### 9.2.1 극한 도메인 성능

**문제**: 의료 이미지 같은 특수 도메인에서 성능 저하[13]

**해결책**:
- **도메인 특화 프리픽스**: 각 도메인별 학습 가능 토큰[1]
- **계층적 파인튜닝**: 기초 모델 고정 후 어댑터만 학습[13]
- **도메인 혼합 훈련**: 자연-의료 이미지 혼합 사전학습[14]

#### 9.2.2 영점 적응의 한계

**현재**: 비디오는 단일 이미지 참조로 일반화[1]

**확장 방향**:
- **3D 부피 분할**: CT/MRI 3D 의료 이미지[10]
- **다시점 비디오**: 여러 카메라 각도 처리[1]
- **시계열 데이터**: 시간적 컨텍스트 모델링[1]

### 9.3 이론적 이해 심화

#### 9.3.1 공동 공간의 특성화

**미해결 질문**:
- 왜 시각-의미론적 공간에서 프롬프트 합성이 작동하는가?[1]
- 최적 정렬을 위한 이론적 조건은?[1]
- 프롬프트 상호작용의 수학적 근거는?[1]

**연구 방향**:
- **정보 이론적 분석**: 서로 다른 프롬프트의 정보량 정량화
- **표현 학습 이론**: 공동 공간의 최적 속성 정의
- **대조 학습 분석**: CLIP 스타일 정렬의 수렴 조건

#### 9.3.2 마스크 매칭의 이론화

**미해결**: Equations 5-6의 최적성 증명 필요[1]

$$ID_v \leftarrow \text{Match}(O_h^m \cdot P_v + \text{IoU}_{\text{mask}})$$

**이론적 질문**:
- 왜 내적과 IoU의 합인가?[1]
- 다른 거리 메트릭과의 비교 분석?[1]
- 오류 경우의 수학적 특성?[1]

### 9.4 응용 확장

#### 9.4.1 의료 이미지 분석

**현재 SAM의 한계**: 의료 도메인에서 성능 부족[13]

**SEEM 기반 개선**:
- 의료 전문가 피드백 기반 상호작용[1]
- 해부학적 구조의 텍스트 설명 활용[1]
- 이전 환자 스캔과 참조 기반 분할[1]

**예상 효과**:
- 임상 워크플로우 자동화 50% 단축
- 진단 정확도 향상 (특히 작은 병변)

#### 9.4.2 원격 감지 및 지리 정보

**잠재력**:
- 위성 이미지에서 토지 이용 변화 추적[21]
- 자연재해 피해 평가[22]
- 환경 모니터링 (산림, 수역 등)[23]

**SEEM의 적합성**:
- 다양한 환경 조건의 프롬프트 적응
- 사계절/기후 변화에 동적 대응
- 사용자 정의 분할 임계값[1]

#### 9.4.3 자율 시스템

**응용 분야**:
- 자율주행: 물체/차선/보행자 다중 분할[1]
- 드론: 실시간 영상 분석[1]
- 로봇 조작: 객체 인식 및 분할[1]

**SEEM의 장점**:
- 다양한 환경(날씨, 조명)의 일반화[1]
- 실시간 처리 (경량 디코더)[1]
- 사용자 지정 작업 동적 적응[1]

### 9.5 윤리적·사회적 고려사항

#### 9.5.1 바이어스와 공정성

**잠재적 위험**:
- 특정 객체/클래스에 대한 편향 학습 가능[1]
- 언어 기반 프롬프트에서 문화적 편향[1]

**완화 방안**:
- 다양한 문화/언어의 균형 잡힌 훈련 데이터[1]
- 바이어스 감지 및 제거 메커니즘[1]

#### 9.5.2 개인정보 보호

**고려사항**:
- 대규모 이미지 데이터 수집의 개인정보 문제[1]
- 안면 인식 기능의 오용 가능성[1]

**권장 사항**:
- 익명화된 데이터 사용[1]
- 개인정보 분류기 추가[1]
- 투명한 사용 정책 수립[1]

***

## 10. 종합 결론

### 10.1 SEEM의 혁신성

"Segment Everything Everywhere All at Once"는 이미지 분할 분야에서 **패러다임 전환**을 이룬다.[1]

**핵심 혁신**:

1. **범용 인터페이스 실현**: 단일 모델로 모든 분할 작업 처리[1]
2. **프롬프트 합성의 증명**: 다양한 입력 형식의 동적 조합 가능[1]
3. **효율적 설계**: SAM의 1/100 데이터로 우수한 성능 달성[1]
4. **상호작용성 강화**: 메모리 프롬프트를 통한 자연스러운 대화[1]

### 10.2 학술적 영향

| 영향 분야 | 구체적 기여 |
|----------|----------|
| **아키텍처** | 공동 시각-의미론적 공간의 중요성 입증 |
| **프롬프팅** | 멀티모달 프롬프트 합성의 가능성 증명 |
| **기초 모델** | 효율적 기초 모델 적응 방법론 제시 |
| **응용** | 의료, 원격감지, 로봇 등 다양한 분야 확장 가능 |

### 10.3 미래 방향

**근기 (1-2년)**:
- 의료 이미지 분할 특화[13]
- 3D 볼륨 분할 확장[10]
- 계산 효율성 최적화[20]

**중기 (3-5년)**:
- 멀티태스크 기초 모델 통합
- 극한 도메인 일반화 해결
- 윤리적 AI 프레임워크 정립

**장기 (5년+)**:
- 명시적 학습 없는 영점 분할
- 자율 시스템의 지각 모듈로 통합
- 인간-AI 협업 상호작용의 자연화

### 10.4 최종 평가

SEEM은 단순히 하나의 모델이 아니라, **분할 연구의 새로운 생태계**를 제시한다. LLM이 자연어처리를 변혁했듯이, 이 작업이 컴퓨터 비전의 미래를 형성할 것으로 예상된다.[1]

***

## 참고 자료 목록

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c9b0f8a3-2bd4-4e02-b6a9-da6ccc4d1539/2304.06718v4.pdf)
[2](https://dl.acm.org/doi/10.1145/3746224)
[3](https://openaccess.thecvf.com/content/CVPR2023/papers/Zou_Generalized_Decoding_for_Pixel_Image_and_Language_CVPR_2023_paper.pdf)
[4](https://ieeexplore.ieee.org/document/11012116/)
[5](https://arxiv.org/abs/2306.09316)
[6](https://huggingface.co/docs/transformers/model_doc/detr)
[7](https://arxiv.org/abs/2304.02643)
[8](https://arxiv.org/abs/2408.00714)
[9](https://arxiv.org/pdf/2306.01567.pdf)
[10](https://arxiv.org/abs/2404.15028)
[11](https://arxiv.org/pdf/2401.04651.pdf)
[12](https://arxiv.org/html/2503.19647v1)
[13](https://arxiv.org/abs/2501.11734)
[14](https://arxiv.org/abs/2509.02379)
[15](http://arxiv.org/pdf/2306.00499.pdf)
[16](https://openaccess.thecvf.com/content/CVPR2024/papers/Zhang_Improving_the_Generalization_of_Segmentation_Foundation_Model_under_Distribution_Shift_CVPR_2024_paper.pdf)
[17](https://openaccess.thecvf.com/content/CVPR2023W/L3D-IVU/papers/Shin_Zero-Shot_Unsupervised_Transfer_Instance_Segmentation_CVPRW_2023_paper.pdf)
[18](https://papers.miccai.org/miccai-2025/paper/5230_paper.pdf)
[19](https://arxiv.org/html/2408.09886v1)
[20](http://arxiv.org/pdf/2312.13789.pdf)
[21](https://ieeexplore.ieee.org/document/11082481/)
[22](https://ieeexplore.ieee.org/document/10640396/)
[23](https://onlinelibrary.wiley.com/doi/10.1002/esp.6053)
[24](https://ieeexplore.ieee.org/document/10981021/)
[25](https://arxiv.org/abs/2508.20909)
[26](https://arxiv.org/html/2507.16406v1)
[27](https://arxiv.org/abs/2502.20749)
[28](https://qims.amegroups.com/article/view/138057/html)
[29](https://arxiv.org/abs/2503.24368)
[30](https://arxiv.org/pdf/2408.12957.pdf)
[31](https://arxiv.org/html/2502.20749)
[32](http://arxiv.org/pdf/2310.10912.pdf)
[33](http://arxiv.org/pdf/2404.13239.pdf)
[34](https://arxiv.org/pdf/2401.11311.pdf)
[35](https://arxiv.org/pdf/2304.12306.pdf)
[36](http://arxiv.org/pdf/2404.09957.pdf)
[37](https://arxiv.org/pdf/2311.01989.pdf)
[38](https://hiringnet.com/image-segmentation-state-of-the-art-models-in-2025)
[39](https://arxiv.org/html/2408.15178v1)
[40](https://arxiv.org/abs/2307.09220)
[41](https://arxiv.org/html/2408.12957v1)
[42](https://www.arxiv.org/abs/2312.04089)
[43](https://www.nature.com/articles/s41467-024-44824-z)
[44](https://huggingface.co/docs/transformers/en/model_doc/detr)
[45](https://openaccess.thecvf.com/content/CVPR2024/html/Liu_Open-Vocabulary_Segmentation_with_Semantic-Assisted_Calibration_CVPR_2024_paper.html)
[46](https://averroes.ai/blog/best-image-segmentation-models)
[47](https://arxiv.org/html/2510.07041v1)
[48](https://arxiv.org/abs/2005.12872)
[49](https://arxiv.org/abs/2311.17095)
[50](https://arxiv.org/pdf/2510.09586.pdf)
[51](https://arxiv.org/html/2304.09854v4)
[52](https://arxiv.org/html/2210.00379v7)
[53](https://github.com/daekeun-ml/gitbook/blob/master/ml/computer-vision-transformer-based/detr-for-object-detection.md)
[54](https://kimjy99.github.io/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0/detr/)
[55](https://arxiv.org/abs/2408.06305)
[56](https://www.mdpi.com/2306-5338/11/2/17)
[57](https://sol.sbc.org.br/index.php/sibgrapi_estendido/article/view/31661)
[58](https://ieeexplore.ieee.org/document/10626911/)
[59](https://pubs.aip.org/jasa/article/156/4_Supplement/A48/3330970/Implementing-artificial-intelligence-models-for)
[60](https://ieeexplore.ieee.org/document/10495706/)
[61](https://ieeexplore.ieee.org/document/10803288/)
[62](https://ieeexplore.ieee.org/document/10678549/)
[63](https://arxiv.org/html/2307.04767)
[64](https://arxiv.org/pdf/2401.10228.pdf)
[65](http://arxiv.org/pdf/2408.06305.pdf)
[66](http://arxiv.org/pdf/2304.09324v1.pdf)
[67](http://arxiv.org/pdf/2406.09627.pdf)
[68](https://arxiv.org/pdf/2410.09714.pdf)
[69](https://encord.com/blog/segment-anything-model-explained/)
[70](https://www.youtube.com/watch?v=TxT6GfVHZAg)
[71](https://papers.miccai.org/miccai-2024/paper/0293_paper.pdf)
[72](https://jliu4ai.github.io/cplot_projectpage/)
[73](https://github.com/microsoft/X-Decoder)
[74](https://pmc.ncbi.nlm.nih.gov/articles/PMC12128912/)
[75](https://blog.roboflow.com/how-to-use-segment-anything-model-sam/)
[76](https://arxiv.org/abs/2306.01567)
[77](https://arxiv.org/html/2403.09199v1)
[78](https://arxiv.org/abs/2304.12308)
[79](https://openaccess.thecvf.com/content/ICCV2025/papers/Liu_Unified_Open-World_Segmentation_with_Multi-Modal_Prompts_ICCV_2025_paper.pdf)
[80](https://arxiv.org/abs/2212.11270)
[81](https://arxiv.org/html/2506.11170v2)
[82](https://arxiv.org/abs/2304.06718)
[83](https://arxiv.org/abs/2211.15037)
[84](https://academic.oup.com/jac/article-lookup/doi/10.1093/jac/dkt306)
[85](https://www.semanticscholar.org/paper/3dad8904e9e547cc328a300cab5134ee394d2020)
[86](https://www.semanticscholar.org/paper/1f2121d173a9a491d73151bb0605b607f3e5a2b5)
[87](https://www.semanticscholar.org/paper/9b7f9fba5c87ae010b5d8ac4de588f5f4df24d10)
[88](http://www.tandfonline.com/doi/abs/10.1080/10163279209464434)
[89](https://www.jstor.org/stable/357627?origin=crossref)
[90](https://ualberta.scholaris.ca/handle/123456789/3628)
[91](https://www.semanticscholar.org/paper/25099230b702ebf20a89b3f3fe8b17d446438699)
[92](https://arxiv.org/html/2312.09128)
[93](https://arxiv.org/html/2403.14103)
[94](https://arxiv.org/html/2404.05331v1)
[95](https://arxiv.org/html/2502.00630v1)
[96](https://arxiv.org/pdf/2112.12782.pdf)
[97](https://proceedings.neurips.cc/paper_files/paper/2023/file/3ef61f7e4afacf9a2c5b71c726172b86-Paper-Conference.pdf)
[98](https://www.reddit.com/r/MachineLearning/comments/12lf2l3/r_seem_segment_everything_everywhere_all_at_once/)
[99](https://www.emergentmind.com/topics/promptable-segmentation)
[100](https://arxiv.org/abs/2503.00450)
[101](https://github.com/UX-Decoder/Segment-Everything-Everywhere-All-At-Once)
[102](https://arxiv.org/abs/2404.11732)
[103](https://www.sciencedirect.com/science/article/abs/pii/S0925231224006003)
[104](https://arxiv.org/pdf/2506.02854.pdf)
[105](https://arxiv.org/html/2508.03300v1)
[106](https://arxiv.org/html/2508.04424v2)
[107](https://arxiv.org/html/2505.11980v1)
[108](https://openaccess.thecvf.com/content/ICCV2025/papers/Cao_Refer_to_Any_Segmentation_Mask_Group_With_Vision-Language_Prompts_ICCV_2025_paper.pdf)
[109](https://arxiv.org/html/2507.09562v1)
[110](http://papers.neurips.cc/paper/8338-zero-shot-semantic-segmentation.pdf)
