# FlexCap: Describe Anything in Images in Controllable Detail

FlexCap은 웹 규모의 이미지–텍스트 데이터를 이용해 “박스 단위(localized)로, 원하는 길이(정보량)만큼” 캡션을 생성하는 길이‑제어(region‑level controllable) 캡션러를 제안하고, 이를 LLM과 조합해 다양한 VQA·Dense Captioning·Open‑ended Detection·Video QA까지 강한 제로샷 성능을 보인다는 것이 핵심 주장입니다.[^1][^2][^3][^4]

***

## 1. 핵심 주장과 주요 기여 (간단 요약)

- **핵심 주장**
    - 단일 모델 FlexCap이 “이미지 내 임의의 박스(객체/영역)”를 입력받아, **단어 수로 직접 제어되는 길이‑조건부(length‑conditioned) 캡션**을 생성할 수 있으며, 이를 통해 짧은 카테고리 레이블부터 풍부한 문장 수준까지 **정보량을 연속적으로 조절하는 flexible captioning**이 가능하다고 주장합니다.[^3][^4][^1]
    - 이 지역 캡션들을 LLM 입력으로 사용하면, 별도의 멀티모달 파라미터 튜닝 없이도 기존 SOTA VLM에 필적하거나 일부 벤치마크에서 능가하는 **제로샷 VQA/Video‑QA/Visual Dialog 성능**을 달성한다고 보고합니다.[^5][^1][^3]
- **주요 기여**[^4][^1][^3]

1. **Controllable localized captioning**
        - 단어 수를 표현하는 length token을 입력 프리픽스로 넣어 박스 단위 캡션의 길이와 정보량을 세밀하게 제어.
2. **Localized Captions Dataset (대규모 박스‑캡션 3쌍)**
        - WebLI·YFCC100M 등의 웹 캡션 이미지에서 OWL‑ViT 기반 텍스트‑박스 매칭으로 **32B(WebLI) + 0.2B(YFCC)** 수준의 image–box–caption triplet 생성.[^1][^3]
3. **아키텍처 및 학습**
        - SigLIP 기반 SO‑ViT‑400M/14 비전 인코더 + 디코더‑only Transformer 텍스트 디코더(총 590M 파라미터)로 구성된 지역 캡션링 모델 설계 및 길이‑조건부 NLL 학습.[^1]
4. **성능**
        - Visual Genome dense captioning, MS‑COCO region classification 등에서 기존 RegionCLIP·GRiT 등을 능가하는 성능.[^1]
        - FlexCap‑LLM 조합이 VQAv2, GQA, OK‑VQA, VizWiz, MSRVTT‑QA, MSVD‑QA 등 다양한 VQA/Video‑QA 벤치마크에서 강한 제로샷 성능.[^3][^4][^1]
5. **프리픽스 기반 속성 추출 / 설명 형식 제어**
        - “The color is …”, “This is used for …”, “The photo was taken in …” 등의 프리픽스를 통해 **색상, 재질, 용도, 장소, 텍스트(OCR)** 등 다양한 속성을 동일 모델로 추출 가능함을 시연.[^1]

***

## 2. 논문이 해결하려는 문제

### 2.1 문제 정의

기존 이미지 캡션/비전‑언어 모델의 한계는 크게 세 가지로 요약됩니다.[^3][^1]

1. **전역 캡션에 치우침**
    - 대부분 “전체 이미지”에 대한 한두 개의 캡션만 생성하며, **지역(박스) 단위로 정보량을 세밀하게 제어**하기 어렵습니다.[^4][^3]
2. **길이 제어의 한계**
    - LIC 등 기존 length‑controllable 캡셔들은 대개 **이미지 전체 캡션**을 coarse‑grained 수준(short/medium/long 등)으로만 제어합니다.[^6][^7]
    - 세밀한 단어 수 수준의 제어 및 **지역 캡션 + 길이 제어를 동시에 지원**하는 모델은 부족했습니다.[^3][^1]
3. **VLM과 LLM 결합 방식의 제약**
    - Flamingo, PaLI, BLIP‑2, InstructBLIP 등은 **시각·언어를 단일 거대 멀티모달 모델 내에서 결합**하거나, 이미지 임베딩을 LLM 토큰으로 넣는 방식이라 해석 가능성 및 모듈성 측면에서 제약이 있습니다.[^2][^4][^3]
    - **“이미지를 사람이 읽을 수 있는 풍부한 텍스트 구조(지역 캡션 집합)로 변환한 후 LLM에 넘기는 파이프라인”**은 상대적으로 덜 탐구되었습니다.[^8][^1]

FlexCap은 **“임의 박스에 대해, 원하는 길이만큼, 컨텍스트를 반영한 캡션을 생성하는 지역 캡션러”**를 만들고, 이것을 LLM 앞단의 범용 시각 모듈로 쓰는 것을 목표로 합니다.[^5][^3][^1]

***

## 3. 제안 방법: 길이‑조건부 지역 캡션링

### 3.1 데이터셋 및 문제 설정

하나의 학습 샘플은 이미지 $X$, 박스 좌표 $B$, 토큰화된 캡션 시퀀스 $W$의 삼중 튜플로 표현됩니다.[^1]

$$
T = (X, B, W), \quad W = \{\text{LENGTH-}K, w_1, w_2, \dots, w_k\} 
$$

여기서 LENGTH‑ $K$ 토큰은 **타겟 캡션의 단어 수 $K$** 를 의미하는 특별 토큰이며, 모형은 **해당 길이로 끝나도록** 다음 단어를 생성하도록 학습됩니다.[^1]

데이터셋 전체는 $\mathcal{D} = \{T_1, \dots, T_N\}$ 입니다.[^1]

- WebLI 기반 Localized Captions Dataset: 약 **32B image–box–caption triplets**.[^3][^1]
- YFCC100M 기반 Localized Captions Dataset: 약 **0.2B triplets**.[^3][^1]

이 triplet은 다음 파이프라인으로 자동 생성됩니다.[^3][^1]

1. 이미지‑캡션 쌍에서 n‑gram($n=1,\dots,8$)을 추출.
2. 비문장/비정보적인 n‑gram 필터링(전치사로 시작, 관사로 끝나는 표현 등 제거).[^1]
3. OWL‑ViT 같은 open‑vocabulary detector에 n‑gram을 질의해 박스 후보 및 similarity score를 얻고, 일정 threshold 이상인 text–box 매칭을 채택.[^1]
4. 동일 박스에 대해 여러 길이의 캡션(“dog” / “brown dog” / “brown dog playing with a frisbee”)을 보존해 **한 박스를 여러 길이로 설명하는 supervision**을 제공.[^1]

### 3.2 모델 구조 (FlexCap 아키텍처)

모델은 크게 비전 인코더와 텍스트 디코더 두 부분으로 구성됩니다.[^1]

1. **비전 인코더**
    - SigLIP으로 contrastive pretraining된 SO‑ViT‑400M/14를 사용.[^3][^1]
    - 입력 이미지 $X$를 패치 토큰 시퀀스 $\{v_1, \dots, v_n\}$ (차원 $d_v$)로 인코딩.[^1]
2. **박스 좌표 인코딩**
    - 박스 $B = (x_1, y_1, x_2, y_2)$를 정규화 후 선형층으로 투영해 **좌표 토큰** $b \in \mathbb{R}^{d_v}$ 생성.[^1]
3. **텍스트 디코더**
    - 디코더‑only Transformer (12‑layer, hidden dim=768, 12 heads) 구조.[^1]
    - 비전/박스 토큰은 **언마스킹된 상태**로, 텍스트 토큰은 causal mask를 적용해 **다음 단어 예측**.[^1]
    - 비전 인코더 출력(예: 1152차원)을 768차원으로 투영 후 텍스트 디코더 입력으로 사용.[^1]
4. **입력 시퀀스 구성**
    - 입력 토큰 시퀀스는

$$
[v_1, \dots, v_n, b, \text{LENGTH-}K, w_1, \dots, w_M] 
$$

형태이며, 여기서 $M$은 패딩 포함 최대 길이입니다.[^1]

### 3.3 학습 목표 (수식)

각 샘플 $T_j = (X_j, B_j, W_j)$ 에서 패딩 포함 텍스트 길이를 $M$이라 할 때, 기본 objective는 **길이‑조건부 다음 단어 예측 log‑likelihood**입니다.[^1]

$$
\ell(X_j, B_j, W_j)
= \sum_{i=1}^{M} \log p\left( (w_j)_i \mid (w_j)_{<i}, X_j, B_j \right) 
$$

전체 데이터셋 손실은

$$
\mathcal{L}(\mathcal{D})
= \sum_{j=1}^{N} \ell(X_j, B_j, W_j)
= \sum_{j=1}^{N} \sum_{i=1}^{M} \log p\left( (w_j)_i \mid (w_j)_{<i}, X_j, B_j \right) 
$$

이며, 패딩 토큰에 대한 loss는 무시합니다.[^1]

학습 세부사항:[^1]

- Optimizer: AdamW, cosine LR schedule, peak LR $1.6\times 10^{-4}$, 10k warmup.[^1]
- Batch size 4096, 이미지 해상도 224×224, 텍스트 길이 32 토큰.[^1]
- 각 이미지당 최대 8개 박스를 샘플링해 학습.[^1]


### 3.4 길이‑조건부 토큰의 역할과 프리픽싱

```
- 기존에는 BOS 토큰 \<s\> 뒤에 캡션을 두면, 동일 박스에 대해 \<s>dog\<e>, \<s>dog playing\<e>, \<s>dog playing with a frisbee\<e> 같은 상황에서 prefix 충돌로 loss 상 모호성이 생깁니다.[^1]
```

- Length token을 BOS 대신 사용하면, 서로 다른 길이/내용의 캡션간 prefix 중복이 크게 줄어들며, 실제로 특정 데이터셋에서 caption pair의 prefix 일치 비율이 **30.8% → 11.1%로 감소**했다고 보고합니다.[^1]
- 동시에, length token을 더 복잡한 프리픽스로 확장(예: “LENGTH‑4 The color is …”)함으로써, **속성 질의(attribute querying)**와 텍스트 형식 제어까지 가능하게 합니다.[^1]

***

## 4. FlexCap‑LLM 파이프라인과 성능

### 4.1 FlexCap‑LLM: 지역 캡션 + LLM 기반 VQA/Video‑QA

1. **객체/영역 추출**
    - OWL‑ViT v2 등의 open‑vocab detector로 이미지에서 상위 128개 박스를 추출.[^1]
2. **지역 캡션 생성**
    - 각 박스에 대해 여러 길이·프리픽스 조합으로 지역 캡션을 생성 (예: LENGTH‑4, LENGTH‑8 등).[^1]
    - 박스 중심 좌표 $[c_x, c_y, w, h]$와 함께 “객체 목록” 텍스트로 serialize.[^3][^1]
3. **LLM의 입력으로 사용**
    - PALM2‑S 등 LLM에 **(설명 프리앰블 + 이미지 크기 + 전체 이미지 설명 + 객체 목록 + 질문)**을 하나의 텍스트 프롬프트로 전달.[^3][^1]
    - LLM은 pure‑text 모드에서 답변을 생성하므로, **멀티모달 파라미터 튜닝 없이** 제로샷 VQA가 가능.[^1]

### 4.2 정량 성능 (요약)

- **MS‑COCO Region Classification (GT 박스)**
    - FlexCap Top‑1: 72.0 mAP, Top‑20: 85.0 mAP으로 CLIP/CLIM/RegionCLIP보다 큰 폭으로 우수.[^1]
- **Dense Captioning (Visual Genome)**
    - GT 박스 기준 mAP 46.9로 기존 FCLN, CAG‑Net 등을 상회.[^1]
    - GRiT 박스 기반 dense captioning에서도 FlexCap+GRiT가 GRiT 자체를 능가.[^1]
- **Zero‑shot Image VQA (FlexCap‑LLM)**[^1]
    - VQAv2 Test‑dev: 65.6% (BLIP‑2 65.2%보다 약간 우수).
    - GQA: 48.8%로 ViperGPT(48.1%)와 비슷, InstructBLIP(49.5%)에 근접.
    - OK‑VQA: 52.1%로 Flamingo, BLIP‑2, ViperGPT 등 기존 zero‑shot/weakly‑supervised 시스템을 상회.
    - VizWiz: 41.8%로 Flamingo/Emu/InstructBLIP zero‑shot보다 우수.[^1]
- **Zero‑shot Video QA (MSRVTT‑QA, MSVD‑QA)**
    - 각 비디오에서 8 프레임 샘플 후 프레임별 지역 캡션을 생성해 LLM에 입력.
    - Flamingo 대비 상당한 성능 향상을 보고 (예: MSRVTT‑QA 25.0% vs Flamingo 17.4%).[^1]
- **Open‑ended Object Detection**
    - describe‑then‑localize(LLAVA + OWL‑ViT)와 localize‑then‑describe(FlexCap + OWL‑ViT)를 비교했을 때,
        - FlexCap 기반 localize‑then‑describe가 **특히 small/medium 객체에서 훨씬 높은 recall**과 더 많은 유효 박스를 검출.[^1]

이 결과를 통해, **“지역 캡션 → LLM” 파이프라인이 강한 일반화 성능을 가지며, 기존 엔드‑투‑엔드 VLM과 경쟁 가능하다**고 주장합니다.[^5][^4][^1]

### 4.3 한계

논문이 명시하는 주요 한계는 다음과 같습니다.[^1]

- **데이터 편향 및 노이즈**
    - 학습 데이터는 WebLI alt‑text와 OWL‑ViT 기반 pseudo‑label에 의존하므로, alt‑text의 편향·부정확성이 그대로 전파될 수 있습니다.[^1]
- **비 end‑to‑end 구조**
    - FlexCap‑LLM은 detector + captioner + LLM의 **파이프라인**으로, end‑to‑end 미분 가능 구조가 아닙니다.[^1]
    - 추후에는 localized captioning 데이터셋을 사용해 진정한 end‑to‑end VLM을 학습하는 방향을 제안합니다.[^1]
- **WebLI 데이터 비공개**
    - 핵심 32B triplets는 WebLI 기반이라 외부 연구자가 완전히 재현하기 어렵고, 대신 YFCC100M 기반 0.2B triplets를 공개 가능한 대안으로 제시합니다.[^1]

***

## 5. 일반화 성능(Generalization) 향상 관점에서의 분석

### 5.1 스케일링 법칙과 일반화

논문은 contrastive pretraining, 데이터 규모, 모델 규모 스케일링이 **지역 캡션 generalization**에 미치는 영향을 ablation으로 분석합니다.[^1]

1. **Contrastive Pretraining (CPT)**
    - 동일 구조에서 CPT 없이 WebLI region‑caption만으로 학습한 경우와 SigLIP CPT로 초기화한 경우를 비교.[^1]
    - CPT 사용 시 Visual Genome GT box captioning mAP가 **43.6 → 45.1**로 개선되며, dense captioning에서도 소폭 향상.[^1]
→ 의미: **이미지–텍스트 contrastive pretraining + localized caption fine‑tuning** 조합이 region‑level 언어 이해 일반화를 크게 돕습니다.
2. **Data Scaling**
    - YFCC100M 기반 0.2B triplets vs WebLI 기반 32B triplets 비교에서, GT box captioning mAP가 **38.5 → 45.1**로 약 6.6p 상승.[^1]
→ 의미: **박스 수준의 캡션 데이터도 일반적인 “더 큰 데이터 → 더 나은 일반화” 스케일링 법칙**을 따름을 실증.
3. **Model Scaling**
    - ViT‑B/16 (86M) vs SO‑ViT/14 (428M) 비교에서, GT box captioning mAP가 **45.1 → 46.9**, dense captioning mAP도 향상.[^1]
→ 의미: 대형 비전 backbone이 지역 캡션 generalization을 위한 더 풍부한 표현을 제공합니다.

### 5.2 Cross‑task 일반화

FlexCap‑LLM이 **“단일 vision front‑end + 동일 LLM” 구조로, task‑specific fine‑tuning 없이** 다양한 벤치마크에서 준 SOTA 성능을 내는 점이 일반화 관점에서 중요합니다.[^5][^1]

- 이미지 VQA(“일반” + compositional + external knowledge + atypical images), Video‑QA, dense captioning, open‑ended detection 등에 모두 적용 가능.[^1]
- 특히 VizWiz처럼 **웹 이미지와 분포가 다른 저품질/비정형 이미지**에서도 zero‑shot으로 기존 모델보다 우수한 성능을 보인다는 점이, **FlexCap의 지역 표현이 Distribution shift에 비교적 강건**함을 시사합니다.[^1]


### 5.3 텍스트로의 “완전한 투영”과 LLM 활용

FlexCap은 이미지를 **서로 겹치는 다수의 박스‑캡션 집합**으로 투영합니다.[^1]

- 이 표현은 인간에게 해석 가능하고, LLM이 잘 다루는 토큰 시퀀스이기 때문에, **새로운 다운스트림 태스크(예: Visual Dialog, instruction‑following VQA)로의 전이**가 비교적 용이합니다.[^5][^1]
- 또한, 프리픽스/길이 변조를 통해 “필요한 속성만 빼내기”가 가능하므로,
    - 예: 로봇 조작에서는 “This is used for …” 프리픽스로 affordance를,
    - OCR 태스크에서는 “The sign says …” 프리픽스로 텍스트를 추출.
→ **task‑specific fine‑tuning 없이 프롬프트 수준에서 generalization을 끌어올릴 수 있는 구조**입니다.[^1]

***

## 6. 2020년 이후 관련 최신 연구와의 비교 분석

여기서는 FlexCap과 밀접한 세 가지 축 — (1) 길이‑제어 캡션링, (2) 지역/밀집 캡션 및 open‑vocab detection, (3) VLM‑LLM 결합형 zero‑shot captioning/VQA — 을 중심으로 2020년 이후 대표 연구들을 정리합니다.

### 6.1 길이‑제어 이미지 캡션링 연구

| 논문 | 핵심 아이디어 / FlexCap과의 차이 |
| :-- | :-- |
| **Length‑Controllable Image Captioning (LIC), ECCV 2020** – C. Deng et al.[^6][^9][^7] | 길이 level embedding(short/medium/long 등)을 추가해 전체 이미지 캡션 길이를 제어. FlexCap은 **박스‑단위** 길이 제어 + 단어 수 정확 제어가 가능하다는 점에서 더 fine‑grained. |
| **CLID: Controlled‑Length Image Descriptions with Limited Data, 2022** – H. Benaim et al.[^10] | self‑generated 장문 캡션으로 데이터 희소성 극복, 길이 제어 학습. FlexCap은 **웹‑스케일 박스 캡션 생성 파이프라인**을 제안해 길이·지역 정보 모두를 스케일링. |
| **CIC‑BART‑SSA / CIC‑BART‑SSA, 2024** – controllable contextualized captioning[^11][^12] | 특정 sub‑region을 대상으로 길이‑제어 caption을 생성하는 CIC 모델 제안. SSA로 length 다양성 확대. FlexCap과 유사하게 지역+길이 제어를 수행하지만, FlexCap은 훨씬 큰 웹‑스케일 pseudo‑label 데이터와 open‑vocab detector를 활용. |
| **Flexibly Controlling Language Pattern in Image Captioning, 2025**[^13] | 길이 외에 스타일/구문 패턴까지 제어하는 CIC 확장. FlexCap은 주로 길이/속성 프리픽스에 초점을 둡니다. |

**종합적으로**, 기존 길이‑제어 연구들은 주로 **전체 이미지 캡션 수준**이거나, 소규모 데이터/고정 길이 레벨에 의존하는 경향이 있고, FlexCap은 **박스‑단위 + 웹‑스케일 + 단어 수 기반 정밀 제어**를 제공하며, 이를 LLM 기반 VQA까지 확장했다는 점이 차별점입니다.[^3][^1]

### 6.2 Zero‑shot/컨트롤러블 캡션 + VLM‑LLM 결합

- **ZeroCap: Zero‑Shot Image‑to‑Text Generation for Visual‑Semantic Arithmetic (CVPR 2022) – Y. Tewel et al.**[^14][^15][^16][^17][^18]
    - CLIP + GPT‑2를 조합해 **추가 학습 없이** zero‑shot 캡션을 생성.
    - 이미지 임베딩과 텍스트 임베딩을 묶어 CLIP loss 등을 최적화해 GPT‑2를 “이미지 관련 방향으로” steer.
    - 주로 **전역 캡션**에 초점, region‑level 제어나 길이‑제어는 제공하지 않음. FlexCap은 학습된 region captioner를 사용하지만, 훨씬 강력한 지역 표현과 VQA/generalization을 제공.
- **MacCap: Mining Fine‑Grained Image‑Text Alignment for Zero‑Shot Captioning, 2024**[^19]
    - CLIP sub‑region feature aggregation을 활용해 text‑only 학습된 adaptor decoder로 zero‑shot captioning 수행.
    - MacCap도 sub‑region 정보를 사용하지만, FlexCap처럼 명시적인 **박스‑단위 길이‑조건부 언어 모델**은 아니며, LLM과의 조합도 제한적.
- **Flamingo (2022), PaLI (2022/2023), BLIP/BLIP‑2 (2022–2023), InstructBLIP (2023), Emu (2024)**[^2][^4]
    - 대다수가 **이미지 피처를 LLM 토큰으로 삽입하거나, 크로스‑모달 트랜스포머**를 사용해 end‑to‑end multimodal 모델을 구성.
    - 일반적으로 VQA 성능은 뛰어나지만, **지역 단위 길이‑제어 캡션**이라는 관점에서는 명시적인 제어 기능이 제한적.
- **Ferret: Refer and Ground Anything Anywhere at Any Granularity (2023)**[^2][^1]
    - arbitrary granularity grounding(점/박스/세그먼트)를 다루는 VLM으로, FlexCap과 유사하게 지역 정보를 풍부하게 표현하려는 방향.
    - Ferret은 grounding+captioning, FlexCap은 지역 캡션을 생성해 LLM과 결합한다는 점에서 상호 보완적입니다.[^1]


### 6.3 Dense Captioning / Region‑Level Vision–Language

- **GRiT: Generative Region‑to‑Text Transformer for Object Understanding, 2022**[^4][^1]
    - 제너레이티브 region‑to‑text 트랜스포머로 dense captioning 수행.
    - FlexCap은 GRiT 박스 위에 지역 캡션을 올려 dense captioning을 수행할 때 GRiT보다 좋은 mAP를 보고.[^1]
- **RegionCLIP (CVPR 2022)**[^2][^1]
    - region‑level contrastive pretraining으로 open‑vocab detection 성능 향상.
    - FlexCap은 region‑level contrastive가 아니라 **region‑level caption generation**에 초점을 맞추고, 이를 통해 detection, VQA까지 확장.

**표 형태 요약 (일부)**

- **FlexCap: Describe Anything in Images in Controllable Detail (NeurIPS 2024)** – D. Dwibedi et al.[^4][^2][^5][^3]
    - Source/Link: arXiv 2403.12026, NeurIPS proceedings.
    - Localized length‑controlled captioner + LLM 기반 제로샷 VQA/dense captioning.
- **Length‑Controllable Image Captioning (ECCV 2020)** – C. Deng et al.[^9][^7][^6]
    - Source/Link: ECCV 2020 / arXiv 2007.09580.
    - Length level embedding으로 전역 캡션 길이 제어, non‑AR captioning 제안.
- **ZeroCap (CVPR 2022)** – Y. Tewel et al.[^15][^16][^18][^14]
    - Source/Link: CVPR 2022 / arXiv 2111.14447.
    - CLIP + GPT‑2로 training‑free zero‑shot 캡션, 이미지/텍스트 arithmetic.
- **CLID (2022)** – H. Benaim et al.[^10]
    - Source/Link: arXiv 2211.14835.
    - self‑generated 장문 캡션과 joint training으로 길이‑제어 문제에서 데이터 희소성 해결.
- **CIC‑BART‑SSA (2024)**[^11][^12]
    - Source/Link: arXiv.
    - region‑focused controllable captioning, 길이와 기타 control signal을 함께 사용.

이들 대비 FlexCap의 위치를 정리하면, **“웹‑스케일 localized caption dataset + 길이‑조건부 region captioner + LLM 조합을 통한 광범위 제로샷 전이”**에 가장 초점을 둔 모델이라고 볼 수 있습니다.[^2][^5][^1]

***

## 7. 앞으로의 연구에 미치는 영향과 고려할 점

### 7.1 영향: “범용 시각 프론트엔드”로서의 역할

- FlexCap이 보여준 것처럼, 이미지를 **풍부한 지역 캡션 집합으로 변환한 후 LLM에 넘기는 모듈형 파이프라인**은,
    - 멀티모달 모델 설계에서 “모든 것을 end‑to‑end 하나의 거대 VLM으로 푸는 대신,
**범용 vision tool + 범용 LLM** 조합으로 분해”하는 설계 패턴을 강화할 가능성이 큽니다.[^5][^1]
- 특히 robotics, AR/VR, 의료 영상 등에서 **task‑specific LLM (도메인 지식) + 범용 region captioner** 구조가 보편적인 아키텍처로 자리 잡을 수 있습니다.


### 7.2 향후 연구 방향

1. **End‑to‑end 학습 및 미분 가능 파이프라인**
    - 현재 FlexCap‑LLM은 detector + captioner + LLM이 느슨하게 결합된 비‑end‑to‑end 구조입니다.[^1]
    - 향후 연구에서는 localized caption 데이터셋(또는 All‑seeing/URECA 계열 데이터)를 활용해,[^20][^21]
        - detector와 captioner, LLM 사이를 **joint training**하거나,
        - region caption을 latent로 사용하는 end‑to‑end VLM을 설계하는 방향이 중요합니다.
2. **오픈 데이터/모델로의 재현 가능성 강화**
    - WebLI 기반 32B triplets는 외부에서 그대로 재현하기 어렵기 때문에,
        - YFCC100M 기반 오픈 데이터셋의 품질/규모를 늘리거나[ in file:1],
        - Grounding DINO/All‑Seeing/URECA와 같이 공개된 grounded caption 데이터와 결합해 **open‑source FlexCap‑like 모델**을 구축하는 연구가 필요합니다.[^21][^1]
3. **편향(bias)·공정성(fairness) 분석**
    - alt‑text와 web captions의 편향이 지역 캡션 수준에서 어떻게 나타나는지,
        - 예: 사람을 “policeman / nurse / baby / homeless person” 등으로 구분하는 표현이 인구 집단별로 어떻게 불균형한지 분석할 필요가 있습니다.[^1]
    - accessibility, 감시(surveillance) 등 민감한 응용에서 **지역 단위 인식과 서술이 어떤 사회적 영향을 가지는지** 체계적으로 평가해야 합니다.[^1]
4. **일반화 향상을 위한 pre‑prefix 설계와 훈련 전략**
    - 논문에서 보여준 color/usage/OCR/장소/저자 등 속성 프리픽스는 매우 유용한 제어 인터페이스입니다.[^1]
    - 앞으로는
        - 프리픽스 공간을 **meta‑learning/soft prompt** 형태로 확장,
        - 새로운 태스크가 등장했을 때 수‑샷 예시만으로 속성 프리픽스를 자동 학습하는 방법,
        - 길이와 속성, 스타일을 동시에 제어하는 multi‑dimensional controllable captioning
등을 탐구할 수 있습니다.[^13][^12]
5. **다국어·도메인 특화 확장**
    - PaLI/PaLI‑X 스타일의 다국어 지원이나, 의료/법률·산업 도메인 이미지에 특화된 localized captioning을 학습하면,[^2]
        - **다국어 VQA, domain‑expert LLM과의 결합**에서 FlexCap‑류 모델의 일반화 능력을 크게 확장할 수 있습니다.

### 7.3 연구 시 고려할 점

- **데이터 품질 관리**: 자동 생성된 박스‑캡션은 노이즈가 많으므로,
    - n‑gram 필터링, score threshold, human‑in‑the‑loop 검증 등 품질 관리 전략이 필수입니다.[^1]
- **연산 비용 및 latency**: 지역 캡션을 대량으로 생성해 LLM에 넣으면 토큰 수가 급증하므로,
    - 중요한 객체만 선택하거나, caption 길이를 adaptive하게 조정하는 **정보량‑대‑latency trade‑off 설계**가 필요합니다.[^1]
- **평가 지표 다양화**: 기존 Meteor/mAP/accuracy 외에,
    - ITIScore와 같이 short/long caption 품질을 세분화 평가하는 벤치마크,[^22]
    - human preference 기반 평가를 병행해 caption의 유용성/해석 가능성을 함께 고려해야 합니다.

***

## 참고한 주요 자료(제목/링크)

(요구사항에 따라, 본 답변에서 직접 인용하거나 비교 분석에 사용한 대표적인 오픈 액세스 논문들의 제목과 링크를 나열합니다. 각 문헌은 본문 내에, [web:*] 형태로도 인용되어 있습니다.)[^1]

- **FlexCap: Describe Anything in Images in Controllable Detail** – Debidatta Dwibedi et al., NeurIPS 2024, arXiv:2403.12026.[^23][^4][^2][^5][^3][^1]
- **Length‑Controllable Image Captioning** – Chaorui Deng et al., ECCV 2020, arXiv:2007.09580.[^7][^6][^9]
- **Zero‑Shot Image‑to‑Text Generation for Visual‑Semantic Arithmetic (ZeroCap)** – Yoad Tewel et al., CVPR 2022, arXiv:2111.14447.[^16][^18][^14][^15]
- **CLID: Controlled‑Length Image Descriptions with Limited Data** – Hila Ben‑Amin et al., arXiv:2211.14835.[^10]
- **CIC‑BART‑SSA: Controllable Image Captioning with Structured Spatial Anchors** – recent arXiv preprint.[^11]
- **Controllable Contextualized Image Captioning: Directing the Visual Narrative** – arXiv preprint.[^12]
- **Mining Fine‑Grained Image‑Text Alignment for Zero‑Shot Captioning (MacCap)** – arXiv:2401.02347.[^19]
- **ITIScore: An Image‑to‑Text‑to‑Image Rating Framework for Image Captioning** – arXiv preprint.[^22]
<span style="display:none">[^24][^25][^26][^27][^28][^29][^30][^31]</span>

<div align="center">⁂</div>

[^1]: 2403.12026v2.pdf

[^2]: https://arxiv.org/abs/2403.12026

[^3]: https://arxiv.org/html/2403.12026v2

[^4]: https://proceedings.neurips.cc/paper_files/paper/2024/file/c91b6f7e0152b7a95ee777e987fe811e-Paper-Conference.pdf

[^5]: https://openreview.net/forum?id=P5dEZeECGu

[^6]: https://arxiv.org/abs/2007.09580

[^7]: https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123580698.pdf

[^8]: https://arxiv.org/abs/2209.14491

[^9]: https://www.semanticscholar.org/paper/Length-Controllable-Image-Captioning-Deng-Ding/2320f853059c29ce7e70409fa559074d727da5a2

[^10]: https://arxiv.org/html/2211.14835v2

[^11]: https://arxiv.org/html/2407.11393v2

[^12]: https://arxiv.org/html/2407.11449v1

[^13]: https://arxiv.org/html/2507.01409v1

[^14]: https://arxiv.org/abs/2111.14447

[^15]: https://openaccess.thecvf.com/content/CVPR2022/papers/Tewel_ZeroCap_Zero-Shot_Image-to-Text_Generation_for_Visual-Semantic_Arithmetic_CVPR_2022_paper.pdf

[^16]: https://www.semanticscholar.org/paper/ZeroCap:-Zero-Shot-Image-to-Text-Generation-for-Tewel-Shalev/a2502d2cd7144c5e2bc1d0d7ec37d2c84b37d381

[^17]: https://velog.io/@hmym7308/논문-리뷰-Zerocap-Zero-Shot-Image-to-Text-Generation-for-Visual-Semantic-Arithmetic-CVPR-2022

[^18]: https://github.com/yoadtew/zero-shot-image-to-text

[^19]: https://arxiv.org/html/2401.02347v1

[^20]: https://arxiv.org/html/2510.02898v1

[^21]: https://arxiv.org/html/2504.05305v1

[^22]: https://arxiv.org/html/2604.03765v1

[^23]: https://arxiv.org/pdf/2403.12026.pdf

[^24]: https://arxiv.org/html/2501.00437v1

[^25]: https://arxiv.org/pdf/2212.08985.pdf

[^26]: https://arxiv.org/pdf/2305.16311.pdf

[^27]: https://arxiv.org/pdf/2203.04705.pdf

[^28]: http://arxiv.org/pdf/2405.07423.pdf

[^29]: https://tanmingkui.github.io/files/publications/Image_Captioning.pdf

[^30]: https://dl.acm.org/doi/10.5555/3737916.3741446

[^31]: https://aclanthology.org/2021.acl-long.157.pdf

