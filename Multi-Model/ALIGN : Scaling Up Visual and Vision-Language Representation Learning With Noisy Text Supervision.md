# ALIGN : Scaling Up Visual and Vision-Language Representation Learning With Noisy Text Supervision

# 핵심 요약 및 주요 기여

**주요 주장:** 대규모의 노이즈가 포함된 이미지–텍스트 데이터(1.8B alt-text 쌍)만으로도, 정교한 전처리나 크리닝 없이 단순한 듀얼 인코더 및 대조 학습(contrastive learning) 기법을 사용해 시각 및 비전-언어 표현을 효과적으로 학습할 수 있으며, 이는 기존 최첨단(SOTA) 모델들을 능가하거나 견줄 만한 성능을 보여준다는 점이다.[1]

**주요 기여:**  
- 대규모(1.8B) 노이즈 이미지–텍스트 코퍼스 수집 및 최소 필터링 전략 제안  
- EfficientNet 기반 이미지 인코더 + BERT 기반 텍스트 인코더의 듀얼-인코더 구조 설계  
- 대조 학습을 위한 정규화 소프트맥스(normalized softmax) 대조 손실 함수 도입  
- 제로-샷 이미지 분류 및 이미지-텍스트 검색, 크로스 모달 검색(task)에서 SOTA 달성  
- 소규모 미세 조정(fine-tuning) 없이도 뛰어난 일반화 성능 입증  

# 상세 설명

## 1. 해결하고자 하는 문제  
시각 및 비전-언어 표현 학습은 전통적으로 대규모 라벨링된 데이터(예: ImageNet, MSCOCO)나 복잡한 크리닝 과정을 거친 데이터셋(예: Conceptual Captions)을 필요로 한다. 이러한 고품질 데이터셋은 수집·정제 비용이 높아, 규모 확장이 어려운 한계가 있었다.[1]

## 2. 제안하는 방법 및 수식  
ALIGN(A Large-scale ImaGe and Noisy-text embedding)은 다음과 같다.

-  데이터셋: 웹에서 수집한 1.8B 이미지–alt-text 쌍, 빈도 기반 최소 필터링만 수행.[1]
-  모델 구조:  
  - 이미지 인코더 $$f_I$$: EfficientNet (global pooling)  
  - 텍스트 인코더 $$f_T$$: BERT의 CLS 토큰 임베딩, 단어피스 어휘 100K  
  - 두 인코더 출력은 L2 정규화 후 동일 차원으로 매핑  
-  대조 손실(contrastive loss):  
  - 이미지→텍스트 분류 손실  

$$
      \mathcal{L}_{I2T} = -\frac{1}{N}\sum_{i=1}^N \log \frac{\exp(\mathbf{x}_i^\top \mathbf{y}_i / \tau)}{\sum_{j=1}^N \exp(\mathbf{x}_i^\top \mathbf{y}_j / \tau)}
    $$  
  
  - 텍스트→이미지 분류 손실  

$$
      \mathcal{L}_{T2I} = -\frac{1}{N}\sum_{i=1}^N \log \frac{\exp(\mathbf{y}_i^\top \mathbf{x}_i / \tau)}{\sum_{j=1}^N \exp(\mathbf{y}_i^\top \mathbf{x}_j / \tau)}
    $$  
  
  - 최종 손실: $$\mathcal{L} = \mathcal{L}\_{I2T} + \mathcal{L}_{T2I}$$  
  여기서 $$\mathbf{x}_i, \mathbf{y}_i$$는 각각 이미지·텍스트 임베딩, $$\tau$$는 학습 가능한 온도(temperature) 변수이다.[1]

## 3. 모델 구조  
ALIGN은 순수 듀얼-인코더 구조로, 복잡한 크로스-어텐션 레이어 없이도 효율적이며 대규모 이미지-텍스트 검색 시스템에 실시간 적용이 가능하다.[1]

# 성능 향상 및 한계

## 1. 성능 향상  
- 이미지-텍스트 검색(Flickr30K, MSCOCO)에서 정밀도(R@1) 기준 SOTA 달성, CLIP 대비 7%p 이상 개선.[1]
- 제로-샷 ImageNet 분류: top-1 정확도 76.4% 기록, CLIP(76.2%)와 유사하거나 상회.[1]
- ImageNet 미세 조정 후: top-1 88.64%, 다양한 변형(ImageNet-R, A, V2)에도 높은 강인성 보유.[1]
- VTAB, Oxford Flowers, Stanford Cars 등 소규모 데이터셋 전이 학습에서도 기존 비지도·반지도 학습 모델들과 유사하거나 우수하게 비교됨.[1]

## 2. 한계  
- **내재적 노이즈**: 최소 필터링으로 데이터 규모는 확대했으나 잘못된 캡션이 포함되어 있어 일부 과제(문장 간 유사도, intra-modal matching)에서는 학습 목적과 어긋날 수 있음.[1]
- **언어·문화 편향**: 웹 크롤링 기반 데이터는 특정 언어·문화권에 치우칠 수 있으며, 안전성·공정성(fairness) 관점의 추가 분석이 필요.[1]
- **민감 정보 오용 우려**: 대규모 미디어 검색·분류 모델로서 감시(surveillance) 등 악용 가능성 존재.[1]

# 일반화 성능 향상 관점

ALIGN의 가장 두드러진 강점은 **데이터 규모가 클수록 노이즈가 상쇄되어 일반화 성능이 지속 개선**된다는 점이다.  
- Conceptual Captions(3M) vs. ALIGN(3M, 6M, 12M): 동일 크기(3M)에서는 노이즈 때문에 성능이 낮으나, 6M 이상부터 CC-3M 성능을 빠르게 추격·상회함.[1]
- 대규모 학습 시 더 큰 백본(EfficientNet-L2, BERT-Large) 사용 시 일반화 성능이 크게 상승, 이른바 “노이즈 대 규모” 트레이드오프에서 규모가 결정적 역할 수행.[1]

# 앞으로의 영향 및 고려 사항

이 논문은 **노이즈 다량 데이터**와 **단순 대조 학습**으로 시각·비전-언어 표현 학습의 패러다임을 확장했다.  
- **영향:**  
  - 무라벨(raw) 웹 데이터 활용 가능성 제고, 데이터 수집 비용·장벽 대폭 감소  
  - 크로스-모달 검색, 제로-샷 분류 등 응용 분야 확산  
- **향후 연구 고려:**  
  - 데이터 편향·공정성(fairness) 분석 및 보정  
  - 노이즈 유형별 영향 정량적 평가 및 적응적 필터링 전략  
  - 소규모 전문 영역(의료·과학) 데이터에 대한 전이 성능 검증  
  - 안전·윤리적 사용 가이드라인 마련  

위와 같은 방향을 통해 ALIGN 방식의 **확장성·일반화 가능성**을 보다 안전하고 효율적으로 제고할 수 있을 것이다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/d91e7947-0dfe-41cd-9e0e-943f929e5086/2102.05918v2.pdf)
