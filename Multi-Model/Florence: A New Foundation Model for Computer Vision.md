# Florence: A New Foundation Model for Computer Vision

**핵심 주장 및 주요 기여**  
Florence는 대규모 웹 이미지·텍스트 데이터(약 9억 쌍)를 활용해 단일 비전 재단 모델을 구축함으로써, 장면 수준 분류에서 객체 검출, 정적 이미지에서 동적 비디오, RGB에서 깊이·캡션 등 다양한 시공간·모달리티 과제를 하나의 모델로 처리할 수 있음을 제안한다.[1]
- **범용 표현 학습**: 이미지-텍스트 대조학습(UniCL)으로 장·단 문장 모두 활용하는 통합 목적 함수 도입.[1]
- **모델 구조**: 두 개의 토너(tower) 아키텍처(이미지용 CoSwin, 언어용 12-layer Transformer)와 동적 헤드(object-level), METER(VL), Video CoSwin(비디오) 어댑터 확장.[1]
- **성능**: ImageNet-1K 제로샷 83.74% top-1, COCO 객체 검출 62.4 mAP, VQA 80.36, Kinetics-600 87.8 등 44개 벤치마크 대다수에서 SOTA 경신.[1]
- **효율성**: ZeRO, mixed-precision, activation checkpointing 등으로 512 A100 GPU에서 10일 내 학습, 적은 커스터마이징으로 다양한 과제에 전이 학습 가능.[1]

***

## 1. 문제 정의  
기존 비전 재단 모델(CLiP, ALIGN 등)은 이미지-텍스트 크로스모달 표현에 집중되어 Scene 분류·검색 위주로만 활용된다. 반면 **실세계 컴퓨터 비전 과제**는  
  - 공간(Space): **장면(scene)** ↔ **객체(fine-grained)**  
  - 시간(Time): **정적(static)** ↔ **동적(dynamic)**  
  - 모달리티(Modality): **RGB** ↔ **다중 센스(caption, depth)**  
축을 모두 고려해야 하며, 최소한의 파인튜닝으로 제로·소수 샷 전이·완전 파인튜닝을 지원하는 **범용 비전 시스템**이 필요하다.[1]

***

## 2. 제안 방법

### 2.1 UniCL 기반 통합 대조학습  
이미지-텍스트 쌍이 웹 데이터에서 중복 캡션을 가질 때, 동일 텍스트 해시값 $$y$$를 레이블로 삼아 긍정 샘플을 확장하는 UniCL을 도입.[1]

$$ L = L_{i\to t} + L_{t\to i} $$  

$$
L_{i\to t} = -\frac{1}{|B|}\sum_{i\in B} \sum_{k\in P_i} \log \frac{\exp(u_i^\top v_k)}{\sum_{j\in B}\exp(u_i^\top v_j)}
$$  

$$
L_{t\to i} = -\frac{1}{|B|}\sum_{j\in B} \sum_{k\in Q_j} \log \frac{\exp(u_k^\top v_j)}{\sum_{i\in B}\exp(u_i^\top v_j)}
$$  

$$u=f_{img}(x)/\|f_{img}(x)\|$$, $$v=f_{txt}(t)/\|f_{txt}(t)\|$$, $$\tau$$는 온도 파라미터[1].

### 2.2 모델 구조  
- **이미지 인코더**: CoSwin(Hierarchical Vision Transformer + Conv embedding), 637M 파라미터.  
- **언어 인코더**: 12-layer Transformer(BERT 유사), 256M 파라미터.  
- **어댑터**:  
  - **Dynamic Head**: 계층·공간·채널 축별 어텐션으로 객체 표현 학습.[1]
  - **METER**: 비전-언어 VL 과제용 어댑터, co-attention 기반 융합.[1]
  - **Video CoSwin**: 3D 토크나이저·패치 머징, 시공간 윈도우 어텐션 확장.[1]

### 2.3 확장 및 효율화  
- **데이터**: 인터넷 크롤링으로 9억 이미지-텍스트 쌍(FLD-900M) 구축, 필터링·균형 샘플링.[1]
- **학습 인프라**: ZeRO, activation checkpointing, mixed-precision, gradient cache 조합으로 메모리·스루풋 최적화.[1]

***

## 3. 성능 및 한계

### 3.1 성능 향상  
- **제로샷 분류**(ImageNet-1K top-1 83.74%, top-5 97.18)  
- **Linear probe**: 11개 데이터셋 중 9개 SOTA 경신  
- **Fine-tune 분류**(ImageNet-1K top-1 90.05)  
- **Few-shot**(Cross-domain 5-shot 평균 68.5% vs 68.0%)  
- **이미지-텍스트 검색**: Flickr30K R@1 90.9, MSCOCO R@1 64.7 (제로샷)  
- **객체 검출**: COCO mAP62.4, Object365 39.3, Visual Genome AP50 16.2  
- **비디오 검색**(MSR-VTT R@1 37.6)  
- **액션 인식**(Kinetics-600 top-1 87.8)  
이 모든 실험에서 기존 최첨단 대비 유의미한 성능 향상을 보임.[1]

### 3.2 일반화 성능  
- **제로·소샷 전이** 성능 우수: 객체 검출·검색·분류에서 학습하지 않은 도메인 및 클래스에도 견고  
- **도메인간 Few-Shot**: 5-shot 상황에서도 기존 챌린지 우승자 넘어섬  
- **저자극 커스터마이징**: 어댑터 소수 레이어·에폭만 조정하는 효율적인 전이

### 3.3 한계  
- **모델 규모**: 893M 파라미터로 NLP 재단 모델 대비 중간 규모  
- **데이터 편향**: 웹 크롤링 특성상 장르·국적 편향 가능성  
- **극한 저자원 도메인**: 객체 검출 zero-shot에서 소수 데이터셋은 성능 격차 존재[1]

***

## 4. 향후 연구 방향 및 고려 사항  
Florence는 비전 재단 모델 구축의 첫걸음으로, 다음 사항을 고려해 발전시킬 필요가 있다.  
- **추가 과제 확대**: 깊이·흐름 추정, 추적(tracking), 추가 VL 과제 등 통합 지원  
- **모델 경량화**: 모바일·저전력 환경에서 활용 가능한 소형 아키텍처  
- **데이터 다양성**: 특수 도메인(의료, 위성 등) 크롤링·정제 파이프라인 개선  
- **편향·안전성 검증**: 대규모 데이터 편향 완화, 윤리적·사회적 영향 분석  
- **제로샷 성능 격차 해소**: novel 클래스·도메인에서의 객관적 성능 개선법 연구  

이로써 Florence는 인간과 유사한 시공간·모달리티 전반의 시각 이해를 목표로, **최소 커스터마이징**으로 광범위한 비전 과제를 처리할 수 있는 기반을 제시하였다.[1]

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/853eef2d-6b18-4301-8b18-4192dd389822/2111.11432v1.pdf)
