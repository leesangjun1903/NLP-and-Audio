# Parameter Efficient Multimodal Transformers for Video Representation Learning

# 핵심 요약 및 주요 기여

**Parameter Efficient Multimodal Transformers for Video Representation Learning**는 기존의 멀티모달 Transformer가 갖는 과도한 메모리 요구를 해결하고, 오디오-비주얼 동영상 표현 학습을 엔드투엔드(end-to-end)로 가능하게 한 최초의 연구이다.  
주요 기여는 다음 세 가지이다:[1]

1. **파라미터 절감 기법**  
   - Transformer의 가중치를 모달리티 간·계층 간에 공유하고, 저랭크 근사를 통해 분해함으로써 최대 97% 파라미터 절감 달성.  
2. **콘텐츠 인지적 네거티브 샘플링(CANS)**  
   - CNN 임베딩 공간에서 샘플 간 유클리드 거리를 측정하여, 양호한 ‘하드 네거티브’를 확률적으로 뽑아내는 대조 학습 전략 제안.  
3. **모달리티 융합 전략 비교**  
   - 조기(A), 중간(M), 후기(L) 융합 방식의 성능을 체계적으로 비교·검증하여, *중간 융합(mid-level fusion)* 방식이 최적임을 입증.  

# 문제 정의 및 접근 방법

### 해결하고자 하는 문제  
- **장기간 동영상(long video)** 처리 시 Transformer의 메모리 요구가 기하급수적으로 커져, 모델을 엔드투엔드로 학습하기 어려움.[1]
- 기존 연구는 언어 사전학습된 BERT를 고정하고 시각 모듈만 학습하여 교차 모달 학습(cross-modal learning)의 잠재력을 제한.

### 제안하는 방법

1. **아키텍처 구성**  
   - 로컬 특징 추출: 1초 분할 영상 및 오디오 클립에 각각 SlowFast-ResNet50, ResNet50 CNN 적용  
   - 단일 모달 Transformer (L=6, 헤드 수 A=12, 차원 D=768)로 장기간 시퀀스 문맥 학습  
   - 멀티모달 Transformer에서 시각·음성 임베딩을 시간 순으로 직렬화하여 비동기적 상호작용 학습  

2. **저랭크 근사 기반 파라미터 분해**  
   - 각 가중치 $$W\in\mathbb{R}^{M\times N}$$를 공유 행렬 $$U\in\mathbb{R}^{M\times O}$$와 전용 행렬 $$V\in\mathbb{R}^{O\times N}$$로 분해  
   - $$O\ll \min(M,N)$$인 128로 설정해 메모리 절감  
   - $$V x$$를 단위 구면에 투영 후, 직교 제약을 가한 회전 변환을 통해 수치 안정성 확보.[1]

3. **대조 학습 목표**  
   - **Masked Embedding Prediction (MEP)**: InfoNCE 손실  

$$
       \mathcal{L}_{\mathrm{NCE}} = -\mathbb{E}_{t}\log\frac{\exp(\mathrm{FFN}(o_t)^\top W_I x_t)}{\sum_{j\in\mathrm{neg}_t}\exp(\mathrm{FFN}(o_t)^\top W_I x_j)}
     $$  
   
   - **Correct Pair Prediction (CPP)**: 오디오-비주얼 임베딩 쌍의 정합 여부 예측 손실  
   - 최종: $$\mathcal{L}=\mathcal{L}\_{\mathrm{MEP}}+\mathcal{L}_{\mathrm{CPP}}$$.[1]

# 성능 향상 및 한계

### 성능 향상  
- **중간 융합(Mid)**: Kinetics-Sounds top-1 67.5%로 기존 대비 +2.6% 향상  
- **CANS-Similar**: 네거티브 샘플링에서 top-1 67.5%, top-5 92.3% 달성  
- **파라미터 절감**: 128M → 4M (97% 절감)에도 성능 저하 최소화  
- **다운스트림**  
  - UCF-101 시각: 89.9%  
  - ESC-50 음성: 92.3%  
  - Charades mAP: 29.5%[1]

### 한계  
- CNN 임베딩에만 간접적 자가 감독이 이뤄져 시각 표현 학습이 다소 취약  
- 짧은 비디오 사전학습(10초)와의 직접 비교 어려움  
- 추가 모달(예: 텍스트) 확장성 미검증

# 일반화 성능 향상 관점

**파라미터 공유**와 **CANS**는 모델의 일반화 능력을 극대화한다.  
- 저랭크 공유 기법은 과도한 파라미터를 줄여 과적합 위험 감소  
- CANS-Similar는 ‘어려운 네거티브’를 확률적으로 선택함으로써 더 견고한 표현 학습 유도  
- 중간 융합은 모달 누락(missing modality) 상황에서도 견고성 유지.[1]

# 연구적·실용적 영향 및 향후 고려사항

본 연구는 **엔드투엔드 멀티모달 Transformer 학습**의 문을 열었으며, 향후 다음 점을 고려할 필요가 있다:

- **CNN 직접 감독**: CNN에 대한 대조 학습을 추가해 시각 표현 강화  
- **다중 모달 확장**: 예를 들어 텍스트·센서 데이터 통합으로 적용 범위 확대  
- **효율적 학습 스케줄**: 배치 크기·학습률 스케줄링 최적화로 더 적은 자원으로 재현성 확보  
- **온라인 하드 네거티브**: CANS 개선을 통한 비디오 이해 심화  

이러한 방향은 멀티모달 비디오 이해뿐 아니라, 자율주행·의료·로보틱스 등 다양한 응용 분야에서 더욱 강력한 표현 학습을 가능하게 할 것으로 기대된다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/eeb1815b-4796-4bb9-b02a-30aa689f8fc5/2012.04124v2.pdf)
