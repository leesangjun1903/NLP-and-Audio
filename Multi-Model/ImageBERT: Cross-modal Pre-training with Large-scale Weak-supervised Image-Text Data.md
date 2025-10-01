# ImageBERT: Cross-modal Pre-training with Large-scale Weak-supervised Image-Text Data

**주요 주장**  
ImageBERT는 대규모 약지도(image-text) 데이터를 활용한 크로스모달 사전학습을 통해 이미지와 텍스트 간의 조인트 표현을 학습하고, 이를 다양한 비전–언어 작업에 적용했을 때 기존 방법을 능가하는 성능을 입증한다.[1]

**주요 기여**  
- 약지도 방식으로 웹에서 수집한 10M 쌍 규모의 LAIT 데이터셋 제안 및 다단계 사전학습(Multi-stage Pre-training) 전략 도입  
- Masked Language Modeling(MLM), Masked Object Classification(MOC), Masked Region Feature Regression(MRFR), Image-Text Matching(ITM) 4가지 사전학습 과제 설계  
- Transformer 기반 단일 모델에 이미지와 텍스트 토큰을 모두 입력하는 구조로, 이미지 영역 특징(RoI) 및 위치 임베딩을 통합  
- MSCOCO 및 Flickr30k의 텍스트-이미지 검색에서 제로샷 및 파인튜닝 모두에서 새로운 SOTA 달성[1]

***

## 1. 해결하고자 하는 문제  
기존 비전–언어 모델은 언어와 비전 모델을 별도로 사전학습한 뒤 Late Fusion하는 방식을 주로 사용해, 대규모 크로스모달 데이터 부족 문제로 일반화 성능과 데이터 효율성이 떨어진다. ImageBERT는 대규모 약지도 웹 크롤링 데이터(LAIT)와 다단계 사전학습을 통해 이러한 한계를 극복하려 한다.[1]

***

## 2. 제안하는 방법 및 모델 구조

### 2.1 모델 구조  
ImageBERT는 BERT 기반 Transformer 아키텍처를 확장해 다음을 입력으로 받는다:[1]
- 텍스트 토큰: WordPiece 토크나이저로 얻은 $$w_0,\dots,w_{n-1}$$  
- 이미지 토큰: Faster R-CNN으로 추출한 $$o$$개의 RoI 특징 $$r_0,\dots,r_{o-1}$$  

각 토큰은 임베딩 레이어를 통해 공통 차원으로 투영 후, 양방향 Self-attention Transformer에 입력된다(Figure 4).[1]
- **언어 임베딩**: 단어 임베딩 + 세그먼트 임베딩 + 위치 임베딩  
- **이미지 임베딩**: RoI 특징 + 객체 위치 임베딩($$c_i = [\frac{x_{tl}}{W}, \frac{y_{tl}}{H}, \dots]$$) + 세그먼트 임베딩 + 순서 임베딩  

### 2.2 다단계 사전학습 (Multi-stage Pre-training)  
1단계: 웹 크롤링으로 수집한 LAIT(10M 쌍 중 2M 샘플)  
2단계: Conceptual Captions(3M), SBU Captions(1M)  
각 단계에서 동일한 4가지 사전학습 과제(Mask 및 Matching) 수행, 단계별 데이터 특성 차이를 활용하여 모델을 점진적으로 수렴시킴.[1]

### 2.3 사전학습 과제  
- **MLM**: 토큰 확률 $$\displaystyle L_{MLM} = -\sum_{(v,w)\in D}\log P(w_m\mid w_{\setminus m},v)$$  
- **MOC**: 객체 토큰 분류 $$\displaystyle L_{MOC} = \sum CE(l_{v_m},f_{v_m})$$  
- **MRFR**: RoI 특징 회귀 $$\displaystyle L_{MRFR} = \sum \|h_{v_m}-r_{v_m}\|^2$$  
- **ITM**: 이미지-텍스트 매칭 $$\displaystyle L_{ITM} = -\sum [y\log s+(1-y)\log(1-s)]$$  

모든 Mask 과제는 긍정 샘플일 때만 손실 계산(Conditional Mask).[1]

***

## 3. 성능 향상 및 한계

### 3.1 성능 향상  
- **제로샷**: MSCOCO R@1 53.6%, Flickr30k R@1 54.3%로 종전 모델 초과[1]
- **파인튜닝**: MSCOCO R@1 73.6%, Flickr30k R@1 73.1%로 새로운 SOTA 달성[1]

다단계 사전학습과 LAIT의 대규모 약지도 데이터가 성능에 기여함이 Ablation 실험으로 확인됨.[1]

### 3.2 한계  
- 웹 수집 데이터의 노이즈 여전: 약지도 방식으로 품질 보장 어려움  
- 모델 크기(BERT-base) 및 RoI 개수(100)에 따른 계산 비용 부담  
- 조명된 파인튜닝 작업에 한정된 검증: VQA, VCR 등 다른 크로스모달 작업으로 일반화 필요.[1]

***

## 4. 일반화 성능 향상 관점

다단계 사전학습 전략이 핵심이다.  
- 대규모 소위 *out-of-domain* LAIT로 일반적 시각-언어 패턴 학습  
- 이후 *in-domain* 공인 캡션 데이터로 세부 조정  
이 접근이 제로샷 및 파인튜닝에서 모두 강력한 일반화 성능을 보였으며, 특히 파인튜닝 데이터가 부족한 상황에서도 고성능을 유지함을 확인했다.[1]

***

## 5. 향후 연구의 영향 및 고려 사항

- **크로스모달 사전학습 표준화**: ImageBERT의 다단계 프레임워크는 향후 다양한 아키텍처와 데이터 유형에 확장 가능  
- **데이터 품질 관리**: 웹 기반 약지도 데이터 필터링 기법 개선, 노이즈 감소 연구 필요  
- **다양한 크로스모달 작업 적용**: VQA, VCR, 이미지 캡셔닝 등으로 영역 확장  
- **효율성 최적화**: 경량 모델, 선택적 RoI 사용, 지식 증류 등을 통한 실시간 응용 연구  

이들 고려 사항은 비전–언어 모델의 범용성과 실용성을 한층 더 끌어올릴 것이다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/afb30fb5-f5d4-475f-8228-5bde37f28ab7/2001.07966v2.pdf)
