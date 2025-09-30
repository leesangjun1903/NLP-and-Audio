# VL-BERT: Pre-training of Generic Visual-Linguistic Representations

**핵심 주장 및 주요 기여**  
VL-BERT는 **Transformer 기반의 단일 스트림 구조**를 도입하여 시각적 정보(이미지 RoI)와 언어 정보(단어 토큰)를 통합 처리함으로써, 다양한 시각-언어 태스크에서 범용적으로 활용 가능한 **사전학습된 표현**을 제안한다. 주요 기여는 다음과 같다.[1]
- 단일 스트림 멀티모달 Transformer 도입으로 시각·언어 정보가 자유롭게 상호작용  
- Conceptual Captions(3.3M 이미지-캡션)와 BooksCorpus·Wikipedia를 이용한 **공동 사전학습**  
- Masked Language Modeling with Visual Clues와 Masked RoI Classification with Linguistic Clues라는 **이중 마스킹 과제** 설계  
- VCR, VQA, RefCOCO 등 주요 다운스트림 태스크에서 최첨단 성능 달성  

***

## 문제 정의  
시각-언어 태스크(image captioning, VQA, VCR 등)는 기존에 시각 모델과 언어 모델을 분리하여 조합하거나, 태스크별 전용 설계만이 주를 이루어 왔다. 데이터가 부족한 경우 과적합이나 최적화 격차가 발생하며, 멀티모달 단일 사전학습 프레임워크의 부재로 일관된 성능 향상이 어려웠다.[1]

***

## 제안 방법

### 모델 구조  
- 입력 요소: 단어 토큰 $$x_1,\dots,x_N$$ 및 이미지의 RoI  
- 임베딩 합성: 토큰 임베딩 + 세그먼트 임베딩 + 위치 임베딩 + **시각 특징 임베딩**  
- 시각 특징 임베딩:  
  1) Appearance Feature: Faster R-CNN의 RoI 특성 벡터(2048-d)  
  2) Geometry Embedding: $$\bigl[\tfrac{x_{LT}}{W},\tfrac{y_{LT}}{H},\tfrac{x_{RB}}{W},\tfrac{y_{RB}}{H}\bigr]$$를 다양한 주파수의 사인/코사인으로 변환(2048-d)  
  3) 두 특성의 FC 결합 후 2048-d로 투영  
- Transformer 층: BERT와 동일한 다중 헤드 어텐션, 피드포워드  
- 세그먼트: A(문장1)·B(문장2)·C(이미지) 구분[1]

### 사전학습 과제  
1. **Masked Language Modeling with Visual Clues**  
   - 입력 문장 토큰 15%를 [MASK]로 대체  
   - 주변 단어 + 시각 정보로 마스크된 단어 예측  
2. **Masked RoI Classification with Linguistic Clues**  
   - 이미지 RoI 15% 픽셀 영역 0 처리 후, 해당 RoI 카테고리 분류  
   - 기반 Faster R-CNN 예측 레이블 활용  
3. **텍스트 전용 MLM**  
   - BooksCorpus·Wikipedia에 대해 표준 MLM 수행  
   
사전학습 손실:  

$$
\mathcal{L} = \mathcal{L}_{\text{MLM-V}} + \mathcal{L}_{\text{RoI-V}} + \mathcal{L}_{\text{MLM-T}}
$$  

여기서 $$\mathcal{L}\_{\text{MLM-V}}$$ 는 시각 단서를 포함한 단어 예측 교차엔트로피, $$\mathcal{L}\_{\text{RoI-V}}$$ 는 언어 단서를 포함한 RoI 분류 교차엔트로피, $$\mathcal{L}_{\text{MLM-T}}$$는 텍스트 전용 MLM 손실이다.[1]

***

## 성능 향상 및 한계

### 주요 다운스트림 결과  
- **VCR** (Q→A→R): VL-BERT<sub>Large</sub> 78.4% vs. 이전 최고 75.7%  
- **VQA v2.0**: VL-BERT<sub>Large</sub> 72.2% vs. LXMERT 72.54%  
- **RefCOCO** (ground-truth): VL-BERT<sub>Large</sub> 83.62% vs. MAttNet 75.13%

### 한계  
- 사전학습 데이터는 주로 **이미지 캡션**과 **텍스트 코퍼스**에 치중되어 있어, 태스크 특화된 비주얼 질문 데이터와 완전 일치하지 않을 수 있음  
- RoI 개수(최대100개)에 따른 계산 비용 및 메모리 부담  
- 외형적으로 단일 스트림이지만, 복합 질문(장문) 시 어텐션 복잡도가 증가  

***

## 일반화 성능 향상 가능성  
- **텍스트 전용 MLM** 추가로 문장 길이・복잡도에 대한 일반화 강화  
- **RoI 분류 마스킹**으로 시각-언어 간 세밀한 정렬 학습  
- 다중 태스크 사전학습이 모델이 시각 및 언어 간 상호작용 패턴을 폭넓게 습득하도록 유도하여, 데이터가 적은 다운스트림 태스크에서도 **과적합 억제** 및 **전이 학습** 성능 개선 효과 발휘.[1]

***

## 향후 영향 및 고려 사항  
앞으로 시각-언어 모델 연구에서는  
- **더 다양한 멀티모달 태스크**(비디오, 3D, 음성 등)로 사전학습 범위 확장  
- **효율적 어텐션 메커니즘**(선택적 어텐션·저전력 트랜스포머) 적용  
- **대규모 웹 크롤링 멀티데이터** 기반 멀티태스크 학습이 필요  
- 모델 일반화와 **견고성(robustness)** 확보를 위해 소량의 **노이즈 캡션**과 **도메인 편차 이미지** 포함 검토  

이와 더불어, **계산 자원**과 **메모리 제약**을 고려한 경량화 연구가 필수적이며, 태스크 특성에 따른 **적응적 마스킹 전략** 개발이 유망하다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/8b30c867-af3e-4f75-8f1d-a34e719a1910/1908.08530v4.pdf)
