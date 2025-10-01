# Multi-Modality Cross Attention Network for Image and Sentence Matching 

**핵심 주장 및 주요 기여**  
제안된 **MMCA 네트워크**는 이미지 영역과 문장 단어 간의 *교차-모달(inter-modality)* 관계뿐만 아니라 각 모달리티 내의 *내부-모달(intra-modality)* 관계를 통합하여 모델링함으로써, 기존 방법들이 놓쳤던 상호 보완적 정보를 결합한다. 주요 기여는 다음과 같다:[1]
- **통합적 관계 모델링**: 이미지–이미지, 문장–문장, 이미지–문장 간의 관계를 하나의 딥 네트워크로 동시에 학습.  
- **교차-어텐션 모듈**: Self-Attention으로 얻은 조각별 표현을 Transformer 기반의 Cross-Attention으로 재가공하여 양방향 관계를 융합(식 (8)-(14)).  
- **우수한 성능**: Flickr30K, MS-COCO 벤치마크에서 기존 최첨단 대비 Recall@1을 최대 7.1%p 향상.[1]

***

## 1. 문제 정의  
이미지–문장 매칭은 시각적 정보와 언어적 정보를 동일 임베딩 공간에 투영한 뒤 유사도를 측정하는 과제다.  
기존 “1:1 글로벌 매칭” 방법은 전체 이미지와 전체 문장 간의 단일 유사도만 학습했으며, “다:다 매칭”은 이미지 영역(region)과 문장 단어(word) 간 국부적 유사도만 고려했다.[1]
그러나 이들 방법은  
- 이미지 내부의 객체 간 관계  
- 문장 내부의 단어 간 관계  
- 이미지와 문장 간 세밀한 상호작용  
중 일부 정보만 활용하여 **전체 시각-언어 격차(visual–semantic gap)**를 완전히 해소하지 못한다.

***

## 2. 제안 방법  
### 2.1. 모델 구조  
MMCA는 두 단계의 어텐션 모듈로 구성된다:[1]
1. **Self-Attention 모듈**  
   - 이미지: Bottom-up 모델로 추출된 k개의 영역 피처 $$R=[r_1,\dots,r_k]$$를 Transformer 기반 Self-Attention으로 처리하여 $$\text{Rs}=[rs_1,\dots,rs_k]$$ 얻음.  
   - 문장: BERT 토큰 임베딩 $$X=[x_1,\dots,x_n]$$에 1D-CNN(uni/bi/tri-gram)을 적용하여 문장 조각 간 intra-modality 관계를 캡처.  

2. **Cross-Attention 모듈**  
   - 전 단계의 이미지·문장 표현을 연결한 $$Y=[R \| X]$$를 입력으로 Transformer를 통해 다음을 동시 학습[1]:  

$$
       \text{Attention}(Q,K,V) = \text{softmax} \bigl(\tfrac{QK^T}{\sqrt{d}}\bigr)V
     $$
  
   - 이로써 식 (12)-(14)에서 보듯 두 모달리티 내부와 간 상호작용을 *동시에* 모델링.  
   - 최종 출력 $$i_1, c_1$$은 1D-CNN 또는 평균 풀링으로 집계.

### 2.2. 학습 목표  
두 종류 임베딩 $$(i_0,c_0)$$, $$(i_1,c_1)$$ 사이 유사도를  

$$
  S(I,T) = \langle i_0,c_0\rangle + \alpha\langle i_1,c_1\rangle
$$

로 정의하고, hard-negative 기반 bi-directional triplet 손실(식 (15))을 최적화.[1]

***

## 3. 성능 향상 및 한계  
- **Recall@1**:  
  - Flickr30K: 이미지→문장 74.2% (+5.1%p), 문장→이미지 54.8% (+2.4%p) 개선.[1]
  - MS-COCO(1K): 74.8% (+3.6%p), 61.6% (+3.8%p) 달성.[1]
- **한계**:  
  - 학습 복잡도 증가: Transformer 계열 모듈 사용으로 연산량 및 메모리 요구량이 큼.  
  - 하이퍼파라미터 $$\alpha$$, hidden 차원 등에 민감하여 튜닝 필요.[1]
  - BERT 및 bottom-up 모델 고정(frozen) 사용으로 언어·시각 특화 추가 학습 제한.

***

## 4. 모델의 일반화 성능 향상  
Cross-Attention 모듈은 이미지 영역과 문장 단어 간 다양한 상호작용을 학습함으로써, 단일 모달리티에 최적화된 과적합을 억제하고 *fragment-level* 표현의 *robustness*를 강화한다.[1]
이를 통해 서로 다른 도메인(예: Flickr30K↔MS-COCO) 간에도 세밀한 시각-언어 정렬 능력이 향상되어 일반화 성능이 크게 개선된다.

***

## 5. 향후 연구에 미치는 영향 및 고려 사항  
- **영향**: MMCA의 joint intra/inter-modality 어텐션 개념은 이후 *Vision-Language Navigation*, *Visual Question Answering* 등 다양한 멀티모달 과제에 확장 적용되어 세밀한 시각-언어 정합의 새로운 표준이 되었다.  
- **고려점**:  
  - **경량화**: 실시간 응용을 위한 모듈 경량화 및 distillation 기법 필요.  
  - **동적 모달리티 융합**: 이미지·텍스트 외 오디오, 비디오 등 복수 모달리티로 확장 시, 각 모달 중요도에 따른 가변적 어텐션 설계 고려.  
  - **자기 지도 학습**: 대규모 unlabeled 멀티모달 데이터를 활용한 사전학습 프레임워크 통합 가능성.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/7d1fe755-640a-44a1-bbd9-70ad8e7148b7/Wei_Multi-Modality_Cross_Attention_Network_for_Image_and_Sentence_Matching_CVPR_2020_paper.pdf)
