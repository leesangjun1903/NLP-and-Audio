# Behind the Scene: Revealing the Secrets of Pre-trained Vision-and-Language Models

**핵심 주장:**  
Transformer 기반 대규모 사전학습된 비전·언어(V+L) 모델(UNITER, LXMERT 등)은 다양한 V+L 벤치마크에서 최고 성능을 달성했지만, 그 내재된 **작동 원리**에 대한 이해는 부족하다. 이를 위해 VALUE(Vision-And-Language Understanding Evaluation)라는 일련의 **프로빙 과제**(Visual Coreference Resolution, Visual Relation Detection 등)를 제안하여, Attention 패턴과 임베딩을 분석함으로써 모델이 학습한 **암묵적 지식**과 **교차 모달 정렬** 메커니즘을 해석한다.[1]

**주요 기여:**  
1. **VALUE 프레임워크 제안:** V+L 모델의  
   - 모달 융합 정도 (Multimodal Fusion Degree)  
   - 모달 중요도 (Modality Importance)  
   - 교차 모달 상호작용 (Cross-modal Interaction)  
   - 이미지-이미지 관계 (Image-to-Image Interaction)  
   - 텍스트-텍스트 지식 (Text-to-Text Interaction)  
   를 평가하는 5가지 프로빙 과제를 체계화.  
2. **모델 아키텍처별 비교 분석:**  
   - 단일 스트림(UNITER) vs. 이중 스트림(LXMERT) 구조에서 **층별 융합 양상**과 **Attention 헤드 역할** 차이 규명.  
3. **Attention 해석:**  
   - 텍스트 모달리티에 편향된 최종 예측 경향 확인.  
   - 일부 Attention 헤드가 **교차 모달 정렬**에 특화됨.  
   - 이 헤드들이 이미지 영역과 텍스트 단어 간 정렬, 시각 관계, 핵심참조(link) 정보를 효과적으로 캡처.  
4. **임베딩 분석:**  
   - UNITER는 층이 깊어질수록 모달 간 융합이 심화(Fusion↑), LXMERT는 반대 양상 관찰.  
   - 시각·언어 지식 모두 BERT 기반 언어 모델 수준으로 인코딩됨.[1]

***

# 1. 해결 문제

이 논문은 Transformer 기반 V+L 사전학습 모델이  
- **왜** 다양한 다운스트림 V+L 태스크에서 높은 성능을 보이는지  
- 모델 내부의 **Attention 메커니즘**과 **임베딩**이 어떤 지식을 암묵적으로 학습하는지  
를 규명하는 것을 목표로 한다.

***

# 2. 제안 방법

VALUE 프레임워크는 다섯 가지 프로빙 과제로 구성된다.

1. **Multimodal Fusion Degree**  
   -  각 층의 임베딩을 k-means 클러스터링 후 NMI로 평가.  
   -  UNITER: 층이 깊어질수록 NMI↓(융합↑). LXMERT: 역전 추세 관찰.  
2. **Modality Importance**  
   -  [CLS] 토큰이 텍스트/이미지 모달리티에 할당하는 Attention 합산 비율 분석.  
   -  전반적 예측은 텍스트에 더 의존.  
3. **Cross-modal Interaction**  
   -  **Visual Coreference Resolution**: Flickr30k Entities의 이미지–명사구 연결을 Attention 최대치로 탐지.  
   -  개별 헤드 및 헤드 조합(선형 분류기) 모두에서 V→T, T→V 정렬 정보 캡처 성공.  
4. **Image-to-Image Interaction**  
   -  **Visual Relation Detection**: Visual Genome 관계(“on”, “holding” 등) 분류를 Attention 패턴으로 분석.  
   -  특정 헤드들이 시각적 관계를 효과적으로 인코딩.  
5. **Text-to-Text Interaction**  
   -  SentEval 9개 언어 과제(SentLen, TreeDepth, BShift, …)로 언어 지식 평가.  
   -  UNITER(Base) 초기화 시 BERT 가중치 활용으로 언어 능력 강화.[1]

수식 예시:  
-  Modality Importance head j:  

$$
I_{M,j} = \sum_{i \in S} \mathbf{1}(i\in M)\,\alpha_{ij}
$$  

-  Attention 기반 코어퍼런스 프로버:  

$$
p(c|i,j)\propto\sum_{k=1}^N (w_k \alpha^k_{ij} + \mu_k \alpha^k_{ji})
$$

***

# 3. 모델 구조 비교

| 특성             | UNITER (단일 스트림)   | LXMERT (이중 스트림)            |
|------------------|------------------------|---------------------------------|
| 아키텍처         | 12-layer Transformer   | Tₜ(9)→Tᵥ(5)→T𝚌(5)               |
| 모달 융합 위치   | 모든 레이어에서 융합   | 마지막 5개 레이어에서만 융합   |
| 초기화           | BERT 가중치 초기화      | 별도 초기화                     |
| 융합 경향        | 층↑ → 융합↑             | 층↑ → 융합↓                    |

***

# 4. 성능 향상 및 한계

-  **성능 향상:**  
  - V+L 다운스트림 태스크(Visual QA, Retrieval 등)에서 최고 성능 달성.  
  - CONTENT-LEVEL: Attention 헤드 프루닝, 구조 최적화 및 BERT 초기화 활용으로 일반화 성능 개선 기대.  
-  **한계:**  
  - 모델 내부 지식은 Attention/임베딩 레벨에서만 간접 평가.  
  - 프로빙 과제는 파인튜닝 성능 대체 지표로 완전치 않음.  
  - LXMERT는 BERT 초기화를 활용하지 않아 언어 성능이 상대적 저하.

***

# 5. 일반화 성능 향상 가능성

- **Attention 헤드 프루닝:** 중요하지 않은 모달/헤드를 제거하여 파라미터 효율화 및 과적합 감소.  
- **BERT 초기화**를 모든 V+L 모델에 일관 적용하여 언어 능력 유지.  
- **추가 감독신호**(예: 시각 관계, 참조 링크)로 사전학습 태스크 확장 시, 더 견고한 멀티모달 일반화 가능.  
- **단일 스트림 구조**가 더 강력한 융합 능력과 해석 가능성을 제공하므로, 이 구조 중심 연구 권장.

***

# 6. 향후 연구에 미치는 영향 및 고려 사항

**영향:**  
이 연구는 V+L 사전학습 모델 설계 원칙, 프로빙 과제 활용법, 해석 가능한 구조 개선 아이디어를 제시하여, 차세대 멀티모달 AI 모델의 **효율성·일반화·해석력** 강화를 촉진한다.

**고려 사항:**  
- **프로빙 태스크 감독:** 사전학습 단계에 직접 통합 시 모델의 중간 표현 품질을 제어할 수 있음.  
- **모델 크기·파라미터 최적화:** Attention 기반 중요도 분석으로 불필요한 컴포넌트 제거.  
- **다양한 데이터 도메인 확장:** 일반화 검증을 위해 의료·자율주행·리테일 등 도메인별 프로빙 과제 개발.  
- **멀티모달 피드백 루프:** Visual Coreference, Relation Detection 등 태스크를 파인튜닝 전용 Objective로 추가하여 사전학습 품질 강화.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/62bbd310-3970-42af-b69b-bbc13eee0287/2005.07310v2.pdf)
