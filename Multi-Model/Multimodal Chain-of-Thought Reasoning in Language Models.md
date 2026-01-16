# Multimodal Chain-of-Thought Reasoning in Language Models

## 핵심 주장 및 주요 기여  
본 논문은 **텍스트와 이미지의 멀티모달 정보를 활용하여 Chain-of-Thought(CoT) 추론**을 수행하는 새로운 프레임워크인 **Multimodal-CoT**를 제안한다.  
1. **두 단계(two-stage) 학습 구조**  
   - 1단계: 질문·이미지 입력 → 합리적 추론(“rationale”) 생성  
   - 2단계: 질문·이미지·생성된 rationale 입력 → 최종 답안 도출  
2. **비전(이미지) 특징 융합**으로 1B 파라미터급 모델에서도 CoT의 **환각(hallucination) 감소** 및 **수렴 속도 향상**을 달성  
3. ScienceQA, A-OKVQA 벤치마크에서 **기존 대비 SOTA 성능**을 기록[1]

## 문제 정의  
기존 CoT 연구는 텍스트만을 다루어,  
- 1B 이하 모델은 잘못된 intermediate reasoning을 생성하여 오답을 유도  
- 이미지 정보를 텍스트 캡션 방식으로만 활용 시 시멘틱 정보 손실

따라서 멀티모달 맥락에서 **합리적 추론 생성의 정확성**과 **이로 인한 최종 답변 정확도**를 동시에 높이는 방법이 필요하다.[1]

## 제안 방법  
### 1. 두 단계 프레임워크  

$$
\begin{aligned}
\text{Stage 1 (Rationale)}:\;&R = F_{\theta}(X_{\text{text}}, X_{\text{vision}})\\
\text{Stage 2 (Answer)}:\;&A = F_{\theta}(X_{\text{text}}\circ R,\, X_{\text{vision}})
\end{aligned}
$$  

- $$X_{\text{text}}$$: 질문·문맥·보기  
- $$X_{\text{vision}}$$: ViT로 추출한 이미지 패치 특징  
- $$F_{\theta}$$: T5 기반 Transformer encoder–decoder

### 2. 모델 구조  
1) **인코딩**  
   - LanguageEncoder → $$H_{\text{lang}}\in\mathbb{R}^{n\times d}$$  
   - VisionExtractor(ViT)+Proj → $$H_{\text{vis}}\in\mathbb{R}^{m\times d}$$  
2) **상호작용**  
   - 단일 헤드 어텐션: $$\mathrm{Softmax}(\tfrac{QK^\top}{\sqrt{d_k}})V$$  
   - 게이트 융합:  


$$\lambda=\sigma(W_lH_{\text{lang}}+W_vH_{\text{vis}}^{\text{attn}})$$,  
$$H_{\text{fuse}}=(1-\lambda)\odot H_{\text{lang}}+\lambda\odot H_{\text{vis}}^{\text{attn}}$$  

3) **디코딩**  
   - Transformer decoder로 rationale 및 answer 생성[1]

## 성능 향상  
- **ScienceQA**  
  - Base(223M): 74.11% → 85.31%  
  - Large(738M): 86.54% → **90.45%**  
- **A-OKVQA**: 47.86% → 50.57%  
- **효과 분석**:  
  - 환각 비율 56% → 22% 감소  
  - 수렴 속도 초기 에폭부터 크게 향상[1]

## 한계 및 일반화 가능성  
- **Commonsense 오류(80%)**: 지도(map) 이해, 객체 계수, 알파벳 순서 오류  
- **논리 오류(14%)**: 잘못된 비교·추론 모순  
- **비전 피처 의존**: ViT 외 CLIP, DETR, ResNet 대비 성능 우수  
- **라벨 없는 상황**: InstructBLIP·ChatGPT 생성 가짜 rationale로도 87.76% 달성 → human annotation 대체 가능성 확인[1]

## 향후 연구 영향 및 고려 사항  
- **멀티모달 상호작용 강화**: 지도·카운팅 전용 비전 모듈, commonsense 외부 지식 주입  
- **라벨 비용 절감**: 대형 비전언어 모델로 pseudo-rationale 생성 후 fine-tuning  
- **필터링 메커니즘**: 부정확한 CoT 무시하여 추론 견고성 강화  
- **확장성**: 다른 멀티모달·비전 질문 응답 데이터셋으로 일반화 검증  

Multimodal-CoT는 **경량 모델**에서도 **정교한 멀티모달 CoT 추론**을 가능하게 하여, 앞으로의 멀티모달 리서치에서 **효율적인 rationalization**과 **환각 제어**를 위한 기준을 제시한다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/10dfd45a-f6df-4b79-980b-6b967ee02e7a/2302.00923v5.pdf)
