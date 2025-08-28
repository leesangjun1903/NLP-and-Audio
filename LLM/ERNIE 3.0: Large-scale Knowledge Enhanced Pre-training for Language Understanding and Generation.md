# ERNIE 3.0: Large-scale Knowledge Enhanced Pre-training for Language Understanding and Generation

**주요 주장 및 기여:**  
ERNIE 3.0은 10억 매개변수급 대규모 언어 모델에 **지식 그래프**와 **다중 패러다임 학습**을 결합하여, 자연어 이해(NLU)와 생성(NLG) 모두에서 제로·소수학습 및 파인튜닝 성능을 크게 개선한다.  
– 지식 통합을 위한 ‘Universal Knowledge-Text Prediction’ 과제 도입  
– NLU용 양방향 인코더와 NLG용 단방향 인코더를 각기 갖춘 **공유 + 과제별** 네트워크 구조  
– 어휘·구조·지식 수준의 연속적 다중 작업(pre-training)으로 일반화 능력 극대화  

# 문제 정의와 제안 방법

## 해결하고자 하는 문제  
1. 기존 대규모 언어 모델의 **단일 자동회귀**(pre-training) 한계  
2. 순수 텍스트 학습으로 인한 **지식 결여**  
3. NLU·NLG 요구사항의 불일치로 인한 파인튜닝 효율 저하  

## 모델 구조  
– Universal Representation Module (48-layer Transformer-XL, hidden=4096, heads=64)  
- NLU Task-specific Module (12-layer bi-directional Transformer-XL)  
- NLG Task-specific Module (12-layer uni-directional Transformer-XL)  
공유 네트워크로 어휘·구문 특징을 학습하고, 과제별 네트워크로 상위 의미 표현을 분리 학습  

## 학습 과제 및 수식  
1. Knowledge Masked LM: 토큰·개체·구(phrase) 마스킹  
2. Document LM: 전통적 AR 언어 모델 $$\mathcal{L}_{LM}=-\sum_t\log P(x_t\mid x_{<t})$$  
3. Sentence Reordering & Distance: 문장 관계 분류  
4. Universal Knowledge-Text Prediction (UKTP):  
   – 주어진 지식 그래프 triple $$(h,r,t)$$와 해당 문장 $$S$$에서  
   – $$\mathcal{L}\_{UKTP}=-\sum\log P(r\mid S,h,t)-\sum\log P(w_{mask}\mid S,h,t)$$  
   – 지식·언어 간 추론 능력 강화  

## 성능 향상  
– 54개 NLU/NLG 벤치마크에서 전 모델 대비 평균 **5–10%** 절대 성능 향상  
– SuperGLUE 1위(90.6%) 달성  
– 제로샷·소수샷 학습에서도 기존 13B 모델 대비 **20%** 이상 우위  

## 한계  
– **연산·메모리 비용**: 10B 파라미터, 4TB 코퍼스, 384 GPU 필요  
– **영어 외 저자원 언어** 적용 실험 부족  
– **실시간 응용**을 위한 경량화 및 추론 최적화 필요  

# 일반화 성능 향상 메커니즘

1. **다중 패러다임 분리 학습**으로 NLU/NLG 충돌 완화  
2. **지식-텍스트 동시 최적화(UKTP)**로 관계·상식 추론력 강화  
3. **Progressive Learning**: 배치 크기·시퀀스 길이·학습률·드롭아웃 단계적 증가로 초기 수렴 가속  
이들 요소가 결합되어, 새로운 도메인·저자원 과제에서 **일반화(generalization)** 역량을 획기적으로 개선  

# 향후 연구 영향 및 고려사항

– **지식 통합 모델**: 다양한 지식원(멀티모달, 시맨틱 웹) 결합 연구 활성화  
– **경량화·효율화**: 지연 시간·에너지 소비를 줄이는 압축·추론 가속 기법 필요  
– **지속적 학습**: 새로운 지식·언어 추가 시, 전체 재학습 없이 모듈별 업데이트 방안 연구  
– **안정성·윤리**: 대규모 지식 모델의 편향·잘못된 지식 내재화 위험 관리 필수  

위 방향성을 바탕으로, ERNIE 3.0은 지식 강화 언어 모델 연구의 **새로운 표준**을 제시하며, 차세대 범용 AI 시스템의 기반이 될 전망이다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/7bbc362a-0f54-43f5-b3e4-9e1223b7f9df/2107.02137v1.pdf)
