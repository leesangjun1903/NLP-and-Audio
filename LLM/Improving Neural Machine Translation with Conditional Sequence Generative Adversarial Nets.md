# Improving Neural Machine Translation with Conditional Sequence Generative Adversarial Nets 

# 핵심 주장 및 주요 기여 요약

**핵심 주장**  
본 논문은 신경 기계 번역(NMT)에 *Conditional Sequence Generative Adversarial Nets* (BR-CSGAN)를 도입하여, 기존 최대우도추정(MLE) 기반 학습의 한계를 극복하고, 생성된 번역문이 사람 번역문과 구분되지 않도록 **판별자(discriminator)**와 **BLEU 강화 목표(static BLEU objective)**를 결합한 **생성자(generator)** 간의 적대적 학습을 제안한다.

**주요 기여**  
- NMT에 GAN 구조를 처음 적용하여, 번역 품질을 문장 단위로 자동 학습하는 조건부 시퀀스 GAN(framework) 설계  
- 동적 판별자 보상(D)과 정적 BLEU 보상(Q)을 결합한 **BLEU 강화 적대적 학습** 기법 제안  
- RNNSearch와 Transformer 양쪽에서 실험, EN-DE/WMT’14 및 ZH-EN/NIST 데이터셋에서 일관된 BLEU 향상 확인 (최대 +1.83, +1.69 BLEU)  

# 해결 문제 및 제안 방법

## 1. 해결하려는 문제  
기존 NMT는 시퀀스 내 개별 단어 확률을 최대화(MLE)하도록 최적화되나, 문장 수준의 문맥 일관성·자연스러움을 보장하지 못함. 최소 위험 학습(MRT)도 정적 BLEU만 고려하여, 실제 사람 번역과의 분포 차이를 완전 학습하지 못하는 한계 존재.

## 2. 제안하는 방법  
BR-CSGAN은 생성자 G와 판별자 D, 그리고 BLEU 목표 Q를 다음과 같이 결합하여 학습한다.

- 생성자 Gθ: 기존 NMT 모델(RNNSearch 또는 Transformer)  
- 판별자 D(X,Y): 조건부 CNN으로, 번역 (X,Y) 쌍이 “사람 생성”일 확률 예측  
- BLEU 목표 Q(Y, Y*) ∈ : 문장 단위 BLEU 점수[1]

### 전체 목표 함수  

```math
J(\theta) = \mathbb{E}_{Y_{1:T}\sim G_\theta(\cdot|X)}\Bigl[\underbrace{\lambda\,(D(X,Y_{1:T}) - b)}_{\text{동적 보상}} + \underbrace{(1-\lambda)\,Q(Y_{1:T},Y^*)}_{\text{정적 BLEU 보상}}\Bigr]
```

- b: 분산 감소용 기준값(0.5), λ ∈[1]
- Monte Carlo 샘플링(N=20)으로 중간 시퀀스까지의 보상 추정  

### 학습 절차  
1. **MLE 사전 학습**: G를 최대우도로 학습  
2. **판별자 사전 학습**: 실제 병렬문 + G 생성문으로 D 학습(정확도 ξ≈0.8)  
3. **적대적 학습**:  
   - G 업데이트: 정책 경사(policy gradient)+teacher forcing  
   - D 업데이트: G의 최신 생성문으로 재학습(가중치 클램핑)  

# 모델 구조 및 성능 향상

| 모델                         | ZH-EN (평균) | EN-DE |
|------------------------------|--------------|-------|
| RNNSearch (baseline)         | 33.94        | 21.20 |
| RNNSearch + BR-CSGAN(λ=0.7)  | **35.77**    | **22.89** |
| Transformer (baseline)       | 41.80        | 27.30 |
| Transformer + BR-CSGAN(λ=0.8)| **42.61**    | **27.92** |

- BR-CSGAN은 **RNNSearch에서 +1.83**, **Transformer에서 +0.81** BLEU 향상  
- λ=0.7~0.8일 때 최적 성능 달성  

## 한계  
- GAN 학습의 불안정성: D의 초기 정확도(ξ)와 Monte Carlo 샘플링 수 N 민감  
- 계산 비용: MC 샘플링(N≥20) 및 D/G 교대 학습 시 학습 시간이 크게 증가  

# 일반화 성능 향상 가능성

- **동적 판별자**가 문장 품질 분포를 지속 학습하여, 새로운 언어쌍·도메인에도 빠르게 적응 가능  
- **BLEU 강화**로 과도한 GAN 편향 없이 구체적 번역 품질 지표에 맞춰 모델 조정  
- 향후 다중 판별자·다중 생성자(Multi-GAN) 확장 시, 다양한 품질 평가 기준(풍부도·유창성 등) 결합하여 **보다 강건한 일반화** 기대  

# 향후 연구 영향 및 고려 사항

- **적대적 학습의 안정화 기법**: WGAN, gradient penalty 등 적용으로 D/G 균형 조절  
- **다중 평가 보상**: BLEU 외 METEOR·BERTScore·인간 평가 분포 학습 연계  
- **경량화 및 효율화**: MC 샘플링 비용 절감, 판별자 경량화로 실시간 번역 적용  
- **범용성 검증**: 저자원 언어·특수 도메인(의료·법률) 번역에서의 효과 분석  

이 논문은 NMT 품질 평가를 판별자로 학습하는 새로운 패러다임을 제시하여, 향후 기계 번역 및 시퀀스 생성 전반의 **적대적 강화학습 연구**에 중요한 토대를 제공한다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/e51920a8-acd7-4b1a-9d54-2e70a41feb62/1703.04887v4.pdf)
