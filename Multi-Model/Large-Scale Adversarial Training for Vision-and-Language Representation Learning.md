# Large-Scale Adversarial Training for Vision-and-Language Representation Learning

# 핵심 요약 및 기여

**핵심 주장:** 본 논문은 비전과 언어를 융합한 대규모 표현 학습(VL) 모델에 *적대적 학습(adversarial training)*을 도입하여 일반화 성능을 획기적으로 개선할 수 있음을 보인다.  
**주요 기여:**  
- VILLA(Vision-and-Language Large-scale Adversarial training) 프레임워크 제안:  
  1) *태스크 무관 사전학습(Adversarial Pre-training, APT)*  
  2) *태스크 특화 미세조정(Adversarial Fine-tuning, AFT)*  
- 이미지·텍스트 임베딩 공간에 소량의 적대적 섭동(perturbation)을 추가하고, *Free adversarial training*과 *KL-divergence 정규화*를 결합하여 효율적이고 부드러운 학습 목표를 수립  
- UNITER 및 LXMERT 등 최첨단 VL 모델에 VILLA를 적용해 다양한 다운스트림 과제(VQA, VCR, NLVR2, RE, ITR, VE)에서 SOTA 성능 달성

***

# 1. 해결하고자 하는 문제

기존 대규모 VL 모델은 막대한 파라미터 수에도 불구하고 한정된 레이블 데이터로 미세조정 시 *과적합(overfitting)*이 발생하기 쉽다.  
- 공격적(finetune-heavy) 학습은 모델이 특정 다운스트림 태스크에 지나치게 치중되어 일반화 성능이 하락  
- 순수한 데이터 증강이나 정규화만으로는 충분한 내구성 확보 어려움[1]

***

# 2. 제안 방법

## 2.1 전체 프레임워크 개요  
VILLA는 두 단계로 구성된다:[1]
1. **적대적 사전학습 (APT)**  
2. **적대적 미세조정 (AFT)**  

두 단계 모두 동일한 학습 목표로 통합 가능하며, 사전학습에서 얻은 일반화 능력이 미세조정에도 전이된다고 가정한다.

## 2.2 임베딩 공간에서의 섭동  
- 텍스트: 단어 임베딩 $$ \mathbf{e}_\text{txt} $$에 소규모 노이즈 $$ \delta_\text{txt} $$ 추가  
- 이미지: 객체 검출기 기반 지역 특징 $$ \mathbf{e}_\text{img} $$에 노이즈 $$ \delta_\text{img} $$ 추가  
- 임베딩 외 위치·분류 토큰 등은 고정

## 2.3 학습 목표  
학습 손실은 세 부분으로 구성된다:  

```math
\min_\theta \mathbb{E}_{(x,y)\sim D}\Big[ 
L_\text{std} 
+ \alpha \underbrace{\max_{\|\delta\|\le\epsilon}L_\text{CE}(f(x+\delta),y)}_{R_\text{at}}
+ \beta \underbrace{\max_{\|\delta\|\le\epsilon} \mathrm{KL}\big(f(x+\delta)\,\|\,f(x)\big)}_{R_\text{kl}}
\Big]
```

- $$L_\text{std}$$: 일반 교차엔트로피 손실  
- $$R_\text{at}$$: *라벨 보존(label-preserving)* 적대적 손실[1]
- $$R_\text{kl}$$: 예측 확률 벡터 간 유사성 정규화  

*Free adversarial training* 전략을 도입하여 추가 연산 없이 다중 PGD 스텝을 수행하고, KL 정규화로 예측 분포 *부드러움(smoothness)* 확보.[1]

***

# 3. 모델 구조 및 구현

- **베이스 백본:**  
  - *UNITER* (Single-stream Transformer)  
  - *LXMERT* (Two-stream + Cross-attention Transformer)  
- **입력:** 이미지 지역 특징 + 토큰화된 텍스트  
- **융합:** CLS 토큰을 통한 멀티-모달 컨텍스트 학습  
- **적대적 학습 적용 위치:** Transformer 전 임베딩 레이어  
- **하이퍼파라미터:**  
  - 섭동 크기 $$\epsilon$$, 적대적 학습률 $$\alpha,\beta$$, PGD 스텝 수 등 논문 및 부록에 상세[1]

***

# 4. 성능 향상 및 일반화

## 4.1 다운스트림 성능 향상  
- **UNITER-large → VILLA-large:**  
  - VQA: 74.02 → 74.87 (+0.85)  
  - VCR (QAR): 62.8 → 65.7 (+2.9)  
  - NLVR2: 80.49 → 82.57 (+2.08)  
  - 평균 across 6 tasks: +1.15~+2.21[1]

## 4.2 사전학습 vs. 미세조정 기여  
- *APT*만 적용 시 평균 +0.51점, *AFT*만 적용 시 +0.82점 상승  
- 두 단계 결합 시 +1.15점 상승[1]

## 4.3 일반화 및 과적합 방지  
- 학습 곡선에서 VILLA는 초반 과적합을 지연시키고(val loss 낮추며) 최종 검증 정확도 향상[1]
- *VQA-Rephrasings*에서 질문 문장 재구성에도 성능 유지력(robustness) 개선 확인[1]
- Probing 분석: 텍스트-이미지 정합(attention) 지표 전 범주에서 UNITER 대비 주의 가중치 상승(0.223 vs. 0.195 평균)[1]

***

# 5. 제한점 및 한계

- **연산 비용:** 다중 PGD 스텝으로 인한 학습 시간 및 자원 증가  
- **공격 면역성:** 본 연구는 주로 깨끗한(clean) 입력에서의 일반화 향상에 초점, 적대적 공격 방어 효과는 추후 연구 필요  
- **세밀한 최적화:** 적대적 하이퍼파라미터(learning rate, norm bound) 탐색이 성능에 민감

***

# 6. 향후 연구 방향 및 고려 사항

- **학습 효율화:** FreeLB, FastAT 등 *고속 적대적 학습* 알고리즘 추가 도입으로 대규모 사전학습 현실화  
- **공격방어 연구:** 멀티모달 분야에 특화된 *공격 기법* 및 *방어 전략* 개발  
- **환경·비용 절감:** 적대적 학습 에너지 비용 최소화를 위한 *지속가능 학습(sustainable training)*  
- **일반화 이론:** 적대적 정규화가 멀티모달 일반화에 미치는 이론적 해석

***

VILLA는 “멀티모달 표현 학습에도 적대적 학습이 유효하다”는 중요한 시사점을 제공하며, 향후 비전-언어 모델의 *견고함*과 *일반화*를 함께 달성하기 위한 연구 기반을 마련한다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/449f2555-2c3b-408b-b0d0-3eb3241759f4/2006.06195v2.pdf)
