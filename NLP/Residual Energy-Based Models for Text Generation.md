# Residual Energy-Based Models for Text Generation

## 주요 주장 및 기여
“Residual Energy-Based Models for Text Generation” 논문은 기존 *locally normalized* 언어모델의 한계를 극복하기 위해 **사전 학습된 언어모델의 잔여(residual)를 에너지 기반 모델(EBM)으로 확장**했다.  
1. **잔여 EBM 잔여 모델**: $$P_\theta(x)\propto P_{\rm LM}(x)\exp(-E_\theta(x))$$ 형태로, 고정된 언어모델 $$P_{\rm LM}$$ 위에 시퀀스 수준의 에너지 함수 $$E_\theta$$를 학습.[1]
2. **효율적 학습**: *Conditional Noise Contrastive Estimation*을 활용해, 실제 데이터(positive)와 언어모델 생성 샘플(negative)을 구분하는 이진 분류 문제로 환원.[1]
3. **평가 및 생성**: 중요도 샘플링(importance sampling)을 통해 문장 전체의 정규화 상수(partition function)를 추정하여 표준적인 perplexity 계산이 가능하며, 샘플링 알고리즘(알고리즘 1)을 제안했다.  
4. **Bi-directional EBM**: BERT/RoBERTa와 같은 양방향 트랜스포머를 에너지 네트워크로 활용, auto-regressive 모델보다 낮은 perplexity와 우수한 생성 품질을 입증했다.[1]

## 문제 정의 및 제안 방법
### 해결하고자 하는 문제
- **Exposure bias**: 학습 시에는 실제 토큰을, 생성 시에는 모델 예측을 컨텍스트로 사용하여 분포 불일치 발생.  
- **국소적 생략**: 토큰 단위의 greedy 예측으로 긴 문맥 일관성 부족.  
- **정규화 곤란**: 순차적 MCMC/Gibbs 샘플링 비실용적.

### 제안 모델 수식
잔여 EBM의 조건부 생성 확률:  

$$
P_\theta(x_{p+1:T}\mid x_{1:p})
= \frac{P_{\rm LM}(x_{p+1:T}\mid x_{1:p})\exp\bigl(-E_\theta(x_{1:T})\bigr)}
{Z_\theta(x_{1:p})}
$$

$$
Z_\theta(x_{1:p})=\sum_{x_{p+1:T}}P_{\rm LM}(x_{p+1:T}\mid x_{1:p})\exp\bigl(-E_\theta(x_{1:T})\bigr).
$$  

학습 목표(조건부 NCE):

$$
\max_\theta\; 
\mathbb{E}_{x^+\sim p_{\rm data}}\log\frac{1}{1+e^{E_\theta(x^+)}}
+
\mathbb{E}_{x^-\sim P_{\rm LM}}\log\frac{1}{1+e^{-E_\theta(x^-)}}
$$

이진 분류 형태로 실제 데이터와 LM 샘플을 구분하여 $$E_\theta$$ 학습.

### 모델 구조
- **UNIT**: 단방향 트랜스포머 기반 에너지 네트워크, 사전학습된 LM 초기화.  
- **BIT-BASE / BIT-LARGE / BIT-MED**: BERT/RoBERTa 기반 양방향 에너지 네트워크.  
  - *BIT-BASE* (125M 파라미터), *BIT-LARGE* (355M), *BIT-MED* (UNIT과 동일 규모).  
  - 은닉층 출력의 평균 풀링 후 스칼라 에너지값으로 투영.

## 성능 향상
### 자동 평가(Perplexity)
- CC-News, Toronto Book Corpus에서 모든 비잔여 및 잔여 LM 대비 최저 perplexity 달성.  
- 예: CC-News 테스트셋에서 BIT-LARGE* 모델이 13.97–14.00으로, BASE LM(17.57), BALM(15.74) 대비 큰 폭 절감.  
- 생성 길이가 길어질수록(per-step horizon 증가) 잔여 EBM 이득 확대(Fig.1).

### 휴먼 평가
- 333문장 비교 실험에서, BIT-BASE vs BASE LM: 56.25% 선호( $$p=0.015$$ ).  
- BIT-LARGE* vs BASE LM: 58.93% 선호( $$p=0.00084$$ ), BALM 대비 59.89%( $$p=0.00020$$ ).  

## 일반화 성능 개선
- 양방향 컨텍스트 활용으로 중간 토큰 정보 완전 반영, 높은 일반화 매개변수 효율.  
- CC-News 작은 책 코퍼스에 대해도 동일한 구조로 효과 확인.  
- 반복 방지, n-gram 다양성 증가(사람과 더 유사) 확인(Fig.2).  
- log-likelihood 분포 매칭 개선: BASE LM 차이는 21.64, BIT-BASE는 6.20으로 큰 폭 감소(Fig.3).

## 한계
- **기저 LM 의존성**: 기저 LM 품질에 크게 의존. LM 샘플이 진짜 분포를 충분히 커버하지 못하면 학습·생성 한계.  
- **부정 샘플 품질**: 부정 샘플로 LM 생성만 사용하여, LM이 약하면 에너지 함수가 단순히 부정샘플 특징만 학습할 수 있음.

## 향후 연구 영향 및 고려사항
- **부정 샘플 다양화**: LM 외 다른 생성 방식 도입으로 더 유용한 neg 샘플 탐색.  
- **긴 텍스트 생성**: 토큰 대신 문단(chunk) 단위 auto-regressive 활용 연구.  
- **다른 손실 함수**: GAN, 다른 NCE 변형, 추론 효율성 개선 방안 모색.  
- **모델 일반화**: 도메인 적응, 적은 데이터 시나리오에서 EBM 일반화 성능 추가 분석.  

---  
Residual EBM은 **시퀀스 수준 평가**와 **bi-directional 사전학습 표현**을 조합해 텍스트 생성의 일관성과 품질을 크게 높이는 혁신적 접근이며, 향후 다양한 생성 모델의 보완 및 새로운 negative 샘플링 기법 개발에 기여할 전망이다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/287fe305-ee0a-4696-bb76-706eb9f8497e/2004.11714v1.pdf)
