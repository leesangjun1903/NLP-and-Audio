# Unsupervised Machine Translation Using Monolingual Corpora Only

## 핵심 주장 및 주요 기여
이 논문은 **병렬 코퍼스 없이** 순수하게 각각의 언어에 대한 단일 언어 말뭉치(monolingual corpora)만으로 신경망 기계번역(NMT) 모델을 학습할 수 있음을 보인다.  
주요 기여는 다음과 같다 :[1]
- 서로 다른 언어 문장을 **공통 잠재 공간(latent space)**에 매핑하고, 양쪽 언어에 대한 **자동 인코딩(auto-encoding)** 및 **역번역(back-translation)** 기반 재구성 손실을 결합한 새로운 학습 프레임워크 제안.  
- 잠재 분포를 정렬하기 위한 **적대적(adversarial) 정규화** 기법 도입.  
- 병렬 데이터 없이도 WMT와 Multi30k 데이터셋에서 BLEU 15.1–32.8을 달성하며, 이는 약 10만 쌍의 병렬 문장으로 학습된 감독 학습 모델과 유사한 성능을 보임.  

***

## 문제 정의 및 제안 방법

### 해결하고자 하는 문제
- **병렬 데이터 부재**: 기존 NMT는 수백만 문장 규모의 병렬 코퍼스를 필요로 하나, 저자원 언어쌍에는 이 같은 대규모 병렬 자원이 부재.  
- 이에 **전혀 레이블 없는 상태**에서 번역 모델을 학습하는 문제를 다룸.[1]

### 모델 구조
- **공유 인코더·디코더**: 양쪽 언어 모두에 단일 인코더와 디코더 사용. 언어 식별자에 따라 어휘 임베딩만 교체.  
- 인코더: 3-layer bidirectional LSTM (히든 차원 300)  
- 디코더: 3-layer LSTM + 어텐션 (히든 차원 300)  

### 학습 목표 함수

$$
L(\theta) = \lambda_{\mathrm{auto}}\bigl(L_{\mathrm{auto}}^{\mathrm{src}}+L_{\mathrm{auto}}^{\mathrm{tgt}}\bigr)
+\lambda_{\mathrm{cd}}\bigl(L_{\mathrm{cd}}^{\mathrm{src}\to\mathrm{tgt}}+L_{\mathrm{cd}}^{\mathrm{tgt}\to\mathrm{src}}\bigr)
+\lambda_{\mathrm{adv}}\,L_{\mathrm{adv}}
$$  

- **자동 인코딩 손실**  

$$L_{\mathrm{auto}}(\ell)=\mathbb{E}\_{x\sim D_\ell,\tilde x\sim C(x)}\bigl[\Delta(d(e(\tilde x,\ell),\ell),x)\bigr]$$  

- **교차 도메인 손실(역번역)**  

$$L_{\mathrm{cd}}(\ell_1\!\to\!\ell_2)=\mathbb{E}\_{x\sim D_{\ell_1}}\bigl[\Delta(d(e(C(M(x)),\ell_2),\ell_1),x)\bigr]$$  

- **적대적 손실**: 잠재 표현이 언어 구분 불가능하도록 판별기(discriminator)를 속이는 항  

$$L_{\mathrm{adv}}=-\mathbb{E}\bigl[\log p_D(\text{opposite language}\mid e(x,\ell))\bigr]$$  

- 하이퍼파라미터 $$\lambda_{\mathrm{auto}}=\lambda_{\mathrm{cd}}=\lambda_{\mathrm{adv}}=1$$로 설정.[1]

### 학습 절차
1. monolingual 데이터로 fastText 임베딩 학습 후, **비지도(bilingual) 사전** 유도  
2. 초기 모델 $$M^{(1)}$$: 단어 대 단어(word-by-word) 번역  
3. 반복적으로  
   - $$M^{(t)}$$로 역번역 데이터 생성  
   - 전체 목표 함수 최적화하여 $$(\theta_{\mathrm{enc}},\theta_{\mathrm{dec}})$$ 업데이트  
   - $$M^{(t+1)}\gets d\circ e$$  
4. 모델 선택: **무감독 모델 선택 지표**(역번역 후 BLEU)로 결정[1]

***

## 성능 향상 및 한계

| 데이터셋 | 언어쌍    | Supervisied (풀병렬) | Unsupervised (3회 반복) |
|---------|---------|---------------------|-------------------------|
| Multi30k| en–fr   | 56.8 BLEU           | 32.8 BLEU               |
| WMT’14  | en–fr   | 28.0 BLEU           | 15.1 BLEU               |

- **소량 병렬 학습 대비 우수**: 약 100,000 문장 병렬 데이터로 학습한 감독 모델과 유사 성능.[1]
- **초기 단어별 번역보다 대폭 향상**: 1회 반복 후 BLEU en–fr 12.1→27.5, 3회 반복 후 32.8까지 상승.[1]
- **제한점**  
  - 문체·도메인 일반화에는 여전히 한계: 다양한 도메인 분포에 대한 실험 부족  
  - 드문 단어 및 장문 번역 품질 저하  
  - 군집 방식의 잠재 공간이 구조화되지 않은 경우 수렴 불안정  

***

## 일반화 성능 향상 관점
- **적대적 정규화**로 잠재 분포를 정렬하여 **언어간 도메인 불변성** 확보 → 도메인 이동성 및 새로운 도메인 적응에 기여 가능.[1]
- **자동 인코딩 및 역번역 복합 학습**이 잡음을 제거하고 구조적 패턴 학습 → 언어 외 새로운 특성(예: 스타일, 주제)에도 유사한 방식 적용 전망.  
- 반복적 역번역 과정은 **자기지도 학습(self-supervision)** 형태로, 낮은 자원 상황에서 **풍부한 일반화 표현**을 획득.

***

## 향후 연구 영향 및 고려할 점
- **반감독(semisupervised) 학습**: 소량 병렬 데이터와 대규모 monolingual 코퍼스를 결합한 효율적 학습 방법 개발  
- **도메인 적응**: 적대적 손실 및 역번역을 활용한 새로운 도메인·스타일·방언에 대한 무감독 적응  
- **잠재 공간 구조화**: 그래프 정규화, 클러스터링 등의 기법으로 잠재 분포에 의미 있는 구조 부여  
- **확장성**: 더 많은 언어쌍(특히 비유럽계 저자원 언어) 및 더 긴 문장에 대한 적용성 검증 필요  
- **안정성·수렴성 분석**: 손실 비율 조정 및 초기화 기법이 전체 학습 안정성에 미치는 영향 체계적 연구.  

이 논문은 **전혀 병렬 자원이 없는 환경**에서도 경쟁력 있는 기계번역이 가능함을 보이며, 향후 **다양한 자기지도 및 적대적 학습** 연구의 토대를 마련한다.[1]

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/f7f82948-9224-4bbd-807b-0d98d9e0b326/1711.00043v2.pdf)
