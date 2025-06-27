# Parameter-Efficient Fine-Tuning for Large Models

## 핵심 주장과 주요 기여

### 논문의 핵심 주장
이 논문은 대규모 언어 모델의 **전체 파라미터 파인튜닝의 계산 비용 문제**를 해결하기 위한 **Parameter-Efficient Fine-Tuning (PEFT)**의 포괄적 분석을 제시합니다. GPT-3의 175억 개 파라미터처럼 대규모 모델의 전체 파인튜닝이 수천 개의 GPU를 필요로 하는 현실적 한계를 극복하고자 합니다.

### 주요 기여
1. **체계적 분류 체계**: PEFT 방법론을 4가지 유형으로 분류
   - **Additive PEFT**: 새로운 파라미터 추가 (Adapter, Soft Prompt)
   - **Selective PEFT**: 기존 파라미터 일부만 선택적 학습
   - **Reparameterized PEFT**: 저랭크 분해 기반 (LoRA 등)
   - **Hybrid PEFT**: 여러 방법의 조합

2. **시스템 구현 분석**: 실제 시스템 설계와 구현 비용 조사

3. **효율성 기법 종합**: 프루닝, 양자화, 메모리 최적화 등 계산 복잡도 감소 방법들

## 해결하고자 하는 문제

### 핵심 문제들
1. **계산 비용**: 대규모 모델의 전체 파인튜닝 시 막대한 GPU 자원 필요
2. **메모리 제약**: 수십억 개 파라미터의 저장 및 처리 부담
3. **하드웨어 한계**: 제한된 계산 능력을 가진 플랫폼에서의 모델 커스터마이징 어려움
4. **리소스 효율성**: 각 태스크별로 전체 모델을 저장해야 하는 비효율성

# PEFT 방법론 및 모델 구조 상세 분석

Parameter-Efficient Fine-Tuning(PEFT) 기법은 대규모 사전훈련 모델의 대부분 파라미터를 고정하고, 극히 소수의 파라미터만 조정하여 효율적으로 특정 태스크에 적응시키는 방법론으로, 크게 네 가지 카테고리로 분류된다[1]:  
  1. Additive PEFT (추가형)  
  2. Selective PEFT (선택형)  
  3. Reparameterized PEFT (재매개화형)  
  4. Hybrid PEFT (혼합형)  

각 카테고리의 핵심 원리와 주요 수식은 다음과 같다.

## 1. Additive PEFT  
기존 모델에 **새로운 모듈이나 파라미터를 추가**하여, 추가된 부분만 학습하는 방식이다[1].

### 1.1 Adapter  
파라미터 수가 작은 병목 구조를 Transformer 각 블록 내에 삽입한다. 입력 $$x\in\mathbb{R}^d$$에 대하여:  

$$
\mathrm{Adapter}(x)= W_{\mathrm{up}}\sigma\bigl(W_{\mathrm{down}}\,x\bigr)+x
$$  

여기서 $$W_{\mathrm{down}}\in\mathbb{R}^{r\times d}$$, $$W_{\mathrm{up}}\in\mathbb{R}^{d\times r}$$, $$r\ll d$$, $$\sigma(\cdot)$$는 활성화 함수이다[1].

**설계 변형:**
- **Serial Adapter**: FFN 레이어와 Attention 레이어 뒤에 순차적으로 배치
- **Parallel Adapter**: Transformer 서브레이어와 병렬로 실행되는 사이드 네트워크 구성[1]
- **CoDA**: sparse activation 메커니즘을 사용하여 중요한 토큰만 선별적으로 처리[1]

### 1.2 Soft Prompt Tuning  
각 Transformer 레이어 입력에 학습 가능한 프롬프트 벡터 $$s_i^{(l)}$$를 선행시킨다:  

$$
X^{(l)} = \bigl[s^{(l)}_1,\dots,s^{(l)}\_{N_S},x^{(l)}\_1,\dots,x^{(l)}\_{N_X}\bigr]
$$  

$$N_S$$는 프롬프트 토큰 수, $$N_X$$는 원본 입력 토큰 수이다[1].

**주요 방법들:**
- **Prefix-tuning**: 모든 Transformer 레이어의 key와 value에 학습 가능한 벡터 삽입[1]
- **Prompt-tuning**: 입력 임베딩 레이어에만 프롬프트 적용 (11B 이상 모델에서 효과적)[1][5]
- **P-tuning v2**: 재매개화 제거하고 더 넓은 모델 스케일과 태스크에 적용[1]

## 2. Selective PEFT  
모델 파라미터 $$\theta=\{\theta_i\}_{i=1}^n$$ 중 일부만 선택적으로 학습한다. 이진 마스크 $$m_i\in\{0,1\}$$를 적용하여: 

$$
\theta'_i \;=\; \theta_i \;-\;\eta\,m_i\,\frac{\partial L}{\partial \theta_i}
$$  

여기서 $$\eta$$는 학습률, $$L$$은 손실 함수이다[1].  
- **BitFit**: 오직 bias 파라미터만 학습($$m_i=1$$ for bias).  
- **Diff Pruning**: 가중치별 학습 가능 마스크를 $$\ell_0$$ 근사 정규화로 학습.

**주요 방법들:**
- **BitFit**: 편향(bias) 파라미터만 미세 조정[1]
- **Diff Pruning**: 학습 가능한 이진 마스크를 가중치에 적용[1]
- **Fisher-based selection**: Fisher 정보를 사용하여 중요한 파라미터 선택[1]

## 3. Reparameterized PEFT  
기존 가중치 행렬의 업데이트를 **저차원 행렬의 곱**으로 재표현하여 학습하고, 추론 시 원래 형태로 복원한다[1].

### 3.1 LoRA (Low-Rank Adaptation)  
사전훈련된 $$W_0\in\mathbb{R}^{d\times k}$$에 대해 $$r\ll\min(d,k)$$인 저차원 행렬 $$\Delta W = W_{\mathrm{up}}W_{\mathrm{down}}$$을 추가:  

$$
h_{\mathrm{out}}
= W_0\,h_{\mathrm{in}}
+\frac{\alpha}{r}\,\Delta W\,h_{\mathrm{in}}
= W_0\,h_{\mathrm{in}}
+\frac{\alpha}{r}\,W_{\mathrm{up}}\,W_{\mathrm{down}}\,h_{\mathrm{in}}
$$  

$$\alpha$$는 스케일링 팩터, $$W_{\mathrm{down}}$$은 가우시안 초기화, $$W_{\mathrm{up}}$$은 0 초기화한다[1].

**LoRA 변형들:**
- **DyLoRA**: 훈련 중 동적으로 랭크 조정[1]
- **AdaLoRA**: SVD 기반으로 중요도에 따라 특이값 가지치기[1]
- **DoRA**: 가중치를 크기와 방향으로 분해하여 각각 다르게 조정[1]

### 3.2 기타 재매개화 기법  
- **Compacter**: Kronecker 곱 $$\sum_i A_i\otimes B_i$$로 어댑터 가중치를 저차원 재표현.  
- **VeRA**: 모든 레이어에 공유되는 $$\Lambda_b W_{\mathrm{up}}\Lambda_d W_{\mathrm{down}}$$ 형태로 스케일링 벡터만 학습.

## 4. Hybrid PEFT  
여러 PEFT 방법을 **동시에 통합**하고, 게이팅 메커니즘으로 각 방법의 기여도를 조절한다[1].  
예: UniPELT의 출력:  

$$
\mathrm{Output}
=G_{\mathrm{LoRA}}\cdot \mathrm{LoRA}\_{\mathrm{out}}
+G_{\mathrm{Prefix}}\cdot \mathrm{Prefix}\_{\mathrm{out}}
+G_{\mathrm{Adapter}}\cdot \mathrm{Adapter}\_{\mathrm{out}}
$$  

$$G\in[1]$$는 각 서브모듈의 활성화 정도를 나타내는 게이트 값이다.

위 네 가지 카테고리는 적용 위치(Attention/FFN), 파라미터 수, 일반화 효과, 계산·메모리 효율성 등에 따라 선택·결합하여 사용되며, 대규모 모델을 적은 비용으로 다양한 태스크에 효과적으로 적응시키는 핵심 기법으로 자리매김하고 있다[1].

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/e5a2c3ee-e4a7-4007-9add-4cb253db1e5a/2403.14608v7.pdf

## 3. 모델 구조 분석 (LLaMA 기준)

### 3.1 기본 LLaMA 아키텍처

**전체 구조:**
1. **임베딩 블록**: 텍스트를 수치 벡터로 변환
2. **디코더 스택**: Multi-head Self-Attention (MSA) + Feed-Forward Network (FFN)
3. **헤드 블록**: 선형 레이어 + 소프트맥스[1]

**핵심 연산:**
```
Q, K, V = RoPE(W_q x), RoPE(W_k x), W_v x
SA(x) = Softmax(QK^T/√d_head) V
MSA(x) = [SA_1(x); SA_2(x); ...; SA_k(x)] W_o
```

### 3.2 PEFT 적용 위치별 분석

**Attention 블록:**
- **LoRA**: Q, K, V 투영 행렬에 적용
- **Adapter**: MSA 출력 후 삽입
- **Prefix-tuning**: Key, Value에 학습 가능한 프리픽스 추가[1]

**FFN 블록:**
- **LoRA**: FFN의 선형 레이어에 적용
- **Adapter**: FFN 출력 후 삽입
- **BitFit**: FFN의 bias 파라미터만 조정[1]

## 4. 효율성 및 성능 분석

### 4.1 파라미터 효율성 비교

- **LoRA**: 원본 파라미터의 0.1-1%만 학습
- **Adapter**: 2-8% 추가 파라미터
- **Soft Prompt**: 1% 미만
- **실제 사례**: LLaMA-Adapter는 LLaMA-7B에 단 1.2M 파라미터만 추가[1]

### 4.2 메모리 및 계산 효율성

**메모리 사용량:**
- 기존 PEFT: 전체 파인튜닝 대비 ~70% 메모리 사용
- **QLoRA**: 단일 48GB GPU에서 65B 모델 파인튜닝 가능[1]

**훈련 시간:**
- **LLaMA-Adapter**: 8개 A100 GPU에서 1시간 미만 파인튜닝[1]

## 5. 일반화 성능 및 장점

### 5.1 모델 일반화 능력

1. **과적합 위험 감소**: 제한된 파라미터 공간으로 인한 정규화 효과
2. **사전훈련 지식 보존**: 대부분 가중치 고정으로 원본 능력 유지
3. **크로스 태스크 일반화**: 다중 태스크 적응 가능[1]

### 5.2 도메인 적응 및 확장성

- **다중 모달 적용**: 비전, 언어, 멀티모달 태스크에 성공적 적용
- **연속 학습**: 재앙적 망각 방지하면서 성능 유지
- **스케일링 특성**: 모델 크기가 클수록 효과 증대[1]

이러한 PEFT 방법론들은 각각의 고유한 장단점을 가지고 있으며, 특정 태스크나 리소스 제약에 따라 적절한 방법을 선택하거나 조합하여 사용할 수 있습니다. 특히 LoRA와 같은 reparameterized 방법이 현재 가장 널리 사용되고 있으며, 다양한 변형과 개선 방법들이 지속적으로 연구되고 있습니다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/e5a2c3ee-e4a7-4007-9add-4cb253db1e5a/2403.14608v7.pdf
[2] https://www.ibm.com/kr-ko/think/topics/parameter-efficient-fine-tuning
[3] https://data-newbie.tistory.com/982
[4] https://velog.io/@khs0415p/Paper-Prompt-Tuning
[5] https://yumdata.tistory.com/406
[6] https://github.com/Aradhye2002/selective-peft-toolkit
[7] https://x2bee.tistory.com/335
[8] https://aclanthology.org/2022.acl-long.433.pdf
[9] https://www.sec.gov/Archives/edgar/data/1969302/000141057825000895/pony-20241231x20f.htm
[10] https://www.sec.gov/Archives/edgar/data/1823986/000141057825000891/wdh-20241231x20f.htm
[11] https://www.sec.gov/Archives/edgar/data/1477960/000147793225002922/cbbb_10k.htm
[12] https://www.sec.gov/Archives/edgar/data/1768259/000095017025056468/gotu-20241231.htm
[13] https://www.sec.gov/Archives/edgar/data/1530766/000164117225004816/form10-k.htm
[14] https://www.sec.gov/Archives/edgar/data/1921865/000095017025047694/aspi-20241231.htm
[15] https://arxiv.org/abs/2403.14608
[16] https://www.semanticscholar.org/paper/a244d417c6c106a98df19030bc64bcdf5717aa56
[17] https://arxiv.org/abs/2402.02242
[18] https://arxiv.org/abs/2402.04401
[19] https://arxiv.org/abs/2205.05638
[20] https://arxiv.org/abs/2401.04679
[21] https://littlefoxdiary.tistory.com/120
[22] https://jjaegii.tistory.com/34
[23] https://chaksseu.tistory.com/59
[24] https://inpa.tistory.com/entry/GOF-%F0%9F%92%A0-%EC%96%B4%EB%8C%91%ED%84%B0Adaptor-%ED%8C%A8%ED%84%B4-%EC%A0%9C%EB%8C%80%EB%A1%9C-%EB%B0%B0%EC%9B%8C%EB%B3%B4%EC%9E%90
[25] https://chalchichi.tistory.com/125
[26] https://day-to-day.tistory.com/69
[27] https://lsoovmee-rhino.tistory.com/entry/%EB%94%94%EC%9E%90%EC%9D%B8%ED%8C%A8%ED%84%B4-%EA%B5%AC%EC%A1%B0%ED%8C%A8%ED%84%B41-Adapter-Pattern-%EC%96%B4%EB%8C%91%ED%84%B0-%ED%8C%A8%ED%84%B4
[28] https://www.sec.gov/Archives/edgar/data/1915403/000117891325001156/zk2532913.htm
[29] https://www.sec.gov/Archives/edgar/data/1047127/000104712725000030/amkr-20241231.htm
[30] https://www.sec.gov/Archives/edgar/data/1894562/000162828025008884/prme-20241231.htm
[31] https://www.sec.gov/Archives/edgar/data/2018462/000149315225006944/form20-f.htm
[32] https://www.sec.gov/Archives/edgar/data/1680062/000168006225000003/acmr-20241231.htm
[33] https://www.sec.gov/Archives/edgar/data/1780652/000141057825000805/can-20241231x20f.htm
[34] https://arxiv.org/pdf/2205.12309.pdf
[35] https://aclanthology.org/2023.emnlp-main.884.pdf
[36] https://aclanthology.org/2022.emnlp-main.758.pdf
[37] https://arxiv.org/html/2502.12200v1
[38] http://arxiv.org/pdf/2406.19486.pdf
[39] https://arxiv.org/html/2208.10160v2
[40] https://kevin-rain.tistory.com/213
[41] https://www.redhat.com/ko/topics/ai/what-is-peft
[42] https://www.themoonlight.io/en/review/a-survey-on-parameter-efficient-fine-tuning-for-foundation-models-in-federated-learning
[43] https://www.sec.gov/Archives/edgar/data/1744494/000182912625004289/adventtech_10k.htm
[44] https://www.sec.gov/Archives/edgar/data/906107/000095017023058214/eqr-20230930.htm
[45] https://www.sec.gov/Archives/edgar/data/931182/000095017023058214/eqr-20230930.htm
[46] https://www.sec.gov/Archives/edgar/data/931182/000095017024120716/eqr-20240930.htm
[47] https://www.sec.gov/Archives/edgar/data/1978124/000110465923082899/tm2316508-1_s4.htm
[48] https://www.sec.gov/Archives/edgar/data/1744494/000182912624005453/adventtech_10k.htm
[49] https://www.themoonlight.io/ko/review/self-supervised-feature-distillation-and-design-of-experiments-for-efficient-training-of-micromechanical-deep-learning-surrogates
[50] http://m.ekcls.kr/attach/board/1625708720_01.pdf
[51] https://www.ultralytics.com/ko/glossary/parameter-efficient-fine-tuning-peft
[52] https://coco0414.tistory.com/110
[53] https://www.sec.gov/Archives/edgar/data/1329099/000119312525066199/d853848d20f.htm
[54] https://www.ewadirect.com/proceedings/ace/article/view/10370
[55] https://arxiv.org/abs/2304.01933
[56] https://arxiv.org/abs/2405.19597
[57] https://arxiv.org/abs/2406.03792
[58] https://ariz1623.tistory.com/348
[59] https://blog.kbanknow.com/81
[60] https://lanad.tistory.com/40
[61] https://www.semanticscholar.org/paper/4ba0acf0bcc594e5408bd82ec9788f948d0b8b73
[62] https://aclanthology.org/2023.emnlp-main.576.pdf
[63] http://arxiv.org/pdf/2404.14607.pdf
[64] https://aclanthology.org/2022.findings-emnlp.511.pdf
[65] https://velog.io/@becky-kwon/Fine-Tuning%EA%B3%BC-Prompt-Engineering-Prompt-Tuning-PEFT%EC%97%90-%EB%8C%80%ED%95%9C-%EC%9D%B4%EC%95%BC%EA%B8%B0
[66] https://stydy-sturdy.tistory.com/45
[67] https://isaac-christian.tistory.com/entry/AI-%EB%AA%A8%EB%8D%B8-%EC%B5%9C%EC%A0%81%ED%99%94-%EB%B0%A9%EB%B2%95-Fine-Tuning-%EB%B0%8F-Prompt-Tuning
[68] https://aclanthology.org/2024.findings-eacl.122.pdf
[69] https://www.sec.gov/Archives/edgar/data/1474835/000165495423008028/ipii_20f.htm
[70] https://www.sec.gov/Archives/edgar/data/931182/000095017023015831/eqr-20230331.htm
[71] https://www.sec.gov/Archives/edgar/data/906107/000095017023015831/eqr-20230331.htm
[72] https://www.sec.gov/Archives/edgar/data/1805521/000121390025039266/ea0238761-s1a1_faraday.htm
[73] https://koreascience.kr/article/JAKO202430459758868.pdf
[74] https://www.koreascience.kr/article/JAKO202514150407984.pdf
## 성능 향상 및 효율성

### 파라미터 효율성
- **LoRA**: 원본 파라미터의 0.1-1%만 학습
- **Adapter**: 2-8% 추가 파라미터
- **Soft Prompt**: 1% 미만
- **실제 사례**: LLaMA-Adapter는 LLaMA-7B에 단 1.2M 파라미터만 추가

### 메모리 및 시간 효율성
- 기존 PEFT 방법들은 전체 파인튜닝 대비 ~70% 메모리 사용
- QLoRA는 단일 48GB GPU에서 65B 모델 파인튜닝 가능
- LLaMA-Adapter는 8개 A100 GPU에서 1시간 미만 파인튜닝

## 일반화 성능 향상 가능성

### 모델 일반화 능력
1. **과적합 위험 감소**: 제한된 파라미터 공간으로 인한 정규화 효과
2. **사전훈련 지식 보존**: 대부분의 가중치가 고정되어 원본 능력 유지
3. **크로스 태스크 일반화**: AdapterFusion, LoRAHub 등을 통한 다중 태스크 적응

### 도메인 적응 능력
- 비전, 언어, 멀티모달 태스크에 성공적 적용
- 연속 학습 시나리오에서 효과적
- 재앙적 망각 방지하면서 성능 유지

### 스케일링 특성
- 모델 크기가 클수록 효과 증대
- 110억 파라미터 이상 모델에서 프롬프트 튜닝 특히 효과적
- LoRA는 1750억 파라미터 모델까지 확장 가능

## 주요 한계점

1. **하이퍼파라미터 민감성**: LoRA의 랭크, Adapter의 병목 차원 등에 민감
2. **태스크별 최적화**: 모든 시나리오에 범용적으로 최적인 방법 부재
3. **훈련 불안정성**: 소프트 프롬프트 방법의 훈련 불안정성
4. **계산 오버헤드**: 일부 방법에서 여전한 메모리 부담

## 향후 연구에 미치는 영향

### 알고리즘 발전 방향
1. **하이퍼파라미터 자동화**: 수동 튜닝 의존도 감소를 위한 자동 랭크 선택
2. **훈련 효율성 향상**: 메모리 효율적 기법과 모델 압축 기법의 통합
3. **스케일링 법칙 탐구**: 새로운 아키텍처(Mamba, LVM)에 대한 PEFT 적응

### 시스템 레벨 혁신
1. **데이터 프라이버시 강화**: 연합 학습 프레임워크, 암호화 프로토콜
2. **하드웨어 최적화**: PEFT 연산 전용 커널, 압축 모델용 특화 하드웨어
3. **멀티테넌트 서빙**: 동적 어댑터 로딩, 자원 할당 최적화

### 연구 시 고려사항

1. **통합 벤치마킹**: 표준화된 평가 지표와 공정한 비교 프레임워크 필요
2. **이론적 이해**: PEFT 효과성의 수학적 기초와 일반화 보장에 대한 연구
3. **실용적 배포**: 실제 운영 시스템 통합과 비용-효익 분석 프레임워크
4. **윤리적 고려사항**: 편향 완화, 적대적 공격에 대한 강건성, 책임감 있는 AI 배포

이 논문은 PEFT 분야의 **체계적 이해와 실용적 가이드라인**을 제공함으로써, 대규모 모델의 효율적 적응에 대한 연구 방향을 제시하고 있습니다. 특히 **일반화 성능 유지와 계산 효율성의 균형**이라는 핵심 과제를 해결하기 위한 포괄적 접근법을 제안하고 있어, 향후 AI 모델의 실용적 배포에 중요한 기여를 할 것으로 예상됩니다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/e5a2c3ee-e4a7-4007-9add-4cb253db1e5a/2403.14608v7.pdf
