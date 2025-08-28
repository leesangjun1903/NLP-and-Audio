# GLM: General Language Model Pretraining with Autoregressive Blank Infilling

**주요 주장 및 기여 요약**  
GLM은 **자연어 이해(NLU)**, **무조건 생성(unconditional generation)**, **조건부 생성(conditional generation)**을 하나의 사전학습 프레임워크에서 모두 우수하게 수행할 수 있도록 설계된 범용 언어 모델이다.  
- 기존 BERT, GPT, T5 중 하나의 과제군에만 강점을 보이던 한계를 극복  
- **Autoregressive Blank Infilling** 목표함수와 **2D 위치 인코딩** 도입으로 NLU와 생성 과제를 동시에 효과적으로 학습  
- 단일 모델(340M 파라미터급)로 SuperGLUE에서 BERT 대비 평균 +5.0%p 성능 향상, 다양한 생성·인퍼런스 과제에서도 SOTA 성능 달성

***

## 1. 해결하고자 하는 문제  
기존 사전학습 모델은 다음과 같은 한계를 지닌다.  
- BERT: 양방향 문맥 인코딩에 특화되어 생성 능력 부재  
- GPT: 단방향 문맥 모델링으로 NLU 태스크에 취약  
- T5/BART: 인코더-디코더 아키텍처로 모든 과제군 통합 가능하나, **파라미터·데이터 효율이 낮음**  

따라서 **단일 모델**이 NLU·무조건 생성·조건부 생성 세 영역을 모두 **효율적으로** 다루도록 하는 사전학습 프레임워크가 필요하다.

***

## 2. 제안 방법

### 2.1 Autoregressive Blank Infilling 목표  
입력 문장 $$\mathbf{x}=[x_1,\dots,x_n]$$에서 연속 토큰 **스팬** $$\{s_1,\dots,s_m\}$$을 무작위로 선택해 각 스팬을 $$\mathrm{[MASK]}$$로 대체하여 손상 텍스트 $$x_{\text{corrupt}}$$를 생성한다.  
이후 스팬들을 **무작위 순서** $$z\sim \mathrm{perm}(1,\dots,m)$$ 로 채우며, 이전에 생성된 스팬을 조건으로 다음 스팬 $$s_{z_i}$$을 **자동회귀적으로** 예측한다:

```math
\max_\theta \mathbb{E}_{z}\;\sum_{i=1}^m \log p_\theta\bigl(s_{z_i}\mid x_{\text{corrupt}}, s_{z_{ < i}}\bigr).
```

각 스팬 내부는 일반적인 순방향 언어모델처럼 토큰별 확률 분해:

$$
p_\theta\bigl(s_i\bigr)
=\prod_{j=1}^{\ell_i}p\bigl(s_{i,j}\mid x_{\text{corrupt}}, s_{ < j}\bigr).
$$

### 2.2 2D 위치 인코딩  
- 첫 번째 차원: **원본 문장 위치** (스팬 영역은 마스크 위치)  
- 두 번째 차원: **스팬 내 상대 위치** (Part B 토큰만 1부터 증가)  

이를 통해 모델이 **미리 생성할 길이**를 알지 못하는 스팬을 효과적으로 처리하며, NLU·생성 간 일관된 인코딩 보장.

### 2.3 멀티태스크 사전학습  
- **짧은 스팬 마스킹** (NLU 성능 최적화)  
- **문서 수준 스팬** (50–100% 길이): 무조건 생성(pretrain)  
- **문장 수준 스팬** (15% 토큰 문장 단위): 조건부 생성(pretrain)  

세 목표를 배치 내 균등 샘플링으로 학습해 **하나의 모델**에서 모든 과제를 아우름.

### 2.4 모델 구조  
기본은 Transformer 기반  
- 레이어 정규화 및 잔차연결(Reordering)  
- GeLU 활성화  
- 단일 선형층으로 출력  
- Self-attention 마스크로 Part A(문맥)와 Part B(스팬) 역할 분리 및 자동회귀 보장

***

## 3. 성능 향상 및 한계

### 3.1 SuperGLUE 성능  
- GLMBase vs. BERTBase: 평균 +4.6%p  
- GLMLarge vs. BERTLarge: 평균 +5.0%p  
- GLMRoBERTa vs. T5Large: 비슷한 성능이나 파라미터 절반  

### 3.2 생성 과제  
- **추상적 요약**(CNN/DailyMail, XSum): BART 대비 동등 성능  
- **텍스트 인필링**(Yahoo): BLEU +1.3–3.9 향상  
- **언어 모델링**(BookWiki perplexity, LAMBADA 정확도): GPTLarge 대비 1.25× 모델로 동등·우월  

### 3.3 제한점  
- 문서 수준 생성 객관적 이득은 NLU에 비해 다소 제한적  
- 대규모(>500M) 모델 학습 리소스 및 최적화 필요  
- Span shuffle 제거·다중 센티넬 토큰 실험에서 성능 급락: 설계 복잡도·일반화 trade-off 존재

***

## 4. 일반화 성능 향상 관점  
GLM의 **blank infilling**은 NLU를 *조건부 생성* 문제로 환원해, 생성 능력이 NLU 성능에 기여하도록 학습 **일관성**을 확보한다.  
- **2D 위치 인코딩**으로 스팬 길이 정보 은닉, 다양한 길이 정답 처리 일반화  
- **스팬 셔플링**으로 예측 순서 불확정성 도입, 토큰 상호 의존성 학습 강화  
- **멀티태스크 학습**으로 서로 다른 과제 간 파라미터 공유, 새로운 과제 전이학습 효과

***

## 5. 향후 연구 방향 및 고려사항  
- **효율화**: 500M+ 파라미터급 모델 학습·추론 비용 절감 방안 (프루닝, 양자화)  
- **스팬 샘플링 전략**: 데이터·태스크별 최적 마스킹 비율·분포 자동화  
- **다국어·멀티모달** 확장: blank infilling 아이디어를 비영어권·비텍스트 영역에 적용  
- **안정성·편향 제어**: 자동회귀 생성을 통한 부적절 출력 리스크 완화 기법 통합  

GLM은 NLU와 생성 과제의 **통합 사전학습** 가능성을 제시하며, 범용 언어 모델 연구의 중요한 전환점을 마련했다. 미래 연구는 자원 효율성과 안전성, 적용 범위 확대에 초점을 맞춰야 할 것이다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/e6b3390c-f20e-40b6-a2b8-9b4e0ae996f5/2103.10360v2.pdf)
