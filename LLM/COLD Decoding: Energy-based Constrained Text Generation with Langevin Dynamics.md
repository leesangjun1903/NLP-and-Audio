# COLD Decoding: Energy-based Constrained Text Generation with Langevin Dynamics

**주요 주장:** COLD(Constrained Decoding with Langevin Dynamics)은 에너지 기반 모델링을 통해 다양한 **하드**(키워드 포함) 및 **소프트**(좌·우 문맥 일치) 제약을 하나의 에너지 함수로 통합하고, 이 함수의 그라디언트를 이용해 Langevin Dynamics 샘플링을 수행함으로써, 사전학습된 언어모델을 미세조정 없이도 제약을 만족하는 고품질 텍스트를 생성할 수 있음을 보인다.[1]

**주요 기여:**
- 제약 조건을 에너지 함수 $$E(\tilde y)= -\sum_i\lambda_i f_i(\tilde y)$$ 로 통합하여 단일 프레임워크로 처리.[1]
- 이산 텍스트를 연속 벡터 공간으로 완화하고, 그 위에서 Langevin Dynamics를 적용하여 효율적이고 미분 가능한 제약 추론 수행.[1]
- “Top-k 필터링” 기반의 가이디드 이산화(discretization)로 유창성과 제약 준수를 동시에 확보.[1]
- 세 가지 응용(어보덕티브 추론, 반사실적(reasoning), 키워드 제약)에 걸쳐 자동·인간 평가에서 기존 방법 대비 우수한 성능 입증.[1]

***

## 1. 해결하고자 하는 문제

텍스트 생성 시
- **하드 제약**: 반드시 특정 키워드를 포함해야 함(e.g., 지식 기반 생성)  
- **소프트 제약**: 좌·우 문맥 일치, 최소 편집 제한 등  
전통적 **오토리그레시브** 디코딩(빔 서치, nucleus 샘플링)은 이러한 복합 제약을 실시간으로 만족시키기 어려우며, 작업별 미세조정 비용이 과도하다.[1]

***

## 2. 제안하는 방법

### 2.1 에너지 기반 생성

생성 분포를 다음의 볼츠만 형태로 정의:  

$$
p(\tilde y)\;=\;\frac{\exp\bigl(-E(\tilde y)\bigr)}{Z},\quad
E(\tilde y)= -\sum_i \lambda_i f_i(\tilde y)
$$

여기서 $$f_i$$는 유창성, 키워드 포함, 문맥 일치 등의 제약 함수, $$\lambda_i$$는 가중치이다.[1]

### 2.2 Langevin Dynamics를 통한 샘플링

$$\tilde y\in\mathbb R^{T\times|V|}$$을 연속 로그잇으로 정의하고, 반복적으로

$$
\tilde y^{(n+1)} = \tilde y^{(n)} - \eta\,\nabla_{\tilde y}E\bigl(\tilde y^{(n)}\bigr) \;+\;\epsilon^{(n)},\quad
\epsilon^{(n)}\sim\mathcal N(0,\sigma)
$$

을 수행하여 최종 연속 샘플 $$\tilde y^{(N)}$$를 얻는다.[1]

### 2.3 주요 제약 함수

- **소프트 유창성** $$f_{\text{LM}}$$: LM 예측 확률과 연속 분포 간의 교차엔트로피  
- **미래 문맥 일치** $$f_{\text{pred}}$$: 우측 컨텍스트 $$x_r$$의 예측 우도  
- **n-그램 유사도** $$f_{\text{sim}}$$: 주어진 키워드 또는 참조문장과의 differentiable BLEU-n 근사  
- **역방향 LM** $$f_{\leftarrow\text{LM}}$$: 우측 컨텍스트 유창성 강화.[1]

### 2.4 연속→이산 변환

Top-k 필터링: 각 위치별 상위 $$k$$ 후보 토큰을 LM이 제안하면, 연속 로그잇 $$\tilde y_t$$ 값을 기준으로 선택함으로써 유창성과 제약 준수를 보장.[1]

***

## 3. 모델 구조

- 백본으로 **사전학습된 GPT-2/XL** 사용  
- 제약별 $$f_i$$는 LM 연산(양방향), n-그램 매칭, 키워드 임베딩 등으로 구성  
- 에너지 함수는 $$\sum_i\lambda_i f_i$$ 형태로 결합  
- Langevin Dynamics 단계 후 top-k 이산화.[1]

***

## 4. 성능 향상 및 한계

### 4.1 어보덕티브 추론(abductive reasoning)

- **자동 평가**: BLEU↑·ROUGE↑·CIDEr↑·BERTScore↑  
- **인간 평가**: 좌·우 문맥 일관성 및 문법성 동시 향상  
- **기존 DELOREAN 대비** 전체 일관성 7%p 이상 향상.[1]

### 4.2 반사실적 스토리 생성(counterfactual reasoning)

- **최소 편집**과 **문맥 일관성** 균형 확보  
- Mix-and-Match 대비 샘플링 효율 및 품질 우수.[1]

### 4.3 키워드 제약(lexically-constrained decoding)

- **키워드 커버리지** 94.5%로 NEUROLOGIC(91.0%) 상회  
- 플루언시(Perplexity)는 소폭 하락하나 인간 평가는 양호.[1]

### 한계

- **고정 길이** 생성 후 후처리 필요  
- **Langevin 스텝 수** 및 **$$\lambda_i$$** 민감도  
- **대규모 LM** 사용 시 계산 비용 증가(file:1).

***

## 5. 일반화 성능 향상 관점

- **제약 조합 유연성**: 새로운 $$f_i$$ 정의만으로 다양한 제약 추가 가능  
- **미분 가능성**: 그라디언트 기반 탐색으로 전역 탐색과 국부 최적화 균형  
- **샘플 앤 셀렉트**: 다중 샘플링 후 평가 지표별 선택으로 일반화 저하 방지.[1]

***

## 6. 향후 연구 영향 및 고려사항

- **프롬프트 제약**: 제약 함수 학습, 자동 가중치 조정(메타러닝)  
- **효율적 샘플링**: 절단된 Langevin Dynamics, 하이브리드 MCMC  
- **대체 백본 모델**: GPT-3, PaLM 등 초거대 모델과 연동 시 계산 최적화  
- **제약 함수 확장**: 스타일, 윤리, 논리적 일관성 등 추가 가능성  
- **편향 및 악용 방지**: 제약을 통한 편향 억제, 안전성 검증 필요.[1]

이와 같이 COLD는 **제약 기반 텍스트 생성** 연구 방향에 유연하고 강력한 샘플링 기반 해법을 제시하여, 사전학습 모델의 범용 디코딩 역량을 확장한다.[1]

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/07dd048e-1dab-4c04-9f0e-9172fdf784c5/2202.11705v3.pdf)
