# Large Language Models Post-training: Surveying Techniques from Alignment to Reasoning

### 1. 논문의 핵심 주장 및 주요 기여

**"Large Language Models Post-training: Surveying Techniques from Alignment to Reasoning"** 논문은 LLM 분야의 최초의 포괄적 포스트트레이닝 서베이입니다. 본 논문의 핵심 주장은 사전학습(pre-training)만으로는 LLM의 제한된 추론 능력, 윤리적 불확실성, 도메인 특화 성능 부족을 해결할 수 없으므로, **체계적인 포스트트레이닝 기법이 필수**라는 것입니다.[1]

**주요 기여**는 다음과 같습니다:[1]

- **포괄적 역사 통합**: ChatGPT의 RLHF에서 DeepSeek-R1의 혁신적 추론 기법까지의 진화 과정 체계화
- **구조화된 분류 체계**: 5가지 핵심 패러다임(Fine-tuning, Alignment, Reasoning, Efficiency, Integration and Adaptation)으로 포스트트레이닝 기법 조직화
- **미래 방향 제시**: 특히 대규모 RL을 활용한 Large Reasoning Models(LRMs)의 중요성 강조

***

### 2. 해결하고자 하는 문제와 제안 방법

#### 2.1 핵심 문제점[1]

LLM의 사전학습된 아키텍처는 다음과 같은 근본적 한계를 드러냅니다:

1. **제한된 추론 능력**: 다단계 논리 추론 및 복잡한 문제 해결 어려움
2. **윤리적 불확실성**: 안전성, 편향성, 정렬도 문제
3. **도메인 특화 성능 부족**: 특정 분야의 미세한 뉘앙스 미이해

#### 2.2 제안하는 포스트트레이닝 기법 체계

**5가지 핵심 패러다임**:[1]

##### (1) **Fine-tuning (미세조정)**

**Supervised Fine-Tuning (SFT)**의 핵심 목적함수:

$$L_{\text{fine-tune}} = -\frac{1}{N} \sum_{i=1}^{N} \sum_{j=1}^{C} y_{ij} \log P(y_j|x_i)$$

여기서 $y_{ij}$는 샘플 $i$의 카테고리 $j$ 참 레이블, $P(y_j|x_i)$는 모델의 예측 확률입니다[1].

**Dataset Quality Screening Function**:

$$\mathcal{D}' = \{(I_k, X_k) \in \mathcal{D} : r(I_k, X_k) \geq \tau\}$$

여기서 $r$은 품질 평가 함수, $\tau$는 품질 임계값입니다.[1]

**Parameter-Efficient Fine-Tuning (PEFT)** 중 LoRA의 공식:

$$h_{\text{out}} = W_0 h_{\text{in}} + \Delta W h_{\text{in}} = W_0 h_{\text{in}} + W_{\text{up}} W_{\text{down}} h_{\text{in}}$$

여기서 $W_{\text{up}} \in \mathbb{R}^{d \times r}$, $W_{\text{down}} \in \mathbb{R}^{r \times k}$입니다.[1]

##### (2) **Alignment (정렬)**

**RLHF (강화학습 with 인간 피드백)**:[1]

조건부 확률 모델:

$$\pi_\theta(x_0, \ldots, x_{n-1}) = \prod_{k=0}^{n} p_\theta(x_k | x_0 \cdots x_{k-1})$$

기본 목적함수:

$$\max_\theta \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_\theta} R(x, y)$$

여기서 $R(x,y)$는 보상 함수, $\pi_\theta$는 정책입니다.[1]

**PPO (Proximal Policy Optimization) 클리핑 목적함수**:[1]

$$L^{\text{CLIP}}_t(\theta) = \mathbb{E}_t \left[\min(r_t(\theta) A_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t)\right]$$

여기서 $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\text{old}}(a_t|s_t)}$는 확률 비율, $A_t$는 추정 이득입니다[1].

**DPO (Direct Preference Optimization)**:[1]

KL-제약 보상 최대화에서 도출된 최적 정책:
$$\pi^*(y|x) = \frac{1}{Z(x)} \pi_{\text{ref}}(y|x) \exp\left(\frac{1}{\beta} r(x,y)\right)$$

여기서 $Z(x) = \sum_y \pi_{\text{ref}}(y|x) \exp(\frac{1}{\beta} r(x,y))$는 분할 함수입니다[1].

보상을 정책으로 재매개변수화:
$$r(x,y) = \beta \log \frac{\pi^*(y|x)}{\pi_{\text{ref}}(y|x)} - \beta \log Z(x)$$

Bradley-Terry 선호도 모델:
$$p(y_w > y_l|x) = \frac{\exp(r(x,y_w))}{\exp(r(x,y_w)) + \exp(r(x,y_l))}$$

DPO 목적함수:

$$\mathcal{L}_{\text{DPO}}(\pi_\theta; \pi_{\text{ref}}) = -\mathbb{E}_{(x,y_w,y_l) \sim \mathcal{D}} \left[\log \sigma\left(\beta \log\frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log\frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right)\right]$$

여기서 $\sigma$는 로지스틱 시그모이드 함수입니다.[1]

**GRPO (Group Relative Policy Optimization)**:[1]

그룹 기반 이득 추정으로 PPO를 개선:

$$\mathcal{J}_{\text{GRPO}} = \mathbb{E}_{q} \sum_{i=1}^{G} \mathbb{E}_{o \sim \mathcal{P}(q,o_i)} \left[\min\left(\frac{\pi(o|q)}{\pi_{\text{old}}(o|q)} A_{i,t}, \text{clip}\left(\frac{\pi(o|q)}{\pi_{\text{old}}(o|q)}, 1-\epsilon, 1+\epsilon\right) A_{i,t}\right) - \lambda D_{\text{KL}}(\pi_{\text{ref}}, \pi)\right]$$

여기서 $A_{i,t}$는 그룹 내 상대 보상에 기반한 이득입니다.[1]

##### (3) **Reasoning (추론)**

**MDP로의 추론 공식화**:[1]

마르코프 결정 과정: $\mathcal{M} = \langle \mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma \rangle$

누적 보상 최대화 목적:
$$J = \mathbb{E}\left[\sum_{t=1}^{T} \gamma^{t-1} R(s_t, a_t)\right]$$

**보상 설계**:[1]

- **이진 정확성 보상**: $r_T = 1$ (정답) 또는 $r_T = 0$ (오답)
- **단계별 정확성 보상**: 각 추론 단계별 피드백
- **자기 일관성 보상**: 여러 경로의 일관성 측정
- **선호도 기반 보상**: 인간 또는 AI 피드백에서 파생

**적응적 탐색 메커니즘**:[1]

$$\mathcal{L}_{\text{PPO}} = \mathbb{E}\left[\min\left(\frac{\pi_\theta(a_s)}{\pi_{\text{old}}(a_s)} A_t, \text{clip}\left(\frac{\pi_\theta(a_s)}{\pi_{\text{old}}(a_s)}, 1-\epsilon, 1+\epsilon\right) A_t\right) - \lambda_t H(s)\right]$$

여기서 $\lambda_t = \alpha \exp(\text{Var}[R_{1:t}])$는 보상 분산에 따라 적응적으로 조정됩니다.[1]

##### (4) **Efficiency (효율성)**

**양자화 공식**:[1]

$$X_L = \text{Round}(K \cdot X_H), \quad K = \frac{\text{absmax}(X_L)}{\text{absmax}(X_H)}$$

여기서 $X_H$는 고정밀(fp32), $X_L$은 저정밀(int8) 표현입니다.[1]

**어댑터 구조**:

$$\text{Adapter}(x) = W_{\text{up}} \sigma(W_{\text{down}} x) + x$$

여기서 $\sigma$는 활성화 함수, 차원 축소를 통해 계산 효율성을 향상합니다.[1]

**지식 증류 손실함수**:[1]

$$\mathcal{L}_{\text{KD}} = (1-\alpha) \mathcal{L}_{\text{CE}}(y, y_s) + \alpha \mathcal{L}_{\text{KL}}(p^t, p^s)$$

여기서 $\mathcal{L}\_{\text{CE}}$ 는 교차 엔트로피, $\mathcal{L}_{\text{KL}}$는 쿨백-라이블러 발산입니다.[1]

##### (5) **Integration and Adaptation (통합 및 적응)**

**Model Merging - SLERP (구면 선형 보간)**:

$$\theta_{\text{merged}} = \text{SLERP}(\theta_1, \theta_2, \lambda)$$

가중치 기반 머징:
$$\theta_{\text{merged}} = \lambda \theta_1 + (1-\lambda) \theta_2, \quad \lambda \in $$[1]

***

### 3. 모델 구조 및 아키텍처

#### 3.1 Transformer 기반 아키텍처[1]

LLM의 기본 구조는 Transformer 인코더-디코더이며, 자기주의(self-attention) 메커니즘을 활용합니다.

**Self-Attention 공식**:[1]

$$\text{SA}(x) = \text{Softmax}\left(\frac{Q l_k K^T}{\sqrt{d_h}}\right) l_v V$$

여기서 $Q, K, V$는 쿼리, 키, 값 행렬, $l_k, l_v$는 학습 가능한 스케일 벡터입니다.[1]

**Feed-Forward 네트워크**:[1]

$$\text{FFN}_{\text{transformer}}(x) = W_{\text{up}} \sigma(W_{\text{down}} x)$$

#### 3.2 Mixture of Experts (MoE) 아키텍처[1]

**DeepSeek-V2 및 V3의 경우**:
- 236B 총 파라미터 (V2)
- 671B 총 파라미터 (V3)
- 160개 전문가 중 선택적 활성화

라우팅 메커니즘:
$$z_{i,j} = \text{Router}(x_i) \rightarrow \text{TopK 선택}$$

#### 3.3 DeepSeek-R1의 다단계 트레이닝 파이프라인[1]

1. **Stage 0**: 기본 모델 (DeepSeek-V3-Base)
2. **Stage 1**: Cold-start RL (소량의 구조화된 데이터)
3. **Stage 2**: 대규모 RL (추론 + 비추론 데이터)
4. **Stage 3**: 최종 정렬 (도움됨/해로움 이중 목적)

***

### 4. 성능 향상 및 벤치마크 평가

#### 4.1 DeepSeek-R1의 주요 성과[2][3][1]

**수학 추론**:
- **AIME 2024**: DeepSeek-R1-Zero는 15.6% → 71.0% (다수결 투표: 86.7%)
- **MATH-500**: 97.3% (OpenAI o1-1217와 동등)
- **AIME 2025**: 40.6% (o3과 비교 가능)

**코딩 성능**:
- **Codeforces**: 전문가 수준 (상위 9%)
- **Aider-Polyglot**: 92.3% (AlpacaEval 2.0 기준)

**일반 지식**:
- **MMLU**: 90.8%
- **MMLU-Pro**: 84.0%
- **GPQA Diamond**: 71.5%

#### 4.2 지식 증류 효과[2][1]

DeepSeek-R1의 추론 능력을 더 작은 모델로 전이:

| 모델 | AIME 2024 | MATH-500 |
|-----|-----------|----------|
| DeepSeek-R1-Distill-Qwen-1.5B | 28.9% | 83.9% |
| DeepSeek-R1-Distill-Qwen-7B | 55.5% | 94.2% |
| DeepSeek-R1-Distill-Qwen-32B | 72.6% | 94.3% |

#### 4.3 RL 훈련 과정에서의 성능 궤적[1]

DeepSeek-R1-Zero의 AIME 2024 성과:
- 초기 (Step 0): 15.6%
- 중기 (Step 8200): 71.0%
- 최종 (다수결): 86.7%

**평탄화 완화 현상**: 훈련 초기 급속한 개선, 이후 안정화

***

### 5. 모델의 일반화 성능 향상

#### 5.1 도메인 외 평가[1]

**OlympiadBench**: PLAN-TUNING 적용 시 ~10% 개선[4]
**AIME 변형**: 일관된 성능 이전[1]

#### 5.2 토큰 효율성과 추론 길이[3]

- **자동 추론 길이 조정**: 문제 복잡도에 따라 동적 할당
- **다수결 투표**: 5-10개 샘플에서 성능 향상 포화

#### 5.3 다층 추론 능력[1]

**Self-Verification**: 단계별 검증으로 오류 조기 감지
**Reflection**: 실패 원인 분석 및 전략 조정
**Backtracking**: 잘못된 경로에서 복구

***

### 6. 주요 한계 및 도전

#### 6.1 기술적 한계[3][2][1]

1. **토큰 효율성 부족**
   - 일부 단순 문제에서 과도한 사고 (overthinking)
   - 추론 길이 최적화 미흡

2. **구조화된 출력 제한**
   - JSON/XML 형식 생성 어려움
   - 복잡한 데이터 구조 처리 불가

3. **도구 사용 부재**
   - 계산기, 검색 엔진 통합 미약
   - 외부 환경 상호작용 제한적

4. **소프트웨어 엔지니어링**
   - 코드 최적화 및 레거시 코드 이해 부분적 개선만 달성
   - RL 평가 인프라 구축의 어려움

#### 6.2 데이터 및 계산 문제

1. **콜드-스타트 의존성**: 초기 고품질 데이터 필요
2. **보상 설계의 복잡성**: 스파스 vs 밀집 보상의 트레이드오프
3. **확장성**: 대규모 RL에 필요한 막대한 계산 자원

#### 6.3 정렬 및 안전성[1]

1. **다목표 정렬의 어려움**: 추론 성능과 일반 능력의 균형
2. **간섭 효과**: 추론 RL이 다른 능력에 미치는 부정적 영향

***

### 7. 모델 일반화 능력 향상 메커니즘

#### 7.1 Weak-to-Strong Generalization[1]

약한 교사 모델의 노이즈 있는 데모에서 강한 학생 모델이 효과적으로 학습하는 프레임워크:

- **앙상블 학습**: W2SG 강화로 노이스 저항성 향상
- **관점 분할**: 다양한 해결 방식 학습으로 일반화 개선

#### 7.2 Multi-Round Reasoning[1]

**다중 라운드 추론의 근사 가능성**:
- 선형 모델 가정 하에서 LTR vs CTL 분석
- 반복적 정제로 지수적 성능 개선

#### 7.3 Domain Adaptation 기법[1]

**Knowledge Editing**:
- 특정 사실의 선택적 업데이트
- 기존 지식 보존과 신규 정보 통합

**Retrieval-Augmented Generation (RAG)**:
- 외부 지식베이스 동적 접근
- 사실성 및 신선도 향상

***

### 8. 2020년 이후 최신 연구 비교 분석

#### 8.1 핵심 연구 진화 단계[5][6][7][1]

| 기간 | 주요 기법 | 특징 |
|------|---------|------|
| 2020-2021 | Prefix-Tuning, Prompt-Tuning | 파라미터 효율성 초점 |
| 2022 | RLHF, ChatGPT | 인간 선호도 기반 정렬 |
| 2023 | DPO, Domain Adaptation | 직접 선호도 최적화, RAG |
| 2024 | DeepSeek-R1, o1 | 대규모 RL 기반 추론 |
| 2025 | PLAN-TUNING, AR3PO | 효율적 추론, 채점 기반 RL |

#### 8.2 최신 혁신 모델[8][5][2]

**OpenAI o1/o3 (2024-2025)**:
- 장기 CoT 추론으로 수학, 코딩 벤치마크 혁신
- 테스트 타임 계산 스케일링 선도

**DeepSeek-R1 (2025)**:
- 오픈소스 모델로 o1 성능 달성
- 콜드-스타트 RL로 계산 효율성 증대
- 지식 증류로 소형 모델에 추론 능력 전이

**QwQ-32B-Preview (2024)**:
- 32B 규모의 수학/프로그래밍 강화

#### 8.3 DPO vs RLHF 비교[9][10]

**DPO의 장점**:
- 계산 효율: 보상 모델 불필요
- 안정성: 온라인 샘플링 없음
- 구현 단순성

**DPO의 한계**:[10]
- 분포 이동(distribution shift)에 민감
- 아웃-오브-디스트리뷰션 응답 활용 편향
- PPO 적절히 최적화 시 성능 격차

**개선 방안**:
- 반복 DPO (Iterative DPO): 이전 DPO를 새 기준 모델로 사용
- 대비 RL (Contrastive RL): 강한/약한 경로 비교

#### 8.4 효율성 개선 기법[11][12][13]

**토큰 효율성**:[12]
- 자기 훈련으로 30% 토큰 감소 (정확도 유지)
- Best-of-N 샘플링 + Few-shot 컨디셔닝

**적응 롤아웃 (AR3PO)**:[14]
- GRPO 대비 4.2배 롤아웃 비용 절감
- 응답 재사용으로 훈련 신호 활용

**잠재 추론 개선**:[11]
- 대조적 추론 피드백: 강한/약한 기준 비교
- 잔차 임베딩 정제: 경사도 통합의 안정화

#### 8.5 멀티모달 확장[1]

**모달 연결 기법**:
- **투영 기반**: LLaMA-Adapter (이미지 → 텍스트 임베딩)
- **쿼리 기반**: BLIP-2 (학습 가능 쿼리 토큰)
- **융합 기반**: Flamingo (교차 주의 레이어)

**최신 모델**:
- DeepSeek-VL2: 동적 타일링으로 고해상도 이미지
- Qwen2.5-VL: 재설계된 Vision Transformer
- InternVL: 60억 파라미터 비전 인코더

***

### 9. 향후 연구에 미치는 영향 및 고려사항

#### 9.1 패러다임 시프트[5][3][1]

**사전학습 → 포스트트레이닝 중심**:
- 데이터 규모 한계에 직면한 사전학습의 한계 인식
- 구조화된 피드백을 통한 동적 학습으로 전환
- 계산 자원의 추론 시간으로의 재배치

**SFT → 대규모 RL 전이**:
- 예제 모방 학습의 한계 극복
- 자기 진화적 학습 능력 발현

#### 9.2 핵심 연구 방향[15][5][1]

1. **적응형 RL 프레임워크**
   - 문제 난이도별 동적 보상 설계
   - 멀티 태스크 RL의 간섭 최소화

2. **공정성 기반 최적화**
   - 집단 간 편향 최소화
   - 다양한 인구통계학적 그룹에 대한 공정한 성능

3. **설명 가능성 강화**
   - 추론 과정의 투명성 증진
   - 단계별 검증 메커니즘의 견고성

4. **하이브리드 모델 개발**
   - 프로세스 기반 보상(추론 과정) + 결과 기반 보상(최종 답)
   - 교육적 중간 단계와 최종 성능의 균형

#### 9.3 실무 적용 시 고려사항

**데이터 획득 및 품질**:
- 고품질 보상 신호 생성의 비용
- 도메인 특화 데이터의 한계성

**계산 자원의 효율화**:
- 중소 조직의 대규모 RL 구현 어려움
- 지식 증류를 통한 경량 모델 개발 필수

**정렬과 안전성**:
- 다양한 문화적 가치관의 반영
- 멀티모달 정렬의 복잡성

#### 9.4 개방형 문제[1]

1. **스케일의 한계**: 어느 규모에서 대규모 RL이 비효율화?
2. **보상 설계의 과학화**: 최적 보상 함수 구성의 일반 원칙?
3. **전이 학습**: RL로 학습한 추론이 새로운 도메인으로 전이되는가?
4. **중장기 일관성**: 추론 능력 강화가 다른 능력을 손상시키지는 않는가?

***

### 10. 결론 및 종합 평가

본 논문은 **LLM 포스트트레이닝의 첫 포괄적 프레임워크**를 제시하며, 특히 다음 세 가지 측면에서 혁신적입니다:

1. **이론적 통합**: 5가지 패러다임을 일관된 원칙 하에 분류
2. **실증적 검증**: DeepSeek-R1을 통한 대규모 RL의 효과 입증
3. **미래 지향적**: LRM의 부상과 효율성, 정렬의 동시 달성 제시

**한계**에도 불구하고:
- 도구 사용 부재
- 토큰 효율성 미흡
- 구조화된 출력 제한

**이러한 도전들은 향후 5년 간 LLM 연구의 핵심 과제**가 될 것이며, **멀티모달 대규모 RL**, **적응형 보상 설계**, **효율성-성능 트레이드오프 최적화**가 주요 연구 방향이 될 것으로 예상됩니다.

***

### 참고문헌 (논문 ID 인용)

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/bd647476-98e8-44a4-8fb9-1b5c3c9890a2/2503.06072v2.pdf)
[2](https://arxiv.org/pdf/2501.12948.pdf)
[3](https://www.nature.com/articles/s41586-025-09422-z)
[4](https://arxiv.org/abs/2507.07495)
[5](https://arxiv.org/abs/2502.21321)
[6](https://arxiv.org/pdf/2502.21321.pdf)
[7](https://aclanthology.org/2025.acl-long.140.pdf)
[8](https://arxiv.org/html/2501.12948v1)
[9](https://arxiv.org/pdf/2503.12854.pdf)
[10](https://arxiv.org/html/2404.10719v1)
[11](https://arxiv.org/abs/2506.08552)
[12](https://arxiv.org/abs/2502.20122)
[13](https://arxiv.org/html/2502.04463v4)
[14](https://arxiv.org/abs/2509.25808)
[15](https://dev.datascienceassn.org/sites/default/files/pdf_files/LLM%20Post-Training%20-%20A%20Deep%20Dive%20into%20Reasoning%20Large%20Language%20Models.pdf)
[16](http://medrxiv.org/lookup/doi/10.1101/2025.06.06.25329104)
[17](https://arxiv.org/abs/2507.00726)
[18](https://www.semanticscholar.org/paper/6c67d0c638be9be7f2d251091f103a9c3ac1e7d1)
[19](https://arxiv.org/abs/2503.06072)
[20](https://arxiv.org/abs/2505.02686)
[21](https://arxiv.org/html/2503.22732v1)
[22](http://arxiv.org/pdf/2410.02108.pdf)
[23](http://arxiv.org/pdf/2503.06692.pdf)
[24](https://aclanthology.org/2023.findings-acl.67.pdf)
[25](http://arxiv.org/pdf/2503.03128.pdf)
[26](https://cameronrwolfe.substack.com/p/direct-preference-optimization)
[27](https://www.interconnects.ai/p/the-state-of-post-training-2025)
[28](https://turingpost.co.kr/p/topic-45-dpo-rrhf-rlaif-rlhf-ai)
[29](https://www.siliconflow.com/articles/en/the-best-deepseek-ai-models-in-2025)
[30](https://openreview.net/forum?id=OuMNJoKJBQ)
[31](https://huggingface.co/deepseek-ai/DeepSeek-R1)
[32](https://magazine.sebastianraschka.com/p/llm-research-papers-2025-part2)
[33](https://kimjy99.github.io/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0/diffusion-dpo/)
[34](https://kimjy99.github.io/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0/deepseek-r1/)
[35](https://arxiv.org/pdf/2509.09055.pdf)
[36](https://arxiv.org/pdf/2509.04419.pdf)
[37](https://arxiv.org/html/2506.19780v2)
[38](https://arxiv.org/abs/2505.15694)
[39](https://arxiv.org/html/2502.02523v2)
[40](https://arxiv.org/html/2511.03939v1)
[41](https://arxiv.org/abs/2502.17947)
[42](https://arxiv.org/html/2507.00726v2)
[43](https://arxiv.org/html/2508.07137v2)
[44](https://pubmed.ncbi.nlm.nih.gov/40962978/)
