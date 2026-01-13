
# A Survey on Progress in LLM Alignment from the Perspective of Reward Design

## 1. 논문의 핵심 주장과 주요 기여

"A Survey on Progress in LLM Alignment from the Perspective of Reward Design"은 대규모 언어 모델(LLM)의 정렬(Alignment)이 본질적으로 **보상 설계(Reward Design)의 지속적 정제 과정**이라는 핵심 주장을 제시합니다.[1]

### 주요 기여도

1. **보상 모델링의 체계적 조직화**: 수학적 공식화, 구축 실제, 최적화 패러다임과의 상호작용이라는 세 가지 핵심 차원을 기반으로 보상 메커니즘의 분류 체계를 개발

2. **하이브리드 보상 설계의 포괄적 분석**: 다양한 신호 소스의 통합 전략과 이들 간의 상충 관계 해결 방법을 체계화

3. **패러다임 시프트의 종합**: RL 기반에서 RL-free 방법으로의 전환, 단일 작업에서 다중 목표·작업·모달 설정으로의 확대라는 진화의 궤적을 분석

## 2. 해결하고자 하는 문제와 제안하는 방법

### 2.1 LLM Alignment의 근본적 도전 과제

LLM이 직면한 주요 문제는 다음과 같습니다:[1]

- **인간 피드백의 주관성**: 주석자 간 불일치로 인한 노이즈 및 최적화 불안정성
- **고품질 피드백의 부족**: 의료, 법률 등 전문 도메인에서의 높은 비용
- **Mode Collapse**: 좁은 보상 신호에 대한 과도 최적화로 인한 반복적 출력
- **Reward Hacking**: 보상 함수의 결함을 악용한 의도하지 않은 행동
- **Reward Misspecification**: 설계된 보상과 실제 인간 선호도 간의 괴리

### 2.2 수학적 공식화

#### Pointwise Reward Modeling (MSE 손실)
$$L_{MSE}(\phi) = \frac{1}{N} \sum_{i=1}^{N} [r_\phi(x^{(i)}, y^{(i)}) - s^{(i)}]^2$$

여기서 $r_\phi(x, y)$는 보상 모델, $s^{(i)}$는 인간 평가 점수입니다.

#### Pairwise Preference Modeling (Bradley-Terry 모델)
$$p^* (y_w \succ y_\ell | x) = \sigma(r_\phi(x, y_w) - r_\phi(x, y_\ell))$$

손실 함수:
$$L_{pairwise}(r_\phi, D) = -E_{(x,y_w,y_\ell) \sim D} [\log \sigma(r_\phi(x, y_w) - r_\phi(x, y_\ell))]$$

#### Listwise Preference Modeling (Plackett-Luce 모델)
$$p^*(y_1 \succ \cdots \succ y_K | x) = \prod_{j=1}^{K} \frac{\exp(r_\phi(x, y_j))}{\sum_{t=j}^{K} \exp(r_\phi(x, y_t))}$$

#### Token-level Reward Modeling
집계된 손실:
$$L_{token-agg}(q_\phi, D) = -E_{(x,y_w,y_\ell) \sim D} [\log \sigma(R_\phi(x, y_w) - R_\phi(x, y_\ell))]$$

단계별 손실:
$$L_{token-step}(q_\phi, D) = -E_{(x,y_w,y_\ell) \sim D} [\sum_{t=1}^{\max(T_w, T_\ell)} \log \sigma(q_\phi(x, y_{w,t}) - q_\phi(x, y_{\ell,t}))]$$

### 2.3 RLHF의 최적화 목표
$$\max_{\pi_\theta} E_{x \sim D, y \sim \pi_\theta(y|x)} [r_\phi(x, y)] - \beta D_{KL}[\pi_\theta(y|x) || \pi_{ref}(y|x)]$$

- $\pi_\theta$: 언어 모델의 정책
- $r_\phi(x, y)$: 보상 모델
- $\beta$: KL 발산 페널티 가중치

### 2.4 Direct Preference Optimization (DPO)

DPO는 명시적 보상 모델 없이 선호도를 직접 최적화하는 획기적 방법입니다.[1]

이상적 정책:
$$\pi_r(y|x) = \frac{1}{Z(x)} \pi_{ref}(y|x) \exp(\frac{1}{\beta} r(x, y))$$

암묵적 보상:

```math
r^*(x, y) = \beta \log \frac{\pi_r^*(y|x)}{\pi_{ref}(y|x)} + \beta \log Z(x)
```

DPO 손실 함수:
$$L_{DPO}(\pi_\theta; \pi_{ref}) = -E_{(x,y_w,y_\ell) \sim D} [\log \sigma(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_\ell|x)}{\pi_{ref}(y_\ell|x)})]$$

## 3. 모델 구조와 분류 체계

논문은 보상 모델을 다음 차원에서 분류합니다:[1]

### 수치형 vs 비수치형 RM
- **수치형**: 스칼라 또는 벡터 점수 출력
- **비수치형**: 자연언어 피드백, 정성적 평가

### 규칙 기반 vs 데이터 기반 vs 하이브리드 RM

**규칙 기반 RM**[1]
- 전문가가 정의한 명시적 규칙
- 높은 해석가능성과 통제가능성
- 확장성 제한 (오픈엔디드 작업에서 부족)
- 예: 독성 필터, 문법 규칙 검사

**데이터 기반 RM**[1]
- 레이블된 데이터로부터 학습
- 초기: Classification/Regression 기반
- 발전: Preference learning (Bradley-Terry, Plackett-Luce)
- 최신: Inverse RL를 통한 잠재 보상 함수 추론

**하이브리드 RM**[1]
- 규칙 기반과 데이터 기반의 장점 결합
- Multi-source fusion (예: 규칙 + 데이터 기반)
- 적응적 가중치 조정
- 문맥-인식 보상 선택

### Explicit vs Implicit RM

**Explicit RM**: 별도의 신경망으로 학습되는 독립적 모델 (RLHF에서 사용)
**Implicit RM**: 최적화 목표에 내재된 보상 신호 (DPO에서 사용)

## 4. 성능 향상 메커니즘과 한계

### 4.1 성능 향상 메커니즘

**Token-level RM의 개선**
- 더 세밀한 신용 할당으로 인한 안정성 향상
- TLCR, TDPO 등의 방법으로 12-18% 추론 성능 개선[2]

**Multi-value RM의 개선**
- 앙상블 방식을 통한 과적합 감소
- 다각적 선호도 신호 포착
- 도메인별 특성 반영

**Hybrid RM의 개선**
- 규칙 기반 신호로 초기 안전성 강화
- 데이터 기반 신호로 유연성 제공
- 적응적 가중치로 컨텍스트-인식 최적화

**DPO와 변형들의 효율성**
- 계산 비용 10-100배 감소[3]
- 더 안정적인 학습 (gradient variance 감소)
- 샘플 효율성 개선[4]

### 4.2 식별된 한계

**DPO의 이론적 문제** (2025년 최신 연구)[5]
$$\text{DPO implicit reward가 실제 인간 선호도를 misspecify할 수 있음}$$

특정 조건에서 DPO는 두 번째 최고 보상 응답을 선호하는 성능 저하 현상 발생

**일반화 성능의 제약**
- 도메인 간 분포 이동에 취약
- Out-of-distribution 데이터에 대한 성능 저하 (20-30%)[6]
- 새로운 작업에 대한 전이 학습 어려움

**Data Annotation의 문제**
- 고품질 피드백 수집 비용 증가
- 주석자 편향 (Annotator bias) 문제
- 주관적 선호도의 다양성과 불일치[1]

**Optimization 안정성 문제**
- Mode collapse 현상
- Reward hacking 위험
- RL 기반 방법의 convergence 불안정성
- Hyperparameter sensitivity[1]

## 5. 모델의 일반화 성능 향상 가능성

### 5.1 현재의 도전 과제

**분포 이동 (Distribution Shift)**
- 훈련 데이터와 실제 배포 간의 괴리
- 문화 간 가치 차이[1]
- 시간에 따른 선호도 변화

**Cross-domain 견고성**
- 의료, 법률 등 고위험 도메인의 특화된 요구사항
- 도메인 간 규범의 충돌
- 제한된 레이블 데이터에서의 학습

### 5.2 향상 가능성을 높이는 최신 방법들

#### 1) Meta-learning 접근 (OOD Preference Learning)

**Bilevel Optimization**으로 다양한 분포에서 일반화 가능한 보상 모델 학습:[6]
- 20개 도메인에서 검증
- 기존 방법 대비 15-25% 성능 향상
- 수렴 속도 증명

#### 2) Foundation Reward Model (GRAM)

사전학습된 생성형 보상 모델 기반 접근:[7]
- 레이블 스무딩으로 정규화된 pairwise ranking loss 최적화
- Cross-domain 일반화 성능 우수
- 재훈련 없이 새로운 도메인 적응 가능
- **성능**: LLaMA-3.1-8B 기준 11% 향상

#### 3) Principle-following Reward Models (RewardAnything)

자연언어 원칙에 따라 동적으로 적응하는 RM:[8]
- 고정된 preference dataset에서 벗어남
- 새로운 원칙에 대한 재훈련 불필요
- RABench 벤치마크로 평가
- Nuanced safety 및 helpfulness 달성

#### 4) Domain-Invariant RM

Source domain의 knowledge를 target domain으로 전이:[9]
- Shared representations를 통한 knowledge transfer
- Preference distribution shift 감소
- 제한된 레이블 데이터에서의 효과적 학습

#### 5) Distributionally Robust Optimization

선호도 분포 변화에 강인한 DPO 변형들:[10]
- **WDPO (Wasserstein DPO)**: Wasserstein distance 기반
- **KLDPO**: KL divergence 기반
- 다항식 샘플 복잡도 달성
- Out-of-distribution accuracy 20-30% 개선

#### 6) Multi-Objective Adaptation

**Preference Orchestrator (Pro)** - 문맥별 적응형 가중치:[11]
- 프롬프트별로 적절한 선호도 가중치 자동 추론
- 이론적으로 고정 가중치 방식보다 superior 성능 보증
- Multi-objective alignment gap 감소

**RiC (Rewards-in-Context)**:[12]
- In-context learning을 통한 동적 선호도 조정
- MORLHF 대비 50% 이상 계산량 감소
- Pareto-optimal 성능 달성

#### 7) Token-level 고도화 (최신 2024-2025)

**TGDPO (Token-level reward Guidance for DPO)**:[13]
- 토큰 수준의 최적 정책 및 보상 유도
- Bradley-Terry 모델과의 통합
- Partition function 제거로 계산 효율성 증대

**T-REG (Token-level Reward as weakly supervised)**:[14]
- 시퀀스 수준 최적화 유지 + 토큰 수준 약한 감시
- 기존 DPO/SimPO 대비 AlpacaEval 2와 Arena-Hard에서 일관적 개선

### 5.3 성공 사례와 벤치마크 성과

| 방법 | 벤치마크 | 성능 | 특징 |
|------|---------|------|------|
| α-DPO | 다양한 모델/데이터셋 | +5-8% | 적응적 선호 분포 |
| β-DPO | NeurIPS 2024 | 우수 | 동적 β 및 데이터 필터링 |
| RewardAnything | RABench | SOTA | 원칙 기반 일반화 |
| GRAM | Cross-domain | +11% (LLaMA 3.1) | 사전학습 기반 |
| DG-PRM | PRMBENCH | SOTA | Process reward 동적 구성 |
| ArmoRM | RewardBench | 80+ | Multi-objective 해석가능성 |

## 6. 논문이 제시하는 영향과 향후 연구 방향

### 6.1 주요 영향

**패러다임 시프트의 확인**:
1. **RL 기반 → RL-free 방법으로의 전환**
   - RLHF의 계산 복잡성 문제 해결
   - DPO와 변형들의 실용적 우위 입증[3]
   - Online vs offline 성능 격차 감소

2. **단일-목표 → 다중-목표/다중-작업/다중-모달로의 확대**
   - 실제 응용의 복잡성 반영
   - 다양한 휴먼 밸류의 동시 고려
   - Cross-cultural alignment 필요성 인식

3. **정적 보상 설계 → 동적 적응 시스템으로의 진화**
   - 시간에 따른 선호도 변화 추적
   - 사용자별 맞춤형 정렬
   - 진화하는 사회 규범 반영

### 6.2 특히 강조되는 관찰

**Reward Design의 중심적 역할**:
논문은 보상 설계가 단순한 기술적 구성요소가 아니라 LLM alignment의 **핵심 메커니즘**임을 강조합니다. 이는 다음을 의미합니다:[1]

- 보상 설계의 선택이 하위 최적화 알고리즘의 선택보다 더 중요할 수 있음
- Alignment 연구의 발전이 보상 전략의 정제와 직결됨
- 향후 breakthrough는 보상 메커니즘의 혁신에서 올 가능성 높음

### 6.3 향후 연구 시 고려할 점

#### 1) Value Co-creation 패러다임 (6장 Discussion)[1]

정적 annotation에서 벗어나 다음을 추구:
- 집단 인간 상호작용 학습 (대화, 피드백, 커뮤니티 입력)
- 시간적으로 적응하는 보상 모델
- 다양한 문화 규범의 명시적 통합

#### 2) Meta-learning Framework 구축

다양한 도메인/작업/선호도에 빠르게 적응:
- Few-shot alignment learning 능력
- Continual learning 지원
- 새로운 규범에 대한 유연한 대응

#### 3) Out-of-Distribution Robustness 강화

분포 이동에 강인한 보상 시스템:
- Domain-invariant representation learning[9]
- Distribution shift 감지 및 대응 메커니즘
- Uncertainty quantification

#### 4) Principle-following Capability 개발[8]

- 고정된 데이터셋에서 벗어남
- 자연언어 원칙에 대한 직접 적응
- 사용자 정의 alignment 비용 감소

#### 5) Multi-modal Integration

다양한 입력 형식에 대한 통합 보상 모델:
- 텍스트, 이미지, 오디오의 일관된 평가
- Cross-modal consistency 보장
- Vision-Language Models (VLMs)에 대한 지원

#### 6) Governance와 Fair Alignment

사회적 책임 있는 정렬:
- 이해관계자 간 선호도 충돌 해결
- 집단 의사결정 원칙 통합
- Pluralistic alignment (다원적 정렬)

## 7. 2020년 이후 최신 연구와의 비교 분석

### 7.1 주요 연구 단계별 진화

**초기 단계 (2020-2022)**:
- RLHF 기반 InstructGPT (2022)
- Bradley-Terry 모델의 도입
- 단순 numerical reward modeling

**발전 단계 (2023)**:
- **DPO의 혁신적 도입** (Rafailov et al., 2023)
  - 명시적 RM 제거로 계산 효율성 10-100배 개선
  - RLHF와 동등한 성능 달성
  
- Token-level reward modeling 개척[2]

**고도화 단계 (2024-2025)**:
- DPO 변형들의 폭증
  - α-DPO, β-DPO, T-DPO, Step-DPO 등[2]
- Foundation reward models 개발[7]
- Principle-following RMs 등장[8]
- Distributionally robust 방법들[10]

### 7.2 Generalization Performance 개선의 최신 성과

**메타러닝 기반** (2024):
- OOD preference learning으로 20개 도메인 검증
- 기존 방법 대비 15-25% 향상[6]

**Foundation Model 기반** (2025):
- 사전학습된 generative RM (GRAM)
- Cross-domain 성능 11% 향상[7]
- 재훈련 없는 적응

**Principle-following** (2024-2025):
- RewardAnything: 자연언어 원칙 기반 적응
- 재훈련 불필요한 새로운 도메인 진입
- RABench로 일반화 평가[8]

**Domain-Invariant** (2024):
- Source-target 도메인 간 knowledge transfer
- Distribution shift 감소
- 제한된 데이터에서의 효과성[9]

### 7.3 계산 효율성 비교

| 방법 | 명시적 RM | RL 필요 | 상대 계산비용 |
|------|---------|---------|---------|
| RLHF (PPO) | ✓ | ✓ | 1.0x |
| DPO | ✗ | ✗ | 0.1-0.2x |
| RLAIF | ✓ | ✓ | 0.6-0.8x |
| RiC | ✓ | ✗ | 0.4-0.5x |
| RewardAnything | ✓ | ✓ | 0.8x |

## 8. 결론 및 종합 평가

### 8.1 논문의 중요성

이 논문은 LLM alignment 연구의 전체 궤적을 **보상 설계의 진화**라는 렌즈로 재해석합니다. 이는 다음을 의미합니다:[1]

1. **Conceptual Clarity**: 산발적 기법들을 체계적 분류 체계로 통일
2. **Methodological Insight**: 패러다임 시프트의 원동력을 명확히 함
3. **Predictive Value**: 미래 연구의 방향을 제시

### 8.2 향후 alignment 연구의 전망

**단기 (1-2년)**:
- DPO 변형들의 수렴과 표준화
- Multi-objective alignment의 실용화
- Domain-specific RM의 체계화

**중기 (3-5년)**:
- Principle-following RMs의 광범위 채택
- Foundation reward models의 확립
- Multi-modal alignment의 성숙

**장기 (5년 이상)**:
- Value co-creation framework의 구현
- Continual learning 기반 동적 정렬
- Cross-cultural 및 pluralistic alignment의 실현

### 8.3 최종 평가

이 논문은 LLM alignment 분야의 **교과서적 문헌**으로서의 위상을 확립합니다. 특히 다음 측면에서 중요합니다:

- **이론적 엄밀성**: 수학적 공식화와 분류 체계의 정교함
- **실용적 유용성**: 다양한 alignment 기법의 선택과 설계에 직접 적용 가능
- **미래 지향성**: 새로운 연구 방향을 명확히 제시

보상 설계가 LLM alignment의 핵심이라는 논문의 주장은, 앞으로의 alignment 연구가 단순히 최적화 알고리즘의 개선에서 벗어나 **인간의 다양한 가치를 보상 신호로 어떻게 효과적으로 인코딩할 것인가**에 집중해야 함을 명확히 합니다.

***

## 참고문헌 표기

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/11cdab63-4e0f-4c27-ab2e-0bb433cb0b75/2505.02666v2.pdf)
[2](https://arxiv.org/abs/2502.01992)
[3](https://arxiv.org/abs/2410.03145)
[4](https://arxiv.org/abs/2508.13382)
[5](https://arxiv.org/abs/2509.22611)
[6](https://www.semanticscholar.org/paper/f64ab88f0326d0473a820f2284509cdae619d542)
[7](https://www.semanticscholar.org/paper/0840e1801d92d06cb2ba4d60ecf4ec3fa460b316)
[8](https://www.journaljemt.com/index.php/JEMT/article/view/1351)
[9](https://ashpublications.org/blood/article/146/Supplement%201/1631/554197/Evolving-use-of-blast-percentage-thresholds-and)
[10](https://www.semanticscholar.org/paper/4e946c4e20f06092b4d3c76421c69adf1a874d04)
[11](https://link.springer.com/10.1007/s00405-025-09524-4)
[12](http://arxiv.org/pdf/2407.04185.pdf)
[13](https://aclanthology.org/2023.emnlp-main.844.pdf)
[14](https://arxiv.org/pdf/2403.06754.pdf)
[15](http://arxiv.org/pdf/2405.12739.pdf)
[16](https://arxiv.org/pdf/2402.18571.pdf)
[17](https://arxiv.org/html/2503.10093v1)
[18](https://arxiv.org/html/2502.04357v1)
[19](http://arxiv.org/pdf/2410.14660.pdf)
[20](https://www.emergentmind.com/topics/llm-based-reward-generation)
[21](https://arxiv.org/html/2410.15595v3)
[22](https://cbtw.tech/insights/rlhf-alternatives-post-training-optimization)
[23](https://arxiv.org/html/2505.02666v2)
[24](https://cameronrwolfe.substack.com/p/direct-preference-optimization)
[25](https://magazine.sebastianraschka.com/p/llm-training-rlhf-and-its-alternatives)
[26](https://aclanthology.org/2025.findings-emnlp.1237.pdf)
[27](https://www.superannotate.com/blog/direct-preference-optimization-dpo)
[28](https://towardsdatascience.com/llm-alignment-reward-based-vs-reward-free-methods-ef0c0f6e8d88/)
[29](https://dl.acm.org/doi/10.24963/ijcai.2025/1141)
[30](https://arxiv.org/html/2410.15595v2)
[31](https://intuitionlabs.ai/articles/reinforcement-learning-vs-rlhf)
[32](https://openreview.net/forum?id=ruvzyT9HqJ&noteId=nzxFSonN3W)
[33](https://bohrium.dp.tech/paper/arxiv/2410.15595)
[34](http://arxiv.org/pdf/2407.16216.pdf)
[35](https://arxiv.org/pdf/2505.02666.pdf)
[36](https://arxiv.org/html/2510.24636v1)
[37](https://arxiv.org/pdf/2503.11701.pdf)
[38](https://arxiv.org/html/2312.15997v1)
[39](https://arxiv.org/html/2505.10597v1)
[40](https://arxiv.org/html/2504.15843v3)
[41](https://arxiv.org/html/2401.11458v1)
[42](https://arxiv.org/html/2504.13134v2)
[43](https://arxiv.org/abs/2503.11701)
[44](https://arxiv.org/html/2402.14740v1)
[45](https://arxiv.org/html/2512.09212v1)
[46](https://arxiv.org/html/2511.10985v1)
[47](https://arxiv.org/html/2309.00267v3)
[48](http://arxiv.org/pdf/2404.14723.pdf)
[49](https://arxiv.org/pdf/2407.08639.pdf)
[50](https://arxiv.org/pdf/2502.07599.pdf)
[51](https://arxiv.org/pdf/2503.02832.pdf)
[52](https://arxiv.org/pdf/2502.01930.pdf)
[53](https://arxiv.org/pdf/2501.01544.pdf)
[54](https://arxiv.org/pdf/2501.06645.pdf)
[55](https://arxiv.org/html/2506.14574v1)
[56](https://openreview.net/forum?id=t5mpbfpZuF)
[57](https://arxiv.org/html/2402.10207v3)
[58](https://aclanthology.org/2025.acl-long.1353.pdf)
[59](https://www.emergentmind.com/papers/2402.14760)
[60](https://arxiv.org/abs/2406.12845)
[61](https://proceedings.iclr.cc/paper_files/paper/2025/file/7fb9f39075a5202472676a7531568212-Paper-Conference.pdf)
[62](https://aclanthology.org/2024.findings-acl.511.pdf)
[63](https://aclanthology.org/2024.findings-emnlp.620.pdf)
[64](https://arxiv.org/html/2505.19653v1)
[65](https://latitude-blog.ghost.io/blog/how-unsupervised-domain-adaptation-works-with-llms/)
[66](https://aclanthology.org/2024.findings-acl.630.pdf)
[67](https://liner.com/ko/review/%CE%B2dpo-direct-preference-optimization-with-dynamic-%CE%B2)
[68](https://pmc.ncbi.nlm.nih.gov/articles/PMC12286580/)
[69](https://proceedings.mlr.press/v235/yang24q.html)
[70](https://arxiv.org/html/2506.03637v1)
[71](https://arxiv.org/html/2511.10656v1)
[72](https://arxiv.org/html/2508.18312v1)
[73](https://arxiv.org/pdf/2506.14175.pdf)
[74](https://arxiv.org/html/2510.20413v1)
[75](https://arxiv.org/pdf/2506.03637.pdf)
[76](https://arxiv.org/abs/2512.10601)
[77](https://arxiv.org/html/2501.00911v1)
[78](https://arxiv.org/html/2510.01167v1)
[79](https://arxiv.org/html/2410.10148v4)
[80](https://arxiv.org/html/2510.27556v1)
[81](https://arxiv.org/html/2505.10892v1)
[82](https://aclanthology.org/2025.acl-long.212.pdf)
[83](https://www.emergentmind.com/topics/general-reward-model-grm)
[84](https://www.nature.com/articles/s41524-025-01564-y)
[85](https://www.emergentmind.com/topics/dpo)
[86](http://mlg.postech.ac.kr/~jtkim/papers/icml_2025_b.pdf)
[87](https://openaccess.thecvf.com/content/CVPR2024/papers/Hwang_Promptable_Behaviors_Personalizing_Multi-Objective_Rewards_from_Human_Preferences_CVPR_2024_paper.pdf)
[88](https://www.ijcai.org/proceedings/2025/1198.pdf)
