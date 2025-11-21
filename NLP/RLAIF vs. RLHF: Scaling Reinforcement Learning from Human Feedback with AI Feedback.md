
# RLAIF vs. RLHF: Scaling Reinforcement Learning from Human Feedback with AI Feedback

## 1. 핵심 주장과 주요 기여

**"RLAIF vs. RLHF: Scaling Reinforcement Learning from Human Feedback with AI Feedback"** 논문의 핵심 주장은 **인간 피드백 대신 AI 피드백을 사용하여 대규모 언어모델 정렬을 달성할 수 있다**는 것입니다. 이 연구는 네 가지 주요 기여를 제시합니다:[1]

1. **RLAIF의 실질적 동등성**: RLAIF는 요약(summarization), 도움이 되는 대화 생성(helpful dialogue), 무해한 대화 생성(harmless dialogue) 등 세 가지 과제에서 RLHF와 비교할 수 있는 성능을 달성합니다.

2. **자체 개선 가능성**: AI 레이블러가 정책과 동일한 크기이거나 심지어 동일한 체크포인트일 때도 RLAIF가 감독 학습(SFT) 기준선을 능가할 수 있음을 입증합니다. 이는 LLM의 자체 개선 가능성을 시사합니다.

3. **Direct-RLAIF(d-RLAIF) 소개**: 보상 모델 학습을 우회하고 RL 중에 오프더셀프 LLM에서 직접 보상을 얻는 기법을 제안합니다.

4. **AI 피드백 정렬 최적화**: 체인-오브-싱크(Chain-of-Thought) 추론, 상세한 프리앰블, 몇 샷 프롬팅 등 AI 생성 선호도와 인간 선호도의 정렬을 최대화하는 기법을 연구합니다.[1]

***

## 2. 해결하고자 하는 문제와 동기

### 문제 정의

RLHF의 근본적인 제약은 **고품질 인간 선호도 레이블 수집의 비용과 시간 소비**입니다. 현대의 LLM들이 인간 판단과 높은 일치도를 보이는 만큼, AI 생성 피드백이 인간 피드백의 실질적 대체제가 될 수 있는지 검증하는 것이 핵심 문제입니다.[1]

### 기술적 도전 과제

- 보상 모델이 정책 학습 중에 "낡아지는"(stale) 문제로 인한 성능 저하
- AI 피드백이 인간 선호도와 충분히 정렬되지 않을 가능성
- 위치 편향(position bias)으로 인한 AI 라벨러의 편향성

***

## 3. 제안 방법(수식 포함)

### 3.1 기본 RLHF 파이프라인

논문의 Appendix A에서 RLHF의 기본 구조를 다음과 같이 정의합니다:[1]

**Step 1: 보상 모델 학습**

$$L_r(\phi) = -\mathbb{E}_{(x,y_w,y_l) \sim D} \left[ \log \sigma(r_\phi(x, y_w) - r_\phi(x, y_l)) \right]$$

여기서:
- $r_\phi(x, y)$: 입력 $x$와 응답 $y$에 대한 보상 모델
- $\sigma$: 시그모이드 함수
- $y_w, y_l$: 선호도 높음/낮음 응답

**Step 2: 정책 최적화**

$$J(\theta) = \mathbb{E}_{y \sim \pi_\theta(\cdot|x)} \left[ (1-\beta)r_\phi(y|x) - \beta D_{KL}\left(\pi^{RL}_\theta(y|x) \parallel \pi^{SFT}(y|x)\right) \right]$$

여기서:
- $\pi_\theta$: 정책 모델
- $\beta$: KL 발산 가중치(기본값 0.05)
- $D_{KL}$: Kullback-Leibler 발산

### 3.2 REINFORCE 알고리즘 적용

언어 모델 정책 최적화를 위해 REINFORCE를 사용합니다:[1]

$$L_{PG}(\theta) = -\sum_t \log \pi_\theta(A_t|X_t) \left[ Z_t - V_\psi^\pi(X_t) \right]$$

여기서:
- $Z_t = R_T$: 누적 보상 (터미널 보상만 0이 아님)
- $V_\psi^\pi(x)$: 기준 가치 함수

가치 함수 손실:
$$L_V(\psi) = \sum_t (Z_t - V_\psi^\pi(X_t))^2$$

### 3.3 AI 피드백 생성

**Step 1: 선호도 추출**

오프더셀프 LLM에 구조화된 프롬프트를 제시하고 로그 확률을 이용해 선호도 분포를 계산합니다:[1]

$$P^{AI}_{i,j} = \text{softmax}(\log P(\text{"1"}), \log P(\text{"2"}))$$

**Step 2: 위치 편향 완화**

두 번의 추론을 수행하여 후보 순서를 반전시킨 후 평균화합니다:[1]

$$P^{final} = \frac{1}{2}[P^{order1} + P^{order2}]$$

**Step 3: Chain-of-Thought(CoT) 추론**

두 단계 추론 절차를 통해 추론 과정을 유도합니다:[1]

1. 설명 요청: "각 요약의 일관성, 정확성, 커버리지, 전체 품질을 고려하고 어느 것이 더 나은지 설명하세요."
2. 선호도 추출: 생성된 설명을 원래 프롬프트에 결합하여 최종 선호도 분포 추출

**Step 4: 보상 모델 학습**

AI 생성 소프트 레이블에 대해 크로스 엔트로피 손실로 훈련합니다:[1]

$$L_{RM}^{AI} = -\mathbb{E}_{(x,y_1,y_2) \sim D_{AI}} \left[ P^{AI}_{i,1} \log P_{RM}(1) + P^{AI}_{i,2} \log P_{RM}(2) \right]$$

### 3.4 Direct-RLAIF (d-RLAIF)

d-RLAIF는 보상 모델을 우회하고 직접 LLM 피드백을 사용합니다:[1]

**점수 계산:**

```math
s(y|x) = \frac{\sum_{i=1}^{10} i \cdot P(i|y,x)}{\text{max score}}
```

정규화를 위해 점수를 $[-1, 1]$ 범위로 조정합니다.

***

## 4. 모델 구조 및 아키텍처

### 4.1 전체 파이프라인 구조

**RLAIF 파이프라인:**[1]

1. **오프더셀프 LLM** → AI 선호도 레이블 생성
2. **보상 모델 학습** → AI 레이블로부터 RM 훈련
3. **정책 최적화** → REINFORCE로 정책 업데이트

**d-RLAIF 파이프라인:**[1]

1. **오프더셀프 LLM** → 직접 1-10 점수 제공
2. **정책 최적화** → 직접 점수를 보상으로 사용

### 4.2 구현 세부사항

**모델 선택:**[1]
- SFT 모델: PaLM 2 Extra-Small (XS)
- AI 레이블러: PaLM 2 Large (L) (기본) / PaLM 2 XS (동일 크기 실험)
- 보상 모델: PaLM 2 XS에서 초기화

**학습 설정:**[1]
- 배치 크기: 128
- 학습률: 10^-5
- 옵티마이저: Adafactor
- KL 가중치: β = 0.05
- RL 에폭: 8
- 온도: T = 0.9 (탐색 장려)

**프롬프팅 기법:**

| 프롬프트 유형 | 설명 | 정렬도 |
|---|---|---|
| Base 0-shot | 기본 질문만 | 76.1% |
| Base + CoT 0-shot | 설명 요청 추가 | 77.5% |
| Detailed 0-shot | 상세한 지침 제공 | 77.4% |
| Detailed + CoT 0-shot | 상세 + 설명 | 78.0% |

***

## 5. 성능 향상 결과

### 5.1 주요 성과

**승률 비교 (인간 평가 기준):**[1]

| 과제 | RLAIF vs SFT | RLHF vs SFT | RLAIF vs RLHF |
|---|---|---|---|
| 요약 | 71% | 73% | 50% |
| 도움이 되는 대화 | 63% | 64% | 52% |
| **무해한 대화** | **88%** | **76%** | **RLAIF 우수** |

**길이 보정 후 결과:**[1]

요약 과제에서 길이 편향을 제어한 후:
- RLAIF vs SFT: 59% (조정 전 71%)
- RLHF vs SFT: 61% (조정 전 73%)

이는 여전히 SFT보다 우월함을 입증합니다.

### 5.2 AI 레이블러 크기의 영향

| 모델 크기 | AI 정렬도 |
|---|---|
| PaLM 2 Large (L) | 78.0% |
| PaLM 2 Small (S) | 73.8% |
| PaLM 2 XS | 62.7% |

더 큰 모델이 인간 선호도와 더 잘 정렬됨을 보여줍니다.[1]

### 5.3 자체 개선 증거

**동일 크기 RLAIF:**
- 같은 크기 AI 레이블러 사용: 68% 승률 vs SFT
- 레이블러가 없을 때보다 3% 낮음 (71%)이지만 여전히 상당한 개선

**Strict Self-Improvement (d-RLAIF):**[1]
- 도움이 되는 대화 과제에서 초기 정책 = 보상 제공 LLM
- 66% 승률 달성 (SFT 대비)

***

## 6. 일반화 성능과 한계

### 6.1 일반화 능력 분석

**개발 데이터셋 범위:**[1]
- Reddit TL;DR (요약): 123k 포스트
- Anthropic Helpful/Harmless: 각 40k+ 훈련 예제
- 다운샘플링: 각 과제당 3-4k 예제

**Out-of-Distribution(OOD) 성능:**

논문은 명시적인 OOD 평가를 수행하지 않았으나, 최근 연구에 따르면:[2]
- 암시적 보상 모델(implicit reward model)은 프롬프트 시프트에서 평균 3% 정확도 하락
- 명시적 보상 모델은 OOD에서 훨씬 더 견고함

### 6.2 식별된 한계

**1. AI 피드백의 질 의존성**

AI 레이블러 정렬도가 직접 성능에 영향을 미칩니다:[1]

- Base 0-shot (76.1% 정렬도) vs Detailed + CoT 0-shot (78.0% 정렬도)
- 정렬도 개선 → 정책 성능 개선 (2% 승률 차이)

**2. 위치 편향 문제**

작은 모델에서 더 심각합니다:[1]

| 모델 크기 | 같은 위치 선호도(%) |
|---|---|
| PaLM 2 L | 18% |
| PaLM 2 S | 21% |
| PaLM 2 XS | 56% |

**3. 정책 수렴 문제**

- 보상 모델이 정책 변화에 따라 out-of-distribution 데이터에 노출됨
- d-RLAIF가 이를 부분적으로 해결하지만 매 단계마다 LLM 호출 필요

**4. 질적 차이**

RLAIF 생성 요약의 특성:[1]

- 긍정: RLHF보다 낮은 할루시네이션 비율
- 부정: 때로 RLHF보다 덜 자연스러운 표현 (run-on sentences 등)

**5. 인간-AI 피드백 결합의 한계**

RLHF + RLAIF 하이브리드 접근:
- RLHF (74%) vs RLHF + RLAIF (71%)
- 결합이 성능 개선을 가져오지 못함[1]

### 6.3 데이터셋 제한

Stanford Human Preferences 데이터셋에서는 의미 있는 개선을 관찰하지 못함 - 이는 과제-특정 한계를 시사합니다.[1]

***

## 7. 최신 연구 기반 영향 및 향후 고려사항

### 7.1 최신 연구 동향 (2024-2025)

**1. Direct Preference Optimization (DPO) vs RLAIF**[3][4]

최근 DPO 조사에 따르면:[3]
- DPO는 RLAIF보다 계산상 더 효율적 (RL 필요 없음)
- 그러나 DPO의 **암시적 보상 모델은 OOD 데이터에서 덜 견고**[2]
- RLAIF의 명시적 보상 모델은 분포 외 시나리오에서 평균 3% 더 높은 정확도[2]

**RLAIF의 우월성:**
- 더 예측 가능한 보상 신호
- 단계별 개선 추적 가능
- 복잡한 정렬 요구사항에 더 적합

**2. Generative Reward Models**[5]

최근 연구(2024)에서:
- RLAIF와 생성적 보상 모델의 하이브리드 접근
- 합성 선호도와 인간 선호도 라벨의 통일된 프레임워크
- RLAIF의 단일 출력 제약 극복

**3. 다중모드 LLM에서의 자체 개선**[6][7]

RLAIF 개념의 확장:
- RLAIF-V: 시각-텍스트 피드백 결합
- 멀티모달 자체 개선 프레임워크에서 RLAIF 원리 적용[7]
- 조건부 자체 개선(Conditional Self-Improvement) 수준 도입[7]

**4. 권장 시스템에의 적용**[8]

RecLAIF 프레임워크:
- 추천 시스템에 RLAIF 적용
- AI 판사가 관련성, 다양성, 설명 가능성 평가[8]

### 7.2 향후 연구 시 고려 사항

**1. 일반화 성능 강화**
- **권장사항**: 명시적 보상 모델 사용으로 OOD 견고성 보장[2]
- 강화학습 기반 접근법이 DPO보다 OOD 상황에서 더 적합[2]

**2. 비용-성능 트레이드오프**
- RLAIF 레이블링 비용: 약 $0.06/예제 (GPT-4 기준)
- 인간 레이블링: 약 $0.67/예제[1]
- **권장사항**: 10배 비용 절감으로 더 큰 데이터셋 활용 가능

**3. 보상 모델 견고성**
- **권장사항**: 
  - 반복 RLAIF로 정책 변화에 따라 RM 재훈련[1]
  - 또는 d-RLAIF로 RM 낡음 문제 회피
  - 최신 DPO+명시적 RM 하이브리드 고려

**4. AI 레이블러 선택**
- 더 큰 모델 사용이 AI-인간 정렬도 향상[1]
  - PaLM 2 L: 78% → PaLM 2 XS: 62.7%
- **권장사항**: 프로덕션에서는 최상의 성능 가능한 최대 크기 모델 활용

**5. 프롬프팅 전략**
- Chain-of-Thought가 일관되게 도움됨 (+1-2% 정렬도)[1]
- Few-shot은 혼합된 결과 (요약에서 해로움)[1]
- **권장사항**: 과제별 최적화된 프롬프팅 템플릿 개발

**6. 자체 개선의 한계**
- 동일 크기 자체 개선은 3% 성능 저하[1]
- **권장사항**: 
  - 보다 큰 초기 모델에서 시작하여 자체 개선
  - Curriculum learning: RLAIF → RLHF 시퀀셜 적용 검토

**7. 멀티모달 확장**
- RLAIF 원리가 시각-언어 모델로 확장[6][7]
- **권장사항**: 멀티모달 alignment에서 RLAIF 활용 고려

**8. 윤리적 고려사항**
- AI 피드백의 편향 전가 위험[1]
- **권장사항**: 
  - 고위험 도메인(의료, 법률)에서는 인간 전문가 감시 필수
  - 생성된 선호도의 정기적 감시
  - 편향 증폭 메커니즘 모니터링

**9. 하이브리드 접근의 재검토**
- 순수 RLAIF + RLHF 결합이 효과 없었음[1]
- **권장사항**: 
  - Curriculum learning: RLAIF (warm-up) → RLHF (refinement)
  - 비균등 비율: 많은 AI + 적은 인간 피드백
  - 순차적 적용 고려

***

## 종합 결론

"RLAIF vs. RLHF" 논문은 **AI 피드백이 LLM 정렬의 스케일링 솔루션**임을 입증했습니다. 10배 비용 절감과 인간 피드백과 동등한 성능은 산업적 의의가 큽니다.[1]

그러나 일반화 성능 측면에서는:
- **OOD 견고성**: 명시적 보상 모델(RLAIF)이 암시적 모델(DPO)보다 우수[2]
- **자체 개선**: 초기 성능 한계 존재[1]
- **편향 전가**: 고위험 도메인에서 주의 필수[1]

최신 연구는 RLAIF를 **DPO, 생성적 보상 모델, 멀티모달 학습과 결합**하는 방향으로 진화 중입니다. 향후 연구자들은 **OOD 견고성 강화**, **curriculum learning 기반 하이브리드 접근**, **멀티모달 확장**에 집중해야 합니다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/5953f63e-5515-4df8-8e15-7909d004be60/2309.00267v3.pdf)
[2](https://aclanthology.org/2024.findings-emnlp.940.pdf)
[3](https://arxiv.org/pdf/2503.11701.pdf)
[4](https://arxiv.org/html/2410.15595v3)
[5](https://arxiv.org/pdf/2410.12832.pdf)
[6](https://openreview.net/forum?id=NRrXHppaBg)
[7](https://arxiv.org/html/2510.02665v1)
[8](https://oars-workshop.github.io/papers/He2025.pdf)
[9](http://arxiv.org/pdf/2406.07295.pdf)
[10](http://arxiv.org/pdf/2405.07863.pdf)
[11](http://arxiv.org/pdf/2309.00267.pdf)
[12](https://arxiv.org/pdf/2402.02423.pdf)
[13](https://arxiv.org/pdf/2311.02379.pdf)
[14](http://arxiv.org/pdf/2410.15181v1.pdf)
[15](https://arxiv.org/abs/2406.15568v1)
[16](https://proceedings.mlr.press/v235/lee24t.html)
[17](https://arxiv.org/abs/2309.00267)
[18](https://usajournals.org/index.php/6/article/view/723)
[19](https://cameronrwolfe.substack.com/p/rlaif-reinforcement-learning-from)
[20](https://www.sapien.io/blog/rlaif-vs-rlhf-understanding-the-differences)
[21](https://www.linkedin.com/posts/carl-hendrick-33948155_how-does-artificial-intelligence-compare-activity-7381004583176941570-btjb)
[22](https://labelbox.com/blog/rlhf-vs-rlaif/)
[23](https://arxiv.org/abs/2504.10961)
[24](https://pietromingotti.com/inside-llms-rlhf-rlaif-the-evolution-of-model-alignment/)
[25](https://arxiv.org/pdf/2211.10892.pdf)
[26](http://arxiv.org/pdf/2403.06392.pdf)
[27](https://arxiv.org/pdf/2111.09190.pdf)
[28](http://arxiv.org/pdf/2408.07772.pdf)
[29](https://arxiv.org/pdf/2304.11327.pdf)
[30](https://arxiv.org/pdf/2106.03721.pdf)
[31](http://arxiv.org/pdf/2106.04496.pdf)
[32](https://arxiv.org/html/2408.14950v1)
[33](https://aws.amazon.com/blogs/machine-learning/fine-tune-large-language-models-with-reinforcement-learning-from-human-or-ai-feedback/)
[34](https://papers.miccai.org/miccai-2025/paper/1273_paper.pdf)
[35](https://proceedings.neurips.cc/paper_files/paper/2024/hash/285cf10c8c6153d66b8cd6a3ab0d69ce-Abstract-Conference.html)
