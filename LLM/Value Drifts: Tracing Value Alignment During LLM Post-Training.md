# Value Drifts: Tracing Value Alignment During LLM Post-Training

## 핵심 주장 및 주요 기여

이 논문은 **대규모 언어 모델(LLM)의 사후 학습(post-training) 단계에서 가치 정렬이 어떻게 발생하는지**를 추적한 최초의 체계적 분석을 제시합니다. 기존 연구가 학습 완료 후 모델을 평가하는 사후 평가(post-hoc evaluation)에 집중했다면, 이 논문은 **학습 동역학(training dynamics)**에 초점을 맞추어 값이 언제, 어떻게 습득되는지 규명합니다.[1]

논문의 핵심 발견은 다음과 같습니다:

1. **감독 미세조정(SFT)은 가치 정렬의 지배적 드라이버**: SFT 단계에서 모델이 급격하게 가치를 습득하며, 이후 단계에서는 크게 변하지 않습니다.[1]

2. **선호도 최적화의 제한된 효과**: 표준 데이터셋을 사용할 때 선호도 최적화는 SFT로 설정된 값을 재정렬하지 못합니다.[1]

3. **가치 간격의 중요성**: 합성 데이터를 통한 제어된 실험에서 "선택됨"과 "거부됨" 응답 사이에 명확한 가치 대비(value gap)가 있을 때만 선호도 최적화가 효과적입니다.[1]

4. **알고리즘 의존적 결과**: PPO, DPO, SimPO는 동일한 선호도 데이터에서도 서로 다른 값 정렬 결과를 도출합니다.[1]

***

## 문제 정의 및 제안 방법

### 해결하고자 하는 문제

LLM이 사회에서 점점 더 중요한 역할을 하면서 경제 성장 vs 기후 변화 대응, 이민 정책 등 **가치 관련 질문에 직면**합니다. 그러나 모델이 이러한 값을 어떻게 습득하고 어느 단계에서 변하는지 이해하지 못하고 있습니다.[1]

### 값(Values)의 정의

논문은 **값을 표정(stance)**으로 규정합니다. 이는 특정 주제에 대한 질문에 응답할 때 모델이 채택하는 명시적 입장이며, {support, neutral, oppose}의 삼원 분류입니다.[1]

특정 주제 T에 대한 모델 θ의 값은 다음과 같이 정의됩니다:

$$v_{\theta}(T) = \left[E_{x \in X_T, y \sim \pi_{\theta}(\cdot|x)}[p(s|x,y,T)]\right]_{s \in S}$$

여기서 S = {support, neutral, oppose}입니다.[1]

### 평가 메트릭

#### 1. 드리프트 크기(Drift Magnitude)

두 체크포인트 t와 t' 사이의 표정 분포 변화:

$$M_{s,\theta,T}(t, t') = v_{\theta,t'}(T)_s - v_{\theta,t}(T)_s$$

#### 2. 드리프트 시간(Drift Time)

값이 최댓값의 95% 신뢰도에 도달하기까지의 학습 진행 비율:

$$\eta_{s,\theta,T}(t, t') = \frac{\eta^{ext}}{\eta_{total}}$$

여기서 ηext는 극값에 도달하는 데 필요한 학습 스텝이고, ηtotal은 전체 학습 스텝입니다.[1]

### V-PRISM 평가 데이터셋

PRISM 데이터셋의 8,100개 질문에서:
- 다단계 파이프라인으로 가치 관련 질문 선별
- 11개 의미적 범주로 군집화
- 각 범주에서 50개씩 선택 → 총 550개 프롬프트[1]

토픽 범주: 낙태, 이민, 기후변화, 경제정책, 종교, 가족관계, 사형 등

***

## 모델 구조 및 학습 방법

### 1. 감독 미세조정(SFT)

$$L_{SFT}(\theta; D_{SFT}) = -E_{(x,y) \sim D_{SFT}}[\log \pi_{\theta}(y|x)]$$

**실험 설정:**
- 모델: Llama-3 (3B, 8B), Qwen-3 (4B, 8B)
- 데이터셋: WildChat (실제 사용자 상호작용), Alpaca (합성 데이터)
- 3 epoch 학습, 500(100) 스텝마다 체크포인트 저장[1]

### 2. 선호도 최적화 알고리즘

#### PPO (Proximal Policy Optimization)

$$L_{PPO}(\theta; D_{Pref}) = -E_{x \sim D_x, y \sim \pi_{\theta}(\cdot|x)}[r(x,y)] + \beta D_{KL}(\pi_{\theta}(y|x)||\pi_{ref}(y|x))$$

보상 모델 r(x,y)를 별도로 학습하고, KL 정규화로 참조 모델과의 편차를 제어합니다.[1]

#### DPO (Direct Preference Optimization)

$$L_{DPO}(\theta; D_{Pref}) = -E_{(x,y_w,y_l) \sim D_{Pref}}\left[\log \sigma\left(\beta \log \frac{\pi_{\theta}(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi_{\theta}(y_l|x)}{\pi_{ref}(y_l|x)}\right)\right]$$

Bradley-Terry 순위 모델을 정책 모델에 통합하여 명시적 보상 모델 없이 학습합니다.[1]

#### SimPO (Simple Preference Optimization)

$$L_{SimPO}(\theta; D_{Pref}) = -E_{(x,y_w,y_l) \sim D_{Pref}}\left[\log \sigma\left(\frac{\beta}{|y_w|}\log \pi_{\theta}(y_w|x) - \frac{\beta}{|y_l|}\log \pi_{\theta}(y_l|x) - \gamma\right)\right]$$

참조 모델을 제거하고 목표 마진 γ를 도입하여 더 단순화합니다.[1]

***

## 성능 향상 결과

### SFT의 가치 정렬 효과

**발견 1**: SFT는 매우 초기 단계부터 값을 설정합니다.
- Llama-3-3B (WildChat): 드리프트 크기 = 0.38 (neutral), 드리프트 시간 = 0.09
- Qwen-3-4B (WildChat): 유사 패턴 관찰[1]

**발견 2**: SFT 데이터셋이 최종 값을 결정합니다.
- WildChat: 높은 중립 표정 비율 → GPT-3.5의 거부 경향 반영
- Alpaca: 높은 지지 표정 비율 → 합성 데이터의 내재적 편향[1]

### 선호도 최적화의 제한된 효과

**표준 데이터셋 사용 시** (UltraFeedback, HH-RLHF):
- 드리프트 크기: 매우 낮음 (0.01-0.27 범위)
- 드리프트 시간: 낮음 (0.14-0.42 범위)

**원인**: 선호 쌍의 "선택됨"과 "거부됨" 응답이 매우 유사한 값 분포를 보입니다. 스타일/톤 차이는 있지만 기본 입장(stance)은 일치합니다.[1]

### 합성 데이터를 통한 가치 신호 차별화

**PPO의 특성**: KL 정규화 때문에 SFT 이전 정책에 강하게 고정됩니다.
- 지지 정렬 설정: 드리프트 크기 = 0.0
- 반대 정렬 설정: 드리프트 크기 = -0.02[1]

**DPO의 특성**: 선호-민감적 증폭(preference-sensitive amplification)
- SFT 이전 확률이 높은 입장 → 강력 증폭 (드리프트 = 0.53)
- SFT 이전 확률이 낮은 입장 → 부분적 전환 (드리프트 = 0.46)

이는 DPO 손실이 정책과 참조 정책의 로그 비율을 최적화하기 때문입니다.[1]

**SimPO의 특성**: 온건한 값 드리프트
- DPO보다 작은 크기와 느린 변화 속도
- 목표 마진 γ에 의해 조절됨[1]

| 알고리즘 | SFT 이전 확률 낮음 | SFT 이전 확률 높음 | 특징 |
|---------|-----------------|-----------------|------|
| **PPO** | 0.0 (변화 없음) | 0.0 (변화 없음) | KL 정규화로 보수적 |
| **DPO** | 0.46 (부분 전환) | 0.53 (강력 증폭) | 선호 신호에 민감 |
| **SimPO** | 0.15 (온건) | 0.15 (온건) | 목표 마진으로 조절 |

***

## 일반화 성능 향상

논문은 **모델 능력 보존**을 중요하게 다룹니다. MMLU, HellaSwag, GPQA Diamond, PiQA 벤치마크에서 미세조정 후 성능을 평가합니다.[1]

### 일반화 성능의 함의

**긍정적 측면:**
1. **SFT의 선택성**: 낮은 값 드리프트는 기존 능력 손실을 최소화
2. **선호도 최적화의 안정성**: 작은 변화는 과도한 분포 이동 방지

**위험 요소:**
1. **강제된 값 정렬**: 합성 데이터의 명확한 가치 간격은 초과 최적화 위험
2. **데이터 분포 불일치**: 표준 데이터셋과 실제 모델 출력 분포의 괴리

### 하이퍼파라미터 영향

PPO의 β (KL 정규화 계수):
- 큰 β → 참조 정책에 강하게 고정 → 최소 값 드리프트
- 작은 β → 상대적으로 큰 값 드리프트 가능[1]

DPO의 β:
- 값 드리프트의 강도를 직접 제어

SimPO의 γ (목표 마진):
- 값 드리프트 크기에 미미한 영향 (0.34 vs 0.36)
- 마진의 효과는 학습 과정에서 이미 내재됨[1]

***

## 한계(Limitations)

### 1. 방법론적 한계

**표정 기반 값 측정의 제한:**
- 이진 입장(support/oppose)은 미묘한 차이 손실
- 경제적 이유와 문화적 이유로 인한 반대는 동일 분류
- GPT-4o의 분류 오류 가능성 (수동 검증 100개만 수행)[1]

**지리적 편향:**
- PRISM 데이터셋이 영어 사용자 중심 (USA, UK, Europe)
- 원주민 토지권 등 특정 지역 이슈 누락[1]

### 2. 실험적 한계

**평가 데이터셋 제약:**
- 550개 프롬프트만 분석 (비용 제약)
- 11개 토픽만 검토

**합성 데이터의 제한:**
- Qwen2.5-72B로만 생성
- 실제 다양한 인간 가치 표현 부재

### 3. 일반화 성능 측면

**알려진 문제:**
- 미세조정은 일반 능력 저하 가능성[1]
- 이 연구는 값 드리프트만 추적하고 일반 성능 저하는 부분적만 측정

***

## 최신 연구와의 비교

### 2024-2025년 관련 최신 연구 분석

#### 1. 선호도 최적화 알고리즘 비교

**"Is DPO Superior to PPO for LLM Alignment?" (Xu et al., 2024)**[2]
- DPO가 분포 외 응답을 악용할 수 있다는 이론적 증명
- PPO가 다양한 벤치마크에서 우수성 입증
- 이 논문과의 관계: Value Drifts는 제어된 합성 데이터에서 DPO의 선호-민감적 증폭 특성을 실증적으로 확인[2]

**Q♯: Provably Optimal RL (Zha et al., 2025)**[3]
- 가치 기반 RL 알고리즘으로 KL 제약 최적화
- 이 논문의 시사점: PPO와 다른 신규 알고리즘의 선호도 최적화 평가 필요[3]

#### 2. SFT 데이터와 일반화

**"Massive Supervised Fine-tuning Experiments" ()**
- 1,000개 이상의 SFT 모델 분석
- 주요 발견: 훈련 데이터의 낮은 perplexity가 SFT 효과 예측
- 중간 레이어 가중치 변화가 성능 향상과 상관
- Value Drifts와 연관: SFT의 초기 값 설정은 중간 레이어 변화와 일관성[4]

**PiKa: Expert-Level Synthetic Datasets ()**
- 30k SFT 데이터로 10M 데이터 경쟁 모델 성능 달성
- 합성 데이터의 품질 > 규모
- Value Drifts와의 시사점: 고품질 합성 데이터로 값 간격 확대 가능[5]

#### 3. 데이터 선택 전략

**"Improving LLM Alignment via Preference Data Selection" (Deng et al., 2025)**[6]
- 마진 최대화 원칙으로 10% 데이터만 사용 가능
- 3-8% 성능 향상 달성
- Value Drifts와의 통합: 값 간격 기반 데이터 선택이 값 정렬 효율성 향상[6]

**"The Best Instruction-Tuning Data are Those That Fit" (GRAPE, 2024)**[7]
- SFT는 사전학습 분포와 일치한 데이터에서 가장 효과적
- 콘텐츠 유사성보다 확률이 중요
- Value Drifts와의 연관: 값 정렬도 동일한 분포 정렬 원칙 적용 가능[7]

#### 4. 값과 정렬의 다각도 연구

**"Internal Value Alignment in LLMs" (Jin et al., 2025)**[8]
- LLM이 인간 가치 체계를 정확히 복제하지 않음
- 상충하는 값들이 유사한 방향으로 인코딩됨
- Value Drifts의 함의: 표정 기반 측정의 제한성 재확인[8]

**"Mind the Value-Action Gap" (2025)**[9]
- LLM 생성 가치 서술과 실제 행동의 불일치
- Schwartz 가치 이론 적용
- Value Drifts와의 차별성: 표정만이 아닌 행동 일관성도 필요[9]

#### 5. 합성 데이터 기반 정렬

**"ICON2: Aligning LLMs Using Self-Synthetic Preference Data" (2025)**[10]
- 내재적 제어로 선호도 쌍 생성
- 다중 샘플링 필요 없음
- Value Drifts와의 보완: 더 효율적인 값 간격 생성 방법[10]

***

## 앞으로의 연구 방향 및 고려사항

### 1. 방법론 개선

#### 값 측정의 정교화
- **현재**: 이진 표정 분류
- **개선 방향**: 
  - 연속 값 점수 (0-1)로 미묘한 차이 포착
  - 다양한 이유(경제, 도덕, 문화)에 따른 입장 차별화
  - 모델 자체 해석 능력 활용

#### 데이터셋 다양성 확대
- 비영어권 가치 질문 포함
- 문화적으로 다양한 표현 수집
- 갈등 상황에서의 가치 프로파일링

### 2. 선호도 최적화 알고리즘 발전

#### 가치 인식적 선호도 최적화
- 선호도 쌍이 동일 값 공간에 있는지 확인
- 값 변화를 의도한 데이터만 선택
- 값 드리프트와 일반 능력의 트레이드오프 최적화

#### 다중 가치 최적화
- 상충하는 가치 간 균형 유지
- 파레토 최적 정책 생성 (PEO 등 참고)[11]
- 사용자 선호도에 따른 동적 값 조정

### 3. 일반화 성능 보증

#### 이중 목적 최적화
- 값 정렬 + 일반 능력 보존
- LEVI(Layer-wise Ensemble) 등 방법론 적용
- 분포 외 성능 평가 확대

#### 통제된 미세조정
- 학습률 감소로 일반 능력 손실 완화 ()
- 중간 레이어만 업데이트 ( 발견)
- LoRA 등 파라미터 효율적 방법 활용

### 4. 실제 응용을 위한 고려사항

#### 다중 이해관계자 가치 정렬
- 단일 가치 벡터의 한계 인식
- 복수 가치 프로필 제공 (의 복수주의 접근)
- 사용자별 값 선호도 맞춤화

#### 투명성과 책임성
- 값 정렬의 출처와 방법론 공개
- 의도하지 않은 값 변화 감시
- 모델 "가치"의 의인화 위험 경고

#### 데이터 큐레이션 가이드라인
- SFT 데이터셋 선택 기준 명시
- 합성 데이터 생성 규범화
- 값 간격 정량화 방법 표준화

### 5. 이론적 심화

#### 값 습득의 메커니즘
- 어떤 모델 파라미터가 가치 표현 담당하는가?
- 값 드리프트와 표현 공간 변화의 관계
- 사전학습 가치 VS 사후학습 가치의 상호작용

#### 일반화 경계
- 값 정렬이 얼마나 강할 때 일반화 손상되는가?
- 값 영역 VS 비-값 영역의 학습 기울기 차이
- 값 복잡도와 학습 난제의 관계

***

## 실무적 권고사항

### SFT 단계에서

1. **신중한 데이터셋 선택**
   - 기본값이 되므로 명확한 값 정렬 의도 필요
   - WildChat (중립) vs Alpaca (지지) 특성 파악
   - 사전학습 분포와의 일치도 검증

2. **조기 검증**
   - SFT 초기 (첫 10%)부터 값 프로파일 모니터링
   - 의도하지 않은 값 변화 조기 감지

### 선호도 최적화 단계에서

1. **가치 간격 검증**
   - 선호 데이터의 표정 분포 분석
   - 간격이 작으면 DPO/SimPO는 효과 제한적
   - 명확한 값 신호 필요시 합성 데이터 추가

2. **알고리즘 선택**
   - **PPO**: 기존 값 최대한 보존 (보수적)
   - **DPO**: 선호도 신호에 민감 (공격적)
   - **SimPO**: 중간 수준의 값 조정

3. **하이퍼파라미터 튜닝**
   - PPO β: 크면 보수적, 작으면 공격적
   - DPO β: 값 변화의 강도 제어
   - 일반 능력 보존과의 균형 중시

### 평가 단계에서

1. **값 일관성 모니터링**
   - V-PRISM 같은 다주제 평가 필수
   - 사전/사후 값 프로파일 비교
   - 의도하지 않은 값 변화 검출

2. **일반 능력 검증**
   - 가치 관련 없는 벤치마크도 평가
   - 특히 중간 레이어 변화 모니터링

***

## 결론

"Value Drifts" 논문은 LLM 값 정렬을 **동적 프로세스**로 새롭게 인식하게 합니다. 기존의 "최종 모델이 올바른 값을 가지는가?"라는 정적 질문에서 "어느 단계에서, 어떤 메커니즘으로 값이 변하는가?"라는 동적 질문으로 전환합니다.[1]

가장 놀라운 발견은 **SFT의 지배적 역할**입니다. 수십만 개의 인간 선호도 데이터로 학습하는 선호도 최적화 단계보다, 작은 규모의 지시 데이터로만 학습하는 SFT가 최종 값을 거의 결정합니다. 이는 LLM 개발 단계에서 **초기 데이터 큐레이션의 중요성**을 강조합니다.[1]

또한 논문은 **데이터 품질의 조용한 위기**를 지적합니다. 표준 선호도 데이터셋의 "선택됨"과 "거부됨" 응답이 근본적인 가치 차이가 없다면, 막대한 계산 비용으로 진행하는 선호도 최적화는 제한된 효과만 거둘 수 있습니다.[1]

향후 연구는 이 발견들을 바탕으로:
- 더 정교한 값 측정 방법론
- 값 인식적 데이터 선택 전략
- 다중 가치를 균형있게 다루는 기술
- 값 정렬과 일반 능력의 동시 최적화

를 개발해야 할 것입니다.[4][5][3][2][6][7][1]

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c1fae81c-7ced-42d5-9197-5b1988cb9700/2510.26707v1.pdf)
[2](https://arxiv.org/html/2404.10719v1)
[3](https://arxiv.org/abs/2502.20548)
[4](https://arxiv.org/html/2506.14681v1)
[5](https://arxiv.org/html/2510.06670v1)
[6](https://arxiv.org/abs/2502.14560)
[7](https://arxiv.org/html/2502.04194v3)
[8](https://aclanthology.org/2025.acl-long.1326.pdf)
[9](https://arxiv.org/html/2501.15463v4)
[10](https://arxiv.org/html/2509.05605v1)
[11](https://arxiv.org/abs/2503.01233)
[12](https://arxiv.org/abs/2510.26707)
[13](https://arxiv.org/abs/2508.16982)
[14](https://arxiv.org/abs/2507.16679)
[15](http://journal.unm.ac.id/index.php/JMATHCOS/article/view/7309)
[16](https://journal.icca.web.id/index.php/StrengthandConditioning/article/view/10)
[17](https://journal.icca.web.id/index.php/StrengthandConditioning/article/view/13)
[18](https://gemangabdi.unram.ac.id/index.php/gemangabdi/article/view/631)
[19](https://arxiv.org/abs/2506.00676)
[20](http://arxiv.org/pdf/2405.13578.pdf)
[21](https://arxiv.org/pdf/2411.00062.pdf)
[22](http://arxiv.org/pdf/2403.04224.pdf)
[23](https://aclanthology.org/2023.emnlp-main.844.pdf)
[24](https://arxiv.org/pdf/2503.02832.pdf)
[25](http://arxiv.org/pdf/2305.13735.pdf)
[26](http://arxiv.org/pdf/2503.01864.pdf)
[27](http://arxiv.org/pdf/2408.10392.pdf)
[28](https://openreview.net/pdf/40bbbfdb18c5d9d3ed6fbdde26b4d3e7f56ba8ba.pdf)
[29](https://toloka.ai/blog/llm-fine-tuning-unlocking-the-true-potential-of-large-language-models/)
[30](https://arxiv.org/pdf/2508.16982.pdf)
[31](https://www.youtube.com/watch?v=-YZ-YZ05VXU)
[32](https://www.nature.com/articles/s41586-025-09937-5)
[33](https://cameronrwolfe.substack.com/p/direct-preference-optimization)
[34](https://aclanthology.org/2025.findings-emnlp.854.pdf)
[35](https://apxml.com/courses/rlhf-reinforcement-learning-human-feedback/chapter-6-advanced-rlhf-techniques/comparing-ppo-dpo-practice)
[36](https://openreview.net/pdf?id=5HCnKDeTws)
[37](https://www.lgresearch.ai/blog/view?seq=565)
[38](https://www.together.ai/blog/direct-preference-optimization)
[39](https://arxiv.org/abs/2505.00661)
[40](https://arxiv.org/html/2505.02666v2)
[41](https://arxiv.org/pdf/2404.10719.pdf)
[42](https://arxiv.org/html/2403.00625v1)
[43](https://arxiv.org/pdf/2507.20067.pdf)
[44](https://arxiv.org/html/2410.15595v3)
[45](https://arxiv.org/html/2502.04419v1)
[46](https://arxiv.org/html/2506.19780v2)
[47](https://arxiv.org/html/2505.20903v1)
[48](https://arxiv.org/html/2507.11316v1)
[49](https://arxiv.org/abs/2404.10719)
[50](https://arxiv.org/html/2504.05632v1)
[51](https://arxiv.org/abs/2512.00778)
[52](https://ieeexplore.ieee.org/document/10823837/)
[53](https://www.semanticscholar.org/paper/8880022a7d461c8c51266fa2908a8669d770772e)
[54](https://ieeexplore.ieee.org/document/10659799/)
[55](https://ieeexplore.ieee.org/document/10531528/)
[56](https://ieeexplore.ieee.org/document/10782018/)
[57](https://ieeexplore.ieee.org/document/10912895/)
[58](https://ieeexplore.ieee.org/document/10841600/)
[59](https://ieeexplore.ieee.org/document/10584876/)
[60](https://ieeexplore.ieee.org/document/10471046/)
[61](https://arxiv.org/abs/2401.07037)
[62](http://arxiv.org/abs/2110.04366)
[63](https://arxiv.org/pdf/2109.05687.pdf)
[64](https://arxiv.org/pdf/2106.15434.pdf)
[65](https://arxiv.org/pdf/2103.01542.pdf)
[66](https://arxiv.org/pdf/2301.05487.pdf)
[67](http://arxiv.org/pdf/2402.15082.pdf)
[68](https://arxiv.org/pdf/2406.07337.pdf)
[69](https://arxiv.org/pdf/2402.04644.pdf)
[70](https://www.nature.com/articles/s41467-024-51844-2)
[71](https://www.anyscale.com/blog/direct-preference-optimization-with-synthetic-data)
[72](https://aclanthology.org/2025.emnlp-main.131.pdf)
[73](https://aclanthology.org/2025.emnlp-main.196.pdf)
[74](https://openreview.net/forum?id=ru93xpQFi1)
[75](https://www.tensorflow.org/tutorials/images/transfer_learning)
[76](https://www.philschmid.de/rl-with-llms-in-2025-dpo)
[77](https://www.innovatiana.com/en/post/sft-dataset)
[78](https://arxiv.org/html/2411.01195v1)
[79](https://arxiv.org/abs/2402.08005)
[80](https://huggingface.co/learn/llm-course/chapter11/3)
[81](https://openreview.net/forum?id=PeLLMw3wLX)
[82](https://openreview.net/forum?id=7visV100Ms&noteId=6vVcxpCHKW)
[83](https://labelyourdata.com/articles/llm-fine-tuning/supervised-fine-tuning)
[84](https://arxiv.org/pdf/2503.20110.pdf)
[85](https://arxiv.org/html/2509.23753v1)
[86](https://arxiv.org/html/2508.05685v6)
[87](https://arxiv.org/pdf/2505.14826.pdf)
[88](https://arxiv.org/html/2503.20110v1)
[89](https://arxiv.org/html/2506.01901v1)
[90](https://arxiv.org/abs/2405.16236)
[91](https://arxiv.org/html/2410.06961v1)
[92](https://arxiv.org/html/2508.16546v1)
