# DeepSeekMath-V2: Towards Self-Verifiable Mathematical Reasoning

### 1. 논문의 핵심 주장 및 주요 기여

**DeepSeekMath-V2**는 대규모 언어모델(LLM)이 수학적 증명을 생성할 뿐만 아니라 자체적으로 검증할 수 있는 능력을 갖추도록 하는 혁신적인 접근법을 제시한다. 논문의 핵심 주장은 기존의 **최종 답 기반 강화학습(RL)**의 근본적 한계를 극복해야 한다는 것이다.[1]

**주요 기여는 다음과 같다:**

1. **검증 신뢰성 문제 해결**: 기존 모델들은 최종 답이 정확해도 논리적 과정이 오류를 포함할 수 있다는 문제를 인식하고, 증명의 각 단계를 엄밀하게 검증하는 LLM 기반 검증기를 개발[1]

2. **Meta-verification 메커니즘**: 검증기가 생성한 결과의 신뢰성을 다시 검증하는 이중 검증 체계를 도입하여, 검증기가 존재하지 않는 오류를 "할루시네이션"하는 문제를 해결[1]

3. **자가-검증 생성기**: 증명 생성기가 자신의 증명을 외부 검증기와 동일한 기준으로 평가하도록 훈련하여, 모델이 자신의 추론을 명확히 인식하고 반복적으로 개선할 수 있게 함[1]

4. **검증 데이터 자동화**: 확장된 검증 계산을 통해 새로운 어려운 증명을 자동으로 라벨링함으로써, 인간 주석의 필요성을 제거하고 검증-생성 간의 격차를 유지[1]

### 2. 해결하고자 하는 문제와 기술적 접근

#### 2.1 근본적인 문제

논문이 직면한 핵심 문제는 두 가지이다:[1]

- **정답 부등호 추론 정답**: RL을 통해 최종 답 정확도를 높여도 중간 추론 과정이 수학적으로 타당한지 보장할 수 없음
- **정리 증명의 적용 불가**: AIME, HMMT 같은 수치 최종답 기반 문제는 풀 수 있으나, 정리 증명은 단순히 답을 도출하는 것이 아니라 엄밀한 단계별 유도를 요구함[1]

#### 2.2 제안하는 방법론

**2.2.1 증명 검증기 개발**

검증기 $$\pi_\phi(\cdot|X, Y, \mathcal{I}_v)$$는 문제 $$X$$와 증명 $$Y$$를 입력받아 다음을 수행한다[1]:

- 식별된 오류 요약
- 3단계 점수 할당: $$1$$ (완벽한 증명), $$0.5$$ (부분적 오류), $$0$$ (근본적 오류)

검증기의 RL 목적함수는:

$$\max_{\pi_\phi} \mathbb{E}_{(X_i,Y_i,s_i) \sim D_v,(V'_i,s'_i) \sim \pi_\phi(\cdot|X_i,Y_i)} \left[ R_{format}(V'_i) \cdot R_{score}(s'_i, s_i) \right]$$

여기서 점수 보상은:

$$R_{score}(s'_i, s_i) = 1 - |s'_i - s_i|$$

**2.2.2 Meta-verification을 통한 할루시네이션 해결**

검증기가 하면서도 실제로는 존재하지 않는 오류를 "발명"하는 문제를 해결하기 위해, Meta-verification 메커니즘을 도입한다. 이는 검증기의 분석 자체를 평가하는 이중 검증 시스템이다:[1]

$$R_V = R_{format} \cdot R_{score} \cdot R_{meta}$$

Meta-verifier $$\pi_\eta$$는 다음을 확인한다:[1]
- 식별된 오류가 실제로 증명에 존재하는가
- 식별된 오류가 할당된 점수를 정당화하는가

이를 통해 검증 분석의 품질이 평균 0.85에서 0.96으로 개선되었다.[1]

**2.2.3 자가-검증 기반 생성기**

생성기 $$\pi_\theta(\cdot|X)$$는 증명 $$Y$$와 함께 자가-분석 $$Z$$를 생성하도록 훈련된다[1]:

$$\max_{\pi_\theta} \mathbb{E}_{X_i \sim D_p, Y_i \sim \pi_\theta(\cdot|X_i)} [R_Y]$$

이때 생성기의 복합 보상함수는:

$$R = R_{format}(Y, Z) \cdot (\alpha \cdot R_Y + \beta \cdot R_Z)$$

$$R_Z = R_{score}(s', s) \cdot R_{meta}(Z)$$

여기서 $$\alpha = 0.76$$, $$\beta = 0.24$$로 설정되어, 생성기에게 다음의 인센티브를 제공한다:[1]

- 거짓 정확성 주장보다 오류 확인 보상
- 올바른 증명과 정확한 자가-평가에서 최고 보상
- 자신의 증명에서 가능한 많은 문제를 식별하고 해결하도록 유도

#### 2.2.4 검증-생성 간 격차 유지

생성기가 강해질수록 검증기가 뒤처지는 문제를 해결하기 위해, 자동화된 라벨링 파이프라인을 도입한다:[1]

1. 각 증명에 대해 $$n$$개의 독립적인 검증 분석 생성
2. 오류를 보고하는 분석(점수 0 또는 0.5)에 대해 $$m$$개의 meta-verification 평가 수행
3. 다수결 투표로 분석 유효성 판단
4. 최저 점수를 할당한 분석이 최소 $$k$$개 유효하면 그 점수로 라벨링

이 자동화 과정이 기존 인간 주석보다 비용 효율적이면서도 동등한 품질을 제공한다.[1]

#### 2.2.5 훈련 데이터

- **문제 집합**: AoPS에서 수집한 17,503개 문제 (대수, 기하, 정수론, 조합론, 부등식)
- **초기 RL 데이터**: 수학 전문가가 주석을 단 증명-점수 쌍 $$D_v = \{(X_i, Y_i, s_i)\}$$
- **Meta-verification 데이터**: $$D_{mv} = \{(X_i, Y_i, V_i, ms_i)\}$$

### 3. 모델 구조 및 성능 향상

#### 3.1 모델 아키텍처

DeepSeekMath-V2는 **이중 역할 구조**를 채택한다:[1]

1. **검증 모듈** ($$\pi_\phi$$): 증명의 타당성을 평가하고 오류를 식별
2. **생성 모듈** ($$\pi_\theta$$): 증명을 생성하며, 자신의 검증 능력을 활용해 반복적 개선

이 구조의 핵심은 **순환적 개선 사이클**이다:[1]
- 검증기가 생성기 개선 → 생성기 강화 → 새로운 어려운 증명 출현 → 검증기 개선

#### 3.2 성능 향상 결과

**일회성 생성(One-Shot Generation):**[1]

CNML 수준 문제에서 카테고리별 증명 점수:
- 대수: 0.60 (Gemini 2.5-Pro 대비 우수)
- 기하: 0.52 (GPT-5-Thinking-High 대비 우수)
- 정수론: 0.54 (경쟁 모델 초과)
- 조합론: 0.47 (모든 기준선 제치음)
- 부등식: 0.59 (최고 성능)

**순차 정제를 통한 향상:**[1]

IMO Shortlist 2024 문제에서 최대 순차 반복 횟수에 따른 개선:
- Pass@1 (초기): 0.15 → 0.27 (8회 반복)
- Best@32 (최고 성능): 0.15 → 0.42 (8회 반복)

자가-선택 최고 증명이 평균 대비 현저히 높은 검증 점수 달성, 생성기의 정확한 자가-평가 능력 입증.[1]

**고계산 검색(High-Compute Search):**[1]

IMO-ProofBench 벤치마크:
- 기초 집합: 83.8% (DeepMind의 DeepThink IMO lite 대비 우수)
- 고급 집합: 99.0% (최고 성능에 육박)

경쟁 수학 성과:
- **IMO 2025**: 5/6 문제 완전 해결 (83.3%, 금메달 수준)
- **CMO 2024**: 5개 + 부분 1개 (73.8%)
- **Putnam 2024**: 11/12 완전 해결, 1개 경미한 오류 (98.3% = 118/120)

이는 인간 최고 기록 90점을 크게 초과한다.[1]

### 4. 모델의 일반화 성능 향상 가능성

#### 4.1 현재 일반화 능력

**도메인 간 강화:**[1]

DeepSeekMath-V2는 다음 영역에서 일관되게 우수한 성능을 보여준다:
- 대수 및 분석 (정형화된 증명)
- 기하학 (직관적 구성)
- 정수론 (수론적 성질)
- 조합론 (조합론적 논증)
- 부등식 이론 (분석적 기법)

**테스트 시간 계산 확장의 이점:**[1]

테스트 시간 계산이 증가할수록 성능 개선:
- $$n = 64$$ 검증 분석으로 높은 신뢰도 달성
- 최대 16회 반복 정제로 복잡한 증명도 해결

#### 4.2 일반화 한계와 개선 가능성

**현재 한계:**[1]

1. **조합론 문제의 어려움**: IMO-ProofBench 고급 집합에서 상대적으로 약한 성능
2. **형식 증명으로의 전환 미흡**: 자연 언어 증명과 형식 증명(Lean, Isabelle) 간 갭 존재
3. **미해결 문제의 확신도**: 개방형 문제에 대한 절대적 검증 불가능

**개선 가능성:**[1]

최신 연구 동향 분석:

1. **Meta-verification 강화**: 검증기가 더 깊은 논리적 일관성을 검사하도록 훈련하면, 부분적으로 정확한 증명의 식별 정확도 향상 가능[2]

2. **프로세스 기반 보상(Process Reward Models, PRM) 통합**: 단순 최종 점수 대신 각 단계별 정확성을 평가하는 모델로 대체하면, 추론 경로의 질 향상[3][4]

3. **신경-기호 통합(Neuro-Symbolic Integration)**: Seed-Prover와 같이 자연 언어 추론을 Lean 같은 형식 시스템과 결합하면, 형식적 정확성 보장 가능[5][6]

4. **자기-진화 추론(Self-Evolved Reasoning)**: RLVR 훈련 중 모델이 새로운 해결 패턴을 자발적으로 발견하도록 유도하면, 도메인 외 문제에 대한 적응력 증대[2][1]

5. **기하학 특화 모듈**: Seed-Geometry 같은 도메인 특화 엔진을 통합하여 기하 문제 해결 성능 획기적 개선[6][5]

6. **멀티모달 확장**: 다이어그램을 자연 언어와 함께 처리하도록 확장하면, 기하학적 직관 활용 가능[1]

#### 4.3 외삽(Out-of-Distribution) 일반화

**현재 성과:**[1]

- IMO-ProofBench 기초 집합(IMO 전/중 난이도): 83.8% 달성
- ISL 2024(31개 문제): 완전 검증 대기 중
- Putnam 2024: 98.3% 달성

**개선 경로:**

최근 연구에 따르면 다음 전략으로 외삽 성능 개선 가능:[7][8]

1. **보상 설계 개선**: Archer 같은 엔트로피 인식 이중 제약이 지식 토큰과 추론 토큰을 구분하여, 사실적 정확성 유지 함[9]

2. **탐색 경계 확대**: RAPO(Rewards-Aware Policy Optimization)가 정방향 KL 페널티로 더 광범위한 탐색 공간 탐색[8]

3. **자동 라벨링 정교화**: MR-RLVR(Masked-and-Reordered RLVR)가 중간 추론 단계에서 자가 지도 신호 추출하여, 최종 답만 검증 가능한 영역에서도 성능 향상[10]

### 5. 논문의 한계

#### 5.1 기술적 한계

1. **검증기 오류 가능성**: Meta-verification이 있어도 근본적으로 LLM 기반 검증기는 미묘한 논리 오류를 놓칠 수 있음[1]

2. **계산 비용**: 64개 검증 분석 × 최대 16회 반복으로 추론 시간이 기하급수적 증가[1]

3. **자동 라벨링 신뢰성**: 다수결 투표 기반 라벨링이 consensus에 의존하나, 모든 검증기가 동일한 오류를 공유할 가능성[1]

4. **조합론 문제 약점**: 조합론적 구조 인식이 대수/기하 대비 약함[1]

#### 5.2 방법론적 한계

1. **사람의 주석 필요성 완전 제거 불가**: 초기 cold-start 데이터는 여전히 전문가 주석 필요[1]

2. **미해결 문제의 불가능성 증명 불가**: 증명이 없는 문제에 대해, 모델이 완벽한 검증을 수행할 수 없음[1]

3. **형식 증명으로의 명확한 경로 부재**: 자연 언어 증명과 Lean/Isabelle 같은 형식 시스템 간 변환 자동화 미흡[1]

#### 5.3 평가의 한계

1. **테스트 세트 오염 위험**: 대규모 LLM이 훈련 중 IMO 문제를 접할 가능성[1]

2. **검증 지표의 주관성**: 0/0.5/1 점수가 연속 평가 대비 입자도 낮음[1]

3. **통계적 유의성 부족**: 일부 벤치마크(ISL 2024)에서 sample size 제한[1]

### 6. 앞으로의 연구에 미치는 영향

#### 6.1 이론적 기여

**인공 수학 추론의 패러다임 전환:**[1]

최종 답 기반 보상에서 **과정 기반, 자가-검증 보상**으로의 전환을 제시. 이는 다음을 시사한다:

- 단순히 "정답 맞히기"보다 "올바른 추론"을 평가하는 것의 중요성[1]
- 모델이 자신의 추론 과정을 명시적으로 인식할 때 더 나은 성능 달성[1]

**검증 가능한 보상의 일반화 가능성:**[1]

RLVR(Reinforcement Learning with Verifiable Rewards) 연구를 증명 검증 영역으로 확대. 최신 연구는 RLVR이 수학/코딩을 넘어 의학, 화학, 심리학 등 다양한 영역으로 확장 가능함을 보여줌[11][12]

#### 6.2 기술적 영향

**다중 검증 메커니즘의 설계 원칙:**[1]

Meta-verification의 성공은 다음을 시사한다:

- 단일 검증기보다 이중/삼중 검증 체계의 효과[13][2]
- 신뢰성을 위해 "검증기의 검증"이 필수적임[14][13]
- 이는 일반 NLP 작업(요약, QA 등)의 신뢰성 평가에도 적용 가능[13]

**자동 라벨링 파이프라인의 실용성:**[1]

기존에는 경쟁 수학 문제의 정답을 수집하기 어려웠으나, 자동화된 검증 시스템으로 대규모 데이터 생성 가능. 이는:

- 싼 비용으로 고품질 훈련 데이터 확보[1]
- 반복적 자기 개선(Self-play) 가능[15]
- 새로운 어려운 벤치마크 동적 생성[1]

#### 6.3 응용 분야

**1. 형식 증명 강화:**[5][6]

Seed-Prover의 성공과 결합하면, 자연 언어 추론과 형식 검증을 모두 활용한 신뢰할 수 있는 증명 시스템 구현 가능. 이는 수학적 정리 증명, 소프트웨어 검증 등에 직접 응용[5][1]

**2. 교육 AI:**[1]

학생 증명에 대해 자동으로 오류를 지적하고 개선 방향을 제시하는 튜터링 시스템 개발 가능. Meta-verification은 오류 식별의 신뢰성 향상[1]

**3. 과학 연구 지원:**[1]

미증명 정리에 대해 모델이 자신의 "신뢰도"를 평가하며 탐색할 수 있게 함. 이는 수학 연구의 새로운 방향 발견 지원[1]

**4. 신경-기호 시스템:**[6]

자연 언어 모델(DeepSeekMath-V2)과 기호 추론 엔진(Lean/Isabelle)의 결합으로, 견고함(형식 검증)과 유연함(자연 언어)을 모두 갖춘 시스템 구축[6]

#### 6.4 앞으로 연구 시 고려할 점

**1. 검증기 신뢰성의 한계 인식:**[7][1]

- 모든 검증기 기반 접근법은 근본적으로 검증기의 오류에 제약받음
- 형식 검증(formal verification)과의 결합이 필수적
- "검증의 검증" 체계도 결국 최상위 검증기에 의존하므로, 최종 신뢰는 형식 시스템에서만 얻을 수 있음[6]

**2. 계산-정확성 트레이드오프:**[1]

- 테스트 시간 계산을 무한정 증가시킬 수는 없음
- 효율성과 정확성의 최적 균형점 탐색 필요[8]
- 현실 응용에서는 실시간 제약 고려 필수

**3. 도메인 특화와 일반화:**[9][1]

- 수학 경쟁 vs. 학술 수학: 서로 다른 스타일과 난이도 분포
- 기하학 같은 특정 도메인은 별도 모듈 개발이 효과적[5][6]
- 하지만 모든 도메인을 커버하는 단일 모델의 한계 인식 필요

**4. 데이터 오염(Data Contamination) 대책:**[1]

- 대규모 사전훈련된 모델이 테스트 데이터를 이미 알 가능성
- 새로운 벤치마크의 지속적 개발 필요
- 평가 시 이중 맹검(double-blind) 원칙 준수 권장

**5. 공정한 비교 기준 수립:**[1]

- 테스트 시간 계산량을 통제한 비교 필수
- 같은 계산 예산(computational budget)에서의 성능 비교
- 테스트 시간 구성 상세 공개

**6. 형식과 비형식의 통합 전략:**[5][6]

- Lean/Isabelle과의 상호작용 자동화
- 형식 증명이 비형식 증명으로 피드백되는 순환 구조 설계
- 신뢰성(formal guarantee)과 적응성(informal flexibility)의 조화

**7. 일반화 성능 추적:**[16][8][1]

- In-domain(IMO 유사 문제) vs. out-of-domain 성능 분리 평가
- 분포 외(OOD) 데이터에 대한 명시적 테스트
- Spurious correlation 탐지 및 제거

### 7. 결론

**DeepSeekMath-V2**는 인공 수학 추론 분야에서 **질적 전환점**을 나타낸다. 최종 답 검증에 국한된 기존 방식을 벗어나 증명의 **논리적 엄밀성**을 직접 평가하고, 모델이 **자신의 추론을 명시적으로 인식**하게 함으로써, 더욱 신뢰할 수 있는 수학 AI 시스템의 가능성을 제시했다.[1]

특히 IMO 2025에서 금메달 수준, Putnam 2024에서 인간 최고 기록 초과라는 성과는 단순히 성능 수치를 넘어, **AI가 형식적 수학 영역에 도입될 수 있는 기초를 마련**했음을 의미한다.[1]

그러나 **검증기 신뢰성의 근본적 한계**, **조합론 문제의 약점**, **형식 증명과의 갭** 등은 여전히 해결할 과제이며, 앞으로의 연구는 다음 방향으로 진행될 것으로 예상된다:[6][5][1]

1. **신경-기호 통합의 심화**: 자연 언어 추론과 형식 검증의 더욱 긴밀한 결합
2. **프로세스 기반 보상의 확대**: 단계별 추론 품질 평가로 일반화 성능 향상
3. **도메인 특화 모듈의 발전**: 기하학, 조합론 등 특정 영역의 혁신적 개선
4. **검증 신뢰도의 형식화**: 확률적 검증에서 형식적 보장으로의 전환

DeepSeekMath-V2의 기여는 **"AI가 증명할 수 있는가?"**라는 질문에 "네, 하지만 신중하게 검증하면서"라는 명확한 답을 제시한 것이며, 이는 수학 연구, 형식 검증, 교육 AI 등 다양한 분야에서 새로운 가능성을 열어줄 것으로 예상된다.[15][5][6][1]

***

### 참고: 주요 수식 모음

**검증 점수 보상:**

$$R_{score}(s'_i, s_i) = 1 - |s'_i - s_i|$$

**검증기 RL 목적:**

$$\max_{\pi_\phi} \mathbb{E}_{(X_i,Y_i,s_i) \sim D_v,(V'_i,s'_i) \sim \pi_\phi(\cdot|X_i,Y_i)} \left[ R_{format}(V'_i) \cdot R_{score}(s'_i, s_i) \right]$$

**Meta-verification 보상:**

$$R_V = R_{format} \cdot R_{score} \cdot R_{meta}$$

**생성기 RL 목적:**

$$\max_{\pi_\theta} \mathbb{E}_{X_i \sim D_p, Y_i \sim \pi_\theta(\cdot|X_i)} [R_Y]$$

**생성기 복합 보상:**

$$R = R_{format}(Y, Z) \cdot (\alpha \cdot R_Y + \beta \cdot R_Z)$$

$$R_Z = R_{score}(s', s) \cdot R_{meta}(Z)$$

여기서 $$\alpha = 0.76$$, $$\beta = 0.24$$

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/75fe392b-6d77-4fee-81f1-024ee89edb74/DeepSeekMath_V2.pdf)
[2](https://arxiv.org/abs/2505.13445)
[3](http://arxiv.org/pdf/2410.15115.pdf)
[4](https://arxiv.org/html/2412.11006v1)
[5](https://www.youtube.com/watch?v=d-N20H22tcM)
[6](https://www.arxiv.org/pdf/2507.23726v1.pdf)
[7](https://arxiv.org/abs/2510.27044)
[8](https://arxiv.org/abs/2510.03865)
[9](https://arxiv.org/abs/2507.15778)
[10](https://www.semanticscholar.org/paper/95e2edef77de43ace03136648fbbcde74a9dcd9a)
[11](https://arxiv.org/abs/2503.23829)
[12](https://arxiv.org/abs/2507.03112)
[13](https://aclanthology.org/2025.findings-naacl.433.pdf)
[14](https://arxiv.org/pdf/2311.05232.pdf)
[15](https://proceedings.mlr.press/v267/dong25h.html)
[16](https://arxiv.org/abs/2506.14245)
[17](https://arxiv.org/abs/2505.19914)
[18](https://arxiv.org/pdf/2407.21009v2.pdf)
[19](http://arxiv.org/pdf/2407.00695v2.pdf)
[20](https://arxiv.org/pdf/2402.06332.pdf)
[21](https://arxiv.org/pdf/2503.10460.pdf)
[22](https://arxiv.org/pdf/2501.00059.pdf)
[23](https://arxiv.org/pdf/2407.04078.pdf)
[24](http://arxiv.org/pdf/2410.12608.pdf)
[25](https://arxiv.org/pdf/2407.08733.pdf)
[26](https://dev.to/czmilo/2025-major-release-how-does-deepseekmath-v2-achieve-self-verifying-mathematical-reasoning-3pje)
[27](https://openreview.net/forum?id=FAe9Gts2Qd)
[28](https://theaiinnovator.com/how-ai-is-transforming-math-the-rise-of-automated-theorem-proving/)
[29](https://indianexpress.com/article/technology/artificial-intelligence/deepseeks-math-v2-ai-model-self-verify-complex-theorems-10390760/)
[30](https://ai.princeton.edu/news/2025/princeton-researchers-unveil-improved-mathematical-theorem-prover-powered-ai)
[31](https://apidog.com/blog/deepseekmath-v2/)
[32](https://www.themoonlight.io/en/review/deeptheorem-advancing-llm-reasoning-for-theorem-proving-through-natural-language-and-reinforcement-learning)
[33](https://sciencemediacentre.es/en/ai-system-could-win-medal-international-mathematical-olympiad-according-study)
[34](https://huggingface.co/deepseek-ai/DeepSeek-Math-V2)
[35](https://arxiv.org/abs/2505.23754)
[36](https://arxiv.org/abs/2509.21128)
[37](https://arxiv.org/pdf/2503.23829.pdf)
[38](http://arxiv.org/pdf/2503.03746.pdf)
[39](http://arxiv.org/pdf/2501.09686.pdf)
[40](https://arxiv.org/html/2403.04642v1)
[41](https://arxiv.org/pdf/2408.15240.pdf)
[42](https://arxiv.org/pdf/2503.12937.pdf)
[43](https://huggingface.co/papers/2506.14245)
[44](https://labelyourdata.com/articles/llm-fine-tuning/llm-hallucination)
[45](https://www.emergentmind.com/topics/reinforcement-learning-with-verified-reward-rlvr)
[46](https://arxiv.org/html/2511.18760v1)
[47](https://labelstud.io/blog/reinforcement-learning-from-verifiable-rewards/)
