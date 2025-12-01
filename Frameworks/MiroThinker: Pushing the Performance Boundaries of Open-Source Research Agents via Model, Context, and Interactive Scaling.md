# MiroThinker: Pushing the Performance Boundaries of Open-Source Research Agents via Model, Context, and Interactive Scaling

### 1. 핵심 주장과 주요 기여 요약

**MiroThinker v1.0**은 오픈소스 기반의 고성능 연구 에이전트로서, 기존의 모델 크기 및 컨텍스트 길이 확장을 넘어 **상호작용 확장(interactive scaling)**이라는 제3의 성능 개선 차원을 도입한 혁신적인 시스템입니다.[1]

주요 기여는 다음과 같습니다:

**핵심 주장:** 연구 능력 개선은 단순히 모델을 더 크게 만들거나 컨텍스트 길이를 늘리는 것만으로는 부족하며, 모델이 **더 깊고 빈번한 에이전트-환경 상호작용을 처리하도록 훈련하는 것**이 중요합니다.[1]

**주요 기여:**
- **상호작용 확장의 체계화**: 강화학습을 통해 모델이 작업당 최대 600개의 도구 호출을 수행하도록 훈련하며, 이는 이전의 오픈소스 모델(<100)과 비교하여 획기적인 발전입니다.[1]
- **벤치마크에서의 최첨단 성능**: GAIA에서 81.9%, HLE에서 37.7%, BrowseComp에서 47.1%, BrowseComp-ZH에서 55.6%의 정확도 달성[1]
- **일반화 가능한 3차원 확장 프레임워크**: 모델 크기, 컨텍스트 길이, 상호작용 깊이 세 가지 차원의 균형잡힌 확장 전략 제시

***

### 2. 해결하고자 하는 문제와 제안하는 방법

#### 2.1 문제 정의

기존의 오픈소스 연구 에이전트들은 다음과 같은 한계를 지닙니다:[1]

1. **성능 격차**: 상용 시스템(ChatGPT Agent, Claude Research)과 현저한 성능 차이
2. **상호작용 깊이의 제한**: 이전 오픈소스 모델들은 작업당 100회 미만의 도구 호출만 지원
3. **확장 전략의 불완전성**: 모델 크기와 컨텍스트만 확장하는 방식의 한계
4. **테스트 타임 스케일링의 위험**: 장사슬 추론에서 성능 저하 가능성

#### 2.2 제안하는 방법

**ReAct 패러다임 기반 아겐틱 워크플로우**

상태 $H_t = \{(T_1, A_1, O_1), \ldots, (T_{t-1}, A_{t-1}, O_{t-1})\}$에서 사고(T), 행동(A), 관찰(O)의 반복 루프를 수행합니다.[1]

사고 생성:
$$T_t = f_\theta(q, \hat{H}_t)$$

행동 정책:
$$A_t = \pi_\theta(\hat{H}_t, T_t)$$

관찰 획득:
$$H_{t+1} = H_t \cup \{(T_t, A_t, O_t)\}$$

**상황 관리 전략: 근본성 기반 컨텍스트 유지**

$K$를 유지 예산으로 정의할 때, 최근 $K$개의 도구 응답을 선택적으로 유지합니다:[1]

$$S_t(K) = \{ i \in \{1, \ldots, t-1\} \mid i \geq t - K \}$$

필터링된 이력:

```math
\hat{H}_t = \left\{\left(T_i, A_i, \hat{O}_i\right)\right\}_{i=1}^{t-1}, \quad \hat{O}_i \triangleq \begin{cases} O_i, & i \in S_t(K) \\ \emptyset, & \text{otherwise} \end{cases}
```

이 전략을 통해 256K 컨텍스트 윈도우 내에서 600개의 도구 호출을 지원할 수 있습니다.[1]

#### 2.3 모델 구조

**세 단계 훈련 파이프라인**

**1단계: 감독 미세조정(SFT)**

전문가 궤적 

```math
 \mathcal{D}_{\text{SFT}}=\{(x_{i},H_{i})\}_{i=1}^{N}
```

에서 학습:

$$L_{\text{SFT}}(\theta) = -\mathbb{E}_{(x,H)} \left[\sum_{t=1}^{T_H} \log \pi_\theta(T_t, A_t | x, H_{ < t})\right]$$

여기서 모델은 사고와 행동 시퀀스를 생성하도록 훈련됩니다.[1]

**2단계: 선호도 최적화(DPO)**

선호 쌍 데이터셋 $\(\mathcal{D}\_{\text{DPO}}=\{(x_{i},H_{i}^{+},H^{-}\_i)\}_{i=1}^{M}\)$ 을 사용하여:

$$L_{\text{DPO}}(x, H^+, H^-) = -\log \sigma\left(\beta \left[(\log \pi_\theta(H^+|x) - \log \pi_\theta(H^-|x)) - (\log \pi_{\text{ref}}(H^+|x) - \log \pi_{\text{ref}}(H^-|x))\right]\right)$$

최종 목적함수:

$$L_{\text{PO}}(\theta) = \mathbb{E}_{(x,H^+,H^-)} [L_{\text{DPO}}(x, H^+, H^-)] + \lambda L^{(+)}_{\text{SFT}}(\theta)$$

여기서 $\beta$는 참조 모델과의 편차를 제어합니다.[1]

**3단계: 강화학습(GRPO)**

$G$개의 궤적을 샘플링하여 그룹 평균에 상대적 이점을 계산합니다:[1]

$$\hat{A}_i = R(x, H_i) - \frac{1}{G} \sum_{j=1}^{G} R(x, H_j)$$

목적함수:

$$L_{\text{GRPO}}(\theta) = \mathbb{E}_{x \sim D} \mathbb{E}_{H \sim \pi_\theta(\cdot|x)} \left[\hat{A}(x, H) \cdot \log \pi_\theta(H|x) - \beta_{\text{KL}} \cdot D_{\text{KL}}(\pi_\theta(\cdot|x) \| \pi_{\text{ref}}(\cdot|x))\right]$$

보상 함수:
$$R(x, H) = \alpha_c R_{\text{correct}}(H) - \alpha_f R_{\text{format}}(H)$$

정확도와 형식 준수를 균형있게 평가합니다.[1]

#### 2.4 도구 인터페이스

MiroThinker는 네 가지 주요 도구 범주를 제공합니다:[1]

| 범주 | 도구 | 기능 |
|------|------|------|
| **실행 환경** | `create_sandbox`, `run_command`, `run_python_code` | 격리된 Linux 샌드박스에서 코드 실행 |
| **파일 관리** | `upload_file_from_local_to_sandbox`, `download_file_from_sandbox_to_local` | 양방향 파일 전송 |
| **정보 검색** | `google_search`, `scrape_and_extract_info` | 웹 검색 및 정보 추출 |
| **컨텍스트 관리** | 근본성 기반 유지, 결과 절단 | 효율적인 메모리 관리 |

***

### 3. 성능 향상 분석

#### 3.1 벤치마크 성능

**절대 성능 달성**[1]

| 벤치마크 | MiroThinker-72B | 이전 최고 기록 | 개선 폭 | 상용 시스템 |
|----------|-----------------|---------------|--------|-----------|
| **GAIA** | 81.9% ± 1.5 | MiniMax-M2 75.7% | +6.2% | GPT-5-high 76.4% |
| **HLE** | 37.7% ± 0.5 | Tongyi 32.9% | +4.8% | GPT-5-high 35.2% |
| **BrowseComp** | 47.1% ± 0.7 | MiniMax-M2 44.0% | +3.1% | ChatGPT-Agent 68.9% |
| **BrowseComp-ZH** | 55.6% ± 1.1 | GLM-4.6 49.5% | +6.1% | OpenAI o3 58.1% |

#### 3.2 상호작용 확장의 효과

강화학습 훈련을 통해 모델의 상호작용 행동이 현저하게 변화합니다.[1]

**SFT vs RL 비교 (MiroThinker-30B)**

| 벤치마크 | SFT 정확도 | RL 정확도 | 개선 | 도구 호출 증가 |
|----------|----------|---------|------|--------------|
| **GAIA** | 65.4% ± 4.0 | 73.5% ± 2.6 | +8.1% | ~100→600회 |
| **BrowseComp** | 32.2% ± 1.1 | 41.2% ± 1.3 | +9.0% | ~100→400회 |
| **BrowseComp-ZH** | 37.6% ± 2.2 | 47.8% ± 1.1 | +10.2% | ~100→350회 |
| **HLE** | 24.1% ± 0.7 | 33.4% ± 0.2 | +9.3% | ~50→250회 |

강화학습이 모델을 더 깊고 체계적인 상호작용으로 유도하면서, 정확도가 8-10% 포인트 향상됩니다.[1]

#### 3.3 스케일 변형 간 성능 차이

모델 크기에 따른 성능 확장성:[1]

| 모델 | GAIA | HLE | BrowseComp | BrowseComp-ZH |
|------|------|-----|-----------|----------------|
| **8B** | 66.4% | 21.5% | 31.1% | 40.2% |
| **30B** | 73.5% | 33.4% | 41.2% | 47.8% |
| **72B** | 81.9% | 37.7% | 47.1% | 55.6% |

***

### 4. 모델의 일반화 성능 향상 가능성

#### 4.1 상호작용 확장의 일반화 효과

**핵심 발견**: 상호작용 깊이는 모델 크기, 컨텍스트 길이와 유사하게 **확장 법칙(scaling laws)**을 따릅니다.[1]

수학적 표현:
$$\text{Performance} = f(\text{Model Size}, \text{Context Length}, \text{Interaction Depth})$$

각 차원이 독립적으로 성능에 기여하며, 이들의 조화로운 확장이 최적의 성능을 제공합니다.[1]

#### 4.2 도메인 간 일반화

**다언어 능력 증진**[1]

- **영문**: GAIA 81.9% (최고 성능)
- **중문**: BrowseComp-ZH 55.6% (오픈소스 최고)
- **교차 언어 전이**: 영문 데이터 중심 훈련에서도 중문 성능이 우수

이는 모델이 언어를 초월한 일반화된 추론 능력을 획득했음을 시사합니다.[1]

#### 4.3 다양한 작업 유형에 대한 확장성

**작업 복잡도별 성능**[1]

MiroThinker는 다음과 같은 다양한 작업 유형에서 일관된 성능 향상을 보입니다:

1. **정보 검색 작업**: GAIA에서 81.9% 달성
2. **다중 문서 추론**: HLE에서 37.7% 달성
3. **웹 네비게이션**: BrowseComp에서 47.1% 달성
4. **교육적 추론**: FRAMES, SEAL-0 등 다양한 벤치마크에서 경쟁력 있는 성능

#### 4.4 테스트 타임 계산 vs 상호작용 확장

**결정적 차이**[1]

테스트 타임 스케일링(test-time scaling)과 상호작용 확장의 근본적 차이:

| 특성 | 테스트 타임 스케일링 | 상호작용 확장 |
|------|------------------|-------------|
| **의존성** | 모델 내부 계산만 | 외부 환경 피드백 활용 |
| **오류 보정** | 제한적 (내부 논리에만 의존) | 강력함 (환경 피드백으로 수정) |
| **장사슬 위험** | 높음 (누적 오류 위험) | 낮음 (검증된 외부 정보) |
| **일반화성** | 제한적 (과적합 위험) | 높음 (다양한 환경 경험) |

상호작용 확장은 환경 피드백과 외부 정보 획득을 활용하여 오류를 수정하고 궤적을 개선합니다.[1]

***

### 5. 논문의 한계

#### 5.1 도구 사용 품질 문제

강화학습 훈련이 에이전트를 더 빈번한 도구 호출로 유도하지만, 일부 호출이 **한계적이거나 중복된 기여**를 합니다. 이는 효율성 개선의 여지를 시사합니다.[1]

#### 5.2 과도한 사고 사슬

강화학습이 **정확도를 위해 더 길고 반복적인 추론 사슬**을 유도하면서:
- 응답 가독성 저하
- 작업 완료 시간 증가
- 사용자 경험 악화

#### 5.3 언어 혼용 문제

비영문 입력(예: 중문)에서 모델이 **영문과 중문을 혼용**하여:
- 중문 성능 최적화 부족
- 출력 일관성 문제

#### 5.4 제한된 샌드박스 능력

모델이 코드 실행 및 파일 관리 도구를 완전히 숙달하지 못하여:
- 샌드박스 타임아웃 발생
- 웹 스크래핑에 코드 실행 도구 오용
- 샌드박스 ID 관리 미숙

***

### 6. 2020년 이후 관련 최신 연구 탐색

#### 6.1 에이전트 기반 재강화학습 진화

**주요 연구 발전**[2][3][4]

- **DeepSeek-R1 (2024)**: 강화학습을 활용한 추론 모델의 확장 법칙 연구
- **Agent Lightning (2024)**: AI 에이전트에 대한 범용 RL 훈련 프레임워크 제시
- **Kimi k1.5 (2025)**: 에이전트 기반의 창의적 문제 해결 능력 개선

연구는 **에이전트 RL의 확장성**이 새로운 성능 경계를 결정하는 핵심임을 강조합니다.[4]

#### 6.2 상호작용 스케일링 이론

**포괄적 프레임워크 개발**[5][6]

최근 연구는 상호작용 스케일링을 체계적으로 분석합니다:

$$\text{Performance} = f(\text{Model Size}, \text{Context Length}, \text{Interaction Depth})$$

**생성-실행-피드백(GEF) 루프** 패러다임이 새로운 표준으로 제시되고 있습니다.[5]

#### 6.3 환경 복잡도 스케일링

**근본적 패러다임 전환**[7][5]

기존: 에이전트 개선에만 집중
현재: **환경의 복잡도, 현실성, 상호작용성** 동시 스케일링

주요 연구:
- **AgentScaler (2024)**: 모의 환경을 통한 도구 호출 능력 훈련
- **ARE (2025)**: 비동기 동적 환경에서의 에이전트 학습
- **PAN (2025)**: 일반적 상호작용 가능한 세계 모델

#### 6.4 테스트 타임 계산 vs 에이전트 상호작용

**명확한 구분 강조**[8][9]

최근 연구는 다음을 명확히 합니다:

1. **테스트 타임 스케일링의 한계**: 고립된 내부 계산에만 의존할 경우, 매개변수의 14배 이상 감소 필요[8]

2. **에이전트 상호작용의 우월성**: 환경 피드백을 활용할 경우 더 효율적인 성능 향상[3]

#### 6.5 일반화 성능 향상 연구

**주요 발견**[10][11][12]

- **다중 환경 노출**: 다양한 환경에서의 학습이 에이전트 일반화 능력을 지수적으로 향상
- **교육적 커리큘럼**: 단순 작업에서 복잡한 작업으로의 진화적 학습이 성능 8-12% 개선
- **멀티태스크 확장 법칙**: 작업 수와 다양성 증가에 따른 예측 가능한 성능 개선

***

### 7. 논문의 앞으로의 영향과 연구 시 고려할 점

#### 7.1 AI 연구의 패러다임 전환

**세 차원 확장 모델의 확립**[6][1]

기존의 2차원 확장(모델 크기 + 컨텍스트)에서 3차원 확장(+ 상호작용 깊이)으로의 전환은:

- **이론적 기여**: 에이전트 성능의 새로운 확장 법칙 제시
- **실무적 의미**: 계산 예산 배분 시 세 가지 차원의 균형 필요성 강조
- **산업 영향**: 미래 에이전트 시스템 설계의 기본 원칙 정립

#### 7.2 개방형 에이전트 연구의 공급 부족 해소

**현실적 기여**[13][1]

MiroThinker의 오픈소스 공개:
- 상용 시스템과의 성능 격차 감소 (GAIA에서 상용 시스템보다 2.5% 우수)
- 커뮤니티 기반 개선 생태계 형성
- 연구 투명성 및 재현성 보장

#### 7.3 강화학습 에이전트 훈련의 표준화

**GRPO 기반 훈련 파이프라인의 일반화**[3][1]

세 단계 훈련 (SFT → DPO → GRPO)이 다양한 에이전트에 적용 가능한 표준으로 확립:

1. **SFT 단계**: 기본 행동 학습
2. **DPO 단계**: 선호도 정렬
3. **GRPO 단계**: 환경 상호작용 최적화

이 파이프라인은 다른 도메인의 에이전트 개발에도 적용될 수 있습니다.[3]

#### 7.4 앞으로의 연구 시 고려할 점

**1. 효율성 개선**

- **도구 호출 품질 최적화**: 600개 호출 중 실질적 기여도가 높은 호출만 선별하는 메커니즘 개발
- **계산 비용 대비 성능**: 더 작은 모델에서 유사한 성능 달성 가능성 탐색
- **에너지 효율성**: 긴 상호작용 궤적의 환경적 비용 고려

**2. 도구 능력 개선**

- **샌드박스 도구 마스터리**: 코드 실행, 파일 관리, 환경 변수 관리의 자동화
- **도구 조합 능력**: 단순 도구 호출이 아닌 도구 간 조합 및 파이프라인 구성
- **적응적 도구 선택**: 작업 특성에 따른 동적 도구 선택 메커니즘

**3. 다언어 및 다문화 능력**

- **언어 혼용 해결**: 비영문 입력에서 순수 출력 언어 유지 메커니즘
- **문화적 맥락 이해**: 지역 특수적 정보 검색 및 해석 능력 강화

**4. 일반화 성능 향상**

- **도메인 전이 학습**: 특정 도메인에서 학습한 에이전트가 새로운 도메인에 적응하는 방식
- **분포 외 견고성(OOD Robustness)**: 훈련 분포를 벗어난 작업에서의 성능 보장
- **지속적 학습**: 새로운 도구 및 작업에 동적으로 적응하는 능력

**5. 환경 복잡도 스케일링**

- **현실적 환경 모델링**: 시뮬레이션을 넘어 실제 웹, API, 데이터베이스와의 상호작용
- **비동기 환경**: 에이전트의 행동과 독립적으로 변화하는 환경 대응
- **다중 에이전트 협업**: 단일 에이전트를 넘어 여러 에이전트 간 협력 메커니즘

**6. 보상 신호의 개선**

- **중간 보상 신호 광화**: 최종 결과뿐 아니라 과정 중 중간 단계의 피드백
- **인간 피드백 통합**: 자동 보상과 인간 평가의 조합
- **온라인 학습**: 배포 후 실시간 피드백을 통한 지속적 개선

**7. 이론적 기초 강화**

- **상호작용 확장의 수학적 분석**: 상호작용 깊이와 성능 간의 엄밀한 스케일링 법칙
- **수렴성 보장**: 강화학습 훈련에서의 수렴 조건 및 샘플 효율성 분석
- **일반화 한계 이론**: 에이전트가 달성할 수 있는 최대 성능의 이론적 한계 규명

#### 7.5 산업 응용 전망

**가까운 미래 (1-2년)**
- 소프트웨어 개발, 데이터 분석, 문헌 검색 자동화
- 기업 문서 분석 및 보고서 자동 생성

**중기 전망 (2-5년)**
- 복잡한 다단계 비즈니스 프로세스 자동화
- 과학 연구 에이전트 (가설 생성, 실험 설계, 결과 분석)
- 실시간 의사결정 지원 시스템

**장기 전망 (5년 이상)**
- 완전 자율 연구 에이전트 (새로운 학문적 발견)
- 멀티 도메인 통합 에이전트 (동시에 다양한 영역 작업 수행)
- 인간-에이전트 공동 창의성 시스템

***

### 8. 결론

**MiroThinker v1.0**은 단순히 성능 벤치마크를 넘는 중요한 패러다임 전환을 제시합니다. 

**상호작용 확장**이라는 제3의 확장 차원의 발견은:

1. **이론적 기여**: 에이전트 지능의 성장이 모델 크기와 컨텍스트만으로 결정되지 않으며, **환경과의 상호작용 깊이**가 동등하게 중요함을 입증
2. **실무적 가치**: 제한된 자원으로도 상용 시스템에 버금가는 성능 달성 가능성 제시
3. **향후 방향 설정**: AI 연구자들에게 에이전트 개발 시 고려해야 할 **균형잡힌 세 차원 확장 전략**을 제안

더욱이 오픈소스 공개를 통해 학술 커뮤니티가 이 성과를 기반으로 지속적인 발전을 도모할 수 있는 기반을 마련했습니다. 

앞으로의 연구는 이러한 상호작용 확장의 이론적 한계, 효율성 개선, 그리고 실제 응용 분야에서의 일반화 성능을 중심으로 전개될 것으로 예상됩니다.

***

### 참고문헌 인덱스

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/157298bc-06ac-4fc8-8db7-2bc3acadff4b/2511.11793v2.pdf)
[2](https://arxiv.org/abs/2502.14499)
[3](https://arxiv.org/html/2508.03680)
[4](https://arxiv.org/html/2509.25300v1)
[5](https://arxiv.org/html/2511.09586v1)
[6](https://www.emergentmind.com/topics/interactive-scaling)
[7](https://openreview.net/pdf/b9a1e052ee84e91b9132c3104c2aa94e8781076a.pdf)
[8](https://openreview.net/forum?id=4FWAwZtd2n)
[9](https://iclr.cc/virtual/2025/oral/31924)
[10](https://datarootlabs.com/blog/state-of-reinforcement-learning-2025)
[11](https://arxiv.org/html/2506.18096v2)
[12](https://www.aryaxai.com/article/top-ai-research-papers-of-2025-from-chain-of-thought-flaws-to-fine-tuned-ai-agents)
[13](https://www.lyzr.ai/blog/agentic-ai-vs-llm/)
[14](http://jair.org/index.php/jair/article/view/15348)
[15](https://misq.umn.edu/misq/article/doi/10.25300/MISQ/2024/17339/3260/RADAR-A-Framework-for-Developing-Adversarially)
[16](https://goldncloudpublications.com/index.php/irjaem/article/view/1235)
[17](https://link.springer.com/10.1007/s10846-024-02064-9)
[18](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/13051/3014099/Learning-behavior-of-offline-reinforcement-learning-agents/10.1117/12.3014099.full)
[19](https://www.semanticscholar.org/paper/066170def5c79fc8f708ebc094c83fb0a0b1aa6d)
[20](https://newjaigs.com/index.php/JAIGS/article/view/397)
[21](https://edintegrity.biomedcentral.com/articles/10.1007/s40979-025-00187-6)
[22](https://ieeexplore.ieee.org/document/10903246/)
[23](https://arxiv.org/pdf/2502.14499.pdf)
[24](https://arxiv.org/pdf/2305.10091.pdf)
[25](https://arxiv.org/html/2412.21088v1)
[26](http://arxiv.org/pdf/2405.14751.pdf)
[27](https://arxiv.org/pdf/2110.05128.pdf)
[28](https://arxiv.org/html/2410.19528v3)
[29](http://arxiv.org/pdf/2303.08115.pdf)
[30](http://arxiv.org/pdf/2307.08962v2.pdf)
[31](https://www.emergentmind.com/topics/agentic-llm-systems)
[32](https://openai.com/index/introducing-deep-research/)
[33](https://www.youtube.com/watch?v=o6SQXXRK07c)
[34](https://stepfun.ai/deep-research-invitation)
[35](https://www.nature.com/articles/s41586-025-09761-x)
[36](https://www.semanticscholar.org/paper/4a5bee77f190dd23125bad78be2676a8489cdc09)
[37](https://www.semanticscholar.org/paper/dbff6cb56b6e23f87fb900eb7ad982020e40dc9e)
[38](https://www.nature.com/articles/s41598-025-29286-7)
[39](https://www.researchprotocols.org/2024/1/e60361)
[40](https://ojs.sciencesforce.com/index.php/nois/article/view/612)
[41](https://ojs.unimal.ac.id/game/article/view/24167)
[42](https://arxiv.org/abs/2509.03771)
[43](https://arxiv.org/abs/2510.18560)
[44](https://www.semanticscholar.org/paper/70ed5e0fb459edd6313be7154c7f0940bd816298)
[45](https://academic-publishing.org/index.php/ejel/article/view/4268)
[46](https://arxiv.org/html/2501.10893v1)
[47](http://arxiv.org/pdf/2406.04151.pdf)
[48](http://arxiv.org/pdf/2411.00114.pdf)
[49](https://arxiv.org/abs/2401.03568)
[50](http://arxiv.org/pdf/2404.10179.pdf)
[51](http://arxiv.org/pdf/2407.17789.pdf)
[52](https://arxiv.org/pdf/2402.02053.pdf)
[53](http://arxiv.org/pdf/2309.01784.pdf)
[54](https://www.deeplearningweekly.com/p/deep-learning-weekly-issue-431)
[55](https://research.aimultiple.com/llm-scaling-laws/)
[56](https://aclanthology.org/2025.acl-short.50.pdf)
[57](https://openreview.net/forum?id=bKymxfD47b)
