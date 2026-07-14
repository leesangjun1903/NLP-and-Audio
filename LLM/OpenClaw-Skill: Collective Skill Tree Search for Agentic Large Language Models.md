# OpenClaw-Skill: Collective Skill Tree Search for Agentic Large Language Models 

---

## ⚠️ 사전 고지

본 논문은 **arXiv:2606.16774v1 (2026년 6월 15일)** 에 게재된 프리프린트입니다. 제공된 PDF 원문을 직접 분석하였으며, 확인되지 않은 내용은 명시적으로 표기합니다.

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

기존 LLM 에이전트 스킬 구성 패러다임은 세 가지 근본적 한계(스킬 단편화, 다양성 부족, 전이 가능성 부족)를 가지며, 이를 **집단 지성 기반 트리 탐색(Collective Skill Tree Search, CSTS)** 으로 해결할 수 있다.

### 주요 기여 4가지

| # | 기여 | 설명 |
|---|------|------|
| 1 | **CSTS 프레임워크** | 트리 탐색을 스킬 구성에 도입, 구조적·다양하고 일반화 가능한 스킬 트리 자동 구축 |
| 2 | **스킬 트리 및 학습 데이터** | OpenClaw 스타일 복잡 태스크를 위한 포괄적 스킬 트리 및 스킬 증강 훈련 데이터 구축 |
| 3 | **CSRL (집단 스킬 강화학습)** | 다수의 스킬 조건부 롤아웃 그룹을 비교하여 정책 최적화 |
| 4 | **OpenClaw-Skill 모델** | 장기 계획, 도구 사용, 오류 복구, 교차 태스크 일반화에서 우수한 성능 달성 |

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

논문은 OpenClaw와 같은 실세계 인터랙티브 환경에서 LLM 에이전트가 복잡한 장기 태스크를 수행하기 위한 **스킬 자동 구성**의 세 가지 핵심 문제를 제기합니다.

```
문제 (1): Skill Fragmentation (스킬 단편화)
  → 개별 서브태스크의 로컬 절차만 포착, 스킬 간 의존성/순서 관리 부재

문제 (2): Limited Skill Diversity (스킬 다양성 부족)
  → 단일 모델의 궤적에서만 스킬 추출 → 해당 모델의 편향 내재

문제 (3): Limited Skill Transferability (전이 가능성 부족)
  → 특정 LLM 백본에서 추출한 스킬이 다른 모델에 적용 시 성능 급락
```

---

### 2.2 제안 방법 및 수식

#### 2.2.1 복잡 태스크 분해 (Complex Task Decomposition)

복잡 태스크 $T$를 순서가 있는 서브태스크 시퀀스로 분해:

$$T \rightarrow (t_1, t_2, \ldots, t_M)$$

트리의 $m$번째 레이어가 서브태스크 $t_m$에 대응합니다.

---

#### 2.2.2 CSN-Gen: 집단 스킬 노드 생성

참여 모델 집합:

$$\mathcal{M} = \{M_1, M_2, \ldots, M_N\}$$

각 모델 $M_n$이 서브태스크 $t_m$에 대해 생성하는 실행 궤적:

$$\tau_{m,n} = \pi_{\theta_n}(\cdot \mid t_m) = \left(t_m,\ \{(\psi^\ell_{m,n},\ a^\ell_{m,n},\ o^\ell_{m,n})\}^{L_{m,n}}_{\ell=1},\ r_{m,n}\right)$$

- $\psi^\ell_{m,n}$: $\ell$ 스텝에서의 추론 상태
- $a^\ell_{m,n}$: 에이전트 행동 (도구 호출, 코드 실행 등)
- $o^\ell_{m,n}$: 관찰/실행 피드백
- $r_{m,n}$: 최종 실행 결과 스칼라

공유 스킬 합성기 $\Phi_{\text{skill}}$을 통해 궤적을 스킬 노드로 변환:

$$s_{m,n} = \Phi_{\text{skill}}(t_m, \tau_{m,n})$$

서브태스크 $t_m$에 대한 후보 스킬 집합:

$$\mathcal{S}_m = \{s_{m,1}, s_{m,2}, \ldots, s_{m,N}\}$$

---

#### 2.2.3 CSN-Assess: 집단 스킬 노드 평가

**① 집단 품질 점수 (Collective Quality Score)**

판사 모델 $j$가 스킬 $s_{m,n}$에 부여하는 점수 $q^j_{m,n}$의 평균:

$$Q_{m,n} = \frac{1}{J} \sum_{j=1}^{J} q^j_{m,n}$$

**② 집단 전이 가능성 점수 (Collective Transferability Score)**

스킬 $s_{m,n}$을 다른 모델 $M_k\ (k \neq n)$에 적용한 스킬 조건부 롤아웃:

$$\tilde{\tau}^k_{m,n} \sim \pi_{\theta_k}(\cdot \mid t_m, s_{m,n}), \quad k \neq n$$

전이 가능성 점수:

$$\text{Tran}_{m,n} = \frac{1}{N-1} \sum_{\substack{k=1 \\ k \neq n}}^{N} r^k_{m,n}$$

**③ 최종 스킬 노드 점수**

$$\text{Score}(s_{m,n}) = Q_{m,n} + \text{Tran}_{m,n}$$

**④ 최적 스킬 노드 선택**

$$s^\star_m = \arg\max_{s_{m,n} \in \mathcal{S}_m} \text{Score}(s_{m,n})$$

**⑤ 선택된 스킬 경로 (Compositional Skill Path)**

$$S^\star_T = (s^\star_1, s^\star_2, \ldots, s^\star_M)$$

---

#### 2.2.4 SFT (Supervised Fine-Tuning)

각 태스크에 대한 SFT 인스턴스:

$$(T, S^\star_T, \tau^\star_T) \in \mathcal{D}_{\text{SFT}}$$

SFT 목적함수:

$$\mathcal{L}_{\text{SFT}}(\theta) = -\mathbb{E}_{(T, S^\star_T, \tau^\star_T) \sim \mathcal{D}_{\text{SFT}}} \log \pi_\theta(\tau^\star_T \mid T, S^\star_T)$$

---

#### 2.2.5 CSRL: 집단 스킬 강화학습

서브태스크 $t_m$에 대해 후보 스킬 $s_{m,n}$ 각각으로 $G$개의 롤아웃 샘플링:

$$\tau^g_{m,n} \sim \pi_{\theta_{\text{old}}}(\cdot \mid t_m, s_{m,n}), \quad g = 1, \ldots, G$$

집단 스킬 조건부 그룹:

$$\mathcal{B}_m = \{\tau^g_{m,n} \mid s_{m,n} \in \mathcal{S}_m,\ g = 1, \ldots, G\}$$

**교차 스킬 정규화 어드밴티지:**

$$A^g_{m,n} = \frac{r^g_{m,n} - \mu_m}{\sigma_m + \delta}$$

$$\mu_m = \frac{1}{|\mathcal{B}_m|} \sum_{\tau^g_{m,n} \in \mathcal{B}_m} r^g_{m,n}, \quad \sigma_m = \text{Std}\{r^g_{m,n} \mid \tau^g_{m,n} \in \mathcal{B}_m\}$$

**GRPO 스타일 클리핑 서로게이트:**

확률 비율:

$$\rho^g_{m,n,\ell}(\theta) = \frac{\pi_\theta(a^g_{m,n,\ell} \mid t_m, s_{m,n}, h^g_{m,n,<\ell})}{\pi_{\theta_{\text{old}}}(a^g_{m,n,\ell} \mid t_m, s_{m,n}, h^g_{m,n,<\ell})}$$

클리핑 서로게이트 항:

$$u^g_{m,n,\ell}(\theta) = \min\!\left(\rho^g_{m,n,\ell}(\theta) A^g_{m,n},\ \text{clip}(\rho^g_{m,n,\ell}(\theta), 1-\epsilon, 1+\epsilon) A^g_{m,n}\right)$$

**CSRL 최종 손실함수:**

$$\mathcal{L}_{\text{CSRL}}(\theta) = -\mathbb{E}_{t_m, \mathcal{B}_m} \left[\frac{1}{|\mathcal{B}_m|} \sum_{\tau^g_{m,n} \in \mathcal{B}_m} \frac{1}{L^g_{m,n}} \sum_{\ell=1}^{L^g_{m,n}} u^g_{m,n,\ell}(\theta)\right]$$

---

### 2.3 모델 구조

```
OpenClaw-Skill 전체 파이프라인:

[복잡 태스크 T]
       ↓ 분해
[서브태스크 시퀀스: t₁, t₂, ..., tₘ]
       ↓ 각 서브태스크마다 반복
┌─────────────────────────────────┐
│  CSN-Gen: 다수 모델 병렬 롤아웃 │
│  → 후보 스킬 집합 Sₘ 생성       │
├─────────────────────────────────┤
│  CSN-Assess:                    │
│  ① 집단 품질 평가 (Qₘ,ₙ)       │
│  ② 전이가능성 평가 (Tranₘ,ₙ)   │
│  → 최적 스킬 s*ₘ 선택           │
└─────────────────────────────────┘
       ↓ 스킬 트리 구성
[스킬 경로 S*_T = (s*₁, s*₂, ..., s*ₘ)]
       ↓
[SFT: 스킬 증강 데이터로 기본 훈련]
       ↓
[CSRL: 다수 스킬 조건부 롤아웃 비교 → 정책 최적화]
       ↓
[OpenClaw-Skill 최종 모델]
```

**백본 모델:** Qwen3-4B, Qwen3-8B, Qwen3.5-4B, Qwen3.5-9B

**학습 설정:**
- SFT 데이터: 2K 고품질 스킬 증강 예시
- 학습 에폭: 2
- 하드웨어: 8× H100 GPU
- 학습률: $5 \times 10^{-6}$

---

### 2.4 성능 향상

#### QwenClawBench 결과

| 모델 | Overall | 향상 |
|------|---------|------|
| Qwen3-4B → OpenClaw-Skill-Qwen3-4B | 7.0 → 12.8 | **+5.8** |
| Qwen3-8B → OpenClaw-Skill-Qwen3-8B | 11.5 → 15.8 | **+4.3** |
| Qwen3.5-4B → OpenClaw-Skill-4B | 31.5 → 41.2 | **+9.7** |
| Qwen3.5-9B → OpenClaw-Skill-9B | 34.5 → 44.9 | **+10.4** |

특히 장기 도구 사용 범주에서 두드러진 향상:
- SVM: 33.2 → 70.9 (+37.7)
- CS: 30.2 → 78.4 (+48.2)
- RIR: 24.4 → 54.1 (+29.7)

#### PinchBench 결과 (123-task)

| 모델 | Best(%) | Average(%) |
|------|---------|------------|
| Qwen3.5-9B | 61.1 | 47.1 |
| OpenClaw-Skill-9B | **68.2** | **53.6** |
| Qwen3-4B | 22.4 | 13.6 |
| OpenClaw-Skill-Qwen3-4B | **31.1** | **20.8** |

#### Ablation Study (Qwen3.5-9B 기준)

| 설정 | Overall |
|------|---------|
| Base | 34.5 |
| + CSN-Gen | 39.8 (+5.3) |
| + CSN-Gen + CSN-Assess | 42.8 (+3.0) |
| + CSRL (OpenClaw-Skill) | **44.9 (+2.1)** |

---

### 2.5 한계

논문에서 명시적으로 언급된 한계 및 관찰 가능한 제약:

1. **계산 비용**: 다수 모델의 병렬 롤아웃 + 집단 평가 → 스킬 구성 단계에서 높은 추론 비용 발생
2. **소규모 SFT 데이터**: 2K 예시만 사용 — 더 다양한 태스크로의 확장 시 한계 가능성
3. **벤치마크 제한**: QwenClawBench, PinchBench 두 벤치마크에만 평가 — 더 광범위한 도메인 검증 부재
4. **스킬 수 제한**: 각 서브태스크당 $N$개 모델의 스킬만 탐색 — 트리 폭의 한계
5. **소형 모델 한계**: 4B, 8B 모델은 여전히 대형 클로즈드소스 모델(Claude Opus 4.6: 59.5점)에 비해 성능 격차 존재

---

## 3. 일반화 성능 향상 가능성 (중점 분석)

### 3.1 일반화를 위한 세 가지 설계 원칙

#### 원칙 1: 집단 전이가능성 점수를 통한 명시적 일반화 강제

전이 가능성 점수 $\text{Tran}_{m,n}$은 **"한 모델이 생성한 스킬이 다른 모델들을 도울 수 있는가"** 를 직접 측정합니다.

$$\text{Tran}_{m,n} = \frac{1}{N-1} \sum_{\substack{k=1 \\ k \neq n}}^{N} r^k_{m,n}$$

이 메커니즘의 의의:
- 특정 모델의 추론 스타일에만 최적화된 스킬은 낮은 전이 점수를 받아 **자동으로 필터링**
- 모델에 구애받지 않는(model-agnostic) 절차적 지식만 선발 → **백본 LLM 독립적 스킬 구성**

#### 원칙 2: 다양한 백본 모델 활용으로 단일 모델 편향 제거

CSN-Gen에서 $N$개의 이종(heterogeneous) 모델을 사용:

$$\mathcal{M} = \{M_1, M_2, \ldots, M_N\}$$

- 각 모델은 서로 다른 실패 모드, 검증 전략, 추론 경로를 탐색
- 이로 인해 스킬 공간이 특정 모델의 사전 분포에 수렴하지 않고 **광범위한 행동 패턴을 커버**
- 단일 모델 편향(single-model bias) 감소 → 미학습 태스크 유형에 대한 일반화 향상

#### 원칙 3: CSRL의 교차 스킬 어드밴티지를 통한 적응적 일반화

기존 GRPO는 동일 스킬 내 롤아웃끼리만 비교하지만, CSRL은 **서브태스크 전체 그룹** $\mathcal{B}_m$ 내에서 정규화:

$$A^g_{m,n} = \frac{r^g_{m,n} - \mu_m}{\sigma_m + \delta}$$

이 설계의 일반화 기여:
- 단일 스킬에 과적합(overfitting)되는 현상 방지
- 다양한 스킬 중 **상황에 따라 가장 효과적인 전략을 동적으로 선택**하는 능력 학습
- 동질적(homogeneous) 해결 방식으로의 수렴 방지

### 3.2 일반화 성능의 실험적 근거

**모델 크기 일반화**: 4B~9B 모든 크기에서 일관된 향상 확인

**태스크 유형 일반화**: WAO(웹 자동화), SOA(OS 자동화), KMM(지식 관리), SVM(서비스 관리), CS(코드 실행) 등 다양한 범주에서 향상

**벤치마크 일반화**: QwenClawBench와 PinchBench 두 독립적 벤치마크에서 모두 향상 확인

```
일반화 한계 분석:
- 오픈소스 소형 모델 (4B~9B) vs 대형 클로즈드소스 (Claude Opus: 59.5)
  → 여전히 성능 격차 존재 (OpenClaw-Skill 9B: 44.9)
- 단, 상대적 향상률(Relative Improvement)은 소형 모델에서 더 크게 관찰됨
```

---

## 4. 관련 최신 연구 비교 분석 (2020년 이후)

### 4.1 스킬 기반 LLM 에이전트 연구 비교

| 연구 | 스킬 구성 방식 | 다양성 | 구조화 | 전이성 |
|------|--------------|-------|-------|-------|
| **Voyager (Wang et al., 2023)** [arXiv:2305.16291] | 자동 스킬 생성 (Minecraft) | 제한적 | 코드 라이브러리 | 단일 환경 |
| **ReAct (Yao et al., 2023)** [ICLR 2023] | 추론-행동 인터리빙 | 없음 | 없음 | 제한적 |
| **SkillRL (Xia et al., 2026)** [arXiv:2602.08234] | RL 기반 스킬 뱅크 | 단일 모델 | 플랫 | 미검증 |
| **Trace2Skill (Ni et al., 2026)** [arXiv:2603.25158] | 궤적 로컬 패턴 증류 | 단일 모델 | 플랫 | 제한적 |
| **CoEvoSkills (Zhang et al., 2026)** [arXiv:2604.01687] | 공진화 검증 | 단일 모델 | 플랫 | 제한적 |
| **EvoSkill (Alzubi et al., 2026)** [arXiv:2603.02766] | 멀티에이전트 스킬 발견 | 멀티에이전트 | 플랫 | 부분적 |
| **OpenClaw-Skill (본 논문)** | 집단 트리 탐색 | **다수 모델** | **트리 구조** | **명시적 검증** |

### 4.2 트리 탐색 기반 LLM 연구 비교

| 연구 | 트리 탐색 적용 대상 | 핵심 기여 |
|------|------------------|---------|
| **Tree of Thoughts (Yao et al., NeurIPS 2023)** [arXiv:2305.10601] | 추론 경로 탐색 | 복수 추론 경로 병렬 탐색 |
| **LATS (Zhou et al., ICML 2024)** [PMLR 2024] | 추론+행동+계획 통합 | MCTS와 언어 에이전트 결합 |
| **Tree Search for LM Agents (Koh et al., TMLR 2025)** | 웹 환경 행동 궤적 | 현실적 웹 환경 성공률 향상 |
| **Mulberry (Yao et al., 2024)** [arXiv:2412.18319] | MLLM 추론 | 집단 MCTS로 o1 스타일 추론 |
| **OpenClaw-Skill (본 논문)** | **스킬 구성** | **트리 탐색 → 스킬 자동 구축** |

> **핵심 차별점**: 기존 트리 탐색 연구들이 *추론 경로* 탐색에 집중한 반면, 본 논문은 트리 탐색을 *재사용 가능한 스킬 구성* 에 처음으로 적용합니다.

### 4.3 강화학습 기반 LLM 에이전트 비교

| 연구 | RL 방법 | 스킬 활용 |
|------|---------|---------|
| **DeepSeek-R1 (2025)** [arXiv:2501.12948] | GRPO | 없음 |
| **SkillRL (2026)** | 스킬 뱅크 + RL | 단일 스킬 조건 |
| **OpenClaw-Skill** | **CSRL (집단 스킬 GRPO)** | **다수 스킬 교차 비교** |

---

## 5. 앞으로의 연구에 미치는 영향 및 고려 사항

### 5.1 앞으로의 연구에 미치는 영향

#### ① 스킬 구성 패러다임의 전환

CSTS는 스킬을 **단순 경험 증류**에서 **구조화된 집단 탐색**으로 전환합니다. 향후 스킬 기반 에이전트 연구는:
- 스킬의 계층적 조직화 (스킬 트리 vs 스킬 뱅크) 를 표준 설계 요소로 고려할 가능성
- 전이 가능성을 스킬 품질의 독립적 평가 지표로 채택할 가능성

#### ② 집단 지성의 LLM 훈련 파이프라인 통합

다수 모델의 집단 평가를 **데이터 품질 필터링**에 활용하는 접근법은:
- 단일 모델 기반 데이터 생성의 한계를 넘어선 새로운 데이터 플라이휠(data flywheel) 구축 가능성 제시
- 특히 라벨 없는 실세계 에이전트 태스크에서 자동 품질 평가 메커니즘으로 확장 가능

#### ③ 강화학습의 스킬 인식(Skill-Aware) 정책 최적화

CSRL의 교차 스킬 어드밴티지 계산은:
- 기존 GRPO/PPO의 자연스러운 확장으로 다른 에이전트 도메인에 적용 가능
- 다중 전략 비교 학습을 통한 정책 견고성 향상 연구의 출발점 제공

### 5.2 앞으로 연구 시 고려할 점

#### 기술적 고려사항

**① 계산 효율성 최적화**
- 현재 $N$개 모델 × $M$개 서브태스크 × 전이성 검증의 조합적 복잡도
- 경량 대리 모델(surrogate model)을 활용한 전이성 추정 연구 필요
- 점진적 스킬 트리 업데이트(incremental update) 메커니즘 개발

**② 스킬 트리 구조 최적화**
- 현재 선형 분해 $T \rightarrow (t_1, \ldots, t_M)$ 를 DAG(방향 비순환 그래프) 구조로 확장
- 동적 트리 재구성(adaptive tree restructuring) 연구

**③ 보상 함수 설계**

$$r^g_{m,n} = R(t_m, s_{m,n}, \tau^g_{m,n})$$

현재 보상 함수의 정의가 구체적으로 기술되지 않아, 다양한 보상 신호(execution feedback, partial credit 등)의 효과 비교 연구 필요

**④ 스킬 망각(Skill Forgetting) 문제**
- 지속 학습(continual learning) 시나리오에서 스킬 트리 갱신 시 기존 스킬 퇴화 방지 메커니즘

#### 방법론적 고려사항

**⑤ 더 광범위한 평가 벤치마크**
- QwenClawBench, PinchBench 외 OSWorld, WebArena 등 기존 벤치마크와의 비교
- 도메인 외(out-of-domain) 일반화 테스트 강화

**⑥ 스킬 해석 가능성**
- 구성된 스킬 경로 $S^\star_T$의 인간 해석 가능성 평가
- 스킬 재사용 빈도 분석을 통한 중요 스킬 식별

**⑦ 멀티모달 환경 확장**
- 현재 텍스트 기반 도구 사용에 집중
- 비전-언어 통합 에이전트 환경(GUI 조작, 이미지 기반 검증)으로의 확장

**⑧ 스킬 보안 및 안전성**
- 자동 구성된 스킬이 의도치 않은 행동을 포함할 위험성
- 스킬 검증 단계에서의 안전성 필터링 메커니즘 필요

---

## 📚 참고자료 (출처)

본 분석은 다음 자료를 직접 참조하였습니다:

**주 분석 대상:**
- **OpenClaw-Skill 논문 원문**: Tianyi Lin et al., "OpenClaw-Skill: Collective Skill Tree Search for Agentic Large Language Models," arXiv:2606.16774v1, 2026년 6월 15일

**논문 내 인용 자료 (관련성 높은 것):**
- [6] Jing Yu Koh et al., "Tree search for language model agents," *Transactions on Machine Learning Research*, 2025
- [7] Shunyu Yao et al., "ReAct: Synergizing reasoning and acting in language models," *ICLR 2023*
- [9] Peng Xia et al., "SkillRL: Evolving agents via recursive skill-augmented reinforcement learning," arXiv:2602.08234, 2026
- [14] Hanrong Zhang et al., "CoEvoSkills: Self-evolving agent skills via co-evolutionary verification," arXiv:2604.01687, 2026
- [16] Jingwei Ni et al., "Trace2Skill: Distill trajectory-local lessons into transferable agent skills," arXiv:2603.25158, 2026
- [38] Guanzhi Wang et al., "Voyager: An open-ended embodied agent with large language models," arXiv:2305.16291, 2023
- [44] Shunyu Yao et al., "Tree of Thoughts: Deliberate problem solving with large language models," *NeurIPS 2023*
- [45] Huanjin Yao et al., "Mulberry: Empowering MLLM with o1-like reasoning and reflection via collective Monte Carlo tree search," arXiv:2412.18319, 2024
- [46] Andy Zhou et al., "Language agent tree search unifies reasoning, acting, and planning in language models," *ICML 2024*
- [20] DeepSeek-AI, "DeepSeek-R1: Incentivizing reasoning capability in LLMs via reinforcement learning," arXiv:2501.12948, 2025
