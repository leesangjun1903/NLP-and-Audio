# Harness Updating Is Not Harness Benefit: Disentangling Evolution Capabilities in Self-Evolving LLM Agents

> **주의**: 본 답변은 첨부된 논문 PDF(arXiv:2605.30621v1)를 직접 분석한 내용에 기반합니다. 2020년 이후 비교 연구 일부는 논문 내 인용 문헌을 기반으로 서술하며, 논문에 명시되지 않은 외부 정보는 별도 표기합니다.

---

## 1. 핵심 주장과 주요 기여 요약

### 핵심 주장

본 논문은 **하네스 자기진화(harness self-evolution)** 시스템에서 흔히 혼동되는 두 가지 능력을 명확히 분리(disentangle)합니다:

| 능력 | 정의 | 담당 역할 |
|------|------|-----------|
| **Harness-Updating** | 실행 증거로부터 유용한 하네스 업데이트를 생성하는 능력 | Evolver (진화자) |
| **Harness-Benefit** | 업데이트된 하네스로부터 실제 이득을 얻는 능력 | Task-Solving Agent |

**두 가지 핵심 발견:**

1. **Harness-Updating은 기반 능력에 대해 평탄(flat)하다**: Qwen3.5-9B 같은 소형 모델이 생성한 하네스 업데이트도 Claude Opus 4.6 수준의 하류(downstream) 성능 향상을 유도한다.

2. **Harness-Benefit은 기반 능력에 대해 비단조적(non-monotonic)이다**: 중간 능력 모델(GPT-OSS-120B)이 가장 많은 이득을 얻고, 약한 모델은 두 가지 실패 모드(활성화 실패, 준수 실패)로 인해 이득이 적으며, 강한 모델은 성능 천장(ceiling) 효과로 이득이 제한된다.

### 주요 기여

- **개념적 기여**: Harness-Updating과 Harness-Benefit을 최초로 공식적으로 분리 정의
- **실증적 기여**: 7개 LLM × 3개 벤치마크(SWE-bench, MCP-Atlas, SkillsBench)의 교차 실험
- **실용적 기여**: 능력 예산을 Evolver가 아닌 Task-Solving Agent에 투자해야 한다는 설계 지침 제시
- **진단적 기여**: 약한 모델의 낮은 Harness-Benefit을 설명하는 두 실패 모드(Harness Activation Failure, Harness Adherence Failure) 식별

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

**문제 배경**: 기존 자기진화 에이전트 연구들은 종단간(end-to-end) 성능 향상만 측정하므로, 성능 향상의 원천이 다음 세 가지 중 어디서 오는지 불분명합니다:

1. 에이전트의 기반 능력
2. Evolver의 하네스 업데이트 품질
3. 에이전트의 업데이트 하네스 활용 능력

**구체적 연구 질문**:
- 어떤 모델이 유용한 하네스 업데이트를 생성하는가?
- 어떤 모델이 업데이트된 하네스로부터 가장 많이 이득을 얻는가?

---

### 2.2 제안하는 방법 및 수식

#### 에이전트 하네스 정의

진화 단계 $t$에서 LLM 에이전트는 다음과 같이 정의됩니다:

$$A_t = (f, H_t) \tag{1}$$

여기서 $f$는 고정된 모델 백본(frozen model backbone), $H_t$는 $t$ 단계 이후의 하네스 상태입니다.

#### Evolver 업데이트 과정

이전 하네스 $H_{t-1}$과 누적 실행 증거 $\mathcal{D}_t$가 주어질 때, evolver $e$는:

$$\Delta H_t = e(H_{t-1}, \mathcal{D}_t)$$

$$H_t = \text{Apply}(H_{t-1}, \Delta H_t) \tag{2}$$

#### 태스크 수행 및 증거 수집

에이전트 $A_{t-1}$이 태스크 배치 $\mathcal{X}_t$에서 각 태스크 $x \in \mathcal{X}_t$를 수행:

$$(\tau_{t,x}, y_{t,x}) = \text{Solve}(A_{t-1}, x) \tag{3}$$

실행 증거:

$$\mathcal{D}_t = \{(x, \tau_{t,x}, y_{t,x}) : x \in \mathcal{X}_t\} \tag{4}$$

#### 기반 능력 정의

전체 태스크 집합 $\mathcal{X} = \bigcup_{t=1}^{T} \mathcal{X}_t$에서 모델 $f$의 기반 능력:

$$M_{\text{base}}(f) = J_\mathcal{X}(f, H_0) \tag{5}$$

$J_\mathcal{X}(f, H)$는 에이전트 $(f, H)$의 $\mathcal{X}$에서의 성능을 측정하는 채점 함수입니다.

#### 쌍별 진화 이득 (Pairwise Evolution Gain)

특정 에이전트-evolver 쌍 $(f, e)$에 대해 $H_T^{(f,e)}$를 최종 하네스라 할 때:

$$\Delta(f, e) = J_\mathcal{X}(f, H_T^{(f,e)}) - M_{\text{base}}(f) \tag{6}$$

#### Harness-Updating 능력

앵커 에이전트 집합 $\mathcal{F}^\star$에서 evolver $e$의 평균 쌍별 이득:

$$\Delta_{\text{update}}(e) = \frac{1}{|\mathcal{F}^\star|} \sum_{f \in \mathcal{F}^\star} \Delta(f, e) \tag{7}$$

#### Harness-Benefit 능력

앵커 evolver 집합 $\mathcal{E}^\star$에서 모델 $f$의 최대 쌍별 이득:

$$\Delta_{\text{benefit}}(f) = \max_{e \in \mathcal{E}^\star} \Delta(f, e) \tag{8}$$

#### Harness-Following Rate (HFR)

모델 $f$에 대해 $N_f^{\text{load}}$: 스킬이 로드된 궤적 수, $N_f^{\text{follow}}$: 스킬을 따른 궤적 수일 때:

$$\text{HFR}(f) = \frac{N_f^{\text{follow}}}{N_f^{\text{load}}}$$

---

### 2.3 모델 구조

논문은 새로운 신경망 아키텍처를 제안하지 않고, **하네스 자기진화 프레임워크**의 분석 설계를 제시합니다.

#### 실험 설계 구조

```
┌──────────────────────────────────────────┐
│           Harness Self-Evolution         │
│                                          │
│  ┌─────────────┐    증거     ┌─────────┐ │
│  │Task-Solving │ ─────────► │ Evolver │ │
│  │   Agent     │            │ (LLM)   │ │
│  │  (frozen f) │ ◄───────── └─────────┘ │
│  └─────────────┘  업데이트된 하네스       │
│         ▲                                │
│    H_t (prompts/skills/memory/tools)     │
└──────────────────────────────────────────┘
```

#### 평가 벤치마크

| 벤치마크 | 태스크 수 | 도메인 수 | 평가 내용 |
|---------|---------|---------|---------|
| SWE-bench Verified | 500 | 12 | 소프트웨어 엔지니어링 (GitHub 이슈 해결) |
| MCP-Atlas | 500 | 36 | 실제 MCP 서버 도구 사용 |
| SkillsBench | 86 | 11 | 스킬 기반 다양한 태스크 수행 |

#### 실험에 사용된 모델 (7개)

| 능력 계층 | 모델 |
|---------|------|
| Strong | Claude Opus 4.6, Claude Sonnet 4.6 |
| Mid | GPT-OSS-120B, Claude Haiku 4.5, Qwen3-235B |
| Weak | Qwen3-32B, Qwen3.5-9B (evolver only) |

---

### 2.4 성능 향상 결과

#### Harness-Updating 결과 (Figure 3, Table 5)

| Evolver | SWE $\Delta_{\text{update}}$ (pp) | MCP $\Delta_{\text{update}}$ (pp) | SB $\Delta_{\text{update}}$ (pp) |
|---------|----------------------------------|----------------------------------|----------------------------------|
| Opus 4.6 | 7.4 | **3.6** | 2.3 |
| Sonnet 4.6 | 7.4 | 2.6 | 1.2 |
| Haiku 4.5 | 8.0 | 2.3 | 2.7 |
| Qwen3-235B | **8.2** | 0.6 | 1.5 |
| Qwen3-32B | 7.8 | 2.3 | 0.7 |
| Qwen3.5-9B | 6.8 | 1.0 | **3.8** |
| GPT-OSS-120B | 5.9 | 1.9 | 1.5 |

**핵심 관찰**: 최고-최저 evolver 간 격차는 어떤 벤치마크에서도 최대 **3.1 pp** 이내이며, 단일 evolver가 모든 벤치마크에서 우위를 점하지 않습니다.

#### Harness-Benefit 결과 (Table 1, 7)

| 모델 | SWE Base (%) | SWE $\Delta_{\text{benefit}}$ (pp) | MCP Base (%) | MCP $\Delta_{\text{benefit}}$ (pp) | SB Base (%) | SB $\Delta_{\text{benefit}}$ (pp) |
|------|-------------|-----------------------------------|--------------|------------------------------------|-------------|----------------------------------|
| Qwen3-32B | 3.6 | 4.4 | 3.6 | 1.0 | 0.0 | 5.8 |
| Qwen3-235B | 20.7 | **19.3** | 25.0 | 4.3 | 4.7 | 1.1 |
| GPT-OSS-120B | 26.2 | 15.8 | 28.0 | **7.0** | 0.0 | 7.0 |
| Haiku 4.5 | 66.0 | 2.4 | 42.4 | 3.6 | 5.8 | **15.1** |
| Sonnet 4.6 | 73.2 | 2.8 | 54.0 | 3.2 | 24.4 | 3.5 |
| Opus 4.6 | 74.2 | 2.6 | 61.0 | 3.6 | 25.6 | 5.8 |

**핵심 관찰**: $\Delta_{\text{benefit}}$는 기반 능력과 단조적으로 증가하지 않으며, 중간 계층에서 최대치를 보입니다.

#### 위상별 준수(Adherence) 분석 (Table 3)

| 궤적 단계 | Qwen3-32B (Weak) | GPT-OSS-120B (Mid) | Opus 4.6 (Strong) |
|----------|-----------------|-------------------|------------------|
| 하네스 로드 후 | 0.52 | 0.67 | **0.89** |
| 중간 턴 | 0.22 | 0.48 | 0.79 |
| 최종 턴 | 0.13 | 0.43 | 0.80 |
| **드리프트 (로드→최종)** | **-0.39** | -0.24 | **-0.09** |

약한 모델은 궤적이 진행될수록 하네스 준수가 급격히 저하됩니다.

#### 약한 모델의 두 가지 실패 모드 (Table 2)

| 모델 | SLR (스킬 로드율) | HFR (하네스 준수율) | LPR (로드 후 성공률) |
|------|-----------------|------------------|-------------------|
| Qwen3-32B | 0.251 | 0.142 | 0.023 |
| GPT-OSS-120B | 0.446 | 0.442 | 0.040 |
| Haiku 4.5 | 0.794 | 0.600 | 0.099 |
| Qwen3-235B | **0.961** | 0.350 | 0.022 |
| Sonnet 4.6 | 0.959 | 0.730 | 0.145 |
| Opus 4.6 | 0.957 | **0.757** | **0.177** |

**Qwen3-235B**는 스킬 로드율(0.961)은 높지만 HFR(0.350)이 낮아, **로드 성공 ≠ 준수 성공**임을 명확히 보여줍니다.

---

### 2.5 한계

논문 Section 6에서 명시적으로 다음을 한계로 인정합니다:

1. **파라메트릭 파인튜닝 미포함**: 모델 가중치를 수정하는 파인튜닝, RL, 하이브리드 적응 방법은 평가하지 않음
2. **모델 집합의 제한성**: 7개 LLM만 사용하여 모델 패밀리, 스케일, 훈련 레시피에 따른 세밀한 차이를 완전히 파악하기 어려움
3. **안전성 우려**: 하네스 자기진화는 잘못된 교훈, 편향된 지침, 민감 정보가 하네스에 기록될 수 있는 위험 존재
4. **벤치마크 특수성**: SkillsBench의 저기반 구간에서 비단조 패턴이 더 가변적으로 나타남

---

## 3. 일반화 성능 향상 가능성

본 논문의 발견들은 일반화 성능 향상과 직접적으로 관련된 중요한 통찰을 제공합니다.

### 3.1 Harness-Updating의 일반화 가능성

**핵심 발견**: 소형 모델(Qwen3.5-9B)의 evolver도 대형 모델(Claude Opus 4.6)과 동등한 품질의 하네스 업데이트를 생성할 수 있습니다.

**일반화 관점 의의**:

- Evolver가 생성하는 지식(예: procedural skill)은 **절차적으로 동형(procedurally isomorphic)**한 형태로 수렴하는 경향이 있습니다.

  예시 (`flink-query` 태스크):
  - Qwen3.5-9B 생성 스킬: 배치 스타일 sessionization, ~3,300자
  - Opus 4.6 생성 스킬: `KeyedProcessFunction` 사용, ~3,800자
  - **결과**: 두 스킬 모두 동일한 Opus 4.6 에이전트로 하여금 태스크를 통과(score 1.0)시킴

이는 **harness-updating 능력이 특정 모델 패밀리나 스케일에 특수화되지 않고 일반화**됨을 시사합니다.

### 3.2 Harness-Benefit의 비단조적 일반화 패턴

$$\Delta_{\text{benefit}}(f) = \max_{e \in \mathcal{E}^\star} \Delta(f, e)$$

이 수식에서 $\Delta_{\text{benefit}}$는 모델의 기반 능력 $M_{\text{base}}(f)$에 대해 세 가지 다른 이유로 다른 값을 가집니다:

**① 약한 모델 (Weak-tier)**
```
낮은 Harness-Benefit
  ├── Harness Activation Failure
  │     └── SLR: 25.1% (Qwen3-32B) vs 96%+ (strong)
  └── Harness Adherence Failure
        └── HFR: 0.142 (Qwen3-32B) vs 0.757 (Opus 4.6)
```
일반화 관점: 약한 모델은 **도메인 전이 시 하네스를 올바르게 호출하거나 준수하는 능력이 부족**하여, 새로운 태스크 분포에서 업데이트된 지식을 활용하지 못합니다.

**② 중간 모델 (Mid-tier)**
```
최대 Harness-Benefit
  └── 충분한 기반 능력 + 성능 천장까지 여유 공간
      → 새로운 하네스 정보를 효과적으로 활용
```
일반화 관점: **업데이트된 하네스의 절차적 지침을 새로운 태스크에 전이(transfer)하는 능력이 가장 우수**합니다.

**③ 강한 모델 (Strong-tier)**
```
낮은 Harness-Benefit (다른 이유)
  └── 성능 천장 효과: 이미 높은 기반 능력으로 많은 태스크를 해결
      → 추가 개선 여지(headroom) 부족
```
일반화 관점: 이미 풍부한 내재적 능력이 있어 외부 하네스 의존도가 낮습니다.

### 3.3 일반화 성능 향상을 위한 설계 시사점

논문은 일반화 성능을 향상시키기 위한 세 가지 구체적 방향을 제시합니다:

**① 능력 예산 할당 최적화**

| 투자 방향 | 기대 효과 |
|----------|---------|
| Evolver 스케일업 | $\leq 3.1$ pp 향상 (제한적) |
| Task-Solving Agent 강화 | 훨씬 더 큰 변동성과 향상 가능성 |

**② 하네스 호출(Invocation) 훈련 내재화**

Qwen3-32B의 낮은 SLR(25.1%)은 포맷 프로토콜 실패에서 비롯됩니다:

```json
// 잘못된 형식 (다중 키 - 거부됨):
{
  "analysis": "...",
  "plan": "...",
  "load_skill": "threejs"
}

// 올바른 형식 (단일 키):
{"load_skill": "threejs"}
```

에이전트 훈련 시 **하네스 호출 프로토콜 준수를 명시적 학습 목표**로 포함해야 합니다. 이는 새로운 태스크 분포에서도 일반화될 수 있는 **메타-스킬**입니다.

**③ 장기 지평 지침 준수 능력 강화**

위상별 준수 분석에서 드리프트 패턴:

$$\text{Drift} = \text{Adherence}_{\text{final}} - \text{Adherence}_{\text{loaded}}$$

$$\text{Drift}_{\text{Qwen3-32B}} = 0.13 - 0.52 = -0.39$$
$$\text{Drift}_{\text{GPT-OSS}} = 0.43 - 0.67 = -0.24$$
$$\text{Drift}_{\text{Opus 4.6}} = 0.80 - 0.89 = -0.09$$

약한 모델의 급격한 드리프트는 **장기 지평 지침 준수(long-horizon instruction following)**가 일반화의 핵심 병목임을 보여줍니다. 장기 궤적에서도 일관된 하네스 준수를 유지하는 훈련이 새로운 도메인으로의 일반화에 필수적입니다.

### 3.4 크로스 도메인 일반화 가능성

논문의 실험 설계 자체가 일반화 능력을 테스트합니다:

- **SWE-bench**: 소프트웨어 엔지니어링 도메인
- **MCP-Atlas**: 멀티서버 도구 사용 도메인
- **SkillsBench**: 11개 다양한 도메인 (오디오 합성, 데이터 분석 등)

일반화 관련 핵심 발견: **harness-updating의 평탄성은 도메인에 무관하게 일관**되나, **어떤 evolver가 최선인지는 도메인에 따라 달라집니다** (Qwen3-235B는 SWE에서 최선, MCP에서 최하).

---

## 4. 2020년 이후 관련 최신 연구 비교 분석

### 4.1 자기진화 에이전트 연구 계보

| 연구 | 연도 | 핵심 방법 | 본 논문과의 관계 |
|------|------|---------|--------------|
| Reflexion (Shinn et al.) | 2023 | 언어적 자기반성을 컨텍스트에 저장 | 초기 자기진화 방법; 단일 텍스트 반성만 저장, 구조화된 하네스 없음 |
| Self-Refine (Madaan et al.) | 2023 | 자기피드백 반복 개선 | 태스크 시도 수준에서 작동; 영속적 하네스 미사용 |
| ExpeL (Zhao et al.) | 2024 | 훈련 궤적에서 재사용 가능한 통찰 추출 | 경험 기반 학습의 선구자; 단일 텍스트 교훈 |
| Voyager (Wang et al.) | 2023 | 실행 가능한 스킬 누적 | 스킬 라이브러리 자기진화의 시초 |
| AWM (Wang et al.) | 2024 | 성공 궤적에서 워크플로 유도 | 워크플로 수준 자기진화 |
| PromptWizard (Agarwal et al.) | 2024 | 피드백 기반 프롬프트 최적화 | 프롬프트 수준 하네스 업데이트 |
| EvolveR (Wu et al.) | 2025 | 경험 기반 생명주기 자기진화 | 종단간 평가만 수행; 두 능력 혼재 |
| GEPA (Agrawal et al.) | 2026 | 궤적 수준 반성으로 프롬프트 진화 | 강화학습 대비 성능 비교; 단일 에이전트-evolver |
| SkillRL (Xia et al.) | 2026 | 재귀적 스킬 강화학습 | 스킬 라이브러리 RL 기반 확장 |
| MemMA (Lin et al.) | 2026 | 멀티에이전트 메모리 사이클 조율 | 메모리 수준 하네스 자기진화 |

**본 논문의 차별점**: 기존 연구들은 모두 특정 evolver-에이전트 쌍에 대한 종단간 이득만 보고하여 두 능력을 혼재시킵니다. 본 논문은 **에이전트와 evolver를 독립적으로 교차 실험**하여 두 능력을 처음으로 체계적으로 분리합니다.

### 4.2 연구 패러다임 비교

```
기존 연구:
  [특정 에이전트 + 특정 evolver] → 종단간 성능 향상 측정
  → 능력의 원천 불명확

본 논문:
  [6개 에이전트 × 7개 evolver × 3개 벤치마크]
  → harness-updating과 harness-benefit 독립 측정
  → 각 능력과 기반 능력의 관계 분석
```

### 4.3 하네스 엔지니어링 관련 연구 비교

| 하네스 유형 | 관련 연구 | 본 논문의 관련성 |
|-----------|---------|--------------|
| 프롬프트 | Zhou et al. 2022, Pan et al. 2026 | 프롬프트를 진화 가능한 하네스 구성요소로 포함 |
| 도구 | Qin et al. 2024, Liu et al. 2025 | MCP-Atlas 벤치마크로 도구 사용 평가 |
| 메모리 | Ouyang et al. 2025, Xu et al. 2026 | 메모리를 MCP-Atlas의 진화 가능 구성요소로 포함 |
| 스킬 | Li et al. 2026b, Liu et al. 2026 | SkillsBench에서 스킬 진화 중점 평가 |
| 코드 하네스 | Lee et al. 2026, Ning et al. 2026 | 코드 자체를 하네스로 처리하는 패러다임 |

---

## 5. 향후 연구에 미치는 영향과 고려사항

### 5.1 연구에 미치는 영향

#### 5.1.1 평가 방법론의 패러다임 전환

```
기존: end-to-end 성능 측정
  → "이 자기진화 방법이 효과적인가?"

이후: 분리된 능력 측정
  → "누가 더 좋은 업데이트를 생성하는가?" (harness-updating)
  → "누가 업데이트로부터 더 이득을 얻는가?" (harness-benefit)
```

향후 자기진화 에이전트 연구는 반드시 이 두 능력을 **독립적으로 보고**해야 할 것입니다.

#### 5.1.2 자원 배분 전략의 재정립

$$\max_{\text{budget}} \text{Performance} \approx \max_{\text{budget}} M_{\text{base}}(f_{\text{agent}})$$

evolver에 대한 스케일업보다 **task-solving agent의 기반 능력 향상**이 훨씬 효율적입니다. 이는 대규모 LLM 배포 비용 최적화에 직접적 영향을 미칩니다.

#### 5.1.3 에이전트 훈련 목표의 재정의

기존 훈련 목표: 태스크 성능 향상
새로운 추가 목표:
1. **하네스 호출 능력**: 관련 스킬/메모리를 올바른 형식으로 불러오기
2. **장기 준수 능력**: 긴 궤적에서도 하네스 지침 준수 유지

#### 5.1.4 이론적 기반 강화 필요성

현재 관찰된 비단조 패턴에 대한 이론적 설명이 부족합니다. 향후 연구에서는 다음을 이론적으로 규명해야 합니다:

- 왜 harness-updating은 모델 스케일에 무관한가?
- 중간 능력 모델에서 최대 이득이 나타나는 근본 원인은 무엇인가?

### 5.2 향후 연구 시 고려할 점

#### 5.2.1 실험 설계 측면

| 고려사항 | 이유 |
|---------|------|
| **에이전트-evolver 교차 실험** | 단일 쌍 평가는 두 능력을 혼재시킴 |
| **인시튜(in-situ) 평가 사용** | 진화에 사용된 태스크와 평가 태스크의 분리 |
| **다중 벤치마크 사용** | 단일 벤치마크에서 우수한 evolver가 다른 벤치마크에서 열등할 수 있음 |
| **능력 계층별 모델 포함** | 비단조 패턴 관찰을 위해 약/중/강 계층 모두 필요 |

#### 5.2.2 파라메트릭 적응과의 통합

본 논문이 다루지 않은 **하이브리드 방법** 연구가 필요합니다:

```
하이브리드 적응:
  하네스 업데이트 (본 논문 범위)
      +
  가중치 업데이트 (RL, 파인튜닝)
      ↓
  각 방법의 기여를 어떻게 분리하고 최적화할 것인가?
```

#### 5.2.3 실패 모드 해결을 위한 훈련 방법

**Harness Activation Failure 해결**:
- 프로토콜 준수 훈련 데이터 구축 (올바른/잘못된 형식 쌍)
- 하네스 호출을 명시적 보상 신호로 사용하는 RL

**Harness Adherence Failure 해결**:
- 장기 지평 준수를 측정하는 HFR 기반 보상 함수 설계:

$$R_{\text{adherence}} = \sum_{t=1}^{T} \gamma^t \cdot \text{AdherenceScore}(t)$$

- 다양한 길이의 지침 준수 훈련 커리큘럼

#### 5.2.4 안전성 및 윤리적 고려사항

논문의 Ethics Statement(Section 7)에서 지적된 바와 같이:

| 위험 요소 | 완화 방안 |
|---------|---------|
| 잘못된 교훈의 영속화 | 업데이트 가역성(reversibility) 메커니즘 |
| 편향된 지침 기록 | 업데이트 감사(auditability) 시스템 |
| 민감 정보 하네스 기록 | 동의 기반 데이터 보존 정책 |
| 자율적 하네스 수정 범위 | 인간 감독(oversight) 체계 |

#### 5.2.5 확장 가능성 연구

현재 논문의 제한된 모델 집합을 확장하여:
- 더 다양한 모델 패밀리 (Gemini, Mistral 등)
- 더 세밀한 스케일 구분
- 다국어 및 멀티모달 에이전트
- 실세계 배포 환경에서의 태스크 분포 이동(distribution shift) 시나리오

#### 5.2.6 Evolver의 역할 재정의

harness-updating이 평탄하다는 발견은 evolver의 역할을 재정의합니다:

```
기존 가정: 더 강력한 evolver → 더 좋은 업데이트
본 논문: evolver 품질보다 evolver 종류(유형)가 중요할 수 있음
향후 연구: 어떤 종류의 evolver가 어떤 종류의 하네스 업데이트에 특화되는가?
```

---

## 참고 자료

**주요 참고 자료:**

1. **Lin, M. et al. (2026). "Harness Updating Is Not Harness Benefit: Disentangling Evolution Capabilities in Self-Evolving LLM Agents."** arXiv:2605.30621v1 [cs.AI]. *(본 분석의 주 대상 논문)*

**논문 내 인용 문헌 (본 답변 작성에 참조):**

2. Shinn, N. et al. (2023). "Reflexion: Language agents with verbal reinforcement learning." *NeurIPS 36*, 8634–8652.
3. Madaan, A. et al. (2023). "Self-Refine: Iterative refinement with self-feedback." *NeurIPS 36*, 46534–46594.
4. Zhao, A. et al. (2024). "ExpeL: LLM agents are experiential learners." *AAAI 38*, 19632–19642.
5. Wang, G. et al. (2023). "Voyager: An open-ended embodied agent with large language models." arXiv:2305.16291.
6. Wang, Z.Z. et al. (2024). "Agent workflow memory." arXiv:2409.07429.
7. Agarwal, E. et al. (2024). "PromptWizard: Task-aware prompt optimization framework." arXiv:2405.18369.
8. Wu, R. et al. (2025). "Evolver: Self-evolving LLM agents through an experience-driven lifecycle." arXiv:2510.16079.
9. Agrawal, L.A. et al. (2026). "GEPA: Reflective prompt evolution can outperform reinforcement learning." *ICLR 2026*.
10. Xia, P. et al. (2026). "SkillRL: Evolving agents via recursive skill-augmented reinforcement learning." arXiv:2602.08234.
11. Jimenez, C.E. et al. (2024). "SWE-bench: Can language models resolve real-world GitHub issues?" *ICLR 2024*, 54107–54157.
12. Li, X. et al. (2026b). "SkillsBench: Benchmarking how well agent skills work across diverse tasks." arXiv:2602.12670.
13. Bandi, C. et al. (2026). "MCP-Atlas: A large-scale benchmark for tool-use competency with real MCP servers." arXiv:2602.00933.
14. Wei, J. et al. (2022). "Chain-of-thought prompting elicits reasoning in large language models." *NeurIPS 35*, 24824–24837.
15. Yao, S. et al. (2022). "ReAct: Synergizing reasoning and acting in language models." arXiv:2210.03629.
16. Lin, M. et al. (2026c). "MemMA: Coordinating the memory cycle through multi-agent reasoning and in-situ self-evolution." arXiv:2603.18718.
17. Lee, Y. et al. (2026). "MetaHarness: End-to-end optimization of model harnesses." arXiv:2603.28052.
