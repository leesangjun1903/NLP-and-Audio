# AutoScientists: Self-Organizing Agent Teams for Long-Running Scientific Experimentation 

> **참고 자료**: Gao, S., Fang, A., & Zitnik, M. (2026). *AutoScientists: Self-Organizing Agent Teams for Long-Running Scientific Experimentation*. arXiv:2605.28655v1 [cs.AI].
> 
> **코드**: https://github.com/mims-harvard/AutoScientists  
> **웹사이트**: https://autoscientists.openscientist.ai

---

## 1. 핵심 주장 및 주요 기여 요약

### 1.1 핵심 주장

기존 AI 에이전트 기반 과학 연구 시스템은 **단일 탐색 궤적**을 따르거나 **중앙 조정자(central planner)**를 통해 고정된 목표를 향해 나아가는 방식이었습니다. 이런 접근법은:

- 병렬 가설 탐색(parallel exploration)을 지속하기 어렵고
- 실험 증거가 쌓여감에 따라 방향을 유연하게 전환하지 못하며
- 실패한 방향에 대한 지식을 보존하지 못합니다.

**AutoScientists**는 **중앙 조정자 없이** 에이전트들이 공유 상태(shared state)를 해석하고 자율적으로 팀을 구성하여 장기 실험을 수행하는 **탈중앙화된(decentralized) 멀티 에이전트 프레임워크**입니다.

### 1.2 주요 기여

| 기여 항목 | 내용 |
|-----------|------|
| **자기조직화 에이전트 팀** | 외부 플래너 없이 에이전트 간 토론을 통해 팀과 연구 방향이 창발(emerge) |
| **장기 실험에서의 지속적 성능 향상** | 단일 에이전트 기준선이 정체된 이후에도 지속적으로 개선 발견 |
| **도메인 전반 SOTA** | 생의학 ML, 언어모델 최적화, 단백질 fitness 예측에서 최고 성능 달성 |

---

## 2. 해결하고자 하는 문제, 제안 방법, 모델 구조, 성능 및 한계

### 2.1 해결하고자 하는 문제

과학 연구의 반복적 사이클(가설 생성 → 실험 설계 → 실행 → 수정)을 AI가 장기적으로 자율 수행할 때 직면하는 세 가지 핵심 문제:

1. **단일 탐색 궤적 한계**: AIDE, Autoresearch 같은 단일 에이전트는 하나의 방향만 탐색
2. **중앙화된 다중 에이전트의 유연성 부족**: 시작 시점에 고정된 탐색 공간 분해에 의존
3. **실패 지식의 손실**: 비생산적 방향에 대한 정보가 공유되지 않아 중복 탐색 발생

### 2.2 제안 방법 및 핵심 수식

#### (1) 문제 정형화 (Problem Formulation)

$n$개의 장기 실행 LLM 에이전트 $\mathcal{A} = \{a_1, \ldots, a_n\}$이 프로그램 공간 $\mathcal{P}$에서 최적의 프로그램을 탐색합니다:

$$p^* = \arg\max_{p \in \mathcal{P}} \ell_{\text{eval}}(p; \mathcal{D})$$

여기서:
- $\mathcal{D}$: 훈련 데이터 $\mathcal{D}\_{\text{train}}$ + 평가 프로토콜 ($\mathcal{D}_{\text{val}}$ 또는 CV)
- $\ell_{\text{eval}}$: 평가 지표 (높을수록 좋게 정규화)
- $\mathcal{P}$: 에이전트가 탐색하는 프로그램 공간 (초기 프로그램 $p_0$에서 시작 가능)

#### (2) 잡음 인식 챔피언 검증 (Noise-Aware Champion Validation)

평가 지표 $\ell$이 확률적(stochastic)이므로, 개선이 노이즈인지 진짜 향상인지 판별하기 위한 게이트:

$$\text{promote}(p') = \begin{cases} \texttt{true} & \text{if } \Delta > M\sigma \\ \texttt{confirm}(p', \text{seed}_2) & \text{if } 0 < \Delta \leq M\sigma \\ \texttt{false} & \text{if } \Delta \leq 0 \end{cases}$$

여기서 $\Delta = \ell(p') - \ell(p^*)$, $M = 2$, $\sigma$는 노이즈 플로어.

노이즈 플로어 $\sigma$는 동일 코드의 두 시드 실행 쌍 $(\ell_{1,i}, \ell_{2,i})$으로부터 추정:

$$\sigma = \sqrt{\frac{1}{2n} \sum_{i=1}^{n} (\ell_{1,i} - \ell_{2,i})^2}$$

최소 3쌍 이상 누적 시 활성화되고 5쌍에서 고정(locked).

#### (3) 애널리스트 제안 프로토콜 — 경험적 축 사전(Empirical Axis Priors)

각 (축, 방향) 쌍에 대해 관측된 효과 크기의 평균:

$$\mu_{a,d} = \frac{1}{|E_{a,d}|} \sum_{e \in E_{a,d}} |\Delta_e|$$

- $|E_{a,d}| < 3$인 방향: **cold**로 분류 → 탐색 보너스 부여
- $\mu_{a,d} < \sigma$: 효과가 노이즈 이하 → 우선순위 하향
- 팀 큐 $Q_k$는 이 랭킹에 따라 정렬

#### (4) 로스터 구조 (Self-Organized Team Roster)

$$R = \{(\mathcal{T}_k, \text{axis}_k, \text{members}_k)\}_{k=1}^{K}$$

로스터는 외부 조정자가 아닌 에이전트들의 토론을 통해 생성·갱신됩니다.

### 2.3 모델 구조 (시스템 아키텍처)

```
┌─────────────────────────────────────────────────────────────┐
│                    AUTOSCIENTISTS 시스템                     │
├─────────────────────────────────────────────────────────────┤
│  공유 상태 (Shared State S)                                  │
│  ├── 챔피언 p*: 현재 최고 모델 + 재현 지침                  │
│  ├── 실험 로그 L: 모든 실험 결과 (성공/실패)                 │
│  ├── 공유 포럼 F: 제안 토론, 결과 공유                       │
│  └── 팀 로컬 상태: Q_k(큐), D_k(dead-end 레지스트리)        │
├─────────────────────────────────────────────────────────────┤
│  에이전트 유형                                               │
│  ├── 애널리스트 에이전트 (3명):                              │
│  │   - 미탐색 파라미터 감사                                  │
│  │   - 효과 크기 기반 제안 생성                              │
│  │   - 정체 감지 → 재토론 트리거                            │
│  └── 실험 에이전트 (6명):                                    │
│      - 큐에서 실험 클레임                                    │
│      - p*에 코드 변경 적용, 훈련                             │
│      - 노이즈 인식 게이트 적용                               │
├─────────────────────────────────────────────────────────────┤
│  운영 단계                                                   │
│  1. 토론 단계: 방향 제안 → 비평 → 팀 구성 (로스터 R 작성)   │
│  2. 실행 단계: 팀별 병렬 실험 → 공유 상태 업데이트          │
│  3. 정체 감지 → 재토론 → 팀 재조직                          │
└─────────────────────────────────────────────────────────────┘
```

#### 핵심 알고리즘 흐름 (Algorithm 1 기반)

각 에이전트 호출(heartbeat)에서:
1. 로스터 $R$과 포럼 $\mathcal{F}$ 읽기
2. `DISCUSSION-TRIGGER`가 있거나 $R$이 비어 있으면 → **토론 브랜치** 실행 (Algorithm 2)
3. 실험 에이전트: 큐에서 실험 클레임 → 실행 → 노이즈 게이트 적용 (Algorithm 3)
4. 애널리스트: 정체 여부 확인 → 새 제안 생성 → $Q_k$에 추가 (Algorithm 4)

### 2.4 성능 향상

#### BioML-Bench (24개 생의학 ML 태스크)

| 시스템 | 평균 리더보드 백분위 |
|--------|---------------------|
| MLAgentBench | 21.4% |
| AIDE | 31.8% |
| STELLA | 60.9% |
| Biomni | 60.4% |
| Autoresearch | 66.1% |
| **AutoScientists** | **74.4%** |

도메인별 성능 (Table 1):

| 도메인 | Biomni | Autoresearch | AutoScientists |
|--------|--------|--------------|----------------|
| 생의학 이미징 (n=4) | 19.04% | 39.60% | **45.75%** |
| 약물 발견 (n=9) | 47.91% | 46.16% | **64.52%** |
| 단백질 공학 (n=6) | 93.94% | 96.97% | **96.97%** |
| 단세포 오믹스 (n=5) | 78.00% | 86.00% | **88.00%** |

#### GPT 훈련 최적화

- **베이스라인에서**: $\text{val bpb} \approx 0.978$에 도달하는 데 AutoScientists 34회 실험 vs. Autoresearch 65회 → **1.9× 빠름**
- **챔피언에서 계속**: AutoScientists 93회 중 7개 개선 수용 ($\text{val bpb} = 0.9730$) vs. Autoresearch 100회 중 0개 수용

#### ProteinGym 단백질 Fitness 예측

ProteinGym 지표 공식:

$$\ell_{\text{eval}} = \frac{1}{3} \sum_{s \in \{\text{random, modulo, contiguous}\}} \rho_s$$

| 모델 | 평균 Spearman $\rho$ |
|------|---------------------|
| Kermut (SoTA) | 0.657 |
| **AutoScientists-Kermut** | **0.700** (+6.5%) |

ACE2-Spike 개발 어세이에서: $\rho$: 0.747 → 0.840 (**+12.5%**)

#### Ablation 연구 (Table 3)

| 태스크 | No analyst | No cross-agent | No self-org | Independent agents | **Full** |
|--------|-----------|---------------|-------------|-------------------|----------|
| TDC-hERG (AUROC) | 0.738 | 0.819 | 0.807 | 0.853 | **0.867** |
| Cell-Cell Comm. (OR) | 0.858 | 0.908 | 0.628 | 0.435 | **0.924** |
| HPPB (Pearson r) | 0.813 | 0.714 | 0.811 | 0.784 | **0.873** |
| GPT (val\_bpb↓) | 0.9817 | 0.9814 | 0.9833 | 0.9833 | **0.9777** |

### 2.5 한계

1. **LLM 호출 효율**: Autoresearch 대비 더 많은 LLM 토큰 사용 (Table S8: AutoScientists 총 ~3,984.7M $ vs. Autoresearch ~910.1M $, 같은 차수)
2. **GPU 병렬성 미활용**: BioML-Bench 비교 시 1 H100 GPU만 사용 → 실험이 순차 실행
3. **팀 크기 고정**: 실행 전 팀 크기가 고정되며, 태스크 난이도에 따른 동적 조정 미지원
4. **다목적 최적화 미지원**: ProteinGym에서 Spearman $\rho$ 향상 시 MSE가 소폭 증가 (0.605 → 0.611)
5. **재현성의 확률적 특성**: 3회 독립 실행 시 $\text{val bpb}$ 편차 0.0018 (평균 0.9784, 표준편차 0.0010)

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 도메인 간 일반화 (Cross-Domain Generalization)

AutoScientists는 **단일 시스템 구성**으로 다양한 도메인에서 우수한 성능을 보여줍니다. 이는 LLM-agnostic 설계 덕분입니다:

- 생의학 이미징 (영상 분할, 암 검출)
- 약물 발견 (ADME, hERG 차단)
- 단세포 오믹스 (modality 예측, 세포-세포 통신)
- 단백질 공학 (피트니스 예측)
- 언어모델 훈련 최적화

### 3.2 동결된 레시피의 전이 (Frozen Recipe Transfer)

가장 강력한 일반화 증거는 **ProteinGym** 실험입니다:

- AutoScientists-Kermut 레시피를 **ACE2-Spike** 단 하나의 어세이에서 개발
- **동결(frozen)** 후 수정 없이 **217개 전체 어세이**에 적용
- 결과: 평균 Spearman $\rho$ 0.657 → 0.700 (+6.5%)

이는 도메인 내 일반화(in-domain generalization)의 강력한 증거입니다.

```
개발 어세이 (ACE2-Spike):  ρ: 0.747 → 0.840 (+12.5%)
전체 217 어세이 적용:       ρ: 0.657 → 0.700 (+6.5%)
```

### 3.3 일반화를 가능하게 하는 설계 요소

#### (a) Dead-End 레지스트리를 통한 탐색 공간 효율화

$$\mathcal{D}_k = \{(\text{axis}, \text{direction}, \Delta\ell, \text{rejection reason})\}$$

비생산적 방향을 등록하여 중복 탐색 방지 → **적은 실험으로 더 넓은 공간 탐색**

#### (b) 다양성 기반 제안 제약 (Diversity Constraints)

애널리스트 제안 시:
- 두 제안은 **서로 다른 연구 방향**을 타겟해야 함
- 연속 3개 이상의 동일 축/방향 제안 금지
- Dead-end 레지스트리에 있는 범위 재제안 시 차이점 명시 의무

이 제약들이 **모델 과적합 방지**와 **탐색 다양성 유지**에 기여합니다.

#### (c) 잡음 인식 챔피언 검증의 일반화 효과

**챔피언 오염(champion pollution)** 방지:

$$\text{promote}(p') = \begin{cases} \texttt{true} & \Delta > M\sigma \\ \texttt{confirm}(p', \text{seed}_2) & 0 < \Delta \leq M\sigma \\ \texttt{false} & \Delta \leq 0 \end{cases}$$

두 번째 시드로 검증함으로써 노이즈에 의한 가짜 개선이 챔피언으로 승격되는 것을 막아 **하류 실험의 일관성** 보장.

#### (d) 정량화 목표 정규화 (Quantile Normalization) — ProteinGym 일반화 핵심

AutoScientists-Kermut에서 GP 회귀 타겟을 van der Waerden 점수로 표준 정규 분포에 매핑:

$$\text{rank } r_i \mapsto \Phi^{-1}\!\left(\frac{r_i - 0.5}{N}\right)$$

이를 통해 DMS 점수의 비대칭 분포를 정규화 → 217개 어세이 전반에 걸친 **일관된 순위 학습** 가능. Ablation에서 이를 제거 시 평균 $\rho$: 0.8407 → 0.8070으로 감소.

#### (e) 팀 크기와 일반화

팀 크기 실험 (Table S3)에서 최적 크기는 태스크 의존적:

| 태스크 | 최적 $n$ |
|--------|---------|
| TDC-hERG | $n=9$ (AUROC 0.867) |
| ProteinGym SPIKE-SARS2 | $n=2$ (Spearman 0.874) |
| GPT 훈련 최적화 | $n=2$ 또는 $n=9$ |

**→ 미래 연구 방향**: 태스크 난이도에 따른 동적 팀 크기 조정

### 3.4 일반화 한계 및 주의사항

- **MSE 트레이드오프**: Spearman $\rho$ 향상 시 MSE 소폭 증가 (0.605 → 0.611) → 순위 기반 지표와 절대값 지표 간 상충
- **과최적화 위험**: 개발 어세이에서의 집중 탐색이 해당 어세이 특화로 이어질 가능성
- **생의학 이미징에서의 한계**: 가장 낮은 도메인 성능 유지 (45.75%) — 대규모 이미지 모델 훈련 필요

---

## 4. 앞으로의 연구에 미치는 영향 및 고려사항

### 4.1 미래 연구에 미치는 영향

#### (1) AI for Science 패러다임 전환

AutoScientists는 AI 과학 연구를 **단일 실험 실행**에서 **집합적 탐색 과정(collective search process)**으로 전환합니다. 이는 향후 AI 과학 시스템 설계의 참조 프레임워크가 될 것입니다.

#### (2) 장기 실험 인프라의 표준화 가능성

공유 상태($\mathcal{S}$)와 dead-end 레지스트리 개념은 대규모 컴퓨테이셔널 과학 실험에서 **재현성(reproducibility)**과 **지식 보존**을 위한 표준으로 발전할 수 있습니다.

#### (3) 멀티 에이전트 협업 연구에의 기여

Ablation 연구 결과는 다중 에이전트 시스템의 핵심 통찰을 제공합니다:
- **애널리스트 역할**: 제안 품질이 병목일 때 가장 중요
- **교차 에이전트 피드백**: 개별 에이전트가 부분적 신호만 관측할 때 중요
- **자기조직화**: 실험 중 생산적 방향이 변화할 때 중요
- **공유 실험 기록**: 독립 에이전트들이 중복 작업하거나 호환되지 않는 지역 최적해로 수렴할 때 중요

#### (4) 생의학 AI 연구 가속

ProteinGym에서 +6.5% 향상은 AI 시스템이 도메인 전문가와 동등하거나 그 이상의 방법론 개선을 자율적으로 발견할 수 있음을 시사합니다.

### 4.2 향후 연구 시 고려할 사항

#### (1) 동적 팀 크기 조정
현재 팀 크기는 실행 전 고정됩니다. **태스크 난이도 추정 기반 동적 확장**이 필요합니다:
- 조기 정체 신호 감지 → 팀 규모 축소
- 빠른 개선 신호 → 팀 규모 확대

#### (2) 다목적 최적화 지원
현재 단일 지표($\ell_{\text{eval}}$) 최적화만 지원. 다음을 고려해야 합니다:

$$p^* = \arg\max_{p \in \mathcal{P}} \left[\alpha \cdot \ell_1(p; \mathcal{D}) + (1-\alpha) \cdot \ell_2(p; \mathcal{D})\right]$$

예: Spearman $\rho$와 MSE 동시 최적화 (파레토 프론티어 탐색)

#### (3) 실험 예산 할당 최적화
현재 공평한 실험 예산 비교를 위해 단일 GPU로 제한. **다중 GPU 환경에서의 스케일링**:
- 팀 수 × GPU 수 매핑 최적화
- 실험 복잡도에 따른 적응적 자원 배분

#### (4) 벤치마크 과적합 위험 관리
논문 자체가 지적하듯, **벤치마크 피드백에 과적합**될 위험이 있습니다. 미래 연구에서는:
- 홀드아웃 검증 세트 구성의 엄격한 프로토콜 필요
- 개발 어세이 선택이 전체 성능에 미치는 영향 분석

#### (5) LLM 백엔드 의존성
현재 Claude Sonnet 4.6 단일 백엔드 사용. **LLM-agnostic 설계**를 실제로 검증하기 위한:
- 다양한 LLM 백엔드 (GPT-4o, Gemini, 오픈소스 모델) 비교 실험 필요
- 에이전트 수와 LLM 능력 간 상호작용 효과 분석

#### (6) 신뢰성 및 안전성
특히 생의학 응용에서:
- 자동 발견 모델의 임상/생물학적 해석 가능성 검증 프레임워크 필요
- 에이전트 간 오류 가설 증폭(amplification) 방지 메커니즘

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

| 시스템 | 연도 | 조정 방식 | 병렬 탐색 | 장기 실험 | 실패 지식 보존 | 주요 도메인 |
|--------|------|-----------|-----------|-----------|----------------|-------------|
| **AIDE** [arXiv:2502.13138] | 2025 | 단일 에이전트 | ✗ | 제한적 | ✗ | ML 실험 |
| **Autoresearch** [Karpathy, 2026] | 2026 | 단일 에이전트 | ✗ | ✓ | 제한적 | GPT 훈련 |
| **STELLA** [bioRxiv, 2026] | 2026 | PI-과학자-비평가 | 제한적 | 제한적 | ✗ | 생의학 |
| **Biomni** [bioRxiv, 2025] | 2025 | 범용 생의학 | ✗ | ✗ | ✗ | 생의학 |
| **AI Co-Scientist** [arXiv:2502.18864] | 2025 | 토론 기반 합의 | ✗ | ✓ | ✗ | 범용 과학 |
| **Virtual Lab** [Nature, 2025] | 2025 | Manager-Developer-Critic | ✓ | ✗ | ✗ | 나노바디 설계 |
| **CORAL** [arXiv:2604.01658] | 2026 | 자율 진화 | ✓ | ✓ | ✗ | 오픈엔드 발견 |
| **AlphaEvolve** [arXiv:2506.13131] | 2025 | 진화적 코드 에이전트 | ✓ | ✓ | 제한적 | 알고리즘 발견 |
| **AutoScientists** | 2026 | **탈중앙화 자기조직화** | **✓** | **✓** | **✓** | **범용** |

### 주요 차별점 분석

**vs. AIDE/Autoresearch (단일 에이전트)**:
- AIDE와 Autoresearch는 단일 탐색 궤적으로 인해 지역 최적해에 갇히기 쉬움
- AutoScientists는 병렬 팀이 서로 다른 가설을 동시 탐색 → GPT 최적화에서 1.9× 빠른 수렴

**vs. AI Co-Scientist (합의 기반 토론)**:
- AI Co-Scientist의 토론은 단일 가설로 **수렴**하기 위해 사용
- AutoScientists의 토론은 약한 제안을 **필터링**하되, 에이전트가 서로 다른 방향 탐색 지속

**vs. Virtual Lab (역할 특화 파이프라인)**:
- Virtual Lab은 Manager→Developer→Critic의 고정 파이프라인
- AutoScientists는 방향이 변화함에 따라 팀 구조 자체가 재구성

**vs. AlphaEvolve (진화적 접근)**:
- AlphaEvolve는 코드 변이를 진화적으로 탐색
- AutoScientists는 LLM 기반 가설 생성과 과학적 토론을 결합 → 더 해석 가능한 탐색 과정

---

## 참고 자료

1. **주 논문**: Gao, S., Fang, A., & Zitnik, M. (2026). *AutoScientists: Self-Organizing Agent Teams for Long-Running Scientific Experimentation*. arXiv:2605.28655v1.
2. Miller, H. E., et al. (2025). *BioML-Bench: Evaluation of AI agents for end-to-end biomedical ML*. bioRxiv. doi:10.1101/2025.09.01.673319.
3. Karpathy, A. (2026). *Autoresearch: AI agents running research on single-GPU nanochat training automatically*. GitHub.
4. Groth, P. M., et al. (2024). *Kermut: Composite kernel regression for protein variant effects*. NeurIPS, vol. 37.
5. Notin, P., et al. (2023). *ProteinGym: Large-scale benchmarks for protein fitness prediction and design*. NeurIPS, vol. 36.
6. Jiang, Z., et al. (2025). *AIDE: AI-driven exploration in the space of code*. arXiv:2502.13138.
7. Swanson, K., et al. (2025). *The virtual lab of AI agents designs new SARS-CoV-2 nanobodies*. Nature, 646(8085):716–723.
8. Gottweis, J., et al. (2025). *Towards an AI co-scientist*. arXiv:2502.18864.
9. Novikov, A., et al. (2025). *AlphaEvolve: A coding agent for scientific and algorithmic discovery*. arXiv:2506.13131.
10. Qu, A., et al. (2026). *CORAL: Towards autonomous multi-agent evolution for open-ended discovery*. arXiv:2604.01658.
11. Huang, K., et al. (2025). *Biomni: A general-purpose biomedical AI agent*. bioRxiv.
12. Jin, R., et al. (2026). *STELLA: Towards a biomedical world model with self-evolving multimodal agents*. bioRxiv.
13. Du, Y., et al. (2024). *Improving factuality and reasoning in language models through multiagent debate*. ICML 2024.
