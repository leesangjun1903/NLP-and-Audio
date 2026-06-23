# Automating SKILL.md Generation for Computer-Using Agents via Interaction Trajectory Mining

---

## 1. 핵심 주장과 주요 기여 요약

### 핵심 주장

이 논문은 **진단 연구(diagnostic study)** 로서, GUI 상호작용 궤적(trajectory)으로부터 명시적 스킬 라이브러리(SKILL.md)를 자동으로 발굴할 수 있는지, 그리고 이를 통해 하위 정책(downstream policy)을 개선할 수 있는지를 탐구합니다.

핵심 주장은 다음 세 가지로 요약됩니다:

> **"궤적 마이닝은 검사 가능한 스킬 구조를 드러낼 수 있지만, 현재의 경계 탐지기, 순서 없는 세그먼트 표현, 오프라인 보상 모델은 신뢰할 수 있는 크로스 도메인 정책 개선에 불충분하다."**

### 주요 기여 (3가지)

| 기여 | 내용 |
|------|------|
| **기여 1** | GUI 궤적으로부터 명시적 SKILL.md 스타일 루틴을 마이닝하는 3단계 파이프라인 제시 → 소스 도메인 내 가독성 있는 구조 생성 확인 |
| **기여 2** | 마이닝된 스킬이 다양한 베이스라인 및 전이 환경에서 하위 스킬 구성(skill composition)을 개선하는지 평가 |
| **기여 3** | **부정적 결과 보고**: 현재 학습된 컴포넌트들은 단순 빈도 사전(frequency prior)을 넘어서지 못하며, 검증된 크로스 도메인 성능 향상은 없거나 오히려 감소 |

---

## 2. 상세 분석: 문제 정의 / 방법 / 모델 구조 / 성능 / 한계

### 2.1 해결하고자 하는 문제

**문제 상황:**
- Computer-Using Agents(CUAs)는 GUI에서 클릭, 타이핑, 스크롤, 복사, 붙여넣기 등의 행동을 수행합니다.
- 반복적인 행동 패턴을 **스킬(skill)**로 패키징하면 검사 및 디버깅이 용이해지나, 기존에는 수작업으로 작성해야 합니다.
- 수작업 스킬 라이브러리는 명명, 범위 설정, 문서화, 인터페이스 변경 시 업데이트가 필요하여 **실질적 병목**을 형성합니다.

**핵심 연구 질문:**
> "궤적 데이터로부터 발견된 클러스터가 새로운 작업에서 정책을 도울 수 있는가? (What transfers from mined skills?)"

**공식 문제 정의 (Eq. 1):**

입력 데이터셋을 다음과 같이 정의합니다:

$$\mathcal{D} = \{\tau^{(n)}\}_{n=1}^{N}, \quad \tau^{(n)} = ((o_1, a_1), \ldots, (o_T, a_T))$$

여기서 $o_t$는 GUI 관찰, $a_t \in \mathcal{A}_{\text{low}}$는 원시 UI 액션입니다. 목표는 스킬 어휘 $\mathcal{Z}$를 귀납하고, 각 궤적을 스킬 $z \in \mathcal{Z}$가 할당된 연속 구간으로 분할하는 것입니다.

---

### 2.2 제안 방법 (3단계 파이프라인)

#### Phase 1: 궤적 분할 — 스킬 경계 탐지 (Eq. 2)

인접 액션 간 거리를 변화점 신호로 활용합니다:

$$\Delta a_t = \|a_t - a_{t-1}\|_2, \quad t \in \mathcal{B} \text{ if } \Delta a_t > \theta$$

- $\theta$: IW 데이터에서 경험적 백분위수를 스윕하여 경계 F1을 최대화하는 값으로 선택 ($\theta = 1.545$, 50번째 백분위수)
- 각 액션 벡터는 15개의 정규화된 특징으로 구성:
  - 10-way 원시 액션 원핫 벡터
  - 화면 좌표 $(x, y) \in [0,1]^2$
  - 정규화된 타임스탬프, 텍스트 길이, 스크롤 양
- **결과**: IW에서 Precision=0.419, Recall=0.803, F1=0.538

#### Phase 2: 스킬 임베딩 — 스킬 라이브러리 구성

**Step 1: 세그먼트 표현 (Eq. 3)**

세그먼트 $\tau_i$의 $T_i$개 액션 벡터 $a_{i,1}, \ldots, a_{i,T_i} \in \mathbb{R}^d$를 평균과 대각 분산으로 요약합니다:

$$\mu_i = \frac{1}{T_i}\sum_{t=1}^{T_i} a_{i,t}, \quad \Sigma_i = \text{diag}\!\left(\frac{1}{T_i}\sum_{t=1}^{T_i}(a_{i,t} - \mu_i) \odot (a_{i,t} - \mu_i) + \epsilon\mathbf{1}\right)$$

여기서 $\odot$은 요소별 곱, $\epsilon = 10^{-4}$는 퇴화 분산 방지를 위한 값입니다. 이 표현은 **순서 불변(orderless) bag-of-actions** 요약이며, 세그먼트 내 행동 순서를 보존하지 않습니다.

**Step 2: Wasserstein 클러스터링 (Eq. 4)**

대각 가우시안 $\mathcal{N}(\mu_i, \Sigma_i)$ 간의 제곱 Bures 거리를 사용합니다:

$$D(\tau_i, \tau_j) = \|\mu_i - \mu_j\|_2^2 + \left\|\sqrt{v_i} - \sqrt{v_j}\right\|_2^2 = \|\mu_i - \mu_j\|_2^2 + \sum_{k=1}^{d}\left(\sqrt{v_{i,k}} - \sqrt{v_{j,k}}\right)^2$$

이는 대각 가우시안 측도 간의 폐쇄형 제곱 2-Wasserstein 거리(Fréchet/Bures 거리의 특수 사례)입니다. 이 거리 행렬에 평균 연결 응집 클러스터링(average-linkage agglomerative clustering)을 적용하고 $k = 8 \sim 16$을 스윕합니다.

**Step 3: 지도 대조 정제 (Eq. 5)**

MLP 인코더 $f_\theta: \mathbb{R}^{2d} \to \mathbb{R}^{d_\text{skill}}$를 이용하여 $x_i = [\mu_i; \text{diag}(\Sigma_i)]$를 $\ell_2$-정규화된 임베딩 $z_i = f_\theta(x_i)$로 매핑합니다. Wasserstein 클러스터 할당 $c_i$를 의사 레이블로 사용하여 다음 지도 대조 손실로 $f_\theta$를 훈련합니다:

$$\mathcal{L}_{\text{sup-con}} = \frac{1}{|\mathcal{B}|}\sum_{i \in \mathcal{B}} \frac{-1}{|P(i)|}\sum_{p \in P(i)} \log \frac{\exp(z_i^\top z_p / T)}{\sum_{a \in \mathcal{B} \setminus \{i\}} \exp(z_i^\top z_a / T)}$$

여기서:
- $\mathcal{B}$: 미니 배치
- $P(i) = \{p \in \mathcal{B} \setminus \{i\} : c_p = c_i\}$: 동일 의사 레이블 양성 집합
- $T = 0.07$: 온도 파라미터
- 인코더 구조: $30 \to 64 \to 32 \to d_\text{skill} = 16$ (ReLU, $\ell_2$ 정규화 출력)
- AdamW ($\text{lr}=10^{-3}$, $\text{wd}=10^{-4}$), 200 에폭, 배치 256

#### Phase 3: 스킬 인식 GRPO 훈련

**보상 모델 (Eq. 6):**

스킬 플랜 선호도 쌍에 대한 쌍별 로지스틱 랭킹 손실을 사용합니다:

$$\mathcal{L}_{\text{RM}} = -\log \sigma\!\left(r_\phi(p, y^+) - r_\phi(p, y^-) - m\right)$$

여기서 $p$는 프롬프트, $y^+$는 정답 스킬 흐름 완성, $y^-$는 합성 근사 실패 계획, $m$은 $[0.05, 0.95]$로 클리핑된 선호 마진입니다.

후보 계획 점수 가중 조합:
$$\text{score} = 0.45 \cdot \text{prefix match} + 0.30 \cdot \text{LCS overlap} + 0.20 \cdot \text{unordered overlap} + 0.05 \cdot \text{length agreement}$$

**GRPO 설정:**
- 기본 모델: Qwen3-8B (베이스 모델에서 시작)
- 프롬프트당 후보 응답: 8개, 온도 0.7
- 최대 완성 길이: 192, 학습률 $5 \times 10^{-6}$
- 그래디언트 누적: 8, 보상 클리핑: 5.0, 1 에폭
- 4× NVIDIA H200 NVL GPUs, 6,072초 소요

---

### 2.3 모델 구조 요약

```
[입력: GUI 궤적]
        ↓
[Phase 1: 경계 탐지]
  Δat = ‖at - at-1‖₂ > θ → 세그먼트 분할
        ↓
[Phase 2: 스킬 라이브러리 구성]
  Step 1: 세그먼트 → (μᵢ, Σᵢ) 통계 요약
  Step 2: Bures/Wasserstein 거리 → 응집 클러스터링 (k=8)
  Step 3: MLP 인코더 (30→64→32→16) + SupCon 손실
        ↓
[SKILL.md 생성]
  스킬 설명, 전이 확률, 워크플로우, 오류 처리 패턴
        ↓
[Phase 3: GRPO 정책 훈련]
  Qwen3-8B + 궤적 보상 모델 → 스킬 시퀀스 구성 정책
        ↓
[평가: IW, WebArena, BrowseComp+, WorkArena-NLP]
```

---

### 2.4 성능 향상 결과

#### 클러스터링 품질 (Phase 2)

| 메트릭 | Wasserstein 기준 | SupCon 정제 후 | 향상 |
|--------|-----------------|--------------|------|
| NMI | 0.650 | 0.862 | +33% 상대적 향상 |
| Silhouette | - | 0.554 | - |
| Purity | 0.63 | 0.837 | - |

5/8 클러스터가 IW 레이블 대비 ≥0.95 순도 달성

#### 정책 전이 성능 (Phase 3) — **부정적 결과**

| 모델 | IW Acc. | WebArena Acc. | BrowseComp+ | WorkArena-NLP |
|------|---------|--------------|-------------|---------------|
| Qwen3-8B (zero-shot) | 18.5% | 55.8% | 43.5% | 37.0% |
| **Qwen3-8B (GRPO)** | **20.5%** | **44.2%** | **43.3%** | **37.0%** |
| Frequency (trivial) | **34.9%** | - | - | - |
| MLP (ours) | 23.3% | 28.5% | - | - |
| Transformer (ours) | 34.6% | 41.0% | - | - |
| Llama-3.1-70B | 30.0% | 56.2% | 51.9% | 38.0% |
| GPT-5 | 24.5% | **57.6%** | 59.5% | **40.6%** |
| OLMo-3-7B | 14.3% | 54.5% | **61.4%** | 12.8% |

**핵심 부정적 결과:**
- GRPO는 IW에서 18.5% → 20.5%로 겨우 2%p 향상
- WebArena에서는 오히려 55.8% → 44.2%로 **하락**
- BrowseComp+는 43.5% → 43.3%로 사실상 **변화 없음**
- **단순 Frequency 베이스라인(34.9%)이 MLP(23.3%)와 GRPO(20.5%)보다 우월**
- Exact sequence match: GRPO = **0%**

#### 경계 탐지 도메인 전이 실패

| 데이터셋 | θ | Precision | Recall | F1 |
|---------|---|----------|--------|-----|
| IW (소스) | 1.545 | 0.419 | 0.803 | 0.538 |
| WebArena (IW θ 적용) | 1.545 | **1.000** | **0.100** | **0.119** |
| WebArena (오라클 θ) | 0.603 | - | - | 0.851 |

---

### 2.5 한계

| 한계 | 설명 |
|------|------|
| **순서 불변 세그먼트 표현** | Phase 2의 bag-of-actions 표현이 within-skill 행동 순서를 무시 (click→copy→paste의 순서가 critical) |
| **경계 탐지기의 낮은 정밀도** | F1=0.538으로 over-splitting 발생, 도메인 비안정적 ($\theta$ 이전 불가) |
| **오프라인 보상 모델** | IW 스킬 흐름 유사도를 학습, 크로스 도메인 태스크 성공을 학습하지 않음 |
| **합성 소스 데이터** | IW가 합성 데이터라 실제 기업 복잡성을 포착하지 못할 수 있음 |
| **단순 통계 사전에 미달** | 모든 학습된 변형이 Frequency 베이스라인을 초과하지 못함 |
| **불완전한 전이 평가** | Mind2Web GRPO 평가 및 실제 WorkArena 결과 없음 |

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 현재 일반화 실패의 구조적 원인 분석

논문은 일반화 실패를 **파이프라인 수준 결과**로 명확히 진단합니다:

#### (1) Phase 1: 경계 탐지의 도메인 비안정성

IW에서 $\theta = 1.545$를 적용했을 때:

$$F1_{\text{IW}} = 0.538 \quad \xrightarrow{\text{WebArena 적용}} \quad F1_{\text{WebArena}} = 0.119$$

이는 도메인마다 액션 스케일이 달라 단일 임계값이 이전 불가능함을 보여줍니다. WebArena 오라클 $\theta = 0.603$에서 $F1 = 0.851$이 가능하지만, 이는 타겟 도메인 레이블을 사용하므로 유효한 전이 결과가 아닙니다.

**일반화 향상을 위한 개선 방향:**

$$ \theta^* = \arg\max_\theta F1(\theta; \mathcal{D}_{\text{target}}) $$

이를 학습 기반으로 대체하여 도메인 적응 경계 탐지기를 구현해야 합니다. 논문이 직접 언급한 방향은 Open-World Skill Discovery [21]의 **액션 예측 오류 기반 경계 탐지**:

$$b_t = \mathbb{1}[\underbrace{\mathcal{L}_{\text{pred}}(a_t | a_{1:t-1})}_{\text{예측 오류}} > \eta]$$

#### (2) Phase 2: 순서 불변 표현의 한계

현재 세그먼트 표현 $x_i = [\mu_i; \text{diag}(\Sigma_i)]$는 행동 순서 정보를 완전히 폐기합니다. 스킬별 정확도 분포가 이를 명확히 보여줍니다:

| 스킬 | 정확도 | 이유 |
|------|--------|------|
| search_navigate | 94.1% | 독특한 액션 시그니처 |
| data_transfer | 89.7% | click→copy→switch→paste 고유 패턴 |
| presentation_edit | **55.6%** | click/type/scroll 공유 → 구별 어려움 |

**일반화 향상을 위한 개선 방향:**

순서 인식(order-aware) 인코더로 교체:

$$z_i = \text{Encoder}_\theta(a_{i,1}, a_{i,2}, \ldots, a_{i,T_i})$$

예를 들어 Transformer 기반 세그먼트 인코더나 RNN을 활용하면 select - copy - paste의 순서가 paste - select - copy와 구별 가능합니다.

#### (3) Phase 3: 오프라인 IW 보상의 크로스 도메인 미스매치

현재 보상 모델은 다음을 학습합니다:

$$r_\phi(p, y) \approx \text{IW-skill-flow similarity}$$

그러나 실제 필요한 것은:

$$r^*(p, y) \approx \text{cross-domain task success}$$

이 미스매치가 WebArena에서 GRPO 후 55.8% → 44.2%로 성능 하락을 일으키는 주된 원인입니다.

### 3.2 일반화 향상을 위한 구체적 개선 경로

논문이 제시한 구체적 개선 방향들:

**① 혼합 도메인 프롬프트 훈련:**

$$\mathcal{D}_{\text{train}} = \mathcal{D}_{\text{IW}} \cup \mathcal{D}_{\text{target-domain prompts}}$$

**② 밀집 보상 모델 훈련:**

태스크 성공 레이블을 사용한 행동 수준 또는 태스크 성공 기반 보상:

$$\mathcal{L}_{\text{RM}}^{\text{dense}} = -\log\sigma\!\left(r_\phi(p, y^+) - r_\phi(p, y^-) - m\right), \quad y^+ \sim \mathcal{D}_{\text{target-success}}$$

**③ 요인 절제 실험:**

3단계 중 2단계를 고정하고 1단계만 교체하는 팩토리얼 실험:

$$\{\text{Ground-truth boundaries} \mid \text{Oracle boundaries} \mid \text{Supervised controls}\}$$

**④ 현재 파이프라인의 긍정적 일반화 신호:**

- NMI 33% 상대적 향상 (Wasserstein → SupCon): 소스 도메인 내 일반화 구조 학습 성공
- Auto-SKILL.md가 $N = 100, 250, 2000$에서 수작업 기준선 초과: 데이터 효율적 일반화 가능성 시사

---

## 4. 앞으로의 연구에 미치는 영향과 고려할 점

### 4.1 연구에 미치는 영향

#### (1) 방법론적 기준 설정

이 논문은 **GUI 스킬 발견 연구에서 최소 3가지 통제 베이스라인을 보고해야 함**을 명확히 합니다:

$$\text{Controls} = \{\text{Frequency prior}, \text{Transition-memory prior}, \text{Modality-matched supervised policy}\}$$

이 기준 없이는 스킬 마이닝의 개선이 데이터 불균형이나 출력 형식 적응에서 비롯된 것인지 구별 불가합니다.

#### (2) 인간 중심 AI 설계 원칙 강화

논문은 자동화 에이전트가 인간이 검사, 질문, 수정할 수 있는 중간 구조를 노출해야 한다는 설계 원칙을 강조합니다. 의료 AI, EHR 에이전트 등 고위험 도메인에서도 동일한 원칙이 적용되어야 합니다.

#### (3) 스킬 발견의 한계 명확화

VIC, DIAYN, DADS 등의 상호정보 기반 스킬 발견 방법에 대한 선행 분석 [36]의 경고, 즉 "상호정보 스킬이 모든 하위 보상에 최적이지 않다"는 점이 실증적으로 확인되었습니다. 이는 비지도 스킬 발견 연구 방향에 중요한 경고입니다.

#### (4) 오프라인 vs. 온라인 RL의 중요성

DigiRL [37], WebRL [38]의 온라인 RL 커리큘럼 접근법과 달리, 이 논문의 오프라인 IW 파생 보상이 실패했습니다. 이는 GUI 에이전트 훈련에서 **실제 GUI 상호작용 기반 온라인 피드백**의 중요성을 강조합니다.

### 4.2 앞으로 연구 시 고려할 점

| 고려 사항 | 구체적 내용 |
|-----------|-----------|
| **순서 인식 세그먼트 인코더** | RNN/Transformer 기반 인코더로 within-skill 행동 순서 보존 |
| **학습된 경계 탐지** | 액션 예측 오류 기반 경계 탐지 [21] 비교 실험 필요 |
| **온라인 타겟 도메인 보상** | 오프라인 IW 기반 보상을 타겟 도메인 성공 신호로 보완 |
| **팩토리얼 절제 실험** | 3단계 중 2단계 고정, 1단계만 교체하는 통제된 실험 설계 |
| **실제 기업 데이터** | 합성 IW 데이터를 실제 기업 궤적으로 보완 |
| **시각 및 텍스트 컨텍스트 통합** | DOM, 스크린샷, 접근성 트리 등 언어적 상태 통합 |
| **인간 검토 프로세스** | 생성된 SKILL.md 배포 전 인간 검토 필수화 |
| **멀티 도메인 공동 훈련** | IW + 타겟 도메인 프롬프트를 혼합하여 전이 가능 어휘 학습 |
| **완전한 전이 평가** | Mind2Web GRPO 및 실제 WorkArena 평가 완료 필요 |

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

| 연구 | 연도 | 접근법 | 스킬 소스 | 전이 검증 | 본 논문과의 차이 |
|------|------|--------|----------|----------|----------------|
| **WebShop** [1] | 2022 | 웹 쇼핑 에이전트 | 수작업 정의 | 단일 도메인 | 스킬 자동화 없음 |
| **Mind2Web** [2] | 2023 | 웹 에이전트 | 수작업 어노테이션 | 제한적 | 마이닝 없음 |
| **WebArena** [3] | 2024 | 현실적 웹 환경 | 수작업 | 도메인 내 | 스킬 라이브러리 없음 |
| **AWM** [16] | 2024 | 궤적에서 루틴 귀납 | 자동 (궤적) | 제한적 | 클러스터링 방법 다름, 전이 결과 더 강 (WA: 0.788 vs 본 논문 MLP: 0.285) |
| **SkillWeaver** [17] | 2025 | API 스타일 스킬 증류 | 자동 (실천) | 웹 에이전트 | 웹사이트 실천 기반, 재사용 가능 API 스킬 |
| **AutoManual** [18] | 2024 | 환경 매뉴얼 구성 | 상호작용 학습 | 제한적 | 인간 가독성 강조 측면 유사 |
| **ICAL** [19] | 2024 | 인지 추상화 증류 | 시연 | VLM 에이전트 | 체화 프로그램 기반 |
| **DigiRL** [37] | 2024 | 온라인 RL | 환경 피드백 | 장치 제어 | 온라인 vs 오프라인이 핵심 차이 |
| **WebRL** [38] | 2025 | 자기 진화 온라인 RL | 온라인 커리큘럼 | ICLR 2025 | 실제 GUI 상호작용 기반 |
| **PAE** [41] | 2025 | Proposer-Agent-Evaluator | 평가자 피드백 | 인터넷 에이전트 | 자율 스킬 발견 |
| **Open-World Skill Discovery** [21] | 2025 | 액션 예측 오류 기반 | 비분할 시연 | ICCV 2025 | 경계 탐지 접근법이 더 정교 |
| **Skills-Coach** [42] | 2026 | GRPO 스타일 루프 | 생성된 태스크 | 제한적 | 훈련 없는 GRPO |
| **본 논문** | 2026 | 3단계 파이프라인 | GUI 궤적 마이닝 | IW→WebArena/BrowseComp+ | 부정적 결과 명시, 진단 연구 |

### 주목할 비교 포인트

**AWM [16] vs 본 논문:**
- AWM은 WebArena에서 0.788 WA 정확도를 달성한 반면, 본 논문의 MLP는 0.285, GRPO는 0.442에 그침
- AWM이 전이 성능에서 우월하며, 이는 AWM의 bigram 전이 학습이 본 논문의 bag-of-actions 클러스터링보다 더 유용한 구조를 포착함을 시사

**DigiRL/WebRL vs 본 논문:**
- 온라인 RL 기반 접근법이 오프라인 보상 모델 기반 접근법보다 강력함을 간접적으로 지지
- 본 논문이 약한 GRPO 전이를 "파이프라인 수준 결과"로 해석하며, GUI 에이전트 RL 전체에 대한 반증이 아님을 명시

**Open-World Skill Discovery [21] vs 본 논문:**
- [21]은 액션 예측 오류를 경계 신호로 사용하여 더 정교한 경계 탐지를 제공
- 본 논문의 $\ell_2$ 거리 기반 휴리스틱보다 학습 기반 접근법이 필요함을 지지

---

## 참고문헌 (논문 내 인용 기준)

본 답변은 다음 자료에 근거합니다:

**주요 논문 (직접 분석 대상):**
- Yuexing Hao, Xiaomin Li. "Automating SKILL.md Generation for Computer-Using Agents via Interaction Trajectory Mining." *arXiv:2606.20363v1* [cs.AI], 18 Jun 2026. (NeurIPS 2026 제출)

**논문 내 핵심 인용 문헌:**
- [1] Yao et al. "WebShop." *NeurIPS*, 2022.
- [2] Deng et al. "Mind2Web." *NeurIPS*, 2023.
- [3] Zhou et al. "WebArena." *ICLR*, 2024.
- [16] Wang et al. "Agent Workflow Memory." *arXiv:2409.07429*, 2024.
- [17] Zheng et al. "SkillWeaver." *arXiv:2504.07079*, 2025.
- [21] Deng et al. "Open-World Skill Discovery from Unsegmented Demonstration Videos." *ICCV*, 2025.
- [36] Eysenbach et al. "The Information Geometry of Unsupervised Reinforcement Learning." *ICLR*, 2022.
- [37] Bai et al. "DigiRL." *NeurIPS*, 2024.
- [38] Qi et al. "WebRL." *ICLR*, 2025.
- [41] Zhou et al. "Proposer-Agent-Evaluator (PAE)." *ICML*, 2025.
- [43] Truong et al. "Selective Review of Offline Change Point Detection Methods." *Signal Processing*, 2020.
- [44] Dowson & Landau. "The Fréchet Distance Between Multivariate Normal Distributions." *Journal of Multivariate Analysis*, 1982.
- [45] Peyré & Cuturi. "Computational Optimal Transport." *Foundations and Trends in ML*, 2019.
- [47] Khosla et al. "Supervised Contrastive Learning." *NeurIPS*, 2020.

> **⚠️ 정확도 주의사항:** 이 논문은 *arXiv:2606.20363v1* (2026년 6월 제출)로, 제가 직접 접근한 PDF 문서를 기반으로 분석하였습니다. 논문에 명시되지 않은 외부 비교 데이터(예: 타 논문의 세부 수치)는 논문 내 인용된 내용 범위 내에서만 기술하였으며, 확인되지 않은 주장은 포함하지 않았습니다.
