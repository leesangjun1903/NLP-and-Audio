# AREX: Towards a Recursively Self-Improving Agent for Deep Research 

> **주의사항**: 본 분석은 제공된 PDF 원문(arXiv:2607.21461v2, 2026년 7월 24일)만을 기반으로 작성되었습니다. 논문 외부 데이터나 추측은 명시적으로 구분합니다.

---

## 1. Executive Summary (10문장 이내)

AREX(BAAI, 2026)는 심층 연구(deep research) 태스크에서 **발견-검증 비대칭성(discovery–verification asymmetry)**을 활용하는 재귀적 자기개선(Recursively Self-Improving, RSI) 에이전트이다.  
핵심 통찰은 다수의 제약 조건을 동시에 충족하는 답을 발견하는 것은 비용이 크지만, 후보 답을 제약 조건별로 검증하는 것은 상대적으로 용이하다는 점이다.  
AREX는 내부 연구 루프(inner research loop)와 외부 자기개선 루프(outer self-improvement loop)를 계층적으로 결합하여 검증을 연구 라운드 간 전환 신호로 활용한다.  
장기 연구 궤적을 유지하기 위해 자율적 컨텍스트 업데이트(Autonomous Context Updating, ACU) 도구를 학습시켜 외부 모델 없이 궤적을 압축한다.  
훈련은 합성 태스크 기반 지도학습, 에이전틱 중간 훈련, 장기 강화학습의 3단계로 구성된다. 희소한 최종 보상 문제를 완화하기 위해 핵심 스텝(key-step)에 집중적인 훈련 신호를 부여한다.  
모델은 4B 파라미터의 AREX-Turbo(밀집)와 122B 총 파라미터/10B 활성 파라미터의 AREX-Base(MoE)로 구체화된다.  
BrowseComp에서 AREX-Base는 82.5%를 달성하여 유사 규모 오픈소스 모델들을 상회하고 독점 모델과도 경쟁적인 성능을 보인다.  
핵심 실험인 Table 3에 따르면, ACU와 외부 루프를 모두 제거 시 59.6%에서 완전 시스템 82.5%로 22.9 포인트의 절대적 향상이 관측된다.  
저자들은 검증 주도 상태 정제와 스텝-인식 최적화가 신뢰할 수 있는 장기 연구 에이전트 구축의 유망한 방향임을 제시한다.

### 1-1. 연구의 목적과 필요성

**목적**: 복수의 결합된 제약 조건을 충족하는 답을 요구하는 심층 연구 태스크에서, 단순히 더 오래 탐색하는 것이 아니라 부분적으로 검증된 상태를 이용해 다음 연구 문제를 더 정밀하게 정의하는 재귀적 자기개선 에이전트를 설계하는 것.

**필요성** (p.2, Introduction):
- 기존 시스템은 단일 검색 궤적을 더 긴 추론/도구 사용/컨텍스트로 연장하는 방식에 집중하나, 이는 초기 오류의 지속, 소진된 방향의 재방문, 부분적으로만 유효한 후보의 조기 수락 등의 문제를 해결하지 못함
- **발견-검증 비대칭성**: 모든 제약 조건을 만족하는 답 발견은 탐색 비용이 크지만, 후보 검증은 제약별로 분해하여 처리 가능
- 기존 검증 활용 방식은 완료된 후보 궤적 순위 매기기 또는 진행 중인 궤적 내 결정 정제에 국한되며, 연구 라운드 간 전환 신호로서의 검증 활용은 미개척

---

## 2. 핵심 주장과 근거 표

| # | 핵심 주장 | 근거/증거 | 위치 |
|---|-----------|-----------|------|
| 1 | 발견-검증 비대칭성이 RSI 프레임워크의 동기 | 심층 연구에서 후보 검증은 제약별 분해로 단순화 가능 | p.1-2, Introduction |
| 2 | ACU가 연구 성능을 향상시킴 | BrowseComp에서 ACU 적용 시 59.6→71.4 (+11.8p) | p.13, Table 3 |
| 3 | 외부 자기개선 루프가 추가 이득을 제공 | ACU 있을 때 루프 추가 시 71.4→82.5 (+11.1p) | p.13, Table 3 |
| 4 | 핵심 스텝 집중 감독이 가장 큰 훈련 이득 | 랜덤 스텝 재현 대비 82.5 vs 74.1 (-8.4p) | p.14, Table 4 |
| 5 | 단계적 다중 라운드 능력 훈련이 혼합 직접 훈련보다 우수 | 82.5 vs 77.5 (-5.0p) | p.14, Table 4 |
| 6 | 스텝-인식 RL이 표준 GRPO보다 우수 | 82.5 vs 79.4 (-3.1p) | p.14, Table 4 |
| 7 | 신뢰도 점수가 올바른 출력과 잘 분리됨 | 정답의 95.9%(ACU 포함)가 신뢰도 90-100에 집중 | p.13, Figure 3 |
| 8 | 핵심 스텝은 일반 스텝보다 학습이 어려움 | 키 스텝 평균 손실 0.277-0.300 vs 일반 0.232 | p.14, Figure 4 |
| 9 | AREX-Base(10B 활성)가 Qwen3.5-397B 초과 | 다수 벤치마크에서 더 큰 모델 대비 우수 | p.11, Table 1 |
| 10 | ACU는 주로 능동적 연구 운영으로 활용 | 업데이트의 0.01%만 128K 한계에서 발생 | p.12, Table 2 |

---

## 2-1. 해결 문제, 제안 방법, 모델 구조, 성능 향상 및 한계

### 해결하고자 하는 문제

심층 연구 태스크에서 에이전트가 다수의 결합 제약 조건을 충족하는 답을 찾아야 할 때 발생하는 세 가지 핵심 문제:

1. **초기 오류의 전파**: 단일 탐색 궤적에서 초기 오류가 지속됨
2. **장기 컨텍스트 관리**: 궤적이 길어질수록 누적된 노이즈, 구식 계획, 중복 관찰이 후속 추론을 방해
3. **희소 보상의 신용 할당**: 장기 궤적에서 최종 보상만으로는 어떤 중간 행동이 결정적 진전을 만들었는지 알기 어려움

### 제안하는 방법 (수식 포함)

#### 내부 연구 루프 (Inner Research Loop)

재귀 라운드 $k$의 스텝 $t$에서 상호작용 궤적:

$$h_t^{(k)} = \left[\left(m_i^{(k)}, a_i^{(k)}, o_i^{(k)}\right)\right]_{i=1}^{t} \tag{1}$$

정책과 환경의 상호작용:

$$\left(m_{t+1}^{(k)}, a_{t+1}^{(k)}\right) = \pi_\theta\left(x, q^{(k)}, h_t^{(k)}\right), \quad o_{t+1}^{(k)} = \mathcal{T}\left(a_{t+1}^{(k)}\right) \tag{2}$$

궤적 업데이트:

$$h_{t+1}^{(k)} = h_t^{(k)} \oplus \left(m_{t+1}^{(k)}, a_{t+1}^{(k)}, o_{t+1}^{(k)}\right) \tag{3}$$

#### 자율적 컨텍스트 업데이트 (ACU)

누적 궤적을 압축된 연구 상태로 변환:

$$z_t^{(k)} = f_\theta\left(h_t^{(k)}\right) \tag{4}$$

가장 최근 컨텍스트 업데이트가 스텝 $\tau \leq t$에서 발생했을 때 유효 컨텍스트:

$$\bar{h}_t^{(k)} = z_\tau^{(k)} \oplus \left[\left(m_i^{(k)}, a_i^{(k)}, o_i^{(k)}\right)\right]_{i=\tau+1}^{t} \tag{5}$$

업데이트가 없을 경우:

$$\bar{h}_t^{(k)} = h_t^{(k)} \tag{6}$$

이후 행동 생성:

$$\left(m_{t+1}^{(k)}, a_{t+1}^{(k)}\right) = \pi_\theta\left(x, q^{(k)}, \bar{h}_t^{(k)}\right) \tag{7}$$

#### 구조화된 답변 외재화 (Structured Answer Externalization)

내부 루프의 출력:

$$r^{(k)} = F_\theta\left(\bar{h}_{T_k}^{(k)}\right) = \left(y^{(k)}, \mathcal{E}^{(k)}, s^{(k)}\right) \tag{8}$$

여기서 $y^{(k)}$는 잠정 답변, $\mathcal{E}^{(k)}$는 지지 증거, $s^{(k)} \in [0, 100]$은 신뢰도 점수.

#### 외부 자기개선 루프 (Outer Self-Improvement Loop)

낮은 신뢰도 결과에 대한 궤적 평가:

$$g^{(k)} = G_\theta\left(x, r^{(k)}, \bar{h}_{T_k}^{(k)}\right) = \left(v^{(k)}, \mathcal{P}^{(k)}, \mathcal{I}^{(k)}, q^{(k+1)}\right) \tag{9}$$

완전한 결정 규칙:

$$d^{(k)} = \begin{cases} \text{Accept}, & s^{(k)} \geq \tau, \\ \text{Refine}, & s^{(k)} < \tau \land v^{(k)} = 1, \\ \text{Restart}, & s^{(k)} < \tau \land v^{(k)} = 0. \end{cases} \tag{10}$$

Refine 선택 시 다음 라운드 초기화:

$$h_0^{(k+1)} = \text{Refresh}\left(\bar{h}_{T_k}^{(k)}, \mathcal{P}^{(k)}, \mathcal{I}^{(k)}\right) \tag{11}$$

Restart 선택 시:

$$h_0^{(k+1)} = \text{Init}(x) \tag{12}$$

#### 훈련 파이프라인

**태스크 합성**: 목표 엔티티 $y$에서 검증 가능한 제약 조건 집합 추출:

$$\mathcal{C}(y) = \{c_1, c_2, \ldots, c_n\} \tag{13}$$

최종 쿼리 생성:

$$x = f(y, \mathcal{C}') \tag{14}$$

**교사 궤적 수집**:

$$\tau_i \sim \pi_{\text{teacher}}(\tau \mid x) \tag{16}$$

신뢰도 필터링:

$$s_{\text{conf}} < \tau_{\text{conf}} \Rightarrow \text{제거} \tag{17}$$

최종 궤적 데이터셋:

$$\mathcal{D}_{\text{traj}} = \{(x, \tau) \mid V(x, \tau) = 1\} \tag{18}$$

**핵심 스텝 목적 함수**:

$$\mathcal{L}_{\text{key}} = -\mathbb{E}_{s_j \sim \mathcal{K}}\left[\frac{1}{|s_j|}\sum_{k=1}^{|s_j|} \log \pi_\theta(a_{j,k} \mid c_{j,k})\right]$$

**스텝-인식 정책 최적화**:

토큰 수준 확률 비율:
$$r_{i,j,k}(\theta) = \frac{\pi_\theta(a_{i,j,k} \mid c_{i,j,k})}{\pi_{\theta_{\text{old}}}(a_{i,j,k} \mid c_{i,j,k})}$$

길이 정규화 스텝 수준 정책 비율(기하 평균):
$$\rho_{i,j}(\theta) = \exp\left(\frac{1}{L_{i,j}}\sum_{k=1}^{L_{i,j}} \log r_{i,j,k}(\theta)\right)$$

계층적 정규화 목적:

$$\mathcal{L}_{\text{step}} = -\mathbb{E}_{x \sim \mathcal{D}}\left[\frac{1}{G}\sum_{i=1}^{G}\frac{1}{M_i}\sum_{j=1}^{M_i} \min\left(\rho_{i,j}(\theta)A_{i,j},\ \text{clip}\left(\rho_{i,j}(\theta), 1-\epsilon, 1+\epsilon\right)A_{i,j}\right)\right]$$

KL 페널티 포함 최종 목적:

$$\mathcal{L} = \mathcal{L}_{\text{step}} + \beta_{\text{KL}}\mathbb{E}\left[D_{\text{KL}}\left(\pi_\theta(\cdot \mid c) \| \pi_{\text{ref}}(\cdot \mid c)\right)\right]$$

그룹 상대적 결과 이점:

$$A_i^{\text{out}} = \frac{R_i - \mu_R}{\sigma_R + \epsilon}$$

핵심 스텝 보너스(성공 궤적에서만):

$$\tilde{B}_{i,j} = \mathbb{I}[R_i > 0] \cdot B_{i,j}$$

최종 스텝 수준 이점:

$$A_{i,j} = A_i^{\text{out}} + \lambda_{\text{key}}\tilde{B}_{i,j}$$

### 모델 구조

| 항목 | AREX-Turbo | AREX-Base |
|------|------------|-----------|
| 백본 | Qwen3.5-4B | Qwen3.5-122B-A10B |
| 유형 | Dense | MoE (Mixture-of-Experts) |
| 총 파라미터 | 4B | 122B |
| 활성 파라미터 | 4B | 10B |
| 최대 내부 루프 턴 | 300 | 300 |
| 최대 외부 루프 라운드 | 5 | 5 |
| 컨텍스트 윈도우 | 128K 토큰 | 128K 토큰 |

### 성능 향상

| 벤치마크 | AREX-Turbo | AREX-Base | 주요 비교 대상 |
|----------|-----------|-----------|--------------|
| BrowseComp | 70.7 | 82.5 | Qwen3.5-35B: 61.0 |
| GAIA | 81.6 | 85.4 | Qwen3.5-397B: 83.5 |
| xbench-2510 | 57.0 | 71.0 | Qwen3.5-35B: 50.3 |
| DeepSearchQA | 78.5 | 89.9 | Qwen3.5-397B: 82.1 |
| WideSearch-en | 68.5 | 82.0 | Kimi-K2.6: 80.8 |
| HLE (tool) | 40.6 | 52.4 | DeepSeek-V4-Pro: 48.2 |

*(Table 1, p.11)*

### 한계

논문에서 명시적으로 언급된 한계:
1. 핵심 스텝 탐지가 규칙 기반(rule-based)으로, 스텝 유용성 추정의 보다 일반적이고 자율적인 메커니즘 필요 (p.14, Conclusion)
2. 궤적 자기 증류(self-distillation)는 예비 탐색 수준이며, 완전 시스템과의 조합 효과 미검증 (p.19, Appendix B)
3. 자기 생성 궤적이 교사 모델의 편향과 실패 모드를 상속하거나 증폭할 수 있음 (p.19)
4. 외부 자기개선 루프의 최대 라운드(5회) 제한의 영향에 대한 분석 부재

---

## 3. 각 주장의 페이지/Figure/Table 번호

| 주장 | 위치 |
|------|------|
| 발견-검증 비대칭성 정의 | p.1-2, Abstract & Introduction |
| RSI 프레임워크 구조 | p.3, Figure 2 |
| 내부 루프 수식 (1)-(8) | p.4-5 |
| ACU 수식 (4)-(7) | p.5 |
| 외부 루프 수식 (9)-(12) | p.6 |
| 태스크 합성 방법 수식 (13)-(18) | p.6-7 |
| 핵심 스텝 목적 함수 | p.9 |
| 스텝-인식 RL 수식 | p.9-10 |
| ACU 동작 분석 | p.12, Table 2 |
| ACU vs 외부 루프 ablation | p.13, Table 3 |
| 신뢰도 분포 분석 | p.13, Figure 3 |
| 스텝 손실 분석 | p.14, Figure 4 |
| 훈련 ablation 결과 | p.14, Table 4 |
| 전체 벤치마크 비교 | p.11, Table 1 |
| 예비 자기 증류 실험 | p.18-19, Appendix B, Table 5 |

---

## 4. 저자 보고 결과 vs. 내 해석 분리

### 저자가 직접 보고한 결과

**연구 주제**: 다중 제약 조건 심층 연구를 위한 재귀적 자기개선 에이전트

**방법**:
- 이중 루프 구조(내부: 증거 수집/잠정 답변, 외부: 제약별 검증/재목표화)
- 자율적 컨텍스트 업데이트 도구(update_context)를 통한 궤적 압축
- 핵심 스텝 집중 감독 + 스텝-인식 강화학습

**결과 (저자 직접 보고)**:
- BrowseComp: AREX-Base 82.5 (Table 1, p.11)
- ACU+외부 루프 전체 시스템: 82.5 vs 기준 59.6, 차이 22.9p (Table 3, p.13)
- 핵심 스텝 평균 손실 (0.277-0.300) > 일반 스텝 (0.232) (Figure 4, p.14)
- BrowseComp에서 ACU는 80.3% 케이스에서 호출, 평균 25,721 토큰에서 발생 (Table 2, p.12)

### 내 해석 (논문이 직접 주장하지 않은 부분)

⚠️ **주의: 아래는 저자 보고가 아닌 내 해석입니다**

- **ACU의 효과 메커니즘**: ACU가 효과적인 이유는 단순히 컨텍스트를 줄이는 것이 아니라, 연구 목표 중심으로 정보를 재구조화하기 때문으로 보임. 특히 미해결 제약(95.5%)과 다음 단계 계획(96.4%)의 높은 보존율은 ACU가 미래 행동 생성에 최적화된 표현을 만든다는 것을 시사
- **스텝-인식 RL의 제한적 이득(+3.1p)**: 중간 훈련에서 이미 핵심 스텝 학습이 상당 부분 이루어진 후 RL이 적용되기 때문에 상대적으로 작은 이득으로 나타날 수 있음. 순서 의존성이 있을 가능성
- **MoE 구조 선택**: 122B 총 파라미터 중 10B만 활성화하는 MoE 구조는 추론 효율성을 위한 실용적 선택이며, 이는 AREX-Base가 매우 큰 밀집 모델과 경쟁할 수 있는 핵심 요인으로 해석됨

---

## 5. 통계적으로 취약한 부분과 비교 불가능한 수치

### ⚠️ 통계적으로 취약한 부분

| 항목 | 취약점 |
|------|--------|
| BrowseComp 단일 벤치마크 의존 ablation | 모든 ablation (Table 3, 4)이 BrowseComp 단 하나의 벤치마크에서만 수행됨. 다른 벤치마크로의 일반화 여부 불명확 |
| 신뢰도 점수 임계값 $\tau$ 미공개 | 외부 루프의 Accept/Refine/Restart 결정에 사용되는 $\tau$ 값이 논문에 명시되지 않음 |
| ACU 동작 분석이 BrowseComp만 대상 | Table 2의 ACU 동작 통계가 BrowseComp에만 한정. 다른 태스크 특성에서의 동작 미보고 |
| 분산/표준편차 미보고 | Table 1, 3, 4의 모든 성능 수치에 표준편차, 신뢰구간, 또는 반복 실험 횟수가 보고되지 않음 |
| 핵심 스텝 탐지기의 정밀도/재현율 미보고 | "high-precision rule-based detectors"라고 주장하지만 실제 탐지 성능 지표 없음 |
| 자기 증류 실험의 단일 설정 | Table 5는 ACU, 외부 루프, 핵심 스텝 감독 없는 단순화 설정에서만 수행 |

### ⚠️ 비교 불가능한 수치

| 항목 | 비교 불가능 이유 |
|------|----------------|
| HLE 점수의 * 표시 | Table 1 각주: `*`는 전체 HLE 기준, 미표시는 텍스트 전용 부분집합 기준. 동일 지표로 비교 불가 |
| xbench-2510에서의 Kimi-K2.6 (90.0) | Figure 1에는 xbench에서 Kimi-K2.6 미표시. Table 1에서 90.0은 다른 버전/설정일 수 있음 |
| WideSearch 영어 부분집합 | AREX는 영어 부분집합(WideSearch-en)만 보고. 타 모델의 전체/부분집합 여부 불명확 |
| 프론티어 모델(GPT-5.4, Opus-4.6 등)의 GAIA 점수 누락 | Table 1에서 GPT-5.4, Opus-4.6의 GAIA 점수가 "–"로 표시 |
| 검색 환경의 차이 | 각 모델이 사용하는 검색 엔진, 웹 인덱스, 도구 사양이 통일되지 않았을 가능성 |
| 외부 루프 최대 5라운드 제한 | "following Team et al. (2026)" 기준이나, 타 모델들의 동등한 제한 적용 여부 불명확 |

---

## 6. 논문이 답하지 않는 질문

1. **신뢰도 임계값 $\tau$**: Accept/Refine/Restart 결정에 사용되는 정확한 신뢰도 임계값이 공개되지 않음
2. **$\lambda_{\text{key}}$ 하이퍼파라미터**: 핵심 스텝 보너스의 강도를 제어하는 $\lambda_{\text{key}}$의 구체적 값과 민감도 분석 부재
3. **훈련 데이터 규모**: 합성 태스크 및 교사 궤적 데이터셋의 구체적 크기(예: 몇 개의 태스크, 몇 개의 궤적)가 공개되지 않음
4. **교사 모델 정체**: 교사 궤적 수집에 사용된 "strong teacher models"의 구체적 모델명 미공개
5. **외부 루프 라운드별 기여**: 5회 최대 라운드 중 라운드별 성능 향상 곡선 미보고
6. **계산 비용**: 단일 라운드 대비 RSI 적용 시 추론 시간 및 비용 증가량 미보고
7. **ACU 품질 평가**: 압축된 개선 상태($z_t^{(k)}$)가 원본 궤적 대비 정보 손실량을 정량적으로 평가하는 방법 부재
8. **실패 케이스 분석**: AREX가 틀리는 사례의 정성적 분류(false positive 신뢰도 포함) 부재
9. **다국어/다도메인 일반화**: 영어 중심 벤치마크 외 다국어 또는 특수 도메인 성능 미보고
10. **RSI의 수렴 조건**: 재귀 라운드가 수렴하는 이론적 조건이나 실증적 분석 부재
11. **탐지 규칙의 구체적 내용**: 핵심 스텝을 식별하는 규칙 기반 탐지기의 상세 규칙 미공개

---

## 7. 가장 중요한 그림 5개 해석

### Figure 1 (p.1): AREX 벤치마크 성능 요약

**내용**: 6개 벤치마크(BrowseComp, GAIA, xbench-2510, DeepSearchQA, WideSearch, HLE)에서 AREX-Base/Turbo와 다양한 비교 모델의 점수를 수평 막대로 시각화.

**해석**:
- AREX-Base(122B 총/10B 활성)가 BrowseComp(82.5), GAIA(85.4), WideSearch(82.0), HLE(52.4)에서 동급 오픈소스 모델 중 최상위권
- AREX-Turbo(4B)는 Qwen3.5-35B 대비 BrowseComp, GAIA, xbench-2510, DeepSearchQA, WideSearch 5개 벤치마크에서 우위
- **주목할 점**: xbench-2510에서 DeepSeek-Pro(80.0)와 AREX-Base(71.0)의 상당한 격차(9.0p)는 이 벤치마크에서의 상대적 약점을 시사하나, 이는 DeepSeek-Pro가 훨씬 큰 모델이라는 점에서 맥락화 필요

### Figure 2 (p.3): AREX의 재귀적 자기개선 프레임워크

**내용**: 내부 연구 루프(Current Research Objective → Research Action → Observation → Intermediate Analysis → Update Context)와 외부 자기개선 루프(Structured Finish → Confidence Score → Accept/Refine/Restart)의 계층 구조를 도식화.

**해석**:
- **Refreshed Research State의 6개 요소**(Verified Findings, Current Candidates, Unresolved Constraints, Validity Concerns, Rejected Candidates, Next-Step Plan)는 연구 진행 상태를 포괄적으로 표현하는 설계 철학을 보여줌
- 외부 루프의 "Recoverable" 판단이 Refine/Restart를 분기하는 핵심 결정점임을 명확히 시각화
- **중요 관찰**: update_context가 "optional"로 표시되어 있어, ACU가 필수가 아닌 선택적 도구임을 시사. 이는 Table 2에서 80.3% 케이스에서만 호출된 결과와 일치

### Figure 3 (p.13): 정답/오답 출력의 신뢰도 분포

**내용**: ACU 유무 조건에서 정답(Correct)/오답(Incorrect) 출력의 신뢰도 구간(<60, 60-90, 90-100) 분포를 정규화된 막대 차트로 표시.

**해석**:
- ACU 포함 시 정답의 95.9%가 신뢰도 90-100에 집중(ACU 없이 89.3%)
- 오답에서 저신뢰도(<60) 비율: ACU 없이 61.0%, ACU 포함 55.2%
- **중요 함의**: 신뢰도 점수가 외부 루프의 Accept/Refine/Restart 결정을 위한 유효한 신호임을 경험적으로 지지. 단, 오답의 33.0%(ACU 포함)가 여전히 높은 신뢰도(90-100)를 보이는 점은 신뢰도 점수의 한계(false confidence)를 드러냄
- ACU가 신뢰도 점수의 보정(calibration)도 개선함을 시사

### Figure 4 (p.14): 전체 궤적 중간 훈련 후 평균 스텝 손실

**내용**: 일반 스텝(Ordinary steps)과 세 종류의 핵심 스텝(Evidence discovery, Path rejection and redirection, Key context-update)의 평균 스텝 손실 수평 막대 비교.

**해석**:
- 수치: 일반 스텝 0.232, 증거 발견 0.277(+19%), 경로 거부/전환 0.298(+28%), 핵심 컨텍스트 업데이트 0.300(+29%)
- **핵심 인사이트**: 전체 궤적 감독 이후에도 핵심 스텝이 상대적으로 "underfit" 상태임을 증명. 이는 핵심 스텝 집중 감독의 필요성을 사후적으로 정당화
- 경로 거부/전환과 컨텍스트 업데이트의 손실이 유사한 수준(0.298 vs 0.300)이라는 점은 이 두 행동 유형이 모델에게 동등하게 어려운 과제임을 시사
- **한계**: 손실 값 자체가 절대적 난이도를 의미하지 않으며, 데이터 분포와 레이블 노이즈의 영향을 받을 수 있음

### Table 3 (p.13): ACU와 외부 자기개선 루프의 효과 분리

**내용**: ACU 유무 × 외부 루프 유무의 2×2 조합에서 BrowseComp 정확도 비교.

| 방법 | 설정 | 정확도 |
|------|------|--------|
| AREX w/o ACU | w/o outer loop | 59.6 |
| AREX w/o ACU | w/ outer loop | 69.8 |
| AREX w/ ACU | w/o outer loop | 71.4 |
| AREX w/ ACU | w/ outer loop | **82.5** |

**해석**:
- **ACU의 단독 기여**: +11.8p (59.6→71.4), 외부 루프의 단독 기여: +10.2p (ACU 없이) ~ +11.1p (ACU 있이)
- 두 구성 요소가 거의 독립적이고 상가적(additive)으로 작동함을 시사 (11.8 + 11.1 ≈ 22.9p 전체 이득)
- **그러나 주의**: 이 상가성이 통계적으로 유의한지, 또는 다른 벤치마크에서도 성립하는지 검증되지 않음
- 외부 루프의 효과(~10p 수준)가 ACU 없이도 상당함은, 단순한 재시도(retry)만으로도 상당한 개선이 가능함을 시사

---

## 8. 결론, 시사점, 후속 연구

### 8-1. 저자 제시 시사점 및 후속 연구 계획

**저자 직접 제시 (p.14, Conclusion)**:
- 검증 주도 상태 정제(verification-guided state refinement)와 스텝-인식 최적화(step-aware optimization)가 신뢰할 수 있는 장기 연구 에이전트 구축의 유망한 방향
- **후속 연구 계획**: 스텝 유용성 추정과 세밀한 훈련 신호 할당을 위한 더 일반적이고 자율적인 메커니즘 연구
- 핵심 스텝을 중간 행동 전체에 동등하게 처리하지 않고 식별·강화하는 것의 중요성

**예비 탐색 결과 (Appendix B, p.18-19)**:
- 궤적 자기 증류(trajectory self-distillation)가 BrowseComp에서 52.3→57.1 (+4.8p)의 향상을 보임
- 이를 완전 시스템과의 조합에 대한 미래 방향으로 제시

### 8-1. 모델의 일반화 성능 향상 가능성 (심층 분석)

#### 현재 일반화의 강점

논문의 결과는 여러 벤치마크(BrowseComp, GAIA, xbench, DeepSearchQA, WideSearch, HLE)에서 일관된 향상을 보여, **태스크 유형 간 일반화**가 어느 정도 달성되었음을 시사한다 (Table 1, p.11). 특히 AREX-Turbo(4B)가 Qwen3.5-35B 대비 5개 벤치마크에서 우위를 보이는 것은 파라미터 효율적 일반화를 나타낸다.

#### 일반화 관련 주요 우려점

1. **단일 언어 벤치마크**: WideSearch는 영어 부분집합만 보고. 다국어 환경에서의 RSI 메커니즘의 효과 미검증
2. **도메인 특수성**: 태스크 합성 시 "browse-intensive", "reasoning-intensive", "scientific literature" 세 카테고리로 한정. 법률, 의료, 코드 등 특수 도메인 일반화 미검증
3. **훈련 데이터와 평가 분포의 잠재적 중복**: 합성 태스크가 실제 벤치마크와 유사한 패턴을 가질 경우, 보고된 성능이 진정한 일반화보다 과적합을 반영할 수 있음
4. **Ablation의 BrowseComp 편중**: 모든 구성 요소 ablation이 BrowseComp 하나에서만 수행. 다른 벤치마크에서의 동일한 패턴 보장 없음

#### 일반화 성능 향상을 위한 추가 후속 연구 방향

**제안 1: 도메인 적응형 제약 표현**

현재 AREX의 제약 집합 $\mathcal{C}(y) = \{c_1, \ldots, c_n\}$은 도메인 독립적으로 정의됨. 도메인별 제약 스키마를 학습하는 메타 학습(meta-learning) 접근으로 확장 가능:

$$\mathcal{C}_{\mathcal{D}}(y) = g_\phi(\mathcal{D}) \odot \mathcal{C}(y)$$

여기서 $g_\phi$는 도메인 $\mathcal{D}$에 따른 제약 가중치를 학습.

**제안 2: 분포 외(out-of-distribution) 일반화 평가**

훈련 태스크와 의미적으로 거리가 먼 벤치마크에서의 성능을 체계적으로 측정하는 OOD 평가 프로토콜 필요.

**제안 3: 신뢰도 점수 보정 (Calibration) 개선**

Figure 3에서 오답의 33.0%가 여전히 고신뢰도(90-100)를 보임. 이는 외부 루프의 결정 신뢰성을 저하시킴. 보정 손실(calibration loss)을 명시적 훈련 목적에 포함:

$$\mathcal{L}_{\text{cal}} = \mathbb{E}\left[\left(s^{(k)} - \mathbb{I}[y^{(k)} = y^*]\right)^2\right]$$

**제안 4: 적응적 재귀 라운드**

현재 최대 5라운드로 고정됨. 태스크 복잡도에 따라 라운드 수를 동적으로 결정하는 메커니즘 도입시 효율성과 일반화 모두 개선 가능.

---

### 8-2. 2020년 이후 관련 최신 연구 비교 분석

> **주의**: 아래 비교는 논문의 Reference 섹션에 명시된 연구들을 기반으로 합니다. 논문 외부 데이터에 의한 추가 연구 비교는 제한합니다.

#### 논문이 인용한 주요 관련 연구와의 비교

| 연구 | 핵심 방법 | AREX와의 차이 | 출처 |
|------|-----------|--------------|------|
| WebGPT (Nakano et al., 2021) | 브라우저 도움 QA + 인간 피드백 | 단일 궤적, 재귀 자기개선 없음 | arXiv:2112.09332 |
| ReAct (Yao et al., 2023) | 추론-행동 시너지 | 단일 루프, 검증 기반 전환 없음 | ICLR 2023 |
| MemGPT (Packer et al., 2023) | LLM을 OS처럼 메모리 관리 | 외부 메모리, 연구 목표 중심 압축 없음 | arXiv:2310.08560 |
| Let's Verify Step by Step (Lightman et al., 2024) | 과정 감독(PRM) | 수학 도메인 특화, 연구 에이전트 아님 | ICLR 2024 |
| Search-R1 (Jin et al., 2025) | RL로 검색 활용 LLM 훈련 | 단일 탐색 궤적 강화 | COLM 2025 |
| Math-Shepherd (Wang et al., 2024) | 스텝별 검증 및 강화 | 수학 특화, 웹 연구 에이전트 아님 | ACL 2024 |
| HiAgent (Hu et al., 2025) | 계층적 작업 메모리 관리 | 에이전트 태스크 대상, 심층 연구 특화 없음 | ACL 2025 |
| WebDancer (Wu et al., 2025a) | 자율 정보 탐색 에이전시 | AREX보다 단순한 루프 구조 | NeurIPS 2025 |
| MEM1 (Zhou et al., 2026) | 메모리-추론 시너지 학습 | 효율적 장기 에이전트, RSI 메커니즘 없음 | ICLR 2026 |
| Beyond Ten Turns (Gao et al., 2026) | 대규모 비동기 RL | 장기 탐색 강화, 재귀 검증 루프 없음 | ICLR 2026 |
| MiroThinker-H1 (Team et al., 2026) | 검증을 통한 헤비 리서치 에이전트 | 검증 활용하나 외부 루프 설계 상이 | arXiv:2603.15726 |
| ReSum (Wu et al., 2025b) | 컨텍스트 요약 기반 장기 검색 | 고정 임계값 기반 요약, 목표 중심 아님 | arXiv:2509.13313 |

#### AREX가 기존 연구 대비 차별화되는 핵심 3가지

1. **검증을 루프 간 전환 신호로 활용**: 기존 연구들은 검증을 최종 필터(outcome ranking) 또는 로컬 행동 비판(action critique)으로 활용. AREX는 검증을 연구 라운드 간 전환 연산자로 사용하여 제약별 미해결 상태를 다음 연구 목표로 변환
2. **목표 중심 자율 컨텍스트 업데이트**: MemGPT, ReSum 등 기존 메모리 관리 연구와 달리, AREX의 ACU는 외부 모델 없이 현재 연구 목표를 중심으로 궤적을 압축하며, 모델이 스스로 업데이트 시점을 결정
3. **계층적 스텝-인식 RL**: Turn-level credit assignment(Zeng et al., 2025a)를 심층 연구 궤적에 맞게 확장하여, 핵심 스텝에 선택적 보너스를 부여하면서도 최종 보상을 주 최적화 신호로 유지

#### 앞으로의 연구에 미치는 영향

1. **에이전트 설계 패러다임 전환**: 단순히 더 많은 토큰/라운드를 투입하는 것이 아니라, 검증-기반 상태 관리를 통해 연구 진행을 체계적으로 제어하는 패러다임을 제시
2. **Process Supervision의 에이전틱 확장**: Lightman et al.(2024)의 과정 감독 아이디어를 장기 연구 궤적의 핵심 스텝 탐지로 확장하며, 이는 다양한 에이전틱 태스크에 적용 가능한 일반적 원리를 제공
3. **MoE 활성화 효율과 RSI의 결합**: 10B 활성 파라미터로 훨씬 큰 모델과 경쟁하는 결과는, 계산 효율적 모델 구조와 정교한 에이전틱 프레임워크의 조합이 단순 스케일 업의 대안이 될 수 있음을 시사

#### 앞으로 연구 시 고려할 점

1. **재현성**: 교사 모델, 훈련 데이터 크기, 신뢰도 임계값 등 핵심 하이퍼파라미터가 미공개. 후속 연구에서는 완전한 재현 가능성을 위한 상세 사항 공개 필요
2. **핵심 스텝 탐지의 자동화**: 현재 규칙 기반 탐지기의 한계를 극복하기 위해, 궤적 성공 여부와 스텝 기여도를 연결하는 학습 기반 탐지기 개발이 중요한 과제
3. **신뢰도 보정**: Figure 3의 false confidence 문제(오답의 33%가 고신뢰도)를 해결하지 않으면 외부 루프의 결정 신뢰성이 제한됨
4. **다국어 및 실시간 웹 환경**: 정적 벤치마크가 아닌 동적 웹 환경과 다국어 태스크로의 확장 시 RSI 메커니즘의 강건성 검증 필요
5. **안전성 및 편향**: 자기 증류 및 자기개선 루프에서 모델이 자신의 편향을 강화할 가능성에 대한 분석이 필요하며, 특히 "recoverable" 판단 기준이 특정 유형의 오류에 편향될 수 있음

---

## 참고자료 (논문 내 인용 기반)

본 분석에서 직접 인용 또는 참조한 자료:

1. **AREX Team**, "AREX: Towards a Recursively Self-Improving Agent for Deep Research," arXiv:2607.21461v2, BAAI, 2026
2. **Wei et al.** (2025), "BrowseComp: A simple yet challenging benchmark for browsing agents," arXiv:2504.12516
3. **Wong et al.** (2026), "WideSearch: Benchmarking agentic broad info-seeking," ICLR 2026
4. **Gupta et al.** (2026), "DeepSearchQA: Bridging the comprehensiveness gap for deep research agents," arXiv:2601.20975
5. **Center for AI Safety et al.** (2026), "Humanity's Last Exam," Nature, 649(8099):1139–1146
6. **Mialon et al.** (2024), "GAIA: A benchmark for general AI assistants," ICLR 2024
7. **xbench Team** (2025), "xbench-DeepSearch-2510," https://xbench.org/agi/aisearch
8. **Yao et al.** (2023), "ReAct: Synergizing reasoning and acting in language models," ICLR 2023
9. **Nakano et al.** (2021), "WebGPT: Browser-assisted question-answering with human feedback," arXiv:2112.09332
10. **Lightman et al.** (2024), "Let's verify step by step," ICLR 2024
11. **Wang et al.** (2024), "Math-shepherd: Verify and reinforce LLMs step-by-step without human annotations," ACL 2024
12. **Packer et al.** (2023), "MemGPT: Towards LLMs as operating systems," arXiv:2310.08560
13. **Hu et al.** (2025), "HiAgent: Hierarchical working memory management for solving long-horizon agent tasks," ACL 2025
14. **Wu et al.** (2025a), "WebDancer: Towards autonomous information seeking agency," NeurIPS 2025
15. **Wu et al.** (2025b), "ReSum: Unlocking long-horizon search intelligence via context summarization," arXiv:2509.13313
16. **Zhou et al.** (2026), "MEM1: Learning to synergize memory and reasoning for efficient long-horizon agents," ICLR 2026
17. **Gao et al.** (2026), "Beyond ten turns: Unlocking long-horizon agentic search with large-scale asynchronous RL," ICLR 2026
18. **Zeng et al.** (2025a), "Reinforcing multi-turn reasoning in LLM agents via turn-level credit assignment," arXiv:2505.11821
19. **Zeng et al.** (2025b), "Pushing test-time scaling limits of deep search with asymmetric verification," arXiv:2510.06135
20. **Jin et al.** (2025), "Search-R1: Training LLMs to reason and leverage search engines with RL," COLM 2025
21. **Lu et al.** (2026), "Beyond the context window: Scaling agentic RL via end-to-end optimized context compression," ACL 2026
22. **Zhang et al.** (2026), "Memory as action: Autonomous context curation for long-horizon agentic tasks," ACL Findings 2026
23. **Qwen Team** (2026), "Qwen3.5: Towards native multimodal agents," https://qwen.ai/blog?id=qwen3.5
24. **Huang et al.** (2025), "Deep research agents: A systematic examination and roadmap," arXiv:2506.18096
25. **Wang et al.** (2025), "RAGEN: Understanding self-evolution in LLM agents via multi-turn RL," arXiv:2504.20073
