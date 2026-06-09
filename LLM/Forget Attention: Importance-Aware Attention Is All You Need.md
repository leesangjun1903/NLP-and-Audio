# Forget Attention: Importance-Aware Attention Is All You Need

> **참고 자료**
> - Shin, S. & Yang, Y. (2026). *Forget Attention: Importance-Aware Attention Is All You Need*. arXiv:2606.02332v2
> - 논문 내 인용 문헌 전체 (References [1]–[26], 본문 각주 참조)

---

## 1. 핵심 주장 및 주요 기여 요약

### 1.1 핵심 주장

이 논문의 핵심 주장은 다음 한 문장으로 요약된다:

> **Transformer의 "전역 검색(global retrieval)" 능력과 SSM의 "순차적 중요도 신호(sequential importance signal)"를 어텐션 스코어 수준(score level)에서 직접 융합하면, 블록·헤드 수준의 기존 하이브리드보다 근본적으로 우월한 언어 모델을 구성할 수 있다.**

논문은 기존 하이브리드의 근본적 결함을 다음과 같이 진단한다:

| 모델 계열 | 강점 | 약점 |
|:---|:---|:---|
| **Transformer** | 어디든 볼 수 있다 (전역 검색) | 어디를 봐야 할지 모른다 (우선순위 없음) |
| **SSM (Mamba 계열)** | 무엇이 중요한지 안다 (순차 추적) | 한 번 감쇠된 정보를 되찾을 수 없다 |
| **기존 하이브리드 (Jamba, Hymba)** | 두 기제 병용 | 두 기제가 독립적으로 계산 후 결합 → 어텐션 스코어 결정 시 SSM 신호 미반영 |

### 1.2 주요 기여 5가지

1. **Score-level fusion (스코어 수준 융합)** 제안 — SSM 신호를 어텐션 스코어 내부에 직접 추가하는 세 번째 설계 축 정의
2. **Single SDPA 실현** — augmented Q/K 벡터를 통해 추가 커스텀 CUDA 커널 없이 FlashAttention 호환 구현
3. **LAMBADA 성능 향상** — 152M 규모에서 Transformer 대비 +3.4 pp (+24%), Mamba-3 대비 +1.8 pp
4. **NIAH 초고속 수렴** — 학습 Step 1K(전체의 10%)에서 100% 달성, Transformer 대비 7배 빠름
5. **$d_s$ 스케일링 연구** — SSM 채널 차원 $d_s \in \{16, 32, 64, 128\}$에 대한 체계적 ablation 및 스케일별 가이드라인 제시

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

논문이 정의하는 핵심 문제는 **"어텐션 계산 자체에 SSM의 중요도 신호를 통합하는 방법의 부재"**이다.

```
기존 하이브리드의 구조적 한계:
  Block-level (Jamba): [Transformer layer] → [Mamba layer] → [Transformer layer] → ...
                        ← 독립 계산, 출력만 교환 →
  Head-level (Hymba):  [Attn heads ‖ SSM heads] → concat → output
                        ← 어텐션 스코어 결정 전까지 SSM 신호 개입 없음 →
```

이 구조에서는 **각 메커니즘이 독립적으로 출력을 생성한 후 결합**되므로, SSM의 중요도 신호가 어텐션 가중치 결정 자체에 개입하지 못한다.

---

### 2.2 제안 방법: SISA (SSM-Informed Softmax Attention)

#### 2.2.1 SISA 스코어 함수

$$s_{ij}^{\text{SISA}} = \underbrace{\frac{\mathbf{q}_i^\top \mathbf{k}_j}{\sqrt{d_h}}}_{\text{content match}} + \underbrace{\lambda \cdot \bar{\mathbf{C}}_i^\top \bar{\mathbf{B}}_j}_{\text{importance match}}, \quad i \geq j \tag{2}$$

- $\lambda > 0$: 헤드별 학습 가능한 양의 스칼라 (fp32로 유지 필수)
- $\bar{\mathbf{C}}_i, \bar{\mathbf{B}}_j$: SSM 동역학을 담은 투영 벡터
- 인과성(causality)은 SDPA의 내장 causal flag로 보장

#### 2.2.2 SSM 채널 계산

입력 $\mathbf{x}_t$로부터:

**투영(Projection):**

$$\mathbf{B}_t = \mathbf{W}_B \mathbf{x}_t, \quad \mathbf{C}_t = \mathbf{W}_C \mathbf{x}_t \in \mathbb{R}^{d_s}$$

**감쇠(Decay):**

$$\alpha_t = \exp(-\text{softplus}(\mathbf{w}_\alpha^\top \mathbf{x}_t + b_\alpha)) \in (0, 1)$$

초기값 $b_\alpha = -5$ (반감기 약 100 토큰)

**위상(Phase):**
$$\theta_t = \mathbf{W}_\theta \mathbf{x}_t \in \mathbb{R}^{d_s/2}$$

**누적 수량 (FP32):**
$$g_t = \sum_{k \leq t} \log \alpha_k, \quad \Phi_t = \sum_{k \leq t} \theta_k, \quad c = \frac{\max_t g_t + \min_t g_t}{2} \tag{3}$$

**최종 채널:**
$$\bar{\mathbf{C}}_i = e^{g_i - c} \cdot R(\Phi_i) \cdot \mathbf{C}_i, \qquad \bar{\mathbf{B}}_j = e^{-(g_j - c)} \cdot R(\Phi_j) \cdot \mathbf{B}_j \tag{4}$$

여기서 $R(\Phi)$는 블록 대각 $2\times2$ 회전 행렬이며, 두 요소의 역할은:

- **감쇠 항** $e^{g_i - c} \cdot e^{-(g_j-c)} = e^{g_i - g_j}$: 위치 $j$의 중요도가 $i$까지 얼마나 유지되는지 (ALiBi의 고정 기울기와 달리 입력 의존적)
- **회전 항** $R(\Phi_i)^\top R(\Phi_j) = R(\Phi_i - \Phi_j)$: 감쇠가 비슷하더라도 시퀀스 내 역할이 다른 위치를 구분 (데이터 의존적 상대 위치 인코딩)

#### 2.2.3 Augmented Q/K: Single-SDPA 실현

**Proposition 1.** $s = d_h^{1/4}\sqrt{\lambda}$, $\hat{\mathbf{Q}}_i = [\mathbf{q}_i;\, s\bar{\mathbf{C}}_i]$, $\hat{\mathbf{K}}_j = [\mathbf{k}_j;\, s\bar{\mathbf{B}}_j]$ 로 정의하면:

$$\frac{\hat{\mathbf{Q}}_i^\top \hat{\mathbf{K}}_j}{\sqrt{d_h}} = \frac{\mathbf{q}_i^\top \mathbf{k}_j + s^2 \bar{\mathbf{C}}_i^\top \bar{\mathbf{B}}_j}{\sqrt{d_h}} = \frac{\mathbf{q}_i^\top \mathbf{k}_j}{\sqrt{d_h}} + \frac{s^2}{\sqrt{d_h}} \cdot \bar{\mathbf{C}}_i^\top \bar{\mathbf{B}}_j \tag{5}$$

$s^2 = \sqrt{d_h} \cdot \lambda$ 이므로 두 번째 항이 $\lambda \cdot \bar{\mathbf{C}}\_i^\top \bar{\mathbf{B}}\_j$로 환원 → $s_{ij}^{\text{SISA}}$ 성립 $\square$

따라서 전체 레이어 연산은:

$$\mathbf{Y} = \text{SDPA}(\hat{\mathbf{Q}}, \hat{\mathbf{K}}, \mathbf{V},\; \text{scale}=1/\sqrt{d_h},\; \text{causal}=\text{True})$$

> ⚠️ **구현 주의사항**: SDPA scale은 반드시 $1/\sqrt{d_h}$이어야 하며, PyTorch의 기본값인 $1/\sqrt{d_h + d_s}$를 사용하면 SISA 스코어가 깨진다. 또한 $\lambda_{\text{raw}}$는 반드시 fp32로 유지해야 한다 (bf16에서는 AdamW 업데이트 크기 $\sim10^{-5}$가 bf16 최소 표현 단위 $0.0078$보다 작아 $\lambda$가 학습되지 않음).

#### 2.2.4 파라미터 예산

SSM 관련 추가 파라미터 수:

$$P_{\text{SSM}} = 2 \cdot d \cdot h \cdot d_s + d \cdot h + h + d \cdot h \cdot \frac{d_s}{2} + h \tag{6}$$

이 비용은 FFN 차원 $d_{\text{ff}}$를 줄여 상쇄 → 전체 파라미터 수를 Transformer와 동일하게 유지 (152M 기준 차이 288 params, 0.0002%)

---

### 2.3 모델 구조

```
입력 x
  │
  ├─── [Content Attention Stream] ────────────────────┐
  │      WQ→q (+ RoPE)                                │
  │      WK→k (+ RoPE)                                │
  │      WV→v                                         │
  │                                                   │
  ├─── [SSM Signal Stream] ────────────────────────── │
  │      WB→B, WC→C (projection)                      │
  │      wα→αt (decay) ──→ cumsum→ gt                 │
  │      Wθ→θt (phase) ──→ cumsum→ Φt                 │
  │      → C̄i = e^(gi-c)·R(Φi)·Ci                    │
  │      → B̄j = e^(-(gj-c))·R(Φj)·Bj                 │
  │                                                   │
  │   Augmented Q/K 생성:                              │
  │      Q̂i = [qi ; s·C̄i]                            │
  │      K̂j = [kj ; s·B̄j]                            │
  │                                                   ▼
  └────────────────────────────→ SDPA(Q̂, K̂, V, scale=1/√dh, causal=True)
                                          │
                                     Residual + SwiGLU FFN
                                          │
                                       Output x'
```

---

### 2.4 성능 향상

#### 주요 벤치마크 결과 (152M, 5B tokens)

| 모델 | LAMBADA↑ | NIAH↑ | HellaSwag↑ | ARC-E↑ | WinoG↑ |
|:---|:---:|:---:|:---:|:---:|:---:|
| Transformer | 13.9 | **100.0** | 25.4 | 33.3 | 51.2 |
| Mamba-2 | 12.7 | 82.5 | 26.5 | **36.4** | 51.4 |
| Mamba-3 | 15.5 | 99.0 | 26.0 | 34.9 | **52.7** |
| **SISA $d_s=16$** | **17.3** | **100.0** | **26.9** | 34.7 | 52.5 |

#### 스케일별 성능 요약

| 규모 | SISA 최적 $d_s$ | LAMBADA | NIAH |
|:---|:---:|:---:|:---:|
| 50M | 64 | 14.4 (+1.0 vs. Transformer) | 100% |
| 152M | 16 | **17.3 (+3.4 vs. Transformer)** | 100% |
| 369M | 128 | 14.8 (Mamba-3: 17.4) | **100%** (Mamba-3: 86.5%) |

#### NIAH 수렴 속도 비교 (152M)

| Step | Transformer | **SISA** | Mamba-2 | Mamba-3 |
|:---:|:---:|:---:|:---:|:---:|
| 1K | 61.5% | **100.0%** | 0.0% | 0.0% |
| 3K | 93.5% | **100.0%** | 42.0% | 96.5% |
| 7K | **100.0%** | **100.0%** | 78.5% | 97.5% |
| Final | **100.0%** | **100.0%** | 82.5% | 99.0% |

#### 처리량 비교 (152M, H100)

| 모델 | tok/s | 상대적 속도 |
|:---|:---:|:---:|
| Transformer | 27,714 | 1.00× |
| **SISA** | 16,783 | 0.61× |
| Mamba-3 | 13,460 | 0.49× |
| Mamba-2 | 10,719 | 0.39× |

> SISA는 Mamba-3 대비 **+25% 처리량** 우위를 가지면서 FlashAttention 완전 호환

---

### 2.5 한계점

| 한계 | 설명 |
|:---|:---|
| **규모 제한** | 최대 369M 파라미터, 5B 토큰 학습에서만 검증 |
| **369M Chinchilla 미달** | 369M 모델은 토큰/파라미터 비율 13.5×로 Chinchilla 최적(~20×) 미달 |
| **Softmax 희석 문제** | SSM 중요도 편향이 softmax 정규화에 의해 희석됨 → sigmoid 어텐션으로 개선 가능(SISA-2 예정) |
| **처리량 오버헤드** | Transformer 대비 39% 처리량 감소 (augmented 내부 차원 $d_h + d_s$에 의한 SDPA 비용 증가) |
| **RoPE 외삽 한계** | 학습 길이 2,048 초과 시 모든 어텐션 기반 모델과 동일하게 성능 저하 |
| **$d_s$ 비단조적 최적화** | 스케일별 최적 $d_s$가 비단조적(50M: 64, 152M: 16, 369M: 128)으로 자동 선택 불가 |
| **단일 학습 시드** | 369M에서 bootstrap CI 분석 시 SISA와 Transformer가 통계적으로 구별 불가 |

---

## 3. 일반화 성능 향상 가능성

### 3.1 일반화에 기여하는 메커니즘

SISA의 일반화 성능 향상은 **"데이터 의존적 순차 사전(data-dependent sequential prior)"의 주입**에서 비롯된다.

#### (1) 빠른 표현 수렴 → 학습 효율성 향상

$$\text{Step 1K: SISA LAMBADA} = 8.4\% \quad \text{vs.} \quad \text{Transformer} = 3.5\%$$

SISA는 학습 초기(Step 2K, 전체의 21%)에 Transformer의 최종 정확도(13.9%)에 도달한다. 이는 SSM 편향이 **의미론적 표현이 성숙하기 전에 위치적 구조를 제공**하여 수렴을 가속화함을 의미한다. 더 적은 토큰으로 동등한 성능을 달성할 수 있다는 것은 다양한 도메인과 태스크로의 **전이 학습(transfer learning) 비용 감소**로 이어질 수 있다.

#### (2) 내용 + 중요도 이중 편향 → 더 강건한 어텐션

기존 어텐션:
$$s_{ij}^{\text{standard}} = \frac{\mathbf{q}_i^\top \mathbf{k}_j}{\sqrt{d_h}}$$

SISA 어텐션:
$$s_{ij}^{\text{SISA}} = \frac{\mathbf{q}_i^\top \mathbf{k}_j}{\sqrt{d_h}} + \lambda \cdot \bar{\mathbf{C}}_i^\top \bar{\mathbf{B}}_j$$

두 번째 항은 **내용 유사도와 독립적인 정규화 역할**을 수행한다. 내용 기반 어텐션만 사용할 때 발생하는 과적합 패턴(특정 표면형에만 주목)을 SSM 중요도 편향이 완화할 수 있다.

#### (3) 감쇠 함수의 입력 의존성 → OOD(Out-of-Distribution) 강건성

ALiBi의 고정 감쇠 $\alpha_t = \text{const}$와 달리 SISA의 감쇠는:

$$\alpha_t = \exp(-\text{softplus}(\mathbf{w}_\alpha^\top \mathbf{x}_t + b_\alpha))$$

입력 내용에 따라 동적으로 변한다. 이는 훈련 분포와 다른 입력에서도 시퀀스 구조를 적응적으로 반영할 수 있어 **분포 외 일반화(OOD generalization)**에 유리하다.

#### (4) NIAH 초기 수렴 → 구조적 검색 편향

$$\text{Step 1K NIAH: SISA} = 100\% \quad \text{vs.} \quad \text{Transformer} = 61.5\%$$

이는 SISA가 **의미론적 표현 학습과 무관하게 구조적 검색 능력을 사전 보유**함을 의미한다. 이 특성은 새로운 도메인에서도 정보 검색 능력이 즉각 발현될 수 있음을 시사한다.

#### (5) 스케일 향상에 따른 편익 증가

| 길이 L=2048에서의 NIAH | 50M | 152M |
|:---|:---:|:---:|
| Transformer | 88% | 82% |
| **SISA** | 83% | **91%** |

모델 크기가 커질수록 SSM 편향의 이점이 증가하는 패턴은 대규모 모델에서의 일반화 가능성을 지지한다.

### 3.2 일반화 한계와 주의사항

그러나 다음 요소들이 일반화를 제한한다:

- **태스크 의존적 최적 $d_s$**: LAMBADA(장거리 컨텍스트)와 ARC-Easy(사실 지식)는 서로 다른 최적 $d_s$를 선호 → 단일 하이퍼파라미터로 모든 태스크 일반화 어려움
- **softmax 희석**: 많은 토큰이 동등하게 관련된 태스크(HellaSwag 류)에서 SSM 편향이 정규화로 인해 희석됨
- **검증 규모 제한**: 369M/5B 토큰까지만 검증되어 1B+ 규모에서의 일반화는 미확인

---

## 4. 2020년 이후 관련 최신 연구 비교 분석

### 4.1 SSM-Transformer 하이브리드 아키텍처 비교

| 논문 | 연도 | 융합 수준 | 핵심 기제 | SISA와의 차이 |
|:---|:---:|:---:|:---|:---|
| **Jamba** (Lieber et al.) | 2024 | Block | 1:7 Mamba:Attn 레이어 교대 | SSM이 어텐션 스코어에 개입하지 않음 |
| **Samba** (Ren et al.) | 2024 | Block | Mamba + SWA + MLP | 동일한 독립 계산 문제 |
| **Griffin** (De et al.) | 2024 | Block | RG-LRU + 로컬 어텐션 | 로컬 어텐션으로 전역 검색 불가 |
| **Hymba** (Dong et al.) | 2024 | Head | Attn ‖ SSM 헤드 병렬 | 출력 수준 결합, 스코어 수준 아님 |
| **Falcon-H1** (Zuo et al.) | 2025 | Head | Attn + Mamba-2 헤드 | 동일한 헤드 수준 결합 |
| **Nemotron-H** (NVIDIA) | 2025 | Block | Mamba-2 + Full Attn | 대규모 검증, 스코어 융합 없음 |
| **Zamba/Zamba-2** (Glorioso et al.) | 2024 | Block | Mamba + 공유 전역 어텐션 | 공유 어텐션으로 효율화, 스코어 융합 없음 |

### 4.2 어텐션 스코어 편향 연구 비교

| 논문 | 연도 | 편향 형태 | 인코딩 내용 | FlashAttn 통합 |
|:---|:---:|:---:|:---|:---:|
| **ALiBi** (Press et al., ICLR 2022) | 2022 | 고정 스칼라 | 거리 | 네이티브 (FA≥2.4) |
| **T5 relative** (Raffel et al., JMLR 2020) | 2020 | 학습 스칼라 | 거리 | 커널 수정 필요 |
| **DAPE** (Zheng et al., NeurIPS 2024) | 2024 | 데이터 의존 스칼라 | 거리 + 내용 | 커널 수정 필요 |
| **FoX** (Lin et al., ICLR 2025) | 2025 | 누적 스칼라 | 감쇠만 | 수정된 FA 구현 필요 |
| **SISA (ours)** | 2026 | $d_s$차원 내적 | **감쇠 + 회전** | **Stock SDPA 호환** |

**FoX와 SISA의 핵심 차이:**

FoX (Forgetting Transformer):
$$\mathbf{A} = \text{softmax}(\mathbf{Q}\mathbf{K}^\top + \mathbf{D})$$
여기서 $D_{ij}$는 위치 쌍별 스칼라 (감쇠만 인코딩)

SISA:
$$s_{ij}^{\text{SISA}} = \frac{\mathbf{q}_i^\top \mathbf{k}_j}{\sqrt{d_h}} + \lambda \cdot \bar{\mathbf{C}}_i^\top \bar{\mathbf{B}}_j$$

$\bar{\mathbf{C}}_i^\top \bar{\mathbf{B}}_j$는 $d_s$차원 내적으로 **감쇠( $e^{g_i-g_j}$ )와 데이터 의존 회전( $R(\Phi_i - \Phi_j)$ ) 동시 인코딩**. 또한 FoX는 전체 $L \times L$ 편향 행렬을 구체화해야 하지만 SISA는 augmented Q/K로 단일 SDPA 호출로 처리.

### 4.3 위치 인코딩 연구 맥락에서의 위치

```
위치 인코딩 발전 계보:
  절대 위치 → RoPE (Su et al., 2021) → ALiBi (2022, 길이 외삽)
                                      → T5 relative (2020)
                                      → DAPE (2024, 내용 의존)
                                      → FoX (2025, 감쇠 편향)
                                      → SISA (2026, 벡터 감쇠+회전)
                                         ↑ 최초의 벡터값 데이터 의존 스코어 편향
```

---

## 5. 앞으로의 연구에 미치는 영향 및 고려사항

### 5.1 연구에 미치는 영향

#### (1) 새로운 설계 공리(Design Axis) 확립
SISA는 하이브리드 시퀀스 모델의 **세 번째 융합 수준**을 정의함으로써, 이후 하이브리드 모델 연구의 설계 공간을 다음과 같이 확장한다:

```
기존: {Block-level, Head-level}
이후: {Block-level, Head-level, Score-level, 그리고 이들의 조합?}
```

#### (2) SSM을 "메모리"가 아닌 "중요도 렌즈"로 재해석
기존에 SSM은 시퀀스 기억 기제로 이해되었으나, SISA는 이를 **어텐션에 순차적 사전(prior)을 공급하는 도구**로 재정의한다. 이 패러다임 전환은 SSM을 완전한 시퀀스 처리 기제로 사용하는 것이 아니라 어텐션의 보조 신호 생성기로 활용하는 새로운 연구 방향을 열어준다.

#### (3) Score-level fusion의 일반화 가능성
SISA의 원리는 다른 순차 모델(예: xLSTM, 선형 어텐션 변형)의 신호를 어텐션 스코어에 주입하는 데도 적용 가능하다. 이는 더 광범위한 "정보 원천 + Softmax Attention" 형태의 모델 패밀리 연구를 촉진할 것이다.

#### (4) 학습 효율성 연구에 대한 기여
"5배 적은 토큰으로 동등한 성능"은 데이터 효율적 사전학습(data-efficient pretraining) 연구에 직접적 영향을 미친다. 특히 저자원 언어나 특수 도메인에서의 모델 학습 비용 절감 가능성을 제시한다.

### 5.2 앞으로 연구 시 고려할 점

#### (1) 대규모 검증 필수성
현재 369M/5B 토큰까지만 검증. **1B, 7B, 70B 규모에서의 검증**이 필요하며 특히:
- $d_s$ 스케일링 법칙의 대규모 일반화 여부
- Chinchilla-optimal 학습에서의 SISA 성능

#### (2) Softmax → Sigmoid 전환 (SISA-2)
논문이 제안하는 가장 중요한 후속 작업:

$$\text{기존: } \mathbf{A} = \text{softmax}\left(\frac{\hat{\mathbf{Q}}\hat{\mathbf{K}}^\top}{\sqrt{d_h}}\right)$$

$$\text{SISA-2 (제안): } \mathbf{A} = \text{sigmoid}\left(\frac{\hat{\mathbf{Q}}\hat{\mathbf{K}}^\top}{\sqrt{d_h}}\right)$$

Sigmoid self-attention (Ramapuram et al., ICLR 2025)은 합계가 1로 강제되지 않아 SSM 중요도 편향이 희석되지 않는다. 특히 "많은 토큰이 동등하게 관련된" 태스크(HellaSwag, WinoGrande)에서 SISA의 약점을 보완할 가능성이 높다.

#### (3) $d_s$ 자동화 선택 메커니즘
현재 $d_s$는 수동으로 선택되며 비단조적 최적 패턴을 보인다. 연구 방향:
- **Per-layer $d_s$**: 각 레이어마다 다른 $d_s$ 적용
- **Sparse SSM channels**: 필요한 채널만 선택적 활성화
- **$d_s$ scaling law**: "데이터/파라미터 비율 × 헤드별 표현력 수요" 함수로서의 $d_s$ 예측 공식 도출

#### (4) 장문 컨텍스트 대응
현재 RoPE 외삽 한계(2,048)로 인해 장문에서 실패한다. **RoPE 스케일링** 또는 **SISA + RoPE-free 위치 인코딩** 조합이 필요하다. 특히 SSM의 데이터 의존적 회전이 이미 상대적 위치 정보를 인코딩하므로, 별도의 위치 인코딩 없이 장문 처리가 가능한지 탐구할 가치가 있다.

#### (5) $\lambda$의 레이어·헤드별 분포 분석
현재 $\lambda$는 헤드별 스칼라로만 분석됨. **레이어 깊이와 헤드 인덱스에 따른 $\lambda$ 분포**를 해석 가능성(interpretability) 관점에서 분석하면 모델이 각 레이어에서 SSM 편향을 어떻게 활용하는지 이해할 수 있다.

#### (6) 다운스트림 태스크별 $d_s$ 최적화
논문은 LAMBADA(장거리 컨텍스트)와 ARC-Easy(사실 지식)가 다른 $d_s$를 선호함을 보였다. **태스크 특화 SISA**를 위한 $d_s$ 선택 가이드라인 연구가 필요하다.

#### (7) Block/Head-level과 Score-level의 결합
SISA는 Score-level fusion을 블록·헤드 수준과 독립적인 제3의 축으로 정의했지만, 실제로 이 세 가지를 결합한 **4세대 하이브리드** 연구도 자연스러운 후속 방향이다. 예를 들어 SISA 레이어를 Jamba 스타일로 교대 배치하거나 Hymba 스타일로 SISA 헤드와 순수 SSM 헤드를 병렬 배치하는 실험이 가능하다.

---

**참고 자료 목록**

1. Shin, S. & Yang, Y. (2026). *Forget Attention: Importance-Aware Attention Is All You Need*. arXiv:2606.02332v2
2. Vaswani, A., et al. (2017). *Attention is all you need*. NeurIPS.
3. Gu, A. & Dao, T. (2023). *Mamba: Linear-time sequence modeling with selective state spaces*. arXiv:2312.00752
4. Dao, T. & Gu, A. (2024). *Transformers are SSMs*. ICML. arXiv:2405.21060
5. Lahoti, A., et al. (2026). *Mamba-3: Improved Sequence Modeling using State Space Principles*. ICLR. arXiv:2603.15569
6. Lieber, O., et al. (2024). *Jamba*. arXiv:2403.19887
7. Ren, L., et al. (2024). *Samba*. ICLR 2025. arXiv:2406.07522
8. Dong, X., et al. (2024). *Hymba: A Hybrid-head Architecture for Small Language Models*. ICLR 2025. arXiv:2411.13676
9. Zuo, J., et al. (2025). *Falcon-H1*. arXiv:2507.22448
10. De, S., et al. (2024). *Griffin: Mixing Gated Linear Recurrences with Local Attention for Efficient Language Models*. arXiv:2402.19427
11. Press, O., Smith, N. A., & Lewis, M. (2022). *Train Short, Test Long: ALiBi*. ICLR. arXiv:2108.12409
12. Raffel, C., et al. (2020). *Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer*. JMLR 21(140)
13. Zheng, C., et al. (2024). *DAPE: Data-Adaptive Positional Encoding*. NeurIPS. arXiv:2405.14722
14. Lin, Z., et al. (2025). *Forgetting Transformer: Softmax Attention with a Forget Gate*. ICLR. arXiv:2503.02130
15. Ramapuram, J., et al. (2024). *Theory, Analysis, and Best Practices for Sigmoid Self-Attention*. ICLR 2025. arXiv:2409.04431
16. Glorioso, P., et al. (2024a). *Zamba: A Compact 7B SSM Hybrid Model*. arXiv:2405.16712
17. Glorioso, P., et al. (2024b). *The Zamba2 Suite: Technical Report*. arXiv:2411.15242
18. NVIDIA. (2025). *Nemotron-H*. arXiv:2504.03624
19. Hoffmann, J., et al. (2022). *Training compute-optimal large language models (Chinchilla)*. arXiv:2203.15556
