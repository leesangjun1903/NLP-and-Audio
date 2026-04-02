# LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels

---

## 1. 핵심 주장 및 주요 기여 요약

LeWorldModel(LeWM)은 **Joint Embedding Predictive Architecture(JEPA)** 프레임워크 기반의 잠재 세계 모델(latent world model)로, 기존 JEPA 방법들이 겪던 **표현 붕괴(representation collapse)** 문제를 복잡한 다중 손실 함수, EMA(Exponential Moving Average), stop-gradient, 사전 학습 인코더 등의 휴리스틱 없이 해결한 **최초의 end-to-end 학습 가능한 JEPA**이다.

### 주요 기여

| 기여 항목 | 내용 |
|---------|------|
| **안정적 end-to-end 학습** | 오직 2개의 손실 항(예측 손실 + SIGReg 정규화)만 사용, 튜닝 가능한 하이퍼파라미터를 6개(PLDM)에서 1개($\lambda$)로 축소 |
| **효율성** | 15M 파라미터, 단일 GPU에서 수 시간 내 학습 가능, 파운데이션 모델 기반 대비 최대 48× 빠른 계획(planning) 속도 |
| **경쟁적 성능** | 2D/3D 다양한 제어 태스크에서 PLDM을 능가하고 DINO-WM과 경쟁적 성능 달성 |
| **물리적 이해도** | 잠재 공간에서 물리량 프로빙 및 VoE(Violation-of-Expectation) 테스트를 통해 의미 있는 물리 구조 인코딩 확인 |
| **이론적 보장** | Cramér–Wold 정리에 기반한 **증명 가능한 anti-collapse 보장** 제공 |

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

JEPA 기반 세계 모델은 관측을 저차원 잠재 공간으로 인코딩하고 미래 잠재 상태를 예측하여 환경 역학을 모델링한다. 그러나 핵심적인 문제가 존재한다:

1. **표현 붕괴(Representation Collapse)**: 인코더가 모든 입력을 동일한 표현으로 매핑하여 예측 목표를 사소하게(trivially) 만족시키는 실패 모드
2. **기존 해결책의 한계**:
   - **PLDM**: end-to-end이지만 VICReg 기반 7개 손실 항 사용 → 6개 하이퍼파라미터, 불안정한 학습
   - **DINO-WM**: 사전 학습된 DINOv2 인코더 동결 사용 → end-to-end 학습 불가, 사전 학습 지식에 제한
   - **Dreamer/TD-MPC**: 태스크 특화(보상 신호, 상태 정보 필요)

### 2.2 제안하는 방법

#### 모델 구조

LeWM은 두 가지 핵심 컴포넌트로 구성된다:

$$
\text{Encoder:} \quad \mathbf{z}_t = \text{enc}_\theta(\mathbf{o}_t)
$$

$$
\text{Predictor:} \quad \hat{\mathbf{z}}_{t+1} = \text{pred}_\phi(\mathbf{z}_t, \mathbf{a}_t)
$$

- **인코더**: ViT-Tiny (~5M 파라미터), 패치 크기 14, 12 레이어, 3 어텐션 헤드, 히든 차원 192. [CLS] 토큰 임베딩을 Batch Normalization이 포함된 1-layer MLP 프로젝터로 투영
- **예측기(Predictor)**: 트랜스포머 (6 레이어, 16 어텐션 헤드, ~10M 파라미터). 액션은 **Adaptive Layer Normalization (AdaLN)**을 통해 주입. 시간적 인과 마스킹(causal masking) 적용

#### 학습 목적 함수

전체 학습 목적 함수는 다음과 같이 정의된다:

$$
\mathcal{L}_{\text{LeWM}} \triangleq \mathcal{L}_{\text{pred}} + \lambda \, \text{SIGReg}(\mathbf{Z})
$$

**① 예측 손실 (Prediction Loss)**:

$$
\mathcal{L}_{\text{pred}} \triangleq \|\hat{\mathbf{z}}_{t+1} - \mathbf{z}_{t+1}\|_2^2, \quad \hat{\mathbf{z}}_{t+1} = \text{pred}_\phi(\mathbf{z}_t, \mathbf{a}_t)
$$

**② SIGReg 정규화 (Sketched-Isotropic-Gaussian Regularizer)**:

잠재 임베딩 $\mathbf{Z} \in \mathbb{R}^{N \times B \times d}$를 $M$개의 랜덤 단위 벡터 $\mathbf{u}^{(m)} \in \mathbb{S}^{d-1}$에 투영:

$$
\mathbf{h}^{(m)} = \mathbf{Z}\mathbf{u}^{(m)}
$$

각 1차원 투영에 대해 Epps–Pulley 정규성 검정 통계량 $T(\cdot)$을 적용:

$$
\text{SIGReg}(\mathbf{Z}) \triangleq \frac{1}{M} \sum_{m=1}^{M} T(\mathbf{h}^{(m)})
$$

Epps–Pulley 검정 통계량은 다음과 같다:

$$
T^{(m)} = \int_{-\infty}^{\infty} w(t) \left| \phi_N(t; \mathbf{h}^{(m)}) - \phi_0(t) \right|^2 dt
$$

여기서 $\phi_N(t; \mathbf{h}) = \frac{1}{N}\sum_{n=1}^{N} e^{ith_n}$은 경험적 특성 함수(ECF)이고, $\phi_0$는 표준 가우시안 $\mathcal{N}(0,1)$의 특성 함수이다.

**Cramér–Wold 정리**에 의해, 모든 1차원 주변 분포가 일치하면 전체 결합 분포도 일치하므로:

$$
\text{SIGReg}(\mathbf{Z}) \to 0 \iff \mathbb{P}_{\mathbf{Z}} \to \mathcal{N}(\mathbf{0}, \mathbf{I})
$$

이를 통해 **증명 가능한 anti-collapse 보장**을 제공한다.

#### 잠재 공간 계획(Latent Planning)

추론 시 Model Predictive Control (MPC)을 사용:

$$
\hat{\mathbf{z}}_{t+1} = \text{pred}_\phi(\hat{\mathbf{z}}_t, \mathbf{a}_t), \quad \hat{\mathbf{z}}_1 = \text{enc}_\theta(\mathbf{o}_1)
$$

목표 매칭 비용 함수:

$$
\mathcal{C}(\hat{\mathbf{z}}_H) = \|\hat{\mathbf{z}}_H - \mathbf{z}_g\|_2^2, \quad \mathbf{z}_g = \text{enc}_\theta(\mathbf{o}_g)
$$

최적 행동 시퀀스:

$$
\mathbf{a}_{1:H}^* = \arg\min_{\mathbf{a}_{1:H}} \mathcal{C}(\hat{\mathbf{z}}_H)
$$

이 최적화는 **Cross-Entropy Method (CEM)**으로 풀어진다 (300 샘플, 30 반복, top-30 엘리트 선택).

### 2.3 성능 향상

| 환경 | LeWM | PLDM | DINO-WM | 특이사항 |
|------|------|------|---------|---------|
| **Push-T** | **96%** | 78% | 74% (pixels) / 92% (+prop) | LeWM이 PLDM 대비 +18%, DINO-WM+prop도 능가 |
| **Reacher** | **86%** | 78% | 79% | 일관적 우위 |
| **OGBench-Cube** | 74% | 65% | **86%** | 3D 시각 복잡성으로 인한 한계 |
| **Two-Room** | 87% | 97% | 100% | 낮은 내재 차원에서 SIGReg 한계 |

- **계획 속도**: DINO-WM 대비 **48× 빠름** (0.98s vs 47s)
- **학습 안정성**: 2-항 손실의 단조(monotonic) 수렴 vs PLDM의 7-항 noisy/non-monotonic 수렴
- **물리량 프로빙**: Agent Location, Block Location, Block Angle 등에서 PLDM을 일관적으로 능가하고 DINO-WM과 경쟁적 (Table 1)
- **Temporal Latent Path Straightening**: 명시적 정규화 없이도 PLDM보다 높은 시간적 직선성 달성 (emergent phenomenon)

### 2.4 한계

1. **단기 계획 제한**: 현재 잠재 세계 모델은 짧은 계획 수평에 제한됨. 자기회귀 롤아웃에서 예측 오류가 누적
2. **저복잡도 환경에서의 SIGReg 한계**: Two-Room과 같이 내재 차원이 낮은 환경에서 고차원 잠재 공간에 등방 가우시안을 강제하면 덜 구조화된 표현을 초래
3. **3D 시각 복잡성**: OGBench-Cube에서 DINO-WM 대비 열위 — 사전 학습된 대규모 비전 인코더의 풍부한 사전 지식이 유리할 수 있음
4. **오프라인 데이터 의존성**: 충분한 상호작용 커버리지를 가진 오프라인 데이터셋에 의존
5. **액션 레이블 필요**: 미래 상태 예측을 위해 명시적 액션 라벨이 필요

---

## 3. 모델의 일반화 성능 향상 가능성

LeWM의 일반화 성능과 관련하여 논문에서 드러나는 핵심 포인트들을 정리하면 다음과 같다:

### 3.1 아키텍처 불가지론(Architecture-Agnostic) 특성

LeWM은 ViT-Tiny와 ResNet-18 모두에서 경쟁적 성능을 달성한다 (Push-T: 96.0 vs 94.0). 이는 **인코더 아키텍처에 대한 일반화 가능성**을 시사한다.

### 3.2 하이퍼파라미터 강건성

- $\lambda \in [0.01, 0.2]$ 범위에서 성공률 80% 이상 유지
- SIGReg의 내부 파라미터(투영 수 $M$, 적분 노트 수)에 대해 성능이 거의 불변
- **단 하나의 효과적 하이퍼파라미터** $\lambda$만 존재 → $\mathcal{O}(\log n)$ 이분 탐색으로 튜닝 가능 (PLDM: $\mathcal{O}(n^6)$ )

### 3.3 태스크 간 일반화

동일한 하이퍼파라미터 설정으로 2D 내비게이션(Two-Room), 2D 조작(Push-T), 3D 조작(OGBench-Cube), 로코모션(Reacher) 등 다양한 태스크에 걸쳐 평가되었다. **태스크별 튜닝 없이** 경쟁적 성능을 보인다.

### 3.4 물리적 구조 인코딩을 통한 일반화

- **선형/비선형 프로브** 결과, 잠재 공간이 에이전트 위치, 블록 위치, 블록 각도 등 물리량을 높은 상관계수($r > 0.97$)로 인코딩
- **디코더 시각화**: 학습 중 재구성 손실을 사용하지 않았음에도 192차원 잠재 임베딩으로부터 시각적 장면 복원 가능
- **t-SNE 시각화**: 잠재 공간이 환경의 공간적 구조를 보존
- **VoE 테스트**: 물리적으로 불가능한 이벤트(텔레포트) 감지 능력 — 학습된 역학 모델이 물리적 규칙성을 내재화

### 3.5 일반화 향상을 위한 미래 방향

1. **대규모 자연 비디오 사전 학습**: 다양한 데이터셋에서의 사전 학습이 표현 선행 지식을 강화하고 도메인 특화 데이터 의존성 감소 가능
2. **계층적 세계 모델링**: 장기 계획 수평에서의 예측 오류 누적 문제 해결
3. **역동역학 모델링을 통한 액션 표현 학습**: 명시적 액션 주석 필요성 감소
4. **적응적 잠재 차원 선택**: 환경의 내재 차원에 맞는 잠재 공간 차원 적응으로 저복잡도 환경에서의 SIGReg 한계 극복

---

## 4. 연구 영향 및 향후 연구 시 고려할 점

### 4.1 앞으로의 연구에 미치는 영향

**① JEPA 학습 안정성의 새로운 패러다임 제시**
- EMA, stop-gradient 없이 end-to-end 학습이 가능함을 실증적으로 보여줌
- SIGReg의 이론적 보장(Cramér–Wold 기반)이 JEPA 커뮤니티에서 휴리스틱 의존을 줄이는 방향성 제시

**② 접근성 민주화**
- 단일 GPU에서 수 시간 내 학습 → 대규모 컴퓨트 자원 없는 연구 그룹도 참여 가능
- 15M 파라미터의 경량 모델로 실시간 제어에 근접한 계획 속도 달성

**③ 잠재 세계 모델의 물리적 이해 평가 프레임워크**
- 프로빙 + VoE 평가를 결합한 체계적 평가 방법론 제시
- Temporal Path Straightening의 자연 발현(emergent) 현상 발견 — 잠재 표현의 기하학적 특성 연구에 새로운 관점 제공

**④ 실용적 로보틱스 및 제어 시스템에 대한 시사점**
- 보상 신호 없는(reward-free), 태스크 비의존적(task-agnostic) 학습 → 범용 세계 모델로의 확장 가능성

### 4.2 향후 연구 시 고려할 점

| 고려 사항 | 구체적 내용 |
|----------|----------|
| **장기 계획(Long-Horizon Planning)** | 현재 계획 수평 $H=5$ (프레임 스킵 5 적용 시 25 환경 스텝). 계층적 잠재 공간 또는 다중 스케일 예측 필요 |
| **데이터 다양성** | SIGReg의 효과는 데이터 다양성에 의존. 낮은 내재 차원의 환경에서는 대안적 정규화 전략 탐색 필요 |
| **스케일링** | 현재 ViT-Tiny 수준. 더 큰 모델과 더 복잡한 환경(Minecraft, 실세계 로보틱스)에서의 확장성 검증 필요 |
| **멀티모달 입력** | 현재 pixels-only. 고유수용감각(proprioception), 촉각 등 멀티모달 통합 시 성능 향상 가능성 |
| **온라인 학습** | 현재 완전 오프라인 설정. 온라인 미세 조정이나 지속 학습(continual learning)과의 통합 탐색 |
| **계획 알고리즘** | CEM은 고차원 액션 공간에서 차원의 저주에 취약. 미분 가능한 계획 방법(differentiable planning) 탐색 |

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

| 연구 | 연도 | 접근 방식 | 학습 방식 | Anti-Collapse 전략 | 하이퍼파라미터 수 | 태스크 비의존성 | LeWM 대비 차이점 |
|------|------|---------|---------|------------------|---------------|-------------|---------------|
| **Ha & Schmidhuber (World Models)** [2] | 2018 | VAE + RNN | End-to-end | VAE 재구성 | 다수 | ✗ (RL 보상 필요) | 픽셀 재구성 의존, 보상 필요 |
| **DreamerV1/V2/V3** [27-29] | 2020-2023 | RSSM (Generative) | End-to-end | 이미지 재구성 + 보상 예측 | 다수 | ✗ | 태스크 특화, 보상 신호 필요 |
| **VICReg** [23] | 2022 (ICLR) | 자기지도 학습 | End-to-end | Variance-Invariance-Covariance | 3 | - | SIGReg의 이론적 선행 연구. LeWM은 이를 대체하는 더 원칙적인 정규화 사용 |
| **I-JEPA** [12] | 2023 (CVPR) | 마스크 예측 JEPA | EMA + SG | EMA/Stop-gradient | - | - (이미지 SSL) | EMA/SG 필요, 이론적 보장 부재 |
| **V-JEPA** [13] | 2023 | 비디오 JEPA | EMA + SG | EMA/Stop-gradient | - | - (비디오 SSL) | 세계 모델이 아닌 표현 학습 목적 |
| **IRIS** [3] | 2023 (ICLR) | VQ-VAE + Transformer | End-to-end | 이산 토큰화 | 다수 | ✗ | 생성적 세계 모델, 보상 필요 |
| **DIAMOND** [6] | 2024 (NeurIPS) | 확산 기반 | End-to-end | 확산 모델 | 다수 | ✗ | 픽셀 공간에서 예측, 계산 비용 높음 |
| **TD-MPC2** [32] | 2024 (ICLR) | 잠재 모델 + TD 학습 | End-to-end | 보상/상태 예측 | 다수 | ✗ | 상태 기반, 보상 필요 |
| **PLDM** [21, 22] | 2022-2025 | JEPA (VICReg 기반) | End-to-end (SG 없음) | VICReg + 시간 정규화 | **6** | ✓ | 7-항 손실, 불안정 학습, LeWM이 직접 개선 |
| **DINO-WM** [18] | 2025 (ICML) | JEPA (동결 인코더) | 예측기만 학습 | 사전학습 인코더 동결 | 0 (인코더) | ✓ | End-to-end 아님, 사전학습 지식에 제한 |
| **V-JEPA 2** [14] | 2025 | 비디오 JEPA | EMA + SG | EMA/Stop-gradient | - | 부분적 | 대규모 비디오 사전학습, 계획 능력 일부 |
| **DreamerV4** [4] | 2025 | 확장 가능한 생성적 WM | End-to-end | 재구성 + RL | 다수 | ✗ | 대규모 생성 모델, 보상 의존 |
| **LeJEPA** [25] | 2025 | 자기지도 학습 | End-to-end | **SIGReg** | 1 | - (이미지 SSL) | LeWM의 SIGReg 원천 연구. 세계 모델 아닌 이미지 SSL |
| **LeWM (본 논문)** | 2026 | JEPA | **End-to-end** | **SIGReg** (이론적 보장) | **1** | **✓** | — |

### 핵심 비교 분석

**PLDM vs LeWM**: 가장 직접적인 비교 대상. PLDM은 VICReg에서 파생된 7개 손실 항( $\mathcal{L}\_{\text{pred}} + \alpha\mathcal{L}\_{\text{var}} + \beta\mathcal{L}\_{\text{cov}} + \gamma\mathcal{L}\_{\text{time-sim}} + \zeta\mathcal{L}\_{\text{time-var}} + \nu\mathcal{L}\_{\text{time-cov}} + \mu\mathcal{L}_{\text{IDM}}$ )을 사용하며, 6개 하이퍼파라미터의 탐색이 $\mathcal{O}$ $(n^6)$ 복잡도를 요구한다. LeWM은 이를 2-항 손실과 1개 하이퍼파라미터( $\mathcal{O}(\log n)$ )로 단순화하면서 Push-T에서 +18% 성능 향상을 달성하고, 학습 곡선도 안정적(단조 수렴)이다.

**DINO-WM vs LeWM**: DINO-WM은 DINOv2(~124M 이미지로 사전학습)를 동결 인코더로 사용하여 collapse를 회피하지만, (1) end-to-end 학습 불가, (2) 사전학습 분포에 없는 도메인에서 한계, (3) ~200× 더 많은 토큰 → 48× 느린 계획. LeWM은 pixels-only로도 Push-T에서 DINO-WM+proprioception을 능가(96% vs 92%)하지만, 3D 복잡 환경(OGBench-Cube)에서는 대규모 사전학습의 시각적 사전 지식이 유리(86% vs 74%).

**Generative WMs (Dreamer, IRIS, DIAMOND) vs LeWM**: 생성적 접근은 픽셀 공간에서 예측하여 모든 시각적 세부사항을 모델링하지만, 보상 신호 의존, 높은 계산 비용, 태스크 특화 학습이라는 한계가 있다. LeWM은 보상 없이 태스크 비의존적으로 학습하며, 잠재 공간에서의 예측으로 계산 효율성을 확보한다.

---

## 참고자료

1. Maes, L., Le Lidec, Q., Scieur, D., LeCun, Y., & Balestriero, R. (2026). "LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels." *arXiv:2603.19312v1* [cs.LG].
2. Sobal, V. et al. (2022). "Joint Embedding Predictive Architectures Focus on Slow Features." *arXiv:2211.10831*.
3. Bardes, A., Ponce, J., & LeCun, Y. (2022). "VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning." *ICLR 2022*.
4. Balestriero, R. & LeCun, Y. (2025). "LeJEPA: Provable and Scalable Self-Supervised Learning without the Heuristics." *arXiv:2511.08544*.
5. Zhou, G. et al. (2025). "DINO-WM: World Models on Pre-trained Visual Features Enable Zero-shot Planning." *ICML 2025*.
6. LeCun, Y. (2022). "A Path Towards Autonomous Machine Intelligence Version 0.9.2." *Open Review*, 62(1):1–62.
7. Assran, M. et al. (2023). "Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture." *CVPR 2023*.
8. Hafner, D. et al. (2025). "Training Agents Inside of Scalable World Models." *arXiv:2509.24527*.
9. Epps, T.W. & Pulley, L.B. (1983). "A Test for Normality Based on the Empirical Characteristic Function." *Biometrika*, 70(3):723–726.
10. Cramér, H. & Wold, H. (1936). "Some Theorems on Distribution Functions." *Journal of the London Mathematical Society*, 1(4):290–294.
