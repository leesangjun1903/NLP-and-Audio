
# Self-Supervised Flow Matching for Scalable Multi-Modal Synthesis (Self-Flow)

> **📌 논문 정보**
> - **제목:** Self-Supervised Flow Matching for Scalable Multi-Modal Synthesis
> - **저자:** Hila Chefer, Patrick Esser, Dominik Lorenz, Dustin Podell, Vikash Raja, Vinh Tong, Antonio Torralba, Robin Rombach
> - **arXiv ID:** [2603.06507](https://arxiv.org/abs/2603.06507) (March 2026)
> - **발표:** ICML 2026 (Black Forest Labs / MIT)
> - **코드:** [github.com/black-forest-labs/Self-Flow](https://github.com/black-forest-labs/Self-Flow)

---

## 1. 핵심 주장 및 주요 기여 (간결 요약)

### 핵심 주장

강력한 의미론적 표현(semantic representation)은 확산(diffusion) 및 플로우 모델의 수렴 속도와 생성 품질을 향상시킨다. 그러나 기존 접근법은 외부 모델에 크게 의존하며, 이러한 외부 모델은 별도의 학습이 필요하고 목적 함수가 정렬되지 않으며 예상치 못한 스케일링 거동을 보인다. 이러한 의존성은 모델의 학습 목적 함수 자체에서 비롯되는데, 기존의 디노이징(denoising) 태스크는 의미론적 표현을 학습할 유인이 거의 없다.

### 주요 기여 요약

| 기여 | 설명 |
|---|---|
| **Self-Flow 프레임워크** | 외부 모델 없이 표현 학습을 생성 프레임워크 내부에 통합 |
| **Dual-Timestep Scheduling** | 토큰 간 이질적인 노이즈 수준을 적용해 정보 비대칭 유발 |
| **Teacher-Student EMA 구조** | 자기 지도 방식으로 의미 표현 학습 |
| **멀티모달 일반화** | 이미지, 비디오, 오디오 전 모달리티에 단일 프레임워크 적용 |
| **스케일링 법칙 준수** | 외부 정렬 기법의 스케일링 실패 문제 해결 |

---

## 2. 상세 분석

### 2-1. 해결하고자 하는 문제

이 논문은 생성 모델링에서의 고질적인 한계, 특히 이미지·오디오·비디오 합성을 위한 플로우/확산 모델에서 외부 사전 학습된 의미 인코더에 대한 의존이 만들어내는 병목 현상을 해결하고자 한다. 저자들은 외부 정렬 방식이 일관된 스케일링에 실패하고, 모달리티 간 효과적인 일반화에 실패하며, 증가된 모델 용량을 충분히 활용하지 못한다는 점을 지적한다.

더 구체적으로, 기존 확산 모델은 랜덤 노이즈를 디노이징하는 방식으로 학습되는데, 이 목적 함수에는 의미론적 이해를 발전시킬 명시적 압력이 없다. 모델은 얕은 패턴 매칭(shallow pattern matching)으로도 디노이징 태스크를 해결할 수 있고, 이 때문에 실무자들은 외부 표현 모델을 가이드로 추가해왔다.

또한 외부 정렬은 기대되는 스케일링 법칙을 유지하지 못하며, 더 강한 인코더를 사용할수록 오히려 성능이 감소하거나 부정적 효과가 나타나는 경우도 있다.

---

### 2-2. 제안하는 방법 (수식 포함)

#### (A) 표준 Flow Matching 목적 함수

Flow Matching의 기본 생성 목적 함수 $\mathcal{L}_{\text{gen}}$은 다음과 같이 정의된다:

$$\mathcal{L}_{\text{gen}} = \mathbb{E}_{\mathbf{x}_0, \mathbf{x}_1, t} \left\| v_\theta(\mathbf{x}_t, t) - (\mathbf{x}_1 - \mathbf{x}_0) \right\|^2$$

여기서 $\mathbf{x}\_t = (1 - t)\mathbf{x}\_0 + t\mathbf{x}\_1$은 노이즈 $\mathbf{x}\_0$과 데이터 $\mathbf{x}\_1$ 사이의 선형 보간이며, $v_\theta$는 속도장(velocity field)을 예측하는 신경망이다.

#### (B) Dual-Timestep Scheduling

핵심 메커니즘인 Dual-Timestep Scheduling은 토큰 전반에 걸쳐 이질적인 노이즈 수준을 적용하여, 모델이 손상된 입력으로부터 누락된 정보를 추론하도록 강제하는 정보 비대칭을 만들어낸다.

토큰 집합을 두 그룹으로 나누어, 각각 $t$ (높은 노이즈 수준)와 $s$ (낮은 노이즈 수준)의 타임스텝을 적용한다:

$$\boldsymbol{\tau} = (t, s), \quad t > s, \quad \tau_{\min} = \min(t, s) = s$$

$$\mathbf{x}_{\boldsymbol{\tau}} = \{(\mathbf{x}_t^{(i)}, \mathbf{x}_s^{(j)}) \mid i \in \text{group}_{\text{high}},\ j \in \text{group}_{\text{low}}\}$$

#### (C) Teacher-Student EMA 구조 및 표현 정렬 손실

두 개의 모델을 유지한다: 이질적으로 노이즈가 가해진 입력 $\mathbf{x}\_{\boldsymbol{\tau}}$로부터 학습하는 학생 네트워크 $f_\theta$와, $\tau_{\min} = \min(\boldsymbol{\tau})$로 노이즈가 가해진 더 깨끗한 $\mathbf{x}\_{\tau_{\min}}$을 관측할 수 있는 EMA 교사 네트워크 $f_{\theta'}$이다. 이 설정을 기반으로, 학생이 입력의 부분적·손상된 뷰로부터 교사의 특징을 재구성하도록 학습하는 특징 정렬 손실을 고안한다.

표현 정렬 목적 함수는 교사 네트워크 $f_{\theta'}$를 표현 네트워크로 사용하고, 이중 타임스텝 $\boldsymbol{\tau}$를 코사인 유사도 기반의 정렬 지표로 통합하여 다음과 같이 정식화된다:

$$\mathcal{L}_{\text{rep}} = -\mathbb{E}_{\mathbf{x}_0, \mathbf{x}_1, \boldsymbol{\tau}} \cos\!\left( h_{\theta}^{(l)}(\mathbf{x}_{\boldsymbol{\tau}}, \boldsymbol{\tau}),\ f_{\theta'}^{(k)}(\mathbf{x}_{\tau_{\min}}, \tau_{\min}) \right) \tag{6}$$

여기서:
- $h_\theta^{(l)}$: 학생 네트워크의 $l$번째 레이어 출력 (프로젝션 헤드 포함)
- $f_{\theta'}^{(k)}$: EMA 교사 네트워크의 $k$번째 레이어 출력
- $l < k$ (얕은 레이어의 학생이 깊은 레이어의 교사를 모방)

#### (D) 최종 통합 목적 함수

학습 목적 함수는 생성 목표와 표현 학습을 결합하며, 스케일링 팩터 $\gamma$로 조절된다:

$$\mathcal{L} = \mathcal{L}_{\text{gen}} + \gamma \cdot \mathcal{L}_{\text{rep}} \tag{7}$$

---

### 2-3. 모델 구조

Self-Flow는 플로우 매칭 목적 함수와 자기 지도 특징 재구성 목적 함수를 결합한 학습 프레임워크이다.

이 방법은 Dual-Timestep Scheduling과 Teacher-Student EMA 설정을 활용하여 이미지, 비디오, 오디오 합성을 위한 의미론적 특징을 강건하게 학습한다.

구조적 특징을 정리하면 다음과 같다:

```
입력 데이터 x₀ (노이즈) → x₁ (데이터)
      │
      ├─ 고노이즈 토큰 (timestep t): Student f_θ에 입력
      └─ 저노이즈 토큰 (timestep s): EMA Teacher f_θ'에 입력

Student f_θ ──────────────────────────────→ L_gen (생성 손실)
      │
      └─ Layer l 출력 h_θ^(l)
              ↓
         코사인 유사도 비교
              ↑
      EMA Teacher f_θ' Layer k 출력 f_θ'^(k)
                                    └─ L_rep (표현 정렬 손실)

최종: L = L_gen + γ · L_rep
```

---

### 2-4. 성능 향상

텍스트-이미지 생성에서 제안 방법은 외부 모델이나 감독 없이 기존 외부 정렬 방법의 대표 격인 REPA(Yu et al., 2024)보다 약 2.8배 빠르게 수렴한다. 특히 REPA는 성능이 정체되는 반면 제안 방법은 계속해서 개선된다.

바닐라 플로우 매칭과 비교하여, 제안 방법은 구조적 일관성, 텍스트 렌더링, 시간적 일관성을 향상시킨다.

실험적 평가에서 다양한 모달리티에 걸쳐 전통적인 외부 정렬 방법 대비 우수한 FID 및 FVD 점수를 달성하며 향상된 확장성과 성능을 보여준다.

특히 ImageNet 학습에 크게 의존하는 DINOv2를 사용하는 REPA보다 높은 성능을 보임으로써, 외부 인코더 없이도 더 나은 결과를 달성할 수 있음을 증명한다.

---

### 2-5. 한계

논문의 한계로 확인된 사항은 다음과 같다:

1. **하이퍼파라미터 민감성**: $\gamma$ (표현 손실 가중치), $l$, $k$ (레이어 선택), 타임스텝 이중 스케줄 $\{t, s\}$ 등 조정이 필요한 파라미터가 다수 존재한다.
2. **EMA 교사-학생 구조의 메모리 비용**: 두 개의 네트워크를 동시에 유지해야 하므로 추가적인 메모리와 연산 비용이 발생한다.
3. **이론적 분석 부족**: Dual-Timestep Scheduling이 실제로 의미 있는 표현을 강제하는 메커니즘에 대한 엄밀한 이론적 분석이 제한적이다 (현재 arXiv preprint 기준).

---

## 3. 일반화 성능 향상 가능성

Self-Flow의 방법은 모달리티 전반에 걸쳐 일반화되며, 기대되는 스케일링 법칙을 따르면서 멀티모달 학습을 가능하게 하고, 우수한 이미지, 비디오, 오디오 생성을 달성한다.

일반화 성능 향상의 핵심 근거를 다음과 같이 분석할 수 있다:

### 3-1. 외부 인코더 의존 제거 → 도메인 편향 최소화

Self-Flow는 생성 목적 함수 내에 표현 학습을 명시적으로 통합하는 자기 지도 플로우 매칭 패러다임을 제안함으로써, 외부 모델에 대한 의존성을 제거한다.

REPA와 같은 기존 방법은 DINOv2 같은 외부 인코더에 의존하는데, 이는 특정 데이터셋(예: ImageNet)에 편향될 수 있다. Self-Flow는 이 의존성을 제거함으로써 새로운 모달리티나 도메인으로의 전이(transfer)가 훨씬 용이하다.

### 3-2. 스케일링 법칙 준수

의미론적 표현이 외부 감독 신호 없이 자동으로 발전하고, 모델 크기가 증가함에 따라 기대되는 스케일링 거동을 유지하며, 단일 통합 프레임워크 내에서 멀티모달 학습이 가능하고, 모든 테스트된 모달리티에서 품질 향상이 나타난다.

이는 모델이 커져도 일반화 성능이 저하되지 않음을 의미하며, 기존 외부 정렬 방법의 스케일링 실패 문제를 근본적으로 해결한다.

### 3-3. 정보 비대칭 기반 표현 학습의 범용성

학생은 더 깨끗한 토큰을 적극적으로 활용하여 더 노이즈가 많은 토큰의 표현을 추론하도록 권장되며, 이는 단순한 지역성을 초월하는 전역적 연결 관계를 형성하도록 한다.

이 메커니즘은 특정 도메인에 종속되지 않는 범용적 표현 학습 원리이므로, 다양한 모달리티와 데이터 분포에 걸쳐 일반화될 수 있는 이론적 근거를 제공한다.

### 3-4. Train-Inference Gap 제거

자기 지도 표현 학습을 플로우 매칭에 직접 통합하는 것은 모든 모달리티에서 외부 정렬을 능가하고, 건강한 스케일링 법칙을 따르며, 도메인별 인코더 선택 없이 원활한 멀티모달 학습을 가능하게 한다. Self-Flow는 마스킹 기반 접근법을 괴롭히는 "학습-추론 격차(train-inference gap)"를 제거한다.

---

## 4. 앞으로의 연구에 미치는 영향 및 고려할 점

### 4-1. 연구에 미치는 영향

#### 🔵 생성 모델 패러다임의 전환

Self-Flow는 외부 인코더 기반 정렬이라는 기존의 지배적 패러다임을 정면으로 도전한다. Self-Flow는 명시적 자기 지도 표현 학습을 플로우 매칭에 내재화함으로써 모달리티에 구애받지 않고 고도로 확장 가능한 생성 모델링을 향한 경험적으로 검증된 경로를 제시한다. 외부 정렬의 한계를 제거하고, 효율적인 스케일링을 지원하며, 이미지, 오디오, 비디오, 통합 예측 태스크 전반에 걸쳐 성능 향상을 실현한다.

#### 🔵 REPA 계열 연구의 재편

관련 후속 연구로는 OneFlow (동시 혼합 모달 및 인터리브드 생성), TwinFlow (자기 적대적 플로우를 통한 원스텝 생성), High-Performance Self-Supervised Learning by Joint Training of Flow Matching (2025) 등이 있다.

#### 🔵 멀티모달 AI 연구 통합

표현 안내와 학습 자유 접근법에 관심 있는 연구자들은 학습 목적 함수를 더 효과적으로 내재화하는 방법을 이해함으로써 잠재적으로 혜택을 받을 수 있다.

---

### 4-2. 앞으로 연구 시 고려할 점

| 고려 항목 | 상세 내용 |
|---|---|
| **타임스텝 분할 전략** | Dual-Timestep Scheduling에서 토큰을 어떻게 분할하고 $t$와 $s$를 어떻게 스케줄링할지에 대한 심화 연구 필요 |
| **$\gamma$ 최적화** | 생성 손실과 표현 손실의 균형 파라미터 $\gamma$의 적응형(adaptive) 조정 방법 탐색 |
| **레이어 선택 ($l < k$)** | 학생 레이어 $l$과 교사 레이어 $k$의 최적 조합에 대한 체계적 분석 |
| **더 복잡한 모달리티 확장** | 3D 포인트 클라우드, 분자 구조, 로보틱스 등 새로운 모달리티로의 적용 가능성 검증 |
| **이론적 보장** | 왜 정보 비대칭이 의미 있는 표현을 유도하는지에 대한 이론적 분석 강화 |
| **계산 효율성** | Teacher-Student 이중 네트워크 구조의 메모리/연산 비용 최적화 |
| **결합 학습 안정성** | $\mathcal{L} = \mathcal{L}\_{\text{gen}} + \gamma \cdot \mathcal{L}_{\text{rep}}$ 형태의 다목적 학습에서 발생할 수 있는 그래디언트 충돌(gradient conflict) 방지 전략 |

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

| 연구 | 연도 | 핵심 방법 | 외부 모델 | 스케일링 | 멀티모달 |
|---|---|---|---|---|---|
| **DDPM** (Ho et al.) | 2020 | U-Net 기반 확산 모델 | ❌ | 제한적 | ❌ |
| **Flow Matching** (Lipman et al.) | 2022 | 선형 플로우 기반 생성 | ❌ | 우수 | 제한적 |
| **DiT** (Peebles & Xie) | 2022 | 트랜스포머 기반 확산 | ❌ | 우수 | ❌ |
| **REPA** (Yu et al.) | 2024 | 외부 DINOv2 정렬 | ✅ (DINOv2) | **불안정** | 제한적 |
| **Self-Flow (본 논문)** | 2026 | 자기 지도 Dual-Timestep | ❌ | **안정적** | ✅ |

텍스트-이미지 생성에서 제안 방법은 어떠한 외부 모델이나 감독 없이 REPA보다 약 2.8배 빠르게 수렴하며, REPA가 정체되는 반면 본 방법은 계속해서 향상된다.

Self-Flow의 가장 큰 차별점은 다음 세 가지다:
1. **자기 완결성(Self-Containment)**: 외부 인코더 없이 자체적으로 의미 표현 학습
2. **스케일링 법칙 준수**: 모델이 커질수록 비례적으로 성능 향상
3. **범용성**: 단일 프레임워크로 이미지·비디오·오디오 동시 처리

---

## 📚 참고 자료 및 출처

| # | 출처 | URL |
|---|---|---|
| 1 | arXiv 논문 원문 (2603.06507) | https://arxiv.org/abs/2603.06507 |
| 2 | arXiv PDF 전문 | https://arxiv.org/pdf/2603.06507 |
| 3 | Hugging Face Papers 페이지 | https://huggingface.co/papers/2603.06507 |
| 4 | GitHub 공식 코드 (Black Forest Labs) | https://github.com/black-forest-labs/Self-Flow |
| 5 | Emergent Mind 분석 페이지 | https://www.emergentmind.com/papers/2603.06507 |
| 6 | AI Models FYI 논문 상세 | https://www.aimodels.fyi/papers/arxiv/self-supervised-flow-matching-scalable-multi-modal |
| 7 | ResearchGate 논문 페이지 | https://www.researchgate.net/publication/401692324 |

> ⚠️ **정확도 관련 고지**: 본 답변은 arXiv 공개 초록, PDF 전문, GitHub 공식 저장소, Hugging Face 논문 페이지 등 1차 출처에서 확인된 정보만을 기반으로 작성되었습니다. 논문의 세부 실험 수치(FID 테이블 전체, 모든 ablation 결과 등) 및 모델 세부 아키텍처의 일부 사항은 arXiv PDF 전문의 직접 열람을 권장합니다.
