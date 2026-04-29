
# Navigation World Models

> **논문 정보**
> - **제목**: Navigation World Models
> - **저자**: Amir Bar, Gaoyue Zhou, Danny Tran, Trevor Darrell, Yann LeCun
> - **소속**: FAIR at Meta, New York University, Berkeley AI Research
> - **발표**: CVPR 2025
> - **arXiv**: [2412.03572](https://arxiv.org/abs/2412.03572) (2024.12.04 제출, 2025.04.11 v2 업데이트)
> - **공식 코드**: [github.com/facebookresearch/nwm](https://github.com/facebookresearch/nwm)

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

NWM(Navigation World Model)은 과거 관측(observations)과 내비게이션 액션(navigation actions)을 기반으로 미래의 시각적 관측을 예측하는 제어 가능한 비디오 생성 모델(controllable video generation model)이다.

친숙한 환경에서 NWM은 탐색 궤적(navigation trajectories)을 시뮬레이션하고 목표를 달성하는지 평가함으로써 계획을 수립할 수 있다. 고정된 행동을 갖는 지도 학습 기반 내비게이션 정책과 달리, NWM은 계획 과정에서 제약 조건을 동적으로 반영할 수 있다.

### 주요 기여 (3가지)

1. **CDiT 제안**: NWM을 위한 새로운 Conditional Diffusion Transformer(CDiT)를 제안하며, 이는 표준 DiT에 비해 현저히 줄어든 연산량으로 최대 10억(1B) 파라미터까지 효율적으로 스케일링된다.

2. **다중 환경 학습 및 계획**: 다양한 로봇 에이전트의 비디오 영상과 내비게이션 액션으로 CDiT를 학습하여, 외부 내비게이션 정책과 함께 또는 독립적으로 내비게이션 계획 시뮬레이션을 가능하게 하며 SOTA 시각 내비게이션 성능을 달성한다.

3. **비레이블 데이터 활용**: 미지의 환경에서 NWM은 Ego4D의 레이블 없는(action-free, reward-free) 비디오 데이터로부터 학습함으로써 이점을 얻는다. 단일 이미지에서의 비디오 예측 및 생성 성능이 질적으로 향상되고, 추가 비레이블 데이터로 학습 시 held-out Stanford Go 데이터셋에서 더 정확한 예측을 수행한다.

---

## 2. 해결하고자 하는 문제, 제안 방법, 모델 구조, 성능 및 한계

### 2-1. 해결하고자 하는 문제

현재 SOTA 내비게이션 정책들은 전형적으로 "하드코딩(hard-coded)"되어 있어, 학습 완료 후 새로운 제약 조건 추가가 어렵다. 또한 기존의 지도 학습 기반 시각 내비게이션 모델들은 더 복잡한 내비게이션 태스크에 직면했을 때 추가적인 연산 자원을 효과적으로 활용하지 못한다.

구체적으로 해결하고자 하는 문제는 두 가지이다:
- **Known environment**: 알려진 환경에서 주어진 목표 이미지까지의 최적 궤적 계획
- **Unknown environment**: 단 한 장의 이미지만으로 미지의 환경에서 궤적 상상(imagination)

---

### 2-2. 제안 방법 (수식 포함)

#### (A) 상태 표현 (State Representation)

과거 시각 관측은 $\mathbf{s}\_\tau = (s_\tau, \ldots, s_{\tau-m})$으로 표현되며, 사전 학습된 VAE를 통해 인코딩된다. VAE를 사용하면 압축된 잠재 변수(latent)를 활용할 수 있어, 예측 결과를 픽셀 공간으로 디코딩하여 시각화가 가능하다.

$$
\mathbf{s}_\tau = (s_\tau, s_{\tau-1}, \ldots, s_{\tau-m}) \quad \leftarrow \text{VAE 인코딩된 잠재 벡터}
$$

모델의 목표는 이전 시각 관측과 액션을 입력받아 미래 상태를 예측하는 확률적 매핑 함수 $F_\theta(s_{t+1} \mid s_t, a_t)$를 학습하는 것이다.

#### (B) 액션 표현 (Action Representation)

이 포뮬레이션의 단순성 덕분에, 다양한 환경 사이에서 자연스럽게 공유되고, 로봇 팔 제어와 같은 더 복잡한 액션 공간으로 쉽게 확장 가능하다.

에이전트 간 스텝 크기를 표준화하기 위해, 에이전트가 프레임 간 이동한 거리를 평균 스텝 크기(미터)로 나누어, 서로 다른 에이전트 간 액션 공간을 유사하게 만든다.

$$
a_\tau = \left(\Delta x,\ \Delta y,\ \Delta\theta,\ k\right)
$$

여기서 $\Delta x, \Delta y$는 2D 이동, $\Delta\theta$는 회전, $k$는 시간 이동(time-shift) 파라미터이다.

#### (C) 학습 목표 (Training Objective)

모델은 깨끗한 타겟(clean target)과 예측 타겟 사이의 평균 제곱 오차(MSE)를 최소화하도록 학습되어, 디노이징 프로세스를 학습하는 것을 목표로 한다. 이 손실을 최소화함으로써, 모델은 컨텍스트와 액션 $a_\tau$에 조건화되어 $s_{\tau+1}^{(t)}$를 재구성하는 법을 배우고, 이를 통해 사실적인 미래 프레임을 생성한다.

$$
\mathcal{L}_{\text{simple}} = \mathbb{E}_{t, s_{\tau+1}, \epsilon}\left[\left\|\epsilon - \epsilon_\theta\!\left(s_{\tau+1}^{(t)},\ t,\ \mathbf{s}_\tau,\ a_\tau\right)\right\|^2\right]
$$

또한, 노이즈의 공분산 행렬을 예측하고 이를 변분 하한(variational lower bound) 손실로 지도한다.

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{simple}} + \lambda \cdot \mathcal{L}_{\text{VLB}}
$$

#### (D) 계획 목표 (Planning Objective)

목표 도달의 비정규화된 지각적 유사도 점수를 따라가며 상태와 액션에 대한 잠재적 제약들을 고려하는 에너지 함수는 다음과 같이 정의된다:

$$
E(s_0, a_0, \ldots, a_T, s_T) = -\mathcal{S}(s_T, s^*) + \sum_t C(s_t, a_t)
$$

여기서 $\mathcal{S}(s_T, s^*)$는 최종 프레임과 목표 프레임 사이의 지각적 유사도 점수(LPIPS 등), $C(s_t, a_t)$는 제약 조건 비용이다.

---

### 2-3. 모델 구조 (CDiT Architecture)

#### Conditional Diffusion Transformer (CDiT)

CDiT는 초기 처리 블록에서 어텐션을 디노이징 중인 타겟 프레임의 토큰으로 제한하여 시간의 자기회귀적 모델링을 향상시킨다. 과거 프레임에 대한 조건화를 위해 크로스 어텐션(cross-attention) 레이어를 사용하며, 현재 프레임 토큰이 과거 프레임 토큰에 어텐션을 기울일 수 있도록 한다.

모델의 복잡도는 컨텍스트 프레임 수에 대해 선형적이어서 효율적이고 확장 가능하며, 10억 파라미터 규모에서도 운용 가능하다. 이러한 CDiT의 구조적 효율성은 기존 DiT 모델 대비 **4배 적은 FLOPs**라는 결과를 낳는다.

```
[입력: 과거 m개 프레임 + 액션]
        ↓ (VAE 인코딩)
[잠재 컨텍스트 토큰]
        ↓ (Self-Attention: 타겟 프레임 토큰 내부)
[CDiT 블록 × N]
        ↓ (Cross-Attention: 타겟 ← 컨텍스트)
[미래 프레임 잠재 벡터 예측]
        ↓ (VAE 디코딩)
[미래 시각 관측 이미지]
```

Stable Diffusion VAE 토크나이저(Blattmann et al., 2023)를 사용하며, DiT(Peebles and Xie, 2023)와 유사한 방식을 따른다. AdamW 옵티마이저를 사용한다.

#### 학습 데이터셋

로봇공학 데이터셋으로는 SCAND, TartanDrive, RECON, HuRoN을 사용하며, 이 데이터셋들을 통해 로봇의 위치와 회전 정보에 접근하여 상대적 액션을 추론한다. 에이전트 간 스텝 크기를 표준화하고 후진 동작을 필터링한다. 추가로 레이블이 없는 Ego4D 비디오도 사용하며, 이 경우 time shift만을 액션으로 간주한다. SCAND는 다양한 환경에서의 사회적 내비게이션, TartanDrive는 오프로드 주행, RECON은 오픈 월드 내비게이션, HuRoN은 사회적 상호작용을 다룬다.

#### 평가 지표

모델 성능은 LPIPS, DreamSim, PSNR을 통해 ground truth 프레임과 비교하여 측정된다.

---

### 2-4. 성능 향상

친숙한 환경 내비게이션에서 모델 용량이 가장 중요하며, CDiT가 최대 2× FLOPs에 해당하는 모델 크기에서 표준 DiT보다 더 우수한 성능을 보인다.

Go Stanford 데이터셋에서 NWM의 Success Rate(SR)는 0.45인 반면, 이를 능가하는 UniWM의 SR은 0.71로, NWM이 강력한 비교 기준선으로 활용되고 있음을 보여준다.

실험들은 NWM이 처음부터 궤적을 계획하거나 외부 정책에서 샘플링된 궤적들을 랭킹하는 방식 모두에서 효과적임을 보여준다.

---

### 2-5. 한계 (Limitations)

미지의 환경에서 흔한 실패 사례는 모드 붕괴(mode collapse)로, 모델 출력이 학습 데이터와 점점 유사해지는 현상이 나타난다.

In-domain 환경에 비해 모델이 더 빨리 붕괴되고, 상상된 환경의 경로를 생성할 때 예상대로 환각(hallucination)이 발생한다.

성능 측면의 여러 한계로는: 고도로 동적인 환경에서의 성능 미검증, 계산 요구량이 실제 응용을 제한할 수 있음, 고품질 시각 입력에 대한 의존성이 신뢰성에 영향을 줄 수 있음 등이 있다.

기존 내비게이션 월드 모델들은 VLN 태스크에서: (1) 이산적 상태 다이나믹스에 의존하여 연속적 액션-상태 전이를 모델링하는 능력이 제한되고, (2) 정적인 사전 학습 모델을 사용하여 새롭고 동적인 환경에 대한 적응성이 제한되며, (3) 픽셀 수준의 미래 예측을 사용하여 높은 계산 비용이 발생하는 문제가 있다.

---

## 3. 모델의 일반화 성능 향상 가능성

### 3-1. 알려진 환경에서의 일반화

DIAMOND 및 GameNGen과 같은 모델들과 달리, NWM은 광범위한 환경 및 에이전트 구현에 걸쳐 학습된다. 이 다양한 데이터셋을 활용하여, 연구자들은 여러 환경에 걸쳐 일반화할 수 있는 대형 확산 트랜스포머 모델을 성공적으로 학습시켰다. 이 일반화 능력은 특정 환경이나 태스크에 제한되었던 이전 모델들과의 중요한 차별점이다.

### 3-2. 미지의 환경에서의 일반화 (핵심)

NWM은 학습된 시각적 사전 지식(visual priors)을 활용하여 단일 입력 이미지만으로 낯선 환경에서 궤적을 상상할 수 있어, 차세대 내비게이션 시스템을 위한 유연하고 강력한 도구가 된다.

Ego4D의 레이블 없는 비디오의 서브셋(time-shift 액션만 사용)과 함께 모든 in-domain 데이터셋으로 모델을 학습시키고 Go Stanford 데이터셋과 기타 랜덤 이미지들로 테스트한다. 레이블 없는 데이터로 학습 시 모든 메트릭에서 유의미하게 개선된 비디오 예측이 나타난다.

주목할 실험으로, 미지의 환경에서 NWM이 Ego4D의 레이블 없는(action-free, reward-free) 비디오 데이터로 학습함으로써 이점을 얻는다. 개별 이미지에서 향상된 비디오 예측 및 생성을 질적으로 보여준다. 추가적인 레이블 없는 비디오 데이터로 학습 시 Stanford Go 데이터셋에서 더 정확한 미래 예측을 달성한다. 이 결과들은 실세계 내비게이션 태스크에서 중요한 강점인 미관측 환경에 대한 NWM의 효과적인 일반화 능력을 강조한다.

### 3-3. 일반화 성능의 확장 방향

단순한 포뮬레이션 덕분에, 이는 자연스럽게 환경 간에 공유될 수 있고 로봇 팔 제어와 같은 더 복잡한 액션 공간으로 쉽게 확장 가능하다.

미지의 분포 외(out-of-distribution) 환경에서, 장기 계획(long-term planning)은 상상(imagination)에 의존할 수 있다.

---

## 4. 관련 최신 연구 비교 분석 (2020년 이후)

| 모델 | 연도 | 핵심 방법 | NWM과의 차이점 |
|------|------|-----------|---------------|
| **DreamWalker** | 2022 | World model for VLN | 이산 상태 다이나믹스, 환경 특화 |
| **PathDreamer** | 2022 | Visual future obs. generation | 이산 관점(viewpoint) 기반 |
| **ViNT/NoMaD** | 2023/2024 | Foundation model for navigation | 액션 예측 정책, 월드 모델 없음 |
| **DIAMOND** | 2024 | Diffusion-based WM (UNet) | 게임 환경에 특화, 내비게이션 일반화 부족 |
| **NWM** | 2024 | CDiT + 다중 에이전트 데이터 | **범용 내비게이션 월드 모델** |
| **UniWM** | 2025 | Memory-augmented WM | NWM보다 SR 성능 향상(+26%p in Go Stanford) |
| **NavMorph** | 2025 | Self-evolving WM for VLN-CE | 연속 액션-상태 전이 모델링 |

NWM, DreamWalker, PathDreamer와 같은 선구적인 접근들은 내비게이션에서 월드 모델의 잠재력을 보여주었지만, 기저의 액션-상태 전이를 학습하는 능력이 부족하거나 이산적 상태 다이나믹스에 의존하여 VLN 태스크에 내재된 공간-시간 다이나믹스의 연속적 특성을 포착하는 능력이 제한된다.

UniWM은 모든 SOTA 기준선과 비교하여 일관되게 우수한 결과를 제공한다. 메모리 증강 없이도 Go Stanford에서 NWM(0.45) 대비 SR 0.71 등 ATE/RPE에서 상당한 향상을 달성한다. 인트라-스텝 메모리와 크로스-스텝 메모리를 결합하면 장기 일관성이 더욱 향상된다.

NavMorph는 인간 인지에서 영감을 받아, VLN-CE 태스크에서 환경 이해와 의사결정을 향상시키는 자기-진화(self-evolving) 월드 모델 프레임워크이다. 컴팩트한 잠재 표현을 사용하여 환경 다이나믹스를 모델링하고, 적응적 계획과 정책 개선을 위한 예지력(foresight)을 갖춘다. 새로운 Contextual Evolution Memory를 통합하여 온라인 적응성을 유지한다.

---

## 5. 앞으로의 연구에 미치는 영향과 고려할 점

### 5-1. 연구에 미치는 영향

#### ① 비디오 생성 + 내비게이션의 통합 패러다임 확립
Navigation World Model(NWM)은 로봇 내비게이션에서 강력한 도약을 대표한다. 시뮬레이션하고, 계획하며, 새로운 제약 조건에 적응하는 능력은 더 자율적이고 유연한 로봇 시스템 구축을 위한 유망한 접근법이 된다.

#### ② 레이블 없는 대규모 데이터 활용 방향 제시
미지의 환경에서 NWM이 Ego4D의 레이블 없는, 액션 및 보상 없는(action- and reward-free) 비디오 데이터로부터의 학습을 통해 이점을 얻음을 보여준다.

이는 향후 대규모 비레이블 데이터를 활용한 사전 학습 → 소수의 로봇 데이터로 파인튜닝하는 연구 방향을 제시한다.

#### ③ MPC(Model Predictive Control)와의 결합
NWM은 CDiT를 기반으로 MPC 프레임워크 내에서 월드 모델링을 활용하는 방식으로 활용될 수 있음이 확인되었다.

#### ④ 관련 분야로의 파급 효과
체현 AI에서의 월드 모델은 조작(manipulation), 내비게이션, 자율 주행 등 다양한 태스크를 다루어야 하며, 이질적인 자원과 엄격한 평가가 요구된다.

---

### 5-2. 향후 연구 시 고려할 점

#### ① 계산 효율성 문제
체현 AI 태스크는 실시간 응용에서 계산 효율성 면의 중대한 도전에 직면한다. Transformer 및 Diffusion 네트워크와 같은 모델들이 강력한 성능을 보이지만, 높은 추론 비용이 로봇 시스템의 실시간 제어 요구와 충돌한다. 결과적으로 RNN과 같은 전통적 방법들이 여전히 널리 사용되는데, 장기 의존성 포착의 한계에도 불구하고 더 큰 계산 효율성을 제공하기 때문이다.

**→ 고려 사항**: 양자화(quantization), 프루닝(pruning), 희소 연산(sparse computation) 등의 기법으로 추론 지연을 줄이는 연구 필요

#### ② 데이터 이질성 및 부족 문제
데이터 관점에서 핵심 과제는 기존 말뭉치의 희소성과 이질성에 있다. 체현 AI가 내비게이션, 조작, 자율 주행 등 다양한 도메인에 걸쳐 있지만, 통합된 대규모 데이터셋이 아직 부재하다. 이러한 분절화는 월드 모델의 능력을 제한하고 일반화 능력을 실질적으로 저해한다.

#### ③ 평가 지표의 한계
FID 및 FVD와 같은 메트릭은 픽셀 충실도를 강조하지만 물리적 일관성, 다이나믹스, 인과관계를 무시한다. EWM-Bench와 같은 최근 벤치마크들이 새로운 측정 방법을 도입하지만 태스크 특화적이며 크로스-도메인 표준이 부재하다.

#### ④ 모드 붕괴 및 환각 문제 해결
특히 미지의 환경에서 흔한 실패 사례 중 하나는 모드 붕괴(mode collapse)로, 모델이 학습 데이터와 점점 더 유사한 프레임을 천천히 생성하는 현상이다.

**→ 고려 사항**: 분포 외(OOD) 환경 감지 메커니즘, 온라인 적응 전략(예: NavMorph의 Contextual Evolution Memory) 결합 탐색 필요

#### ⑤ 연속적 액션-상태 전이 모델링
기존 내비게이션 월드 모델들은 이산적 상태 다이나믹스 의존, 정적 사전 학습 모델, 픽셀 수준의 미래 예측으로 인한 높은 계산 비용 등의 한계가 있다. 이를 해결하기 위해 잠재 표현 학습을 통한 연속적 액션-상태 전이 모델링, 자기-진화 메커니즘을 통한 동적 적응, 특징 수준에서의 미래 예측이 필요하다.

#### ⑥ 장기 공간적 일관성 유지
이 문제를 해결하기 위해 양자화, 프루닝, 희소 연산 등의 기술을 사용하여 성능 저하 없이 추론 지연을 줄이는 모델 아키텍처 최적화에 초점을 맞춰야 한다. 또한 SSM(State Space Model)과 같은 새로운 시간적 방법을 탐색하면 실시간 효율성을 유지하면서 장거리 추론을 향상시킬 수 있다.

---

## 📚 참고 자료 (출처)

| # | 제목 및 출처 |
|---|-------------|
| 1 | **Navigation World Models** - arXiv:2412.03572, Amir Bar et al. (CVPR 2025) https://arxiv.org/abs/2412.03572 |
| 2 | **Navigation World Models (HTML 전문)** - https://arxiv.org/html/2412.03572v2 |
| 3 | **Navigation World Models (Project Page)** - https://www.amirbar.net/nwm/ |
| 4 | **Navigation World Models (Official Code, CVPR 2025)** - https://github.com/facebookresearch/nwm |
| 5 | **Navigation World Models (ResearchGate PDF)** - https://www.researchgate.net/publication/386454839 |
| 6 | **[Literature Review] Navigation World Models** - https://www.themoonlight.io/en/review/navigation-world-models |
| 7 | **Yann LeCun Team's New Research: Revolutionizing Visual Navigation** - Synced Review https://syncedreview.com/2024/12/09/ |
| 8 | **A Comprehensive Survey on World Models for Embodied AI** - arXiv:2510.16732 https://arxiv.org/html/2510.16732v1 |
| 9 | **Unified World Models: Memory-Augmented Planning and Foresight for Visual Navigation** - arXiv:2510.08713 https://arxiv.org/html/2510.08713v1 |
| 10 | **NavMorph: A Self-Evolving World Model for Vision-and-Language Navigation in Continuous Environments** - arXiv:2506.23468 (ICCV 2025) https://arxiv.org/html/2506.23468v1 |
| 11 | **Toward Memory-Aided World Models: Benchmarking via Spatial Consistency** - arXiv:2505.22976 https://arxiv.org/html/2505.22976v1 |
| 12 | **Navigation World Models - AI Models FYI** - https://www.aimodels.fyi/papers/arxiv/navigation-world-models |
| 13 | **Navigation World Models - alphaXiv** - https://www.alphaxiv.org/abs/2412.03572 |
