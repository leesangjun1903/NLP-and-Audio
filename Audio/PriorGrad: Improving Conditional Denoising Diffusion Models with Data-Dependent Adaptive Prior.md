# PriorGrad: Improving Conditional Denoising Diffusion Models with Data-Dependent Adaptive Prior
### 1. 핵심 주장 및 기여도

**PriorGrad**는 조건부 확산 확률 모델의 효율성을 개선하기 위해 표준 가우시안 선행분포 대신 조건 정보로부터 도출된 데이터 의존적 적응형 가우시안 선행분포를 사용하는 방법을 제안한다. 본 논문의 주요 기여는 다음과 같다:[1]

- **첫 번째**: 조건부 생성 모델에서 비표준 가우시안 분포를 선행분포로 사용하는 효과를 체계적으로 조사한 최초 연구
- **두 번째**: 조건 정보를 활용한 적응형 선행분포로 표준 가우시안 대비 성능 대폭 향상 실증
- **세 번째**: 음성 합성 영역에서 스펙트럼 및 시간 도메인 모두에서 포괄적 분석 및 빠른 수렴, 개선된 품질, 향상된 매개변수 효율성 입증

### 2. 해결하고자 하는 문제

기존 DDPM 기반의 조건부 음성 합성 모델들은 몇 가지 근본적인 비효율성을 보인다:[1]

**문제의 본질**: 표준 가우시안 $$N(0, I) $$를 선행분포로 사용하면서 실제 데이터 분포와의 괴리가 발생한다. 예를 들어 시간 도메인 음성에서 유성(voiced) 및 무성(unvoiced) 음절은 극도로 다른 가변성을 가지는데, 같은 표준 가우시안으로 이를 모델링하려 하므로 다음과 같은 문제 발생:

- 훈련 중 느린 수렴 (약 7일/단일 A40 GPU로 1M 반복 필요)
- 모든 데이터 모드를 포괄하기 어려움으로 인한 학습 비효율
- 추론 시 높은 계산 비용

### 3. 제안하는 방법 (수식 포함)

#### 3.1 일반적 공식화

**Proposition 1** (수정된 ELBO): $$\mathcal{N}_0 $$를 선행분포로 하는 경우, 수정된 ELBO 손실함수는 다음과 같다:[1]

$$
\text{ELBO} = C + \sum_{t=1}^{T} \lambda_t \mathbb{E}_{x_0, \epsilon} \left\| \tilde{\epsilon} - \epsilon_{t}(x_t, c, t) \right\|^2 - \epsilon_1
$$

여기서:
- $$x_t = \alpha_t x_0 + \sqrt{1 - \alpha_t} \epsilon $$ (forward process)
- $$\mu(\mu, \Sigma) $$는 데이터 통계로부터 계산된 적응형 평균/분산
- $$\lambda_t = \frac{\sigma_t^2}{2\bar{\sigma}_t^2} $$, $$\bar{\sigma}_t^2 = \frac{(1-\bar{\alpha}_t)\sigma_t^2}{\bar{\alpha}_t} $$

#### 3.2 적응형 선행분포 계산

기존의 고정 선행분포 대신:

$$
p(x_T) = \mathcal{N}(x_T; 0, I) \rightarrow p(x_T | c) = \mathcal{N}(x_T; \mu(c), \Sigma(c))
$$

#### 3.3 훈련 및 샘플링 알고리즘

**알고리즘 1 (PriorGrad 훈련)**:[1]
```
repeat
  μ, Σ ← 데이터 의존 선행분포
  x₀ ~ q_data
  t ~ U{1, ..., T}
  xₜ ← √ᾱₜ x₀ + √(1-ᾱₜ) ε
  ℒ ← ||x̃ₜ - εθ(xₜ, c, t)||²₁
  모델 매개변수 θ로 ℒ 역전파
until 수렴
```

**알고리즘 2 (PriorGrad 샘플링)**:[1]
```
μ, Σ ← 데이터 의존 선행분포
xₜ ~ N(0, Σ)
for t = T, T-1, ..., 1 do
  z ~ N(0, I) if t > 1 else 0
  xₜ₋₁ ← (1/√ᾱₜ)(xₜ - (1-ᾱₜ)/√(1-ᾱₜ) ε̂ₜ) + √(1-σₜ²) z
end for
return x₀
```

#### 3.4 음성 합성 응용

**Vocoder 응용** (mel-spectrogram 조건):[1]
- 프레임 레벨 에너지: $$E_i = \sqrt{\sum_{f} c_{i,f}^2} $$
- 정규화된 에너지: $$\sigma_i^c = (E_i - E_{\min})/(E_{\max} - E_{\min}) $$
- 클리핑: $$\sigma_i^c = \max(0.1, \min(1.0, \sigma_i^c)) $$

**음향 모델 응용** (음소 조건):[1]
- 음소별 통계 수집: 훈련 데이터에서 각 음소 발생별 80차원 평균/분산 계산
- 지속 시간 기반 업샘플링으로 프레임 레벨 선행분포 구성

### 4. 이론적 분석

**Proposition 2 (모델 단순화)**: $$\epsilon $$가 선형 함수이고 $$\det(\Sigma) = \det(I) $$ 제약 하에서:[1]

$$
\min_{\theta} L(\mu, \Sigma, x_0) \leq \min_{\theta} L(0, I, x_0)
$$

이는 **공분산 정렬**이 더 단순한 모델로 같은 정확도를 달성할 수 있음을 의미한다.

**수렴 속도 분석**: 헤시안 조건수 $$\kappa(H) = \frac{\lambda_{\max}(H)}{\lambda_{\min}(H)} $$에 대해:[1]
- $$L(\mu, \Sigma, x_0) $$: $$\kappa = 1 $$ (최적)
- $$L(0, I, x_0) $$: $$\kappa \geq 1 $$ (비최적)

따라서 적응형 선행분포 사용으로 더 빠른 수렴 보장.

### 5. 모델 구조 및 성능 향상

#### 5.1 모델 구조

**음성신경망 (DiffWave 기반)**:[1]
- 2.62M 매개변수, 12개 dilated 컨볼루션 층
- mel-spectrogram 조건 입력
- 확산 단계 임베딩 + 조건부 프로젝션 층

**음향 모델 (FastSpeech2 + DiffWave)**:[1]
- 인코더: 11.5M (Transformer 기반 음소 인코더)
- 디코더: 10M/3.5M (dilated 컨볼루션 기반 확산 디코더)
- 교차 주의로 음소 조건 주입

#### 5.2 성능 향상 결과

**수렴 속도**:[1]
- LS-MAE 메트릭 기준, 200K 반복에서 기저선(DiffWave)의 1M 반복 성능 달성
- **약 5배 가속화**: 7일 → 1.4일 훈련 시간

**객관적 지표 (1M 반복)**:[1]

| 지표 | DiffWave | PriorGrad | 개선율 |
|------|----------|-----------|--------|
| LS-MAE | 0.5264 | 0.5048 | 4.1% |
| MR-STFT | 1.0920 | 0.9976 | 8.6% |
| MCD | 9.7822 | 9.2820 | 5.1% |
| F0 RMSE | 16.4035 | 15.5542 | 5.2% |
| Sinkhorn S(xT,x0) | 72698.62 | 42236.93 | 41.9% |

**주관적 품질 (MOS 5단계)**:[1]

| 모델 | T_infer=6 | T_infer=50 |
|------|-----------|------------|
| GT | - | 4.42 ± 0.07 |
| DiffWave 500K | - | 4.01 ± 0.08 |
| PriorGrad 500K | 4.02 ± 0.08 | **4.14 ± 0.08** |
| DiffWave 1M | - | 4.12 ± 0.08 |
| PriorGrad 1M | 4.21 ± 0.08 | **4.25 ± 0.08** |

**음향 모델 성능 (60K 반복)**:[1]

| 모델 | 매개변수 | MOS |
|------|----------|-----|
| 기저선 10M | 10M | 3.84 ± 0.10 |
| **PriorGrad 10M** | 10M | **4.04 ± 0.07** |
| 기저선 3.5M | 3.5M | 3.87 ± 0.09 |
| **PriorGrad 3.5M** | 3.5M | **3.96 ± 0.07** |

#### 5.3 매개변수 효율성[1]

**축소 모델 성능** (1.23M 음성신경망 vs 2.62M 기저선):

| 모델 | MOS |
|------|-----|
| DiffWave (2.62M) | 4.06 ± 0.08 |
| DiffWave Small (1.23M) | 3.90 ± 0.09 |
| **PriorGrad Small (1.23M)** | **4.02 ± 0.08** |

PriorGrad 소규모 모델이 기저선 대규모 모델과 거의 동등한 성능 달성.

### 6. 모델 일반화 성능 향상 가능성

#### 6.1 PriorGrad의 일반화 특성

**강점**:[1]
1. **빠른 수렴으로 인한 과적합 감소**: 더 적은 반복으로 수렴하므로 규제 효과
2. **매개변수 효율성**: 같은 품질을 더 작은 모델로 달성 가능
3. **추론 효율**: T_infer=6으로도 T_infer=50 수준의 기저선과 비슷한 성능

**한계**:[1]
1. **Task-specific 설계 필요**: 각 도메인(음성신경망 vs 음향 모델)마다 다른 선행분포 설계 필요
2. **조건 정보 소스 선택의 경험성**: mel-spectrogram 에너지, 음소 통계 등 수동 선택
3. **도메인 간 전이성 부재**: 이미지 등 다른 도메인 확장 검증 없음

#### 6.2 시뮬레이션 분석을 통한 일반화

**멘트로피 제약에서의 비교** (부록 A.2):[1]

선형 근사 하에서 헤시안:

$$
H = \nabla^2_{\theta} L(\mu, \Sigma, x_0) = \sum_{t=1}^{T} \sigma_t^2 I
$$

vs 표준 선행분포:

$$
H = \sum_{t=1}^{T} \sigma_t^2(1 - t) I
$$

$$1 - t < 1 $$이므로 PriorGrad의 헤시안이 더 **well-conditioned**, 더 일반성 있는 최적화 경로 제공.

#### 6.3 실증적 일반화 지표

**객관적 메트릭 개선** (Sinkhorn 발산):[1]
- 선행분포 ↔ 실제 데이터: 72698.62 → 42236.93 (41.9% 개선)
- 생성 샘플 ↔ 실제 데이터: 1650.22 → 1608.89 (2.5% 개선)

더 나은 선행분포 정렬로 학습된 역확산 프로세스의 **표현 능력 향상**.

### 7. 한계

논문이 명시하는 한계:[1]

1. **Task-specific 설계**: "task-specific design to compute the data-dependent statistics or its proxy, which may be unsuitable depending on the granularity of the conditional information"

2. **조건 정보 선택의 경험성**: 부록 A.3에서 voiced/unvoiced 라벨, 음소 레벨 통계 등 여러 선택 시도 후 프레임 레벨 에너지 선택

3. **도메인 한정성**: 음성 합성에 중점, 이미지 초해상도 등 다른 도메인 확장은 향후 과제로 제시

4. **학습 가능한 선행분포 실패**: 부록 A.6에서 관절 학습 시도는 선행분포 추정 노이즈, 분산 붕괴로 실패

### 8. 2020년 이후 관련 최신 연구 비교 분석

#### 8.1 Improved DDPM (Nichol & Dhariwal, 2021)[2][3]

**개요**: 역확산 프로세스의 분산을 학습하여 log-likelihood 개선

**vs PriorGrad**:
- **공통점**: 모두 표준 DDPM 개선을 목표
- **차이점**:
  - Improved DDPM: 분산 학습 (forward process는 고정)
  - PriorGrad: forward process 선행분포 적응화
  - PriorGrad가 더 **근본적 개선** (선행분포 자체 개선)

#### 8.2 Grad-TTS (Popov et al., 2021)[4][5][6]

**개요**: 텍스트 조건 음성 생성을 위한 스코어 기반 디코더로 mel-spectrogram 생성

**선행분포 관점**:
- 인코더로부터 평균 이동 가우시안 $$N(\mu(c), I) $$ (분산은 항등행렬)
- 인코더 손실로 특정 제약 강제

**vs PriorGrad**:[4][1]
| 측면 | Grad-TTS | PriorGrad |
|------|----------|-----------|
| 선행분포 | 학습 가능 (인코더) | 데이터 통계 (고정) |
| 분산 모델링 | 항등행렬로 고정 | 데이터로부터 추정 |
| 인코더 제약 | 있음 (additional loss) | 없음 |
| 수렴 보장 | 선행분포 변화로 인한 불안정 | 고정 선행분포로 보장[1] |
| 추가 오버헤드 | 인코더 손실 | 사전 계산만 |

**실무적 차이**: 
- Grad-TTS (NeurIPS 2021, T_infer=2 MOS=3.43): 빠른 추론 가능
- PriorGrad (ICLR 2022): T_infer=6 MOS=4.21, 수렴 안정성 우수

#### 8.3 DOSE: Diffusion Dropout with Adaptive Prior (2023, NeurIPS)[7][8][9]

**개요**: 훈련 중 dropout + 조건으로부터 적응형 선행분포 활용

**선행분포 전략**:

$$
p_\psi(x_\tau | y) = \mathcal{N}(x_\tau; g(\mu(y), \tau), \Sigma(y))
$$

여기서 $$\tau < T $$ 선택으로 에러 누적 회피

**vs PriorGrad**:[7]
| 측면 | DOSE | PriorGrad |
|------|------|-----------|
| 도메인 | 음성 향상 | 음성 합성 |
| 선행분포 원천 | 조건 정보 | 조건 정보 |
| Dropout 메커니즘 | 있음 | 없음 |
| τ 선택 | 동적 ( $$\tau < T $$ ) | 고정 ( $$\tau = T $$ ) |
| 응용 특화 | 향상 작업 중심 | 합성 작업 중심 |

**상보적 관계**: 두 방법 모두 적응형 선행분포 개념 공유하나 응용 영역 다름

#### 8.4 RestoreGrad (2025, ICML)[10][11][12]

**개요**: VAE와 DDPM 통합으로 **관절 학습된 선행분포**

**핵심 혁신**:

$$
\text{Prior} = \text{Encoder}(y) \rightarrow \mathcal{N}(\mu_{\text{enc}}, \Sigma_{\text{enc}})
$$

여기서 $y$ = 열화된 신호 (음성/이미지)

**vs PriorGrad**:[10]

| 측면 | PriorGrad | RestoreGrad |
|------|-----------|-------------|
| 선행분포 계산 | 데이터 통계 (사전 계산) | 인코더 (관절 학습) |
| 학습 복잡도 | 낮음 | 높음 |
| 수렴 속도 | 1M 반복 | 5-10배 빠름 (관절 학습이므로) |
| 일반화성 | Task-specific | 더 적응형 (신호 특정) |
| 추론 효율 | T_infer=6 동등 | 2-2.5배 단계 감소 |
| 응용 | 음성 합성 | 신호 복원 (음성/이미지) |

**학습 가능성**: PriorGrad의 부록 A.6에서 관절 학습 실패 시연, RestoreGrad가 개선된 접근

#### 8.5 최신 경향 (2024-2025) 종합 분석

**1. 선행분포 설계의 진화**:

```
표준 DDPM (2020)
    ↓
N(0,I) → Learned Variance (Improved DDPM, 2021)
    ↓
조건부 선행분포 시작 (Grad-TTS, 2021)
    ↓
데이터 의존 고정 선행분포 (PriorGrad, 2022)
    ↓
적응형 고급 선행분포 + dropout (DOSE, 2023)
    ↓
관절 학습된 선행분포 (RestoreGrad, 2025)
```

**2. 도메인별 적용 현황**:[13][14][15][16][17][10]
- **음성**: PriorGrad, DOSE, GLA-Grad++ (2025)
- **이미지**: FACDIM (2025), CN-Diff (조건부 비선형 변환)
- **신호 복원**: RestoreGrad (2025)
- **합성**: 조건부 확산 모델의 모든 영역 확장

**3. 이론적 진전**:[18][19][20]
- Classifier-free guidance 정밀 분석
- Bayesian conditioning (BCDM, 2024)
- 경사 기반 안내의 최적성 증명 (REG, 2025)

**4. 실무적 혁신**:[21][22][23]
- 미세 조정 중 무조건 선행분포 보존의 중요성 강조
- 비선형 데이터 변환과 조건부 가우시안 결합 (CN-Diff)
- 사전 학습 모델의 선행분포 저하 현상 분석

### 9. 향후 연구에 미치는 영향과 고려 사항

#### 9.1 학문적 영향

**기여 영역**:

1. **선행분포 설계의 중요성 인식**:
   - DDPM의 "black-box" 선행분포 선택에 대한 비판적 관점 제시
   - 조건부 모델에서 선행분포 최적화의 1차적 중요성 입증

2. **이론-실무 연결**:
   - Proposition 1-2로 선행분포 개선의 이론적 정당성 제공
   - 헤시안 조건수 분석으로 수렴 메커니즘 설명

3. **도메인별 응용 가능성 확대**:
   - 음성 합성 이외 음성 향상(DOSE), 신호 복원(RestoreGrad) 파생 연구 촉발
   - 이미지 도메인 적용 가능성 제시 (초해상도 등)

#### 9.2 실무적 영향

**산업 적용**:

1. **효율성 향상**:
   - 훈련 시간 5배 단축으로 GPU 자원 절감
   - 매개변수 효율성으로 모바일/엣지 배포 가능성 증가

2. **품질-속도 트레이드오프 개선**:
   - T_infer=6으로 고품질 생성 (기존 T_infer=50 필요)
   - 실시간 음성 합성 시스템에 실용성 증대

3. **비용 절감**:
   - 더 작은 모델 (1.23M vs 2.62M)로 성능 유지
   - 배포 시 메모리/연산 오버헤드 감소

#### 9.3 향후 연구 시 고려할 점

**1. Task-Specific 설계 자동화**:[1]

현재 한계: 각 응용마다 선행분포 수동 설계 필요

향후 과제:
```python
# 자동 선행분포 발견
선행분포 = 조건 정보로부터 자동 추출(조건)
# 메타 학습 기반 접근
학습 가능한 통계 추출 함수 = MetaLearner(조건)
```

**2. 비선형 조건 함수 이론 확장**:[1]

Proposition 2는 선형 $\epsilon$ 가정. 실무에서는 비선형 신경망:

$$
\text{향후}: \text{Non-linear } \epsilon_\theta \text{에 대한 최적성 조건 유도}
$$

**3. 도메인 일반화**:[1]

- **이미지**: 패치 레벨 에너지 vs 색상 히스토그램 vs 중력
- **기하학**: 거리 지도 조건화
- **3D**: 점구름 밀도 기반 선행분포

**4. 학습 가능한 선행분포 재검토**:[1]

부록 A.6의 실패 원인 분석:
- 선행분포 추정기 불안정성 해결
- VAE 틀(RestoreGrad) vs 경량 추정기 비교
- 규제 기법 도입으로 붕괴 방지

**5. 조건부 분포 이론 심화**:

최신 연구(2024-2025) 방향:
- **Bayesian 관점**: 사전 분포의 베이지안 최적성 분석
- **정보 이론**: 상호 정보로 조건과 선행분포의 정렬도 측정
- **최적 수송**: Wasserstein 거리 최소화로 선행분포 학습

**6. 멀티모달 조건화**:

최신 연구 트렌드:
- 텍스트 + 음향 조건 결합
- 사전 학습 대형 언어 모델(LLM) 활용
- 크로스 모달 선행분포 학습

**7. 안정성 및 강건성**:[1]

실무 배포 시 고려사항:
- 분포 외(OOD) 조건에 대한 선행분포 행동 분석
- 노이즈 로버스트 선행분포 추정
- 다양한 데이터 통계에 대한 민감도 분석

### 10. 결론

**PriorGrad**는 조건부 확산 모델의 **근본적 개선**을 제시한 중요한 연구이다. 표준 가우시안 선행분포 대신 데이터 의존적 적응형 선행분포를 사용함으로써:

- **이론적 보증**: 수렴 속도 및 모델 단순화의 수학적 증명
- **실증적 성과**: 음성 합성에서 5배 수렴 가속화, MOS 개선
- **효율성 향상**: 매개변수 50% 감소로도 성능 유지

이러한 성과는 후속 연구(DOSE, RestoreGrad 등)로 이어졌으며, 현재까지 조건부 확산 모델의 **선행분포 최적화**가 핵심 연구 주제로 자리잡게 하였다.

**핵심 의의**: 확산 모델 설계에서 "선행분포는 고정된 하이퍼파라미터가 아닌 **최적화 대상**"임을 입증한 선구적 연구로, 향후 생성 모델 개발에서 선행분포 설계의 중요성을 강조하는 기초를 마련했다.

***

### 참고문헌 인덱싱

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c6e570af-9bfe-4fa2-954f-27a1e310afaa/2106.06406v2.pdf)
[2](https://arxiv.org/abs/2102.09672)
[3](https://proceedings.mlr.press/v139/nichol21a/nichol21a.pdf)
[4](https://proceedings.mlr.press/v139/popov21a/popov21a.pdf)
[5](https://arxiv.org/abs/2105.06337)
[6](https://www.semanticscholar.org/paper/Grad-TTS:-A-Diffusion-Probabilistic-Model-for-Popov-Vovk/2e32cde6e080f990873638f2e113767a6a19c824)
[7](https://papers.nips.cc/paper_files/paper/2023/file/7e966a12c2d6307adb8809aaa9acf057-Paper-Conference.pdf)
[8](https://openreview.net/forum?id=2C2WZfCfo9)
[9](https://arxiv.org/pdf/2312.04370.pdf)
[10](https://arxiv.org/abs/2502.13574)
[11](https://icml.cc/virtual/2025/poster/45491)
[12](http://www.arxiv.org/abs/2502.13574)
[13](https://arxiv.org/abs/2509.15182)
[14](http://cjlcd.lightpublishing.cn/thesisDetails#10.37188/CJLCD.2025-0045)
[15](https://www.mdpi.com/2079-9292/14/10/2070)
[16](https://arxiv.org/abs/2507.20478)
[17](https://ieeexplore.ieee.org/document/11104385/)
[18](https://arxiv.org/abs/2501.18865)
[19](https://arxiv.org/abs/2403.11968)
[20](https://arxiv.org/html/2410.16415v1)
[21](https://arxiv.org/html/2503.20240v1)
[22](https://openreview.net/forum?id=kcUNMKqrCg)
[23](https://randomsampling.tistory.com/424)
[24](https://arxiv.org/abs/2508.19581)
[25](https://www.semanticscholar.org/paper/b9a5d69c4663ce5b80e9cda35c195a83683ed753)
[26](https://arxiv.org/abs/2504.17253)
[27](https://arxiv.org/pdf/2106.06406.pdf)
[28](http://arxiv.org/pdf/2406.09768.pdf)
[29](https://arxiv.org/pdf/2402.03570.pdf)
[30](https://arxiv.org/html/2406.12120)
[31](https://arxiv.org/abs/2302.11710)
[32](https://arxiv.org/html/2405.18782v1)
[33](https://icml.cc/virtual/2025/poster/44243)
[34](https://openreview.net/pdf?id=_BNiN4IjC5)
[35](https://www.ijcai.org/proceedings/2025/0728.pdf)
[36](https://huggingface.co/papers/2106.06406)
[37](https://dlaiml.tistory.com/entry/Improved-DDPM-Improved-Denoising-Diffusion-Probabilistic-Models)
[38](https://arxiv.org/html/2502.13574v3)
[39](https://www.ijcai.org/proceedings/2022/0577.pdf)
[40](https://learnopencv.com/denoising-diffusion-probabilistic-models/)
[41](https://www.sciencedirect.com/science/article/abs/pii/S0893608025000838)
[42](https://arxiv.org/pdf/2405.04235.pdf)
[43](https://arxiv.org/html/2511.22293v1)
[44](https://arxiv.org/abs/2006.11239)
[45](https://arxiv.org/html/2510.10807v3)
[46](https://arxiv.org/abs/2106.06406)
[47](https://ar5iv.labs.arxiv.org/html/2102.09672)
[48](https://openaccess.thecvf.com/content/CVPR2025/papers/Tang_Exploring_the_Deep_Fusion_of_Large_Language_Models_and_Diffusion_CVPR_2025_paper.pdf)
[49](https://arxiv.org/html/2409.05730v1)
[50](https://arxiv.org/html/2410.23318v1)
[51](https://arxiv.org/html/2507.16406v1)
[52](https://proceedings.neurips.cc/paper/2020/file/4c5bcfec8584af0d967f1ab10179ca4b-Paper.pdf)
[53](https://dmqa.korea.ac.kr/activity/seminar/411)
[54](https://arxiv.org/abs/2301.12686)
[55](https://ieeexplore.ieee.org/document/10657237/)
[56](https://dl.acm.org/doi/10.1145/3664647.3681556)
[57](https://arxiv.org/abs/2403.15316)
[58](https://ieeexplore.ieee.org/document/10446692/)
[59](https://arxiv.org/abs/2509.22414)
[60](https://arxiv.org/abs/2305.04391)
[61](https://arxiv.org/abs/2401.05907)
[62](https://arxiv.org/abs/2410.17521)
[63](https://arxiv.org/html/2407.03635v2)
[64](https://arxiv.org/html/2412.07149)
[65](https://arxiv.org/html/2403.06054v4)
[66](http://arxiv.org/pdf/2502.20679.pdf)
[67](http://arxiv.org/pdf/2311.14900v2.pdf)
[68](https://arxiv.org/html/2410.04618)
[69](http://arxiv.org/pdf/2310.01391.pdf)
[70](https://chatpaper.com/de/chatpaper/paper/109804)
[71](https://www.youtube.com/watch?v=fznLSLSglm0)
[72](https://github.com/ICDM-UESTC/DOSE)
[73](https://arxiv.org/html/2502.13574v2)
[74](https://www.semanticscholar.org/paper/Residual-Denoising-Diffusion-Models-Liu-Wang/4daf51de81e5cf3846482548196a8275369b27c4)
[75](https://arxiv.org/html/2507.15272v1)
[76](https://arxiv.org/html/2509.15952v1)
[77](https://www.semanticscholar.org/paper/69614f326557928d9d142ca0de2e5f572d813f04)
[78](https://arxiv.org/pdf/2406.07646.pdf)
