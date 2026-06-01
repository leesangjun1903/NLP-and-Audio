# ELF: Embedded Language Flows 

---

## 1. 핵심 주장과 주요 기여 (간결 요약)

### 핵심 주장

ELF(Embedded Language Flows)의 핵심 주장은 다음과 같습니다:

> **"연속형 확산 언어 모델(Continuous DLM)은 이산 도메인에 대한 최소한의 적응만으로도 효과적으로 만들 수 있다."**

기존의 주류 확산 언어 모델(DLM)들이 이산 토큰 공간에서 작동하는 것과 달리, ELF는 **연속 임베딩 공간** 에서 전 과정을 수행하고 오직 **마지막 시간 스텝(t=1)** 에서만 이산 토큰으로 변환합니다.

### 주요 기여 요약

| 기여 항목 | 내용 |
|-----------|------|
| **연속 시간 Flow Matching** | 언어 생성에 연속 시간 Flow Matching 최초 적용 |
| **최종 단계 이산화** | 디노이징 전 과정을 연속 공간에서 유지, 마지막에만 이산화 |
| **공유 가중치 네트워크** | 디노이저와 디코더가 동일 네트워크 공유 → 추가 모듈 불필요 |
| **CFG 자연스러운 적용** | 이미지 도메인 기법(Classifier-Free Guidance)을 언어에 직접 적용 |
| **데이터 효율성** | 경쟁 모델 대비 **10배 적은 학습 토큰** 으로 더 나은 성능 |
| **증류 불필요** | 증류(distillation) 없이 소수 샘플링 스텝에서 우수한 성능 |

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

ELF가 해결하려는 핵심 문제는 다음과 같습니다:

**문제 1: 연속형 DLM의 성능 열위**
- 기존 연속형 DLM들은 이산형 DLM(MDLM, Duo 등)에 비해 성능이 낮았음
- 중간 디노이징 단계마다 이산화 손실(cross-entropy)을 적용하여 연속 공간의 자유도를 제한

**문제 2: 이산-연속 공간 인터페이스 설계의 미흡**
- 기존 방법들은 별도의 디코더 모듈이 필요하거나, 매 스텝마다 토큰 레벨 감독이 필요
- 이미지 도메인의 CFG 등 강력한 기법들을 언어에 직접 적용하기 어려움

**문제 3: 학습 비용 문제**
- 경쟁 모델들은 500B+ 토큰으로 학습 후 추가 蒸留까지 필요

---

### 2.2 제안하는 방법 (수식 포함)

#### (1) 이산 토큰 → 연속 임베딩

이산 토큰 시퀀스 $\mathbf{s} = [s_1, \ldots, s_L] \in \mathcal{V}^L$을 사전학습된 T5 인코더를 통해 연속 임베딩 $\mathbf{x}$로 변환합니다.

$$\mathbf{x} = \text{encode}(\mathbf{s})$$

#### (2) Flow Matching (Rectified Flow)

노이즈 $\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$와 데이터 임베딩 $\mathbf{x} \sim p_{\text{data}}(\mathbf{x})$ 사이의 선형 보간(Rectified Flow):

$$\mathbf{z}_t = t\mathbf{x} + (1-t)\boldsymbol{\epsilon}, \quad t \in [0,1]$$

연속 시간에서의 유속(velocity):

$$\mathbf{v} = \frac{d\mathbf{z}_t}{dt} = \mathbf{x} - \boldsymbol{\epsilon}$$

#### (3) x-예측(x-prediction) 파라미터화

ELF는 속도 $\mathbf{v}$를 직접 예측하는 대신, 정제된 임베딩 $\mathbf{x}$를 예측하는 **x-prediction** 방식을 채택합니다. 네트워크 출력 $\mathbf{x}\_\theta = \text{net}_\theta(\mathbf{z}_t, t)$에 대한 MSE 손실:

$$\mathcal{L}_{\text{MSE}} = \mathbb{E}_{t,\mathbf{x},\boldsymbol{\epsilon}} \left\| \mathbf{v}_\theta(\mathbf{z}_t, t) - \mathbf{v} \right\|^2 = \mathbb{E}_{t,\mathbf{x},\boldsymbol{\epsilon}} \frac{1}{(1-t)^2} \left\| \mathbf{x}_\theta(\mathbf{z}_t, t) - \mathbf{x} \right\|^2$$

여기서 $\mathbf{v}(\mathbf{z}_t, t) = (\mathbf{x} - \mathbf{z}_t)/(1-t)$ 관계를 활용합니다.

#### (4) 최종 단계 이산화 (Cross-Entropy Loss)

마지막 타임스텝 $t = 1$에서만 토큰 레벨 이산화를 수행합니다. 손상된 임베딩 $\tilde{\mathbf{z}}$를 입력으로 받아:

$$\mathcal{L}_{\text{CE}} = \mathbb{E}_{\tilde{\mathbf{z}}} \left[ \text{CrossEnt}(\mathbf{W} \mathbf{x}_\theta(\tilde{\mathbf{z}}), \mathbf{s}) \right]$$

여기서 $\mathbf{W}$는 학습 가능한 unembedding 행렬입니다. 전체 학습 손실은 두 목적함수의 조합으로:

- **디노이징 브랜치 (80%)**: $\mathcal{L}_{\text{MSE}}$ 적용
- **디코딩 브랜치 (20%)**: $\mathcal{L}_{\text{CE}}$ 적용

#### (5) Classifier-Free Guidance (CFG)

CFG는 연속 속도장에 대해 자연스럽게 적용됩니다:

$$\mathbf{v}_{\text{cfg}}(\mathbf{z}_t \mid \mathbf{c}) = \omega \mathbf{v}(\mathbf{z}_t \mid \mathbf{c}) + (1-\omega)\mathbf{v}(\mathbf{z}_t \mid \varnothing)$$

여기서 $\omega$는 가이던스 스케일, $\mathbf{c}$는 self-conditioning 신호, $\varnothing$는 무조건부 대응값입니다.

**Training-time CFG**의 회귀 타겟:

$$\mathbf{v}_{\text{target}} = \mathbf{x} - \boldsymbol{\epsilon} + \left(1 - \frac{1}{\omega}\right)\left(\mathbf{v}_\theta^{\text{cfg}}(\mathbf{z}_t \mid t, \mathbf{c}, \omega) - \mathbf{v}_\theta^{\text{cfg}}(\mathbf{z}_t \mid t, \varnothing, \omega)\right)$$

#### (6) Self-Conditioning

Self-conditioning은 이전 스텝의 예측값 $\hat{\mathbf{x}}'$를 현재 스텝의 입력으로 활용합니다:

$$\hat{\mathbf{x}} = \text{net}_\theta(\mathbf{z}_t \mid \hat{\mathbf{x}}', t)$$

훈련 시 50% 확률로 적용하고, 나머지 50%는 영벡터 $\mathbf{0}$을 조건으로 사용합니다.

#### (7) 샘플링 과정

**ODE 샘플러** ($\mathbf{z}_0 \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$에서 시작):

$$\frac{d\mathbf{z}_t}{dt} = \mathbf{v}_\theta(\mathbf{z}_t, t), \quad \mathbf{z}_{t+\Delta t} = \mathbf{z}_t + \Delta t \cdot \mathbf{v}_\theta(\mathbf{z}_t, t)$$

**SDE 샘플러** (노이즈 재주입 스케일 $\gamma$):

$$\alpha = 1 - \gamma \Delta t, \quad t_{\text{back}} = \alpha t, \quad \mathbf{z}_{\text{back}} = \alpha \mathbf{z}_t + (1-\alpha)\boldsymbol{\epsilon}$$

---

### 2.3 모델 구조

```
[이산 토큰 입력]
      ↓
[T5-small Encoder (35M, frozen)] → 연속 임베딩 x (512차원)
      ↓
[Bottleneck 투영] → 128차원 압축
      ↓
[ELF Transformer (Diffusion Transformer)]
  - 구조: SwiGLU, RMSNorm, RoPE, QK-Norm
  - In-context conditioning (시간 토큰, CFG 스케일 토큰, 모드 토큰)
  - Self-conditioning (채널 연결 후 선형 투영)
  - 공유 가중치로 디노이징(t≠1) + 디코딩(t=1) 수행
      ↓
[t=1에서만] → Unembedding 행렬 W → 토큰 로짓 → argmax → 이산 토큰
```

**모델 크기별 구성:**

| 모델 | 레이어 수 | 히든 크기 | 헤드 수 | 파라미터 수 |
|------|-----------|-----------|---------|-------------|
| ELF-B | 12 | 768 | 12 | 105M |
| ELF-M | 24 | 1056 | 16 | 342M |
| ELF-L | 32 | 1280 | 16 | 652M |

**학습 설정:**
- 옵티마이저: Muon (lr = 0.002)
- 배치 크기: 512
- OWT 5 에폭 (약 95K 스텝), 약 45.2B 유효 학습 토큰
- 시간 샘플링: Logit-normal 분포 ($P_{\text{mean}} = -1.5$, $P_{\text{std}} = 0.8$)

---

### 2.4 성능 향상

#### 비조건부 생성 (OpenWebText, Gen. PPL 기준)

| 모델 | 파라미터 | 학습 토큰 | Gen. PPL (32 스텝) | 蒸留 |
|------|----------|-----------|-------------------|------|
| MDLM | 170M | ~524B | >100 | ✗ |
| Duo | 170M | ~524B | ~80 | ✗ |
| MDLM+SDTT | 170M | ~550B | ~50 | ✓ |
| Duo+DCD | 170M | ~550B | ~45 | ✓ |
| FLM/FMLM | 170M | ~524-577B | ~40 | ✓ |
| **ELF-B** | **105M** | **45B** | **~24** | **✗** |

#### 조건부 생성 (WMT14 De-En & XSum)

| 모델 | BLEU ↑ | ROUGE-1 ↑ | ROUGE-2 ↑ | ROUGE-L ↑ |
|------|--------|-----------|-----------|-----------|
| AR | 25.2 | 30.5 | 10.2 | 24.4 |
| MDLM | 18.4 | 33.4 | 11.6 | 25.8 |
| Duo | 21.3 | 31.4 | 10.1 | 25.0 |
| E2D2 | 24.8 | 28.4 | 8.3 | 22.0 |
| **ELF-B** | **26.4** | **36.0** | **12.2** | **27.8** |

---

### 2.5 한계점

논문에서 명시적으로 인정하거나 분석에서 추론 가능한 한계는 다음과 같습니다:

1. **우도(likelihood) 평가 불가**: Flow 기반 모델의 특성상 likelihood 직접 계산이 어려워 검증 퍼플렉시티 미사용 (별도의 likelihood 특화 학습 필요)

2. **인코더 의존성**: 사전학습된 T5 인코더에 의존하므로, 인코더 품질이 ELF 성능에 직접 영향

3. **평가 지표 한계**: Gen. PPL은 GPT-2 Large 기준으로, 특정 스타일의 텍스트에 편향될 수 있음

4. **대규모 확장 미검증**: 현재 최대 652M 파라미터까지만 실험; GPT 계열의 수십억 파라미터 수준 확장성 미검증

5. **학습 수렴 한계**: OWT에서 추가 학습 시 더 이상 성능 향상이 없음을 확인 ("did not observe further performance improvement")

6. **CFG 포화 효과**: CFG 스케일 3 이상에서 효과 역전, 다양성(엔트로피) 감소

7. **SDE 샘플러의 근사성**: 논문이 제안하는 SDE 샘플러는 정확한 확률 SDE가 아닌 근사적 구현

---

## 3. 일반화 성능 향상 가능성

ELF의 일반화 성능 향상 가능성은 여러 측면에서 분석할 수 있습니다.

### 3.1 연속 공간 디노이징의 일반화 이점

기존 방법들은 중간 디노이징 단계마다 이산화 손실을 부과함으로써 모델이 **어휘 수준의 예측에 과도하게 특화**되는 문제가 있었습니다. ELF는 거의 모든 단계에서 연속 임베딩 공간에서만 디노이징을 수행함으로써:

- **유연한 잠재 궤적**: 어휘 수준 제약 없이 최적 디노이징 경로 탐색 가능
- **의미론적 보간**: 연속 공간에서 의미론적으로 유사한 표현들 사이를 자연스럽게 이동
- **표현력 증가**: 이산 토큰 공간에서 직접 확산하는 것보다 더 풍부한 잠재 표현 활용

### 3.2 사전학습 임베딩의 전이 학습 효과

ELF는 동결된(frozen) T5 인코더의 **맥락적 임베딩(contextual embedding)**을 사용합니다. 이는:

- T5가 대규모 코퍼스에서 학습한 언어 지식을 ELF가 상속
- 임베딩 공간이 이미 의미론적으로 구조화되어 있어 Flow Matching이 의미 있는 궤적을 학습하기 용이
- Ablation 실험 결과, 맥락적 임베딩이 비맥락적 임베딩보다 일관되게 우수

#### 임베딩 선택에 따른 성능 비교 (논문 Fig. 5a):

| 임베딩 유형 | 특성 | 성능 순위 |
|-------------|------|-----------|
| 사전학습 T5 인코더 | 맥락적, 고정 | 1위 (최고) |
| 스크래치 학습 인코더 | 맥락적, 고정 | 2위 |
| 사전학습 토큰 임베딩 | 비맥락적, 고정 | 3위 |
| 가우시안 임베딩 | 비맥락적, 고정 | 4위 |
| 학습 가능 임베딩 | 비맥락적, 가변 | 5위 (최저) |

이 결과는 **사전학습된 풍부한 표현이 일반화에 결정적으로 중요함**을 보여줍니다.

### 3.3 CFG를 통한 조건부 생성 일반화

이미지 도메인에서 검증된 CFG를 언어 모델에 직접 적용할 수 있음은 중요한 일반화 이점입니다:

$$\mathbf{v}_{\text{cfg}}(\mathbf{z}_t \mid \mathbf{c}) = \omega \mathbf{v}(\mathbf{z}_t \mid \mathbf{c}) + (1-\omega)\mathbf{v}(\mathbf{z}_t \mid \varnothing)$$

- **품질-다양성 제어**: CFG 스케일 $\omega$로 생성 품질과 다양성 간 트레이드오프 조절
- **텍스트-투-텍스트 일반화**: 번역, 요약 등 다양한 조건부 생성 태스크에 동일 프레임워크 적용
- **이산형 DLM에서 CFG 미적용**: 이산 DLM(MDLM, Duo 등)에서는 CFG가 효과적이지 않음을 논문이 확인

### 3.4 스케일링에 따른 일반화

ELF-B (105M) → ELF-M (342M) → ELF-L (652M)으로 모델 크기를 키울수록:

- 동일 엔트로피에서 Gen. PPL 일관되게 감소
- 동일 Gen. PPL에서 엔트로피(다양성) 증가
- **스케일링 법칙(Scaling Law)** 이 연속형 DLM에서도 성립함을 시사

이는 ELF 프레임워크가 더 큰 모델로 확장할 때 더 나은 일반화를 달성할 가능성을 보여줍니다.

### 3.5 SDE 샘플러의 일반화 기여

SDE 샘플러는 매 단계에서 소량의 노이즈를 재주입함으로써:

- **오류 누적 방지**: ODE의 결정론적 경로에서 초기 오류가 증폭되는 문제 완화
- **Few-step 체제에서의 강건성**: 적은 샘플링 스텝에서도 더 낮은 Gen. PPL 달성
- 이는 모델이 **분포 외(out-of-distribution) 중간 상태**에도 강건하게 대응할 수 있음을 시사

### 3.6 다중 태스크 일반화 증거

동일한 ELF-B 모델(추가 파인튜닝 포함)이 다음 태스크에서 모두 최고 성능을 달성:

- **비조건부 생성**: OWT Gen. PPL = 24 (32 스텝, SDE)
- **기계 번역**: WMT14 De-En BLEU = 26.4 (모든 비교 대상 중 최고)
- **요약**: XSum ROUGE-L = 27.8 (모든 비교 대상 중 최고)

이는 ELF의 연속 임베딩 공간 접근법이 **다양한 언어 생성 태스크에 걸쳐 일반화**됨을 보여줍니다.

---

## 4. 2020년 이후 관련 최신 연구 비교 분석

아래 분석은 ELF 논문의 참고문헌 및 본문에 명시된 관련 연구들을 기반으로 합니다.

### 4.1 이산형 확산 언어 모델 계열

#### DDPM 기반 이산 마스킹 모델

| 연구 | 출판 | 핵심 방법 | ELF 대비 |
|------|------|-----------|----------|
| **D3PM** (Austin et al., NeurIPS 2021) | 2021 | 이산 상태 공간의 일반적 손상 과정 | ELF의 비교 기준점 |
| **DiffusionBERT** (He et al., ACL 2023) | 2023 | BERT + 마스킹 확산 | 이산 공간에서 작동, CFG 미지원 |
| **MDLM** (Sahoo et al., NeurIPS 2024) | 2024 | 단순하고 효과적인 마스킹 확산 | ELF-B가 Gen. PPL에서 크게 앞섬 |
| **Duo** (Sahoo et al., ICML 2025) | 2025 | 균일 분포로의 확산 이중성 | ELF-B가 같은 스텝에서 낮은 PPL |
| **LLaDA** (Nie et al., NeurIPS 2025) | 2025 | 대규모 마스킹 확산 LLM | 대형 모델이나 데이터 효율성 낮음 |
| **DREAM 7B** (Ye et al., 2025) | 2025 | 7B 파라미터 확산 LLM | 대형 모델, ELF와 직접 비교 미수행 |

#### 반자동회귀 및 블록 확산

| 연구 | 출판 | 핵심 방법 | ELF 대비 |
|------|------|-----------|----------|
| **E2D2** (Arriola et al., NeurIPS 2025) | 2025 | 인코더-디코더 확산 LM | 조건부 생성에서 ELF에 열위 |

### 4.2 연속형 확산 언어 모델 계열

#### 임베딩 공간 확산

| 연구 | 출판 | 핵심 특징 | ELF와의 차이 |
|------|------|-----------|-------------|
| **Diffusion-LM** (Li et al., NeurIPS 2022) | 2022 | 최초 연속 확산 LM | 매 스텝 CE 손실, DDPM 방식 |
| **CDCD** (Dieleman et al., 2022) | 2022 | Score-ODE 기반 | 매 스텝 이산화, 별도 디코더 불필요 |
| **DiffuSeq** (Gong et al., ICLR 2023) | 2023 | Seq2Seq 확산 | 매 스텝 CE 손실 |
| **SeqDiffuSeq** (Yuan et al., NAACL 2024) | 2024 | 인코더-디코더 Transformer | ELF에 모든 지표에서 열위 |

#### 잠재 확산 모델 계열

| 연구 | 출판 | 핵심 특징 | ELF와의 차이 |
|------|------|-----------|-------------|
| **LD4LG** (Lovelace et al., NeurIPS 2023) | 2023 | 고정 인코더 잠재 확산 | **별도 AR 디코더 필요**, DDPM |
| **PLANNER** (Zhang et al., NeurIPS 2023) | 2023 | 단락 레벨 잠재 확산 | 별도 디코더, 반자동회귀 |
| **TEncDM** (Shabalin et al., AAAI 2025) | 2025 | 인코더 공간 특성 분석 | VP-DDPM, 별도 디코더 |
| **Cosmos** (Meshchaninov et al., NeurIPS 2025) | 2025 | 압축 잠재 공간 | VP-DDPM, 별도 디코더 |
| **CoDAR** (Shen et al., 2026) | 2026 | VP-SDE, 고정 인코더 | 별도 디코더 필요 |

#### Flow Matching 기반 언어 모델 (동시대 연구)

| 연구 | 출판 | 핵심 특징 | ELF와의 차이 |
|------|------|-----------|-------------|
| **CFM/Categorical FM** (Roos et al., 2026) | 2026 | 심플렉스 공간 FM | 매 스텝 CE 손실, 蒸留 사용 |
| **FLM/FMLM** (Lee et al., 2026) | 2026 | 원-핫 인코딩 FM | 매 스텝 CE 손실, 蒸留 필요 |
| **DFM** (Potaptchik et al., 2026) | 2026 | 이산 Flow Map | 심플렉스 공간, 매 스텝 이산화 |
| **LangFlow** (Chen et al., 2026) | 2026 | Bregman FM + 학습 가능 임베딩 | 매 스텝 CE 손실, ELF에 열위 |

### 4.3 핵심 비교 요약표

논문 Table 2를 기반으로 ELF의 차별점을 정리합니다:

| 방법 | 프로세스 | 상태 공간 | 훈련 중 중간 이산화 | 추론 중 중간 이산화 | 별도 디코더 |
|------|----------|-----------|--------------------|--------------------|-------------|
| Diffusion-LM | DDPM | 학습 임베딩 | ✓ | ✓ | ✗ |
| LD4LG | DDPM | 고정 인코더 | ✗ | ✗ | **✓** |
| FLM | FM | 원-핫 | **✓** | ✗ | ✗ |
| LangFlow | Bregman FM | 학습 임베딩 | **✓** | ✗ | ✗ |
| DFM | FM | 심플렉스 | **✓** | ✗ | ✗ |
| **ELF** | **FM** | **고정 인코더** | **✗** | **✗** | **✗** |

ELF는 중간 이산화도, 별도 디코더도 필요 없는 **유일한 FM 기반 고정 인코더 방식**입니다.

---

## 5. 향후 연구에 미치는 영향과 고려할 점

### 5.1 향후 연구에 미치는 영향

#### 영향 1: 연속형 DLM의 재평가 촉진

ELF는 "연속형 DLM이 이산형 DLM보다 근본적으로 열등하다"는 기존 인식을 뒤집었습니다. 이는 앞으로:

- 연속 임베딩 공간에서의 언어 모델링 연구 활성화
- 이산형 vs. 연속형의 이분법적 구분 대신 **하이브리드 설계** 탐구 촉진
- 이미지 생성 도메인의 기술들(CFG, SDE 샘플러, 蒸留 등)을 언어 도메인에 적용하는 연구 증가

#### 영향 2: 이미지-언어 통합 생성 모델의 가능성

ELF가 이미지 도메인의 Flow Matching을 언어에 직접 적용할 수 있음을 보였으므로, 향후 **이미지와 언어를 동일한 연속 임베딩 공간에서 동시에 생성하는 통합 모델** 개발이 가능해집니다.

#### 영향 3: 데이터 효율적 언어 모델 학습 패러다임

ELF가 45B 토큰으로 500B+ 토큰으로 학습한 모델들을 능가한 것은, 향후 연구에서 **단순히 더 많은 데이터를 사용하는 것보다 설계 선택(아키텍처, 목적함수, 샘플러)이 더 중요할 수 있음**을 시사합니다.

#### 영향 4: 공유 가중치 네트워크 설계 원칙

디노이저와 디코더가 동일 네트워크를 공유하는 설계는 **모듈 수 최소화**의 원칙을 언어 확산 모델에 확립합니다. 이는 멀티모달 생성이나 다중 태스크 학습에서 유사한 설계 원칙 적용으로 이어질 수 있습니다.

#### 영향 5: 통제 가능한 텍스트 생성 연구

CFG가 언어 모델에서도 효과적으로 작동함을 보임으로써, **텍스트 생성의 세밀한 제어**(스타일, 감정, 주제 등)를 위한 연속 확산 기반 접근법 연구가 증가할 것으로 예상됩니다.

---

### 5.2 향후 연구 시 고려할 점

#### 고려점 1: 더 나은 평가 지표 개발

현재 Gen. PPL (GPT-2 Large 기준)은 다음 한계가 있습니다:
- 특정 텍스트 스타일에 편향
- 사실성(factuality), 일관성(coherence) 등 미반영

**제안**: 다양한 평가 지표(다운스트림 태스크 성능, 인간 평가, 사실성 검증 등)를 종합적으로 활용

#### 고려점 2: 우도(Likelihood) 평가 메커니즘 통합

현재 Flow 기반 모델의 우도 계산은 별도의 학습이 필요합니다. 향후 연구에서는:
- Flow 기반 모델의 효율적 우도 추정 방법 (예: 논문 [1]의 공동 蒸留 접근법)
- 또는 우도 없이도 신뢰할 수 있는 평가 프레임워크 개발이 필요

#### 고려점 3: 대규모 확장 검증

현재 ELF는 최대 652M 파라미터까지만 실험되었습니다. **수십억 파라미터 규모**에서도:
- 스케일링 법칙이 유지되는지
- 계산 비용 대비 성능 향상이 유지되는지
- 장문 생성에서의 일관성이 유지되는지 검증 필요

#### 고려점 4: 다양한 언어 및 도메인 일반화

현재 실험이 주로 영어 코퍼스(OWT, WMT14, XSum)에 집중되어 있습니다. 향후 연구에서는:
- 다국어 설정에서의 성능 검증
- 도메인 특화(의료, 법률, 코드) 텍스트에서의 성능 평가
- 인코더를 다국어 모델(mT5, mBERT 등)로 교체 시의 성능 변화

#### 고려점 5: 인코더 설계의 최적화

ELF는 현재 T5-small (35M) 고정 인코더를 사용합니다. 향후 연구에서는:
- 더 강력한 인코더 (T5-large, 또는 LLaMA 계열) 사용 시의 성능 영향
- 인코더를 ELF와 공동 파인튜닝하는 효율적인 방법 탐구
- ELF 자체를 위한 특화 인코더 설계

#### 고려점 6: 실시간 생성 효율성

현재 ELF는 32 스텝에서 Gen. PPL = 24를 달성하지만, 실제 응용에서는:
- 단일 스텝 또는 소수 스텝 생성을 위한 蒸留 방법 탐구
- 자동회귀 모델과의 추론 속도 직접 비교 (토큰/초 기준)
- KV-cache 등 추론 최적화 기법 적용 가능성 검토

#### 고려점 7: 더 긴 시퀀스 처리

현재 ELF는 시퀀스 길이 1024에서 실험되었습니다. 더 긴 컨텍스트를 처리하기 위해서는:
- 비자기회귀 특성상 시퀀스 전체를 동시 처리하므로 메모리 요구량이 크게 증가
- Linear Attention 또는 Sparse Attention 기법과의 결합 필요
- 슬라이딩 윈도우 방식 또는 계층적 생성 방식 탐구

#### 고려점 8: 이론적 기반 강화

ELF의 연속 공간 접근법이 이산 언어 모델링에서 이점을 갖는 이유에 대한:
- **이론적 분석**: 연속 임베딩 공간의 기하학적 구조와 Flow Matching의 관계
- **수렴성 보장**: 유한 스텝 ODE/SDE 샘플러의 이론적 오차 한계
- **표현력 분석**: 연속 확산 모델이 이산 분포를 얼마나 정확하게 근사할 수 있는지

---

## 참고 자료

**주요 참고 논문 (ELF 논문 내 인용 기준):**

1. **ELF 원논문**: Hu, K., Qiu, L., Lu, Y., Zhao, H., Li, T., Kim, Y., Andreas, J., He, K. (2026). *ELF: Embedded Language Flows*. arXiv:2605.10938v1
2. **Flow Matching**: Lipman, Y., et al. (2023). *Flow matching for generative modeling*. ICLR 2023
3. **Rectified Flow**: Liu, X., et al. (2023). *Flow straight and fast: Learning to generate and transfer data with rectified flow*. ICLR 2023
4. **DDPM**: Ho, J., Jain, A., Abbeel, P. (2020). *Denoising diffusion probabilistic models*. NeurIPS 2020
5. **Score-based**: Song, Y., et al. (2021). *Score-based generative modeling through stochastic differential equations*. ICLR 2021
6. **MDLM**: Sahoo, S., et al. (2024). *Simple and effective masked diffusion language models*. NeurIPS 2024
7. **Duo**: Sahoo, S. S., et al. (2025). *The diffusion duality*. ICML 2025
8. **Diffusion-LM**: Li, X., et al. (2022). *Diffusion-LM improves controllable text generation*. NeurIPS 2022
9. **CDCD**: Dieleman, S., et al. (2022). *Continuous diffusion for categorical data*. arXiv:2211.15089
10. **DiffuSeq**: Gong, S., et al. (2023). *DiffuSeq: Sequence to sequence text generation with diffusion models*. ICLR 2023
11. **LD4LG**: Lovelace, J., et al. (2023). *Latent diffusion for language generation*. NeurIPS 2023
12. **CFG**: Ho, J., Salimans, T. (2021). *Classifier-free diffusion guidance*. NeurIPS Workshops 2021
13. **Self-conditioning**: Chen, T., Zhang, R., Hinton, G. (2023). *Analog bits: Generating discrete data using diffusion models with self-conditioning*. ICLR 2023
14. **x-prediction/Back to Basics**: Li, T., He, K. (2025). *Back to basics: Let denoising generative models denoise*. arXiv:2511.13720
15. **LDM**: Rombach, R., et al. (2022). *High-resolution image synthesis with latent diffusion models*. CVPR 2022
16. **Mean Flows**: Geng, Z., et al. (2025). *Mean flows for one-step generative modeling*. NeurIPS 2025
17. **T5**: Raffel, C., et al. (2020). *Exploring the limits of transfer learning with a unified text-to-text transformer*. JMLR 2020
18. **SiT**: Ma, N., et al. (2024). *SiT: Exploring flow and diffusion-based generative models with scalable interpolant Transformers*. ECCV 2024
19. **E2D2**: Arriola, M., et al. (2025). *Encoder-decoder diffusion language models for efficient training and inference*. NeurIPS 2025
20. **LangFlow**: Chen, Y., et al. (2026). *Langflow: Continuous diffusion rivals discrete in language modeling*. arXiv:2604.11748
21. **FLM/FMLM**: Lee, C., et al. (2026). *Flow map language models: One-step language modeling via continuous denoising*. arXiv:2602.16813
22. **DFM**: Potaptchik, P., et al. (2026). *Discrete flow maps*. arXiv:2604.09784
23. **Muon Optimizer**: Jordan, K., et al. (2024). *Muon: An optimizer for hidden layers in neural networks*. Technical report
24. **DiT**: Peebles, W., Xie, S. (2023). *Scalable diffusion models with Transformers*. ICCV 2023
25. **Survey on DLMs**: Li, T., et al. (2025). *A survey on diffusion language models*. arXiv:2508.10875
