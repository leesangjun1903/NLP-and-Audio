# DiffGAN-TTS: High-Fidelity and Efficient Text-to-Speech with Denoising Diffusion GANs

1. 핵심 주장과 주요 기여 (간결 요약)
-----------------------------------

- **핵심 문제의식**  
  DDPM 계열 diffusion TTS는 음질은 매우 좋지만, 수십~수백 번의 denoising step이 필요해 **실시간 TTS에 쓰기 어렵다**는 한계를 가진다.[1][2][3]

- **핵심 아이디어**  
  DiffGAN‑TTS는 “denoising 분포” $\(q(x_{t-1}\mid x_t)\)$ 를 **조건부 GAN으로 직접 근사**함으로써,  
  - denoising step 수를 극단적으로 줄이면서도  
  - 다화자(highly multi-speaker) 환경에서 **고음질·고다양성 TTS**를 달성하는 것을 목표로 한다.[2][1]

- **주요 기여 요약**
  1. **Denoising Diffusion GAN 기반 비자기회귀(non‑AR) TTS**  
     - DDPM의 reverse process를 **Gaussian 가정 대신 조건부 GAN**으로 모델링하여, 큰 step size로도 안정적인 denoising을 가능하게 함.[1][2]
  2. **Multi-speaker TTS에 특화된 구조**  
     - FastSpeech2 스타일 encoder + variance adaptor + **WaveNet 기반 diffusion decoder**,  
     - diffusion‑step & speaker‑aware **JCU(조건+비조건) discriminator** 설계.[4][1]
  3. **Active Shallow Diffusion + 2‑Stage 학습**  
     - 1단계: 기본 acoustic 모델(FastSpeech2 유사)을 학습해 “coarse mel” prior를 만들고.[1]
     - 2단계: 이 coarse mel 기반 **shallow(1 step) diffusion**으로 후처리(post‑filter)하는 구조 제안 → **1 step 만으로도 높은 MOS** 확보.[1]
  4. **성능**  
     - 4‑step 버전(T=4): DiffSpeech(60 step)보다 빠르면서, SSIM/MCD/F0 RMSE 등 객관 지표에서 우수.[1]
     - 2‑stage(1 step): FastSpeech2 수준의 속도(비슷한 RTF)를 유지하면서, MOS 4.17로 GANSpeech 등 강력한 baseline과 비슷하거나 우수.[1]
  5. **한계 인식 및 향후 방향**  
     - 여전히 “acoustic model + HiFi‑GAN vocoder”의 **2‑stage pipeline**이고, text‑to‑wave end‑to‑end 구조로의 확장이 필요하다고 스스로 명시.[1]
     - diffusion step 수 증가 시 speaker similarity가 감소하는 trade‑off를 관찰.[1]

***

2. 해결하고자 하는 문제
-----------------------

### 2.1 기존 TTS의 한계

1. **Autoregressive TTS (Tacotron 계열 등)**  
   - 장점: 매우 높은 자연스러움.  
   - 단점:
     - 프레임 단위 autoregressive decoding → **느린 속도**.  
     - error accumulation으로 인한 **word skipping / repetition** 문제.[1]

2. **Non‑AR TTS (FastSpeech2, GANSpeech 등)**  
   - FastSpeech2: duration/pitch/energy 예측 + transformer 기반 non‑AR mel decoder.[1]
   - GANSpeech: FastSpeech2 구조에 GAN 훈련 추가로 oversmoothing을 완화.[1]
   - 한계:
     - 대부분 **Gaussian‑like 단일 모드 분포**를 가정 → 다화자·다양 prosody에서 **일대다(one‑to‑many) 매핑**을 충분히 표현하기 어려움.  
     - oversmoothing으로 인한 **멜 스펙트로그램 blur** 및 prosody 다양성 부족.[1]

3. **Diffusion TTS (Grad‑TTS, DiffSpeech, Diff‑TTS 등)**  
   - 장점:
     - 매우 복잡한 멜 스펙트로그램 분포를 잘 모델링 → **최상급 음질**.[5][3]
   - 치명적 단점:
     - **수십~수백 step** denoising 필요 → **실시간 TTS 부적합**.[3][5][1]

### 2.2 DiffGAN‑TTS가 노리는 지점

- **질문:** “Diffusion의 표현력 + GAN의 효율성”을 결합해  
  - **TTS 품질은 diffusion 수준**,  
  - **속도는 FastSpeech2/GANSpeech 수준**까지 끌어올릴 수 있는가?  
- 특히 **다화자(multi‑speaker)**, prosody 다양성, one‑to‑many 매핑을 자연스럽게 처리하며,  
  **4 step 이하, 궁극적으로 1 step** denoising으로 고품질을 달성하는 것이 목표.[2][1]

***

3. 제안 방법 (수식 중심 설명)
----------------------------

### 3.1 기본 diffusion 정식화

데이터 분포 \(q(x_0)\) (여기서 \(x_0\): 멜 스펙트로그램)를 대상으로,  
**순방향(diffusion) 과정**은 다음과 같은 Markov chain으로 정의된다:[1]

$$
q(x_{1:T}\mid x_0)
= \prod_{t=1}^T q(x_t\mid x_{t-1}), 
\quad
q(x_t\mid x_{t-1}) = \mathcal N\!\bigl(x_t; \alpha_t x_{t-1}, \beta_t I\bigr)
$$

여기서 $\(\{\beta_t\}\)$는 사전 정의된 variance schedule이며, $\(\alpha_t\)$는 $\(\beta_t\)$로부터 유도되는 scale이다.[1]
이 과정을 여러 step 반복하면 $\(x_T \sim \mathcal N(0,I)\)$와 유사한 잡음 분포로 수렴한다.

**역방향(denoising) 과정**은 파라미터화된 Markov chain:

$$
p_\theta(x_{0:T})
= p(x_T)\prod_{t=1}^T p_\theta(x_{t-1}\mid x_t),
\quad p(x_T)=\mathcal N(0,I)
$$

일반적인 DDPM에서는 각 step을 Gaussian으로 가정한다:

$$
p_\theta(x_{t-1}\mid x_t)
= \mathcal N\!\bigl(x_{t-1}; \mu_\theta(x_t,t), \Sigma_\theta(x_t,t)\bigr)
$$

그러나 **큰 step size**를 쓰면 실제 denoising 분포 $\(q(x_{t-1}\mid x_t)\)$가 **multi‑modal·비Gaussian**이 되어 (3)의 가정이 깨지고, 품질이 나빠진다.[2][1]

### 3.2 Denoising 분포의 GAN 근사

DiffGAN‑TTS의 가장 중요한 수식적 아이디어는:

- “**각 diffusion step에서의 denoising 분포**”

$\[q(x_{t-1}\mid x_t)\]$ 를 **조건부 GAN generator** $\(p_\theta(x_{t-1}\mid x_t)\)$ 로 직접 근사하자는 것.[2][1]

이를 위해 각 step에서 아래와 같은 **divergence 최소화 문제**를 푼다:

$$
\min_\theta
\sum_{t=1}^T 
\mathbb E_{q(x_t)}
\Bigl[
D_{\text{adv}}\bigl( q(x_{t-1}\mid x_t)\;\|\; p_\theta(x_{t-1}\mid x_t) \bigr)
\Bigr]
$$

여기서 $\(D_{\text{adv}}\)$ 는 LSGAN 기반 adversarial divergence이다.[1]

#### (1) LSGAN 기반 discriminator 손실

Discriminator $\(D_\phi(x_{t-1}, x_t, t, s)\)$ (speaker ID $\(s\)$ 포함)는  
**JCU(joint conditional & unconditional)** 구조를 가지며,  
실·가짜 쌍 $\((x_{t-1}, x_t)\)$ 를 구분한다.[4][1]

LSGAN 손실은 다음과 같이 정의된다:[1]

$$
\begin{aligned}
L_D
&= \sum_{t=1}^T
\mathbb E_{q(x_t)q(x_{t-1}\mid x_t)}
\bigl[(D_\phi(x_{t-1},x_t,t,s)-1)^2\bigr] \\
&\quad +
\mathbb E_{q(x_t)p_\theta(x_{t-1}\mid x_t)}
\bigl[D_\phi(\tilde x_{t-1},x_t,t,s)^2\bigr]
\end{aligned}
$$

#### (2) Generator(adversarial) 손실

Generator(= acoustic generator)의 adversarial 손실은

$$
L_{\text{adv}}
= \sum_{t=1}^T
\mathbb E_{q(x_t)p_\theta(x_{t-1}\mid x_t)}
\bigl[(D_\phi(\tilde x_{t-1},x_t,t,s)-1)^2\bigr]
$$

으로 정의된다.[1]

#### (3) Feature matching + 재구성 손실

GAN만으로는 multi-speaker TTS를 안정적으로 학습하기 어렵기 때문에,  
다음 두 가지 regularization을 추가한다:[1]

1. **Feature matching 손실**

```math
L_{\text{fm}}
=
\mathbb E_{q(x_t)}
\sum_{i=1}^N
\left\|D_\phi^{(i)}(x_{t-1},x_t,t,s)
      - D_\phi^{(i)}(\tilde x_{t-1},x_t,t,s)
\right\|_1
```

2. **FastSpeech2 스타일 재구성 손실**

```math
\begin{aligned}
L_{\text{recon}}
&=
L_{\text{mel}}(x_0,\hat x_0)
+ \lambda_d L_{\text{duration}}(d,\hat d) \\
&\quad
+ \lambda_p L_{\text{pitch}}(p,\hat p)
+ \lambda_e L_{\text{energy}}(e,\hat e)
\end{aligned}
```

여기서 $\(\lambda_d,\lambda_p,\lambda_e\)$ 는 모두 0.1로 설정.[1]
멜은 MAE, duration/pitch/energy는 MSE를 사용한다.

3. **최종 Generator 손실**

$$
L_G
= L_{\text{adv}} + L_{\text{recon}} + \lambda_{\text{fm}} L_{\text{fm}}
$$

$\(\lambda_{\text{fm}}\)$ 은 Yang et al.(2021) 방식으로 **동적으로 scaling**한다.[1]

### 3.3 Active shallow diffusion & 2‑stage 학습

DiffGAN‑TTS의 “two‑stage” 변형(1 step denoising)은 다음 아이디어에 기반한다:[1]

1. **Stage‑1: 기본 acoustic 모델 $\(G_{\text{base}}\)$ 학습**  
   - FastSpeech2와 유사한 구조로, 입력 텍스트 $\(y\)$, speaker $\(s\)$ → coarse mel $\(\hat x_0\)$ 를 예측.  
   - 단순 멜 MSE만 쓰지 않고, **“diffused space에서 GT와 예측이 가까워지도록”** 학습한다.

수식으로는,

$$
\min_\psi
\sum_{t=0}^T
\mathbb E_{(x_0,y,s)\sim D}
\Bigl[
\text{Div}\bigl(
q_t^{\text{diff}}(G_{\text{base}}(y,s)),
q_t^{\text{diff}}(x_0)
\bigr)
\Bigr]
$$

여기서 $\(q_t^{\text{diff}}\)$ 는 (1)의 diffusion process를 한 step $\(t\)$ 까지 적용하는 연산이다.[1]

2. **Stage‑2: DiffGAN‑TTS diffusion decoder 학습**  
   - Stage‑1의 encoder/variance adaptor/mel decoder weight를 초기값으로 복사 & freeze.  
   - coarse mel $\(\hat x_0\)$ 를 생성한 뒤, 이를 diffusion decoder의 conditioning으로 사용하여,
     1 step denoising으로 고해상도 mel을 복원하도록 GAN 방식으로 학습.[1]

3. **Inference (two‑stage, 1 step)**

- $\(x_0^{\text{coarse}} = G_{\text{base}}(y,s)\)$  
- diffusion forward로 $\(x_1 \sim q(x_1\mid x_0^{\text{coarse}})\)$ 샘플링  
- diffusion decoder에서 1 step denoising으로 최종 mel $\(\hat x_0\)$ 생성.[1]

이 구조는 diffusion을 **“강력한 post‑filter/super‑resolution 모듈”**로 활용하는 셈이며,  
실험적으로 **1 step만으로도 높은 MOS와 객관 지표**를 달성한다.[1]

***

4. 모델 구조
------------

### 4.1 전체 파이프라인

1. **텍스트 전처리(front‑end)** → phoneme 시퀀스 $\(y\)$.  
2. **Acoustic generator (DiffGAN‑TTS 본체)**:[1]
   - Transformer encoder (FastSpeech2 FFT block 4층, hidden 256, head 2 등)  
   - Variance adaptor  
     - duration predictor, pitch predictor, energy predictor  
     - phoneme‑level F0·energy label 사용.[1]
   - Diffusion decoder  
     - 비인과(non‑causal) WaveNet 구조, dilation=1, residual block 20층, hidden 256.[1]
     - diffusion step embedding + speaker embedding을 residual block에 주입.  
3. **HiFi‑GAN vocoder**  
   - 모든 비교 모델에서 동일한 HiFi‑GAN 설정을 사용해 공정 비교.[6][1]

### 4.2 JCU Discriminator

- Conv‑only 아키텍처로 구성되며,  
  - **unconditional branch**: 순수 real/fake 판별  
  - **conditional branch**: diffusion step embedding, speaker embedding 조건 사용  
- Joint Conditional & Unconditional(JCU) 방식으로,  
  **다화자 scene에서 안정적으로 고품질·고 speaker‑similarity를 유지**하도록 설계.[4][1]

***

5. 성능 향상과 한계
-------------------

### 5.1 객관·주관 평가 및 속도

논문에서 Mandarin 228명, 200시간 다화자 코퍼스로 비교한 결과:[1]

- 비교 모델: FastSpeech2, GANSpeech, DiffSpeech(60 step), DiffGAN‑TTS (T=1,2,4, two‑stage).  
- Objective 지표: SSIM, MCD, F0 RMSE, voice cosine similarity.  
- Subjective 지표: MOS(9‑point), 95% CI.  
- 효율: real‑time factor(RTF) on T4 GPU.

**주요 관찰점**:[1]

1. **DiffGAN‑TTS T=4**
   - SSIM, MCD, F0 RMSE에서 모든 baseline을 **일관되게 상회**.  
   - MOS 4.22로 DiffSpeech(60 step)에 이어 **2위**, 하지만 RTF는 0.0176으로 **한 자릿수 step**임을 감안하면 매우 효율적.

2. **DiffGAN‑TTS two‑stage (1 step, active shallow diffusion)**
   - SSIM·F0 RMSE는 T=1,2보다 더 좋거나 비슷, MCD도 유사 수준.[1]
   - MOS 4.17로 GANSpeech(4.12), FastSpeech2(4.03)보다 우수하며, DiffSpeech(4.28)에 매우 근접.[1]
   - RTF 0.0097로 FastSpeech2(0.0058), GANSpeech(0.0058)와 동급 수준의 **실시간성**.[1]

3. **T step 증가 vs speaker similarity**
   - voice cosine similarity는 T=2에서 최고(0.823), T=1이 그 다음(0.819), T=4가 소폭 하락(0.806).[1]
   - DiffSpeech(60 step)의 speaker similarity는 더 낮음 → **step 수 증가가 speaker identity 유지에는 불리**할 수 있음을 시사.[1]

4. **속도 스케일링**
   - 텍스트 길이(phoneme 수) 대비 inference time 그래프에서,  
     DiffGAN‑TTS T=1은 FastSpeech2와 거의 동일한 scaling, T=2,4 및 two‑stage도 크게 뒤처지지 않음.[1]

### 5.2 Ablation과 한계

Ablation 실험 결과:[1]

- **GAN만(L_adv만) 사용 → 학습 실패**  
  → 멀티스피커, 복잡 one‑to‑many 상황에서 adversarial loss만으로는 불안정.  
- **L_mel 제거 vs L_fm 제거**  
  - L_mel 제거 시 성능 급락 → mel reconstruction이 더 중요.  
  - L_fm 제거 시 성능도 감소하지만 상대적으로 덜.  
- **latent z(StyleGAN식) 추가**  
  - 오히려 SSIM/MCD/F0/cosine similarity 모두 악화.[1]
  - variance adaptor + speaker conditioning만으로 variation modeling이 충분하다는 해석.

**저자들이 명시하는 한계**:[1]

1. 여전히 **acoustic model + neural vocoder의 2‑stage 구조**  
   - end‑to‑end text‑to‑wave로 통합하면 latency 단축·오류 누적 감소 여지가 큼.  
2. 다화자 Mandarin에 대해서만 실험  
   - 언어·도메인·zero‑shot 화자 일반화에 대한 평가는 제한적.  
3. GAN 기반 adversarial 학습의 안정성·hyper‑parameter 민감성  
   - dual loss(L_adv+L_recon+L_fm)의 균형 조절이 필요.

***

6. 일반화 성능 관점에서의 분석
-----------------------------

질문에서 강조한 “**모델의 일반화 성능 향상 가능성**”을 중심으로 정리하면:

### 6.1 왜 diffusion‑GAN이 일반화에 유리할 수 있는가

1. **denoising distribution 직접 학습**  
   - $\(q(x_{t-1}\mid x_t)\)$ 는 “잡음이 추가된 멜에서 깨끗한 멜로 복원하는 조건부 분포”이다.  
   - 이를 Gaussian이 아닌 **고표현력 조건부 GAN**으로 모델링하면,  
     - 복잡한 멀티모달 prosody·speaker variation을 더 자연스럽게 반영할 수 있고,  
     - 훈련 분포 주변에서 **강한 local regularization(denoising as manifold projection)** 역할을 한다.[3][2][1]
   - 이는 unseen 텍스트·prosody에 대해 자연스러운 멜을 생성하는 데 도움이 될 수 있다.

2. **다화자 + speaker‑aware JCU discriminator**  
   - discriminator가 diffusion step·speaker를 명시적으로 conditioning하기 때문에,  
     - 화자별 acoustic 특징을 분리하고,  
     - step마다 필요한 denoising 형태를 학습한다.[4][1]
   - 이 구조는 **화자 간 분포 혼선(“speaker leakage”)**을 줄이고,  
     - 다화자 환경에서도 모델이 잘 일반화하도록 돕는다.

3. **variance adaptor 기반 factorization**  
   - duration/pitch/energy 예측을 통해 text → prosody → mel로 factorization함으로써,  
     - 텍스트·prosody·speaker 간 변동성을 어느 정도 disentangle한다.[1]
   - 이는 새 텍스트나 발화 길이, prosody 패턴에 대한 **조합적 일반화(combinatorial generalization)**에 유리하다.

### 6.2 Active shallow diffusion의 일반화 효과

- Stage‑1 기본 acoustic 모델은 비교적 간단한 MSE/MAE 기반 학습으로,  
  - **coarse but robust한 멜 prior**를 형성한다.[1]
- Stage‑2 diffusion decoder는 이 prior를 기반으로  
  - “세부 texture 보정 + 고주파수 정보 복원”에 집중한다 (post‑filter 역할).[1]

이 구조는 일반화 측면에서 다음처럼 해석할 수 있다:

1. **Base model이 global structure를, diffusion이 local detail을 담당**  
   - base model이 alignment·내용·거친 prosody 등 “전역적 요소”를 책임지고,  
   - diffusion은 oversmoothing을 해소하면서도 base mel에서 크게 벗어나지 않는 범위에서 세부 구조를 추가.  
   - 따라서 overfitting risk가 줄고,  
     - 특히 데이터가 적은 화자나 텍스트에 대해 더 **안정된 출력**을 기대할 수 있다.

2. **Diffusion step 수 축소와 generalization의 trade‑off**  
   - 실험적으로 step 수가 많아질수록 **speaker similarity 감소**가 관측.[1]
   - 이는 과도한 refinement가 **화자 고유 특성까지 변경**할 수 있음을 시사한다.  
   - shallow diffusion(1~4 step)은  
     - base 모델이 이미 학습한 speaker manifold 근처에서만 국소 보정을 수행하므로,  
     - **보수적인 수정 → 더 나은 speaker/generalization trade‑off**를 제공할 수 있다.

3. **다양성 vs speaker consistency**  
   - DiffGAN‑TTS는 동일 텍스트·화자에 대해 여러 번 샘플링 시 다양한 pitch contour를 생성할 수 있음이 F0 실험으로 확인된다.[1]
   - 이는 **stochasticity를 통한 데이터 증강·style 다양화**에 유리하지만,  
   - 지나친 다양성은 화자 일관성을 해칠 수 있음 → 실제 시스템 설계 시 균형 조절 필요.

***

7. 2020년 이후 관련 최신 연구 동향
---------------------------------

DiffGAN‑TTS(2022)를 기점으로, **Diffusion + GAN 결합 또는 효율적인 diffusion TTS** 연구가 활발해졌다.

### 7.1 Diffusion‑GAN 계열 후속 TTS 연구

1. **MixGAN‑TTS (2023)**[7]
   - DiffGAN‑TTS와 유사하게 diffusion의 denoising 분포를 GAN으로 모델링하여 step 수를 줄이는 non‑AR TTS.  
   - linguistic encoder에서 soft phoneme alignment + hard word alignment로 **단어 레벨 semantic 정보**를 잘 추출하고,  
   - HiFi‑GAN vocoder를 사용해, 4 step만으로도 다른 모델들보다 좋은 품질을 보고.

2. **Adversarial Training of Denoising Diffusion Model Using Dual Discriminators (2023)**[8][9]
   - DiffGAN‑TTS를 직접 언급하며,  
   - **2개의 discriminator**(diffusion discriminator + spectrogram discriminator)를 도입.  
   - reverse process 분포와 최종 mel 분포를 각각 학습하여, SSIM/MCD/F0/MOS 등에서 DiffGAN‑TTS를 상회하는 결과 보고.[9][8]
   - 이는 **denoising 단계와 최종 결과 분포를 분리해서 감시**하는 것이 성능·일반화에 도움이 될 수 있음을 시사.

3. **Prosodic Diff‑TTS (Style Description 기반 Conditional PL‑Norm + Diffusion GAN, 2023)**[10]
   - 텍스트 + style description(예: pitch, speaking speed, emotion, gender 등) 기반으로 멀티스피커 음성을 생성.  
   - BERT에서 추출한 style embedding을 **conditional prosodic layer norm**으로 encoder/decoder에 주입하고,  
   - diffusion‑GAN 구조로 4 step 이내에서 **style‑controllable 고품질 음성**을 생성.[10]
   - DiffGAN‑TTS 아이디어를 **style controllability** 방향으로 확장한 사례.

4. **DETS: End‑to‑End Single‑Stage TTS via Hierarchical Diffusion GAN (2024)**[11]
   - 기존 2‑stage TTS 대신, **단일 단계에서 text→waveform**까지 hierarchical diffusion‑GAN으로 모델링.  
   - duration predictor와 speech decoder를 각각 diffusion‑GAN으로 parameterize하여,  
   - 자연스러운 prosody 다양성과 고음질을 1 step denoising으로 달성.[11]
   - DiffGAN‑TTS가 제안했던 “denoising distribution을 GAN으로 모델링” 아이디어를 **end‑to‑end TTS** 방향으로 진화시킨 예.

5. **Bilingual/Code‑switching TTS with Diffusion + GAN (IFS‑DiffGAN, 2024)**[12][13]
   - Mandarin–English bilingual 및 code‑switching TTS에서 diffusion+GAN을 사용해 품질 및 speaker consistency 향상.  
   - DiffGAN‑TTS를 참조하며 **멀티링구얼·코드스위칭 일반화**에 diffusion‑GAN 구조가 효과적임을 보여준다.[13][12]

6. **DiffGAN‑ZSTTS (Zero‑Shot Speaker Adaptation with Diffusion GAN, 2025)**[14][15]
   - FastSpeech2 프레임워크 위에 diffusion‑based decoder를 더해,  
     - **zero‑shot unseen speaker**에 대한 음질·speaker similarity를 크게 개선.  
   - encoder에 SE‑Res2Net, decoder에 multi‑head speaker encoder(MHSE)를 도입해,  
     - **unseen speaker generalization**에 초점을 맞춘 DiffGAN 계열 연구.[15][14]

7. **DiffGAN‑VC (Voice Conversion, 2023)**[16]
   - non‑parallel many‑to‑many voice conversion에서 denoising diffusion GAN 구조를 적용해,  
   - 큰 step size, 적은 denoising step으로도 high‑quality VC 달성.

**정리:** DiffGAN‑TTS 이후, diffusion‑GAN 아이디어는  
– 속도 개선 뿐 아니라, **multi‑lingual, style control, zero‑shot, VC 등 일반화 어려운 시나리오**로 확장되고 있다.

### 7.2 효율적 diffusion TTS (비‑GAN 계열)

1. **ResGrad (Residual DDPM for TTS, 2022)**[5]
   - 멜 전체 대신 **잔차(residual)**만 diffusion으로 모델링해 계산량을 줄이면서도 고품질 유지.  
   - DiffGAN‑TTS처럼 diffusion의 역할을 “refinement”로 한정한다는 점에서 개념적으로 유사.

2. **CM‑TTS (Consistency Model 기반 Real‑Time TTS, 2024)**[17]
   - diffusion + GAN 결합의 adversarial instability를 지적하고,  
   - **Consistency Model(CM)**을 사용해 매우 적은 step으로도 고품질 멜 생성.  
   - weighted sampler를 도입하여 학습 시 전체 noise level을 고르게 커버하고,  
   - 기존 single‑step TTS보다 품질이 우수하다고 보고.[17]
   - 이는 DiffGAN‑TTS류 “GAN 기반 가속”에 대한 **비‑adversarial 대안**으로 볼 수 있다.

3. **Sample‑Efficient Diffusion for TTS (2024)**[6]
   - EnCodec 기반 latent space에서 diffusion을 수행하여,  
   - **latent diffusion**으로 sample 효율·속도를 크게 개선.  
   - speaker‑prompted TTS, audio inpainting 등을 동시에 수행하면서, 텍스트 only·multi‑speaker 모두에서 강력한 성능을 보임.[6]

4. **Semantic Latent Space of Diffusion TTS (2024)**[18]
   - diffusion TTS의 latent space가 어떤 **semantic control**(예: 발화 스타일, 감정)을 허용하는지 분석.  
   - 향후 diffusion‑GAN 계열 모델에서도 **해석 가능한 latent control**을 도입하는 방향으로 이어질 수 있다.

***

8. 향후 연구에의 영향과 앞으로 고려할 점
--------------------------------------

### 8.1 DiffGAN‑TTS가 남긴 영향

1. **“Diffusion = 느리다” 고정관념을 깬 사례**  
   - 4 step(또는 1 step) diffusion으로도  
     - 멀티스피커 환경에서 고품질 TTS가 가능함을 보였고,[1]
     - 이후 MixGAN‑TTS, DETS, DiffGAN‑ZSTTS 등 다수 연구가 **few‑step diffusion‑GAN**을 채택하게 만드는 계기.[14][7][15][11]

2. **Denoising distribution을 직접 GAN으로 모델링하는 프레임워크**  
   - reverse process 각 step을 explicit GAN mapping으로 보는 시각은,  
     - TTS뿐 아니라 gesture generation 등 다른 modality에도 확장되고 있다.[19]

3. **Shallow diffusion + base model prior라는 설계 패턴**  
   - DiffGAN‑TTS two‑stage, DiffSpeech, ResGrad, Sample‑Efficient Diffusion 등  
     - “기본 모델 + diffusion refinement” 구조가 표준 패턴으로 자리잡는 데 기여.[5][6][1]

4. **다화자·zero‑shot·multi‑lingual 일반화 연구의 기초**  
   - 이후 DiffGAN‑ZSTTS, code‑switching TTS, zero‑shot VC 등에서  
     - DiffGAN‑TTS의 구조와 아이디어가 반복적으로 인용·활용된다.[12][15][16][13][14]

### 8.2 앞으로 연구 시 고려할 점 (연구자 관점 제언)

1. **모델 구조 관점**
   - **End‑to‑end text‑to‑wave**  
     - DETS와 같이 diffusion‑GAN을 waveform까지 확장해 pipeline을 단순화하고,  
     - acoustic/vocoder 경계에서의 mismatch를 줄이는 연구 필요.[11]
   - **Latent‑space diffusion‑GAN**  
     - Sample‑Efficient Diffusion처럼 auto‑encoder latent에서 diffusion‑GAN을 돌려  
       - step 수와 계산량을 줄이면서,  
       - semantic control을 용이하게 하는 방향.[18][6]

2. **학습 안정성과 GAN 대안**
   - dual discriminator, feature matching, reconstruction loss 등 **안정화 테크닉**을 체계적으로 비교·분석할 필요.[8][9]
   - CM‑TTS처럼 **consistency model / distillation** 기반 가속이  
     - DiffGAN‑TTS류 구조를 대체하거나 보완할 수 있는지도 중요한 연구 질문.[20][17]

3. **일반화 평가 프로토콜 강화**
   - 현재 대부분 연구가 **단일 언어·제한된 화자 집합**에서 평가.  
   - 향후에는  
     - zero‑shot unseen speaker, cross‑lingual, noisy 환경, OOD 텍스트 등에 대한  
       **체계적인 generalization benchmark**가 필요.  
   - 특히 DiffGAN‑계열은 adversarial training 특성상  
     - 데이터 분포 밖에서의 행동을 면밀히 분석해야 한다.

4. **Style/Prosody 제어와 해석 가능한 latent**
   - Prosodic Diff‑TTS, StyleTTS‑ZS 등에서 보듯,  
     - 텍스트 외 style description, reference audio를 통해  
       **세밀한 prosody·style control**이 점점 중요해지고 있다.[20][10]
   - diffusion‑GAN의 denoising step/latent를  
     - interpretable하게 제어할 수 있는 기법 (semantic latent 발견, controllable diffusion path 등)이 향후 큰 주제가 될 것.

5. **실시간·대규모 서비스 관점**
   - RTF, 메모리 footprint, GPU 병렬성, quantization·distillation 등을 고려한  
     - **산업적 배치 가능성** 연구가 필수.  
   - 특히 multi‑speaker·multi‑lingual 대규모 시스템에서는  
     - speaker encoder, language embedding, prosody embedding 등이  
       - latency와 generalization 간 trade‑off에 어떤 영향을 주는지 분석이 필요하다.

***

정리하면, DiffGAN‑TTS는  
- diffusion의 표현력과 GAN의 효율성을 결합해  
- 소수 step denoising으로도 고품질·다화자 TTS를 달성할 수 있음을 처음으로 명확히 보여준 작업이며,[2][1]
- 이후 diffusion‑GAN, shallow diffusion, zero‑shot generalization, end‑to‑end TTS 등 여러 연구 방향에 중요한 **방법론적 토대**를 제공했다.  

앞으로는 이 프레임워크를  
- end‑to‑end, latent diffusion, consistency model, interpretable control 등과 결합하여  
- **일반화 성능과 효율성을 동시에 극대화하는 방향**으로 확장하는 것이 핵심 과제가 될 것이다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/f3eb55b9-3906-41b6-ac09-6b376e5e1c01/2201.11972v1.pdf)
[2](https://www.semanticscholar.org/paper/098bbff093da93e7838c9faf5936ccb6562cf2a1)
[3](https://arxiv.org/pdf/2104.01409.pdf)
[4](https://kimjy99.github.io/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0/diffgantts/)
[5](https://arxiv.org/pdf/2212.14518.pdf)
[6](https://www.arxiv.org/pdf/2409.03717.pdf)
[7](https://ieeexplore.ieee.org/document/10145456/)
[8](https://ieeexplore.ieee.org/document/10494889/)
[9](https://arxiv.org/abs/2308.01573)
[10](https://arxiv.org/abs/2310.18169)
[11](https://ieeexplore.ieee.org/document/10446855/)
[12](https://www.isca-archive.org/interspeech_2024/yang24i_interspeech.html)
[13](https://www.isca-archive.org/interspeech_2024/yang24i_interspeech.pdf)
[14](https://www.nature.com/articles/s41598-025-90507-0)
[15](https://pmc.ncbi.nlm.nih.gov/articles/PMC11842752/)
[16](https://arxiv.org/pdf/2308.14319.pdf)
[17](https://arxiv.org/abs/2404.00569)
[18](http://arxiv.org/pdf/2402.12423.pdf)
[19](https://arxiv.org/html/2410.20359v1)
[20](https://aclanthology.org/2025.naacl-long.242.pdf)
[21](https://ieeexplore.ieee.org/document/10986870/)
[22](https://arxiv.org/abs/2201.11972)
[23](https://aclanthology.org/2023.findings-acl.437.pdf)
[24](https://www.isca-archive.org/interspeech_2024/janiczek24_interspeech.pdf)
[25](https://aclanthology.org/2024.findings-emnlp.533.pdf)
[26](https://openreview.net/pdf?id=c7vkDg558Z)
