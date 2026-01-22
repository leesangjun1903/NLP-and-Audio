# One Transformer Fits All Distributions in Multi-Modal Diffusion at Scale

# 1. 논문의 핵심 주장과 주요 기여 (간결 요약)

이 논문의 핵심 주장은 다음 한 문장으로 정리할 수 있습니다.

> **하나의 Transformer 기반 diffusion 모델(UniDiffuser)만으로, 동일한 멀티모달 데이터셋에서 정의되는 “모든” 관련 분포(주변, 조건부, 결합 분포)를 동시에 학습하고, 추가 비용 없이 다양한 생성 태스크(텍스트, 이미지, 텍스트→이미지, 이미지→텍스트, 이미지–텍스트 공동 생성)를 수행할 수 있다**는 것입니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/ba129847-02ab-42e5-9c8e-73ebe7861663/2303.06555v2.pdf)

주요 기여를 정리하면:

1. **통합 이론 프레임워크**  
   - 소거 분포 $\(q(x_0)\)$ , 조건부 분포 $\(q(x_0\mid y_0)\)$ , 결합 분포 $\(q(x_0,y_0)\)$ 등을 **모두 “노이즈 예측” 문제**로 통일하는 관점을 제시. [proceedings.mlr](https://proceedings.mlr.press/v202/bao23a/bao23a.pdf)
   - 모달리티별로 서로 다른 timestep $\((t_x,t_y)\)$ 를 부여하면, 적절한 $\((t_x,t_y)\)$ 선택만으로 주변/조건부/결합 분포를 모두 표현할 수 있음을 보임.

2. **UniDiffuser: 단일 노이즈 예측 네트워크**  
   - 두 모달리티(이미지 $\(x\)$ , 텍스트 $\(y\))$ 에 대해, 모든 쌍 $\((t_x,t_y)\)$ 에 대해 노이즈 $\((\epsilon_x,\epsilon_y)\)$ 의 조건부 기댓값  

$\(\mathbb{E}[\epsilon_x,\epsilon_y\mid x_{t_x},y_{t_y}]\)$ 을 하나의 Transformer로 예측하는 **공동 노이즈 예측 네트워크**를 제안. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/ba129847-02ab-42e5-9c8e-73ebe7861663/2303.06555v2.pdf)
   - 학습 목적 함수는 표준 DDPM과 거의 동일한 단일 MSE 손실.

3. **“Classifier-free guidance를 공짜로” 구현**  
   - 하나의 모델이 이미 주변 및 조건부 분포를 모두 모델링하고 있기 때문에, **추가 모델이나 null token 없이** classifier-free guidance(CFG)를 자연스럽게 구현하는 수식적 해석을 제시. [proceedings.mlr](https://proceedings.mlr.press/v202/bao23a/bao23a.pdf)

4. **대규모 이미지–텍스트 실험 및 성능 검증**  
   - LAION-5B 서브셋 기반 대규모 학습으로,  
     - 텍스트→이미지, 이미지→텍스트, 이미지/텍스트 단일 모달 생성, 이미지–텍스트 공동 생성 등 다양한 태스크를 **추가 파라미터나 별도 모델 없이** 수행. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/ba129847-02ab-42e5-9c8e-73ebe7861663/2303.06555v2.pdf)
   - 대표적인 “범용” 모델인 Versatile Diffusion(VD)보다 여러 지표(FID, CLIP score)에서 우수하고, 텍스트→이미지에서는 Stable Diffusion, DALL·E 2 등 bespoke 모델에 근접한 성능 달성. [openaccess.thecvf](https://openaccess.thecvf.com/content/ICCV2023/papers/Xu_Versatile_Diffusion_Text_Images_and_Variations_All_in_One_Diffusion_ICCV_2023_paper.pdf)

5. **멀티모달 일반화 및 응용 데모**  
   - 하나의 학습된 분포를 통해 **이미지·텍스트 변형, 블록 기브스 샘플링, 두 이미지 사이의 인터폴레이션 등** 다양한 응용이 자연스럽게 가능함을 보임. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/ba129847-02ab-42e5-9c8e-73ebe7861663/2303.06555v2.pdf)


***

# 2. 논문이 다루는 문제, 제안 방법(수식 포함), 모델 구조, 성능 및 한계

## 2.1 해결하고자 하는 문제

기존 대규모 생성 모델의 한계:

1. **태스크/분포별 별도 모델 필요**  
   - GLIDE, Imagen, Stable Diffusion 등은 주로 **텍스트→이미지 하나의 조건부 분포** $\(p(\text{Image}\mid \text{Text})\)$ 만을 학습하는 bespoke 시스템. [proceedings.mlr](https://proceedings.mlr.press/v162/nichol22a.html)
   - 이미지→텍스트(캡셔닝), 이미지 단일 모달 생성, 텍스트 단일 모달 생성 등은 별도의 모델이나 아키텍처가 필요.

2. **공동 분포 기반 멀티모달 모델의 비효율성**  
   - 고전적 접근: $\(p(x_0,y_0)\)$ 를 학습한 뒤 MCMC 등으로 주변/조건부 분포를 얻는 방식은, LAION-5B급 대규모 데이터에서는 **추론 비용이 지나치게 큼**. [arxiv](https://arxiv.org/abs/2211.14842)

3. **멀티모달 “범용” 확률 모델의 부재**  
   - Versatile Diffusion(VD)은 멀티플로우 구조로 여러 태스크를 수행하지만, **태스크별 loss, 플로우별 gradient tuning** 등 복잡한 multi-task 설계가 필요하고, joint generation에서의 이론적 정합성은 상대적으로 약함. [arxiv](https://arxiv.org/pdf/2211.08332.pdf)

이에 대해 논문은 다음을 목표로 합니다.

> 하나의 diffusion 모델로, 같은 이미지–텍스트 데이터셋에서 정의되는  
> $\(\{q(x_0),q(y_0),q(x_0\mid y_0),q(y_0\mid x_0),q(x_0,y_0)\}\)$  
> 를 **명시적으로 동시에 근사**하고, 어떤 태스크든 “timestep 설정만 바꾸어” 수행 가능하게 만들기.


## 2.2 제안 방법: 통합된 노이즈 예측 관점

### 2.2.1 단일 모달 DDPM 복습

표준 diffusion model(DDPM)에서는 데이터 $\(x_0\sim q(x_0)\)$ 에 대해 전방 노이즈 주입 과정이 [arxiv](http://arxiv.org/abs/2011.13456)

$$
q(x_{1:T}\mid x_0) = \prod_{t=1}^T q(x_t\mid x_{t-1}), \quad
q(x_t\mid x_{t-1}) = \mathcal{N}\bigl(x_t\mid \sqrt{\alpha_t}x_{t-1}, \beta_t I\bigr),
$$

로 정의되고, 누적 계수 $\(\bar\alpha_t=\prod_{i=1}^t(1-\beta_i)\)$ 에 대해

$$
x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon_x,\quad \epsilon_x\sim\mathcal{N}(0,I).
$$

역과정은

$$
p_\theta(x_{t-1}\mid x_t) = \mathcal{N}\bigl(x_{t-1}\mid \mu_\theta(x_t,t),\sigma_t^2 I\bigr)
$$

이고, 최대우도 기준 최적 평균은: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/ba129847-02ab-42e5-9c8e-73ebe7861663/2303.06555v2.pdf)

$$
\mu_t^\ast(x_t) = \frac{1}{\sqrt{\alpha_t}}
\Bigl(
x_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\,\mathbb{E}[\epsilon_x\mid x_t]
\Bigr).
$$

따라서 학습은 **노이즈 예측 네트워크** $\(\epsilon_\theta(x_t,t)\)$ 의 회귀로 귀결됩니다.

$$
\min_\theta \mathbb{E}_{t,x_0,\epsilon_x}
\bigl\| \epsilon_x - \epsilon_\theta(x_t,t)\bigr\|_2^2
\quad\Rightarrow\quad
\epsilon_\theta^\ast(x_t,t)=\mathbb{E}[\epsilon_x\mid x_t]. 
$$

조건부 생성 $\(q(x_0\mid y_0)\)$ 에서는 조건 $\(y_0\)$ 을 함께 넣어  
$\(\epsilon_\theta(x_t,y_0,t)\)$ 를 학습하면 되고, 이때 최적 평균은: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/ba129847-02ab-42e5-9c8e-73ebe7861663/2303.06555v2.pdf)

$$
\mu_t^\ast(x_t,y_0) = \frac{1}{\sqrt{\alpha_t}}
\Bigl(
x_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\,\mathbb{E}[\epsilon_x\mid x_t,y_0]
\Bigr). 
$$


### 2.2.2 UniDiffuser의 핵심 통찰: 모든 분포를 하나의 조건부 기댓값으로 통합

2모달리티 $\((x_0,y_0)\sim q(x_0,y_0)\)$ 가 있을 때, 양쪽에 독립적으로 forward diffusion을 적용합니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/ba129847-02ab-42e5-9c8e-73ebe7861663/2303.06555v2.pdf)

$$
\begin{aligned}
x_{t_x} &= \sqrt{\bar\alpha_{t_x}}x_0 + \sqrt{1-\bar\alpha_{t_x}}\,\epsilon_x,\\
y_{t_y} &= \sqrt{\bar\alpha_{t_y}}y_0 + \sqrt{1-\bar\alpha_{t_y}}\,\epsilon_y,
\end{aligned}
\quad
\epsilon_x,\epsilon_y\sim\mathcal{N}(0,I).
$$

여기서 **관찰**은 다음과 같습니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/ba129847-02ab-42e5-9c8e-73ebe7861663/2303.06555v2.pdf)

- 주변 분포:
  - $\(q(x_0)\)$ 를 diffusion으로 모델링하는 데 필요한 것은  
    $\(\mathbb{E}[\epsilon_x\mid x_{t_x}]\)$ .
- 조건부 분포:
  - $\(q(x_0\mid y_0)\)$ 에는  
    $\(\mathbb{E}[\epsilon_x\mid x_{t_x},y_0]\)$ .
- 결합 분포:
  - $\(q(x_0,y_0)\)$ 에는  
    $\(\mathbb{E}[\epsilon_x,\epsilon_y\mid x_t,y_t]\)$ .

이 **모든 양**이 사실은 하나의 더 일반적인 조건부 기댓값

$$
\mathbb{E}[\epsilon_x,\epsilon_y\mid x_{t_x},y_{t_y}], \quad 0\le t_x,t_y\le T
$$

의 특수한 경우라는 점입니다. [proceedings.mlr](https://proceedings.mlr.press/v202/bao23a/bao23a.pdf)

여기서 timestep 선택에 따라 의미가 바뀝니다.

- ** $\(t_y=T\)$ ** (완전한 노이즈):  
  $\(y_{t_y}\approx\epsilon_y\)$ 이므로, $\(y\)$ 정보를 사실상 버린 상태 →  
  $\(\mathbb{E}[\epsilon_x\mid x_{t_x},y_T]\approx \mathbb{E}[\epsilon_x\mid x_{t_x}]\)$  
  ⇒ 주변 분포 $\(q(x_0)\)$ .
- ** $\(t_y=0\)$ ** (깨끗한 텍스트):  
  $\(\mathbb{E}[\epsilon_x\mid x_{t_x},y_0]\)$  
  ⇒ 조건부 분포 $\(q(x_0\mid y_0)\)$ .
- ** $\(t_x=t_y=t\)$ **:  
  $\(\mathbb{E}[\epsilon_x,\epsilon_y\mid x_t,y_t]\)$  
  ⇒ 결합 분포 $\(q(x_0,y_0)\)$ .

결국 **모든 목표 분포**를 하나의 통합된 노이즈 예측 문제로 표현할 수 있습니다.


### 2.2.3 통합 학습 목적 함수

이를 위해 논문은 **공동 노이즈 예측 네트워크**

$$
\epsilon_\theta(x_{t_x},y_{t_y},t_x,t_y)
\in\mathbb{R}^{d_x+d_y}
$$

를 도입하고, $\(\epsilon_x,\epsilon_y\)$ 를 concat한 벡터를 예측하도록 학습합니다. [proceedings.mlr](https://proceedings.mlr.press/v202/bao23a/bao23a.pdf)

학습 손실:

```math
\mathcal{L}(\theta)
=
\mathbb{E}_{x_0,y_0,\epsilon_x,\epsilon_y,t_x,t_y}
\bigl\|
\epsilon_\theta(x_{t_x},y_{t_y},t_x,t_y)
-
[\epsilon_x,\epsilon_y]
\bigr\|_2^2.
```

- $\(t_x,t_y \sim \text{Uniform}\{1,\dots,T\}\)$ 독립 샘플.  
- forward noising은 각각의 모달리티에 대해 독립적으로 수행.  
- **한 번의 forward–backward pass**에서 “모든” 분포 관련 학습 신호를 동시에 받음.

이렇게 학습된 $\(\epsilon_\theta\)$ 를 이용해, 적절한 $\((t_x,t_y)\)$ 와 샘플러(DPM-Solver 등)를 지정하면:

- $\(t_y=0\)$ : $\(x\)$ 를 $\(y\)$ 에 조건부로 생성 (텍스트→이미지)  
- $\(t_x=0\)$ : $\(y\)$ 를 $\(x\)$ 에 조건부로 생성 (이미지→텍스트)  
- $\(t_x=t_y\)$ : 공동 샘플링 (이미지–텍스트 pair joint generation)  
- $\(t_y=T\)$ : $\(x\)$ 주변 분포 샘플링 (unconditional image)  
- $\(t_x=T\)$ : $\(y\)$ 주변 분포 샘플링 (unconditional text)

이 모두가 **동일한 샘플링 루프** 안에서 단지 timestep 설정만 다르게 하는 것으로 구현됩니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/ba129847-02ab-42e5-9c8e-73ebe7861663/2303.06555v2.pdf)


### 2.2.4 Classifier-Free Guidance(CFG)를 “공짜로” 구현

기존 CFG는 조건부 모델 $\(\epsilon_\theta(x_t,y_0,t)\)$ 과 무조건 모델 $\(\epsilon_\theta(x_t,t)\)$ 를 선형 결합합니다. [proceedings.mlr](https://proceedings.mlr.press/v162/nichol22a.html)

```math
\hat\epsilon_\theta(x_t,y_0,t)
=
(1+s)\,\epsilon_\theta(x_t,y_0,t)
-
s\,\epsilon_\theta(x_t,t).
```

UniDiffuser에서는 **하나의 네트워크**가 이미 조건부와 무조건부 모두를 커버합니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/ba129847-02ab-42e5-9c8e-73ebe7861663/2303.06555v2.pdf)

예를 들어 텍스트→이미지에서:

- 조건부: $\(\epsilon^x_\theta(x_t,y_0,t,0)\)$
- 무조건: $\(\epsilon^x_\theta(x_t,\epsilon_y,t,T)\)$ (텍스트 모달리티를 최대 timestep으로 노이즈화)

따라서

```math
\hat\epsilon^x_\theta(x_t,y_0,t)
=
(1+s)\,\epsilon^x_\theta(x_t,y_0,t,0)
-
s\,\epsilon^x_\theta(x_t,\epsilon_y,t,T).
```

joint sampling에서도 비슷한 방식으로, joint score를 조건부 score와 주변 score의 조합으로 해석하여 CFG를 정의합니다. CFG용으로 **별도 모델이나 null token이 필요 없다는 점**이 구조적으로 깔끔한 부분입니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/ba129847-02ab-42e5-9c8e-73ebe7861663/2303.06555v2.pdf)


## 2.3 모델 구조

### 2.3.1 1단계: 이미지·텍스트를 공통 latent space로 인코딩

UniDiffuser는 Latent Diffusion Models(LDM)의 성공을 그대로 가져와, **이미지와 텍스트를 모두 연속 latent로 변환한 뒤** diffusion을 latent 상에서 수행합니다. [arxiv](http://arxiv.org/pdf/2112.10752.pdf)

**이미지 인코더–디코더**

- Stable Diffusion의 VAE encoder $\(E_{\text{AE}}\)$ 로 재구성용 latent $\(x_0^{\text{AE}}\)$ 추출. [github](https://github.com/Stability-AI/stablediffusion)
- CLIP 이미지 인코더로 의미 latent $\(x_0^{\text{CLIP}}\)$ 추출. [openaccess.thecvf](https://openaccess.thecvf.com/content/CVPR2022/papers/Rombach_High-Resolution_Image_Synthesis_With_Latent_Diffusion_Models_CVPR_2022_paper.pdf)
- 최종 이미지 latent:

$$
  x_0 = [x_0^{\text{AE}}, x_0^{\text{CLIP}}].
  $$
  
  - AE 부분: 재구성 품질 담당.
  - CLIP 부분: 이미지–텍스트 의미 alignment에 도움 (이미지→텍스트, joint generation 성능 향상). [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/ba129847-02ab-42e5-9c8e-73ebe7861663/2303.06555v2.pdf)

**텍스트 인코더–디코더**

- CLIP 텍스트 인코더(77×768 벡터)를 통과시킨 후, 선형층으로 64차원으로 축소:

$$
  y_0 = \text{Linear}_\phi(\text{CLIP text}(T)). 
  $$

- 디코더는 GPT-2를 prefix-tuning 형태로 사용: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/ba129847-02ab-42e5-9c8e-73ebe7861663/2303.06555v2.pdf)
  - $\(y_0\)$ 를 prefix embedding으로 넣고, 토큰 시퀀스 $\(T\)$ 를 auto-regressive하게 복원:

$$
    \min_\varphi \mathbb{E}_T\Bigl[-\log p_\varphi(T\mid y_0)\Bigr]
    = \mathbb{E}_T\sum_{i=1}^N -\log p_\varphi(T_i\mid T_{1:i-1},y_0).
    $$

텍스트 재구성 BLEU-4 기준 0.894로, 64차원이라는 낮은 차원에서도 원문을 잘 복원함을 보입니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/ba129847-02ab-42e5-9c8e-73ebe7861663/2303.06555v2.pdf)


### 2.3.2 2단계: Transformer(U-ViT) 기반 공동 노이즈 예측기

latent $\((x_0,y_0)\)$ 에 대해, Section 2.2의 통합 loss (식 (4))로 joint noise predictor를 학습합니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/ba129847-02ab-42e5-9c8e-73ebe7861663/2303.06555v2.pdf)

- 백본: U-ViT – Vision Transformer 기반 diffusion 전용 아키텍처. [arxiv](http://arxiv.org/pdf/2212.09748v2.pdf)
- 설계 특징:
  - 이미지 latent patch, 텍스트 latent 토큰, 두 모달리티의 timestep $\((t_x,t_y)\)$ 를 모두 **토큰으로 처리**.
  - U자형 long skip connection으로 shallow–deep feature를 연결.
  - mixed precision 학습 안정성을 위해 pre-LN 대신 post-LN 및 skip 이후 별도 LayerNorm 사용. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/ba129847-02ab-42e5-9c8e-73ebe7861663/2303.06555v2.pdf)
- 모델 크기: 31-layer, hidden dim 1536, 24-head, 약 952M 파라미터. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/ba129847-02ab-42e5-9c8e-73ebe7861663/2303.06555v2.pdf)

학습 스케줄:

- 데이터: LAION-5B 서브셋 세 단계(laion2B-en, high-resolution, aesthetics 5+). [arxiv](http://arxiv.org/pdf/2112.10752.pdf)
- 해상도: 256→512 fine-tuning, 최종 512×512.  
- 최종 학습은 88×A100(80GB)에서 약 28일. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/ba129847-02ab-42e5-9c8e-73ebe7861663/2303.06555v2.pdf)

샘플링은 DPM-Solver/++ 등을 이용해 50 step 정도로 수행합니다. [arxiv](https://arxiv.org/abs/2206.08265)


## 2.4 성능: 개선점과 수치

### 2.4.1 텍스트→이미지: MS-COCO Zero-shot FID 비교

논문 Table 1의 요약: [arxiv](https://arxiv.org/abs/2205.11487)

| 모델 유형        | 모델              | MS-COCO Zero-shot FID↓ |
|------------------|-------------------|------------------------|
| bespoke T2I      | GLIDE             | 12.24 [proceedings.mlr](https://proceedings.mlr.press/v202/bao23a/bao23a.pdf)        |
|                  | Make-A-Scene      | 11.84 [proceedings.mlr](https://proceedings.mlr.press/v202/bao23a/bao23a.pdf)             |
|                  | DALL·E 2          | 10.39 [proceedings.mlr](https://proceedings.mlr.press/v202/bao23a/bao23a.pdf)             |
|                  | Stable Diffusion  | 8.59 [proceedings.mlr](https://proceedings.mlr.press/v202/bao23a/bao23a.pdf)    |
|                  | Imagen            | 7.27 [proceedings.mlr](https://proceedings.mlr.press/v202/bao23a/bao23a.pdf)    |
|                  | Parti             | 7.23 [proceedings.mlr](https://proceedings.mlr.press/v202/bao23a/bao23a.pdf)         |
| 범용 멀티모달    | Versatile Diffusion| 10.09 [proceedings.mlr](https://proceedings.mlr.press/v202/bao23a/bao23a.pdf)     |
|                  | **UniDiffuser**   | **9.71** [proceedings.mlr](https://proceedings.mlr.press/v202/bao23a/bao23a.pdf)          |

해석:

- UniDiffuser는 **멀티태스크 멀티모달** 모델임에도,
  - 범용 모델인 Versatile Diffusion보다 낮은 FID.
  - GLIDE, DALL·E 2보다도 좋은 FID이며, [semanticscholar](https://www.semanticscholar.org/paper/Photorealistic-Text-to-Image-Diffusion-Models-with-Saharia-Chan/9695824d7a01fad57ba9c01d7d76a519d78d65e7)
    Stable Diffusion/Imagen/Parti 등 최신 bespoke T2I와는 근접한 수준.

또한 CLIP score–FID trade-off 곡선에서도, 동일 guidance scale에서 Versatile Diffusion보다 항상 더 좋은 위치에 있습니다. [openaccess.thecvf](https://openaccess.thecvf.com/content/ICCV2023/papers/Xu_Versatile_Diffusion_Text_Images_and_Variations_All_in_One_Diffusion_ICCV_2023_paper.pdf)

### 2.4.2 이미지→텍스트 및 기타 태스크

- 이미지→텍스트에서 CLIP score 기준으로 **모든 guidance scale에서 Versatile Diffusion보다 우수**. [openaccess.thecvf](https://openaccess.thecvf.com/content/ICCV2023/papers/Xu_Versatile_Diffusion_Text_Images_and_Variations_All_in_One_Diffusion_ICCV_2023_paper.pdf)
- 이미지 단일 생성/텍스트 단일 생성, 이미지–텍스트 joint generation에서도 시각적으로 자연스러운 샘플을 제시. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/ba129847-02ab-42e5-9c8e-73ebe7861663/2303.06555v2.pdf)
- 이미지/텍스트 variation, 이미지–텍스트 블록드 기브스 체인, 이미지 인터폴레이션 등,  
  **모든 것이 하나의 joint 모델에서 파생**된다는 점이 구조적 강점. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/ba129847-02ab-42e5-9c8e-73ebe7861663/2303.06555v2.pdf)

### 2.4.3 효율성

Table 4 요약: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/ba129847-02ab-42e5-9c8e-73ebe7861663/2303.06555v2.pdf)

- 파라미터:
  - Stable Diffusion: 860M
  - Versatile Diffusion: 2.57B
  - UniDiffuser: 952M
- 10개 샘플(25 step) 생성 시간 (A100 80GB):
  - Stable Diffusion: 25.43s
  - Versatile Diffusion: 23.89s
  - **UniDiffuser: 19.77s**
- 메모리 사용:
  - Stable Diffusion: 67.83GB
  - Versatile Diffusion: 76.53GB
  - **UniDiffuser: 48.30GB**

즉, **범용 모델 중에서는 훨씬 가볍고 빠르며**, bespoke Stable Diffusion 대비 10% 남짓한 파라미터 증가로 여러 태스크를 동시에 처리합니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/ba129847-02ab-42e5-9c8e-73ebe7861663/2303.06555v2.pdf)


## 2.5 한계 및 단점

논문과 최근 문헌을 종합한 한계는 다음과 같습니다. [openaccess.thecvf](https://openaccess.thecvf.com/content/CVPR2025/papers/Li_OmniFlow_Any-to-Any_Generation_with_Multi-Modal_Rectified_Flows_CVPR_2025_paper.pdf)

1. **텍스트 생성 품질 제약**  
   - 텍스트 인코더/디코더는 비교적 작은 GPT-2 기반이고, 학습 데이터인 웹 캡션 품질이 낮아,  
     “텍스트 자체”의 유창성은 최신 LLM 기반 captioner에 비해 부족함을 저자들도 인정. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/ba129847-02ab-42e5-9c8e-73ebe7861663/2303.06555v2.pdf)

2. **모달리티 확장 실험 부재**  
   - 이론상 오디오, 비디오 등으로 확장 가능하지만, 실제 구현/실험은 이미지–텍스트에 한정. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/ba129847-02ab-42e5-9c8e-73ebe7861663/2303.06555v2.pdf)
   - 이후 OmniFlow 등에서 텍스트–이미지–오디오로 확장된 any-to-any 모델이 등장하며 후속 방향성을 보여줌. [openaccess.thecvf](https://openaccess.thecvf.com/content/CVPR2025/papers/Li_OmniFlow_Any-to-Any_Generation_with_Multi-Modal_Rectified_Flows_CVPR_2025_paper.pdf)

3. **훈련 비용**  
   - 88×A100, 수주 단위 학습은 연구 조직/산업에서나 가능한 스케일로,  
     실무에서 그대로 모사하기는 어렵다는 현실적 제약이 큼. [arxiv](https://arxiv.org/abs/2307.01952)

4. **표현력 vs 안정성의 trade-off**  
   - 하나의 네트워크가 모든 분포를 담당하기 때문에,  
     특정 태스크(예: 고품질 텍스트 생성)에만 완전히 최적화된 bespoke 모델보다 품질이 약간 떨어질 수 있음.  
   - OmniFlow, CoDi 등은 모듈화를 통해 “범용성 vs 최적화” 사이의 새로운 균형점을 제시하며 발전 중. [openaccess.thecvf](https://openaccess.thecvf.com/content/CVPR2025/papers/Li_OmniFlow_Any-to-Any_Generation_with_Multi-Modal_Rectified_Flows_CVPR_2025_paper.pdf)


***

# 3. 모델의 일반화 성능 향상 가능성

UniDiffuser가 “일반화”를 어떻게 돕는지, **세 가지 관점**에서 볼 수 있습니다.

## 3.1 분포–태스크 간 파라미터 공유에 따른 통계적 효율성

통합 loss (식 (4))는, 각 미니배치가 동시에 다음을 포함하게 합니다.

- 이미지 주변 분포 학습 신호 $(\(t_y=T\))$  
- 텍스트 주변 분포 학습 신호 $(\(t_x=T\))$ 
- 텍스트→이미지 조건부 $(\(t_y=0\))$  
- 이미지→텍스트 조건부 $(\(t_x=0\))$  
- joint 분포 $(\(t_x=t_y\))$

즉, 동일 파라미터 $\(\theta\)$ 가 **여러 서로 다른 주변/조건부/결합 분포에서 오는 gradient를 동시에** 받습니다. [proceedings.mlr](https://proceedings.mlr.press/v202/bao23a/bao23a.pdf)

이것은 고전적인 multi-task learning에서 보는 것처럼:

- 각 태스크의 데이터가 다른 태스크에 **레귤러라이저**로 작용해 overfitting을 줄이고,  
- 특정 태스크에서 부족한 모드(예: 드문 문장–이미지 조합)를 다른 태스크의 신호가 보완하는 효과를 가집니다.

실제로 Versatile Diffusion과의 직접 비교에서:

- 동일 guidance scale에서 FID, CLIP score가 **항상 더 우수**하고, [openaccess.thecvf](https://openaccess.thecvf.com/content/ICCV2023/papers/Xu_Versatile_Diffusion_Text_Images_and_Variations_All_in_One_Diffusion_ICCV_2023_paper.pdf)
- 텍스트→이미지, 이미지→텍스트 모두에서 격차가 나타난다는 점은  
  “동일 capacity에서 통합 학습이 통계적으로 더 효율적”이라는 주장의 실증적 근거로 볼 수 있습니다.

## 3.2 “타임스텝 조건”을 이용한 연속적인 태스크 스펙트럼

UniDiffuser는 $\((t_x,t_y)\)$ 를 연속적인 조건으로 사용하므로,

- 깨끗한 텍스트–노이즈 이미지 $(\(t_y=0,t_x>0\))$ 에서 텍스트→이미지,
- 노이즈 텍스트–깨끗한 이미지 $(\(t_x=0,t_y>0\))$ 에서 이미지→텍스트,
- 중간 단계 $(\(t_y\in(0,T)\))$ 에서는 **노이즈 텍스트를 조건으로 한 이미지 생성** 같은 중간 형태의 태스크까지 포함합니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/ba129847-02ab-42e5-9c8e-73ebe7861663/2303.06555v2.pdf)

즉, 모델은 실제로는

$$
q(x_0\mid y_{t_y})\ (0\le t_y\le T),\quad
q(y_0\mid x_{t_x})\ (0\le t_x\le T)
$$

와 같이 **연속적인 조건 강도 스펙트럼을 커버**합니다.

이는 다음과 같은 일반화 가능성을 시사합니다.

- 입력 모달리티가 **부분적으로 손상되거나, 노이즈가 있거나, 도메인이 살짝 달라져도**,  
  해당 상황은 어떤 $\(t_x,t_y\)$ 조합과 유사한 분포로 매핑 가능.
- 실제로 논문에서는 이미지/텍스트 variation, 블록 기브스 체인, 와일드 이미지 인터폴레이션 등 **훈련 시 명시적으로 정의하지 않은 조합**도 자연스럽게 수행함을 보입니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/ba129847-02ab-42e5-9c8e-73ebe7861663/2303.06555v2.pdf)

이는 **diffusion SDE 기반 score-model이 다양한 조건 레벨에 걸친 score를 학습하면, 새로운 조건 조합에도 비교적 robust하게 일반화할 수 있다**는 Score-SDE 계열 연구의 통찰과도 상통합니다. [arxiv](https://arxiv.org/pdf/2402.07487.pdf)


## 3.3 최근 후속 연구에서의 활용과 확장

UniDiffuser의 아이디어(“하나의 Transformer/flow로 멀티모달 분포를 통합”)는 이후 논문들에서 여러 방향으로 확장되었습니다.

- **UniD3 / Unified Discrete Diffusion** [huggingface](https://huggingface.co/papers/2211.14842)
  - 이미지 VQ 코드 + 텍스트 토큰을 하나의 discrete 시퀀스로 보고, **단일 discrete diffusion**과 mutual attention으로 양 모달을 동시에 생성·변환.  
  - UniDiffuser가 continuous latent–노이즈 예측 관점에서 한 작업을, **discrete token 공간**으로 옮긴 셈.  
  - 다양한 vision–language 태스크에서 state-of-the-art급 성능을 보이며, discrete token 기반 모델에서도 통합 학습이 유효함을 입증.

- **OmniFlow (Any-to-Any Multi-modal Rectified Flows)** [arxiv](https://arxiv.org/pdf/2506.07903.pdf)
  - 텍스트–이미지–오디오 등 3모달 이상에서 **“any-to-any generation”**을 목표로 하는 rectified flow 기반 모델.  
  - Stable Diffusion 3의 MMDiT 아키텍처를 확장해 multi-modal Transformer backbone을 구축하고, UniDiffuser와 유사하게 **하나의 네트워크**로 다양한 분포 전이를 학습.  
  - 텍스트→이미지, 오디오→이미지 등 다수 태스크에서 UniDiffuser나 CoDi 같은 선행 범용 모델보다 CLIP score 등에서 우수한 성능을 보고. [openaccess.thecvf](https://openaccess.thecvf.com/content/CVPR2025/papers/Li_OmniFlow_Any-to-Any_Generation_with_Multi-Modal_Rectified_Flows_CVPR_2025_paper.pdf)

이들 결과는, UniDiffuser 스타일의 **“단일 백본–다수 분포/태스크”** 설계가 스케일이 커질수록 더 강력해질 수 있음을 시사합니다.


***

# 4. 2020년 이후 관련 최신 연구 비교 분석

여기서는 UniDiffuser를, 2020년 이후 등장한 관련 연구들과 세 축에서 비교합니다.

1. **기초 diffusion/score-based 이론**
2. **단일 태스크(주로 텍스트→이미지) 대형 모델**
3. **통합 멀티태스크/멀티모달 생성 모델**

## 4.1 기초 이론: Score-SDE와 Latent Diffusion

- **Score-Based Generative Modeling via SDEs** (Song et al., 2020) [arxiv](https://arxiv.org/abs/2011.13456)
  - 연속 시간 SDE로 diffusion을 정식화하고, 역 SDE/ODE를 통해 샘플링 및 정확한 likelihood 계산 가능함을 보인 foundational work.  
  - UniDiffuser 역시 **노이즈 예측 네트워크**와 **SDE 기반 샘플링(예: DPM-Solver)**에 의존한다는 점에서 이 계열 위에 서 있음. [arxiv](https://arxiv.org/abs/2206.08265)

- **Latent Diffusion Models / Stable Diffusion** (Rombach et al., 2021–22) [arxiv](https://arxiv.org/abs/2112.10752)
  - 고해상도 이미지 생성을 위해, VAE latent 공간에서 diffusion을 수행하는 LDM 제안.  
  - CLIP 텍스트 인코더와 cross-attention으로 텍스트 조건을 처리.  
  - UniDiffuser는 이 LDM/Stable Diffusion의 VAE·CLIP 스택을 그대로 활용하여, **멀티모달 통합은 diffusion 백본에서만 수행**하는 2-stage 설계를 채택. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/ba129847-02ab-42e5-9c8e-73ebe7861663/2303.06555v2.pdf)

정리하면, UniDiffuser는 **Score-SDE 이론 + Latent Diffusion 실용 설계** 위에, **분포 통합과 멀티모달 통합**이라는 층을 올린 모델이라 볼 수 있습니다.


## 4.2 단일 태스크(텍스트→이미지) 초대형 모델과의 비교

대표적인 모델과 UniDiffuser의 비교 축은 다음과 같습니다. [arxiv](https://arxiv.org/abs/2205.11487)

| 축               | GLIDE [proceedings.mlr](https://proceedings.mlr.press/v162/nichol22a.html) | Imagen [arxiv](https://arxiv.org/abs/2205.11487) | Stable Diffusion/SDXL [arxiv](http://arxiv.org/pdf/2112.10752.pdf) | UniDiffuser [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/ba129847-02ab-42e5-9c8e-73ebe7861663/2303.06555v2.pdf) |
|------------------|-----------------|-----------------------|-----------------------------------------|--------------------|
| 핵심 태스크      | T2I             | T2I                   | T2I (+img2img 등)                      | T2I 포함 멀티태스크 |
| 데이터 도메인    | 이미지–텍스트   | 이미지–텍스트         | 이미지–텍스트 (LAION 등)              | 동일               |
| 아키텍처         | U-Net + 텍스트 인코더 | cascade diff + 대형 LM 텍스트 인코더 | LDM (U-Net) + CLIP | LDM + Transformer(U-ViT) joint noise |
| 분포 관점        | $\(q(x_0\mid y_0)\)$ | $\(q(x_0\mid y_0)\)$    | $\(q(x_0\mid y_0)\)$ (부분적 주변)       | $\(q(x_0),q(y_0),q(x_0\mid y_0),q(y_0\mid x_0),q(x_0,y_0)\)$ |
| 성능 (COCO FID) | 12.24           | 7.27                  | 8.59 / SDXL 더 향상 [semanticscholar](https://www.semanticscholar.org/paper/d7890d1906d95c4ae4c430b350455156d6d8aed9)                | 9.71               |
| 범용성           | 한 태스크 중심   | 한 태스크 중심        | 이미지 도메인 내 여러 태스크          | 멀티모달/멀티태스크 |

**핵심 차이**는:

- GLIDE/Imagen/SD/SDXL은 **텍스트→이미지 품질 극대화**에 집중한 bespoke 설계.  
- UniDiffuser는 **품질을 약간 희생하는 대신**,
  - 텍스트→이미지와 유사한 품질을 유지하면서,
  - 멀티모달 분포 전체를 다루는 일반성을 확보.

따라서 **“텍스트→이미지만 죽어라 쓰겠다”**는 응용이면 Imagen·SDXL 계열이 여전히 우세하고,  
**“이미지–텍스트 쌍과 그 주변·조건부를 모두 통합적으로 다루고 싶다”**면 UniDiffuser 스타일이 유리합니다.


## 4.3 통합 멀티태스크/멀티모달 생성 모델 비교

### 4.3.1 Versatile Diffusion (VD) [semanticscholar](https://www.semanticscholar.org/paper/Versatile-Diffusion:-Text,-Images-and-Variations-in-Xu-Wang/97029b53d0252ea68472423dea33e5aa2316926d)

- 멀티플로우 구조: text→image, image→text, image variation, text variation 등 **여러 플로우를 구성**하고,  
  각 플로우에 대한 loss를 동시에 학습.
- 장점:
  - Stable Diffusion 계열과 호환되는 모듈식 설계.
  - 다양한 조합 태스크를 비교적 직관적으로 구성.
- 단점 (UniDiffuser 대비): [openaccess.thecvf](https://openaccess.thecvf.com/content/ICCV2023/papers/Xu_Versatile_Diffusion_Text_Images_and_Variations_All_in_One_Diffusion_ICCV_2023_paper.pdf)
  - 태스크별 loss, gradient multiplier tuning 등 **훈련 레시피가 복잡**.
  - joint generation(특히 이미지–텍스트 동시 샘플링)에 대한 이론적 정합성이 약하고,  
    성능도 UniDiffuser에 비해 떨어짐 (FID/CLIP 기준).

UniDiffuser는 같은 범주의 문제를 **“timestep 조건화”라는 하나의 축으로 흡수**하여,  
단일 MSE loss로 훨씬 깔끔한 이론·구현을 제공하는 점에서 차별화됩니다.


### 4.3.2 Unified Discrete Diffusion / UniD3 [mhh0318.github](https://mhh0318.github.io/unid3/)

- 이미지 VQ 코드와 텍스트 토큰, [MASK] 상태를 포함한 discrete token 시퀀스에 대해,  
  **공통 transition matrix를 사용하는 discrete diffusion**으로 vision–language joint generation을 수행.
- mutual attention 모듈과 통합 objective로, 텍스트 기반·이미지 기반·동시 생성(all-in-one)을 달성.
- UniDiffuser와 비교:
  - UniDiffuser: continuous latent + 노이즈 예측 (DDPM 스타일).  
  - UniD3: discrete token space + 전이 행렬 기반 discrete diffusion.
  - 두 모델 모두 **“하나의 확률 과정으로 멀티모달 통합”**이라는 공통 철학을 가짐.
  - discrete 방식은 텍스트와 이미지 token space를 단일 vocabulary로 unify하기 쉬운 반면,  
    continuous latent는 고해상도·연속적 visual detail 처리에 강점.

### 4.3.3 OmniFlow 및 최근 any-to-any 모델 [arxiv](https://arxiv.org/pdf/2506.07903.pdf)

- OmniFlow: [openaccess.thecvf](https://openaccess.thecvf.com/content/CVPR2025/papers/Li_OmniFlow_Any-to-Any_Generation_with_Multi-Modal_Rectified_Flows_CVPR_2025_paper.pdf)
  - Stable Diffusion 3의 MMDiT 아키텍처를 기반으로,  
    텍스트–이미지–오디오 간 **any-to-any rectified flow**를 학습.
  - Multi-modal guidance와 modular 설계를 통해, 텍스트→이미지, 이미지→오디오 등 다양한 태스크에서  
    UniDiffuser·CoDi 등보다 CLIP/CLAP score 등에서 우수.
  - UniDiffuser와 달리, **사전 학습된 모듈(예: 텍스트 인코더, 이미지 디코더)을 재활용**하는 설계로,  
    학습 비용을 줄이면서도 범용성을 확보.

- 기타: CoDi, Chameleon, Transfusion 등 [emergentmind](https://www.emergentmind.com/topics/diffusion-vision-language-models-dvlms)
  - autoregressive/diffusion 혼합 아키텍처, 대형 autoregressive backbone을 활용한 멀티모달 통합 등 다양한 설계가 시도 중.

UniDiffuser는 이들보다 **연대기상 이른 “1세대” 통합 멀티모달 diffusion**으로,  
후속 모델들이 rectified flow, discrete diffusion, autoregressive–diffusion 하이브리드 등으로 확장하는 **개념적 출발점 중 하나**로 자리잡았습니다. [arxiv](https://arxiv.org/pdf/2506.07903.pdf)


***

# 5. 앞으로의 연구에 미치는 영향과 연구 시 고려할 점

## 5.1 향후 연구에 미치는 영향

1. **“범용 멀티모달 분포 모델” 패러다임의 개척**

   - UniDiffuser는 “이미지–텍스트 쌍에서 정의되는 모든 분포를 하나의 Transformer로 근사”라는  
     비교적 급진적인 목표를 실제로 구현하고, SOTA에 근접한 성능을 달성했다는 점에서 의미가 큽니다. [proceedings.mlr](https://proceedings.mlr.press/v202/bao23a/bao23a.pdf)
   - 이후 UniD3, OmniFlow, Unified Multimodal Discrete Diffusion 등은 모두  
     **“하나의 generative backbone으로 여러 modality·태스크를 동시에 다룬다”**는 방향을 따르고 있습니다. [arxiv](https://arxiv.org/html/2503.20853v1)

2. **Diffusion Transformer(DiT) 및 scaling law 연구와의 결합**

   - DiT는 Transformer 기반 diffusion 백본이 FLOPs 증가에 따라 FID가 monotonically 감소하는, [arxiv](http://arxiv.org/pdf/2212.09748v2.pdf)
     “Transformer형 diffusion의 scaling law”를 보고했습니다. [arxiv](http://arxiv.org/pdf/2410.08184.pdf)
   - UniDiffuser는 **Transformer 기반 joint noise predictor**를 대규모 데이터에 적용한 초기 사례로,  
     이후 DiffScaler, SDXL, OmniFlow 등에서 Transformer diffusion을 더 크게 스케일링하는 작업으로 이어졌습니다. [arxiv](https://arxiv.org/html/2404.09976)
   - 결과적으로, “하나의 거대한 diffusion transformer + 다양한 조건/모달리티”라는 그림이  
     멀티모달 foundation model의 한 축으로 자리 잡았습니다.

3. **조합적 태스크/조건에 대한 자연스러운 지원**

   - blocking Gibbs, image–text 인터폴레이션, cross-modal variation은  
     기존 모델에서는 별도 설계가 필요했지만, UniDiffuser에서는 **joint 분포 + 조건부 분포를 공유**하기 때문에  
     자연스럽게 파생되는 응용이 되었습니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/ba129847-02ab-42e5-9c8e-73ebe7861663/2303.06555v2.pdf)
   - 이는 향후 “멀티모달 reasoning”이나 “vision–language–action planning” 등에서도  
     **joint generative model 기반 MCMC, planning**을 사용하는 연구의 가능성을 열어 줍니다. [arxiv](https://arxiv.org/html/2508.20072v1)


## 5.2 앞으로 연구 시 고려할 점 (연구 전략 관점)

1. **어디까지 “하나의 모델”로 통합할 것인가?**

   - UniDiffuser는 이미지–텍스트 2모달리티, 여러 분포를 하나의 모델로 통합했지만,  
     OmniFlow처럼 오디오까지 포함한 3모달, 나아가 action(로봇 제어 시퀀스)까지 포함하면  
     representation 공유가 과도해져 **모달리티별 특화 성능이 떨어질 수 있는 위험**이 있습니다. [arxiv](https://arxiv.org/html/2508.20072v1)
   - 실용적 설계에서는:
     - encoder/decoder는 모달리티별로 특화,
     - diffusion backbone만 공유,
     - 일부 레이어만 cross-modal 공유하는 MoE/adapter 구조 [arxiv](https://arxiv.org/abs/2412.12953)
     등의 하이브리드 구조가 바람직할 수 있습니다.

2. **Continuous vs Discrete diffusion 선택**

   - UniDiffuser(continuous latent) vs UniD3(두 모달의 discrete token) vs CoM-DAD 같은 혼합 discrete–continuous 프레임워크. [arxiv](https://arxiv.org/html/2601.04056v1)
   - 연구자가 초점을 두는 응용에 따라:
     - 고해상도 이미지·비디오: continuous latent diffusion 선호.  
     - 텍스트 중심, 토큰 레벨 alignment: discrete diffusion 또는 hybrid.  
   - UniDiffuser의 아이디어(모달리티별 timestep 통합)는 discrete setting으로도 옮길 수 있음을 UniD3 계열이 보여주었으므로,  
     **개념과 구현 레벨을 분리하여 설계**하는 것이 중요합니다.

3. **일반화 성능 평가 지표의 확장**

   - 지금까지는 MS-COCO FID, CLIP score 정도로 평가하지만,  
     범용 모델의 진정한 강점은 **도메인 전이, 태스크 전이, 노이즈/결손 조건에서의 robust한 생성**에 있습니다.  
   - 향후 연구에서는:
     - OOD caption/이미지에 대한 생성 품질,  
     - 부분 관측(노이즈 텍스트/이미지)에서의 복원 성능,  
     - 다운스트림 태스크(예: VQA, captioning, detection) 성능 향상 여부,
     - semi-supervised setting에서의 label 효율성  
     등을 함께 보는 것이 UniDiffuser류 모델의 가치를 제대로 측정하는 방향입니다.

4. **데이터·윤리·안전성**

   - LAION-5B 기반 대규모 멀티모달 모델은, deepfake·저작권·편향 등의 사회적 이슈를 동반합니다. [biorxiv](https://www.biorxiv.org/content/10.1101/2022.11.18.517004v2.full)
   - UniDiffuser 저자들도 watermarking, 책임 있는 공개 프로토콜의 필요성을 언급하며,  
     이후 Stable Diffusion/SDXL, Imagen 계열에서도 RLHF, 콘텐츠 필터, watermarking 등 안전 장치가 연구되고 있습니다. [semanticscholar](https://www.semanticscholar.org/paper/d7890d1906d95c4ae4c430b350455156d6d8aed9)
   - 후속 연구에서는 **생성 품질과 동시에 검출/추적 가능성, 편향 완화**를 함께 설계하는 것이 필수입니다.

5. **이론적 해석과 샘플링 효율 개선의 결합**

   - Analytic-DPM, DPM-Solver/++, ERA-Solver 등은 [ml.cs.tsinghua.edu](https://ml.cs.tsinghua.edu.cn/~jun/pub/analytic-dpm.pdf)
     diffusion 역과정의 **최적 분산·고차 ODE solver** 등을 분석해,  
     10–20 step 내 고품질 샘플링을 가능하게 하고 있습니다.
   - UniDiffuser류의 **대형 멀티모달 모델**은 한 샘플당 계산비용이 크므로,  
     이런 이론적 sampler와 결합해 **few-step generation**을 달성하는 것이 실제 응용에서 매우 중요합니다.

***

# 6. 정리

- 이 논문은 **“모든 관련 분포를 하나의 Transformer diffusion으로 통합한다”**는 매우 명확한 메시지를 갖고 있으며,  
  이를 **노이즈 예측–timestep 조건화–멀티모달 latent 통합**이라는 간단하지만 강력한 프레임으로 구현했습니다. [proceedings.mlr](https://proceedings.mlr.press/v202/bao23a/bao23a.pdf)

- 실험적으로,  
  - 범용 모델인 Versatile Diffusion을 전반적으로 상회하고, [openaccess.thecvf](https://openaccess.thecvf.com/content/ICCV2023/papers/Xu_Versatile_Diffusion_Text_Images_and_Variations_All_in_One_Diffusion_ICCV_2023_paper.pdf)
  - bespoke 텍스트→이미지 모델에 근접한 성능을 유지하면서  
    이미지·텍스트·joint generation·variation·Gibbs 체인 등 다양한 태스크를 하나의 모델로 수행합니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/ba129847-02ab-42e5-9c8e-73ebe7861663/2303.06555v2.pdf)

- 일반화 측면에서,  
  - 여러 분포·태스크 간 파라미터 공유,  
  - timestep을 통한 연속적 조건 레벨 학습,  
  - joint/conditional/marginal 관계의 명시적 학습  
  이 결합되어, 멀티모달 도메인에서 **견고한 representation과 조건부 생성 능력**을 키우는 방향으로 작용합니다.

- 이후 UniD3, OmniFlow, SDXL, UniVG 등은 이 아이디어를 discrete diffusion, rectified flow, 더 큰 DiT backbone, 더 많은 모달리티로 확장하고 있으며, [arxiv](https://arxiv.org/html/2503.12652v1)
  UniDiffuser는 이러한 “범용 멀티모달 diffusion foundation model” 계열의 **기본 레퍼런스**로 자리 잡았다고 평가할 수 있습니다.

연구자로서 이 논문을 기반으로 작업할 때는,  
(1) 어떤 분포/태스크를 한 모델에 통합할지의 범위 설계,  
(2) continuous vs discrete diffusion 선택,  
(3) evaluation protocol 확장,  
(4) 효율·안전·윤리 이슈를 포함한 전체 시스템 관점,  
을 함께 고려하는 것이 중요합니다.

<span style="display:none">[^1_46][^1_47][^1_48][^1_49][^1_50][^1_51][^1_52][^1_53][^1_54][^1_55][^1_56][^1_57][^1_58][^1_59][^1_60][^1_61][^1_62][^1_63][^1_64][^1_65][^1_66][^1_67][^1_68][^1_69][^1_70][^1_71][^1_72][^1_73][^1_74][^1_75][^1_76][^1_77][^1_78][^1_79][^1_80][^1_81][^1_82][^1_83][^1_84][^1_85][^1_86]</span>

<div align="center">⁂</div>

[^1_1]: 2303.06555v2.pdf

[^1_2]: https://proceedings.mlr.press/v202/bao23a/bao23a.pdf

[^1_3]: https://openaccess.thecvf.com/content/ICCV2023/papers/Xu_Versatile_Diffusion_Text_Images_and_Variations_All_in_One_Diffusion_ICCV_2023_paper.pdf

[^1_4]: https://arxiv.org/pdf/2211.08332.pdf

[^1_5]: https://proceedings.mlr.press/v162/nichol22a.html

[^1_6]: https://arxiv.org/abs/2205.11487

[^1_7]: https://openaccess.thecvf.com/content/CVPR2022/papers/Rombach_High-Resolution_Image_Synthesis_With_Latent_Diffusion_Models_CVPR_2022_paper.pdf

[^1_8]: https://arxiv.org/abs/2112.10752

[^1_9]: https://arxiv.org/abs/2211.14842

[^1_10]: http://arxiv.org/abs/2011.13456

[^1_11]: https://arxiv.org/pdf/2011.13456.pdf

[^1_12]: http://arxiv.org/pdf/2112.10752.pdf

[^1_13]: https://github.com/Stability-AI/stablediffusion

[^1_14]: http://arxiv.org/pdf/2212.09748v2.pdf

[^1_15]: https://arxiv.org/abs/2206.08265

[^1_16]: https://link.springer.com/10.1007/s11633-025-1562-4

[^1_17]: http://arxiv.org/pdf/2205.11487.pdf

[^1_18]: https://www.semanticscholar.org/paper/Photorealistic-Text-to-Image-Diffusion-Models-with-Saharia-Chan/9695824d7a01fad57ba9c01d7d76a519d78d65e7

[^1_19]: https://openaccess.thecvf.com/content/CVPR2025/papers/Li_OmniFlow_Any-to-Any_Generation_with_Multi-Modal_Rectified_Flows_CVPR_2025_paper.pdf

[^1_20]: https://arxiv.org/abs/2307.01952

[^1_21]: https://arxiv.org/pdf/2402.07487.pdf

[^1_22]: https://huggingface.co/papers/2211.14842

[^1_23]: https://mhh0318.github.io/unid3/

[^1_24]: https://arxiv.org/pdf/2506.07903.pdf

[^1_25]: https://arxiv.org/abs/2011.13456

[^1_26]: https://proceedings.mlr.press/v162/nichol22a/nichol22a.pdf

[^1_27]: https://papers.neurips.cc/paper_files/paper/2022/file/ec795aeadae0b7d230fa35cbaf04c041-Paper-Conference.pdf

[^1_28]: https://www.semanticscholar.org/paper/d7890d1906d95c4ae4c430b350455156d6d8aed9

[^1_29]: https://www.semanticscholar.org/paper/Versatile-Diffusion:-Text,-Images-and-Variations-in-Xu-Wang/97029b53d0252ea68472423dea33e5aa2316926d

[^1_30]: https://www.emergentmind.com/topics/diffusion-vision-language-models-dvlms

[^1_31]: https://arxiv.org/html/2503.20853v1

[^1_32]: http://arxiv.org/pdf/2410.08184.pdf

[^1_33]: https://arxiv.org/html/2404.09976

[^1_34]: https://arxiv.org/html/2508.20072v1

[^1_35]: https://arxiv.org/abs/2412.12953

[^1_36]: https://arxiv.org/html/2601.04056v1

[^1_37]: https://www.biorxiv.org/content/10.1101/2022.11.18.517004v2.full

[^1_38]: https://arxiv.org/html/2406.14847v1

[^1_39]: https://arxiv.org/pdf/2412.09656.pdf

[^1_40]: https://ml.cs.tsinghua.edu.cn/~jun/pub/analytic-dpm.pdf

[^1_41]: https://arxiv.org/abs/2201.06503

[^1_42]: https://arxiv.org/html/2301.12935v4

[^1_43]: https://www.semanticscholar.org/paper/Analytic-DPM:-an-Analytic-Estimate-of-the-Optimal-Bao-Li/9b7b218b0f4e14f97260b6192add37da5e9ae2c5

[^1_44]: https://transferlab.ai/pills/2022/analytic-dpm/

[^1_45]: https://arxiv.org/html/2503.12652v1

[^1_46]: https://www.semanticscholar.org/paper/93667328d7f9998523f4a0b01ab6f72dea947612

[^1_47]: https://www.semanticscholar.org/paper/fad8bd00bca79005f89a0b0e2aa13fddc864fe22

[^1_48]: https://www.semanticscholar.org/paper/fc719a3a32d328dcb79afe3eef7df28db3e7f480

[^1_49]: https://www.semanticscholar.org/paper/f671a09e3e5922e6d38cb77dda8d76d5ceac2a27

[^1_50]: https://www.semanticscholar.org/paper/baaf5d042fbb4287991efd858ce70081b044a49d

[^1_51]: https://www.semanticscholar.org/paper/2e32cde6e080f990873638f2e113767a6a19c824

[^1_52]: https://www.semanticscholar.org/paper/d49a230b7718bd82fd7816d9d78e3ebd49118d2a

[^1_53]: https://www.semanticscholar.org/paper/509e166d5e66df10675a0e15063daad518dcc5ad

[^1_54]: https://www.semanticscholar.org/paper/9d6e6488bb3d1ecd55e8a50b78c2e4cbedf2f437

[^1_55]: https://link.springer.com/10.1007/s10687-022-00460-8

[^1_56]: https://arxiv.org/pdf/2202.02514.pdf

[^1_57]: http://arxiv.org/pdf/2404.12814.pdf

[^1_58]: https://arxiv.org/pdf/2106.01357.pdf

[^1_59]: https://arxiv.org/abs/2406.04916

[^1_60]: http://arxiv.org/pdf/2412.19114.pdf

[^1_61]: http://arxiv.org/pdf/2208.05003v1.pdf

[^1_62]: https://arxiv.org/html/2408.16626v1

[^1_63]: https://arxiv.org/pdf/2311.11003.pdf

[^1_64]: https://www.pure.ed.ac.uk/ws/portalfiles/portal/237685508/Maximum_Likelihood_SONG_DOA28092021_AFV.pdf

[^1_65]: https://www.youtube.com/watch?v=L9ZegT87QK8

[^1_66]: https://papers.nips.cc/paper/2021/file/0a9fdbb17feb6ccb7ec405cfb85222c4-Paper.pdf

[^1_67]: https://openreview.net/references/pdf?id=9f-ZDS-x9w

[^1_68]: https://papers.nips.cc/paper_files/paper/2021/file/0a9fdbb17feb6ccb7ec405cfb85222c4-Paper.pdf

[^1_69]: https://iclr.cc/virtual/2022/awards_detail

[^1_70]: https://openreview.net/forum?id=PxTIG12RRHS

[^1_71]: https://openreview.net/pdf/ef0eadbe07115b0853e964f17aa09d811cd490f1.pdf

[^1_72]: https://proceedings.mlr.press/v162/lu22f/lu22f.pdf

[^1_73]: https://openreview.net/forum?id=0xiJLKH-ufZ

[^1_74]: https://arxiv.org/pdf/2101.09258.pdf

[^1_75]: https://arxiv.org/html/2406.10808v4

[^1_76]: https://arxiv.org/pdf/2304.13224.pdf

[^1_77]: https://www.semanticscholar.org/paper/Maximum-Likelihood-Training-of-Score-Based-Models-Song-Durkan/9cf6f42806a35fd1d410dbc34d8e8df73a29d094

[^1_78]: https://www.semanticscholar.org/paper/Score-Based-Generative-Modeling-through-Stochastic-Song-Sohl-Dickstein/633e2fbfc0b21e959a244100937c5853afca4853

[^1_79]: https://arxiv.org/html/2508.03636v1

[^1_80]: https://www.semanticscholar.org/paper/On-Maximum-Likelihood-Training-of-Score-Based-Durkan-Song/396acfada641766c18be4c5585ff6f3a7d37272d

[^1_81]: https://arxiv.org/abs/2101.09258

[^1_82]: https://arxiv.org/pdf/2501.08180.pdf

[^1_83]: http://www.arxiv.org/abs/2101.09258

[^1_84]: https://github.com/baofff/Analytic-DPM

[^1_85]: https://arxiv.org/abs/2303.06555

[^1_86]: https://arxiv.org/pdf/2303.06555.pdf
