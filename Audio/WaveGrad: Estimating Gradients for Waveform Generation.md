# WaveGrad: Estimating Gradients for Waveform Generation

"WaveGrad: Estimating Gradients for Waveform Generation"은 음성 합성과 같은 파형 생성 분야에서 확산 확률 모델(diffusion probabilistic models)과 점수 매칭(score matching) 기법을 성공적으로 적용한 선구적인 연구 중 하나입니다. 이 논문은 기존의 자기회귀(autoregressive) 모델이 가진 고품질의 장점과 비자기회귀(non-autoregressive) 모델의 빠른 생성 속도의 장점을 결합하여, 생성 오디오의 품질과 속도 사이의 균형을 맞추는 새로운 접근법을 제시했습니다.

#### 핵심 주장 및 주요 기여

WaveGrad의 핵심 주장은 **데이터 밀도의 기울기(gradient)를 점진적으로 추정**하여, 가우시안 백색 잡음(Gaussian white noise)으로부터 시작해 목표 파형을 반복적으로 정제(refine)해 나갈 수 있다는 것입니다. 이는 기존의 생성 모델들이 데이터의 분포 자체를 직접 모델링하려 했던 것과는 다른 접근 방식입니다.[1]

주요 기여는 다음과 같이 요약할 수 있습니다.

*   **새로운 파형 생성 패러다임 제시:** 점수 매칭과 확산 확률 모델을 조건부 음성 합성에 적용하여, 빠르면서도 고품질의 오디오를 생성하는 새로운 길을 열었습니다.[1]
*   **속도와 품질의 절충:** 단 몇 번의 반복(iteration)만으로도 고품질 오디오를 생성할 수 있어, 추론 속도와 샘플 품질 사이의 유연한 절충이 가능함을 보여주었습니다. 실제로 6번의 반복만으로도 뛰어난 품질의 오디오 생성이 가능했습니다.[1]
*   **뛰어난 성능 입증:** 기존의 비자기회귀 모델들의 성능을 뛰어넘고, 최첨단 자기회귀 모델과 비견될 만한 주관적 자연성(subjective naturalness)을 달성했습니다.[1]
*   **모델 일반화 성능 향상:** 연속적인(continuous) 노이즈 레벨에 모델을 조건화함으로써, 한 번의 학습으로 다양한 추론 횟수에 대응할 수 있는 일반화 성능이 뛰어난 모델을 제안했습니다.[1]

***

### 상세 설명

#### 1. 해결하고자 하는 문제

기존의 음성 생성 모델들은 크게 두 가지로 나뉩니다.

*   **자기회귀 모델 (Autoregressive Models):** WaveNet과 같은 모델들은 오디오 샘플을 순차적으로 하나씩 생성합니다. 이 방식은 매우 높은 품질의 오디오를 생성할 수 있지만, 샘플 하나하나를 순서대로 만들어야 하므로 생성 속도가 매우 느리다는 치명적인 단점이 있습니다. 이는 실시간 음성 서비스에 적용하기 어려운 주요한 장벽이었습니다.[1]
*   **비자기회귀 모델 (Non-autoregressive Models):** GAN이나 Flow 기반 모델들은 오디오 샘플들을 병렬적으로 한 번에 생성하여 속도가 매우 빠릅니다. 하지만 자기회귀 모델만큼의 정교하고 자연스러운 오디오 품질을 달성하는 데에는 어려움을 겪는 경우가 많았습니다.[1]

WaveGrad는 이 두 가지 모델의 장점을 모두 취하고자 했습니다. 즉, **빠른 생성 속도를 유지하면서도 자기회귀 모델 수준의 고품질 오디오를 생성하는 것**이 이 논문이 해결하고자 한 핵심 문제입니다.[1]

#### 2. 제안하는 방법 (수식 포함)

WaveGrad는 **확산 과정(diffusion process)**과 그 역 과정인 **제거 과정(denoising process)**에 기반합니다.

**확산 과정 (Forward Process):** 원본 오디오 신호 $$y_0$$에 점진적으로 가우시안 노이즈를 추가하여, 최종적으로는 순수한 가우시안 노이즈 $$y_N$$으로 만드는 과정입니다. 이 과정은 다음과 같은 마르코프 연쇄(Markov chain)로 정의됩니다.[1]

$$ q(y_{1:N}|y_0) = \prod_{n=1}^{N} q(y_n|y_{n-1}) $$

각 단계에서 노이즈는 다음과 같이 추가됩니다.

$$ q(y_n|y_{n-1}) = \mathcal{N}(y_n; \sqrt{1-\beta_n}y_{n-1}, \beta_n\mathbf{I}) $$

여기서 $$\beta_n$$은 각 단계에서 추가되는 노이즈의 양을 조절하는 스케줄(schedule)입니다. 이 과정의 중요한 특징은, 어떤 단계 $$n$$의 결과인 $$y_n$$을 원본 $$y_0$$로부터 한 번에 계산할 수 있다는 것입니다.[1]

$$ y_n = \sqrt{\bar{\alpha}_n}y_0 + \sqrt{1-\bar{\alpha}_n}\epsilon $$

이때 $$\alpha_n = 1-\beta_n$$, $$\bar{\alpha}\_n = \prod_{s=1}^{n}\alpha_s$$, 그리고 $$\epsilon \sim \mathcal{N}(0, \mathbf{I})$$ 입니다.

**제거 과정 (Reverse Process) 및 학습:** 모델이 학습하는 것은 바로 이 확산 과정의 역 과정, 즉 노이즈가 낀 신호 $$y_n$$에서 원본 신호 $$y_0$$에 추가되었던 노이즈 $$\epsilon$$을 예측하는 것입니다. 모델은 멜-스펙트로그램과 같은 조건 $$x$$와 현재 노이즈 레벨 $$n$$을 입력받아 노이즈 $$\epsilon$$을 예측하는 함수 $$\epsilon_\theta(y_n, x, n)$$를 학습합니다.

학습 목표는 예측된 노이즈와 실제 추가된 노이즈 사이의 오차를 최소화하는 것이며, 손실 함수는 다음과 같이 표현됩니다.[1]

$$ L = \mathbb{E}_{n, y_0, \epsilon} \left\| \epsilon - \epsilon_\theta(\sqrt{\bar{\alpha}_n}y_0 + \sqrt{1-\bar{\alpha}_n}\epsilon, x, n) \right\|^2_2 $$

논문에서는 L2 손실 대신 L1 손실을 사용했을 때 학습 안정성이 더 좋았다고 언급합니다.[1]

**추론 (Sampling):** 새로운 오디오를 생성할 때는 가우시안 노이즈 $$y_N$$에서 시작하여, 학습된 모델 $$\epsilon_\theta$$를 이용해 점진적으로 노이즈를 제거해 나갑니다. 각 단계에서 모델이 예측한 노이즈를 조금씩 빼주면서 최종적으로 깨끗한 오디오 $$y_0$$를 복원합니다.[1]

$$ y_{n-1} = \frac{1}{\sqrt{\alpha_n}}\left(y_n - \frac{1-\alpha_n}{\sqrt{1-\bar{\alpha}_n}}\epsilon_\theta(y_n, x, n)\right) + \sqrt{\beta_n}z $$

여기서 $$z \sim \mathcal{N}(0, \mathbf{I})$$는 약간의 무작위성을 더해주는 항입니다.

#### 3. 모델 구조

WaveGrad의 네트워크 구조는 U-Net과 유사한 형태를 가지며, GAN-TTS 모델에서 큰 영감을 받았습니다.[1]

*   **인코더 (Downsampling Blocks - DBlock):** 노이즈가 섞인 현재의 파형 $$y_n$$을 입력받아 다운샘플링하면서 특징을 추출합니다.
*   **디코더 (Upsampling Blocks - UBlock):** 멜-스펙트로그램 조건 $$x$$를 업샘플링하면서, 인코더에서 추출된 특징과 결합하여 최종 예측을 만듭니다. 이 과정에서 확장된 컨볼루션(dilated convolution)을 사용하여 넓은 수용 영역(receptive field)을 확보합니다.[1]
*   **FiLM (Feature-wise Linear Modulation):** 노이즈 레벨 정보와 멜-스펙트로그램 정보를 효과적으로 결합하기 위해 사용됩니다. 노이즈 레벨(또는 시간 스텝) $$n$$은 Transformer의 위치 인코딩과 유사한 방식으로 임베딩되어 FiLM 모듈에 입력됩니다. 이 모듈은 U-Block의 중간 특징 맵에 적용될 스케일(scale)과 편향(bias)을 생성하여, 각기 다른 노이즈 레벨에 맞춰 모델이 다르게 동작하도록 돕습니다.[1]
*   **비자기회귀 구조:** 전체 모델은 컨볼루션 신경망으로만 구성되어 있어, 학습과 추론 과정 모두에서 높은 병렬 처리가 가능합니다.[1]

#### 4. 성능 향상 및 한계

**성능 향상:**

*   **품질:** 주관적 음질 평가(MOS)에서 기존의 강력한 자기회귀 모델인 WaveRNN과 동등한 수준의 높은 점수를 기록했습니다.[1]
*   **속도:** 비자기회귀 모델로서 매우 빠른 추론 속도를 보여줍니다. NVIDIA V100 GPU 기준으로 WaveRNN보다 100배 빠른 속도를 달성하면서도, 단 6번의 반복만으로 4.4 이상의 높은 MOS 점수를 유지했습니다.[1]

**한계:**

*   **노이즈 스케줄 의존성:** 생성되는 오디오의 품질이 노이즈를 추가하고 제거하는 스케줄에 매우 민감합니다. 특히 적은 반복 횟수로 고품질 오디오를 생성하기 위해서는 이 스케줄을 정교하게 튜닝해야 합니다. 논문에서는 몇 가지 경험적인 가이드를 제시하지만, 최적의 스케줄을 찾는 체계적인 방법에 대한 탐구는 부족합니다.[1]
*   **하이퍼파라미터 튜닝 비용:** 최적의 노이즈 스케줄과 반복 횟수 N을 찾는 과정은 탐색 공간이 넓어 많은 비용과 시간이 소요될 수 있습니다.[1]

***

### 모델의 일반화 성능 향상 가능성

WaveGrad 논문에서 특히 강조하는 부분은 **모델의 일반화 성능 향상**입니다. 이는 두 가지 핵심적인 아이디어에서 비롯됩니다.

첫째, 모델을 **이산적인(discrete) 시간 스텝 `n`이 아닌, 연속적인(continuous) 노이즈 레벨 `ᾱ`에 직접 조건화**하는 것입니다. 기존의 확산 모델들은 1부터 N까지의 정수 스텝 `n`에 대해 모델을 학습시켰습니다. 이 경우, 학습 시 사용한 스텝 수 N과 다른 횟수로 추론을 하려면 모델을 다시 학습해야 하는 번거로움이 있었습니다.[1]

하지만 WaveGrad는 노이즈 레벨 자체를 연속적인 값으로 다루도록 모델을 변경했습니다.

$$ L = \mathbb{E}_{\bar{\alpha}, y_0, \epsilon} \left\| \epsilon - \epsilon_\theta(\sqrt{\bar{\alpha}}y_0 + \sqrt{1-\bar{\alpha}}\epsilon, x, \bar{\alpha}) \right\|_1 $$

이렇게 함으로써 **모델을 한 번만 학습시키면, 추론 시에는 원하는 만큼의 반복 횟수(N)를 자유롭게 설정**할 수 있게 됩니다. 예를 들어, 1000번의 반복을 가정하고 학습한 모델을 가지고, 실제 서비스를 할 때는 50번, 20번, 혹은 단 6번의 반복만으로도 오디오를 생성할 수 있습니다. 이는 실시간성과 품질 사이의 균형을 동적으로 조절할 수 있게 해주는 매우 강력한 장점이며, 모델의 일반화 성능이 크게 향상되었음을 의미합니다.[1]

둘째, **학습과 추론 시의 조건 불일치(mismatch) 문제 완화**입니다. 일반적으로 TTS(Text-to-Speech) 모델은 학습 시에는 실제 녹음된 오디오에서 추출한 '정답' 멜-스펙트로그램을 조건으로 사용하지만, 실제 추론 시에는 텍스트로부터 예측된 멜-스펙트로그램을 사용합니다. 이 둘 사이의 차이 때문에 성능 저하가 발생할 수 있습니다. 하지만 WaveGrad는 실험을 통해 이러한 불일치에도 불구하고 성능 저하가 거의 없었음을 보여주었습니다. 이는 TTS 모델의 학습 파이프라인을 크게 단순화시키는 이점으로 작용합니다.[1]

***

### 향후 연구에 미치는 영향 및 고려할 점

WaveGrad는 음성 생성 분야, 특히 확산 모델 기반 접근법에 큰 영향을 미쳤습니다.

**향후 연구에 미치는 영향:**

*   **확산 모델의 보편화:** WaveGrad와 거의 동시대에 발표된 DiffWave 등의 연구는 확산 모델이 이미지뿐만 아니라 오디오 생성에서도 강력한 성능을 낼 수 있음을 입증하며, 이후 수많은 후속 연구들이 확산 모델을 기반으로 하게 되는 계기가 되었습니다.[2][3]
*   **고속 샘플링 연구 촉진:** WaveGrad가 적은 스텝으로도 고품질 생성이 가능함을 보여주면서, 확산 모델의 샘플링 속도를 더욱 높이기 위한 다양한 기법(예: DDIM, ODE solver 기반 샘플링 등)에 대한 연구가 활발해졌습니다.
*   **다양한 오디오 생성 태스크로의 확장:** 텍스트로부터 오디오를 생성하는 Text-to-Audio(TTA) 시스템(예: AudioLDM )이나, 음악 생성 등 더 넓은 범위의 오디오 생성 분야로 확산 모델이 확장되는 데 기여했습니다.[4]
*   **WaveGrad 개선 연구:** WaveGrad를 직접적으로 개선하려는 연구도 등장했습니다. 예를 들어, `gla-grad`는 기존 WaveGrad의 확산 과정 각 단계에 Griffin-Lim 알고리즘을 적용하여 위상(phase) 복원 성능을 높임으로써, 특히 학습 데이터와 다른 특성을 가진 오디오를 생성할 때 더 나은 결과를 보여주었습니다.[5]

**앞으로 연구 시 고려할 점:**

*   **최적의 노이즈 스케줄 설계:** 여전히 경험적으로 튜닝되고 있는 노이즈 스케줄을 보다 체계적이고 이론적인 방법으로 설계하는 연구가 필요합니다. 이는 모델의 수렴 속도와 최종 품질에 직접적인 영향을 미칩니다.
*   **샘플링 효율성 극대화:** 실시간성을 요구하는 애플리케이션을 위해, 단 한두 번의 스텝으로도 만족스러운 품질을 내는 초고속 샘플링 기법에 대한 연구가 계속될 것입니다. 최근에는 Flow Matching과 같은 새로운 생성 모델링 기법들이 이러한 방향성을 보여주고 있습니다.[6][7]
*   **에일리어싱(Aliasing) 등 아티팩트 해결:** 신경망 기반 모델, 특히 업샘플링 과정에서 발생하는 에일리어싱(aliasing) 현상은 생성된 오디오의 품질을 저하시키는 요인입니다. `Wavehax`와 같은 최근 연구들은 이러한 아티팩트를 줄이는 데 초점을 맞추고 있습니다.[8]
*   **모델 경량화 및 온디바이스(On-device) 적용:** 더 적은 파라미터와 계산량으로도 높은 성능을 내는 경량화된 모델을 개발하여, 스마트폰이나 스마트 스피커와 같은 기기에서 직접 구동될 수 있도록 하는 연구가 중요합니다.

***

### 2020년 이후 관련 최신 연구 탐색

WaveGrad 이후, 오디오 생성을 위한 확산 및 관련 모델 연구는 폭발적으로 증가했습니다. 주요 흐름은 다음과 같습니다.

*   **PeriodWave & PeriodWave-Turbo (2024년 8월):** GAN 기반 모델의 빠른 속도와 확산 모델의 높은 품질을 결합하려는 시도 중 하나로, 적대적 플로우 매칭(Adversarial Flow Matching)이라는 새로운 기법을 제안합니다. 이는 고품질 오디오를 더 효율적으로 생성하는 것을 목표로 합니다.[7][6]
*   **gla-grad (2024년):** WaveGrad를 직접 확장한 모델로, 위상 복원을 위해 Griffin-Lim 알고리즘을 통합하여 특히 학습 데이터와 다른 도메인의 오디오 생성 시 강점을 보입니다.[5]
*   **Wavehax (2024년 11월):** 신경망 보코더에서 발생하는 에일리어싱 문제를 해결하기 위해 2D 컨볼루션과 고조파 사전 정보(harmonic prior)를 활용하는 새로운 구조를 제안합니다.[8]
*   **AudioLDM (2023년):** 사전 학습된 오디오-텍스트 모델(CLAP)을 활용하여, 텍스트 설명으로부터 다양한 종류의 소리를 생성하는 잠재 확산 모델(Latent Diffusion Model)입니다. 이는 확산 모델이 음성 합성을 넘어 일반적인 오디오 생성으로 성공적으로 확장된 사례입니다.[4]
*   **AI 합성 음성 탐지 연구:** WaveGrad와 같은 고품질 보코더의 발전은 진짜와 가짜를 구별하기 어렵게 만들었고, 이에 따라 신경망 보코더가 남기는 미세한 흔적(artifact)을 탐지하여 AI가 생성한 음성을 판별하는 연구도 활발하게 진행되고 있습니다.[9][10]

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/4c78a315-428c-4a66-8eca-aed9fd3f0c9a/2009.00713v2.pdf)
[2](https://wandb.ai/wandb_gen/audio/reports/A-Technical-Guide-to-Diffusion-Models-for-Audio-Generation--VmlldzoyNjc5ODIx)
[3](https://huggingface.co/learn/diffusion-course/unit4/3)
[4](https://audioldm.github.io)
[5](https://www.merl.com/publications/docs/TR2024-014.pdf)
[6](http://arxiv.org/pdf/2408.08019.pdf)
[7](http://arxiv.org/pdf/2408.07547.pdf)
[8](http://arxiv.org/pdf/2411.06807.pdf)
[9](https://openaccess.thecvf.com/content/CVPR2023W/WMF/papers/Sun_AI-Synthesized_Voice_Detection_Using_Neural_Vocoder_Artifacts_CVPRW_2023_paper.pdf)
[10](https://www.albany.edu/faculty/mchang2/files/2022-06_WIFS_ExposeAudio.pdf)
[11](https://ieeexplore.ieee.org/document/8727289/)
[12](https://ieeexplore.ieee.org/document/11233004/)
[13](https://www.semanticscholar.org/paper/187c8e05b8970ebe7be1b2e14c5197142e90bcbc)
[14](https://www.mdpi.com/2076-3263/14/10/263)
[15](https://ieeexplore.ieee.org/document/10974693/)
[16](https://ieeexplore.ieee.org/document/10037964/)
[17](https://www.semanticscholar.org/paper/f3c6dc529fb0fe1652140f9b62e9d74b561df4cf)
[18](https://www.nap.edu/catalog/25761)
[19](https://ieeexplore.ieee.org/document/9254769/)
[20](https://www.semanticscholar.org/paper/db155c75fa75fbfc491986319011ad4a3cdf72a3)
[21](https://arxiv.org/pdf/1910.11480.pdf)
[22](http://arxiv.org/pdf/2407.04544.pdf)
[23](http://arxiv.org/pdf/1904.02892.pdf)
[24](https://arxiv.org/pdf/1910.06711.pdf)
[25](http://arxiv.org/pdf/2407.19862.pdf)
[26](https://openreview.net/pdf?id=NsMLjcFaO8O)
[27](https://openreview.net/pdf/852dc9f15526e62d1a3d57bf2823c4dd74367e31.pdf)
[28](https://arxiv.org/pdf/2009.00713.pdf)
[29](https://www.sciencedirect.com/science/article/abs/pii/S0003682X22005576)
