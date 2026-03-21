
# AnimateDiff: Animate Your Personalized Text-to-Image Diffusion Models without Specific Tuning

**저자**: Yuwei Guo, Ceyuan Yang, Anyi Rao, Zhengyang Liang, Yaohui Wang, Yu Qiao, Maneesh Agrawala, Dahua Lin, Bo Dai  
**발표**: ICLR 2024 Spotlight (arXiv: 2307.04725)

---

## 1. 핵심 주장 및 주요 기여 (요약)

AnimateDiff는 개인화된 T2I(Text-to-Image) 모델을 모델별 특화 튜닝 없이 애니메이션 생성기로 전환하는 실용적 프레임워크이며, 핵심은 한 번 학습하면 동일 기반 T2I에서 파생된 어떤 개인화 모델에나 원활히 통합할 수 있는 **plug-and-play 모션 모듈**이다.

### 주요 기여:
1. **Plug-and-Play Motion Module**: 제안된 학습 전략을 통해 모션 모듈이 실제 동영상에서 전이 가능한 모션 사전(motion priors)을 학습하며, 한 번 학습 후 개인화 T2I 모델에 삽입하면 개인화 애니메이션 생성기가 된다.
2. **MotionLoRA**: 사전 학습된 모션 모듈이 카메라 줌, 패닝 등 새로운 모션 패턴에 적은 학습·데이터 수집 비용으로 적응할 수 있게 하는 경량 파인튜닝 기법이다.
3. **Domain Adapter**: 비디오 데이터셋의 워터마크 등 시각적 결함을 흡수하여 모션과 외형 학습을 분리한다.
4. **일반화 성능 검증**: 애니메 스타일부터 실사 사진까지 다양한 공개 개인화 T2I 모델에서 시간적으로 매끄러운 애니메이션 클립을 생성하면서 도메인과 출력 다양성을 보존함을 입증하였다.

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

Stable Diffusion과 DreamBooth, LoRA 등의 개인화 기법 덕분에 누구나 고품질 이미지를 저비용으로 생성할 수 있게 되었지만, 기존 개인화 T2I 모델에 모션 역동성을 추가하여 애니메이션을 생성하는 것은 여전히 미해결 과제이다.

기존 접근법의 한계:
- T2V(Text-to-Video) 모델: 대규모 비디오 데이터셋으로 처음부터 학습해야 하며 막대한 연산 자원이 필요하고, 개인화 T2I 모델의 특성을 보존하지 못할 수 있다.
- Training-Free 방법: 노이즈 공간 조작만으로는 프레임 간 시간적 일관성 유지에 한계가 있다.
- 모델별 튜닝: 도메인별 비디오 데이터 부족과 연산 비용 문제로 비실용적이다.

### 2.2 제안 방법 및 수식

#### (1) 기본 프레임워크: Stable Diffusion (Latent Diffusion Model)

Stable Diffusion은 VAE 인코더 $\mathcal{E}$와 디코더 $\mathcal{D}$를 사용하여 잠재 공간에서 확산 과정을 수행한다. Forward process에서는 잠재 코드 $z_0$에 노이즈를 점진적으로 추가한다:

$$q(z_t | z_0) = \mathcal{N}(z_t; \sqrt{\bar{\alpha}_t}\, z_0,\; (1-\bar{\alpha}_t)\,\mathbf{I})$$

여기서 $\bar{\alpha}\_t = \prod_{s=1}^{t} \alpha_s$ 는 노이즈 스케줄에 따른 누적 곱이다.

#### (2) 학습 목적 함수 (Training Objective)

모션 모듈은 각 확산 단계에서 예측된 노이즈와 실제 노이즈 간의 MSE(평균 제곱 오차) 손실을 사용하여 학습된다. 학습 목표는 Latent Diffusion Model의 것을 따르며, 잠재 코드에 추가된 노이즈 강도를 예측하고 $L_2$ 손실로 학습한다.

$$\mathcal{L} = \mathbb{E}_{z_0, \epsilon \sim \mathcal{N}(0,\mathbf{I}),\, t}\Big[\|\epsilon - \epsilon_\theta(z_t, t, c)\|_2^2\Big]$$

여기서:
- $\epsilon$: 실제 추가된 노이즈
- $\epsilon_\theta$: 모델이 예측하는 노이즈 (U-Net + Motion Module)
- $z_t$: 시간 $t$에서의 잠재 코드
- $c$: 텍스트 조건 임베딩

#### (3) 네트워크 인플레이션 (5D 텐서)

사전 학습된 이미지 레이어가 비디오 프레임을 독립적으로 처리하도록, 모델을 5D 비디오 텐서 $z \in \mathbb{R}^{b \times c \times f \times h \times w}$를 입력으로 받도록 수정한다. 이미지 레이어를 통과할 때는 시간 축 $f$를 배치 축 $b$에 병합하여 각 프레임을 독립적으로 처리한다.

이미지 레이어 통과 시 reshape:
$$z \in \mathbb{R}^{b \times c \times f \times h \times w} \rightarrow z' \in \mathbb{R}^{(b \cdot f) \times c \times h \times w}$$

모션 모듈 통과 시 reshape:
$$z \in \mathbb{R}^{b \times c \times f \times h \times w} \rightarrow z'' \in \mathbb{R}^{(b \cdot h \cdot w) \times c \times f}$$

#### (4) MotionLoRA

사전 학습된 모션 모듈이 카메라 줌, 패닝, 롤링 등 새로운 모션 패턴에 적은 수의 참조 비디오와 학습 반복으로 효율적으로 적응해야 하며, 이러한 효율성은 비싼 사전 학습 비용을 감당하지 못하는 사용자에게 필수적이다.

LoRA의 핵심 수식:

$$W' = W + \Delta W = W + BA$$

여기서 $B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times k}$ ( $r \ll \min(d,k)$ )로, 학습 가능한 파라미터를 저차원 행렬 쌍으로 분해한다.

MotionLoRA 모델은 약 30MB의 추가 저장 공간만 필요하여 모델 공유 효율성을 극대화한다.

### 2.3 모델 구조

AnimateDiff의 구조는 3단계 학습 파이프라인으로 구성된다:

**Stage 1 — Domain Adapter**: 학습 데이터셋의 워터마크 등 결함적 시각 아티팩트를 적합시키기 위해 도메인 어댑터를 학습하며, 이는 모션과 공간 외형의 분리 학습에도 도움이 된다. 추론 시 어댑터는 제거 가능하다.

**Stage 2 — Motion Module**: 비디오에서 실제 모션 패턴을 학습하기 위해 모션 모듈을 학습한다.

**Stage 3 — MotionLoRA (선택)**: 카메라 줌, 롤링 등 특정 모션 패턴에 모션 모듈을 효율적으로 적응시키기 위해 MotionLoRA를 학습한다.

**Motion Module (Temporal Transformer)의 세부 설계:**

Temporal Transformer는 시간 축을 따라 여러 self-attention 블록으로 구성되며, 사인파 위치 인코딩(sinusoidal position encoding)을 사용하여 애니메이션에서 각 프레임의 위치를 인코딩한다.

Temporal Self-Attention 수식:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

여기서 $Q$, $K$, $V$는 시간 축을 따라 재배열된 feature map에서 추출된다.

추가 모듈의 부정적 영향을 방지하기 위해, Temporal Transformer의 출력 프로젝션 레이어를 **zero initialization**하고 잔차 연결(residual connection)을 추가하여 학습 초기에 모션 모듈이 항등 사상(identity mapping)이 되도록 한다.

모션 모듈은 고정(frozen)된 T2I 모델에 모션 모듈 레이어를 삽입하여 달성되며, 이 모션 모듈들은 Stable Diffusion UNet의 ResNet 및 Attention 블록 뒤에 적용된다.

### 2.4 성능 및 평가

AnimateDiff는 Text2Video-Zero, Tune-a-Video 등 주요 비디오 합성 모델과 비교 평가되었으며, 핵심 평가 지표로는 텍스트-이미지 정합성(text-image alignment), 도메인 유사도(domain similarity), 모션 부드러움(motion smoothness)이 포함된다.

AnimateDiff는 시간적으로 매끄럽고 일관된 애니메이션을 생성하며, Text2Video-Zero와 같은 training-free 방법 대비 프레임 간 콘텐츠 일관성과 자연스러운 모션에서 더 우수한 성능을 보인다.

사용자 연구에서 AnimateDiff는 모션 부드러움과 도메인 보존에서 호의적인 평가를 받았으며, CLIP 유사도 기반 정량 지표에서도 의미적 정합성과 시각적 일관성 모두에서 강한 성능을 보였다.

### 2.5 한계

고도로 양식화되거나 비현실적인 개인화 모델에서는 비디오 학습 데이터 분포와 크게 벗어나 어려움을 겪을 수 있다.

스케일러 파라미터가 기본적인 애니메이션 강도 제어를 제공하지만, 특정 모션에 대한 더 상세한 제어는 여전히 과제이다.

현재 접근법은 비교적 짧은 애니메이션을 위해 설계되었으며, 더 긴 시퀀스로의 확장은 일관성 유지를 위한 추가 메커니즘이 필요할 수 있다.

모션의 품질은 학습 데이터에 민감하며, 학습 데이터에 존재하지 않는 특이한 그래픽은 애니메이션화하기 어렵다.

---

## 3. 모델의 일반화 성능 향상 가능성

AnimateDiff의 일반화 성능은 이 논문의 핵심 강점 중 하나이다.

### 3.1 일반화를 가능케 하는 설계 원리

1. **콘텐츠와 모션의 분리(Decoupling)**: 콘텐츠 생성과 모션 모델링을 분리함으로써, 단일 사전 학습된 모션 모듈로 다양한 개인화 모델의 애니메이션을 가능하게 한다.

2. **대규모 비디오 데이터 학습**: 모션 모듈은 주로 WebVid-10M과 같은 대규모 비디오 데이터셋에서 학습되며, 수백만 개의 실제 비디오 클립에 노출되어 특정 콘텐츠나 도메인에 과적합하지 않고 광범위한 모션 사전을 증류할 수 있다.

3. **Zero Initialization 전략**: Temporal Transformer 모듈은 항등 연산자(identity operator)로 초기화되어 학습 초기 단계에서 안정적인 수렴을 보장한다. 이 전략은 기존 T2I 모델의 지식을 훼손하지 않으면서 점진적으로 모션을 학습하게 한다.

4. **도메인 어댑터**: 비디오 데이터셋의 아티팩트(워터마크 등)를 별도로 흡수하여, 모션 모듈이 순수한 모션 패턴에 집중할 수 있게 한다.

### 3.2 일반화 향상을 위한 향후 방향

- **스케일업 학습**: 더 큰 해상도와 배치 크기에서 학습된 v2 모션 모듈은 모션 품질과 다양성 향상에 크게 기여하는 것으로 확인되었다.
- **새로운 베이스 모델 확장**: SDXL 등 다른 Stable Diffusion 변형에 대한 지원이 진행 중이다.
- **MotionLoRA의 조합**: 개별적으로 학습된 MotionLoRA 모델들을 결합하여 추론 시 복합 모션 효과를 달성할 수 있으며, 이는 저차원(low-rank) 속성 덕분이다.
- 다양한 프롬프트와 개인화 스타일에 대해 강건하며, ToonYou, Lyriel, majicMIX Realistic 등 광범위한 커뮤니티 학습 모델과 호환된다.

---

## 4. 향후 연구에 미치는 영향 및 고려사항

### 4.1 연구 영향

1. **"Inflate-then-Animate" 패러다임 확립**: AnimateDiff는 기존 이미지 모델을 비디오 모델로 확장하는 "inflation" 패러다임을 실용적으로 검증하여, 후속 연구(PIA, AnimateZero, MotionDirector 등)의 기반을 마련하였다.

2. **커뮤니티 모델 생태계 활성화**: ICLR 2024 Spotlight으로 선정된 이 연구의 공식 구현은 대부분의 커뮤니티 T2I 모델을 추가 학습 없이 애니메이션 생성기로 전환하는 plug-and-play 모듈이다.

3. **효율적 적응의 새 표준**: MotionLoRA는 소규모 참조 비디오(20~50개)만으로도 특정 모션 패턴을 학습할 수 있음을 보여, 효율적 파인튜닝의 새로운 방향을 제시하였다.

4. **후속 확장 연구 촉진**: SparseCtrl을 통한 제어 가능한 애니메이션, FreeNoise를 통한 긴 비디오 생성 등 다양한 확장 연구가 가능해졌다.

### 4.2 향후 연구 시 고려할 점

| 고려사항 | 설명 |
|---------|------|
| **장기 시간 일관성** | 현재 16프레임 기반 설계의 한계를 극복하여 분 단위 이상의 비디오 생성 |
| **세밀한 모션 제어** | 텍스트 프롬프트만으로 특정 동작 시퀀스를 정교하게 제어하는 메커니즘 |
| **물리 법칙 준수** | 생성 비디오의 물리적 현실성 (중력, 충돌, 유체 역학 등) |
| **고해상도 확장** | SDXL 이상의 대규모 모델에서의 안정적 동작 |
| **평가 메트릭** | 시간적 일관성, 모션 품질을 정량적으로 측정하는 표준화된 벤치마크 |
| **윤리적 고려** | 딥페이크 등 오용 가능성에 대한 안전장치 |

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

| 연구 | 연도 | 접근법 | AnimateDiff와의 차이점 |
|------|------|--------|----------------------|
| **Make-A-Video** (Singer et al.) | 2022 | 사전 학습된 확산 이미지 모델을 시간 차원으로 확장하며, 기본 T2I 모델 + 시간 레이어로 구성된다. | 대규모 사전 학습 필요; 개인화 모델 호환 미고려 |
| **Tune-A-Video** (Wu et al.) | 2023 | 소수의 파라미터를 단일 비디오에 파인튜닝한다. | 비디오별 튜닝 필요; 일반화 성능 제한 |
| **Text2Video-Zero** (Khachatryan et al.) | 2023 | 사전 정의된 어파인 행렬 기반 잠재 래핑으로 학습 없이 T2I를 애니메이션화한다. | Training-free이나 시간적 일관성 한계 |
| **Align-Your-Latents** (Blattmann et al.) | 2023 | 일반 비디오 생성기 내 고정된 이미지 레이어를 개인화할 수 있음을 보여주었다. | 비디오 모델 기반; AnimateDiff는 이미지 모델 기반 |
| **Sora** (OpenAI) | 2024 | DiT(Diffusion Transformer) 아키텍처를 활용하여 시공간 패치 위에서 동작하며, 더 나은 스케일링을 달성한다. | 대규모 자원 필요; 비공개 모델 |
| **CogVideoX** (Yang et al.) | 2024 | 3D causal convolution tokenizer를 사용하는 2B/5B 오픈 가중치 비디오 모델로, 확산 공식으로 학습된다. | End-to-end 비디오 모델; AnimateDiff의 모듈성과 대조적 |
| **FreeNoise** (Qiu et al.) | 2023 | 노이즈 재스케줄링을 통해 짧은 비디오 생성 모델로 더 긴 비디오를 생성할 수 있는 샘플링 메커니즘이다. | AnimateDiff와 상호 보완적으로 사용 가능 |
| **AnimateLCM** | 2024 | 분리된 일관성 학습(decoupled consistency learning)으로 AnimateDiff를 가속화 | AnimateDiff 기반 속도 최적화 |
| **MotionDirector** | 2024 | 외형과 모션의 학습을 분리하기 위한 이중 경로 LoRA 아키텍처와, 시간 학습 목표에서 외형의 영향을 줄이는 새로운 손실을 설계한다. | AnimateDiff의 모션-외형 분리를 더욱 정교화 |

### 패러다임 변화 요약

최근 확산 모델의 발전은 비디오 생성에 혁명을 가져왔으며, 전통적인 GAN 기반 접근법 대비 우수한 시간적 일관성과 시각 품질을 제공하지만, 모션 일관성, 연산 효율성, 윤리적 고려 등에서 여전히 상당한 과제가 있다.

AnimateDiff는 이 생태계 내에서 **모듈형 접근법의 효과성**을 입증하여, 대규모 end-to-end 비디오 모델(Sora 등)과 경량 plug-and-play 접근법 사이의 중요한 중간 지점을 차지하고 있다.

---

## 참고자료 출처

1. **[arXiv] AnimateDiff: Animate Your Personalized Text-to-Image Diffusion Models without Specific Tuning** — https://arxiv.org/abs/2307.04725
2. **[Project Page] AnimateDiff** — https://animatediff.github.io/
3. **[GitHub] guoyww/AnimateDiff** — https://github.com/guoyww/AnimateDiff
4. **[OpenReview] AnimateDiff (ICLR 2024)** — https://openreview.net/forum?id=Fx2SbBgcte
5. **[HuggingFace] AnimateDiff Paper Page** — https://huggingface.co/papers/2307.04725
6. **[HuggingFace] Diffusers AnimateDiff Documentation** — https://huggingface.co/docs/diffusers/en/api/pipelines/animatediff
7. **[alphaXiv] AnimateDiff Overview** — https://www.alphaxiv.org/overview/2307.04725
8. **[arXiv HTML] AnimateDiff v2 Full Paper** — https://arxiv.org/html/2307.04725v2
9. **[Open Laboratory] SDXL Motion Model / AnimateDiff** — https://openlaboratory.ai/models/sdxl-motion-model
10. **[Semantic Scholar] AnimateDiff** — https://www.semanticscholar.org/paper/AnimateDiff
11. **[HuggingFace] animatediff-motion-adapter-v1-5** — https://huggingface.co/guoyww/animatediff-motion-adapter-v1-5
12. **[Lil'Log] Diffusion Models for Video Generation** — https://lilianweng.github.io/posts/2024-04-12-diffusion-video/
13. **[Springer] Video diffusion generation: comprehensive review and open problems** — https://link.springer.com/article/10.1007/s10462-025-11331-6
14. **[arXiv] Survey of Video Diffusion Models: Foundations, Implementations, and Applications** — https://arxiv.org/abs/2504.16081
15. **[Yen-Chen Lin Blog] Video Generation Models Explosion 2024** — https://yenchenlin.me/blog/2025/01/08/video-generation-models-explosion-2024/
16. **[ResearchGate] AnimateDiff PDF** — https://www.researchgate.net/publication/372248525
17. **[Medium] AnimateDiff: Revolutionize Text-to-Image Generation** — https://panditshivam.medium.com/animatediff-revolutionize-text-to-image-generation-and-animation
18. **[Stable Diffusion Art] AnimateDiff Guide** — https://stable-diffusion-art.com/animatediff/

> ⚠️ **참고**: 본 분석에서 수식은 논문의 공개된 방법론 설명과 확산 모델의 표준 수학적 공식에 기반하여 재구성한 것입니다. 특정 세부 수식이 원본 논문 PDF에서 약간 다른 표기법을 사용할 수 있으나, 수학적 의미는 동일합니다.
