# Hunyuan3D 2.0: Scaling Diffusion Models for High Resolution Textured 3D Assets Generation

## 1. 핵심 주장 및 주요 기여 (요약)

Hunyuan3D 2.0은 단일 이미지로부터 고해상도 텍스처드 3D 자산을 생성하는 대규모 오픈소스 3D 합성 시스템입니다. 핵심 주장은 "이미지·비디오 생성 분야에서 입증된 **Latent Diffusion 패러다임**과 **Flow Matching** 학습 기법을 3D 도메인에 본격적으로 스케일업하면, 폐쇄형 상용 모델을 능가하는 형상·텍스처 품질을 오픈소스로 달성할 수 있다"는 것입니다.

주요 기여는 네 가지로 정리됩니다. 첫째, **Hunyuan3D-ShapeVAE**는 vector set 표현(3DShape2VecSet 계열)에 *importance sampling* 전략을 도입해 모서리·코너 등 고주파 영역의 정보 손실을 줄이고, 가변 토큰 길이(최대 3072) 학습으로 미세 디테일을 보존합니다. 둘째, **Hunyuan3D-DiT**는 FLUX 스타일의 dual/single-stream 트랜스포머 위에서 flow-matching 목적함수로 학습된 형상 생성용 latent diffusion 모델입니다. 셋째, **Hunyuan3D-Paint**는 mesh-conditioned multi-view diffusion 모델로, image delighting, double-stream reference-net, multi-task attention, dense-view inference, view dropout 등을 결합해 조명 불변·다중뷰 일관성을 갖춘 텍스처를 굽습니다(bake). 넷째, **Hunyuan3D-Studio** 플랫폼을 통해 sketch-to-3D, low-poly stylization, character animation 등 다운스트림 응용을 통합합니다.

정량 평가에서 Trellis 및 3개 비공개 상용 모델 대비 V-IoU 93.6%, S-IoU 89.16% (재구성), ULIP/Uni3D 계열의 조건 정합도, FID_CLIP·CMMD·CLIP-score(텍스처) 모두 1위를 기록했고, 50명 사용자 스터디에서도 가장 높은 만족도를 받았습니다.

---

## 2. 문제 정의·제안 방법·구조·성능·한계

### 2.1 해결하고자 하는 문제

3D 자산 제작은 스케치, 모델링, UV 언랩, 텍스처링, 베이킹 등 다단계 숙련 노동을 요구합니다. 한편 이미지/비디오 생성은 Stable Diffusion·HunyuanVideo 같은 강력한 오픈소스 파운데이션 모델 덕에 폭발적으로 발전했지만, 3D 분야는 (i) 표현 형식이 통일되지 않고(voxel/point/mesh/SDF/Gaussian), (ii) 데이터 규모가 작으며, (iii) 형상과 텍스처를 동시에 학습하기 어렵다는 한계로 정체되어 있었습니다. 저자들은 이를 "**오픈소스 대규모 3D 파운데이션 모델의 부재**"로 진단합니다.

### 2.2 제안 방법: 2단계 파이프라인

전체 파이프라인은 *bare mesh 생성 → 텍스처 합성*의 디커플링 구조입니다. 이 구조는 형상과 텍스처의 난이도를 분리하고, 사용자가 직접 제작한 mesh에도 텍스처 모듈을 재사용할 수 있게 합니다.

**(A) Hunyuan3D-ShapeVAE — 형상 인코딩.** 표면에서 두 종류의 점군을 샘플링합니다: 균일 샘플 $P_u \in \mathbb{R}^{M\times 3}$ 와 중요도 샘플 $P_i \in \mathbb{R}^{N\times 3}$. 각각에 FPS(Farthest Point Sampling)를 적용해 쿼리 $Q_u, Q_i$를 얻고 결합한 뒤, Fourier positional encoding과 선형 사영을 거쳐 cross-attention + self-attention 스택으로 latent token sequence $Z_s \in \mathbb{R}^{(M'+N')\times d_0}$를 만듭니다. 디코더는 3D grid query $Q_g \in \mathbb{R}^{(H\times W\times D)\times 3}$로 SDF $F_{\text{sdf}}$를 회귀하고, marching cubes로 메시를 추출합니다.

학습 손실은 KL 항이 결합된 reconstruction loss입니다.

$$\mathcal{L}_r = \mathbb{E}_{x\in\mathbb{R}^3}\big[\mathrm{MSE}(\mathcal{D}_s(x|Z_s),\,\mathrm{SDF}(x))\big] + \gamma\,\mathcal{L}_{KL}$$

여기서 SDF 전체 계산이 무거우므로 공간/표면에서 무작위 샘플링한 점들에 대한 기댓값으로 근사합니다. 또한 **multi-resolution** 토큰 길이(최대 3072)를 무작위로 사용해 학습을 가속하고 고해상도까지 커버합니다.

**(B) Hunyuan3D-DiT — 형상 생성.** FLUX-style의 dual-stream(16개) + single-stream(32개) 트랜스포머 백본을 채택했습니다. Dual-stream에서는 latent 토큰과 이미지 조건 토큰이 별도 QKV/MLP를 가지면서 attention 안에서만 상호작용하고, single-stream에서는 두 토큰을 concat해 spatial/channel attention을 병렬 처리합니다. 이미지 조건은 **DINOv2-Giant**로 인코딩한 $518\times 518$ 입력 패치 토큰(헤드 토큰 포함)을 사용합니다.

학습 목적은 **Flow Matching** (Lipman et al., 2022; Esser et al., 2024)으로, 가우시안 $x_0$와 데이터 $x_1$ 사이의 affine path

$$x_t = (1-t)\,x_0 + t\,x_1, \qquad u_t = x_1 - x_0$$

를 정의하고 속도장 $u_\theta$를 예측합니다.

$$\mathcal{L} = \mathbb{E}_{t,x_0,x_1}\Big[\,\big\| u_\theta(x_t, c, t) - u_t \big\|_2^2 \,\Big],\quad t\sim\mathcal{U}(0,1)$$

추론 시 1차 Euler ODE 솔버로 $x_0\sim\mathcal{N}(0,I)$에서 시작해 $x_1$로 적분합니다. 또 latent 토큰이 3D grid의 고정 위치에 대응하지 않기 때문에 **positional embedding을 의도적으로 제거**한 점이 이미지/비디오 DiT와의 결정적 차이입니다.

**(C) Hunyuan3D-Paint — 텍스처 합성.** SD2.1 ZSNR v-model을 백본으로 시작해 다음 모듈을 추가합니다.

*Image Delighting*: 입력 이미지의 음영을 제거(InstructPix2Pix 계열)하여 조명-불변 텍스처를 학습 가능하게 만듭니다. 학습 시 동일 메시를 random HDRI와 균일 백색광 두 가지로 렌더링한 페어 데이터를 사용합니다.

*View Selection*: 4개의 직교 뷰를 고정하고, 다음 식을 최대화하는 그리디 탐색으로 8–12뷰까지 확장합니다.

```math
\mathcal{F}(v_i, \mathbb{V}_s, \mathbf{M}) = \mathcal{A}_{\text{area}}\!\left\{\mathcal{UV}_{\text{cover}}(v_i,\mathbf{M})\setminus\!\Big[\mathcal{UV}_{\text{cover}}(v_i,\mathbf{M})\cap\Big(\bigcup_{s\in\mathbb{V}_s}\mathcal{UV}_{\text{cover}}(v_s,\mathbf{M})\Big)\Big]\right\}
```

이는 "가장 많이 보이지 않은 영역을 새로 커버하는 뷰"를 우선 선택하여 텍스처 인페인팅 부담을 줄입니다.

*Multi-task Attention*: 원본 self-attention $Z_{SA}$에 reference branch와 multi-view branch 어텐션을 *병렬*로 더합니다.
$$Z_{MVA} = Z_{SA} + \lambda_{ref}\cdot\mathrm{Softmax}\!\left(\frac{Q_{ref}K_{ref}^{\top}}{\sqrt{d}}\right)V_{ref} + \lambda_{mv}\cdot\mathrm{Softmax}\!\left(\frac{Q_{mv}K_{mv}^{\top}}{\sqrt{d}}\right)V_{mv}$$

병렬 구조 덕분에 reference 보존(이미지 정합)과 view 일관성이라는 두 목표 간 충돌이 완화됩니다.

*Geometry Conditioning*: canonical normal map과 canonical coordinate map(CCM)을 VAE로 인코딩한 뒤 latent noise와 채널 concat하고, 학습 가능한 viewpoint embedding을 더해 강건한 뷰포인트 단서를 제공합니다.

*Dense-view + View Dropout*: 사전 정의된 44뷰 중 매 학습 step마다 6뷰만 샘플링해 학습합니다. 추론 시 임의 뷰 출력이 가능해 dense view 인페인팅 부담을 줄입니다.

학습 설정: $512\times 512$ 해상도, batch 48, lr $5\times10^{-5}$, 80,000 step, 1000 warm-up, ZSNR trailing 스케줄러.

### 2.3 성능 향상 (논문 표 1, 2, 3, 4 요약)

형상 재구성에서 V-IoU 93.6% / S-IoU 89.16%로 Direct3D(88.43% / 81.55%), 3DShape2VecSet, Michelangelo를 모두 앞섭니다. 형상 생성에서는 Trellis 및 3개 비공개 모델 대비 ULIP-T 0.0771, Uni3D-T 0.2519, Uni3D-I 0.3151로 1위 또는 공동 1위, 텍스처 합성에서는 CMMD 2.318, FID_CLIP 26.44, CLIP-score 0.8893으로 TexPainter·Paint3D·SyncMVD 등을 모두 능가합니다. 엔드투엔드 평가에서도 CLIP-score 0.809, FID_CLIP 49.165로 Trellis(0.787, 54.639) 및 비공개 모델 1–3을 앞섰고, 사용자 스터디 만족도 약 79–82%로 최고점입니다.

### 2.4 한계

논문이 명시하지 않거나 시사만 한 한계는 다음과 같습니다. (i) 텍스처가 **albedo-only(RGB)**이고 PBR(roughness/metallic) 출력이 없어 photorealistic 렌더링/리라이팅에 제약이 있습니다 — 실제로 이는 후속 Hunyuan3D 2.1에서 albedo·roughness·metallic 동시 생성과 3D-Aware RoPE 도입으로 보완되었습니다. (ii) 본질적으로 SDF→marching cubes 기반이라 **개곡면(open surfaces)·비매니폴드·내부 구조**를 잘 다루지 못합니다(이 한계는 TRELLIS.2의 O-Voxel 표현이 직접적으로 공격하는 지점입니다). (iii) Mesh topology가 dense triangle이라 게임 파이프라인이 요구하는 artist-style low-poly 토폴로지는 별도 후처리(Hunyuan3D-Studio의 quadric error metric 단순화)에 의존합니다. (iv) 데이터 규모·라이선스가 명확히 공개되지 않았고, 사람·얼굴 등 도메인 편향이 잠재합니다. (v) Diffusion sampling이 여전히 다단계라 실시간성에는 미흡합니다.

---

## 3. 일반화 성능 향상 가능성 (집중 분석)

이 논문에는 *형상 분포 일반화*와 *조건(이미지) 일반화* 두 축에서 의미 있는 설계 결정이 모여 있습니다.

**(a) Vector set + Importance Sampling으로 표현 병목 완화.** 균일 샘플만으로는 복잡한 형상의 모서리 정보가 latent로 압축되며 손실되지만, 중요도 샘플로 별도 FPS-쿼리를 구성한 뒤 두 점군을 concat하면 *동일한 토큰 예산 내에서* 고주파 영역 정보가 더 보존됩니다. 이는 "처음 보는 카테고리의 날카로운 디테일"에 대해서도 재구성/생성이 무너지지 않는 일반화 효과를 줍니다. V-IoU/S-IoU에서의 큰 폭 향상이 이 효과를 시사합니다.

**(b) 가변 토큰 길이 학습.** 짧은 시퀀스와 긴 시퀀스를 무작위로 섞어 학습하면, 추론 시 해상도-품질 트레이드오프를 사용자가 선택할 수 있을 뿐 아니라 *형상 복잡도가 다른 데이터에 대한 일반화*가 향상됩니다. 단순 객체에는 짧은 시퀀스로도 안정적이고, 복잡 객체에는 3072 토큰까지 확장 가능한 *zero-shot 길이 외삽*에 가까운 효과를 노립니다.

**(c) Positional Embedding 제거.** 3D latent token이 grid 위 특정 위치에 묶이지 않으므로 모델은 "어디에 무엇이 있는지"를 토큰 내용 자체로 학습합니다. 이는 객체 스케일·위치·축 배향 변화에 대한 일반화에 기여합니다.

**(d) DINOv2-Giant + 518×518 + 배경 제거.** 강력한 self-supervised 비전 인코더를 큰 입력 해상도로 사용하면, 학습 분포 밖의 사진·일러스트·스케치까지도 의미적/구조적 단서가 보존됩니다. 배경을 제거하고 객체를 정중앙·균일 크기로 재배치하는 전처리는 "야생 이미지(in-the-wild)"에 대한 도메인 갭을 효과적으로 좁힙니다. 실제로 sketch-to-3D 모듈이 이 강건성을 활용합니다.

**(e) Flow Matching의 안정성.** DDPM 대비 flow matching은 분포 간 직선 경로에 가까워 학습이 안정적이고, 적은 step에서도 distribution shift에 강건한 sample을 만드는 경향이 보고됩니다. 3D처럼 데이터가 적은 도메인에서 일반화의 표본효율을 끌어올리는 핵심 선택입니다.

**(f) Image Delighting + Double-stream Reference-Net의 "soft regularization".** 텍스처 모듈에서 reference branch의 가중치를 *얼리고* timestep을 0으로 고정해 원본 SD2.1 분포를 anchor로 삼습니다. 저자들은 이를 "rendered image distribution으로 drift하는 것을 막는 soft regularization"이라고 명시하는데, 이는 곧 **3D 렌더 데이터로 학습하면서도 실세계 이미지에 대한 일반화 성능이 유지**된다는 핵심 일반화 메커니즘입니다.

**(g) View Dropout.** 44뷰 중 매 step 6뷰를 무작위 샘플링해 학습하므로, 추론 시 학습에서 한 번도 함께 본 적 없는 뷰 조합도 합리적으로 생성합니다. 이는 dense-view inference 일반화의 토대이며, 자기-가림(self-occlusion)이 심한 임의 메시에 대한 강건성을 높입니다.

종합하면, Hunyuan3D 2.0의 일반화 전략은 (i) *표현(VAE)에서의 정보 보존*, (ii) *학습 목적함수의 안정성*, (iii) *조건 인코더의 사전학습 강도*, (iv) *분포 anchor 정규화*, (v) *학습-시간 randomization*이라는 다섯 축의 합으로 설명됩니다.

---

## 4. 향후 연구에의 영향과 고려할 점

**영향.** 첫째, vector-set 기반 latent + flow-matching DiT 조합이 3D 생성의 **새로운 사실상 표준**으로 자리잡는 데 결정적인 오픈소스 레퍼런스가 되었습니다. 후속 Hunyuan3D 2.1, Step1X-3D, MeshGen 등이 동일 골격 위에서 PBR·multimodal 확장을 시도하고 있습니다. 둘째, "shape→texture" 디커플링 파이프라인은 사용자 제작 메시에도 동일 텍스처 모듈을 적용할 수 있어 산업 파이프라인 친화적 설계의 모범이 되었습니다. 셋째, image-conditioning에 DINOv2 대형 모델을 그대로 쓴다는 디자인이 텍스처/형상 모두에서 효과적임을 보여, 3D 도메인에서 비전 파운데이션 모델 활용의 표준 레시피를 제시합니다.

**고려할 점.**

(1) **표현의 한계 극복.** SDF 기반은 closed manifold에 본질적으로 편향됩니다. TRELLIS의 SLat(sparse voxel + DINO 특징) 및 TRELLIS.2의 O-Voxel처럼 *open/non-manifold/내부 구조까지 표현 가능한 native 3D representation*을 채택하는 후속 연구가 늘 것으로 보이며, vector-set과의 하이브리드 설계가 유망합니다.

(2) **PBR 재질·리라이팅.** Hunyuan3D 2.1의 albedo+roughness+metallic 동시 합성, 3D-Aware RoPE 같은 cross-view 일관성 보강이 사실상 다음 세대 baseline입니다. RGB-only 텍스처 모델은 게임·VFX 파이프라인에서 곧 폐기 후보가 됩니다.

(3) **Artist-style 토폴로지.** Marching cubes 결과물은 dense triangle이라 LOD/리깅에 부적합합니다. MeshAnything v2, EdgeRunner, MeshRipple 같은 *auto-regressive artist-mesh 생성*과의 결합이 중요한 다음 단계입니다.

(4) **데이터 거버넌스와 도메인 편향.** Objaverse-XL 등 대규모 3D 데이터셋의 라이선스·중복·편향 문제가 상용화의 실질 장벽이 됩니다. 이 논문은 데이터 큐레이션 세부를 공개하지 않았는데, 재현성과 윤리적 검증을 위해 후속 연구에선 더 투명한 데이터 카드가 요구됩니다.

(5) **샘플링 효율과 실시간성.** Diffusion 기반의 멀티뷰 합성과 베이킹은 여전히 비싸며, Hunyuan3D-Paint-Turbo처럼 distillation/단계 축소가 후속 트렌드입니다.

(6) **조건 모달리티 확장.** 현재는 단일 이미지 위주이지만 *멀티뷰 sparse 이미지, 텍스트, 스케치, 부분 메시 보완*을 한 모델에서 처리하는 통합 조건화가 다음 도전 과제입니다.

(7) **평가지표.** ULIP/Uni3D, FID_CLIP, CMMD는 모두 2D 렌더링 기반의 간접 지표라 *형상 정합도와 물리적 정확성*을 충분히 측정하지 못합니다. 기하·재질 오류를 직접 평가하는 metric 개발이 후속 연구의 정량성에 큰 영향을 줄 것입니다.

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

| 연구(연도) | 표현(Representation) | 생성 백본 | 주요 특징/차이 |
|---|---|---|---|
| 3DShape2VecSet (Zhang et al., SIGGRAPH 2023) | Vector set (point query latent) | Diffusion (UNet/Transformer) | Hunyuan3D-ShapeVAE의 출발점. 균일 샘플만 사용해 디테일 손실 |
| Michelangelo (Zhao et al., NeurIPS 2023) | Learnable vector set | Shape-Image-Text aligned latent + diffusion | 다중 모달 정렬에 강점, 재구성 디테일은 열위 |
| CLAY (Zhang et al., SIGGRAPH 2024) | Multi-resolution VAE → vector set | DiT, 1.5B 파라미터, progressive 학습 | 최초로 대규모(1.5B) 3D-native diffusion 입증, 2K PBR 다중뷰 텍스처 |
| Direct3D (Wu et al., 2024) | Triplane latent | Latent diffusion transformer | 공간 prior 보존, 토큰 수에 민감(3072 필요) |
| Craftsman 1.5 (Li et al., 2024) | Vector set | Diffusion + interactive geometry refiner | 인터랙티브 정제 강점 |
| TRELLIS (Xiang et al., CVPR 2025 Spotlight) | **SLat**: sparse voxel + DINOv2 feature | Rectified Flow Transformer, 최대 2B | 다중 출력(NeRF/3DGS/mesh), 로컬 편집 |
| **Hunyuan3D 2.0 (Zhao et al., 2025)** | Vector set + **importance sampling**, 가변 토큰 | Flow-matching DiT (dual+single stream) | 형상-텍스처 디커플링, 멀티태스크 어텐션, oss best |
| Hunyuan3D 2.1 (Tencent, 2025.06) | 동일 + 학습 코드 전면 공개 | 동일 + PBR multi-view diffusion | albedo/roughness/metallic 동시, 3D-Aware RoPE |
| TRELLIS.2 (Microsoft, 2025) | **O-Voxel** (field-free sparse voxel) | Sparse Compression VAE + flow DiT | open/non-manifold/내부 표현, PBR 통합 |

비교 관점에서 Hunyuan3D 2.0의 위치는 다음과 같이 요약됩니다.

3DShape2VecSet→Michelangelo→CLAY로 이어진 **vector-set 계열의 최적화 정점**에 해당합니다. 표현 측면에서 importance sampling으로 3DShape2VecSet의 디테일 한계를, 가변 토큰 길이로 CLAY의 multi-resolution 아이디어를 더 단순한 형태로 흡수했습니다. 한편 같은 시기 발표된 TRELLIS는 *표현 자체를 sparse voxel + DINO feature로 바꿔* 다양한 출력 포맷(NeRF·3DGS·mesh)과 로컬 편집을 가능하게 했다는 점에서 **표현 다양성**으로 차별화하고, Hunyuan3D 2.0은 **이미지 정합 품질·텍스처 충실도·오픈소스 완성도**로 차별화합니다. 실험 표에서 Hunyuan3D 2.0이 Trellis를 ULIP/Uni3D 및 사용자 스터디에서 앞서지만, Trellis는 출력 형식 유연성에서 우위입니다. 이후 TRELLIS.2가 O-Voxel로 표현 한계까지 깬 흐름은 *vecset → sparse voxel + 학습된 latent*로의 표현 전환 가능성을 보여줍니다. 즉 2025년 현재 3D 생성 모델 경쟁의 축은 (i) latent representation의 표현력(특히 open/non-manifold/PBR), (ii) flow-matching DiT의 스케일, (iii) 멀티뷰 텍스처의 일관성과 PBR 충실도로 수렴하고 있으며, Hunyuan3D 2.0/2.1은 (ii)·(iii)에서, TRELLIS.x는 (i)에서 각각 최전선을 형성하고 있습니다.

---

## 참고 자료

1. Zhao, Z. et al. *Hunyuan3D 2.0: Scaling Diffusion Models for High Resolution Textured 3D Assets Generation*. arXiv:2501.12202v3, Feb 2025. (사용자 업로드 원문)
2. arXiv 페이지: https://arxiv.org/abs/2501.12202
3. Hugging Face Paper: https://huggingface.co/papers/2501.12202
4. Tencent-Hunyuan/Hunyuan3D-2 GitHub: https://github.com/Tencent-Hunyuan/Hunyuan3D-2
5. Hunyuan3D 2.1 (Tencent, 2025): *Hunyuan3D 2.1: From Images to High-Fidelity 3D Assets with Production-Ready PBR Material*, arXiv:2506.15442. https://arxiv.org/abs/2506.15442
6. Xiang, J. et al. *Structured 3D Latents for Scalable and Versatile 3D Generation* (TRELLIS), arXiv:2412.01506; CVPR 2025. https://trellis3d.github.io
7. Microsoft TRELLIS GitHub: https://github.com/microsoft/TRELLIS
8. TRELLIS.2 프로젝트 페이지: https://microsoft.github.io/TRELLIS.2/ (arXiv:2512.14692)
9. Zhang, L. et al. *CLAY: A Controllable Large-scale Generative Model for Creating High-quality 3D Assets*, ACM TOG / SIGGRAPH 2024. https://dl.acm.org/doi/10.1145/3658146
10. Zhang, B. et al. *3DShape2VecSet*, ACM TOG / SIGGRAPH 2023. https://dl.acm.org/doi/abs/10.1145/3592442
11. Zhao, Z. et al. *Michelangelo*, NeurIPS 2023. (Hunyuan3D 2.0 참고문헌 [118])
12. Lipman, Y. et al. *Flow Matching for Generative Modeling*, arXiv:2210.02747, 2022.
13. Esser, P. et al. *Scaling Rectified Flow Transformers for High-Resolution Image Synthesis*, ICML 2024. (Hunyuan3D 2.0 참고문헌 [24])
14. Oquab, M. et al. *DINOv2: Learning Robust Visual Features without Supervision*, 2023. (Hunyuan3D 2.0 참고문헌 [64])
15. Black Forest Labs, *FLUX*, 2024. https://github.com/black-forest-labs/flux
16. The Decoder, "CLAY creates detailed 3D objects from text and images", Jul 2024. https://the-decoder.com/clay-creates-detailed-3d-objects-from-text-and-images/
17. Moonlight Literature Review: https://www.themoonlight.io/en/review/hunyuan3d-20-scaling-diffusion-models-for-high-resolution-textured-3d-assets-generation

본 답변에서 논문 내부 수치(IoU, ULIP/Uni3D, CMMD/FID_CLIP/CLIP-score, 사용자 스터디 백분위, 학습 하이퍼파라미터)와 구조적 사실은 모두 업로드된 원문에서 직접 확인했습니다. 후속 모델(2.1, TRELLIS, TRELLIS.2)의 차별점은 위 외부 출처에서 교차 확인했습니다. 그 외에 100% 확신이 서지 않는 세부(예: 정확한 학습 데이터 규모·라이선스, 비공개 "Shape Model 1/2/3"의 정체)는 추측을 피하고 비공개로 명시했습니다.
