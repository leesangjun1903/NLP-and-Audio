# Hunyuan3D 2.1: From Images to High-Fidelity 3D Assets with Production-Ready PBR Material

## 1. 핵심 주장 및 주요 기여 요약

**Hunyuan3D 2.1**은 Tencent Hunyuan 팀이 2025년 6월 arXiv에 공개한 논문(2506.15442v1)으로, **단일 이미지로부터 PBR(Physically-Based Rendering) 머티리얼이 적용된 고품질 3D 에셋을 생성하는 오픈소스 시스템**입니다. 본질적으로는 시스템 논문이자 튜토리얼 형태의 보고서로, 데이터 처리부터 학습·평가까지 전체 파이프라인을 공개하는 데 의의가 있습니다.

**주요 기여 4가지**:

1. **모듈식 2단계 파이프라인**: **Hunyuan3D-DiT**(형상 생성) + **Hunyuan3D-Paint**(PBR 텍스처 합성)를 분리해 사용자 유연성(텍스처 없는 메쉬만 생성, 외부 메쉬에 텍스처 적용 등)을 확보.
2. **Hunyuan3D-ShapeVAE**: 메쉬 표면의 importance sampling과 **가변 토큰 길이(maximum 3072)** 학습 전략으로 sharp edge와 미세 디테일을 모두 보존.
3. **PBR 머티리얼 직접 생성**: Disney Principled BRDF를 따라 albedo, metallic, roughness 맵을 다중 뷰에서 동시 생성하며, **Spatial-Aligned Multi-Attention**, **3D-Aware RoPE**, **Illumination-Invariant Training** 세 가지 새로운 기법 도입.
4. **완전 오픈소스화**: 데이터 처리, 학습 파이프라인, 모델 가중치를 모두 공개하여 PBR 텍스처가 적용된 3D 에셋 생성 분야 최초의 완전 오픈소스 솔루션을 표방.

---

## 2. 문제·제안 방법·모델 구조·성능·한계

### 2.1 해결하고자 하는 문제

논문이 명시한 문제 의식은 다음과 같습니다.

- 2D 이미지·비디오 생성은 Stable Diffusion, LLaMA, HunyuanVideo, Wan 2.1과 같은 **개방형 파운데이션 모델 생태계**가 형성되어 있으나, 3D 영역은 **분절되어 있고 확장 가능한 오픈소스 기반이 부족**.
- 기존 방법들은 (a) 형상 디테일 부족, (b) 텍스처가 RGB 컬러에 그쳐 **PBR 머티리얼 부재** → 게임/영화/산업 디자인의 실제 렌더링 파이프라인에 사용 불가, (c) 데이터 수집·전처리·학습의 복잡성 때문에 일반 사용자 접근이 어려움.

### 2.2 제안 방법 (수식 포함)

#### (1) 데이터 전처리 — Watertight 변환

defective 메쉬에서 IGL 라이브러리로 SDF(Signed Distance Field)를 만들고 marching cubes로 watertight 메쉬를 추출합니다.

```math
\text{SDF}(q) = \underbrace{\text{distance\_to\_mesh}(q, V, F)}_{\text{nearest surface distance}} \cdot \underbrace{\text{sign}(\omega(q))}_{\text{inside/outside sign}}
```

여기서 $V$는 vertex, $F$는 face, $\omega(q)$는 generalized winding number이며 $\omega \approx 1$이면 내부, $\omega \approx 0$이면 외부로 분류합니다 (논문에서는 $\omega > 0.5$ 임계).

#### (2) Hunyuan3D-ShapeVAE 학습 손실

VecSet 기반 인코더-디코더로, 표면에서 균등 샘플링한 점 $P_u \in \mathbb{R}^{M \times 3}$과 importance sampling한 점 $P_i \in \mathbb{R}^{N \times 3}$을 입력해 latent $Z_s$를 만든 뒤 SDF를 복원합니다. 학습 목적식:

$$\mathcal{L}_r = \mathbb{E}_{x \in \mathbb{R}^3}\bigl[\text{MSE}(\mathcal{D}_s(x|Z_s),\ \text{SDF}(x))\bigr] + \gamma \mathcal{L}_{KL} $$

여기서 $\gamma$는 KL 가중치, $\mathcal{L}_{KL}$은 latent space를 컴팩트하고 연속적으로 만들기 위한 KL-divergence 손실입니다. **다중 해상도 학습 전략**으로 latent 토큰 시퀀스 길이를 동적으로 변화시키며 최대 3072까지 확장합니다.

#### (3) Hunyuan3D-DiT — Flow Matching

이미지 조건 $c$ 하에서 latent 토큰 시퀀스를 생성하는 flow-based diffusion model. Conditional optimal transport schedule의 affine path:

$$x_t = (1-t)\, x_0 + t\, x_1, \qquad u_t = x_1 - x_0$$

학습 목적식:

$$\mathcal{L} = \mathbb{E}_{t, x_0, x_1}\bigl[\| u_\theta(x_t, c, t) - u_t \|_2^2\bigr] $$

여기서 $t \sim \mathcal{U}(0,1)$이고, 추론 시에는 첫 번째 차수 Euler ODE solver로 $x_1$을 구합니다. 조건 인코더로는 **DINOv2 Giant** ($518 \times 518$ 입력)를 사용하며, 21개의 Transformer layer를 stack하고 각 layer에 dimension-concatenation 방식 skip connection, cross-attention(이미지 조건 주입), **MoE layer**(표현력 확장)를 둡니다.

#### (4) Hunyuan3D-Paint — PBR 다중 뷰 확산

Disney Principled BRDF를 따라 **albedo / metallic-roughness(MR)** 두 분기를 가진 dual-branch UNet으로 다중 뷰 PBR 맵을 동시 생성합니다. 핵심 모듈 3가지:

- **Spatial-Aligned Multi-Attention Module**: self / multi-view / reference attention을 albedo와 MR 분기에 병렬 배치하고, **albedo reference attention의 출력을 MR 분기로 직접 전파**해 두 맵 간 공간 정렬 보장.
- **3D-Aware RoPE**: 3D 좌표 볼륨을 다운샘플링해 UNet 계층과 정렬되는 다중 해상도 3D 좌표 인코딩을 만들고 hidden state에 가산 융합 → **이웃 뷰 간 텍스처 seam과 ghosting 완화**.
- **Illumination-Invariant Training Strategy**: 동일 객체를 다른 조명으로 렌더링한 두 세트를 사용해 일관성 손실(consistency loss)을 부과 → **light-free albedo**와 정확한 MR 맵 생성.

초기화는 Stable Diffusion 2.1의 Zero-SNR 체크포인트, 학습률 $5 \times 10^{-5}$의 AdamW, warm-up 2000 steps, 약 **180 GPU-days** 소요.

### 2.3 모델 구조 요약

| 구성 요소 | 역할 | 핵심 설계 |
|---|---|---|
| Hunyuan3D-ShapeVAE | 메쉬 ↔ latent 토큰 | VecSet 기반, FPS query, 가변 길이(max 3072) |
| Hunyuan3D-DiT | 이미지 → latent 토큰 | DINOv2-Giant + 21-layer Transformer + MoE + Flow Matching |
| Hunyuan3D-Paint | 메쉬 + 이미지 → PBR maps | Dual-branch UNet, Spatial-Aligned Multi-Attention, 3D-Aware RoPE |

### 2.4 성능 향상

**형상 생성** (Table 1, 자체 보고치):

| 모델 | ULIP-T ↑ | ULIP-I ↑ | Uni3D-T ↑ | Uni3D-I ↑ |
|---|---|---|---|---|
| Michelangelo | 0.0752 | 0.1152 | 0.2133 | 0.2611 |
| Craftsman 1.5 | 0.0745 | 0.1296 | 0.2375 | 0.2987 |
| TripoSG | 0.0767 | 0.1225 | 0.2506 | 0.3129 |
| Step1X-3D | 0.0735 | 0.1183 | 0.2554 | 0.3195 |
| Trellis | 0.0769 | 0.1267 | 0.2496 | 0.3116 |
| Direct3D-S2 | 0.0706 | 0.1134 | 0.2346 | 0.2930 |
| **Hunyuan3D-DiT** | **0.0774** | **0.1395** | **0.2556** | **0.3213** |

**텍스처 생성** (Table 2): Hunyuan3D-Paint가 CLIP-FID 24.78, CMMD 2.191, CLIP-I 0.9207, LPIPS 0.1211로 SyncMVD-IPA, TexGen, Hunyuan3D-2.0 대비 모두 우위.

### 2.5 한계

논문은 튜토리얼·시스템 보고서 성격이라 한계 분석을 명시적으로 거의 다루지 않습니다. 본 논문 텍스트로부터 확인 가능하거나 합리적으로 추론 가능한 한계는 다음과 같습니다.

- **데이터 규모**: 형상 100K+, 텍스처 70K+로, Trellis의 500K(공개 발표 기준)나 TripoSG가 강조하는 large-scale 데이터에 비해 작음.
- **학습 비용**: Paint만 약 180 GPU-days로 재현 비용이 높음.
- **평가의 한계**: ULIP/Uni3D는 점군-텍스트/이미지 임베딩 유사도이며 메쉬 토폴로지 품질, manifold/non-manifold 처리, open surface 등에 대한 직접 측정이 아님. 사용자 선호도 평가 결과는 논문 본문에서 정량 표로 구체화되지 않음.
- **표현 방식의 제약**: SDF + marching cubes 기반이므로 **얇은 개방형 표면(천, 머리카락 등)이나 토폴로지가 복잡한 객체**에서는 본질적 표현력 한계 존재(이는 TRELLIS.2가 명시적으로 지적하는 occupancy/SDF-based 방법의 공통 약점).
- **PBR 분리의 ill-posed성**: 단일 이미지에서 albedo/metallic/roughness를 동시 추정하는 것은 본질적으로 ill-posed이며, illumination-invariant 학습이 완전한 해결은 아님.

---

## 3. 모델의 일반화 성능 향상 가능성

논문이 일반화에 직접적으로 기여하는 설계와, 이를 더 끌어올릴 수 있는 가능성을 정리합니다.

### 3.1 논문이 일반화에 기여하는 설계 요소

1. **DINOv2-Giant 조건 인코더 (518×518)**
   DINOv2는 자기지도 학습으로 학습된 **도메인 일반적 비전 백본**이며, OOD 이미지(만화·실사·AI 생성 이미지 혼재)에서도 강건한 표현을 제공합니다. TripoSG도 같은 이유로 DINOv2를 채택해 다양한 이미지 스타일에 대한 generalization을 강조합니다.

2. **Flow Matching + MoE**
   Flow matching은 noise schedule에 덜 민감하고 학습이 안정적이며, MoE layer는 파라미터를 늘리지 않고도 도메인 다양성(가구, 캐릭터, 기계, 자연물 등)에 대해 **conditional capacity**를 확장하는 효과가 있습니다.

3. **가변 토큰 길이(최대 3072) 학습**
   객체별 기하 복잡도에 맞춰 latent 길이를 동적으로 사용함으로써, 단순 객체에서는 효율을, 복잡 객체에서는 표현력을 확보 → **다양한 복잡도의 객체에 대한 일반화**.

4. **Importance + Uniform 이중 샘플링**
   sharp edge가 많은 객체(기계 부품)와 부드러운 곡면 객체(인형, 동물) 모두를 학습 분포에 균형 있게 노출시켜 카테고리 간 편향을 줄이는 효과.

5. **Illumination-Invariant Training**
   같은 객체의 서로 다른 조명 렌더 쌍에 대해 일관성 손실을 부과하므로, **테스트 시 입력 이미지의 조명 변동에 대한 강건성** 확보 → 실제 사진 입력에서 일반화 성능 개선.

6. **3D-Aware RoPE**
   다중 해상도의 3D 좌표를 attention에 주입함으로써 학습 시 보지 못한 카메라 배치·뷰 개수에 대해서도 **공간적 일관성**을 유지할 수 있는 inductive bias 제공.

### 3.2 일반화 성능을 더 끌어올릴 수 있는 방향

논문이 직접 제안한 것은 아니지만, 본 구조의 약점을 토대로 합리적으로 추론할 수 있는 방향입니다.

- **데이터 스케일링**: 100K → 수백만 단위로 데이터 확장 (Trellis-500K, Objaverse-XL 활용). 다만 데이터 라벨 품질이 핵심.
- **표현 방식 보강**: SDF 단일 표현의 한계를 극복하기 위해 sparse voxel(O-Voxel/SLat 계열) 또는 hybrid 표현 도입.
- **합성 데이터 + 실세계 데이터 혼합**: 현재는 주로 Objaverse 계열 합성 데이터 → 실사진 페어로 미세조정 시 in-the-wild 일반화 향상 기대.
- **PBR map의 명시적 disentanglement loss 강화**: 현재 illumination consistency 외에 reflection, normal 등 추가 supervision을 결합하면 ill-posed성 완화 가능.
- **카테고리/도메인 적응을 위한 LoRA·adapter 모듈식 미세조정**: 산업 도메인(가구, 의류, 기계 부품)별 소규모 adapter로 일반화 향상.

---

## 4. 향후 연구에 미치는 영향과 고려할 점

### 4.1 영향

1. **3D 생성 분야의 “Stable Diffusion 모멘트” 가속**: 데이터 파이프라인부터 가중치까지 전부 공개되어, 향후 follow-up 연구의 baseline 및 fine-tuning 출발점이 될 가능성이 큼.
2. **PBR 표준화**: 단순 RGB 텍스처가 아닌 albedo/metallic/roughness 동시 생성을 표준 스코프로 끌어올렸음. 이후 연구는 PBR을 default 평가 항목으로 고려해야 할 가능성.
3. **모듈성에 따른 산업 적용성**: 형상-텍스처 분리는 게임/3D 프린팅/디자인 워크플로의 기존 자산 재활용과 결합되기 쉬움.
4. **벤치마크 인플레이션 우려**: 자체 보고 수치만으로 우열을 단정하기 어렵고, 다른 모델(Trellis, TripoSG 등) 측에서 수행한 비교와 다를 수 있음 — 향후 표준 벤치마크의 필요성을 부각.

### 4.2 향후 연구 시 고려할 점

- **공정한 평가 프로토콜**: ULIP/Uni3D는 의미 정렬 척도이며 메쉬 품질의 직접 척도가 아님. Chamfer Distance, F-Score, normal consistency, manifoldness, watertightness 등 **기하 품질 지표**를 함께 보고해야 함.
- **사용자 평가의 통계적 신뢰성**: human preference study는 평가자 수, instruction, 표본 다양성을 사전 등록(preregistration)하는 관행이 필요.
- **재현성 및 학습 비용**: 180 GPU-days 수준의 학습은 학술 연구실에서 어려움 → distillation, lightweight backbone, EMA-based fine-tuning 같은 **저비용 재현** 연구가 함께 진행되어야 함.
- **윤리·저작권**: Objaverse-XL은 다양한 출처 자산을 포함하므로 라이선스 필터링과 출처 추적이 중요한 연구 주제로 부각.
- **에셋 호환성**: UV 매핑, 토폴로지 정리, retopology 등 실제 게임 엔진 워크플로와의 인터페이스는 여전히 후처리에 의존하므로, **end-to-end 산업용 메쉬 생성**이 다음 과제.

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

다음 표는 본 논문 및 검색된 자료에서 확인된 사실 기반으로 정리한 것입니다. 정확한 수치는 각 논문의 자체 보고치이므로 직접 비교는 주의가 필요합니다.

| 연구 (연도) | 표현 방식 | 생성 백본 | 텍스처/머티리얼 | 차별점 |
|---|---|---|---|---|
| **3DShape2VecSet** (Zhang et al., SIGGRAPH 2023) | VecSet latent + neural field SDF | latent diffusion | 없음 | 고정 길이 latent set 표현의 시초 |
| **Michelangelo** (2023, 본 논문 인용 [8]) | latent diffusion | transformer | 없음 | 이미지-3D 정렬 학습 강조 |
| **CLAY** (Zhang et al., ACM TOG 2024) | multi-resolution VAE + Vecset latent | DiT (1.5B) | 다중 뷰 PBR (diffuse/roughness/metallic, 2K) | 본 분야의 첫 대규모 시도, 비공개 |
| **Trellis** (Xiang et al., arXiv 2412.01506, CVPR'25 Spotlight) | **SLat (sparse structured latent)** | Rectified Flow Transformer (최대 2B) | Radiance Field/3D Gaussian/메쉬 다중 출력 | 500K 데이터셋, 출력 형식 유연 |
| **TripoSG** (Li et al., arXiv 2502.06608, 2025) | VecSet latent SDF | **Rectified Flow Transformer (1.5B)** | 형상 중심(별도 텍스처 모듈 없음) | 대규모 RF 모델, 스타일 일반화 강조 |
| **Step1X-3D** (2025, 본 논문 비교 대상) | latent diffusion + texture | — | PBR | 오픈소스 |
| **Direct3D-S2** (2025, 본 논문 비교 대상) | sparse 3D representation | — | — | 본 논문에서 비교 대상 |
| **Hunyuan3D 2.1** (Tencent, 2025) | VecSet latent SDF (가변 길이 max 3072) | Flow Matching DiT (21-layer) + MoE | **dual-branch PBR (albedo + MR)**, 3D-aware RoPE, illumination-invariant | **PBR end-to-end 완전 오픈소스** |
| **TRELLIS.2** (Microsoft, 2025/26 후속) | **O-Voxel + Sparse Compression VAE** | structured latent | PBR (Base Color/Roughness/Metallic/Opacity) | open surface·non-manifold·내부 구조까지 처리, field-free |

**비교 관점에서의 위치**: Hunyuan3D 2.1은 (a) **표현 방식**으로는 CLAY/TripoSG와 같은 VecSet+SDF 계열을 따르며, (b) **백본**은 Flow Matching DiT로 TripoSG/Trellis와 흐름을 공유하고, (c) **PBR 머티리얼 직접 생성을 완전 오픈소스로 제공**한다는 점에서 CLAY(비공개)와 차별화됩니다. 다만 Trellis 계열의 **sparse structured representation**이 표현력 면에서 유리하다는 후속 흐름(TRELLIS.2)이 등장하고 있어, **표현 방식의 다음 세대 전환**이 향후 일반화 성능의 분기점이 될 가능성이 높습니다.

---

## 참고한 자료 (출처)

1. **Hunyuan3D 2.1 본 논문 (PDF, 사용자 업로드)**: Tencent Hunyuan, "Hunyuan3D 2.1: From Images to High-Fidelity 3D Assets with Production-Ready PBR Material," arXiv:2506.15442v1 [cs.CV], 18 Jun 2025.
2. **TripoSG 논문**: Li et al., "TripoSG: High-Fidelity 3D Shape Synthesis using Large-Scale Rectified Flow Models," arXiv:2502.06608 (v1: Feb 2025, v3: Mar 2025). https://arxiv.org/abs/2502.06608
3. **TripoSG GitHub**: https://github.com/VAST-AI-Research/TripoSG
4. **TRELLIS 논문**: Xiang et al., "Structured 3D Latents for Scalable and Versatile 3D Generation," arXiv:2412.01506, CVPR 2025 Spotlight. https://huggingface.co/papers/2412.01506
5. **TRELLIS GitHub & 프로젝트 페이지**: https://github.com/microsoft/TRELLIS, https://microsoft.github.io/TRELLIS/
6. **TRELLIS.2 프로젝트 페이지**: https://microsoft.github.io/TRELLIS.2/
7. **3DShape2VecSet (ACM TOG 2023)**: Zhang, Tang, Nießner, Wonka. https://dl.acm.org/doi/10.1145/3592442, arXiv:2301.11445
8. **CLAY 논문**: Zhang et al., "CLAY: A Controllable Large-scale Generative Model for Creating High-quality 3D Assets," ACM TOG 2024, arXiv:2406.13897. https://arxiv.org/html/2406.13897v1

---

**정확도에 관한 주의**: 위 분석은 본 논문에 명시된 내용과 검색된 후속/관련 연구 정보에 기반합니다. 표 1·2의 수치, 각 연구의 파라미터 규모(예: TripoSG 1.5B, Trellis 최대 2B), 학습 GPU-days(Paint 약 180 GPU-days)는 각 논문의 자체 보고치입니다. 다른 연구가 동일 셋업에서 재측정한 결과와 다를 수 있으며, 본 논문에는 한계 분석 절이 따로 없으므로 §2.5의 일부 항목은 본문 사실(SDF 사용, 데이터 규모 등)에서 합리적으로 추론한 내용임을 명시했습니다.
