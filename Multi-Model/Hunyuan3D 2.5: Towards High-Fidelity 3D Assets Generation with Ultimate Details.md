# Hunyuan3D 2.5: Towards High-Fidelity 3D Assets Generation with Ultimate Details

논문의 본문(arXiv:2506.16504v1)과 관련 최신 연구를 함께 검토하여 작성한 분석입니다. 본문에 명시되지 않은 내용은 분명하게 구분해 표시했습니다.

---

## 1. 핵심 주장과 주요 기여 (Executive Summary)

Hunyuan3D 2.5는 Tencent Hunyuan3D 팀이 발표한 이미지-투-3D 생성 모델군으로, 전작 Hunyuan3D 2.0/2.1의 **2단계 파이프라인(형상 생성 → 텍스처 생성)** 을 계승하면서 두 단계 모두에서 큰 개선을 이룬 기술 보고서입니다. 핵심 기여는 다음 네 가지로 요약됩니다.

1. **LATTICE라는 새 형상 파운데이션 모델 도입**: 최대 **10B 파라미터** 규모로 확장(scale-up)되어, 데이터·모델·연산량 증가에 따라 안정적으로 품질이 향상된다고 보고합니다. 손가락 개수, 자전거 바퀴 패턴, 장면 안의 작은 그릇 같은 미세 디테일 생성, 그리고 "예리한 에지(sharp edges)와 매끈한 표면(smooth surfaces)"의 양립을 강조합니다.
2. **PBR(Physically-Based Rendering) 멀티뷰 텍스처 생성 프레임워크**: 단일 RGB가 아니라 **albedo + MR(metallic-roughness) + normal** 채널을 동시에 생성하며, 이를 가능하게 하는 **Dual-Channel Attention(공유 attention mask)** 을 도입했습니다.
3. **Dual-Phase Resolution Enhancement 학습 전략**: 1단계에서 6-view 512×512로 일관성 학습, 2단계에서 zoom-in 방식으로 고해상도 디테일을 학습 → 추론 시 768×768까지 가능.
4. **종합 평가**: 정량 지표(ULIP, Uni3D, FID, CLIP-FID, CMMD, LPIPS)와 사용자 스터디에서 오픈소스/상용 베이스라인 대비 우위를 보고. 사용자 스터디에서는 image-to-3D 기준 **상용 모델 1 대비 72.25% 승률**을 기록했다고 제시합니다.

---

## 2. 자세한 설명: 문제, 방법, 모델 구조, 성능, 한계

### 2.1 해결하고자 하는 문제

논문이 명시적으로 지적하는 미해결 문제는 다음과 같습니다.

- **형상 측면**: 기존 모델(Hunyuan3D 2.0, TripoSG, Trellis 등)은 복잡한 객체에서 **fine-grained detail, sharp edges, smooth surface 세 가지를 동시에** 만족시키지 못합니다. 디테일을 키우면 표면이 거칠어지고, 표면을 매끄럽게 하면 디테일이 무뎌지는 trade-off가 존재합니다.
- **텍스처 측면**: (i) 멀티뷰 일관성 부족으로 baking 시 솔기/아티팩트 발생, (ii) **오픈소스 PBR 솔루션 부재** — 기존 RGB 텍스처는 사실적 렌더링·재조명에 부적합. 또한 albedo와 MR 같이 도메인 갭이 큰 채널 간 **공간적 정합(spatial alignment)** 유지가 어렵습니다.

### 2.2 모델 구조 (Pipeline Overview)

전체 파이프라인은 다음과 같이 구성됩니다(논문 Fig. 3).

$$
\text{Input Images} \;\to\; \text{Image Preprocessing} \;\to\; \text{Shape Generation (LATTICE)} \;\to\; \text{Mesh Post-processing} \;\to\; \text{Texture Generation (Hunyuan3D-Paint-PBR)} \;\to\; \text{Output Mesh}
$$

#### (1) Shape Generation – LATTICE

논문은 LATTICE를 "단일 이미지 또는 4-view 멀티뷰로부터 고해상도 메시를 생성하는 대규모 디퓨전 모델"이라고 설명합니다. **단, LATTICE의 내부 아키텍처(블록 구성, latent 형태, 손실함수 등)에 대한 수식·세부 사양은 본 보고서에 거의 기술되어 있지 않습니다.** 보고서는 계열(family) 가운데 최대 모델이 10B 파라미터이며, 추론 가속을 위해 **guidance distillation과 step distillation** 을 사용한다고만 명시합니다. 따라서 LATTICE의 정확한 수식은 본문에 부재하며, 추측해서 적지 않겠습니다.

(맥락적으로) 같은 계열이 따르는 vecset 디퓨전 패러다임은 일반적으로 다음과 같이 동작합니다 — Zhang et al., 3DShape2VecSet (SIGGRAPH 2023). 3D 형상을 학습 가능한 latent vector set $\mathbf{Z}=\{\mathbf{z}_i\}_{i=1}^{N}$ 으로 압축하는 cross-attention 기반 오토인코더와, 이 latent set 위에서 동작하는 디퓨전 모델로 구성됩니다.

$$
\mathcal{L}_{\text{diff}} = \mathbb{E}_{\mathbf{Z}_0, \boldsymbol{\epsilon}, t}\left[\, \big\| \boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\mathbf{Z}_t,\, t,\, c)\big\|_2^2\,\right]
$$

여기서 $c$ 는 이미지 조건, $\mathbf{Z}_t$ 는 timestep $t$ 에서의 노이즈가 섞인 latent set입니다. 이 부분은 vecset 계열의 일반적인 정식화이며, **LATTICE가 정확히 이 형태를 따르는지에 대한 본문 명시는 없습니다.**

#### (2) Texture Generation – Hunyuan3D-Paint-PBR

이 부분은 본문에 수식과 함께 명시되어 있습니다. 핵심은 **albedo, MR, normal** 세 채널에 대한 학습 가능한 임베딩 $\mathbf{E}_{\text{albedo}},\, \mathbf{E}_{\text{mr}},\, \mathbf{E}_{\text{normal}} \in \mathbb{R}^{16\times 1024}$ 을 도입하고, 이를 cross-attention으로 각 분기에 주입하는 것입니다.

**Dual-Channel Attention의 핵심 수식 (논문 Eq. 1, 2)**: 본 논문이 발견한 멀티채널 misalignment의 원인은 분기마다 attention mask가 다르다는 점입니다. 따라서 **basecolor(albedo) 분기에서 계산된 attention mask를 다른 분기에 공유** 하고, value만 분기별로 다르게 두는 방식을 채택합니다.

$$
M_{\text{attn}} \;=\; \text{Softmax}\!\left(\frac{Q_{\text{albedo}}\, K_{\text{ref}}^{\top}}{\sqrt{d}}\right) 
$$

$$
z^{\text{new}}_{\text{albedo}} \;=\; z_{\text{albedo}} \;+\; \text{MLP}_{\text{albedo}}\!\left[\, M_{\text{attn}} \cdot V_{\text{albedo}} \,\right]
$$

$$
z^{\text{new}}_{\text{MR}} \;=\; z_{\text{MR}} \;+\; \text{MLP}_{\text{MR}}\!\left[\, M_{\text{attn}} \cdot V_{\text{MR}} \,\right] 
$$

이 설계의 직관은 albedo가 RGB 공간상 reference image와 가장 의미적으로 가까우므로 **가장 신뢰할 수 있는 mask 추정 분기로 활용**하고, 다른 채널은 그 mask를 따라가게 함으로써 픽셀-수준 정합을 유지하는 것입니다. 추가로 학습 시 **illumination-invariant consistency loss**(He et al., 2025의 MaterialMVP)를 도입해 재질과 조명을 디스인탱글한다고 보고합니다.

**3D-Aware RoPE**: 멀티뷰 일관성을 위해 RomanTex(Feng et al., 2025)에서 제안된 3D-aware Rotary Positional Embedding을 그대로 계승합니다.

**Dual-Phase Resolution Enhancement**:
- Phase 1: 6-view 512×512 멀티뷰로 일관성과 기본 텍스처-지오메트리 대응 학습
- Phase 2: reference와 멀티뷰를 무작위로 zoom-in 하여 메모리 제약 없이 고주파 디테일 학습
- Inference: 768×768까지의 멀티뷰를 UniPC 샘플러(Zhao et al., 2023)로 가속 생성

### 2.3 성능 (Quantitative & Qualitative)

**Shape generation** (논문 Table 1):

| 모델 | ULIP-T↑ | ULIP-I↑ | Uni3D-T↑ | Uni3D-I↑ |
|---|---|---|---|---|
| Michelangelo | 0.0752 | 0.1152 | 0.2133 | 0.2611 |
| Craftsman 1.5 | 0.0745 | 0.1296 | 0.2375 | 0.2987 |
| Trellis | 0.0769 | 0.1267 | 0.2496 | 0.3116 |
| Commercial Model 1 | 0.0741 | 0.1308 | 0.2464 | 0.3106 |
| Commercial Model 2 | 0.0746 | 0.1284 | 0.2516 | 0.3131 |
| Hunyuan3D 2.0 | 0.0771 | 0.1303 | 0.2519 | 0.3151 |
| **Hunyuan3D 2.5** | **0.07853** | 0.1306 | **0.2542** | **0.3151** |

ULIP-T, Uni3D-T, Uni3D-I에서 1위, ULIP-I에서는 Commercial Model 1(0.1308)이 근소하게 앞섭니다. 저자들도 "이러한 지표가 모델 능력을 충분히 반영하지 못한다"고 직접 인정하며, **시각적 비교(Fig. 6)와 사용자 스터디**가 더 결정적이라고 주장합니다.

**Texture generation** (논문 Table 2):

| 방법 | CLIP-FID↓ | FID↓ | CMMD↓ | CLIP-I↑ | LPIPS↓ |
|---|---|---|---|---|---|
| Text2Tex | 31.83 | 187.7 | 2.738 | – | 0.1448 |
| SyncMVD | 29.93 | 189.2 | 2.584 | – | 0.1411 |
| Paint-it | 33.54 | 179.1 | 2.629 | – | 0.1538 |
| Paint3D | 26.86 | 176.9 | 2.400 | 0.8871 | 0.1261 |
| TexGen | 28.23 | 178.6 | 2.447 | 0.8818 | 0.1331 |
| **Hunyuan3D 2.5** | **23.97** | **165.8** | **2.064** | **0.9281** | **0.1231** |

다섯 지표 모두에서 1위. **사용자 스터디(Fig. 8)** 에서는 image-to-3D 기준 vs Commercial Model 1에서 19.48% Inferior / 8.27% Same / 72.25% Better, vs Commercial Model 2에서 32.01%/18.63%/49.36%, vs Commercial Model 3에서 26.14%/12.99%/60.87%로 모든 비교에서 우위를 보고합니다.

### 2.4 한계 (논문 명시 + 분석적 관찰)

논문에는 **공식적인 "Limitations" 섹션이 없습니다.** 보고서 형식의 기술 리포트에 가까운 글이라는 점을 감안해, 본문에서 식별 가능한 한계와 분석적으로 식별 가능한 한계를 구분합니다.

- **본문에서 직접 시사된 한계**:
  - 저자들 스스로 ULIP/Uni3D 같은 메트릭이 모델 능력을 충분히 반영하지 못한다고 인정합니다(§3.1).
  - 풀 고해상도 멀티뷰 학습은 메모리 제약 때문에 불가능해, zoom-in 우회를 도입했다고 명시합니다(§2.2).
- **본문에 부재하여 검증 어려운 한계 (분석적 관찰)**:
  - LATTICE 자체의 아키텍처·하이퍼파라미터·VAE 구조·학습 손실에 대한 디테일이 거의 공개되지 않아 **재현성(reproducibility) 정보가 제한적**입니다.
  - **제거 실험(ablation)** 이 부재합니다 — 예: dual-channel attention vs 독립 attention의 정량 비교, dual-phase 학습 vs 단일 phase 비교, 모델 사이즈별(예: 1B vs 10B) 정량 비교가 표로 제시되지 않습니다.
  - 상용 모델 1·2·3의 정체가 익명화되어 있어 외부 검증이 어렵습니다.
  - **속도/메모리 코스트** 의 정량 보고가 없습니다(특히 10B 모델 추론 시간/GPU 메모리).

---

## 3. 모델의 일반화 성능 향상 가능성 (집중 분석)

이 항목은 사용자께서 중점적으로 요청하셨으므로, **본문 근거 → 추론적 함의** 순으로 분리해 정리합니다.

### 3.1 본문이 명시적으로 보고하는 일반화 관련 근거

1. **데이터·모델·연산 스케일링 안정성**: §2.1에서 "this new model exhibits stable improvement when scaling up the model"이라고 보고. 이는 LLM에서 관찰된 **scaling law 유사 현상이 3D shape diffusion에도 성립**할 가능성을 시사합니다. 10B까지 키워도 collapse 없이 단조 개선된다는 점은, 더 큰 데이터셋과 모델로 확장 시 **OOD(out-of-distribution) 객체에서의 일반화 여지**를 함축합니다.
2. **In-the-wild 입력에서의 정성 우수성**: §3에서 "diverse range of in-the-wild input images"에 대해 평가했다고 명시. Fig. 6의 비교 사례(군용 차량, 메카, 핑크 자판기, 마법사 캐릭터, 도시 주택)는 **카테고리 간 도메인 다양성**이 큰 입력이고, 여기서 Trellis/Hunyuan3D 2.0/상용 모델 대비 일관되게 우수하다는 점은 카테고리 일반화의 간접 증거입니다.
3. **단일 이미지 + 4-view 듀얼 모드**: 단일 뷰가 모호한 경우 멀티뷰 입력으로 fall back할 수 있어, **입력 모달리티 측면의 일반화** 폭이 넓습니다.
4. **PBR로의 출력 일반화**: 텍스처를 RGB가 아니라 albedo/MR/normal로 분해함으로써, **새로운 조명 환경(relighting)·새로운 렌더링 엔진** 으로 자산을 옮길 때의 일반화가 구조적으로 보장됩니다. 이는 게임 엔진/언리얼·유니티/필름 파이프라인 호환성을 높입니다.

### 3.2 일반화에 작용할 수 있는 구조적 강점 (분석적)

- **Vecset 표현의 카테고리 비종속성**: 3DShape2VecSet 계열 latent는 voxel/triplane 같은 그리드 표현보다 **객체 토폴로지에 둔감**합니다(Zhang et al., 2023). LATTICE도 이 계열을 따른다고 추정되어 — 본문이 명시 부재하므로 단정은 피합니다 — 토폴로지 다양성에 강할 가능성이 있습니다.
- **공유 attention mask**: dual-channel attention은 albedo가 reference와 가장 가까운 도메인이라는 사전(prior)을 코드에 박는 것으로, **도메인 갭이 큰 채널을 새로 추가**할 때(예: emission, height, anisotropy)에도 같은 원리로 일반화하기 좋은 설계 패턴입니다.
- **Distillation 기반 가속**: guidance/step distillation으로 추론 비용을 낮추어 **실시간 어플리케이션**(인터랙티브 모델링, 게임 내 자산 생성)으로의 일반화 가능성이 열립니다 — 이 방향은 같은 팀의 후속 연구 FlashVDM(Lai et al., 2025, ICCV'25 Highlight)에서 vecset 디퓨전을 1초 내 실행하는 방향으로 이미 확장됨이 확인됩니다.

### 3.3 일반화의 잠재적 취약점 (본문 부재로 추정 영역임을 명시)

- **데이터 큐레이션 의존성**: §2.1에서 "extensive and high-quality 3D dataset"라고만 표현되어, 데이터 분포 편향이 다음과 같은 OOD에서 어떻게 작용할지 검증되지 않습니다 — (i) 비유클리드/유체/유연체, (ii) 매우 얇은 구조(머리카락·잎맥), (iii) 투명/굴절 재질(유리, 액체), (iv) 인체 사실주의(피부 SSS).
- **메트릭 일반화 갭**: ULIP/Uni3D/CLIP-FID 모두 **이미지 임베딩 기반 간접 메트릭**입니다. 손가락 개수·기계 부품 수 같은 **이산 구조 정확성(structural correctness)** 을 직접 측정하지 못해, 이 측면에서의 일반화는 메트릭상 보이지 않을 수 있습니다.
- **멀티뷰 일관성의 수치 평가 부재**: 텍스처 평가는 단일 뷰 기반 메트릭이며, 멀티뷰 일관성(예: cross-view PSNR, seam metric)에 대한 수치 보고가 없어 이 차원의 일반화는 정성 결과(Fig. 7)에만 의존합니다.

요약하면, LATTICE의 scale-up 안정성과 PBR 분해는 **상용·산업 파이프라인으로의 일반화에 매우 유리**하지만, **이산 구조의 정확성과 비표준 재질로의 일반화**는 추가 검증이 필요합니다.

---

## 4. 향후 연구에 미치는 영향과 고려할 점

### 4.1 영향 (Likely Impact)

- **3D 생성에서의 "스케일이 답이다" 패러다임 강화**: 10B 파라미터 vecset 디퓨전이 안정적으로 개선됨을 보고함으로써, 후속 연구는 **데이터·모델 양쪽의 스케일링**을 표준 절차로 채택할 가능성이 큽니다. 이미 동일 계열에서 TripoSG(Li et al., 2025), Trellis(Xiang et al., 2024), CLAY(Zhang et al., 2024b)가 같은 추세를 따르고 있습니다.
- **PBR 텍스처 생성의 오픈 소스화 가속**: 오픈소스 진영의 가장 큰 갭이 PBR이었음을 본 논문이 짚었고, dual-channel attention은 이후 텍스처 모델의 표준 구성요소가 될 가능성이 있습니다(MaterialMVP, RomanTex, TRELLIS.2 같은 동시기 작업이 같은 방향).
- **상용 모델과의 격차 축소**: 사용자 스터디에서 상용 모델 대비 50–72% 승률은, **오픈소스 계열이 폐쇄형 상용 제품을 따라잡기 시작했다**는 신호로 해석될 수 있습니다.

### 4.2 향후 연구 시 고려할 점

1. **체계적 ablation과 scaling law 정립**: 모델 사이즈(0.5B→1B→3B→10B)와 데이터 규모를 축으로 한 scaling curve를 제시하면, 후속 연구가 자원 계획을 세우기 쉬워집니다. 본문은 이 부분이 약합니다.
2. **메트릭 개선**: ULIP/Uni3D 한계를 저자도 인정한 만큼, **구조적 정확도(part counting, fine-detail F-score), 멀티뷰 일관성(cross-view consistency PSNR/SSIM), PBR 정확도(BRDF reconstruction error)** 같은 새로운 벤치마크가 필요합니다.
3. **재현성과 공개**: LATTICE의 구조적 디테일과 학습 데이터 통계가 공개되어야 학계 비교가 가능합니다. (Hunyuan3D 2.0/2.1은 GitHub 공개되어 있으나, 2.5의 10B 모델 공개 여부는 본 보고서에서 명시되지 않습니다.)
4. **위상학적/구조적 정확성**: 손가락·치아 같은 이산 카운트 정확성과, **메시 토폴로지(쿼드 도미넌트, 인공물 아티스트 친화 토폴로지)** 는 별도의 후처리·표현(MeshGPT, BPT, Meshtron 같은 autoregressive 계열)과의 결합이 유망합니다.
5. **재질·조명의 완전한 디스인탱글**: illumination-invariant loss를 더 발전시키고, anisotropic·subsurface scattering·clearcoat 같은 **확장 BRDF** 로의 일반화가 필요합니다.
6. **윤리·라이선스**: 대규모 3D 데이터 큐레이션은 IP 문제(아티스트 자산, 캐릭터 IP)를 동반합니다. 본 보고서는 데이터 출처에 대한 언급이 없어, 후속 연구는 데이터 거버넌스를 함께 다루어야 합니다.
7. **속도-품질 trade-off의 정량화**: distillation 후 생성 시간/품질 곡선을 명시해야 산업 도입이 가능합니다(이는 동일 팀의 FlashVDM에서 일부 다뤄지고 있음).

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

3D 생성 분야의 주요 흐름을 패러다임별로 정리하고 Hunyuan3D 2.5의 위치를 표시합니다.

### 5.1 Score-Distillation 기반 (2022–2023 초)

**대표작**: DreamFusion (Poole et al., ICLR 2023), Magic3D, Fantasia3D.
**원리**: 사전학습된 2D 디퓨전을 score function으로 사용해 NeRF/메시를 최적화.

$$
\nabla_\theta \mathcal{L}_{\text{SDS}}(\phi, g(\theta)) \;=\; \mathbb{E}_{t,\boldsymbol{\epsilon}}\!\left[\, w(t)\,\big(\boldsymbol{\epsilon}_\phi(\mathbf{x}_t; y, t) - \boldsymbol{\epsilon}\big)\,\frac{\partial \mathbf{x}}{\partial \theta}\,\right]
$$

**한계**: 객체당 분 단위 최적화, Janus 문제, 디테일 부족. Hunyuan3D 2.5는 **feed-forward 디퓨전**으로 이를 우회.

### 5.2 Feed-Forward 재구성 (2023–2024)

**대표작**: LRM (Hong et al., 2023), Hunyuan3D 1.0 (Yang et al., 2024), LGM (Tang et al., 2024).
**원리**: 트랜스포머가 단일/희소 뷰에서 triplane/3DGS를 직접 회귀.
**한계**: 디테일·기하 충실도 한계.

### 5.3 Native 3D Diffusion – Vecset 계열 (2023–2025, **현재 주류**)

3DShape2VecSet (Zhang et al., SIGGRAPH 2023)이 시작점. 형상을 **잠재 벡터 셋**으로 압축한 뒤 latent diffusion. 후속작:

| 모델 | 연도 | 파라미터 | 특이점 |
|---|---|---|---|
| Michelangelo | 2024 | – | shape-image-text 정렬 latent |
| CLAY (Zhang et al.) | SIGGRAPH 2024 | 1.5B | 점진적 학습, 2K PBR |
| Craftsman 1.5 | 2024 | – | 기하 refiner |
| Hunyuan3D 2.0 | 2025 | – | 본 논문의 직전 버전 |
| TripoSG | 2025 | 4B | rectified flow + DINOv2 조건 |
| **Hunyuan3D 2.5 (LATTICE)** | **2025** | **10B** | scale-up + PBR + dual-phase 학습 |

학습 목적함수는 일반적으로 다음 디퓨전 손실:

$$
\mathcal{L} \;=\; \mathbb{E}_{\mathbf{Z}_0,\,\boldsymbol{\epsilon}\sim\mathcal{N}(0,I),\,t}\!\left[\, \big\| \boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\mathbf{Z}_t,\, t,\, c_{\text{img}})\big\|_2^{2}\,\right]
$$

또는 TripoSG처럼 **rectified flow** 손실:

$$
\mathcal{L}_{\text{RF}} \;=\; \mathbb{E}_{t,\,\mathbf{Z}_0,\,\boldsymbol{\epsilon}}\!\left[\, \big\| v_\theta(\mathbf{Z}_t,\, t,\, c) - (\mathbf{Z}_0 - \boldsymbol{\epsilon})\big\|_2^{2}\,\right]
$$

여기서 $\mathbf{Z}_t = (1-t)\boldsymbol{\epsilon} + t\,\mathbf{Z}_0$.

### 5.4 Native 3D Diffusion – Sparse Voxel/Structured Latent 계열

**대표작**: TRELLIS (Xiang et al., CVPR 2025 Spotlight, 2B 파라미터, 500K 자산), Direct3D (triplane), TRELLIS.2.
**핵심 차이**: 표면 voxel에 DINOv2 기반 dense feature를 붙인 **SLAT** 표현, 메시·3DGS·NeRF로 multi-decoding 가능. Hunyuan3D 2.5와는 **표현 방식의 양대 산맥**(vecset vs structured voxel)을 형성합니다.

### 5.5 Autoregressive 메시 생성

**대표작**: MeshGPT (Siddiqui et al., CVPR 2024), BPT (Weng et al., 2024), Meshtron (Hao et al., 2024).
**장점**: 아티스트 친화적 토폴로지(쿼드 도미넌트, 균일 폴리곤). **단점**: 디테일 한계, 시퀀스 길이 폭발.

### 5.6 텍스처/PBR 생성 흐름

- **Inpainting 기반(2023)**: TEXTure (Richardson), Text2Tex (Chen) – 일관성 문제.
- **Multiview Diffusion(2024–2025)**: MVDream, Zero123++, Wonder3D, Era3D, MV-Adapter — Hunyuan3D 2.5의 모태.
- **PBR 전용**: MatFusion (Sartor & Peers, SIGGRAPH Asia 2023), Collaborative Control (Vainer et al., ECCV 2024), MaterialMVP (He et al., 2025), DreamMat (Zhang et al., TOG 2024). Hunyuan3D 2.5는 이들 중 **dual-channel attention과 dual-phase resolution** 으로 차별화됩니다.

### 5.7 Hunyuan3D 2.5의 위치 정리

- **표현 패러다임**: vecset 계열의 가장 큰 모델(10B)
- **텍스처 패러다임**: multiview PBR diffusion, 채널 정합 문제를 attention mask 공유로 해결
- **속도**: distillation 사용(상세 수치 부재), 같은 팀의 FlashVDM(ICCV'25 Highlight)으로 1초대 생성도 가능
- **경쟁자**: 폐쇄형 상용(Tripo, Meshy 등 추정 — 본문은 익명) 및 Trellis/TRELLIS.2(Microsoft, sparse voxel 계열)

---

## 참고한 자료 (출처)

본 답변에서 직접 인용·참조한 자료입니다.

1. **Hunyuan3D 2.5 본 논문**: Tencent Hunyuan3D, "Hunyuan3D 2.5: Towards High-Fidelity 3D Assets Generation with Ultimate Details", arXiv:2506.16504v1, 2025년 6월 19일. https://arxiv.org/abs/2506.16504
2. **GitHub 저장소**: Tencent/Hunyuan3D-2. https://github.com/Tencent/Hunyuan3D-2
3. **3DShape2VecSet** (vecset 패러다임의 출발): Biao Zhang, Jiapeng Tang, Matthias Nießner, Peter Wonka, ACM TOG (SIGGRAPH 2023). https://arxiv.org/abs/2301.11445 / 프로젝트 페이지 https://1zb.github.io/3DShape2VecSet/
4. **TRELLIS**: Jianfeng Xiang et al., "Structured 3D Latents for Scalable and Versatile 3D Generation", CVPR 2025 Spotlight, arXiv:2412.01506. https://arxiv.org/abs/2412.01506 / https://microsoft.github.io/TRELLIS/
5. **TRELLIS.2** (후속 연구): https://microsoft.github.io/TRELLIS.2/
6. **TripoSG**: Yangguang Li et al., "TripoSG: High-Fidelity 3D Shape Synthesis using Large-Scale Rectified Flow Models", arXiv:2502.06608. https://github.com/VAST-AI-Research/TripoSG
7. **FlashVDM** (Hunyuan 팀의 vecset 가속): Zeqiang Lai et al., "Unleashing Vecset Diffusion Model for Fast Shape Generation", ICCV 2025 Highlight, arXiv:2503.16302. https://github.com/Tencent-Hunyuan/FlashVDM
8. **CLAY**: Zhang et al., ACM TOG 2024, "CLAY: A Controllable Large-scale Generative Model for Creating High-quality 3D Assets" (본 논문 §4 인용).
9. **MaterialMVP** (PBR 일관성 손실의 출처): He et al., 2025, arXiv:2503.10289 (본 논문 §2.2 인용).
10. **RomanTex** (3D-Aware RoPE의 출처): Feng et al., 2025, arXiv:2503.19011 (본 논문 §2.2 인용).
11. **Hunyuan3D 2.1**: Hunyuan3D 팀, arXiv:2506.15442 (본 논문의 직전 버전).
12. **Papers With Code 페이지**: https://paperswithcode.com/paper/hunyuan3d-2-5-towards-high-fidelity-3d-assets
13. **Hugging Face Papers 페이지**: https://huggingface.co/papers/2506.16504
14. **Emergent Mind 분석**: https://www.emergentmind.com/papers/2506.16504

---

## 정확도에 관한 자기-검증 (사용자 요청 사항 반영)

다음 항목은 **본문에 명시되어 있지 않아 단정 불가**하므로 의도적으로 추측을 피했습니다.

- LATTICE의 정확한 네트워크 구조(블록 수, latent set 크기 $N$, VAE 손실 형태).
- 학습 데이터셋의 정확한 규모·출처.
- 10B 모델 추론 시간·GPU 메모리 사용량.
- "Commercial Model 1/2/3"의 실제 제품명.
- 모델 사이즈 별 정량 ablation 결과.
- LATTICE라는 이름의 약어 풀이 — 보고서가 약어를 풀어 쓰지 않습니다.

이 부분은 추가 자료(향후 저자들의 후속 보고서, 코드 공개)를 통해 확인해야 합니다. 위 모든 분석은 첨부된 PDF와 명시된 외부 자료의 내용에 한정합니다.
