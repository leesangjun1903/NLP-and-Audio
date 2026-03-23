# VR-GS: A Physical Dynamics-Aware Interactive Gaussian Splatting System in Virtual Reality

---

## 1. 핵심 주장 및 주요 기여 요약

**VR-GS**는 3D Gaussian Splatting(GS)으로 표현된 3D 콘텐츠를 Virtual Reality(VR) 환경에서 **물리 법칙에 기반하여 실시간으로 상호작용**할 수 있도록 설계된 최초의 인터랙티브 시스템이다. 핵심 주장은 다음과 같다:

1. **물리 기반 인터랙티브 시스템**: eXtended Position-based Dynamics(XPBD)와 3D GS를 결합하여, 사용자가 VR에서 변형 가능한 가상 객체를 물리적으로 그럴듯하게 조작할 수 있는 실시간 시스템을 구축하였다.
2. **Two-Level Deformation Embedding**: Gaussian 커널을 시뮬레이션 메시에 직접 임베딩할 때 발생하는 spiky artifact를 해결하기 위해, **로컬-글로벌 2단계 임베딩** 전략을 제안하여 부드러운 변형장(deformation field)을 보장하였다.
3. **통합 파이프라인**: 장면 재구성 → 객체 분할(Segmentation) → 인페인팅(Inpainting) → 메시 생성 → 물리 시뮬레이션 → 실시간 렌더링(그림자 포함)까지 일관된 워크플로우를 제공한다.

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

- **기존 3D 콘텐츠 제작의 한계**: 전통적 그래픽 파이프라인은 고품질 메시, UV 맵, 텍스처 등을 요구하여 비전문가에게 접근이 어렵다.
- **NeRF의 실시간 인터랙션 부적합성**: NeRF는 암묵적(implicit) 표현으로 인해 변형 시 역변형 맵(inverse deformation map)을 통한 레이 벤딩이 필요하며, 충돌 감지가 어렵고 볼륨 렌더링이 느려 실시간 VR에 부적합하다.
- **GS 기반 물리 시뮬레이션의 성능 병목**: PhysGaussian [Xie et al. 2024]은 Gaussian 커널 단위(per-Gaussian)로 MPM(Material Point Method)을 사용하여 상세한 역학을 제공하지만, 복잡한 장면에서 실시간 성능이 불가능하다.
- **Spiky Artifact 문제**: GS 커널을 테트라헤드랄 메시에 단순 임베딩할 경우, 커널이 테트라헤드론 경계를 넘어 불연속적 변형 그라디언트로 인한 시각적 아티팩트가 발생한다.

### 2.2 제안하는 방법 (수식 포함)

#### (1) Gaussian Splatting 렌더링

3D 장면은 학습 가능한 평균 $\mu$, 불투명도 $\sigma$, 공분산 행렬 $\Sigma$, 구면 조화 계수 $C$를 가진 비등방성 3D Gaussian 커널 집합으로 표현된다. 픽셀 색상은 $\alpha$-블렌딩으로 계산된다:

$$C = \sum_{i \in N} c_i \alpha_i \prod_{j=1}^{i-1}(1 - \alpha_j) \tag{1}$$

여기서 $c_i$는 구면 조화(SH)에서 평가된 색상, $\alpha_i$는 커널의 불투명도와 2D Gaussian 가중치의 곱이다.

#### (2) Segmentation Loss

2D 분할 모델 [Cheng et al. 2023]로 생성된 마스크를 기반으로, 3D Gaussian 커널에 추가적인 학습 가능한 RGB 속성을 부여하여 분할 손실을 정의한다:

$$L_{\text{seg}} = L_1(M_{2d}, I) \tag{2}$$

전체 학습 손실:

$$L_{\text{total}} = (1 - \lambda)L_1 + \lambda L_{\text{SSIM}} + \lambda_{\text{seg}} L_{\text{seg}} \tag{3}$$

여기서 $\lambda = 0.2$, $\lambda_{\text{seg}} = 0.1$이다.

#### (3) 변형 그라디언트 (Deformation Gradient)

테트라헤드론의 정지 형태 $\{x_0^0, x_1^0, x_2^0, x_3^0\}$과 현재 형태 $\{x_0, x_1, x_2, x_3\}$에 대해:

$$F = \left[x_1 - x_0,\; x_2 - x_0,\; x_3 - x_0\right] \left[x_1^0 - x_0^0,\; x_2^0 - x_0^0,\; x_3^0 - x_0^0\right]^{-1} \tag{4}$$

변형된 Gaussian 커널의 평균과 공분산:

$$\boldsymbol{\mu} = \sum_i w_i \boldsymbol{x}_i, \quad \boldsymbol{\Sigma} = F \boldsymbol{\Sigma}_0 F^T \tag{5}$$

여기서 $w_i$는 정지 형태에서 초기 중심 $\boldsymbol{\mu}_0$의 무게 중심 좌표(barycentric coordinates), $\boldsymbol{\Sigma}_0$는 초기 공분산 행렬이다.

#### (4) Two-Level Embedding

단순 임베딩의 spiky artifact를 해결하기 위한 2단계 전략:

- **Local Embedding**: 각 Gaussian 커널을 독립적인 최소 바운딩 테트라헤드론에 개별적으로 포함시킨다. 이 로컬 테트라헤드라 간에는 연결성이 없다.
- **Global Embedding**: 로컬 테트라헤드론의 꼭짓점들을 글로벌 시뮬레이션 메시에 임베딩한다.

이를 통해 하나의 로컬 테트라헤드론이 여러 글로벌 테트라헤드라에 걸칠 수 있으며, 해당 Gaussian의 변형은 주변 글로벌 테트라헤드라의 **평균 변형**으로 결정되어 부드러운 변형장을 생성한다.

#### (5) XPBD 기반 실시간 시뮬레이션

PhysGaussian의 MPM 대신 XPBD [Macklin et al. 2016]를 채택하여:
- 변형 에너지 제약조건(strain energy constraint)을 탄성 모델로 사용
- 속도 기반 감쇠(velocity-based damping) 모델 적용
- 테트라헤드랄 메시의 정점 수를 10K–30K로 제한하여 실시간 성능과 물리적 정확도의 균형 달성

#### (6) Shadow Map

GS는 원래 구면 조화에 그림자를 텍스처로 학습하므로 객체 이동 시 그림자가 갱신되지 않는다. 본 시스템은 광원으로부터 식 (1)을 이용해 깊이 맵을 추정하고, 각 Gaussian의 가시성을 테스트하여 **동적 그림자**를 실시간으로 생성한다.

### 2.3 모델 구조 (파이프라인)

전체 파이프라인은 3단계로 구성된다 (Fig. 2 참조):

| 단계 | 구성 요소 |
|------|---------|
| **Object-level 3D Scene Reconstruction** | 실제 장면 촬영 → COLMAP 보정 → 3D GS 학습 → 분할 → 인페인팅 |
| **GS Embedded Geometry Reconstruction** | VDB 재구성 → Marching Cubes → TetGen → 글로벌 임베딩 → 로컬 임베딩 (Two-level) |
| **VR-GS Simulation and Rendering** | XPBD 시뮬레이션 → 충돌 처리 → GS 임베딩 보간 → Shadow Ray → Gaussian Rasterizer (좌/우 눈 렌더링) |

### 2.4 성능 향상

#### 정량적 비교 (프레임당 시뮬레이션 시간, 초 단위)

| 예제 | PAC-NeRF | PhysGaussian | **VR-GS (Ours)** |
|------|----------|-------------|----------------|
| Stool | 0.750 | 0.112 | **0.017** |
| Chair | 0.813 | 0.219 | **0.022** |
| Materials | 0.625 | 0.39 | **0.021** |

VR-GS는 PhysGaussian 대비 약 **5~19배**, PAC-NeRF 대비 약 **30~44배** 빠른 시뮬레이션 속도를 달성하며, 시각적 품질은 PhysGaussian에 필적하고 PAC-NeRF를 크게 상회한다.

#### 실시간 성능 (FPS)

다양한 데모에서 **24.3~161.2 FPS**를 달성하여 VR 인터랙션에 적합한 프레임률을 보장한다 (Table 1 참조).

#### 사용자 연구

- 10명의 참가자에 대한 사용자 연구에서 물리 기반 인터랙션(6.1점)이 변환 기반 인터랙션(4.8점)보다 **몰입감과 사실성에서 유의미하게 높은 점수** (7점 리커트 척도, $p = .0227$)
- System Usability Scale(SUS) 점수 **83.5** (\"Excellent\" 등급)

### 2.5 한계

1. **대규모 장면에서의 렌더링 비용**: 2K 해상도의 대규모 장면에서 고충실도 GS 렌더링은 잠재적 지연 문제를 유발할 수 있다.
2. **수동 물리 파라미터 설정**: Young's modulus ($E$), Poisson ratio ($\nu$), 밀도 ($\rho$)를 수동으로 조정해야 하며, 약 10회의 반복 실험이 필요하다.
3. **재질 범위 제한**: 현재 탄성체(deformable body)만 지원하며, 유체(fluid)나 천(cloth) 등의 재질은 미지원이다.
4. **메시 해상도와 성능의 트레이드오프**: 저해상도 메시는 세밀한 역학을 포착하지 못하고, 고해상도 메시는 XPBD 수렴 부족으로 과도하게 부드러운 시뮬레이션을 초래할 수 있다.

---

## 3. 모델의 일반화 성능 향상 가능성

VR-GS의 일반화 성능은 다음과 같은 관점에서 논의할 수 있다:

### 3.1 다양한 장면 및 객체에 대한 일반화

- **다양한 데이터셋 호환성**: Instant-NGP 데이터셋, Instruct-NeRF2NeRF, Tanks and Temples 등 다양한 소스에서 재구성된 장면에 적용 가능함을 실험으로 입증하였다.
- **텍스트-to-3D 통합**: LucidDreamer [Liang et al. 2023]로 생성된 가상 객체도 기존 장면에 원활히 통합 가능하여, 촬영 기반 재구성 외에도 생성 모델 기반 객체까지 일반화 범위를 확장하였다.
- **가상 + 실제 혼합**: Blender로 모델링한 가상 객체(링, 벽돌)를 실제 촬영 장면에 삽입하여 인터랙션하는 하이브리드 시나리오를 시연하였다.

### 3.2 일반화 성능 향상을 위한 개선 방향

1. **자동 물리 파라미터 추정**: 현재 수동 설정 방식에서, PAC-NeRF [Li et al. 2023a]처럼 비디오에서 재질 파라미터를 자동 추정하거나, 대규모 비전-언어 모델(LVM)을 활용하여 자동화하면 새로운 장면/객체에 대한 일반화가 크게 향상될 수 있다.
2. **다양한 재질 확장**: 유체 [Feng et al. 2024a], 천 등 다양한 재질 시뮬레이션을 통합하면 일반화 범위가 확대된다.
3. **적응적 메시 해상도**: 객체의 복잡도와 하드웨어 제약에 따라 자동으로 메시 해상도를 조절하는 적응적 전략이 일반화에 기여할 수 있다.
4. **대규모 멀티모달 모델 활용**: 생성된 역학의 충실도를 평가하기 위해 대규모 멀티모달 모델을 활용하는 방향이 제안되었다.

### 3.3 구조적 일반화 장점

Two-level embedding 전략 자체가 일반화에 유리한 구조적 특성을 가진다:
- 로컬 테트라헤드론이 Gaussian 커널의 형상에 맞게 독립적으로 생성되므로, **임의 형태의 객체**에 적용 가능하다.
- 글로벌 메시에 대한 의존성이 완화되어, 메시 품질이 다소 낮더라도 부드러운 변형이 보장된다.
- VDB + Marching Cubes + TetGen 파이프라인이 완전 자동화되어 있어, 새로운 GS 재구성에 범용적으로 적용 가능하다.

---

## 4. 향후 연구에 미치는 영향 및 고려할 점

### 4.1 연구에 미치는 영향

1. **실시간 물리 기반 3D 인터랙션 패러다임 제시**: NeRF/GS 기반 3D 콘텐츠를 단순 시각화를 넘어 **물리적으로 상호작용 가능한 대상**으로 전환하는 패러다임을 확립하였다. 이는 VR/AR 기반 교육, 엔터테인먼트, 원격 협업 등에 직접적 영향을 미친다.

2. **렌더링-시뮬레이션 통합 설계 원칙**: "What you see is what you simulate" 원칙 하에 렌더링과 시뮬레이션을 통합하는 설계 방법론은 향후 인터랙티브 3D 시스템 설계의 표준이 될 수 있다.

3. **Gaussian Splatting 기반 물리 시뮬레이션의 실용화**: PhysGaussian이 제시한 GS-물리 통합의 개념을 실시간 수준으로 끌어올려, 실용적 응용의 가능성을 입증하였다.

4. **비전문가 접근성 확대**: 약 2시간 30분의 준비 시간으로 실제 장면을 물리 기반 인터랙티브 VR 환경으로 변환할 수 있음을 보여, 전문가와 비전문가 모두의 3D 콘텐츠 제작 접근성을 높였다.

### 4.2 향후 연구 시 고려할 점

1. **확장성(Scalability)**: 현재 시스템은 개별 객체 수준의 시뮬레이션에 최적화되어 있으며, 대규모 장면(수백 개 객체)에서의 확장성 검증이 필요하다.

2. **물리 파라미터 자동화**: 비디오 기반 역문제(inverse problem) 해결이나, diffusion 기반 재질 추정 모델을 통한 자동 파라미터 설정이 핵심 연구 방향이다.

3. **다중 재질 시뮬레이션**: 유체, 천, 파괴 역학(fracture mechanics) 등을 통합한 멀티-재질 시뮬레이션이 필요하다.

4. **네트워크 기반 다중 사용자 인터랙션**: 현재 단일 사용자 시스템에서 다중 사용자 협업 VR 환경으로의 확장이 고려되어야 한다.

5. **품질-성능 자동 최적화**: 메시 해상도, XPBD 반복 횟수, GS 커널 수를 하드웨어 성능에 따라 자동 조절하는 적응적 LOD(Level of Detail) 메커니즘이 필요하다.

6. **평가 프로토콜 표준화**: 물리 기반 GS 인터랙션의 충실도를 정량적으로 평가할 수 있는 표준 벤치마크와 메트릭의 개발이 요구된다.

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

| 연구 | 연도 | 표현 방식 | 물리 시뮬레이션 | 실시간 인터랙션 | 주요 특징 |
|------|------|---------|------------|------------|---------|
| **NeRF** [Mildenhall et al.] | 2021 | Implicit (MLP) | ✗ | ✗ | 고품질 novel view synthesis의 기초 |
| **D-NeRF** [Pumarola et al.] | 2021 | Implicit + 시간 | ✗ (데이터 기반) | ✗ | 동적 장면을 canonical + deformation field로 분리 |
| **Instant-NGP** [Müller et al.] | 2022 | Hash encoding | ✗ | △ (빠른 학습) | 멀티해상도 해시 인코딩으로 학습/렌더링 가속 |
| **3D Gaussian Splatting** [Kerbl et al.] | 2023 | Explicit (Gaussian) | ✗ | ✓ (렌더링만) | 명시적 3D 표현, 실시간 래스터화 |
| **PAC-NeRF** [Li et al.] | 2023 | Implicit (NeRF) + MPM | ✓ (MPM) | ✗ | 비디오에서 재질 파라미터 추정, 낮은 렌더링 품질 |
| **PhysGaussian** [Xie et al.] | 2024 | Explicit (Gaussian) + MPM | ✓ (MPM) | ✗ | GS 커널 단위 물리 시뮬레이션, 고품질 but 느림 |
| **PIE-NeRF** [Feng et al.] | 2024 | Implicit (NeRF) + FEM | ✓ (FEM) | △ (제한적) | NeRF에 인터랙티브 탄성역학 통합 |
| **4D Gaussian Splatting** [Yang et al.] | 2023 | Explicit (4D Gaussian) | ✗ (데이터 기반) | ✓ (렌더링) | 시간에 따라 변화하는 4D Gaussian primitive |
| **SC-GS** [Huang et al.] | 2023 | Sparse-Controlled GS | ✗ (학습 기반) | △ | 편집 가능한 동적 장면을 위한 희소 제어 |
| **GaussianEditor** [Chen et al.] | 2023 | Explicit (Gaussian) | ✗ | △ | 텍스트 기반 GS 편집 |
| **Gaussian Splashing** [Feng et al.] | 2024 | Explicit (Gaussian) + 유체 | ✓ (유체) | ✗ | GS 기반 동적 유체 합성 |
| **VR-NeRF** [Xu et al.] | 2023 | Implicit (NeRF) | ✗ | ✓ (다중 GPU) | 다중 GPU 기반 고품질 VR 볼루메트릭 렌더링 |
| **VR-GS (본 논문)** | **2024** | **Explicit (Gaussian) + XPBD** | **✓ (XPBD)** | **✓ (실시간)** | **GS + XPBD 통합, Two-level embedding, VR 인터랙션** |

### 핵심 차별점 분석

- **vs. PhysGaussian**: PhysGaussian은 MPM으로 per-Gaussian 시뮬레이션을 수행하여 세밀한 역학을 제공하지만 실시간이 불가능하다. VR-GS는 XPBD + 케이지 메시 임베딩으로 **5~19배 빠른 속도**를 달성하면서 비슷한 시각적 품질을 유지한다.
- **vs. PAC-NeRF**: PAC-NeRF는 NeRF 기반으로 렌더링 품질이 낮고 시뮬레이션도 느리다. VR-GS는 GS의 명시적 표현과 효율적 래스터화를 활용하여 **렌더링 품질과 속도 모두에서 우위**를 점한다.
- **vs. PIE-NeRF**: PIE-NeRF는 NeRF에 FEM 기반 탄성역학을 통합하지만, 암묵적 표현의 렌더링 비용으로 인해 VR 수준의 실시간 인터랙션에는 미치지 못한다.
- **vs. 4D GS / D-NeRF**: 이들은 입력 데이터에서 포착된 동작만 재현할 수 있으며, **새로운(unseen) 역학을 합성**할 수 없다. VR-GS는 물리 법칙에 기반하여 임의의 새로운 인터랙션을 생성한다.

---

## 참고자료

1. Jiang, Y., Yu, C., Xie, T., et al. (2024). "VR-GS: A Physical Dynamics-Aware Interactive Gaussian Splatting System in Virtual Reality." *arXiv:2401.16663v2*.
2. Kerbl, B., Kopanas, G., Leimkühler, T., & Drettakis, G. (2023). "3D Gaussian Splatting for Real-Time Radiance Field Rendering." *ACM Transactions on Graphics*, 42(4).
3. Xie, T., Zong, Z., Qiu, Y., et al. (2024). "PhysGaussian: Physics-Integrated 3D Gaussians for Generative Dynamics." *Proceedings of the IEEE/CVF CVPR*.
4. Li, X., Qiao, Y.-L., Chen, P.Y., et al. (2023). "PAC-NeRF: Physics Augmented Continuum Neural Radiance Fields for Geometry-Agnostic System Identification." *arXiv:2303.05512*.
5. Macklin, M., Müller, M., & Chentanez, N. (2016). "XPBD: Position-Based Simulation of Compliant Constrained Dynamics." *Proceedings of MIG '16*.
6. Feng, Y., Shang, Y., Li, X., et al. (2024b). "PIE-NeRF: Physics-based Interactive Elastodynamics with NeRF." *Proceedings of IEEE/CVF ICCV*.
7. Feng, Y., et al. (2024a). "Gaussian Splashing: Dynamic Fluid Synthesis with Gaussian Splatting." *arXiv:2401.15318*.
8. Mildenhall, B., et al. (2021). "NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis." *Communications of the ACM*, 65(1).
9. Pumarola, A., et al. (2021). "D-NeRF: Neural Radiance Fields for Dynamic Scenes." *Proceedings of IEEE/CVF CVPR*.
10. Yang, Z., et al. (2023). "Real-time Photorealistic Dynamic Scene Representation and Rendering with 4D Gaussian Splatting." *arXiv:2310.10642*.
11. Müller, T., et al. (2022). "Instant Neural Graphics Primitives with a Multiresolution Hash Encoding." *ACM Transactions on Graphics*, 41(4).
12. Suvorov, R., et al. (2022). "Resolution-Robust Large Mask Inpainting with Fourier Convolutions." *Proceedings of IEEE/CVF WACV*.
13. Cheng, H.K., et al. (2023). "Putting the Object Back into Video Object Segmentation." *arXiv:2310.12982*.
14. Huang, Y.-H., et al. (2023b). "SC-GS: Sparse-Controlled Gaussian Splatting for Editable Dynamic Scenes." *arXiv:2312.14937*.
15. Chen, Y., et al. (2023). "GaussianEditor: Swift and Controllable 3D Editing with Gaussian Splatting." *arXiv:2311.14521*.
16. Xu, L., et al. (2023a). "VR-NeRF: High-Fidelity Virtualized Walkable Spaces." *SIGGRAPH Asia 2023 Conference Papers*.
17. Bangor, A., Kortum, P., & Miller, J. (2009). "Determining What Individual SUS Scores Mean: Adding an Adjective Rating Scale." *Journal of Usability Studies*, 4(3).
18. Liang, Y., et al. (2023). "LucidDreamer: Towards High-Fidelity Text-to-3D Generation via Interval Score Matching." *arXiv:2311.11284*.
