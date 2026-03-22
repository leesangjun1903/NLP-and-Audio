# Text2Room: Extracting Textured 3D Meshes from 2D Text-to-Image Models

---

## 1. 핵심 주장과 주요 기여 요약

**Text2Room**은 텍스트 프롬프트만으로 **방 규모(room-scale)의 텍스처가 입혀진 3D 메시(textured 3D mesh)**를 생성하는 최초의 방법을 제시한다. 기존의 text-to-3D 방법들이 단일 객체 생성이나 zoom-out 궤적에 한정되었던 것과 달리, Text2Room은 벽, 바닥, 천장, 가구 등 여러 객체를 포함하는 **완전한 실내 장면**을 명시적 3D 메시 표현으로 생성한다.

### 주요 기여 (3가지)

1. **임의의 텍스트 입력으로부터 room-scale 실내 3D 메시 생성**: 설득력 있는 텍스처와 기하학을 가진 완전한 장면을 생성한다.
2. **반복적 장면 생성 프레임워크**: 2D text-to-image 모델과 단안(monocular) 깊이 추정을 결합하여, **깊이 정렬(depth alignment)**과 **메시 융합(mesh fusion)** 단계를 통해 왜곡 없는 이음매 없는 기하학과 텍스처를 생성한다.
3. **2단계 맞춤형 시점 선택(two-stage tailored viewpoint selection)**: 사전 정의된 궤적에서 방 레이아웃과 가구를 먼저 생성한 후, 남은 구멍을 후처리 방식으로 채워 방수(watertight) 메시를 생성한다.

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

기존 text-to-3D 방법들은 다음과 같은 한계를 가진다:

- **3D 학습 데이터 부족**: 3D 데이터셋(e.g., ShapeNet)은 2D 데이터셋에 비해 규모가 매우 작다.
- **단일 객체 한정**: DreamFusion, Magic3D 등은 radiance field로 단일 객체를 생성하며, room-scale 장면으로 확장이 어렵다.
- **메시 표현의 부재**: 기존 NeRF 기반 방법은 commodity hardware에서의 래스터라이제이션 렌더링에 직접 사용하기 어렵다.
- **장면 일관성 문제**: 대규모 장면에서 다양한 시점 간 밀도 있고 일관된 출력을 보장하기 어렵다.

### 2.2 제안하는 방법 (수식 포함)

#### (A) 반복적 3D 장면 생성 (Iterative 3D Scene Generation)

장면은 메시 $\mathcal{M} = (\mathcal{V}, \mathcal{C}, \mathcal{S})$로 표현되며, 여기서 $\mathcal{V} \in \mathbb{R}^{N \times 3}$은 정점, $\mathcal{C} \in \mathbb{R}^{N \times 3}$은 정점 색상, $\mathcal{S} \in \mathbb{N}_0^{M \times 3}$은 면 집합이다.

입력은 텍스트 프롬프트 집합 $\{P_t\}\_{t=1}^{T}$와 대응하는 카메라 포즈 $\{E_t\}_{t=1}^{T} \in \mathbb{R}^{3 \times 4}$이다.

**Step 1: 래스터라이제이션 렌더링**

$$I_t, d_t, m_t = r(\mathcal{M}_t, E_t) $$

여기서 $r$은 셰이딩 없는 래스터라이제이션 함수, $I_t$는 렌더링 이미지, $d_t$는 렌더링 깊이, $m_t$는 관측되지 않은 픽셀을 표시하는 마스크이다.

**Step 2: Text-to-Image 인페인팅**

$$\hat{I}_t = \mathcal{F}_{t2i}(I_t, m_t, P_t) $$

고정된 text-to-image 모델 $\mathcal{F}_{t2i}$를 사용하여 관측되지 않은 픽셀을 텍스트 프롬프트에 따라 인페인팅한다.

**Step 3: 깊이 예측 및 정렬**

$$\hat{d}_t = \textit{predict-and-align}(\mathcal{F}_d, I_t, d_t, m_t) $$

단안 깊이 추정기 $\mathcal{F}_d$를 적용하고 깊이 정렬을 수행한다.

**Step 4: 메시 융합**

$$\mathcal{M}_{t+1} = \textit{fuse}(\mathcal{M}_t, \hat{I}_t, \hat{d}_t, m_t, E_t) $$

#### (B) 깊이 정렬 단계 (Depth Alignment Step)

서로 다른 시점에서 예측된 단안 깊이는 스케일이 일관되지 않아 3D 기하학에서 불연속성이 발생한다. 이를 해결하기 위해 2단계 깊이 정렬을 수행한다:

1. **깊이 인페인팅 네트워크** (IronDepth): 기존 알려진 깊이 $d$를 조건으로 예측:
$$\hat{d}_p = \mathcal{F}_d(I, d)$$

2. **스케일-시프트 최적화**: 스케일 $\gamma$와 시프트 $\beta \in \mathbb{R}$를 최적화하여 예측 디스패리티와 렌더링 디스패리티를 최소제곱 의미에서 정렬:

$$\min_{\gamma, \beta} \left\lVert m \odot \left( \frac{\gamma}{\hat{d}_p} + \beta - \frac{1}{d} \right) \right\rVert^2 $$

정렬된 깊이는 다음과 같이 추출된다:

$$\hat{d} = \left( \frac{\gamma}{\hat{d}_p} + \beta \right)^{-1}$$

마지막으로 마스크 경계에 $5 \times 5$ 가우시안 커널을 적용하여 $\hat{d}$를 스무딩한다.

#### (C) 메시 융합 단계 (Mesh Fusion Step)

이미지 공간 픽셀을 월드 공간 포인트 클라우드로 역투영:

$$\mathcal{P}_t = \{E_t^{-1} K^{-1} \cdot \hat{d}_t[u,v] \cdot (u, v, 1)^T\}_{u=0,v=0}^{W, H} $$

여기서 $K \in \mathbb{R}^{3 \times 3}$은 카메라 내부 파라미터이다.

인접 4개 픽셀로 두 개의 삼각형을 형성하는 삼각화(triangulation) 후, 두 가지 필터를 적용:

- **에지 길이 필터**: 면의 에지 유클리드 거리가 임계값 $\delta_{edge}$보다 크면 제거
- **표면 법선 필터**:

$$\mathcal{S} = \{(i_0, i_1, i_2) \mid n^T v > \delta_{sn}\} $$

여기서 $n \in \mathbb{R}^3$은 정규화된 면 법선, $v \in \mathbb{R}^3$은 정규화된 시선 방향, $\delta_{sn}$은 임계값이다. 이는 작은 그레이징 각도에서 대량의 메시 영역에 대해 소수의 픽셀로 텍스처가 생성되는 것을 방지한다.

#### (D) 2단계 시점 선택 (Two-Stage Viewpoint Selection)

**생성 단계 (Generation Stage)**:
- 사전 정의된 궤적으로 방 레이아웃과 가구를 생성
- 각 궤적은 관측되지 않은 영역이 많은 시점에서 시작
- 최적 관측 거리 보장: $T_{i+1} = T_i - 0.3L$ (평균 렌더링 깊이가 0.1보다 클 때까지)

**완성 단계 (Completion Stage)**:
- 장면을 균일 복셀로 분할하고 각 셀에서 랜덤 포즈를 샘플링
- 기존 기하학과 너무 가까운 포즈를 제거하고, 관측되지 않은 픽셀을 가장 많이 보는 포즈를 선택
- 소형 구멍은 고전적 인페인팅으로 채우고 남은 구멍을 팽창(dilation)
- 최종적으로 Poisson 표면 재구성을 적용하여 방수 메시 생성

### 2.3 모델 구조

Text2Room은 별도의 학습이 필요 없는 **test-time pipeline**으로, 다음의 사전 학습된 모델을 조합한다:

| 컴포넌트 | 사용 모델 |
|---|---|
| Text-to-Image 인페인팅 $\mathcal{F}_{t2i}$ | Stable Diffusion (인페인팅 finetuned) |
| 단안 깊이 추정 $\mathcal{F}_d$ | IronDepth (ScanNet 학습) |
| 메시 래스터라이제이션 및 융합 | PyTorch3D |
| Diffusion 샘플러 | DPM-Solver++ |
| 최종 메시 후처리 | Poisson Surface Reconstruction |

### 2.4 성능 향상

**정량적 결과** (Table 1):

| Method | CS ↑ | IS ↑ | PQ ↑ | 3DS ↑ |
|---|---|---|---|---|
| PureClipNeRF | 24.06 | 1.26 | 2.34 | 2.38 |
| Outpainting | 23.10 | 1.60 | 2.90 | 2.58 |
| Text2Light+Ours | 25.99 | 2.21 | 2.82 | 2.97 |
| Blockade+Ours | 26.29 | 2.13 | 3.35 | 3.36 |
| **Ours (full)** | **28.02** | **2.31** | **4.01** | **4.19** |

- CLIP Score, Inception Score, 사용자 연구(Perceptual Quality, 3D Structure Completeness) 모두에서 최고 성능
- 61명 사용자 대상 연구에서 PQ 4.01, 3DS 4.19 (5점 만점)

**Ablation 결과**:
- 깊이 정렬 제거 시: 메시 조각이 분리되어 이음매 없는 장면 형성 불가 (CS 26.73, PQ 3.12)
- 스트레치 제거 생략 시: 면이 비자연적으로 늘어남 (PQ 3.28)
- 완성 단계 생략 시: 메시에 구멍 존재 (3DS 3.87)

### 2.5 한계

1. **임계값 기반 필터링의 불완전성**: 고정된 $\delta_{sn}=0.1$, $\delta_{edge}=0.1$로 모든 늘어난 면을 감지하지 못할 수 있음
2. **완성 단계의 불완전한 구멍 채움**: 벽에 가까운 객체 뒤의 좁은 구멍은 적절한 카메라 포즈를 찾기 어려워 Poisson 재구성으로 과도하게 스무딩될 수 있음
3. **재질-조명 분해 부재**: 확산 모델이 생성한 그림자와 밝은 조명이 텍스처에 베이크-인(bake-in)됨
4. **레이아웃 제어 제한**: 카메라 포즈로 부분적으로만 레이아웃을 제어 가능하며, 생성되는 청크의 크기가 가변적
5. **생성 시간**: RTX 3090 GPU에서 장면당 약 50분 소요

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 현재의 일반화 강점

Text2Room의 핵심 일반화 전략은 **사전 학습된 대규모 2D 모델의 지식을 3D로 전이(lift)**하는 것이다:

- **3D 학습 데이터 불필요**: Stable Diffusion이 LAION-5B 같은 대규모 이미지-텍스트 데이터셋에서 학습한 풍부한 2D 지식을 활용하므로, 제한된 3D 데이터셋에 종속되지 않음
- **임의의 텍스트 프롬프트**: "coastal bathroom," "industrial home office," "modern nursery" 등 매우 다양한 실내 장면을 생성 가능
- **다중 프롬프트 혼합**: 공간적으로 다른 텍스트 프롬프트를 사용하여 복합 장면 생성 가능 (예: 거실+주방+욕실+침실)

### 3.2 일반화 성능 향상을 위한 방향

#### (a) 기반 모델의 발전에 따른 자동 향상
Text2Room은 **모듈식 파이프라인**이므로, 각 컴포넌트를 더 강력한 모델로 교체하면 일반화 성능이 자동으로 향상된다:
- **더 나은 text-to-image 모델** (e.g., SDXL, DALL·E 3, Stable Diffusion 3): 더 정확하고 다양한 이미지 생성
- **더 나은 깊이 추정 모델** (e.g., Depth Anything, Marigold): 더 정확한 기하학 복원
- **3D-aware 인페인팅 모델**: 다중 시점 일관성을 고려한 인페인팅으로 기하학 불일치 감소

#### (b) 실외 장면 및 대규모 환경으로의 확장
현재 방법은 실내 장면에 초점을 두지만, 시점 선택 전략과 깊이 정렬을 조정하면 실외 환경이나 도시 규모 장면으로 확장 가능하다.

#### (c) 3D 일관성 강화
현재의 깊이 정렬은 $\gamma, \beta$의 2-파라미터 선형 모델이다. 이를 비선형 정렬이나 **multi-view consistent depth estimation**으로 확장하면 장면 간 기하학적 일관성이 크게 향상될 수 있다.

#### (d) 적응적 메시 필터링
고정 임계값 $\delta_{sn}, \delta_{edge}$ 대신 학습 기반 또는 적응적 필터링을 도입하면, 다양한 장면 구조에 대해 더 나은 일반화가 가능하다.

#### (e) 재질-조명 분해
PBR(physically-based rendering) 재질과 조명을 분리하면, 생성된 장면의 재조명(relighting)이 가능해져 다양한 환경에서의 활용성이 높아진다.

---

## 4. 향후 연구에 미치는 영향과 고려할 점

### 4.1 연구적 영향

1. **2D-to-3D 리프팅 패러다임의 확립**: Text2Room은 "대규모 2D 생성 모델의 지식을 3D로 전이"하는 접근법이 단일 객체를 넘어 **장면 수준**에서도 유효함을 최초로 입증하였다. 이후 SceneScape, LucidDreamer, WonderWorld 등 다수의 후속 연구에 영감을 주었다.

2. **명시적 3D 표현의 가치 재확인**: NeRF/Gaussian Splatting 기반 암시적 표현이 주류인 가운데, Text2Room은 **메시 표현**이 상용 하드웨어 렌더링, AR/VR 자산 생성 등 실용적 활용에서 여전히 중요함을 보여주었다.

3. **대규모 3D 콘텐츠 생성의 민주화**: 전문 모델링 기술 없이 텍스트만으로 3D 장면을 생성할 수 있다는 가능성을 제시하여, 3D 콘텐츠 생성의 접근성을 크게 높였다.

### 4.2 향후 연구 시 고려할 점

1. **다중 시점 일관성(Multi-view Consistency)**: 현재 프레임별 독립적인 인페인팅은 누적 오류를 야기할 수 있으므로, 3D-aware diffusion model이나 multi-view diffusion을 활용한 일관성 보장이 중요하다.

2. **의미론적 제어(Semantic Control)**: 현재 레이아웃 제어가 카메라 포즈에 의존적이므로, 플로어플랜(floor plan)이나 바운딩 박스 등 명시적 레이아웃 조건을 도입하는 연구가 필요하다.

3. **확장성(Scalability)**: 건물 전체나 도시 규모로의 확장 시 메모리 효율적인 메시 관리와 계층적 생성 전략이 필요하다.

4. **품질-속도 트레이드오프**: 장면당 ~50분의 생성 시간을 줄이기 위한 효율적 추론 전략 (e.g., 병렬 인페인팅, fewer iterations)이 필요하다.

5. **윤리적 고려**: Stable Diffusion의 학습 데이터 편향과 유해 콘텐츠 생성 가능성을 상속하므로, 안전 필터링과 편향 완화 전략이 필요하다.

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

| 연구 | 연도 | 핵심 접근법 | Text2Room 대비 차이점 |
|---|---|---|---|
| **DreamFusion** (Poole et al.) | 2022 | SDS loss로 NeRF 최적화, 단일 객체 생성 | 단일 객체 한정; room-scale 불가; 메시 직접 생성 불가 |
| **Magic3D** (Lin et al.) | 2022 | 2단계 coarse-to-fine NeRF→메시 최적화 | 고해상도 단일 객체; 장면 수준 확장 어려움 |
| **SceneScape** (Fridman et al.) | 2023 | Text-driven perpetual view generation, zoom-out 비디오 | Forward-facing 궤적 한정; 완전한 3D 메시 생성 불가 |
| **MVDream** (Shi et al.) | 2023 | Multi-view diffusion prior로 3D 일관성 향상 | 단일 객체 수준; scene-level 확장 미고려 |
| **LucidDreamer** (Chung et al.) | 2023 | 3D Gaussian Splatting + point cloud 기반 장면 생성 | Text2Room과 유사한 iterative 패턴이나, Gaussian 표현 사용; 메시 아님 |
| **SceneDreamer** (Chen et al.) | 2023 | Unconditional scene generation via BEV representation | 실외 자연 장면 초점; 텍스트 조건 없이 학습 기반 |
| **Set-the-Scene** (Cohen-Bar et al.) | 2023 | 여러 NeRF 객체를 장면에 배치 | 사전 정의된 객체별 최적화 필요; 자동 레이아웃 생성 제한 |
| **WonderJourney** (Yu et al.) | 2023 | LLM-guided perpetual 3D scene generation | 연속 장면 탐색 초점; 단일 폐쇄 공간 메시 생성과는 다른 목적 |
| **Ctrl-Room** (Fang et al.) | 2023 | 레이아웃 조건부 room generation (panorama + mesh) | 레이아웃 제어 가능하나, 단일 파노라마 기반으로 가려진 영역 제한 |
| **RoomDreamer** (Song et al.) | 2023 | 기존 3D 장면의 스타일 편집 | 기존 기하학 필요; zero-shot 장면 생성 불가 |
| **Text2Immersion** (Ouyang et al.) | 2023 | Large 3D Gaussian field from text | Gaussian Splatting 기반; 메시 아님; 대규모 장면 가능 |

### 핵심 비교 인사이트

1. **표현 방식**: Text2Room은 **명시적 메시**를 사용하여 상용 렌더링 파이프라인과의 호환성이 높다. 반면 DreamFusion, LucidDreamer 등은 NeRF/Gaussian Splatting 같은 암시적/반암시적 표현을 사용한다.

2. **장면 규모**: Text2Room은 room-scale 메시를 생성하는 **최초의 방법**이며, 대부분의 text-to-3D 방법은 단일 객체에 한정된다. SceneScape는 장면을 다루지만 완전한 3D 메시를 생성하지 않는다.

3. **3D 학습 데이터 불필요**: Text2Room, DreamFusion, SceneScape 등은 사전 학습된 2D 모델만을 활용하므로, 3D 데이터셋에 의존하지 않아 일반화 가능성이 높다. 반면 GAUDI, SceneDreamer 등은 특정 3D 데이터셋에서 학습하므로 도메인이 제한된다.

4. **깊이 정렬의 중요성**: Text2Room이 제안한 scale-shift 깊이 정렬(수식 5)은 후속 연구에서도 널리 채택되는 핵심 기법이 되었다.

---

## 참고자료

1. Höllein, L., Cao, A., Owens, A., Johnson, J., & Nießner, M. (2023). "Text2Room: Extracting Textured 3D Meshes from 2D Text-to-Image Models." *arXiv:2303.11989v2*. https://arxiv.org/abs/2303.11989
2. Poole, B., Jain, A., Barron, J. T., & Mildenhall, B. (2022). "DreamFusion: Text-to-3D using 2D Diffusion." *arXiv:2209.14988*.
3. Lin, C.-H., et al. (2022). "Magic3D: High-Resolution Text-to-3D Content Creation." *arXiv:2211.10440*.
4. Fridman, R., et al. (2023). "SceneScape: Text-Driven Consistent Scene Generation." *arXiv:2302.01133*.
5. Rombach, R., et al. (2022). "High-Resolution Image Synthesis with Latent Diffusion Models." *CVPR 2022*.
6. Bae, G., Budvytis, I., & Cipolla, R. (2022). "IronDepth: Iterative Refinement of Single-View Depth Using Surface Normal and Its Uncertainty." *BMVC 2022*.
7. Kazhdan, M., Bolitho, M., & Hoppe, H. (2006). "Poisson Surface Reconstruction." *Eurographics Symposium on Geometry Processing*.
8. Liu, A., et al. (2021). "Infinite Nature: Perpetual View Generation of Natural Scenes from a Single Image." *ICCV 2021*.
9. Bautista, M. A., et al. (2022). "GAUDI: A Neural Architect for Immersive 3D Scene Generation." *NeurIPS 2022*.
10. Chung, J., et al. (2023). "LucidDreamer: Domain-free Generation of 3D Gaussian Splatting Scenes." *arXiv:2311.13384*.
11. Shi, Y., et al. (2023). "MVDream: Multi-View Diffusion for 3D Generation." *arXiv:2308.16512*.
12. Radford, A., et al. (2021). "Learning Transferable Visual Models From Natural Language Supervision." *ICML 2021* (CLIP).
13. Text2Room 프로젝트 페이지: https://lukashoel.github.io/text-to-room
