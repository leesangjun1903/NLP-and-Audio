# 3D Shape Generation and Completion through Point-Voxel Diffusion

### 1. 핵심 주장 및 주요 기여 요약

**Point-Voxel Diffusion (PVD)** 논문은 3D 형상의 **확률론적 생성 모델링**에 혁신을 가져왔습니다. 논문의 세 가지 핵심 주장은:

1. **통합 확률 프레임워크**: 조건부 형상 완성과 비조건부 형상 생성을 단일 프레임워크에서 수행 가능
2. **다중 모달리티 지원**: 부분적 관찰로부터 다양한 형상 완성 결과 생성으로 인간의 창의성 모방
3. **표현 하이브리드화의 필수성**: 순수 포인트 또는 복셀 표현의 직접 확장 실패, 포인트-복셀 하이브리드가 효과적

주요 기여는 다음과 같습니다:
- 3D 확산 모델의 첫 성공 사례 제시
- 가우시안 노이즈에서 고화질 3D 형상으로의 점진적 생성
- 아키텍처 수정 없이 생성과 완성의 전환 가능
- 실제 RGB-D 스캔 데이터에 대한 적용 입증
- 기존 평가 메트릭의 한계 지적 및 1-NN 메트릭 우월성 주장[1]

***

### 2. 해결하는 문제, 방법론, 모델 구조

#### 2.1 기존 기술의 한계

**복셀 기반 방법**: 메모리 비효율성(분해능 2배 시 메모리 8배), 저해상도로 인한 세부사항 손실, 이진 값으로 인한 확률 모델링 부적합[1]

**포인트 클라우드 기반 방법**: 순열 불변성 제약으로 인한 아키텍처 제한, 결정론적 샘플링(단일 완성), 다중 모달리티 표현 불가능, 확산 모델의 직접 적용 실패[1]

#### 2.2 수학적 기초

**확산 과정 (Forward Process)**:[1]

$$q(x_{0:T}) = q(x_0) \prod_{t=1}^{T} q(x_t | x_{t-1})$$

여기서:
$$q(x_t | x_{t-1}) = \mathcal{N}(\sqrt{1-\beta_t}x_{t-1}, \beta_t I)$$

**생성 과정 (Generative Process)**:[1]

$$p_\theta(x_{0:T}) = p(x_T) \prod_{t=1}^{T} p_\theta(x_{t-1} | x_t)$$

**변분 하한 기반 훈련**:[1]

최종 L2 손실 함수:

$$\mathcal{L}_t = \mathbb{E}_{\epsilon \sim \mathcal{N}(0,I)} \left\| \epsilon - \epsilon_\theta(x_t, t) \|^2 \right\|$$

모델은 현재 형상으로부터 노이즈를 제거하는 데 필요한 노이즈를 예측합니다.

**조건부 형상 완성**:[1]

부분 형상 $$z_0$$이 주어졌을 때, 자유 포인트만 확산/생성:

$$\tilde{q}(x_t | x_{t-1}, x_0, z_0) = \mathcal{N}(\sqrt{1-\beta_t}x_{t-1}, \beta_t I)$$

생성 시 부분 형상은 모든 타임스텝에서 고정 유지되며, 마스킹을 통해 자유 포인트에만 손실 적용:[1]

$$\mathcal{L}_{\text{partial}} = \mathbb{E}_{t,\epsilon} \left\| \epsilon - \epsilon_\theta([x_t, z_0], t) \|^2 \right\|$$

#### 2.3 모델 아키텍처

**백본**: Point-Voxel CNN (Liu et al., 2019)을 기반으로 하며, 포인트 세부사항과 복셀 효율성을 결합[1]

**구성 요소**:[1]
- **Set Abstraction (SA) 모듈 4개**: 계층적 특징 추출, 중심점 점진적 감소 (1024 → 256 → 64 → 16)
- **Feature Propagation (FP) 모듈 4개**: 역순 계층적 정보 전파
- **시간 임베딩**: 정현파 위치 인코딩 (Transformer 기반)
- **Point-Voxel Convolution (PVConv)**: 공간 상관성 활용

**특이한 설계**: 아키텍처를 변경하지 않고 훈련 목적함수만 수정하여 생성과 완성 작업을 전환 가능[1]

***

### 3. 성능 향상 및 한계

#### 3.1 정량적 성능[1]

**ShapeNet 생성 벤치마크** (1-NN 메트릭 with CD/EMD):

| 모델 | Airplane | Chair | Car |
|------|----------|-------|-----|
| PointFlow | 87.30 / 93.95 | 68.58 / 83.84 | 66.49 / 88.78 |
| DPF-Net | 76.05 / 65.80 | 59.21 / 60.05 | 64.77 / 60.09 |
| Shape-GF | 75.18 / 65.55 | 62.00 / 58.53 | 62.35 / 54.48 |
| **PVD** | **73.82 / 64.81** | **56.26 / 53.32** | **54.55 / 53.83** |

EMD 기준으로 최대 10-20% 성능 향상[1]

**형상 완성** (PartNet 다중 모달리티):

| 메트릭 | KNN-latent | CGAN | PVD |
|-------|-----------|------|-----|
| TMD (다양성) | 1.42 | 1.89 | **1.91** |
| MMD (품질) | 1.42 | 1.77 | **1.43** |

PVD는 품질에서 우수하면서도 다양성 유지[1]

#### 3.2 근본적 한계

**메트릭 문제**: CD는 점 밀도 분포를 무시하여 시각적으로 열등한 형상도 높은 점수를 받을 수 있음. PVD의 형상 완성이 CD에서는 낮지만 EMD에서는 높은 이유가 이것[1]

**도메인 갭**: ShapeNet 학습 → PartNet/Redwood 적용 시 성능 저하. 실제 RGB-D 스캔과 합성 데이터 간의 도메인 갭 존재[1]

**생성 속도**: 타임스텝 T=1000 필요로 생성이 느림[1]

**표현의 한계**: 포인트-복셀 하이브리드는 혁신이지만, 고해상도 생성 시 메모리 및 계산 복잡도 증가[1]

***

### 4. 모델 일반화 성능 향상 가능성

#### 4.1 현재 일반화 성능

**범주 내**: ShapeNet의 학습 범주(Airplane, Chair, Car)에서 우수 성능[1]

**범주 간**: PartNet(다른 범주)으로의 전이 시 성능 저하, 다양성 지표는 유지하나 품질 편차 증가[1]

**Sim-to-Real**: Redwood 실제 데이터에서는 부분적 성공만 달성[1]

#### 4.2 향상을 위한 전략

**다중 데이터셋 학습**:

$$\mathcal{L}_{\text{multi}} = \sum_{d \in \text{datasets}} w_d \cdot \mathcal{L}_d(x^d)$$

형상 변동성 증가, 센서 노이즈 다양화, 범주 간 이질성 학습으로 도메인 일반화 개선[2][3][4]

**테스트 타임 적응**:

부분 형상의 특성으로부터 도메인을 추론하고 모델 파라미터 동적 조정:
$$p_\theta(x_{t-1}|x_t, z_0; \Theta_{\text{adapted}})$$

이는 최근 3D 객체 검출에서 효과 입증[3][5]

**아키텍처 개선**:

1. 더 강력한 포인트-복셀 표현 (옥트리 기반)[6]
2. 조건부 인코더 강화로 부분 형상 특성 명시 학습
3. Cross-Attention 메커니즘으로 부분-전체 관계 모델링[7]

**자기 감독 보조 작업**:

- 포인트 밀도 예측: 센서 특성 학습
- 노이즈 레벨 인식: 노이즈 특성에 민감화
- 불완전성 정도 예측: 부분 형상 복잡도 학습

#### 4.3 근본적 제약

**이론적 한계**: 확산 모델의 정규 분포 가정이 크게 다른 도메인에서 성능 보장 불가[2]

**데이터 수집 비용**: 다중 센서/환경의 실제 데이터 수집이 비용이 많이 듦

**계산 복잡도**: T단계 필요로 다중 데이터셋 학습 시 선형 증가[8]

***

### 5. 연구에 미치는 영향과 고려사항

#### 5.1 패러다임 전환의 영향

**직접 후속 연구** (2021-2025):[9][10][6]

- **LION** (2022): PVD의 포인트-복셀을 잠재 공간으로 확장
- **Diffusion-SDF** (2023): 조건부 생성을 암묵적 표현에 적용
- **OctFusion** (2024): 옥트리 기반 계층적 구조로 고해상도 달성
- **HierOctFusion** (2025): 부분-전체 계층 모델링 명시화

모든 최신 논문이 확률론적 모델링과 다중 모달리티 완성을 표준으로 채택[6][9]

**개념적 기여**: 3D 생성을 확률론적 문제로 인식하게 함으로써, 불확실성 정량화와 다양성 모델링이 가능해짐[11]

#### 5.2 앞으로의 연구 시 고려사항

**표현 혁신**:

현재: 포인트-복셀
미래: 메시(위상 제어 가능), SDF(부드러운 표면), 신경 암묵적 표현(유연성)[12][13][14]

**평가 메트릭 재정의**:

학습 기반 메트릭, 기하학적 메트릭, 다중 메트릭 앙상블로 인간 지각과의 상관성 증대[15][16]

**도메인 일반화**:

메타 학습, 테스트 타임 적응, 다중 소스 도메인 학습[17][18][5]

**효율성 향상**:

DDIM(타임스텝 50-100), Consistency Models(한 번의 샘플링), 옥트리 기반 계층적 생성으로 O(n²) → O(n) 복잡도[19][9][2]

***

### 6. 2020년 이후 관련 최신 연구 탐색

#### 6.1 시간대별 진화[10][20][9][11][6][2]

**2021년**: PVD, Shape-GF, 포인트 클라우드 확산 모델 등장[20][2]

**2022년**: LION(잠재 공간 확대), Point-E(다중 단계 조건부 생성)[6]

**2023년**: Diffusion-SDF, IC3D(이미지 조건화), SDF 기반 다양한 변형[21][13]

**2024년**: OctFusion(옥트리 1024³ 고해상도), TripoSG(직류 흐름)[22][12]

**2025년**: OctGPT(자기회귀 결합), HierOctFusion(부분-전체 계층), Scaling Diffusion Mamba(선형 복잡도)[23][9][6]

#### 6.2 멀티모달 3D 생성[24][25][26]

**텍스트-3D**: SCDiff(형상-색상 분리), Fun3D(물리 호환), Phidias(참조 증강)[24][7]

**이미지-3D**: Shape from Semantics(다중 뷰 의미론), SPGen(구면 투영)[25][27]

**스케치-3D**: VRSketch2Shape(시간 순서 정보)[26]

#### 6.3 도메인 일반화[4][5][28][17]

**점 밀도 재샘플링**: 센서별 밀도 차이로 인한 도메인 갭 해결[3]

**메타 학습**: MLDGG로 다중 도메인 빠른 적응[18]

**테스트 타임 적응**: 3D 객체 검출에서 검증된 기법[5][3]

#### 6.4 의료/응용 확대[29]

**의료 형상 완성**: 간, 척추 형상 증강 및 임상 평가[30][29]

**농업**: LiDAR 센서 간 도메인 갭 해결[28]

**로봇공학**: 부분 감지로부터의 객체 인식 개선

***

### 결론

PVD는 2021년의 논문이지만, **확산 모델 기반 3D 생성**이라는 새로운 패러다임을 수립했습니다. 포인트-복셀 하이브리드 표현, 확률론적 모델링, 통합 프레임워크라는 세 가지 혁신은 이후 수백 편의 후속 연구의 영감이 되었습니다.[9][11][6]

현재의 도전은 **도메인 일반화**, **계산 효율성**, **평가 메트릭 개선**입니다. 2024-2025년의 최신 연구들은 이러한 문제를 해결하는 방향으로 진화 중이며, 의료, 로봇공학, 콘텐츠 제작 등 다양한 실제 응용으로의 확대가 진행 중입니다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/0aa870eb-827f-47a8-98cb-4ef4c9b31f3b/2104.03670v3.pdf)
[2](https://ieeexplore.ieee.org/document/9578791/)
[3](https://arxiv.org/abs/2402.04967)
[4](https://www.mdpi.com/1424-8220/25/3/767)
[5](https://arxiv.org/html/2311.10845v2)
[6](https://dl.acm.org/doi/10.1145/3721238.3730601)
[7](https://ieeexplore.ieee.org/document/10855435/)
[8](https://www.semanticscholar.org/paper/95f5bafba97beb9b4f8c1fe607f04ec28efab7f9)
[9](https://ojs.aaai.org/index.php/AAAI/article/view/34144)
[10](https://arxiv.org/abs/2501.12202)
[11](https://arxiv.org/html/2506.22678v1)
[12](https://arxiv.org/html/2408.14732v1)
[13](http://arxiv.org/pdf/2211.13757.pdf)
[14](https://arxiv.org/html/2407.21428)
[15](https://openreview.net/pdf?id=0fJKg3fX41)
[16](https://www.nature.com/articles/s42004-024-01233-z)
[17](https://ieeexplore.ieee.org/document/10341614/)
[18](https://arxiv.org/pdf/2411.12913.pdf)
[19](http://arxiv.org/pdf/2408.06693.pdf)
[20](https://ieeexplore.ieee.org/document/9711332/)
[21](https://arxiv.org/abs/2305.04461)
[22](https://arxiv.org/html/2502.06608)
[23](https://arxiv.org/abs/2508.11106)
[24](https://ieeexplore.ieee.org/document/11227150/)
[25](https://arxiv.org/abs/2502.00360)
[26](https://www.semanticscholar.org/paper/53d8286752f3de332ea6db809de4b3025173d405)
[27](https://www.semanticscholar.org/paper/4306ae5f669f5a7726e0ba0e4c303a7ad3ced98c)
[28](https://www.nature.com/articles/s41598-025-26225-4)
[29](https://www.nature.com/articles/s41598-024-68084-5)
[30](https://arxiv.org/abs/2504.19402)
[31](https://arxiv.org/pdf/2210.06978.pdf)
[32](https://arxiv.org/abs/2212.00842)
[33](https://arxiv.org/pdf/2305.15399.pdf)
[34](https://arxiv.org/abs/2211.10865)
[35](https://arxiv.org/html/2409.11406)
[36](https://www.sciencedirect.com/science/article/abs/pii/S0263224124021316)
[37](https://openaccess.thecvf.com/content_ICCV_2017/papers/Han_High-Resolution_Shape_Completion_ICCV_2017_paper.pdf)
[38](https://www.sciencedirect.com/science/article/abs/pii/S0957417425035468)
[39](https://s-space.snu.ac.kr/handle/10371/177571)
[40](https://onlinelibrary.wiley.com/doi/10.1111/cgf.70198?af=R)
[41](https://www.sciencedirect.com/science/article/pii/S119510362500028X)
[42](https://www.cvlibs.net/publications/Stutz2018CVPR.pdf)
[43](https://www.semanticscholar.org/paper/91b32fc0a23f0af53229fceaae9cce43a0406d2e)
[44](https://www.semanticscholar.org/paper/c940509c5b1ee8db9e4ce70254726719b8d56c54)
[45](https://bmcmedimaging.biomedcentral.com/articles/10.1186/s12880-021-00703-3)
[46](https://www.researchsquare.com/article/rs-956250/v1)
[47](https://ieeexplore.ieee.org/document/9704981/)
[48](https://link.springer.com/10.1007/978-3-030-85540-6_109)
[49](https://onepetro.org/SPERPTC/proceedings/21RPTC/21RPTC/D021S012R001/470455)
[50](https://arxiv.org/html/2401.17603v1)
[51](https://arxiv.org/html/2403.19773)
[52](https://arxiv.org/pdf/2308.07837.pdf)
[53](https://openaccess.thecvf.com/content/CVPR2021/papers/Luo_Diffusion_Probabilistic_Models_for_3D_Point_Cloud_Generation_CVPR_2021_paper.pdf)
[54](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/08142.pdf)
[55](https://openaccess.thecvf.com/content/WACV2025/papers/Li_ShapeMorph_3D_Shape_Completion_via_Blockwise_Discrete_Diffusion_WACV_2025_paper.pdf)
[56](https://arxiv.org/html/2403.10085)
[57](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/04117.pdf)
[58](https://www.nature.com/articles/s41597-025-04897-x)
[59](https://academic.oup.com/bioinformatics/article/41/8/btaf426/8219452)
[60](https://velog.io/@guts4/Diffusion-Probabilistic-Models-for-3D-Point-Cloud-GenerationCVPR-2021)
[61](https://ieeexplore.ieee.org/document/10779892/)
[62](https://ieeexplore.ieee.org/document/11229087/)
[63](https://arxiv.org/abs/2406.14994)
[64](https://www.mdpi.com/1424-8220/23/17/7312)
[65](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/12926/3007664/Deep-implicit-statistical-shape-models-for-3D-lumbar-vertebrae-image/10.1117/12.3007664.full)
[66](https://ieeexplore.ieee.org/document/11142278/)
[67](https://www.aclweb.org/anthology/2020.findings-emnlp.329)
[68](https://arxiv.org/pdf/2211.14058.pdf)
[69](https://arxiv.org/html/2503.06282v1)
[70](http://arxiv.org/pdf/2405.08586.pdf)
[71](http://arxiv.org/pdf/2004.05749.pdf)
[72](http://arxiv.org/pdf/2307.13492.pdf)
[73](https://arxiv.org/pdf/2203.17067.pdf)
[74](https://www.ijcai.org/proceedings/2018/0115.pdf)
[75](https://pmc.ncbi.nlm.nih.gov/articles/PMC9920750/)
[76](https://www.sciencedirect.com/topics/computer-science/generalization-performance)
[77](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136860266.pdf)
[78](https://arxiv.org/pdf/2504.01659.pdf)
[79](https://pmc.ncbi.nlm.nih.gov/articles/PMC10508770/)
[80](https://proceedings.neurips.cc/paper_files/paper/2024/file/6b7e1e96243c9edc378f85e7d232e415-Paper-Conference.pdf)
[81](https://onlinelibrary.wiley.com/doi/10.1111/1365-2478.70020)
[82](https://www.sciencedirect.com/science/article/abs/pii/S0167865525002417)
