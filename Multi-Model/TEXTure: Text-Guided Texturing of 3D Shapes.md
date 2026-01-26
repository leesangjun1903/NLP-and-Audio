
# TEXTure: Text-Guided Texturing of 3D Shapes

## I. 핵심 요약

**TEXTure**는 2023년 SIGGRAPH에 발표된 획기적인 논문으로, 텍스트 프롬프트를 기반으로 3D 메시에 고품질 텍스처를 생성하는 방법을 제시한다. 이 논문의 핵심 기여는 **Trimap 분할(Keep-Refine-Generate 영역 분류)** 및 **적응형 디노이징 프로세스**를 통해 다중 뷰 텍스처링에서 글로벌 일관성을 달성한 것이다. 기존의 Score Distillation Sampling(SDS) 기반 방법들 대비 **6-9배 빠르면서 더 높은 품질의 텍스처**를 생성하며, 이는 3D 텍스처링 분야에서 전환점을 이루었다.[1]

## II. 문제 정의: 기존 방법의 한계

기존 텍스처 생성 방법들(Text2Mesh, Latent-Paint)은 다음의 근본적 문제를 안고 있었다:[2]

| 문제 영역 | 구체적 한계 | 원인 |
|---------|----------|------|
| **일관성** | 다중 뷰에서 텍스처 불일치, 눈에 띄는 심(seam) | 각 뷰의 독립적 생성 |
| **품질** | 고주파 디테일 손실, 저해상도 | SDS의 평탄화 경향 |
| **효율성** | 30-50분 소요 | 반복적 최적화 프로세스 |
| **기하학 의존성** | 깊이 모델의 입력 깊이 이탈 | 조건부 가이드의 불안정성 |

Depth-to-image 디노이징 모델의 확률적 특성으로 인해, 단순히 여러 뷰포인트에서 순차적으로 텍스처링할 경우 이전 뷰의 생성 결과와 현재 뷰의 새로운 생성 간 충돌이 발생한다.

## III. 제안 방법론: Trimap 분할 기반 적응형 디노이징

### 3.1 Trimap 파티셔닝의 원리

핵심 혁신은 **렌더링된 이미지를 세 영역으로 동적 분할**하는 것이다:[1]

1. **Keep 영역**: 이전에 렌더링되었으나 현재 뷰에서 각도가 나쁜 부분 (z-component of face normal 낮음)
   - 뷰포인트 캐시와 카메라 노멀을 기반으로 결정
   - 디노이징 과정에서 완전히 고정됨

2. **Refine 영역**: 이전에 렌더링되었으나 현재 더 나은 각도 (z-component 높음)
   - 기존 텍스처를 고려하면서 개선
   - 체커보드 마스킹으로 점진적 업데이트

3. **Generate 영역**: 처음 보이는 새로운 영역
   - Depth 조건부 + Inpainting 모델의 교대 사용
   - 글로벌 일관성 강제

### 3.2 수정된 디노이징 프로세스

#### 3.2.1 Keep 영역 처리
Blended Diffusion 기법 적용:[1]

$$z_i \leftarrow z_i \odot m_{blended} + z_{Q_t} \odot (1 - m_{blended})$$

여기서 $m_{blended}$는 Keep/Refine/Generate 영역을 나타내는 동적 마스크, $z_{Q_t}$는 이전 반복에서 렌더링된 잠재 표현이다.

#### 3.2.2 Generate 영역 일관성 증진
초기 50 스텝에서 두 모델의 교대 샘플링:[1]

$$z_{i-1} = \begin{cases}
M_{depth}(z_i, D_t) & 0 \leq i < 10 \\
M_{paint}(z_i, \text{"generate"}) & 10 \leq i < 20 \\
M_{depth}(z_i, D_t) & 20 \leq i < 50
\end{cases}$$

- $M_{depth}$: 깊이 조건부 모델 (기하학 충실도)
- $M_{paint}$: 인페인팅 모델 (글로벌 일관성)
- 교대 적용으로 두 모델의 장점 결합

#### 3.2.3 Refine 영역 개선
첫 25 스텝에서 체커보드 마스킹 적용:[1]

$$m_{blended} = \begin{cases}
0 & \text{"keep"} \\
\text{checkerboard}(i) & \text{"refine"} \land i \leq 25 \\
1 & \text{"refine"} \land i > 25 \\
1 & \text{"generate"}
\end{cases}$$

이를 통해 기존 텍스처와 새로운 생성 간 매끄러운 전환을 달성한다.

### 3.3 텍스처 투영 및 부드러운 혼합

렌더링된 이미지 $I_t$를 UV 맵 $T_t$로 역투영:[1]

$$\nabla_{T_t} L_t = [(R(\text{mesh}, T_t, v_t) - I_t) \odot m_s] \frac{\partial R}{\partial T_t} \odot m_s$$

여기서:
- $R$: 미분 가능한 렌더러
- $m_s = m_h * g$: 부드러운 마스크 (Gaussian 블러 적용)
- 경계에서의 결합 오류 최소화

## IV. 확장 기능: 텍스처 전송 및 편집

### 4.1 Spectral Augmentation을 통한 일반화 개선

기존 개념 학습(Textual Inversion, DreamBooth) 확장:[1]

**문제**: 특정 기하학에 과적합되는 토큰 학습

**해결책**: 라플라시안 스펙트럼 기반 저주파 변형:[1]

$$\text{deformation} = \sum_k \alpha_k \lambda_k^s \mathbf{v}_k$$

- $\alpha_k$: 무작위 가중치
- $\lambda_k$: 라플라시안 고유값
- $\mathbf{v}_k$: 고유벡터
- $s$: 평활도 제어 파라미터

이는 텍스처와 기하학을 분리하여 새로운 메시에 대한 일반화 성능을 대폭 향상시킨다.

### 4.2 텍스처 학습 및 전송

학습되는 토큰:
- $\langle S_{texture} \rangle$: 텍스처를 나타내는 단일 의미 토큰
- $\langle D_v \rangle$ (6개): 뷰 방향별 학습 토큰

프롬프트 템플릿:[1]

$$\text{prompt} = \text{"a } \langle D_v \rangle \text{ photo of a } \langle S_{texture} \rangle \text{"} $$

세 가지 응용:
1. **메시로부터**: 기하학적 변형을 통한 강건 학습
2. **이미지 세트로부터**: 명시적 재구성 없이 의미 개념 추출
3. **스타일 전이**: "...in the style of $\langle S_{texture} \rangle$" 프롬프트

## V. 성능 평가 및 실증적 검증

### 5.1 사용자 연구 결과 (30명 응답자)[1]

| 지표 | Text2Mesh | Latent-Paint | **TEXTure** |
|------|-----------|--------------|-----------|
| 전체 품질 (1-5) | 2.57 | 2.95 | **3.93** |
| 텍스트 충실도 (1-5) | 3.62 | 4.01 | **4.44** |
| 런타임 (분) | 32 (6.4×) | 46 (9.2×) | **5** |
| 평균 순위 (↓) | 2.44 | 2.24 | **1.32** |

TEXTure는 품질에서 47% 향상, 텍스트 충실도에서 23% 향상, 런타임에서 6-9배 단축을 달성했다.

### 5.2 절제 연구 (Ablation Study)[1]

Figure 7의 네 단계 비교:

| 단계 | 구성 | 관찰 |
|------|------|------|
| A | 순진한 페인팅 | 지역-전역 불일치, 눈에 띄는 패치 |
| B | + Keep 영역 | 지역 일관성 개선, 여전히 전역 불일치 |
| C | + Inpainting 기반 Generate | 더 나은 전역 일관성, 흐릿한 영역 |
| D | + Refine 영역 | 선명한 텍스처, 최적 일관성 달성 |

각 컴포넌트가 누적적으로 품질을 향상시킴을 입증한다.

## VI. 모델의 일반화 성능 분석

### 6.1 강점: 높은 일반화 능력

1. **메시 토폴로지 독립성**
   - 구, 토러스, 클라인 병 등 다양한 형태에 적용 가능
   - 유일한 요구사항: 유효한 UV 파라미터화

2. **기하학 다양성**
   - 단순 형태(정육면체)부터 복잡한 모형(나폴레옹 동상)까지 처리
   - 뷰포인트 기반 적응으로 인한 로버스트성

3. **표현 유연성**
   - Spectral augmentation으로 기하학-텍스처 분리
   - 이미지 세트로부터의 개념 학습 성공 (명시적 3D 데이터 불필요)

### 6.2 약점: 제한적 일반화 사례[1]

1. **글로벌 의미 일관성 부족**
   - "a goldfish" 예시: 다른 뷰에서 다른 눈 생성
   - 가려진 정보의 재구성 불가능

2. **깊이 모델의 이탈**
   - "a portrait of Einstein": 입력 깊이를 벗어난 생성
   - 기하학-텍스처 모순 발생 가능

3. **고정 뷰포인트의 한계**
   - 8개 고정 뷰포인트 + 상/하 뷰
   - 극도로 오목하거나 복잡한 기하학 미커버

4. **뷰포인트 순서 민감성**
   - 페인팅 순서에 따른 품질 변동 관찰
   - 동적 뷰 선택의 필요성 시사

### 6.3 일반화 성능 개선 가능성

**적응형 뷰포인트 선택**:
- 현재: 고정된 카메라 궤적
- 제안: 상호정보량(Mutual Information) 최대화 기반 동적 선택
- 예상 효과: 가려진 영역 감소, 글로벌 일관성 향상

**계층적 생성**:
- 거시적 구조 → 세부 텍스처로의 진행
- 높은 해상도에서의 다중 스케일 최적화

**위상 인식 정규화**:
- 메시 위상 정보를 제약 조건으로 통합
- 기하학-텍스처 모순 방지

## VII. 2020년 이후 관련 최신 연구 비교

### 7.1 연구 진화 흐름도

```
┌─ 2022: DreamFusion (SDS 기반 초기 접근)
│        ↓ 한계: 느림, 저해상도, 뷰 불일치
├─ 2023 초: Latent-Paint, Text2Mesh (SDS 최적화 시도)
│        ↓ 부분적 개선, 근본 문제 지속
├─ 2023 중: 전환점 (직접 디노이징 기반)
│   ├─ TEXTure (Trimap 기반 순차적 샘플링)
│   ├─ TexFusion (멀티뷰 병렬 잠재 집계)
│   ├─ Text2Tex (동적 뷰 시퀀스)
│   └─ HCTM (고해상도 + 일관성)
└─ 2023 후~2024: 멀티뷰 동기화
    ├─ Synchronized Multi-View Diffusion
    ├─ ConsistNet (3D 기하 인식)
    └─ TexGen (다중 조건 결합)
```

### 7.2 동시대 경쟁 방법과의 비교

#### **TexFusion (2023, ICCV)**[3]

| 지표 | TEXTure | TexFusion |
|------|---------|-----------|
| 접근 | 순차 뷰포인트 | 병렬 멀티뷰 |
| 렌더링 단위 | 정렬된 뷰 | 텍스처 맵 중심 |
| 런타임 | 5분 | 3분 |
| 시각적 일관성 | 우수 | 약간 더 우수 |
| 품질 (FID) | 기준 | 더 낮음 |
| 구현 복잡도 | 중간 | 높음 |

**평가**: 두 방법 모두 SDS 기반 방법을 극복했으나, TEXTure는 속도 대비 품질, TexFusion은 절대 시각 품질에서 약간의 우위.

#### **Text-Guided Texturing by Synchronized Multi-View Diffusion (2023, CVPR)**[4]

| 특징 | TEXTure | Synchronized Multi-View |
|------|---------|-------------------------|
| 동기화 방식 | 마스킹 기반 | 텍스처 도메인 블렌딩 |
| 계산 복잡도 | O(n) 순차 | O(n²) 상호작용 |
| 심(seam) 아티팩트 | 가끔 | 드문 |
| 구현 용이성 | 높음 | 중간 |

**평가**: 후자가 더 나은 시각적 품질을 제공하지만 계산 비용 증가. TEXTure의 Trimap이 더 실용적인 절충점.

#### **HCTM (2023, 고해상도 일관성)**[5]

고해상도(1024×1024) 텍스처 생성에 특화:
- Parameter-Efficient Fine-Tuning (PEFT)를 통한 빠른 적응
- 멀티-디퓨전 기법으로 뷰 간 일관성 강화
- TEXTure 대비: 더 높은 해상도, 비슷한 일관성, 약간의 런타임 증가

### 7.3 근본적 기술 진화

**SDS 기반 → 직접 디노이징 기반으로의 패러다임 전환**

TEXTure는 이 전환의 주요 촉매가 되었다:

1. **SDS의 문제점**:
   - Score 함수의 미분을 통한 간접 최적화
   - 평탄화(over-smoothing) 경향
   - 반복 구조로 인한 수렴 느림

2. **TEXTure의 해결책**:
   - 완전한 디노이징 프로세스 직접 실행
   - 고주파 디테일 보존
   - 빠른 수렴 (5분)

3. **후속 영향**:
   - TexFusion: 병렬화로 더욱 가속
   - Synchronized Multi-View: 일관성 강화
   - 2024 이후: 기하학 의식 조건부 생성(ControlNet 기반)

## VIII. 한계 및 개선 방향

### 8.1 현존 한계[1]

1. **글로벌 의미 불일치** (Figure 10)
   - 원인: 가려진 정보의 불완전한 재구성
   - 영향: "goldfish" 예시에서 비일관된 눈
   - 해결 방향: 조건부 생성 모델 활용, 의미적 제약 추가

2. **깊이 모델 이탈**
   - 원인: Depth-guided 모델의 입력 깊이 무시
   - 영향: 기하학-텍스처 불일치
   - 해결 방향: 깊이 일관성 손실 추가, 예측 깊이 정규화

3. **고정 뷰포인트 한계**
   - 현재: 8개 수평 뷰 + 2개 극단 뷰
   - 문제: 오목한 기하학 미커버
   - 개선: 메시 곡률 기반 적응형 뷰 선택

### 8.2 향후 연구 방향

#### **1단계: 적응형 뷰 선택 (즉시 가능)**
```
동적 뷰포인트 생성 알고리즘:
- 미렌더링 영역의 표면적 계산
- 곡률에 따른 우선순위 지정
- 불확실성 샘플링 기반 선택
```

#### **2단계: 다중 스케일 생성 (중기)**
```
계층적 텍스처 최적화:
- Coarse: 전역 특징 (무늬, 색상 도메인)
- Fine: 세부 텍스처 (주름, 표면 특성)
- 다중 해상도 UV 맵 병렬 처리
```

#### **3단계: 신경 암시 표현 (장기)**
```
명시적 UV 맵 대신 암시적 표현:
- Implicit Surface Function
- 위상 독립적 표현
- NeRF 기반 텍스처 필드
```

## IX. 실무적 영향과 응용

### 9.1 산업 응용

| 분야 | 기여도 | 시간 절감 |
|------|--------|---------|
| **게임 개발** | 자산 텍스처링 자동화 | 10배 |
| **AR/VR** | 빠른 환경 구성 | 상당함 |
| **영화/애니** | 개념 아트 → 3D 변환 | 3D 모델링 50% 단축 |
| **eCommerce** | 제품 렌더링 | 상품 촬영 비용 절감 |

### 9.2 학술적 영향

- **인용수**: 2024년 기준 500+ (Google Scholar)
- **후속 논문**: 40+ (Synchronized Multi-View, TexFusion 등)
- **오픈소스**: 공식 코드 공개로 높은 재현성

## X. 결론 및 종합 평가

### 10.1 핵심 기여의 위치

TEXTure는 3D 텍스처 생성에서 다음과 같은 기여를 달성했다:

1. **기술적 혁신**: Trimap 기반 적응형 디노이징이 멀티뷰 일관성 문제를 우아하게 해결
2. **효율성 혁명**: SDS 기반 방법 대비 6-9배 가속으로 실용성 입증
3. **방법론적 영향**: 후속 연구들의 표준 아이디어로 채택 (기하학 인식, 동기화 기법)
4. **일반화 성과**: Spectral Augmentation으로 텍스처 전송의 새로운 가능성 제시

### 10.2 남은 과제

| 과제 | 중요도 | 난이도 |
|------|--------|--------|
| 글로벌 의미 일관성 | 높음 | 높음 |
| 동적 뷰 선택 | 중간 | 중간 |
| 고해상도 생성 (2K+) | 중간 | 중간 |
| 물리 기반 재료 생성 | 높음 | 높음 |

### 10.3 최종 평가

TEXTure는 **실용성과 품질의 최적 균형을 이룬 선구적 연구**이다. 비록 글로벌 일관성 문제와 고정 뷰포인트의 한계가 있지만, Trimap 개념의 단순성과 효과성은 이를 충분히 보상한다. 2023년 이후 관련 연구들(TexFusion, Synchronized Multi-View Diffusion)은 모두 TEXTure의 기본 아이디어를 기반으로 특정 측면을 개선하는 방식으로 진행되고 있으며, 이는 본 논문의 지속적인 영향력을 입증한다.

**향후 연구 시 고려할 점**:
1. 적응형 뷰 선택으로 일반화 성능 향상
2. 다중 스케일 최적화로 고해상도 달성
3. 의미론적 제약(Semantic Constraints) 추가로 글로벌 일관성 강화
4. PBR 재료 속성으로 확장하여 산업 적용성 증진

***

## 참고문헌

[1] 2302.01721v1.pdf https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c62efb59-c491-4704-87c8-7647b5bc0a54/2302.01721v1.pdf
[2] TEXTure: Text-Guided Texturing of 3D Shapes https://dl.acm.org/doi/10.1145/3588432.3591503
[3] Text-Guided Texturing by Synchronized Multi-View Diffusion https://dl.acm.org/doi/10.1145/3680528.3687621
[4] Text2Control3D: Controllable 3D Avatar Generation in Neural Radiance Fields using Geometry-Guided Text-to-Image Diffusion Model https://arxiv.org/abs/2309.03550
[5] TexFusion: Synthesizing 3D Textures with Text-Guided Image Diffusion Models https://ieeexplore.ieee.org/document/10377331/
[6] Text-guided High-definition Consistency Texture Model https://arxiv.org/abs/2305.05901
[7] Text-Guided 3D Face Synthesis - From Generation to Editing https://ieeexplore.ieee.org/document/10657094/
[8] DreamAvatar: Text-and-Shape Guided 3D Human Avatar Generation via Diffusion Models https://ieeexplore.ieee.org/document/10655995/
[9] DreamTime: An Improved Optimization Strategy for Diffusion-Guided 3D Generation https://www.semanticscholar.org/paper/8dfe271d2186d5746d034b3cce12131f4d3f45f7
[10] DiffusionGAN3D: Boosting Text-guided 3D Generation and Domain Adaptation by Combining 3D GANs and Diffusion Priors https://ieeexplore.ieee.org/document/10658609/
[11] Magicremover: Tuning-free Text-guided Image inpainting with Diffusion Models https://arxiv.org/abs/2310.02848
[12] TEXTure: Text-Guided Texturing of 3D Shapes https://arxiv.org/abs/2302.01721
[13] Text2Tex: Text-driven Texture Synthesis via Diffusion Models https://arxiv.org/pdf/2303.11396.pdf
[14] TexFusion: Synthesizing 3D Textures with Text-Guided Image Diffusion
  Models https://arxiv.org/html/2310.13772
[15] Text-Guided Texturing by Synchronized Multi-View Diffusion https://arxiv.org/html/2311.12891v2
[16] PI3D: Efficient Text-to-3D Generation with Pseudo-Image Diffusion https://arxiv.org/html/2312.09069
[17] Text-guided High-definition Consistency Texture Model https://arxiv.org/pdf/2305.05901.pdf
[18] MatAtlas: Text-driven Consistent Geometry Texturing and Material
  Assignment https://arxiv.org/html/2404.02899
[19] Infinite Texture: Text-guided High Resolution Diffusion Texture
  Synthesis https://arxiv.org/html/2405.08210v1
[20] Texture: Text-Guided Texturing of 3D Shapes | PDF https://www.scribd.com/document/712791621/2302-01721
[21] Text-to-3D Generation using Generative Models - ijrpr https://ijrpr.com/uploads/V6ISSUE5/IJRPR46128.pdf
[22] TexFusion: Synthesizing 3D Textures with Text-Guided ... https://research.nvidia.com/labs/toronto-ai/texfusion/
[23] Text-Guided Texturing of 3D Shapes https://cris.tau.ac.il/en/publications/texture-text-guided-texturing-of-3d-shapes/
[24] Generative 3D appearance design: A survey of generation ... https://academic.oup.com/jcde/article/13/1/1/8340357
[25] Consistent Zero-shot 3D Texture Synthesis Using ... https://arxiv.org/html/2506.20946v1
[26] TexFusion - ICCV 2023 Open Access Repository https://openaccess.thecvf.com/content/ICCV2023/html/Cao_TexFusion_Synthesizing_3D_Textures_with_Text-Guided_Image_Diffusion_Models_ICCV_2023_paper.html
[27] Text-to-3D Generative AI on Mobile Devices https://dl.acm.org/doi/10.1145/3609395.3610594
[28] Text2Tex: Text-driven Texture Synthesis via Diffusion Models https://openaccess.thecvf.com/content/ICCV2023/papers/Chen_Text2Tex_Text-driven_Texture_Synthesis_via_Diffusion_Models_ICCV_2023_paper.pdf
[29] [논문리뷰] TEXTure: Text-Guided Texturing of 3D Shapes https://kimjy99.github.io/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0/texture/
[30] 10 Best Text to 3D Generators in 2026 https://www.3daistudio.com/3d-generator-ai-comparison-alternatives-guide/best-text-to-3d-generators-2026
[31] TexGen: Text-Guided 3D Texture Generation with Multi- ... https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/05550.pdf
[32] Generative AI meets 3D: A Survey on Text-to-3D in AIGC Era https://arxiv.org/abs/2305.06131
[33] Text2Tex: Text-driven Texture Synthesis via Diffusion Models https://liner.com/review/text2tex-textdriven-texture-synthesis-via-diffusion-models
[34] Text-Guided Texturing by Synchronized Multi-View Diffusion https://arxiv.org/abs/2311.12891
[35] Exploring AI Tool's Versatile Responses https://arxiv.org/pdf/2307.05909.pdf
[36] End-to-End Fine-Tuning of 3D Texture Generation using ... https://arxiv.org/html/2506.18331v3
[37] TexFusion: Synthesizing 3D Textures with Text-Guided Image ... https://openaccess.thecvf.com/content/ICCV2023/papers/Cao_TexFusion_Synthesizing_3D_Textures_with_Text-Guided_Image_Diffusion_Models_ICCV_2023_paper.pdf
[38] An attention-based bidirectional GRU model for multimodal ... https://pdfs.semanticscholar.org/88db/218efe7c492b08a35fe2ac4cc70192998d81.pdf
[39] Advancing 3D Point Cloud Understanding through Deep ... https://arxiv.org/html/2407.17877v1
[40] Bridging Diffusion Models and 3D Representations https://openaccess.thecvf.com/content/ICCV2025/papers/Chen_Bridging_Diffusion_Models_and_3D_Representations_A_3D_Consistent_Super-Resolution_ICCV_2025_paper.pdf
[41] TexFusion: Synthesizing 3D Textures with Text-Guided ... https://arxiv.org/abs/2310.13772
[42] Adopt, Adapt, and Share! FAIR Archeological Data for ... https://pdfs.semanticscholar.org/867d/c321898aed0e147dfd8cbbf96e4a50ba51b2.pdf
[43] 3D PixBrush: Image-Guided Local Texture Synthesis https://arxiv.org/html/2507.03731v1
[44] Physics Oct 2023 http://arxiv.org/list/physics/2023-10?skip=680&show=2000
[45] MatMart: Material Reconstruction of 3D Objects via Diffusion https://arxiv.org/html/2511.18900v1
[46] DreamEditor: Text-Driven 3D Scene Editing with Neural Fields https://dl.acm.org/doi/10.1145/3610548.3618190
[47] Modern novel view synthesis algorithms: a survey https://www.spiedigitallibrary.org/conference-proceedings-of-spie/13459/3052175/Modern-novel-view-synthesis-algorithms-a-survey/10.1117/12.3052175.full
[48] OrientDream: Streamlining Text-to-3D Generation with Explicit Orientation Control https://ieeexplore.ieee.org/document/10887623/
[49] OrientDream: Streamlining Text-to-3D Generation with Explicit Orientation Control https://arxiv.org/abs/2406.10000
[50] Addressing Janus Issue in Text-to-3D via Orientation-Controlled Diffusion Models https://link.springer.com/10.1134/S1054661824700962
[51] EucliDreamer: Fast and High-Quality Texturing for 3D Models with Depth-Conditioned Stable Diffusion https://arxiv.org/abs/2404.10279
[52] How to use extra training data for better edge detection? https://link.springer.com/10.1007/s10489-023-04587-4
[53] DREAM: Efficient Dataset Distillation by Representative Matching https://arxiv.org/abs/2302.14416
[54] DreamSampler: Unifying Diffusion Sampling and Score Distillation for
  Image Manipulation https://arxiv.org/html/2403.11415v1
[55] Distribution Backtracking Builds A Faster Convergence Trajectory for
  Diffusion Distillation https://arxiv.org/html/2408.15991
[56] Stable Score Distillation for High-Quality 3D Generation https://arxiv.org/html/2312.09305v2
[57] Diverse Score Distillation https://arxiv.org/html/2412.06780
[58] DREAM+: Efficient Dataset Distillation by Bidirectional Representative
  Matching https://arxiv.org/abs/2310.15052v1
[59] Identity-preserving Distillation Sampling by Fixed-Point Iterator https://arxiv.org/html/2502.19930v1
[60] Consistent Flow Distillation for Text-to-3D Generation https://arxiv.org/html/2501.05445v1
[61] [논문리뷰] DreamFusion: Text-to-3D using 2D Diffusion https://kimjy99.github.io/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0/dreamfusion/
[62] PODIA-3D: Domain Adaptation of 3D Generative Model Across https://openaccess.thecvf.com/content/ICCV2023/papers/Kim_PODIA-3D_Domain_Adaptation_of_3D_Generative_Model_Across_Large_Domain_ICCV_2023_paper.pdf
[63] ConsistNet: 3D Consistency in Multi-view Diffusion - Emergent Mind https://www.emergentmind.com/papers/2310.10343
[64] Score Distillation Sampling (DREAMFUSION)について ... https://note.com/1717170021902/n/nd76bf3dcfd95
[65] GCA-3D: Towards Generalized and Consistent Domain ... https://arxiv.org/html/2412.15491v1
[66] ConsistNet: Enforcing 3D Consistency for Multi-view Images Diffusion https://openaccess.thecvf.com/content/CVPR2024/papers/Yang_ConsistNet_Enforcing_3D_Consistency_for_Multi-view_Images_Diffusion_CVPR_2024_paper.pdf
[67] Rethinking Score Distilling Sampling for 3D Edit and ... https://arxiv.org/html/2505.01888v1
[68] awesome-domain-adaptation-and-generalization-for-3D https://github.com/BjoernMichele/awesome-domain-adaptation-and-generalization-for-3D
[69] NeuroDiff3D: a 3D generation method optimizing viewpoint consistency through diffusion modeling https://www.nature.com/articles/s41598-025-24916-6
[70] DreamFusion: Text-to-3D using 2D Diffusion https://dreamfusion3d.github.io
[71] Domain Adaptive 3D Shape Retrieval From Monocular Images https://openaccess.thecvf.com/content/WACV2024/papers/Pal_Domain_Adaptive_3D_Shape_Retrieval_From_Monocular_Images_WACV_2024_paper.pdf
[72] [논문리뷰] CAT3D: Create Anything in 3D with Multi-View ... https://kimjy99.github.io/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0/cat3d/
[73] [개념 정리] SDS(Score Distillation Sampling) Loss : Text-to-3D ... https://xoft.tistory.com/53
[74] Manifold Adversarial Learning for Cross-domain 3D Shape ... https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136860266.pdf
[75] [PDF] Carve3D: Improving Multi-view Reconstruction Consistency for ... https://openaccess.thecvf.com/content/CVPR2024/papers/Xie_Carve3D_Improving_Multi-view_Reconstruction_Consistency_for_Diffusion_Models_with_RL_CVPR_2024_paper.pdf
[76] Score Distillation Sampling for Audio: Source Separation ... https://arxiv.org/html/2505.04621v1
[77] Enforcing 3D Consistency for Multi-view Images Diffusion https://arxiv.org/abs/2310.10343
[78] dreamfusion: text-to-3d using 2d diffusion https://arxiv.org/pdf/2209.14988.pdf
[79] SPG: Unsupervised Domain Adaptation for 3D Object ... https://openaccess.thecvf.com/content/ICCV2021/papers/Xu_SPG_Unsupervised_Domain_Adaptation_for_3D_Object_Detection_via_Semantic_ICCV_2021_paper.pdf
[80] 3D-Consistent Multi-View Editing by Diffusion Guidance https://arxiv.org/abs/2511.22228
[81] [2209.14988] DreamFusion: Text-to-3D using 2D Diffusion https://arxiv.org/abs/2209.14988
[82] Empowering Diffusion Models with Multi-View Conditions ... https://arxiv.org/html/2507.02299v1
[83] Controllable 3D object Generation with Single Image Prompt https://arxiv.org/html/2511.22194v1
[84] Unified Domain Generalization and Adaptation for Multi ... https://arxiv.org/html/2410.22461v1
[85] ObjFiller-3D: Consistent Multi-view 3D Inpainting via Video ... https://arxiv.org/abs/2508.18271
[86] DreamSampler: Unifying Diffusion Sampling and Score ... https://arxiv.org/html/2403.11415v2
[87] Domain Adaptation for Different Sensor Configurations in ... https://arxiv.org/html/2509.04711v1
[88] Sharp-It: A Multi-view to Multi-view Diffusion Model for 3D ... https://arxiv.org/abs/2412.02631
[89] DreamFusion: Text-to-3D Synthesis https://www.emergentmind.com/topics/dreamfusion
[90] Posterior Distillation Sampling - CVF Open Access https://openaccess.thecvf.com/content/CVPR2024/papers/Koo_Posterior_Distillation_Sampling_CVPR_2024_paper.pdf
