
# ControlNet : Adding Conditional Control to Text-to-Image Diffusion Models
https://github.com/lllyasviel/ControlNet

## 1. 핵심 주장과 주요 기여

ControlNet은 **대규모 사전학습 텍스트-이미지 확산 모델(Stable Diffusion)에 공간적 조건 제어를 추가하는 신경망 구조**를 제시합니다. 이 논문의 핵심 주장은 다음과 같습니다:[1]

**주요 기여:**

1. **구조 보존을 통한 안정적 학습**: 원본 확산 모델의 파라미터를 동결(locked)하고, 학습 가능한 복사본(trainable copy)을 만든 후 제로 컨볼루션(zero convolution)으로 연결하여 사전학습된 지식을 보존하면서 새로운 조건을 학습합니다.[1]

2. **데이터 효율성**: 50k부터 1m 이상의 데이터셋에서 강건한 학습이 가능하며, 단일 NVIDIA RTX 3090Ti GPU에서 산업체 모델 수준의 성능을 달성합니다.[1]

3. **다양한 조건 지원**: Canny 엣지, 깊이 맵, 세분화 마스크, 인간 포즈, 스케치 등 8가지 이상의 조건을 지원하며, 단일 또는 다중 조건의 조합이 가능합니다.[1]

***

## 2. 문제 정의, 제안 방법, 모델 구조 및 성능

### 2.1 해결하고자 하는 문제

기존 텍스트-이미지 생성 모델은 **공간적 구성에 대한 세밀한 제어가 부족**합니다. 특히, 복잡한 레이아웃, 포즈, 형태 등을 정확히 표현하기 어렵고, 소수의 조건-관련 데이터셋(약 100k 정도)으로 대규모 모델(LAION-5B로 학습된 Stable Diffusion)을 미세조정하면 **재난적 망각(catastrophic forgetting)**과 **과적합** 문제가 발생합니다.[1]

### 2.2 제안하는 방법 (수식 포함)

#### 기본 구조

신경망 블록 \(F\)가 입력 특성맵 \(x\)를 출력 \(y\)로 변환한다고 하면:

$$y = F(x; \Theta)$$

ControlNet은 원본 블록을 동결하고, 학습 가능한 복사본을 만들어 제로 컨볼루션으로 연결합니다:

$$y_c = F(x; \Theta) + Z(F(x; \Theta_z); \Theta_{z1}) + Z(c; \Theta_{z2})$$

여기서 \(y_c\)는 ControlNet 블록의 출력, \(c\)는 조건 벡터, \(Z(\cdot; \cdot)\)는 제로 컨볼루션입니다.[1]

#### 제로 컨볼루션의 특성

제로 컨볼루션은 **가중치와 바이어스가 모두 0으로 초기화된 1×1 컨볼루션**입니다. 초기 학습 단계에서:

$$Z(\cdot) = 0$$

따라서 \(y_c = F(x; \Theta)\)이 되어 학습 초기에 해로운 노이즈가 추가되지 않습니다. 학습이 진행되면서 제로 컨볼루션의 파라미터가 점진적으로 성장합니다.[1]

#### 조건 인코딩

입력 조건 이미지 \(c\)는 4개의 컨볼루션 층(커널 4×4, 스트라이드 2×2, ReLU 활성화)으로 구성된 소형 네트워크 \(E\)를 통해 특성공간 조건 벡터로 변환됩니다:

$$c_f = E(c; \Theta_E)$$

입력 이미지(512×512)를 Stable Diffusion의 잠재공간(64×64)에 맞추기 위해 채널 수를 16 → 32 → 64 → 128로 확장합니다.[1]

#### 학습 목적 함수

확산 모델의 학습 목적은 다음과 같습니다:

$$L = E_{z_0, t, c_t, c_f \sim \mathcal{N}(0,1)} \left\| \epsilon - \hat{\epsilon}(z_t, t, c_t, c_f) \right\|^2$$

여기서 \(\epsilon\)는 추가된 노이즈, \(\hat{\epsilon}\)는 모델의 예측값, \(c_t\)는 텍스트 조건, \(c_f\)는 공간 조건입니다. 훈련 중 50% 확률로 텍스트 프롬프트를 공 문자열로 대체하여 모델이 조건 이미지의 의미를 직접 인식하도록 유도합니다.[1]

### 2.3 모델 구조

ControlNet은 Stable Diffusion의 U-Net 구조에 적용됩니다:

- **인코더**: 64×64, 32×32, 16×16, 8×8 해상도의 12개 블록 (각각 3회 반복)
- **중간 블록**: 8×8 해상도의 1개 블록
- **디코더**: 스킵 연결을 통해 입력받음

각 해상도 수준에서 ControlNet의 학습 가능한 복사본을 만들고 제로 컨볼루션으로 원본 모델과 연결합니다. 이를 통해:[1]

- **계산 효율성**: 동결된 파라미터는 그래디언트 계산이 필요 없어 기존 학습 대비 약 **34% 시간 증가, 23% GPU 메모리 증가**에 불과합니다.[1]

### 2.4 성능 향상

#### 정성적 평가

ControlNet은 다양한 조건 입력(Canny 엣지, 깊이 맵, 포즈 스켈레톤 등)에서 높은 품질의 이미지를 생성합니다. 프롬프트가 없는 상황에서도 조건 이미지의 의미를 올바르게 인식하여 생성합니다.[1]

#### 정량적 평가

**사용자 연구** (Table 1): 손그린 스케치 조건으로 20개 샘플에 대해 12명의 사용자가 평가했을 때:

| 방법 | 이미지 품질 (AHR) | 조건 충실도 (AHR) |
|------|----------------|-----------------|
| PITI | 1.10 ± 0.05 | 1.02 ± 0.01 |
| Sketch-Guided (1.6) | 2.31 ± 0.62 | 3.21 ± 0.57 |
| Sketch-Guided (3.2) | 3.28 ± 0.72 | 2.52 ± 0.44 |
| ControlNet-lite | 3.93 ± 0.59 | 4.09 ± 0.46 |
| **ControlNet** | **4.22 ± 0.43** | **4.28 ± 0.45** |

**산업 모델 비교**: Stable Diffusion V2 Depth-to-Image (SDv2-D2I)는 12M 이상의 훈련 이미지와 수천 시간의 GPU로 학습되었으나, ControlNet은 **단 200k 이미지와 5일 단일 GPU로 거의 구별 불가능한 결과(정확도 0.52 ± 0.17)**를 달성합니다.[1]

**분할 조건 평가** (Table 3 - ADE20K):

| 방법 | FID 점수 | CLIP 텍스트-이미지 점수 | CLIP 미적 점수 |
|------|---------|--------------------|-----------| 
| Stable Diffusion | 6.09 | 0.26 | 6.32 |
| LDM | 26.28 | 0.17 | 5.14 |
| PITI | 25.35 | 0.18 | 5.15 |
| ControlNet-lite | 19.74 | 0.20 | 5.77 |
| **ControlNet** | **15.27** | **0.26** | **6.31** |

### 2.5 한계

1. **해상도 제한**: 조건 이미지와 생성된 이미지의 해상도가 고정되어 있으며, 세밀한 텍스트 렌더링이 어렵습니다.[2]

2. **조건 타입별 독립 학습**: 각 조건 타입마다 독립적으로 ControlNet을 학습해야 하므로 **확장성 문제**가 있습니다.[3]

3. **메모리 오버헤드**: 추론 시 약 700M 추가 파라미터가 필요하여 메모리 소비량이 증가합니다.[4]

4. **학습 데이터 구성에 따른 민감성**: 조건-이미지 쌍의 데이터 분포에 크게 의존하며, 분포 외(out-of-distribution) 조건에 대한 일반화가 제한적입니다.[1]

***

## 3. 일반화 성능 향상 가능성

### 3.1 현재 수준의 일반화 성능

**데이터셋 크기에 따른 강건성** (Figure 10): ControlNet은 1k 이미지로 학습해도 인식 가능한 라이온을 생성하고, 데이터가 증가함에 따라 점진적으로 성능이 개선됩니다. 이는 제로 컨볼루션 구조가 사전학습된 백본을 강력한 기초로 활용하기 때문입니다.[1]

**스타일 전이 가능성**: 훈련된 ControlNet은 Comic Diffusion, Protogen 등 다른 커뮤니티 모델로 **재훈련 없이 직접 전이**됩니다. 이는 U-Net 기반 아키텍처의 호환성으로 인한 것입니다.[1]

### 3.2 최신 연구의 일반화 개선 방향

#### (1) 메타 러닝 기반 접근 (Meta ControlNet)

**Meta ControlNet**은 메타 학습 기법을 도입하여 **제로샷 제어**와 **빠른 적응**을 달성합니다:[5]
- 학습 단계를 5000에서 상당히 감소시킵니다.
- 엣지 기반 작업뿐 아니라 다양한 비엣지 기반 작업으로 확대합니다.

#### (2) 통합 프레임워크 (Uni-ControlNet)

**Uni-ControlNet**은 **단일 모델에서 지역 제어(local control)와 전역 제어(global control)를 동시에 수행**합니다:[6]
- 엣지 맵, 깊이 맵, 분할 마스크(지역 제어)
- CLIP 이미지 임베딩(전역 제어)
- 다중 조건의 유연한 조합 가능

#### (3) 데이터 효율성 개선 (CtrLoRA)

**CtrLoRA**는 **LoRA(Low-Rank Adaptation) 기반의 경량 프레임워크**로:[3]
- 각 조건마다 독립적인 훈련 대신 **파라미터 효율적 학습** 구현
- 기존 ControlNet 대비 파라미터 감소로 확장성 개선

#### (4) 도메인 간 일반화 (Generalization by Adaptation with DIDEX)

**DIDEX** 연구는 ControlNet을 활용하여 **도메인 적응(domain adaptation) 성능을 향상**시킵니다:[7]
- 합성 도메인과 실제 도메인 간 갭 해소
- 확산 기반의 세밀한 제어로 도메인 일반화 개선

#### (5) 효율적인 제어 (FlexControl, RelaCtrl)

**FlexControl**과 **RelaCtrl**은 **Diffusion Transformer(DiT) 아키텍처**에서의 효율적 제어를 목표로 합니다:[8][9]

**RelaCtrl의 주요 성과:**
- DiT에서 제어 계층의 관련성(relevance)을 평가하는 "ControlNet 관련성 점수" 도입
- 파라미터와 계산량을 기존 대비 **85% 감소**
- SOTA 성능 유지

#### (6) 의료 이미지 합성 (Adaptively Distilled ControlNet)

**Adaptively Distilled ControlNet**은 **의료 이미지 생성에서의 일반화 개선**:[10]
- 이중 모델 증류(dual-model distillation)를 통한 마스크-정렬 성능 향상
- TransUNet에서 mDice/mIoU **2.4%/4.2% 개선**(KiTS19 데이터셋)

#### (7) 멀티 해상도 유연성 (EasyControl)

**EasyControl**은 DiT 기반 프레임워크로:[11]
- 조건 인젝션 LoRA 모듈로 **플러그 앤 플레이 호환성** 달성
- **위치 인식 학습 패러다임**으로 임의의 종횡비와 해상도 지원
- **제로샷 다중 조건 일반화**: 단일 조건 데이터로만 훈련하면서 다중 조건에 일반화

#### (8) 소수샷 적응 (Universal Few-Shot Spatial Control)

**Universal Few-Shot Spatial Control**은:[12]
- 새로운 공간 조건에 대한 소수샷(few-shot) 적응 능력
- 메타 학습을 통해 빠른 수렴
- 다양한 확산 모델 백본(U-Net, DiT) 지원

### 3.3 일반화 성능 향상의 핵심 요소

| 개선 방향 | 주요 기법 | 효과 |
|---------|--------|------|
| **학습 효율** | 제로 컨볼루션 + 파라미터 동결 | 사전학습된 지식 보존으로 소규모 데이터 학습 가능 |
| **멀티태스크** | 통합 프레임워크 (Uni-ControlNet) | 단일 모델에서 다양한 조건 동시 처리 |
| **도메인 간 전이** | 메타 러닝 (Meta ControlNet) | 신규 조건에 대한 빠른 적응 |
| **계산 효율** | 관련성 기반 설계 (RelaCtrl) | DiT 구조에서 파라미터 85% 감소 |
| **다양한 아키텍처** | 유연한 주입 모듈 (EasyControl) | U-Net과 DiT 모두 지원 |

***

## 4. 앞으로의 연구에 미치는 영향 및 고려사항

### 4.1 ControlNet이 미친 영향

**원본 논문 발표 후 학술/산업계 반응:**

1. **제어 가능한 생성의 기준점**: ControlNet은 조건부 이미지 생성 분야의 **핵심 아키텍처 패러다임**으로 자리 잡았습니다.[13][14][8][5][6][3]

2. **산업 응용 확대**: 의료 이미지 합성, 농업(잡초 탐지 데이터 증강), 텍스트 렌더링(RepText), 각종 디자인 도구에 광범위하게 적용되고 있습니다.[15][2]

3. **다양한 후속 연구 촉발**:
   - 원본의 U-Net 기반에서 **Diffusion Transformer(DiT) 기반**으로의 확장
   - **경량화 및 효율화** 방향의 연구
   - **멀티태스크 통합** 및 **제로샷 전이** 기술

### 4.2 현재 직면 과제

1. **DiT 아키텍처와의 호환성**: SD3.0, FLUX 등 최신 고성능 모델이 DiT 기반으로 전환되면서 ControlNet의 직접 적용이 제한되고 있습니다. 이를 해결하기 위한 **EasyControl, RelaCtrl** 등 새로운 프레임워크가 등장했습니다.[16][11]

2. **조건 타입 확장의 비용**: 각 조건마다 독립 학습이 필요하므로 **스케일 문제**가 있습니다. **CtrLoRA** 같은 경량 기법으로 개선 중입니다.[3]

3. **분포 외 조건에 대한 강건성 부족**: 학습 데이터 분포를 벗어난 입력에 대해 예측 품질이 급격히 저하될 수 있습니다. **도메인 일반화** 기법의 강화가 필요합니다.[17]

4. **텍스트/세밀한 세부 표현 한계**: 특히 비라틴 문자나 작은 텍스트 렌더링에서 여전히 어려움이 있습니다.[2]

### 4.3 앞으로의 연구 방향

#### (1) **효율성과 확장성**
- 단일 모델에서 다양한 조건을 처리하는 **통합 프레임워크** 강화
- **토큰 기반** 또는 **LoRA 기반** 경량 제어 모듈 개발 지속

#### (2) **새 아키텍처 적응**
- DiT, Diffusion Transformers로 완전히 재설계된 **다음 세대 확산 모델**에 대한 ControlNet 변형 연구
- **VideoPoet, Lumiere** 등 비디오 생성 모델로의 확장

#### (3) **도메인 간 일반화**
- **메타 러닝**, **도메인 적응** 기법과의 통합
- 새로운 조건에 대한 **소수샷 또는 제로샷 적응** 성능 향상

#### (4) **의료, 과학 도메인 특화**
- **의료 이미지 합성**에서의 물리적 제약(예: 정확한 장기 위치) 준수
- **분자, 단백질 구조** 등 과학적 데이터 생성에의 응용

#### (5) **다중 모달리티**
- 텍스트 렌더링, 3D 형상 제어, **비디오의 시간적 일관성 유지**
- **멀티센서 조건** (RGB, 적외선, 3D 스캔 등) 통합 제어

#### (6) **표현력 향상**
- **GraphControl** 같이 기존 이미지 기반 조건을 넘어 **그래프, 의미적 구조** 기반 제어
- **자연어와 시각 조건의 통합** 처리

### 4.4 고려사항

1. **계산 비용**: 제어 추가로 인한 메모리 및 시간 오버헤드가 여전하므로, **더 효율적인 구조 설계**가 필수입니다.

2. **데이터 가용성**: 특정 도메인(예: 의료, 산업)에서는 **라벨된 조건-이미지 쌍 수집의 어려움**이 병목입니다.

3. **모델 간 호환성**: 다양한 기초 모델(base model)과의 호환성을 보장하는 것이 **모듈식 설계**의 핵심입니다.

4. **평가 메트릭**: 조건 충실도(condition fidelity)와 이미지 품질 간의 **트레이드오프 측정** 방식의 표준화가 필요합니다.

***

## 결론

ControlNet은 **확산 모델의 조건부 제어를 구조화하는 획기적 방법론**을 제시했습니다. 제로 컨볼루션을 통한 안정적 학습, 강력한 사전학습 백본 보존, 다양한 조건 지원 등이 핵심 강점입니다. 최근 연구들은 DiT 호환성, 효율성, 멀티태스크 통합, 도메인 일반화 등의 측면에서 ControlNet의 한계를 보완하고 있으며, 앞으로도 **효율적 확장성, 신 아키텍처 적응, 도메인 특화 응용**이 주요 연구 방향이 될 것으로 예상됩니다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/7de3e2f9-29d1-4425-b753-3a4442e18568/2302.05543v3.pdf)
[2](https://www.visionhong.com/posts/reptext)
[3](https://arxiv.org/html/2410.09400v2)
[4](https://huggingface.co/blog/controlnet)
[5](https://arxiv.org/html/2312.01255v2)
[6](https://arxiv.org/abs/2305.16322)
[7](https://elib.dlr.de/202784/1/WACV_DIDEX.pdf)
[8](https://arxiv.org/html/2502.10451v1)
[9](https://arxiv.org/html/2502.14377)
[10](https://arxiv.org/html/2507.23652v1)
[11](https://easycontrolproj.github.io)
[12](https://arxiv.org/html/2509.07530v1)
[13](http://arxiv.org/pdf/2310.07365.pdf)
[14](http://arxiv.org/pdf/2404.09967.pdf)
[15](https://www.sciencedirect.com/science/article/abs/pii/S0168169925002297)
[16](https://huggingface.co/papers/2503.07027)
[17](https://ettrends.etri.re.kr/ettrends/207/0905207005/)
[18](https://arxiv.org/abs/2302.05543)
[19](http://arxiv.org/pdf/2404.07987.pdf)
[20](http://arxiv.org/pdf/2502.14779.pdf)
[21](https://arxiv.org/html/2410.10905v2)
[22](https://learnopencv.com/controlnet/)
[23](https://blog.segmind.com/how-to-optimize-controlnetsoft-performance-a-comprehensive-guide/)
[24](https://openreview.net/forum?id=b3bJR1quJ3)
[25](https://huggingface.co/docs/diffusers/api/pipelines/controlnet_sdxl)
[26](https://www.machinelearningmastery.com/control-net-with-stable-diffusion/)
[27](https://kimjy99.github.io/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0/controlnet/)
[28](https://arxiv.org/html/2408.06070v3)
[29](https://arxiv.org/pdf/2403.02714.pdf)
[30](https://blog.gopenai.com/360ai-launches-relactrl-a-new-version-of-controlnet-under-the-dit-architecture-e4e9f85cd623)
[31](https://dmqa.korea.ac.kr/activity/seminar/445)
[32](https://42morrow.tistory.com/entry/Diffusion-Self-DistillationDSD-%ED%99%95%EC%82%B0%EB%AA%A8%EB%8D%B8-%ED%99%9C%EC%9A%A9%ED%95%9C-%EC%A0%9C%EB%A1%9C%EC%83%B7-%EB%A7%9E%EC%B6%A4%ED%98%95-%EC%9D%B4%EB%AF%B8%EC%A7%80-%EC%83%9D%EC%84%B1)
[33](https://blog.naver.com/lee_jyoon/222721772771)


# Reference
- [23′ ICCV] ControlNet : Adding Conditional Control to Text-to-Image Diffusion Models:
https://ffighting.net/deep-learning-paper-review/diffusion-model/controlnet/
