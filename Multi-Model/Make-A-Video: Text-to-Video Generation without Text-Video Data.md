
# Make-A-Video: Text-to-Video Generation without Text-Video Data

## 1. 핵심 주장 및 주요 기여

**Make-A-Video**는 Meta AI에서 발표한 혁신적인 텍스트-투-비디오(T2V) 생성 모델로, 기존 텍스트-투-이미지(T2I) 생성의 발전을 비디오 생성으로 확장하는 새로운 접근 방식을 제시합니다.[1]

논문의 핵심 주장은 명확합니다: **텍스트-비디오 쌍 데이터 없이도 고품질 비디오 생성이 가능하다**는 것입니다. 이를 달성하기 위해 세 가지 전략을 조합합니다.[1]

**주요 기여:**

1. **기존 T2I 모델의 효과적인 확장**: 확산 기반(diffusion-based) T2I 모델을 시공간 분해 확산 모델(spatiotemporally factorized diffusion model)을 통해 T2V로 확장하는 효과적인 방법 제시[1]

2. **텍스트-비디오 데이터 의존성 제거**: 결합된 텍스트-이미지 선행 정보(text-image priors)를 활용하여 대규모 텍스트-비디오 쌍 데이터의 필요성 우회[1]

3. **초해상도 전략의 혁신**: 사용자 제공 텍스트 입력을 통해 **처음으로** 고화질(고해상도, 높은 프레임률) 비디오를 생성하는 시공간 초해상도 전략 제시[1]

4. **포괄적 평가 제시**: 정성적, 정량적 측면 모두에서 최첨단 결과 달성 및 기존 문헌보다 철저한 평가 수행[1]

***

## 2. 문제 정의, 제안 방법 및 모델 구조

### 2.1 해결하고자 하는 문제

**핵심 문제**: 텍스트 기반 이미지 생성은 인터넷 수십억 개의 alt-text-이미지 쌍으로부터 획기적인 진전을 이루었지만, 비디오 생성은 다음과 같은 제약에 직면합니다.[1]

- **데이터 부족**: 고품질 텍스트-비디오 쌍 데이터를 대규모로 수집하기 어려움
- **계산 복잡성**: 동영상은 이미지보다 높은 차원의 데이터로, 모델 학습이 매우 무겁다
- **처음부터 학습의 비효율성**: 이미 존재하는 강력한 T2I 모델을 활용하지 않음

### 2.2 핵심 직관 (Intuition)

Make-A-Video는 다음과 같은 간결하면서도 강력한 직관에 기반합니다:[1]

> **"텍스트와 이미지의 대응 관계는 텍스트-이미지 쌍 데이터에서, 세계의 움직임은 레이블 없는 비디오 데이터에서 학습한다"**

이러한 접근은 비지도 학습(unsupervised learning)의 강력함을 활용합니다. 자연어 처리(NLP)에서와 마찬가지로, 사전학습(pretraining)된 모델은 지도 학습만으로 훈련된 모델보다 현저히 높은 성능을 보입니다.[1]

### 2.3 전체 시스템 구조

**Make-A-Video의 최종 T2V 추론 과정은 다음과 같이 표현됩니다:**[1]

$$y_t = \text{SR}_h(\text{SR}_l^t(F(D_t(P(x, C_x(x))))))$$

여기서:
- $y_t$: 생성된 비디오
- $\text{SR}_h$, $\text{SR}_l$: 공간 및 시공간 초해상도 네트워크
- $F$: 프레임 보간 네트워크
- $D_t$: 시공간 디코더
- $P$: 선행 네트워크 (prior network)
- $x$: BPE 인코딩된 텍스트
- $C_x$: CLIP 텍스트 인코더

### 2.4 모델의 세 가지 주요 구성 요소

#### (1) 기존 T2I 모델 기반

DALL-E 2와 유사한 아키텍처 공유:[1]
- **Prior 네트워크** $P$: 텍스트 임베딩에서 이미지 임베딩 생성
- **Decoder 네트워크** $D$: 이미지 임베딩에서 저해상도(64×64) RGB 이미지 생성
- **초해상도 네트워크** $\text{SR}_l$, $\text{SR}_h$: 256×256 및 768×768 해상도로 확대

#### (2) 시공간 계층(Spatiotemporal Layers)

2D 조건부 네트워크를 시간 차원으로 확장하기 위해 두 가지 핵심 구성 요소 수정:[1]

**Pseudo-3D 합성곱 계층:**

$$\text{ConvP3D}(h) = \text{Conv1D}(\text{Conv2D}(h)^T)^T$$

여기서:
- 입력 텐서 $h \in \mathbb{R}^{B \times C \times F \times H \times W}$ (B: 배치, C: 채널, F: 프레임, H: 높이, W: 너비)
- Conv2D는 사전학습된 T2I 모델에서 초기화
- Conv1D는 항등함수(identity function)로 초기화되어 평활한 초기화 보장[1]

이 구조의 장점:
- 계산 비용 증가 최소화 (3D 합성곱의 무거운 계산량 회피)
- 사전학습된 공간 지식 보존
- 새로운 시간 정보 학습 가능

**Pseudo-3D 주의(Attention) 계층:**

$$\text{ATTN}_{P3D}(h) = \text{unflatten}(\text{ATTN}_{1D}(\text{ATTN}_{2D}(\text{flatten}(h)^T))^T)$$

여기서:
- flatten: 공간 차원을 행렬로 평탄화
- unflatten: 역 연산
- $\text{ATTN}_{2D}$: 사전학습된 공간 주의 계층
- $\text{ATTN}_{1D}$: 항등함수로 초기화된 새로운 시간 주의 계층[1]

**프레임률 조건화 (Frame Rate Conditioning):**

T2I 조건 외에도 fps(frames-per-second)를 추가 조건으로 포함하여:[1]
- 제한된 비디오 데이터에 대한 추가 데이터 증강
- 추론 시 생성되는 비디오의 프레임률 제어 가능

#### (3) 프레임 보간 네트워크 (Frame Interpolation Network)

생성된 16프레임을 더 높은 프레임률로 확대:[1]
- 마스크된 프레임 보간(masked frame interpolation)을 통해 프레임 간 생략된 부분 채우기
- 비디오 업샘플링과 외삽(extrapolation)을 위한 통합 아키텍처
- 시공간 디코더 $D_t$를 미세조정(fine-tuning)하여 구현

마스크된 입력: U-Net에 3채널 RGB 마스크 비디오 + 1채널 이진 마스크 채널 추가[1]

***

## 3. 학습 절차

다양한 구성요소는 **독립적으로 훈련**됩니다:[1]

1. **Prior 네트워크** $P$: 텍스트-이미지 쌍 데이터에서 훈련 (비디오 데이터에서 미세조정 **없음**)

2. **Decoder, Prior, 초해상도 구성요소**: 먼저 이미지 단독으로 훈련
   - Decoder: CLIP 이미지 임베딩 입력
   - 초해상도: 다운샘플된 이미지 입력

3. **시공간 계층**: 새 시간층 추가 및 레이블 없는 비디오 데이터에서 미세조정
   - 원본 비디오에서 16프레임 샘플링 (프레임률 1-30fps 무작위)
   - Beta 함수로 프레임률 샘플링: 높은 fps부터 시작하여 낮은 fps로 전환

4. **마스크된 프레임 보간 구성요소**: 시공간 디코더에서 미세조정

***

## 4. 모델 성능 평가 및 향상

### 4.1 정량적 평가 결과

**MSR-VTT 벤치마크 (영점-샷 평가):**[1]

| 방법 | FID | CLIPSIM | 영점-샷 |
|------|------|----------|---------|
| GODIVA | 24.02 | N/A | 아니오 |
| NUWA | 47.68 | 0.2439 | 아니오 |
| CogVideo (중국어) | 23.59 | 0.2631 | 예 |
| CogVideo (영어) | 24.78 | 0.2614 | 예 |
| **Make-A-Video** | **13.17** | **0.3049** | **예** |

Make-A-Video는 **FID에서 43% 개선**을 달성했습니다.[1]

**UCF-101 벤치마크 (미세조정):**[1]

| 방법 | IS | FVD |
|------|-------|--------|
| MoCoGAN-HD | 33.95 | 70.0 |
| CogVideo | 50.46 | 62.6 |
| TATS-base | 79.28 | 27.8 |
| **Make-A-Video** | **82.55** | **81.25** |

### 4.2 인간 평가

**DrawBench 및 평가 세트:**[1]

| 비교 | 품질(%) | 충실도(%) |
|------|---------|-----------|
| Make-A-Video vs. VDM | 84.38 | 78.13 |
| Make-A-Video vs. CogVideo (영어) | 74.48 | 68.75 |
| Make-A-Video vs. CogVideo (중국어) | 76.88 | 73.37 |

인간 평가자들의 대다수가 Make-A-Video가 더 높은 품질과 텍스트 충실도를 가진다고 평가했습니다.[1]

***

## 5. 모델의 일반화 성능 향상

### 5.1 핵심 일반화 메커니즘

**Transfer Learning 활용:**
Make-A-Video의 가장 혁신적인 측면은 사전학습된 T2I 모델의 강력한 표현을 직접 활용한다는 것입니다. 이는 다음을 보장합니다:[1]

- **시각-텍스트 대응 관계의 즉각적 이전**: T2I 모델이 학습한 텍스트-이미지 매핑이 비디오의 각 프레임에 자동으로 적용
- **의미론적 일관성**: 비디오 전체 프레임이 동일한 텍스트 임베딩 공간에서 생성되어 의미적 일관성 보장

**대규모 비텍스트 비디오 데이터 활용:**

웹에서 쉽게 수집 가능한 레이블 없는 비디오 데이터만 사용:[1]
- WebVid-10M: 1000만 개 비디오
- HD-VILA-100M (10M 부분집합): 추가 1000만 개 비디오 클립

이를 통해 시공간 역학(dynamics) 학습 가능.[1]

### 5.2 영점-샷(Zero-Shot) 일반화

**MSR-VTT에서의 영점-샷 성능:**

Make-A-Video는 훈련 데이터 없이도 MSR-VTT에서 다른 모든 모델을 능가합니다. 이는:[1]

1. **T2I 모델의 광범위한 개념 커버리지**: 23억 개의 텍스트-이미지 쌍에서 학습한 다양한 시각 개념
2. **비지도 학습을 통한 동작 이해**: 레이블 없는 비디오에서 학습한 보편적 동작 패턴

**UCF-101에서의 영점-샷 대 미세조정:**

영점-샷 성능: IS=33.00, FVD=367.23
미세조정 성능: IS=82.55, FVD=81.25

비록 미세조정 후 성능이 대폭 향상되지만, **영점-샷 성능도 이미 경쟁력 있는 수준**을 보여줍니다.[1]

### 5.3 다양성과 창의성 상속

T2I 모델의 미학적 다양성 계승:[1]
- 사진적 사실성에서 환상적 표현까지 광범위한 시각 스타일
- 고해상도 이미지 생성 모델의 세부 사항과 질감

***

## 6. 모델의 한계

논문에서 명시적으로 언급된 주요 한계:[1]

### 6.1 텍스트-비디오 현상에만 내재된 개념 학습 불가

**핵심 문제**: 텍스트-이미지 쌍에서는 추론 불가능한 동작 관련 정보를 생성할 수 없습니다.

예시:
- "사람이 손을 왼쪽에서 오른쪽으로 흔든다" vs "사람이 손을 오른쪽에서 왼쪽으로 흔든다"
- 이 두 명령은 정적 이미지에서 구별 불가능하나 비디오에서는 중요[1]

### 6.2 비디오 길이 제한

현재 시스템은 최대 16프레임(원본 생성)만 처리 가능하며, 프레임 보간을 통해 약 76프레임까지만 확대 가능합니다.[1]

더 긴 비디오, 다중 씬, 복잡한 내러티브 생성은 향후 과제입니다.[1]

### 6.3 사회적 편향(Social Bias)

대규모 웹 데이터 학습으로 인한 문제:[1]
- 모델이 학습 데이터의 사회적 편향 및 해로운 편견 학습 가능성
- NSFW 콘텐츠 필터링 및 독성 단어 제거로 부분적 완화

### 6.4 시공간 초해상도의 한계

**고해상도 초해상도($\text{SR}_h$)의 문제:**
- 메모리 및 계산 제약으로 시간 차원에 확장 어려움
- 현재 공간 차원에서만 작동하며, 프레임당 동일한 노이즈 초기화로 일관성 유지[1]

***

## 7. 논문의 영향 및 향후 연구 고려사항

### 7.1 학계에 미친 영향

Make-A-Video는 텍스트-비디오 생성의 **패러다임 전환**을 이끌었습니다.[2][3]

**주요 영향:**

1. **Transfer Learning의 중요성 증명**: T2I 모델 기반 접근이 표준화되었습니다. 이후 대부분의 T2V 모델들(CogVideo, VDM, Imagen Video 등)은 유사한 전이학습 전략 채택[3][2]

2. **텍스트-비디오 쌍 데이터 없는 학습 가능성**: 대규모 텍스트-비디오 데이터 수집의 필수성이 덜 중요해졌습니다.[4]

3. **시공간 인수분해의 표준화**: Pseudo-3D 설계는 이후 많은 모델에서 영감 제공[3]

### 7.2 최신 연구 동향 (2023-2025)

#### (1) 더 긴 비디오 생성
- **StreamingT2V** (2024): 자동회귀(autoregressive) 접근으로 텍스트-투-장비디오 생성[5]
- **Step-Video-T2V** (2025): 30B 파라미터로 204프레임 비디오 생성 가능[6]

#### (2) 구성적 이해 개선
- **VideoComposer** (2023): 다중 조건(depth, sketch, action 등)으로 비디오 합성[7]
- **VideoComp** (2025): 세분화된 구성적 비디오-텍스트 정렬 개선[8]
- **Compositional Video Synthesis** (2024): 객체 중심 표현으로 편집 가능한 비디오 생성[9]

#### (3) 언어 이해 강화
- **DirecT2V** (2024): 대규모 언어 모델(LLM)을 프레임 수준 감독으로 활용[10]
- **NewMove** (2024): 새로운 동작 패턴을 적응적으로 학습[11]
- **FancyVideo** (2024): 프레임별 텍스트 안내를 통한 시간적 논리 이해 개선[12]

#### (4) 데이터 효율성 개선
- **VideoCrafter2** (2024): 데이터 제한 극복을 위한 전략[13]
- **A Recipe for Scaling** (2023): 텍스트 없는 비디오로 T2V 확장[4]
- **CompPretrain** (2025): 짧은 캡션 비디오 데이터셋 활용 사전학습[8]

#### (5) 일반화 능력 향상
- **MOVAI** (2025): 계층적 장면 이해와 시공간 주의 메커니즘 통합[14]
- **VideoPoet** (2024): 대언어모델 기반 영점-샷 비디오 생성[15]
- **CogVideoX** (2024): 10초 연속 비디오(768×1360) 생성[16]

### 7.3 향후 연구 시 고려할 점

#### (1) 모델 일반화 개선
- **도메인 특이성 극복**: 특정 스타일(만화, 애니메이션 등)이나 분야(의료, 산업)에 더 나은 일반화
- **장시간 일관성**: 200프레임 이상 장비디오에서 주체 일관성 및 시공간 일관성 유지
- **다중 객체 처리**: 복잡한 장면에서 여러 객체의 상호작용 정확히 표현

#### (2) 텍스트-비디오 매핑의 한계 극복
- **암묵적 동작 이해**: 정적 이미지로는 알 수 없는 방향성 동작(좌측 vs 우측 이동) 등 명시적으로 학습하는 방법
- **인과 관계**: "공이 떨어지면 튀어나온다" 같은 물리 법칙 준수
- **다중 도메인 공동 훈련**: 이미지, 비디오, 3D 데이터의 통합 학습

#### (3) 효율성과 확장성
- **계산 효율**: 더 큰 모델로의 확장 시 메모리 및 계산 비용 최적화
- **적응형 계산**: 복잡도에 따른 동적 계산량 조정
- **실시간 생성**: 사용자 대화형 애플리케이션을 위한 빠른 추론

#### (4) 데이터 및 편향 문제
- **사회적 편향 완화**: 더 균형잡힌 데이터셋 개발 및 공정성 평가 지표
- **다언어 지원**: 영어 중심 학습에서 벗어나 다양한 언어 지원
- **투명성 증대**: 학습 데이터 공개 및 재현 가능성 개선

#### (5) 평가 방법 개선
- **인간-중심 평가**: 자동 지표(FID, CLIPSIM)보다 더 정교한 인간 평가 프레임워크
- **일관성 평가**: 시간적, 공간적 일관성을 측정하는 표준화된 지표
- **사용성 평가**: 실제 사용자가 생성된 비디오를 활용하는 시나리오에서의 만족도

#### (6) 새로운 응용 분야
- **대화형 비디오 생성**: 사용자 피드백 기반 실시간 수정
- **개인화된 비디오**: 사용자 스타일 학습 및 맞춤형 생성
- **비디오 편집 및 조작**: 생성 모델을 통한 기존 비디오 편집
- **다중 모달 입력**: 텍스트뿐 아니라 스케치, 이미지, 오디오 등 통합

***

## 요약

**Make-A-Video**는 텍스트-투-비디오 생성의 혁신적인 전환점입니다. 사전학습된 T2I 모델의 강력한 표현과 비지도 학습의 유연성을 결합함으로써, 텍스트-비디오 쌍 데이터 없이도 고품질 비디오를 생성합니다. Pseudo-3D 시공간 계층 설계는 계산 효율성을 유지하면서도 시간 차원을 효과적으로 모델링합니다.

최신 연구 동향은 **네 가지 주요 방향**으로 진화하고 있습니다: (1) 더 긴 비디오 생성, (2) 구성적 이해 강화, (3) 언어 모델 통합을 통한 지능형 안내, (4) 효율적인 데이터 활용. 향후 연구는 일반화 능력 개선, 암묵적 동작 이해, 장비디오 일관성 유지, 그리고 사회적 편향 완화에 중점을 두어야 할 것입니다.[2][16][5][14][15][11][10][12][9][13][7][3][4][8][1]

***

## 참고 자료

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/aa4a56da-f869-431c-b72d-4ace7830af24/2209.14792v1.pdf)
[2](https://ieeexplore.ieee.org/document/10521640/)
[3](https://dl.acm.org/doi/10.1145/3587423.3595503)
[4](https://arxiv.org/html/2312.15770)
[5](http://arxiv.org/pdf/2403.14773.pdf)
[6](https://arxiv.org/html/2502.10248)
[7](https://proceedings.neurips.cc/paper_files/paper/2023/file/180f6184a3458fa19c28c5483bc61877-Paper-Conference.pdf)
[8](https://openaccess.thecvf.com/content/CVPR2025/papers/Kim_VideoComp_Advancing_Fine-Grained_Compositional_and_Temporal_Alignment_in_Video-Text_Models_CVPR_2025_paper.pdf)
[9](https://openreview.net/pdf/26f6122fe171cbcfc979863334b80abfd2d2e122.pdf)
[10](http://arxiv.org/pdf/2305.14330.pdf)
[11](https://arxiv.org/html/2312.04966v1)
[12](https://arxiv.org/html/2408.08189v1)
[13](https://openaccess.thecvf.com/content/CVPR2024/papers/Chen_VideoCrafter2_Overcoming_Data_Limitations_for_High-Quality_Video_Diffusion_Models_CVPR_2024_paper.pdf)
[14](https://arxiv.org/html/2511.00107v1)
[15](https://icml.cc/virtual/2024/poster/34296)
[16](http://arxiv.org/pdf/2408.06072.pdf)
[17](https://dl.acm.org/doi/10.1145/3503161.3548380)
[18](https://www.ssrn.com/abstract=4294796)
[19](https://arxiv.org/abs/2403.03206)
[20](http://www.ghspjournal.org/lookup/doi/10.9745/GHSP-D-23-00199)
[21](https://www.semanticscholar.org/paper/31973a972d4e4b99f17dc1a89c51398d3276a1d6)
[22](https://www.semanticscholar.org/paper/deff0fd3af0bf9b4182f476f26c0cdc8a3e56718)
[23](http://arxiv.org/pdf/2408.12590.pdf)
[24](https://arxiv.org/pdf/2303.13439.pdf)
[25](https://arxiv.org/html/2401.12945v1?s=09)
[26](http://arxiv.org/pdf/2406.04277.pdf)
[27](https://arxiv.org/html/2311.18829v2)
[28](https://arxiv.org/html/2502.03621v1)
[29](https://arxiv.org/abs/2209.14792)
[30](https://openreview.net/pdf?id=qpDqO7qa3R)
[31](https://ostin.tistory.com/130)
[32](https://www.ijcai.org/proceedings/2025/0238.pdf)
[33](https://www.ijcai.org/proceedings/2025/0239.pdf)
[34](https://www.youtube.com/watch?v=KRTEOkYftUY)
[35](https://openaccess.thecvf.com/content/ICCV2021/papers/Aich_Spatio-Temporal_Representation_Factorization_for_Video-Based_Person_Re-Identification_ICCV_2021_paper.pdf)
[36](https://proceedings.neurips.cc/paper_files/paper/2022/hash/39235c56aef13fb05a6adc95eb9d8d66-Abstract-Conference.html)
[37](https://arxiv.org/abs/2412.14989)
[38](https://arxiv.org/abs/2505.06814)
[39](https://arxiv.org/abs/2508.04129)
[40](https://dl.acm.org/doi/10.1145/3746027.3762058)
[41](https://dl.acm.org/doi/10.1145/3706599.3707213)
[42](https://www.semanticscholar.org/paper/6c708659768e470f63d06f791ff8420e7ff0feac)
[43](https://ejournal.papanda.org/index.php/jirpe/article/view/1410)
[44](https://dl.acm.org/doi/10.1145/3746027.3762103)
[45](http://pubs.rsna.org/doi/10.1148/radiol.242167)
[46](https://drpress.org/ojs/index.php/fcis/article/view/31302)
[47](https://arxiv.org/abs/2305.18264)
[48](https://towardsdatascience.com/the-evolution-of-text-to-video-models-1577878043bd/)
[49](https://openreview.net/pdf?id=sgDFqNTdaN)
[50](https://garagefarm.net/blog/the-complete-guide-to-ai-video-generators)
[51](https://arxiv.org/abs/2412.00773)
[52](https://modal.com/blog/text-to-video-ai-article)
[53](https://arxiv.org/html/2412.18688v2)
