# Scaling up GANs for Text-to-Image Synthesis: 종합 분석 보고서

### 1. 핵심 주장 및 주요 기여

**GigaGAN**의 핵심 주장은 단순하면서도 혁신적입니다: **GANs도 충분히 확장 가능하며, 대규모 텍스트-이미지 합성 작업에 여전히 실행 가능한 옵션이다**는 것입니다. 논문 발표 당시 diffusion 모델(DALLE 2, Imagen)과 자동회귀 모델(Parti)이 텍스트-이미지 합성의 표준으로 부상하였으나, 논문 저자들은 **기존 StyleGAN2 구조를 단순히 확장하는 것은 불안정성 문제를 야기한다**는 문제점을 식별했습니다.[1][2]

주요 기여는 다음과 같습니다:

1. **10억 파라미터 규모의 GAN 모델 개발**: StyleGAN2보다 36배, StyleGAN-XL보다 6배 더 큼[2][1]
2. **초고속 추론 속도**: 512px 이미지를 0.13초에 생성 (diffusion 모델보다 수십 배 빠름)[1][2]
3. **디엔탱글된 잠재 공간 유지**: 스타일 믹싱, 프롬프트 보간, 벡터 산술 등 고급 편집 기능 제공[1]
4. **초고해상도 생성**: 4K 해상도(16 메가픽셀)를 3.66초에 생성[1]

### 2. 해결하고자 하는 문제 및 기술적 배경

#### 2.1 근본적 문제

**기존 GAN의 확장성 한계**:

StyleGAN 아키텍처가 대규모 인터넷 이미지 데이터셋(LAION2B-en)에 단순히 적용될 때의 문제점:[1]
- 모델 용량을 늘리면 훈련이 불안정해짐
- 동일한 합성곱 필터가 모든 위치에서 동일하게 적용되어 표현 능력 제한
- 저해상도 블록이 제대로 활용되지 않음

#### 2.2 제안된 핵심 방법론

**가. Sample-Adaptive Kernel Selection (적응형 커널 선택)**

$$\mathcal{K} = \sum_{i=1}^{N} \mathcal{K}_i \cdot \text{softmax}(\mathcal{W}_{\text{filter}} \mathbf{w} + \mathbf{b}_{\text{filter}})_i$$

여기서:
- $\mathcal{K}\_i \in \mathbb{R}^{C_{in} \times C_{out} \times K \times K}$: N개 필터 뱅크
- $\mathbf{w} \in \mathbb{R}^{d}$: StyleGAN의 스타일 벡터
- softmax를 통한 미분 가능한 필터 선택[1]

이어서 생성된 필터로 정규화된 합성곱 수행:

$$\tilde{\mathbf{x}} = \mathcal{K} * (\mathcal{W}_{\text{mod}}(\mathbf{w}) \otimes \mathbf{f})$$

여기서 $\otimes$는 탈모듈레이션(de-modulation) 연산입니다.[1]

**나. Self-Attention과 Cross-Attention 통합**

StyleGAN에 주의 메커니즘을 추가하되, **L2 거리 기반 주의 로짓**을 사용하여 Lipschitz 연속성 유지:

$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(-\|\mathbf{Q} - \mathbf{K}\|_2^2 / \sqrt{d}\right) \mathbf{V}$$

- Self-Attention: 동일 특성에서 장거리 의존성 모델링
- Cross-Attention: 텍스트 임베딩 벡터를 query로 사용하여 조건부 생성[1]

**다. 다중 스케일 학습 및 Multi-Scale Input-Output (MS-IO) 판별기**

생성기는 다중 스케일 이미지 피라미드 생성:

$$\{\mathbf{x}^i\}_{i=0}^{L-1} = \{\mathbf{x}^0, \mathbf{x}^1, ..., \mathbf{x}^{4}\}$$

해상도: {64×64, 128×128, 256×256, 512×512, 1024×1024}[1]

판별기는 각 피라미드 수준에서 다중 스케일에서 예측:

$$\mathcal{V}_{MS-IO}(G, D) = \sum_{i=0}^{L-1} \sum_{j=i}^{L} \left[\mathcal{V}_{GAN}(G_i, D_{\phi_{ij}}) + \mathcal{V}_{\text{match}}(G_i, D_{\phi_{ij}})\right]$$

이는 저해상도 블록의 활용도를 크게 향상시킵니다.[1]

#### 2.3 손실 함수 구성

최종 훈련 목표:

$$\mathcal{L} = \mathcal{L}_{MS-IO} + \lambda_{CLIP} \mathcal{L}_{CLIP} + \lambda_{Vision} \mathcal{L}_{Vision}$$

**CLIP 대조 손실**:[1]

```math
\mathcal{L}_{CLIP} = \mathbb{E}_{c,n} \left[-\log \frac{\exp(\mathcal{E}_{img}(G(c)) \cdot \mathcal{E}_{txt}(c^0) / \tau)}{\sum_{n} \exp(\mathcal{E}_{img}(G(c)) \cdot \mathcal{E}_{txt}(c^n) / \tau)}\right]
```

여기서 $\mathcal{E}\_{img}$, $\mathcal{E}_{txt}$는 CLIP 인코더이고 $\tau$는 온도 파라미터.[1]

**매칭 인식 손실** (진정한 텍스트-이미지 정렬 강화):[1]

$$\mathcal{V}_{\text{match}} = \mathbb{E}_{x,c,c'} \left[\log(1 + e^{-D(x,c)}) + \log(1 + e^{D(G(c),c')})\right]$$

### 3. 모델 구조 상세

#### 3.1 생성기 아키텍처

```
입력: 잠재 코드 z ~ N(0,I), 텍스트 조건 c
↓
[CLIP 인코더] → 토큰 임베딩 (78×768)
↓
[학습가능 텍스트 트랜스포머 T] → 지역 및 전역 텍스트 표현
  - t_local: 개별 단어 임베딩
  - t_global: EOT (End-of-Token) 토큰
↓
[매핑 네트워크 M] z + t_global → 스타일 코드 w
↓
[합성 네트워크]
  ├─ 적응형 커널 선택 (g_adaconv)
  ├─ Self-Attention (g_self)
  └─ Cross-Attention (g_cross, t_local 사용)
↓
[다중 해상도 출력]
  x^0 (64×64) → x^4 (1024×1024)
```

**텍스트 조건부 입력**:[1]

$$\mathbf{t}_{\text{local}}, \mathbf{t}_{\text{global}} = \mathcal{T}(E_{\text{txt}}(c))$$
$$\mathbf{w} = \mathcal{M}(\mathbf{z}, \mathbf{t}_{\text{global}})$$
$$\mathbf{x} = G(\mathbf{w}, \mathbf{t}_{\text{local}})$$

#### 3.2 판별기 아키텍처

```
[이미지-텍스트 쌍] (x, c)
↓
이미지 처리 분기          텍스트 처리 분기
┌─────────────┐         ┌────────────────┐
│ 합성곱 + Self-Attn │  │ CLIP + Attn층  │
│ 다중 스케일처리     │  │ (전역 특성만)  │
└─────────────┘         └────────────────┘
        ↓                           ↓
      ϕ_ij(x)                   t_D(c)
        ↓────────────┬─────────────↓
                 [모듈레이션 예측기]
                      ↓
                 실/가짜 분류
```

판별기 출력:[1]

$$D(x, c) = \text{Conv}_{1×1}(\phi_{ij}(x)) + \Phi_j(\text{Modulate}(\phi_{ij}(x), t_D(c)))$$

### 4. 성능 향상 분석

#### 4.1 정량적 성과

**COCO 2014 제로샷 평가**:[2][1]

| 모델 | FID-30k | CLIP Score | 추론 시간 |
|------|---------|-----------|---------|
| DALLE 2 | 10.39 | - | - |
| Stable Diffusion v1.5 | 9.62 | - | 2.9s |
| **GigaGAN** | **9.09** | **0.307** | **0.13s** |
| Parti-750M | 10.71 | - | - |

**ImageNet 조건부 생성 (256px)**:[1]
- **IS (Inception Score)**: 225.52 (BigGAN-Deep: 224.46, StyleGAN-XL: 297.62)
- **FID**: 3.45 (StyleGAN-XL: 2.32, ADM-G-U: 4.01)

**업샘플러 성능 (128→1024px)**:[1]

| 방법 | FID | Patch-FID | CLIP | LPIPS |
|------|-----|----------|------|-------|
| Real-ESRGAN | 8.60 | 22.8 | 0.314 | 0.363 |
| SD Upscaler | 9.39 | 41.3 | 0.316 | 0.523 |
| **GigaGAN** | **1.54** | **8.90** | **0.322** | **0.274** |

#### 4.2 절제 연구 (Ablation Study)[1]

기본 StyleGAN2+텍스트 조건화부터 시작:

| 구성 | FID-10k | CLIP Score | 파라미터 |
|-----|---------|-----------|---------|
| StyleGAN2 기본 | 29.91 | 0.222 | 27.8M |
| + Attention | 23.87 | 0.235 | 59.0M |
| + Matching Loss (D) | 27.29 | 0.250 | 59.0M |
| + Matching Loss (G,D) | 21.66 | 0.254 | 59.0M |
| + Adaptive Conv | 19.97 | 0.261 | 80.2M |
| + Deeper Architecture | 19.18 | 0.263 | 161.9M |
| + CLIP Loss | 14.88 | 0.280 | 161.9M |
| + Multi-scale Training | 14.92 | **0.300** | 164.0M |
| + Vision-aided GAN | 13.67 | 0.287 | 164.0M |
| **최종 Scale-up** | **9.18** | **0.307** | **652.5M** |

#### 4.3 개선 메커니즘

1. **적응형 커널 선택**: 텍스트 특성에 따라 동적으로 필터 선택 → 표현력 증가
2. **Multi-scale Training**: 저해상도 블록 활용도 향상 → 구조 정보 개선
3. **교차주의 메커니즘**: 텍스트-이미지 정렬 개선 → CLIP 점수 0.300 달성
4. **CLIP 손실**: 대규모 사전 학습 모델 활용 → 의미론적 이해 강화

### 5. 일반화 성능 및 한계

#### 5.1 일반화 능력 분석

**강점**:[3][4][1]
- **제로샷 성능**: COCO에서 FID 9.09 달성 (Stable Diffusion과 경쟁 수준)
- **다양한 텍스트 프롬프트 처리**: 임의의 텍스트 설명으로부터 이미지 생성 가능
- **텍스트-이미지 정렬**: CLIP 점수 0.307로 높은 의미론적 일치도

**약점 및 한계**:[4][3][1]

1. **포토리얼리즘 부족**: DALLE 2와 비교하면 세부사항(디테일) 표현 능력 한계[1]
   - 세밀한 얼굴 특징, 정확한 객체 모양 등에서 성능 부족

2. **구성 일반화(Compositional Generalization)** 문제:
   - 학습하지 않은 형용사-명사 조합("blue petal", "long beak") 생성 어려움[3][4]
   - 복잡한 장면의 다중 객체 생성에서 정확도 감소

3. **모드 커버리지(Mode Coverage)**: 제한된 다양성
   - 동일 프롬프트로 여러 번 생성 시 유사 이미지 반복

4. **저주파 정보**: 
   - 전역 구조 이해도 한계 (예: 침대의 다리 개수 오류)
   - 기하학적 일관성 부족

#### 5.2 일반화 성능 향상 가능성

**논문에서 제시하는 개선 방향**:[1]
1. 더 큰 모델 규모로 확대 (Table 1에서 품질 포화 미관찰)
2. 더 큰 배치 크기 및 더 긴 훈련 일정
3. 다양한 데이터 전처리 및 증강 기법

**최신 연구 기반 개선 전략**:[5][6][7]

**a) 구성 일반화 강화** (CLIP-R-Precision 메트릭):[5][4]
- 지수 기반 학습(Examplar-based learning)을 통한 조합 문제 해결
- 대비 학습(Contrastive Learning) 강화

**b) 효율성 개선** (2024 최신 연구):[6]
- Diffusion vs GAN 비교에서 GAN이 동일한 계산 자원으로 더 나은 성능 가능
- 조건부 생성에서 GAN의 우월성 재평가

**c) 시각-의미 정렬(Visual-Semantic Alignment)** 개선:
- Vision Transformer 기반 판별기 강화
- 다중 예제 기반 메트릭 학습

#### 5.3 현재 한계 인식

논문 저자들의 명시적 언급:[1]
> "본 결과의 시각 품질이 반드시 더 나은 것은 아닙니다... DALLE 2와 같은 프로덕션 급 모델과 비교하면 사실성과 텍스트-이미지 정렬에서 제한이 있습니다."

### 6. 후속 연구에 미치는 영향 및 고려사항

#### 6.1 학술적 영향

**1) GAN의 부활론** (2023-2024):[8][2][1]
- StyleGAN-T (동시 발표), GALIP 등과 함께 GAN 기반 텍스트-이미지 합성의 실행 가능성 재확인
- Diffusion 모델의 독점적 지배 구도 재평가

**2) 아키텍처 설계 원리** (이후 연구의 영감):
- Sample-Adaptive Kernel Selection이 다른 생성 모델 분야로 확산
- Multi-Scale 판별기 설계의 표준화

**3) 효율성 재고** (2024-2025):
- Diffusion 모델이 항상 더 나은가? 동일 자원 조건에서 GAN의 경쟁성 입증[6]
- GAN 기반 초분해상도(SR)에서 Diffusion과 동등 또는 우월 성능

#### 6.2 앞으로의 연구 시 고려사항

**1) 평가 메트릭의 재정의**:[9][4][5]
- **기존 문제**: FID/CLIP Score만으로는 부족
- **개선 방향**: 
  - CLIP-R-Precision: 더 나은 텍스트-이미지 정렬 평가
  - Compositional Alignment (CA): 다중 객체 생성 평가
  - 인간 평가와 상관관계 향상

**2) 구성 일반화 능력 강화**:[4][3]
- 학습하지 않은 속성 조합에 대한 성능 평가 중요
- Zero-Shot Compositional Splits 벤치마크 활용

**3) 데이터 품질 및 전처리**:[1]
- LAION 같은 대규모 웹 크롤 데이터의 노이즈 문제
- CLIP 점수, 미학 점수 기반 고품질 필터링의 중요성 재확인

**4) 계산 효율성과 품질 트레이드오프**:[10][11][12]
- 모바일/엣지 기기 배포 시 GAN의 장점 극대화
- GAN 압축 기법 발전 (Knowledge Distillation, Neural Architecture Search)

**5) 다중 모달 입력 통합**:[13][7]
- 텍스트뿐 아니라 이미지, 스케치, 스타일 입력 조건부 생성
- Vision-Empowered Discriminator 확장

**6) 제어 가능한 생성(Controllable Generation)**:[14][1]
- GAN의 디엔탱글된 잠재 공간 활용 극대화
- 레이아웃 보존 스타일 제어, 프롬프트 기반 세밀 제어

#### 6.3 장기 연구 방향

| 차원 | 고려사항 | 2024-2025년 발전 현황 |
|------|--------|------------------|
| **아키텍처** | Transformer vs CNN 하이브리드 | TransGAN, Vision Transformer 기반 GAN 성숙화 |
| **확장성** | 10B+ 파라미터 모델 | DALLE 3, GPT-4V 수준의 멀티모달 이해 필요 |
| **효율성** | 추론 지연 시간 감소 | 0.13s → <0.05s 목표 (엣지 배포) |
| **안정성** | 훈련 안정성 개선 | Consistent Latent Space 기반 새로운 정규화 |
| **정렬** | 시각-의미 정렬 | Multimodal Pre-training (CLIP 기반) 필수화 |

### 7. 결론

**GigaGAN**은 GANs의 확장 가능성을 명확히 증명했으며, 텍스트-이미지 합성 분야에서 Diffusion/Autoregressive 모델과의 건설적 경쟁 구도를 형성했습니다. 특히 **0.13초의 초고속 추론, 디엔탱글된 잠재 공간, 4K 초고해상도 생성**은 실시간 상호작용 애플리케이션에서 GAN의 고유한 가치를 재확인시켰습니다.

다만 **포토리얼리즘, 구성 일반화, 복잡한 장면 이해** 분야에서는 여전히 Diffusion 모델에 미치지 못하고 있으며, 이는 향후 연구의 핵심 과제입니다. 2024-2025년 최신 연구에서 보듯이, **동일 계산 자원 조건에서 GAN과 Diffusion의 경계가 모호해지고 있으며**, 최적의 선택은 애플리케이션의 우선순위(속도 vs 품질 vs 제어성)에 따라 결정되어야 합니다.

***

### 참고 자료

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/43706893-e6e1-49cc-aea6-da8b222981af/2303.05511v2.pdf)
[2](https://openaccess.thecvf.com/content/CVPR2023/papers/Kang_Scaling_Up_GANs_for_Text-to-Image_Synthesis_CVPR_2023_paper.pdf)
[3](https://openreview.net/forum?id=bKBhQhPeKaF)
[4](https://datasets-benchmarks-proceedings.neurips.cc/paper_files/paper/2021/file/0a09c8844ba8f0936c20bd791130d6b6-Paper-round1.pdf)
[5](https://www.sapien.io/blog/gans-vs-diffusion-models-a-comparative-analysis)
[6](https://openreview.net/forum?id=46mbA3vu25)
[7](https://arxiv.org/html/2501.00116v1)
[8](http://arxiv.org/pdf/2301.09515.pdf)
[9](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136960585.pdf)
[10](https://arxiv.org/html/2411.03999)
[11](https://openaccess.thecvf.com/content_CVPR_2020/papers/Li_GAN_Compression_Efficient_Architectures_for_Interactive_Conditional_GANs_CVPR_2020_paper.pdf)
[12](https://dspace.mit.edu/bitstream/handle/1721.1/143671/2003.08936.pdf?sequence=2&isAllowed=y)
[13](http://arxiv.org/pdf/2501.02167.pdf)
[14](https://arxiv.org/abs/2401.06127)
[15](https://arxiv.org/pdf/2205.11273.pdf)
[16](https://arxiv.org/abs/2303.05511)
[17](https://arxiv.org/pdf/2209.01339.pdf)
[18](https://arxiv.org/abs/2403.19645)
[19](http://arxiv.org/pdf/1909.03611.pdf)
[20](https://mingukkang.github.io/GigaGAN/)
[21](https://www.sabrepc.com/blog/Deep-Learning-and-AI/gans-vs-diffusion-models)
[22](https://kimjy99.github.io/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0/gigagan/)
[23](http://arxiv.org/pdf/2105.13290.pdf)
[24](https://arxiv.org/pdf/2301.00704.pdf)
[25](http://arxiv.org/pdf/2205.11487.pdf)
[26](http://arxiv.org/pdf/2102.07074v2.pdf)
[27](http://arxiv.org/pdf/2407.00752.pdf)
[28](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/00072.pdf)
[29](https://eprints.soton.ac.uk/473733/1/yue_jiao_phd_vlc_ecs_11_05_1_.pdf)
[30](https://openaccess.thecvf.com/content/CVPR2024/papers/Liang_Rich_Human_Feedback_for_Text-to-Image_Generation_CVPR_2024_paper.pdf)
[31](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/08701.pdf)
[32](https://ijrpr.com/uploads/V5ISSUE12/IJRPR36053.pdf)
[33](https://arxiv.org/html/2407.21794v1)
[34](https://dl.acm.org/doi/10.1016/j.imavis.2021.104284)
