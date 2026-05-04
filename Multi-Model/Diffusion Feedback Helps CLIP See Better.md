# Diffusion Feedback Helps CLIP See Better

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

CLIP 모델은 방향(orientation), 수량(quantity), 색상(color), 구조(structure) 등 **세밀한 시각적 세부 정보(fine-grained visual details)를 구별하는 능력이 부족**하며, 이는 CLIP을 비전 인코더로 사용하는 MLLMs(Multimodal Large Language Models)의 성능도 제한한다. 본 논문은 **텍스트-이미지 확산 모델(diffusion model)의 생성적 피드백(generative feedback)을 활용한 자기지도 학습(self-supervised learning) 방식의 사후 훈련(post-training)**으로 이 문제를 해결할 수 있음을 주장한다.

### 주요 기여

1. **최초의 생성 피드백 기반 CLIP 최적화 프레임워크**: 텍스트-이미지 확산 모델의 생성적 피드백을 직접 활용하여 CLIP의 판별적 표현(discriminative representation)을 최적화하는 최초의 연구
2. **DIVA 프레임워크 제안**: 이미지만(텍스트 불필요) 사용하는 자기지도 학습 프레임워크로, **Visual Dense Recap** 전략을 통해 CLIP의 시각 특징을 확산 모델의 조건(condition)으로 활용
3. **광범위한 성능 향상 검증**: MMVP-VLM 벤치마크에서 3~7% 성능 향상, 29개 제로샷 분류·검색 벤치마크에서 기존 일반화 능력 유지

---

## 2. 해결 문제, 제안 방법, 모델 구조, 성능 및 한계

### 2.1 해결하고자 하는 문제

CLIP의 시각적 결함은 두 가지 근본적 원인에서 비롯된다:

**① 학습 패러다임의 문제 (Training Paradigm)**
- CLIP의 대조 학습(contrastive learning)은 positive pair 간 거리를 최소화하고 negative pair 간 거리를 최대화하는 방식
- 이는 고수준 의미론적(semantic) 정보에만 집중하게 하여, 방향·수량·색상·구조 등 세부 시각 정보를 간과

**② 데이터 형식의 문제 (Data Format)**
- CLIP 훈련에 사용된 이미지-텍스트 쌍에서 텍스트의 실질 유효 길이는 ~20 토큰 미만 (Zhang et al., 2024)
- 텍스트에 이미지의 시각적 세부 정보가 충분히 기술되지 않음

### 2.2 제안하는 방법 (수식 포함)

#### 2.2.1 확산 모델 기초 (Preliminaries)

**순방향 확산 과정(Forward Diffusion Process)**:

$$\mathbf{x}_t = \sqrt{1 - \beta_t}\mathbf{x}_{t-1} + \sqrt{\beta_t}\epsilon_t, \quad t = 1, \ldots, T $$

가우시안 분포의 합산 성질을 활용한 재공식화:

$$\mathbf{x}_t = \sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t}\epsilon, \quad t = 1, \ldots, T $$

여기서 $\alpha_t = 1 - \beta_t$, $\bar{\alpha}\_t = \prod_{i=1}^{t}\alpha_i$

**역방향 생성 과정(Reverse Process)**:

$$\mathbf{x}_{t-1} = \frac{1}{\sqrt{\alpha_t}}\left(\mathbf{x}_t - \frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}}\epsilon_\phi(\mathbf{x}_t, t)\right) + \sigma_t\epsilon, \quad t = T, \ldots, 1 $$

**비조건부 확산 모델 학습 목표**:

$$\mathcal{L}(\phi) = \mathbb{E}_{t, \mathbf{x}_0, \epsilon}\left[\|\epsilon - \epsilon_\phi(\sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1-\bar{\alpha}_t}\epsilon, t)\|^2\right] $$

**조건부 확산 모델 학습 목표 (DIVA의 핵심)**:

$$\mathcal{L}(\phi) = \mathbb{E}_{t, \mathbf{x}_0, \epsilon, \mathbf{c}}\left[\|\epsilon - \epsilon_\phi(\sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1-\bar{\alpha}_t}\epsilon, t, \mathbf{c})\|^2\right] $$

여기서 조건 $\mathbf{c}$는 CLIP의 시각 특징으로 구성됨

#### 2.2.2 DIVA 알고리즘 (의사코드 기반)

$$\mathcal{L}(\theta, \phi) = \frac{1}{N}\sum_{i=1}^{B}\|\epsilon_\phi(\sqrt{\bar{\alpha}_{t_i}}\mathbf{x} + \sqrt{1-\bar{\alpha}_{t_i}}\epsilon_i, \mathbf{c}, t_i) - \epsilon_i\|^2 $$

$$\theta^* \leftarrow \theta - \eta \nabla_\theta \mathcal{L}(\theta, \phi) $$

- CLIP 파라미터 $\theta$만 업데이트, 확산 모델 파라미터 $\phi$는 고정(frozen)
- 이미지-텍스트 쌍 불필요: **이미지만으로 자기지도 학습**

#### 2.2.3 Visual Dense Recap 전략

| 조건 구성 | 설명 |
|---|---|
| CLS 토큰만 | 의미론적 정보만 포함, 재구성 난이도 과도하게 높음 |
| CLS + 일부 패치 토큰 | **최적 균형점** → 최대 +6.6% 성능 향상 |
| CLS + 전체 패치 토큰 | 조건 정보 과잉, 재구성 난이도 지나치게 낮아 학습 효과 제한 |

모델별 설정:
- OpenAI CLIP ViT-L-14/224: 로컬 토큰 약 15% 무작위 선택
- OpenAI CLIP ViT-L-14/336: 로컬 토큰 약 30% 무작위 선택
- SigLIP ViT-SO-14/224&384: 윈도우 크기 6, 10의 1D 평균 풀링
- DFN ViT-H-14/378: 50% 무작위 선택

### 2.3 모델 구조

```
입력 이미지 x₀
    │
    ▼
[CLIP 비전 인코더 θ] ──→ CLS 토큰 + 일부 패치 토큰
    │                              │
    │                              ▼
    │                    [조건 c 구성]
    │                    (시각 특징 + 빈 텍스트 임베딩[BOS][EOS])
    │                              │
    ▼                              ▼
노이즈 추가 xₜ ──────────→ [확산 모델 Denoiser εφ] (고정)
                                   │
                                   ▼
                          노이즈 예측 ε̂ → 재구성 손실
                                   │
                                   ▼ (역전파)
                        CLIP 파라미터 θ 업데이트
```

**구현 세부 사항**:
- GPU: 8 × NVIDIA A100 80GB
- 배치 크기: 640
- 옵티마이저: SGD (lr=1e-4, momentum=0.9)
- 훈련 데이터: Conceptual Captions 3M (CC-3M)
- 훈련 스텝: 4,600 스텝 (~1 에폭)
- 총 훈련 시간: 약 66.4 GPU×시간 (≈ 8.3시간 on 8 GPUs)
- 확산 모델: Stable Diffusion 2.1-base (최적)

### 2.4 성능 향상

#### MMVP-VLM 벤치마크 (세밀한 시각 능력 평가)

| 모델 | 기준선 | DIVA 적용 후 | 향상 |
|---|---|---|---|
| OpenAI ViT-L-14/224 | 19.3% | 25.9% | **+6.6%** |
| OpenAI ViT-L-14/336 | 20.0% | 25.2% | **+5.2%** |
| MetaCLIP ViT-H-14/224 | 25.2% | 31.9% | **+6.7%** |
| SigLIP ViT-SO-14/224 | 37.8% | 40.7% | **+2.9%** |
| DFN ViT-H-14/224 | 39.3% | 43.7% | **+4.4%** |

#### LLaVA-1.5 MLLMs 성능 향상 (Table 2)

| 모델 | MMVP | POPE (rand/pop/adv) | MMBench-EN |
|---|---|---|---|
| LLaVA1.5-7B (기준) | 24.7 | 87.3/86.1/84.2 | 64.3 |
| LLaVA1.5-7B + DIVA | **31.3** | **87.9/87.0/84.6** | **66.4** |
| LLaVA1.5-13B (기준) | 30.7 | 87.1/86.2/84.5 | 67.7 |
| LLaVA1.5-13B + DIVA | **35.3** | **88.1/87.4/84.8** | **69.4** |

#### 의미론적 분할 (SAN, Table 3)

| 백본 | ADE20K-150 | Pascal Context-459 | Pascal Context-59 |
|---|---|---|---|
| ViT-L-14/224 (기준) | 29.2 | 14.2 | 55.8 |
| ViT-L-14/224 + DIVA | **30.2** | **15.4** | **56.7** |

### 2.5 한계점

논문에서 명시된 한계:
1. **데이터 스케일 제한**: CC-3M만 사용; 더 큰 데이터셋으로 확장 시 추가 성능 향상 기대 (데이터 스케일링 특성 관측됨)
2. **감독 신호의 단순성**: 단순한 재구성 손실만 사용; 더 세밀한 감독 방식과 결합 가능성
3. **단일 모달리티 집중**: 현재는 이미지-텍스트에 한정; 비디오·오디오 등 추가 모달리티 미탐색
4. **특정 확산 모델 의존성**: DiT-XL/2 사용 시 오히려 성능 저하 관측 (표현 품질이 낮은 확산 모델 적용 불가)

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 제로샷 능력 보존: 핵심 발견

DIVA의 가장 중요한 특성 중 하나는 **세밀한 시각 인식을 향상시키면서도 CLIP의 뛰어난 일반화 능력을 손상시키지 않는다**는 점이다.

**27개 제로샷 이미지 분류 벤치마크 결과 (Table 4)**:

| 모델 | ImageNet-1K | ImageNet-V2 | ImageNet-Adv. | 평균 top-1 |
|---|---|---|---|---|
| OpenAI ViT-L-14 | 75.5 | 69.8 | 70.7 | 69.3 |
| OpenAI ViT-L-14 + DIVA | **75.5** | 69.7 | **70.8** | **69.3** |
| MetaCLIP ViT-H-14 | 78.5 | 72.1 | 69.6 | 74.4 |
| MetaCLIP ViT-H-14 + DIVA | 78.4 | 71.9 | 69.1 | 74.2 |

→ 평균 정확도 차이 **±0.2% 이내**로, 실질적 성능 저하 없음

**제로샷 검색 성능 (Table 5)**:

| 모델 | Flickr30K R@1 (텍스트) | COCO R@1 (텍스트) |
|---|---|---|
| OpenAI ViT-L-14 | 85.1 | 56.4 |
| OpenAI ViT-L-14 + DIVA | **85.3** | **56.7** |

### 3.2 일반화 향상의 메커니즘 분석

#### 메커니즘 1: 자기지도 학습의 도메인 불편성(Domain Agnosticism)

DIVA는 이미지-텍스트 쌍 의존성을 제거하고 **이미지만으로 학습**하므로, 특정 도메인의 텍스트 편향에서 자유롭다. 이는 다음과 같은 일반화 이점을 제공한다:

$$\theta^* = \arg\min_\theta \mathbb{E}_{\mathbf{x} \sim p(\mathbf{x})}\left[\mathcal{L}_{diffusion}(\theta, \phi; \mathbf{x})\right]$$

텍스트 조건 없이 이미지 분포 $p(\mathbf{x})$만을 학습하므로, 텍스트-이미지 쌍의 노이즈 및 편향에 영향을 받지 않음.

#### 메커니즘 2: 데이터 스케일링 특성 (Data Scaling Property)

Table 7에서 확인된 **데이터 스케일에 따른 단조 증가 특성**:

| 데이터 비율 | MMVP-VLM 평균 점수 | 향상폭 |
|---|---|---|
| 0% (기준) | 19.3 | - |
| 25% | 20.7 | +1.4 |
| 50% | 22.2 | +2.9 |
| 75% | 23.7 | +4.4 |
| 100% | 25.9 | +6.6 |

→ **성능 감소 없이 데이터 증가에 비례하여 성능 향상**, 더 대규모 데이터셋 적용 시 추가 향상 가능성 시사

#### 메커니즘 3: 다양한 CLIP 아키텍처에 대한 범용성

DIVA는 다음 모든 아키텍처에 성능 향상을 제공:
- OpenAI CLIP (Radford et al., 2021)
- EVA-CLIP (Fang et al., 2023)
- MetaCLIP (Xu et al., 2023a)
- SigLIP (Zhai et al., 2023)
- DFN (Fang et al., 2023)

→ 모델 크기, 해상도, 훈련 방법론에 무관하게 일관된 향상

#### 메커니즘 4: 적절한 재구성 난이도의 설계

Visual Dense Recap의 **"Goldilocks" 원칙**:

$$\text{최적 조건 밀도} = \arg\max_{\rho} \text{MMVP-VLM 성능}(\rho)$$

- 과도한 조건($\rho \to 1$): 재구성 너무 쉬움 → 표현 최적화 미미
- 부족한 조건($\rho \to 0$): 재구성 불가능 → 학습 실패
- **적절한 밀도(~15-30%)**: 최적 학습 신호 제공

### 3.3 일반화 한계와 가능성

**현재 일반화의 한계**:
- 고수준 의미론적 태스크(ImageNet 분류, 검색)에서는 DIVA 적용 후 큰 향상 없음 → 이미 CLIP이 강한 영역에서는 효과 제한적
- DiT 기반 확산 모델 사용 시 성능 저하 → 확산 모델 품질에 의존

**향후 일반화 가능성**:
- 더 대규모 데이터(CC-12M, LAION 등) 적용 시 선형적 성능 향상 기대
- 비디오/오디오 모달리티로 확장 가능성
- 더 정교한 확산 모델 활용 시 추가 향상 기대

---

## 4. 2020년 이후 관련 최신 연구 비교 분석

### 4.1 CLIP 개선 관련 연구

| 연구 | 방법 | 데이터 요구사항 | 세밀 시각 | 일반화 |
|---|---|---|---|---|
| **CLIP** (Radford et al., 2021) | 대조 학습 | 이미지-텍스트 쌍 | ❌ 취약 | ✅ 우수 |
| **EVA-CLIP** (Sun et al., 2023) | 개선된 사전훈련 | 이미지-텍스트 쌍 | △ 부분적 | ✅ 우수 |
| **SigLIP** (Zhai et al., 2023) | Sigmoid 손실 | 이미지-텍스트 쌍 | △ 부분적 | ✅ 우수 |
| **LongCLIP** (Zhang et al., 2024) | 긴 텍스트 파인튜닝 | 이미지-텍스트 쌍 | △ 텍스트만 | ✅ 유지 |
| **MetaCLIP** (Xu et al., 2023a) | 데이터 큐레이션 | 이미지-텍스트 쌍 | △ 부분적 | ✅ 우수 |
| **DIVA** (본 논문, 2024) | 확산 피드백 | **이미지만** | ✅ 대폭 향상 | ✅ 유지 |

### 4.2 확산 모델 기반 표현 학습 관련 연구

| 연구 | 방법 | 목표 | DIVA와의 차이 |
|---|---|---|---|
| **Diffusion-TTA** (Prabhudesai et al., 2023, NeurIPS) | 테스트 시 적응 | 도메인 적응 | 텍스트 조건 사용, 표현 수준 최적화 아님 |
| **SODA** (Hudson et al., 2024, CVPR) | 병목 확산 모델 | 표현 학습 | 자체 확산 모델 훈련, 더 높은 비용 |
| **StableRep** (Tian et al., 2024, NeurIPS) | 합성 데이터 생성 | 표현 강화 | 합성 데이터 활용, CLIP 직접 최적화 아님 |
| **DDAEs** (Xiang et al., 2023, ICCV) | 확산 오토인코더 | 자기지도 학습 | 분류/분할 특화, CLIP 최적화 아님 |
| **DIVA** (본 논문) | 생성 피드백 직접 활용 | CLIP 표현 최적화 | **CLIP 직접 최적화 + 이미지만 사용** |

### 4.3 MLLM의 시각 인코더 개선 관련 연구

| 연구 | 방법 | CLIP 직접 개선 | 계산 비용 |
|---|---|---|---|
| **BRAVE** (Kar et al., 2024) | 다중 비전 인코더 앙상블 | ❌ 우회 방식 | 높음 |
| **Cambrian-1** (Tong et al., 2024a) | 다중 비전 인코더 | ❌ 우회 방식 | 높음 |
| **Eyes Wide Shut** (Tong et al., 2024b) | 문제 분석 및 진단 | ❌ 해결책 제시 안함 | - |
| **DIVA** (본 논문) | 단일 CLIP 직접 개선 | ✅ | **낮음 (~8.3h)** |

---

## 5. 향후 연구에 미치는 영향과 고려 사항

### 5.1 향후 연구에 미치는 영향

#### 영향 1: 생성 모델과 판별 모델의 상호 보완적 활용 패러다임 확립

DIVA는 **생성 모델(generative)과 판별 모델(discriminative)이 서로를 개선할 수 있다**는 패러다임을 실증적으로 보여준다. 이는:
- 생성 모델 → 판별 모델 지식 전달의 새로운 방향 제시
- 역방향(판별 → 생성) 연구의 가능성도 시사

#### 영향 2: 이미지-텍스트 쌍 의존성 탈피의 중요성

비전-언어 모델 연구에서 **이미지 단독 데이터의 활용 가능성**을 강조:
- 텍스트 어노테이션이 없는 대규모 이미지 데이터 활용 방안 연구 촉진
- 웹 스케일 이미지 데이터(텍스트 없는)를 활용한 모델 개선 방향

#### 영향 3: CLIP 기반 MLLMs의 성능 향상 방향성

현재 MLLMs 연구에서 비전 인코더 교체 대신 **기존 CLIP을 직접 개선**하는 방향의 중요성을 보여줌:
- 계산 비용 절감
- 기존 MLLM 훈련 파이프라인과 직접 호환

#### 영향 4: 자기지도 학습의 새로운 감독 신호

재구성 기반 손실을 자기지도 감독 신호로 활용하는 방식은:
- MAE(Masked Autoencoder)의 픽셀 재구성 + 확산 모델의 의미론적 풍부성을 결합
- 향후 더 복잡한 재구성 목표 설계 연구 자극

### 5.2 향후 연구 시 고려할 사항

#### 고려 사항 1: 데이터 스케일 확장 실험

논문에서 CC-3M만 사용했으나, 데이터 스케일링 특성이 확인되었으므로:
- LAION-400M, LAION-5B 등 대규모 데이터셋 적용 효과 탐구
- 데이터 품질 vs. 양의 트레이드오프 분석 필요

#### 고려 사항 2: 더 정교한 조건 설계

현재 Visual Dense Recap은 휴리스틱하게 설정됨:
- 최적 토큰 선택 전략의 이론적 정립 필요
- 내용 기반 적응적(adaptive) 토큰 선택 메커니즘 연구

#### 고려 사항 3: 더 세밀한 감독 신호와의 결합

논문 자체에서 언급된 한계:
- 주의 메커니즘(attention) 기반 세밀 감독과의 결합
- 대조 학습(contrastive loss)과 재구성 손실의 균형 최적화

$$\mathcal{L}_{total} = \lambda_1 \mathcal{L}_{diffusion} + \lambda_2 \mathcal{L}_{contrastive} + \lambda_3 \mathcal{L}_{fine-grained}$$

#### 고려 사항 4: 다양한 모달리티로의 확장

- **비디오**: 시간적 일관성을 학습하는 확산 피드백 활용
- **3D**: 공간적 구조 인식 개선
- **오디오**: 시청각 정렬(audio-visual alignment) 개선

#### 고려 사항 5: 확산 모델 선택의 이론적 기준

DiT 사용 시 성능 저하 관측 → 확산 모델의 **표현 품질과 CLIP 최적화 효과 간의 관계** 이론적 분석 필요:

$$\text{CLIP 향상} \propto f(\text{확산 모델 표현 품질}, \text{조건 설계})$$

#### 고려 사항 6: 계산 효율성 최적화

- LoRA 등 파라미터 효율적 파인튜닝과 결합 가능성
- 지식 증류(knowledge distillation)와의 결합으로 소형 모델에 적용

#### 고려 사항 7: 평가 벤치마크의 다양화

- MMVP-VLM 외 더 다양한 세밀 시각 인식 벤치마크 필요
- 실제 응용 시나리오(의료 영상, 위성 이미지 등)에서의 검증

---

## 참고 자료

**논문 원문**:
- Wenxuan Wang, Quan Sun, Fan Zhang, Yepeng Tang, Jing Liu, Xinlong Wang. **"Diffusion Feedback Helps CLIP See Better"**. arXiv:2407.20171v4, 2024.

**본 논문에서 인용된 주요 관련 연구**:
- Radford et al. (2021). "Learning Transferable Visual Models from Natural Language Supervision." *ICML 2021*. (CLIP 원논문)
- Tong et al. (2024b). "Eyes Wide Shut? Exploring the Visual Shortcomings of Multimodal LLMs." *CVPR 2024*.
- Ho et al. (2020). "Denoising Diffusion Probabilistic Models." *NeurIPS 2020*.
- Rombach et al. (2022). "High-Resolution Image Synthesis with Latent Diffusion Models." *CVPR 2022*. (Stable Diffusion)
- Liu et al. (2024a). "Improved Baselines with Visual Instruction Tuning." *CVPR 2024*. (LLaVA-1.5)
- Prabhudesai et al. (2023). "Diffusion-TTA: Test-Time Adaptation of Discriminative Models via Generative Feedback." *NeurIPS 2023*.
- Zhai et al. (2023). "Sigmoid Loss for Language Image Pre-Training." *ICCV 2023*. (SigLIP)
- Xu et al. (2023a). "Demystifying CLIP Data." (MetaCLIP)
- Zhang et al. (2024). "Long-CLIP: Unlocking the Long-Text Capability of CLIP." arXiv:2403.15378.
- Tian et al. (2024). "StableRep: Synthetic Images from Text-to-Image Models Make Strong Visual Representation Learners." *NeurIPS 2024*.
- Hudson et al. (2024). "SODA: Bottleneck Diffusion Models for Representation Learning." *CVPR 2024*.
- Peebles & Xie (2023). "Scalable Diffusion Models with Transformers." *ICCV 2023*. (DiT)
- Tong et al. (2024a). "Cambrian-1: A Fully Open, Vision-Centric Exploration of Multimodal LLMs." arXiv:2406.16860.
- Sharma et al. (2018). "Conceptual Captions." *ACL 2018*. (CC-3M 데이터셋)
- Xu et al. (2023c). "Side Adapter Network for Open-Vocabulary Semantic Segmentation." *CVPR 2023*. (SAN)

**프로젝트 페이지**: https://rubics-xuan.github.io/DIVA/

**코드**: https://github.com/baaivision/DIVA
