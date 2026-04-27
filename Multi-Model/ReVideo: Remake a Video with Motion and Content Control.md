
# ReVideo: Remake a Video with Motion and Content Control

> **논문 정보:**
> - 저자: Chong Mou, Mingdeng Cao, Xintao Wang, Zhaoyang Zhang, Ying Shan, Jian Zhang
> - 발표: NeurIPS 2024
> - arXiv: [2405.13865](https://arxiv.org/abs/2405.13865) (2024년 5월 22일)
> - 공식 NeurIPS 페이지: https://neurips.cc/virtual/2024/poster/93082

---

## 1. 핵심 주장 및 주요 기여 요약

확산 모델(Diffusion Model)을 활용한 비디오 생성·편집 분야가 크게 발전했음에도 불구하고, 정확하고 지역화된 비디오 편집은 여전히 중요한 도전 과제로 남아 있다. 또한 기존 대부분의 비디오 편집 방법은 시각적 콘텐츠 변경에 초점을 맞추고 있으며, 모션 편집에 대한 연구는 매우 제한적이다.

이를 해결하기 위해 ReVideo는 다음의 핵심 주장과 기여를 제안한다:

ReVideo는 콘텐츠와 모션 모두를 지정함으로써 특정 영역에서 정밀한 비디오 편집을 가능하게 하는 새로운 시도이다. 콘텐츠 편집은 첫 번째 프레임을 수정함으로써 이루어지며, 궤적(trajectory) 기반의 모션 제어는 직관적인 사용자 인터랙션 경험을 제공한다.

### 주요 기여 (Contributions) 요약

| 기여 항목 | 내용 |
|---|---|
| **신규 태스크 정의** | 콘텐츠 + 모션 동시 로컬 편집 (최초 시도) |
| **3단계 훈련 전략** | Content/Motion 결합 문제를 점진적으로 분리 |
| **SAFM 모듈** | 공간-시간 적응 융합으로 정밀 제어 |
| **멀티-영역 편집** | 별도 훈련 없이 다중 영역 편집 일반화 |

저자들의 표현에 따르면, 이것은 비디오에서 콘텐츠와 모션을 동시에 로컬 편집하는 이 태스크에 대한 최초의 시도이다.

---

## 2. 해결하고자 하는 문제, 제안 방법, 모델 구조, 성능 및 한계

### 2.1 해결하고자 하는 문제

이 새로운 태스크에서 편집되지 않는 콘텐츠와 모션 커스터마이제이션 간에 **결합(coupling) 문제**가 존재한다. 두 제어 조건을 비디오 생성 모델에 직접 동시에 훈련시키면 모션 제어가 무시되는 현상이 발생한다.

구체적으로는 두 가지 핵심 문제가 있다:

1. **콘텐츠-모션 결합 문제 (Content-Motion Coupling):** 콘텐츠와 모션 제어 간의 결합 및 훈련 불균형(training imbalance) 문제.
2. **모션 제어 무시 문제:** Coarse-to-fine 훈련 전략을 사용하더라도, 일부 복잡한 모션 궤적에서 상당한 실패 사례가 관찰된다.

---

### 2.2 제안하는 방법

#### 2.2.1 3단계 훈련 전략 (Three-Stage Training Strategy)

저자들은 이 문제를 해결하기 위해, 두 조건을 거칠게(coarse)에서 세밀하게(fine)로 결합하는 3단계 훈련 전략을 개발한다.

각 단계는 다음과 같이 구성된다 (논문에 기술된 내용 기반):

- **Stage 1: 콘텐츠 제어 훈련 (Content Control Training)**
  - 첫 번째 프레임 수정을 통해 비디오 전체에 콘텐츠 변경을 전파
  - 기반 모델: Stable Video Diffusion (SVD)

- **Stage 2: 모션 제어 훈련 (Motion Decoupling Training)**
  - 궤적 기반 모션 신호를 추가 도입
  - 콘텐츠 인코더는 고정(freeze)하고 모션 인코더를 훈련

- **Stage 3: 디블로킹 훈련 (Deblocking Training)**
  - SVD 모델 및 제어 모듈에서 temporal self-attention layer의 key embedding과 value embedding만 fine-tune하여 편집 영역 경계의 아티팩트를 제거.

---

#### 2.2.2 공간-시간 적응 융합 모듈 (SAFM: Spatiotemporal Adaptive Fusion Module)

콘텐츠 제어와 모션 궤적의 생성 내 제어 역할을 더욱 구분하기 위해, SAFM이 설계된다. SAFM은 직접 합산(direct summing) 대신 가중치 맵 $\mathbf{M}$을 예측하여 모션과 콘텐츠 제어를 융합한다. 또한 확산 생성은 다중 단계 반복 과정이므로, 타임스텝 간 제어 조건 융합에 적응적 조정이 필요하다.

**SAFM의 수식 표현:**

SAFM의 두 인코더 $E_c$ (콘텐츠)와 $E_m$ (모션)은 각 조건을 인코딩하며, 최종 융합은 다음과 같이 공식화된다:

$$\mathbf{F}_{out} = \mathbf{M} \cdot \mathbf{F}_c + (1 - \mathbf{M}) \cdot \mathbf{F}_m$$

여기서:
- $\mathbf{F}_c$: 콘텐츠 인코더 $E_c$의 출력 특징(feature)
- $\mathbf{F}_m$: 모션 인코더 $E_m$의 출력 특징
- $\mathbf{M}$: 공간적 위치 및 타임스텝 $t$에 따라 예측되는 가중치 맵

SAFM의 입력 단계에서, 두 인코더 $E_c$와 $E_m$은 콘텐츠와 모션 조건을 별도로 인코딩한다. 두 인코더는 동일한 저비용(low-cost) 구조를 공유하며, 각 서브블록은 합성곱(convolution)과 다운샘플링(downsampling) 연산으로 구성되어 조건 맵을 잠재 공간(latent)과 동일한 크기로 매핑한다.

**SAFM의 가중치 맵 예측 (타임스텝 적응 포함):**

$$\mathbf{M} = \sigma\left(f\left(\mathbf{F}_c, \mathbf{F}_m, \mathbf{\Gamma}(t)\right)\right)$$

여기서 $\mathbf{\Gamma}(t)$는 타임스텝 임베딩, $\sigma$는 sigmoid 함수, $f$는 예측 네트워크를 나타낸다.

> ⚠️ **주의:** SAFM의 내부 세부 수식 일부는 논문 HTML 렌더링에서 기호 손상이 있어, 위 수식은 논문의 서술적 설명을 LaTeX로 재구성한 것입니다.

**SAFM 효과 검증:**

SAFM의 유효성을 입증하기 위해, SAFM을 모션·콘텐츠 제어의 단순 합산으로 교체한 결과, 단순 합산 방식은 파형 궤적(wavy lines)과 같이 복잡한 모션 궤적에서 정확한 모션 제어에 실패함을 확인하였다.

---

### 2.3 모델 구조

ReVideo의 전체 파이프라인은 다음과 같이 구성된다:

```
[입력]
├── 편집된 첫 번째 프레임 (콘텐츠 신호)
├── 사용자 정의 모션 궤적 (모션 신호)
└── 편집 마스크 (편집 영역 지정)
        ↓
[SAFM (Spatiotemporal Adaptive Fusion Module)]
├── E_c: 콘텐츠 인코더
├── E_m: 모션 인코더
└── 가중치 맵 M 예측 → 융합
        ↓
[SVD 기반 비디오 확산 모델]
├── 3단계 훈련된 제어 모듈
└── Temporal Self-Attention Fine-tuning
        ↓
[출력: 편집된 비디오]
```

이 방법은 단일 제어 모듈을 통해 모션과 콘텐츠 조건을 확산 비디오 생성에 효율적으로 주입한다. 이를 통해 사용자는 첫 번째 프레임을 수정하고 궤적 선을 그리는 것만으로 비디오의 특정 영역을 편리하게 편집할 수 있다. ReVideo는 단일 영역 편집에 국한되지 않고 여러 영역을 병렬로 커스터마이즈할 수 있다.

**모션 궤적 표현:**

궤적은 다음과 같이 좌표 시퀀스로 표현된다:

$$\mathcal{T} = \{(x_1, y_1), (x_2, y_2), \ldots, (x_T, y_T)\}$$

여기서 $T$는 비디오 프레임 수이며, 각 $(x_t, y_t)$는 $t$번째 프레임에서의 객체 위치를 나타낸다.

---

### 2.4 성능 향상

ReVideo는 InsV2V 및 AnyV2V 대비 모든 평가 지표에서 현저히 우월한 성능을 달성한다. 예를 들어 ReVideo는 PSNR 32.85, 텍스트 정렬 CLIP 점수 0.2304, 일관성 CLIP 점수 0.9864를 달성하며, InsV2V(PSNR 29.77, 텍스트 정렬 0.2022)와 AnyV2V(PSNR 29.80, 텍스트 정렬 0.2143)를 크게 앞선다.

Pika와 비교 시 ReVideo는 시간적 일관성(0.9864 vs 0.9956) 및 비편집 콘텐츠 품질(PSNR 32.85 vs 33.07)에서 소폭 낮지만, 텍스트 정렬 CLIP 점수(0.2304 vs 0.2184)와 인간 평가 점수에서 전반적 품질(59.1% vs 27.9%) 및 편집 목표 달성(67.0% vs 23.9%)에서 압도적으로 우월하다.

| 지표 | ReVideo | InsV2V | AnyV2V | Pika |
|---|---|---|---|---|
| PSNR ↑ | **32.85** | 29.77 | 29.80 | 33.07 |
| CLIP (텍스트 정렬) ↑ | **0.2304** | 0.2022 | 0.2143 | 0.2184 |
| CLIP (시간 일관성) ↑ | 0.9864 | 0.9808 | 0.9836 | **0.9956** |
| 전체 품질 (Human) ↑ | **59.1%** | - | - | 27.9% |
| 편집 달성도 (Human) ↑ | **67.0%** | - | - | 23.9% |

---

### 2.5 한계 (Limitations)

오류 누적(error accumulation)이 편집 품질에 영향을 미칠 수 있음이 관찰된다. 이는 장 비디오(long video) 편집의 내재적 문제이며, 더 강력한 기반 모델을 사용함으로써 완화될 수 있다.

첫 번째 프레임에 등장하지 않는 영역에서 생성되는 콘텐츠는 완전히 할루시네이션(hallucination)될 수 있다. ReVideo는 원본 비디오의 정보를 인페인팅(inpainting)하여 이 문제를 해결하려 하지만, 카메라 움직임이 포함된 비디오에서는 이 방식이 근본적으로 무너진다.

추가적인 한계:
- **직사각형 편집 영역으로 훈련**: 비록 직사각형 편집 영역으로만 훈련되었음에도 불구하고, 손으로 그린 불규칙한 편집 영역에서도 안정적인 편집 능력을 유지함이 확인되었다 — 이는 긍정적 일반화 능력이지만, 동시에 훈련-추론 간 분포 차이를 내포한다.
- **단일 프레임 의존성**: 첫 번째 프레임만을 콘텐츠 기준으로 사용하는 구조적 제약.

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 현재의 일반화 성능 근거

ReVideo는 특정 훈련 없이도 멀티-영역 편집으로 이러한 애플리케이션을 원활하게 확장할 수 있으며, 이는 유연성과 견고성을 입증한다.

비록 직사각형 편집 영역으로만 훈련되었지만, 손으로 그린 불규칙한 편집 영역에 대해서도 안정적인 콘텐츠 및 모션 편집 능력을 보인다.

### 3.2 일반화 성능 향상을 위한 핵심 방향

아래는 논문에서 시사하거나 연구자들이 고려할 수 있는 일반화 향상 가능성이다:

#### (A) 더 강력한 기반 모델로의 확장
더 강력한 기반 모델은 장 비디오 편집에서의 오류 누적 문제를 완화할 수 있다. 현재 SVD(Stable Video Diffusion) 기반에서, DiT 기반의 대형 모델(예: CogVideoX, Wan 2.1)로 SAFM 구조를 이식하면 더 높은 일반화 성능을 기대할 수 있다.

#### (B) 임의 프레임 사양(Arbitrary Frame Specification)으로의 확장
ReVideo처럼 첫 번째 프레임에만 의존하는 방식은 편집 가능한 범위를 심각하게 제한하며, 이후에 등장하는 요소들을 처리하지 못한다. 이를 개선하면 영상 중반부에 등장하는 객체에 대한 편집 일반화가 가능해진다.

#### (C) 카메라 모션 인식 일반화
ReVideo와 같은 I2V 기반 방법은 지정된 모션과 함께 단일 이미지로부터 새로운 비디오를 생성할 수 있지만, 상당한 단점이 존재한다. 카메라 움직임(panoramic shot, zoom 등)과 객체 모션을 분리하는 메커니즘 추가가 필요하다.

#### (D) 훈련 없는(Training-Free) 방향
해석 가능한 모션 인식 특징(motion-aware feature)을 사용하여 비디오 모션을 제어하는 새로운 파이프라인은 훈련이 필요 없으며, 다양한 아키텍처 프레임워크에 일반화될 수 있다. 이러한 training-free 접근과 ReVideo의 구조를 결합하면 다양한 도메인에 대한 일반화가 용이해진다.

---

## 4. 관련 최신 연구 비교 분석 (2020년 이후)

| 방법 | 연도 | 발표 | 편집 방식 | 모션 제어 | 콘텐츠 제어 | 주요 한계 |
|---|---|---|---|---|---|---|
| **InsV2V** | 2023 | - | 텍스트 명령 | ❌ | ✅ | 비편집 영역 왜곡 |
| **DragNUWA** | 2023 | arXiv | I2V + 궤적 | ✅ (픽셀 수준) | ✅ | 객체 전체가 아닌 픽셀 단위 모션 |
| **AnyV2V** | 2024 | - | 첫 프레임 기반 | ❌ | ✅ | 모션 제어 부재, 콘텐츠 손실 |
| **DragAnything** | 2024 | ECCV | I2V + 엔티티 표현 | ✅ (엔티티 수준) | ✅ | 비디오 편집보다 생성에 집중 |
| **ReVideo** | 2024 | NeurIPS | 첫 프레임 + 궤적 | ✅ | ✅ | 첫 프레임 의존, 카메라 모션 취약 |
| **MotionV2V** | 2025 | arXiv | 전체 비디오 기반 | ✅ (임의 프레임) | ✅ | - |

일부 세밀한 편집 시나리오(예: 남성에게 선글라스 씌우기)에서 AnyV2V는 편집된 콘텐츠의 손실이 발생하며, InsV2V와 AnyV2V의 비편집 영역에서는 콘텐츠 왜곡이 나타난다. Pika는 부드럽고 고충실도의 결과를 생성하지만, 텍스트로 새 콘텐츠를 정확하게 커스터마이즈하기 어렵다.

모션 제어 부재로 인해 AnyV2V와 Pika는 보통 도로를 달리는 자동차처럼 편집된 콘텐츠의 정적인 모션을 생성한다. 반면 ReVideo는 편집된 콘텐츠를 전체 비디오에 효과적으로 전파하면서 사용자가 편집 영역의 모션을 커스터마이즈할 수 있게 해준다.

**후속 연구와의 비교:**

MotionV2V와 같이, ReVideo처럼 첫 번째 프레임을 콘텐츠 생성에 사용하는 I2V 방식에 의존하는 것은 가능한 편집을 심각하게 제한하고 이후에 등장하는 요소들을 처리하지 못한다.

---

## 5. 앞으로의 연구에 미치는 영향 및 고려사항

### 5.1 연구에 미치는 영향

**① 새로운 태스크 패러다임의 정립**

이 연구는 콘텐츠와 모션 모두의 동시 지정 및 제어를 통해 특정 영역에서의 정밀한 비디오 편집을 가능하게 하는 방향을 설정한다. 이로 인해 단순한 텍스트 기반 편집에서 벗어나 다중 모달 제어(multi-modal control) 기반의 비디오 편집 연구가 활성화될 것으로 예상된다.

**② ControlNet 패러다임의 비디오 도메인 확장**

ReVideo는 ControlNet 적응 구조가 다양한 조건화 유형에 고도로 적응 가능하며, 훈련이 용이하고 편집·창작 시 더 정밀한 조작과 향상된 정확도를 허용함을 보여주었다.

**③ 훈련 불균형 문제의 인식**

콘텐츠와 모션 제어 사이의 결합 및 훈련 불균형은 효과적인 조인트 비디오 편집에서의 도전으로 처음 명시적으로 규명되었다. 이 인식은 향후 다중 조건 생성 모델 설계에 중요한 기반이 된다.

**④ 인간 중심 평가의 중요성 부각**

AnyV2V와 Pika와 같은 기존 방법들이 명시적인 모션 제어 부재로 인해 정적인 모션을 생성하며, 이것이 인접 프레임의 CLIP 유사도로 측정되는 일관성 점수를 인위적으로 부풀릴 수 있다는 점이 분석을 통해 밝혀졌다. 이는 자동 메트릭의 한계를 인식하고 인간 평가를 병행해야 한다는 연구 방향성을 제시한다.

---

### 5.2 앞으로 연구 시 고려할 점

#### (1) 첫 프레임 의존성 탈피
ReVideo와 같은 첫 번째 프레임 보존 방법은 인페인팅을 통해 문제를 해결하려 하지만, 카메라 모션이 초기 프레임에 없는 콘텐츠를 드러낼 때 성능이 저하된다. 따라서 **임의 키프레임(arbitrary keyframe) 기반 편집** 또는 **전체 비디오 컨텍스트 활용** 방식의 연구가 필요하다.

#### (2) 더 강력한 기반 모델과의 통합
최근 연구들은 모션 궤적, 객체 마스크, DiT 프레임워크 내 궤적 조건화 등의 명시적 제어 입력을 활용한다. ReVideo의 SAFM 모듈을 DiT 기반 최신 대형 비디오 생성 모델에 이식하는 연구가 중요한 방향이 될 것이다.

#### (3) 카메라 모션과 객체 모션의 분리
DragNUWA는 카메라와 객체 모션 모두를 모델링하기 위해 궤적을 조건화하고, DragAnything은 엔티티 수준의 제어를 위해 객체 마스크를 활용한다. ReVideo는 이 두 모션을 분리하여 제어하는 기능이 제한적이므로, 카메라 모션 인식 모듈의 통합이 필요하다.

#### (4) 평가 지표의 고도화
현재 자동 메트릭은 텍스트 정렬을 위한 CLIP 점수와 시간적 일관성을 위한 프레임 간 평균 CLIP 코사인 유사도를 사용한다. 동적인 모션 편집 품질을 더 정확하게 측정하는 전문화된 모션 정확도 메트릭 개발이 요구된다.

#### (5) 장 비디오(Long Video) 처리
ReVideo는 첫 번째 프레임의 편집을 90프레임 비디오에 전파할 수 있지만, 오류 누적이 편집 품질에 영향을 미친다. 따라서 청크(chunk) 기반 일관성 유지, 메모리 효율적 아키텍처, 그리고 자동회귀(autoregressive) 방식의 도입이 고려되어야 한다.

#### (6) Training-Free 접근과의 결합
기존의 훈련 기반 패러다임은 모션 조건을 훈련에 통합하고 추가 모듈을 필요로 하며, 상당한 훈련 리소스를 요구하고 다른 모델에 대한 재훈련이 필요하다. 훈련 비용을 줄이면서도 일반화 성능을 높이는 하이브리드 방식 연구가 필요하다.

---

## 참고 자료

1. **[주 논문]** Chong Mou, Mingdeng Cao, Xintao Wang, Zhaoyang Zhang, Ying Shan, Jian Zhang. *"ReVideo: Remake a Video with Motion and Content Control."* NeurIPS 2024. arXiv:2405.13865. https://arxiv.org/abs/2405.13865

2. **[NeurIPS 공식]** NeurIPS 2024 Poster: https://neurips.cc/virtual/2024/poster/93082

3. **[논문 전문 HTML]** arXiv HTML (v1): https://arxiv.org/html/2405.13865v1

4. **[NeurIPS 논문집 PDF]** https://proceedings.neurips.cc/paper_files/paper/2024/file/20e6b4dd2b1f82bc599c593882f67f75-Paper-Conference.pdf

5. **[OpenReview]** https://openreview.net/forum?id=xUjBZR6b1T

6. **[HuggingFace Papers]** https://huggingface.co/papers/2405.13865

7. **[비교 논문]** Wu et al. *"DragAnything: Motion Control for Anything using Entity Representation."* ECCV 2024. arXiv:2403.07420.

8. **[비교 논문]** Ku et al. *"AnyV2V: A Tuning-Free Framework For Any Video-to-Video Editing Tasks."* arXiv:2403.14468.

9. **[비교 논문]** Yin et al. *"DragNUWA: Fine-grained Control in Video Generation by Integrating Text, Image, and Trajectory."* arXiv:2308.08089.

10. **[후속 연구]** *"MotionV2V: Editing Motion in a Video."* arXiv:2511.20640. https://arxiv.org/html/2511.20640

11. **[후속 연구]** *"Ctrl-V: High Fidelity Video Generation with Bounding-Box Controlled Object Motion."* arXiv:2406.05630. https://arxiv.org/html/2406.05630v3

12. **[서베이 논문]** *"Video diffusion generation: comprehensive review and open problems."* Artificial Intelligence Review, Springer Nature, 2025. https://link.springer.com/article/10.1007/s10462-025-11331-6

13. **[리뷰]** Liner AI Quick Review: https://liner.com/review/revideo-remake-a-video-with-motion-and-content-control

14. **[ResearchGate]** https://www.researchgate.net/publication/397203662_ReVideo_Remake_a_Video_with_Motion_and_Content_Control
