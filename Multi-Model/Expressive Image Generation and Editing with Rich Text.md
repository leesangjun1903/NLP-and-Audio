# Expressive Image Generation and Editing with Rich Text

(Songwei Ge, Taesung Park, Jun-Yan Zhu, Jia-Bin Huang; ICCV 2023 / IJCV 2025 확장판)

---

# 1. 핵심 주장과 주요 기여 (요약)

기존 텍스트-이미지 합성은 평문(plain text) 인터페이스에 의존하는데, 이는 사용자 커스터마이징 옵션이 제한적이어서 원하는 출력을 정확히 기술하기 어렵다. 예를 들어, 평문으로는 정확한 RGB 색상값이나 각 단어의 중요도 같은 연속적 수량을 지정하기 힘들며, 복잡한 장면에 대한 상세한 프롬프트를 작성하는 것도 사람에게 번거롭고 텍스트 인코더가 해석하기도 어렵다.

**핵심 주장:** 리치 텍스트(rich text) 에디터가 지원하는 폰트 스타일, 크기, 색상, 텍스처 채우기, 각주, 임베디드 이미지 등의 포맷 정보를 활용하여, 로컬 스타일 제어, 명시적 토큰 가중치 재설정, 정밀한 색상 렌더링, 참조 개념/텍스처를 활용한 상세 영역 합성을 가능하게 한다.

**주요 기여:**
1. 리치 텍스트 기반 이미지 생성 과제(task)를 평가하기 위한 벤치마크 구축 — 복잡한 프롬프트를 활용한 정량적 평가 포함
2. 임베디드 이미지를 활용한 개념 가이딩 및 텍스처 채우기(texture fill)를 활용한 텍스처 렌더링 가이딩이라는 두 가지 새로운 애플리케이션 개발
3. 폰트 스타일, 색상 등의 단어 속성을 추출하여 region-based diffusion process를 통해 정밀 제어를 수행하며, 기존 방법 대비 정확한 색상, 구별되는 스타일, 정교한 디테일 생성에서 우수한 성능 달성

---

# 2. 상세 분석

## 2.1 해결하고자 하는 문제

평문은 텍스트 기반 이미지 합성·편집에서 보편적인 인터페이스가 되었으나, 제한적인 커스터마이징 옵션으로 사용자가 원하는 출력을 정확히 기술하기 어렵다. 정확한 RGB 색상값이나 단어의 중요도 같은 연속적 수량을 지정하기 힘들고, 복잡한 장면을 위한 상세 프롬프트 작성이 번거로우며, 참조 개념이나 텍스처를 평문으로 기술하는 것이 비자명(non-trivial)하다.

구체적으로 다음 4가지 제어 불가 문제를 해결하고자 한다:
- **(1) 로컬 스타일 제어:** 이미지 내 서로 다른 영역에 서로 다른 예술 스타일 적용
- **(2) 토큰 가중치 재설정:** 특정 단어/객체의 중요도를 연속적으로 제어
- **(3) 정밀한 색상 렌더링:** RGB 값 기반의 정확한 색상 지정
- **(4) 상세 영역 합성:** 각주(footnote)를 통한 개별 객체에 대한 보충 설명

## 2.2 제안하는 방법

### 전체 파이프라인

평문 프롬프트를 먼저 디퓨전 모델에 입력하여 self-attention과 cross-attention 맵을 수집한다. Attention 맵은 서로 다른 head, layer, time step에 걸쳐 평균된다. 이후 self-attention 맵을 이용해 spectral clustering으로 세그멘테이션을 생성하고, cross-attention으로 각 세그먼트에 레이블을 부여한다.

리치 텍스트 프롬프트는 에디터에서 JSON 형식으로 저장되어 각 토큰 스팬에 속성을 제공한다. 각 토큰의 속성에 따라 토큰 맵이 가리키는 영역에 denoising prompt 또는 guidance 형태의 제어가 적용되며, 평문 생성 결과로부터 구조와 배경을 보존하기 위해 feature injection 또는 noised sample blending이 수행된다.

### 주요 수식

#### (a) Diffusion 기본 프레임워크

Latent Diffusion Model(LDM)의 기본 목적함수:

$$\mathcal{L}_{LDM} = \mathbb{E}_{z, \epsilon \sim \mathcal{N}(0,1), t}\left[\|\epsilon - \epsilon_\theta(z_t, t, c)\|_2^2\right]$$

여기서 $z_t$는 시간 $t$에서의 노이즈가 추가된 잠재 표현, $c$는 텍스트 조건, $\epsilon_\theta$는 노이즈 예측 네트워크이다.

#### (b) Token Reweighting (폰트 크기 기반)

폰트 크기는 최종 생성에서 각 토큰의 가중치를 나타내며, 이는 각 cross-attention 레이어에서 softmax 이전에 지수적 attention 점수를 재가중(reweighting)하는 방식으로 구현된다.

토큰 $w_j$의 폰트 크기 속성 $a_w$에 따른 cross-attention reweighting:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}} + \log \alpha\right)V$$

여기서 $\alpha$는 폰트 크기로부터 유도된 토큰별 스케일링 팩터이다. 기존 Prompt-to-Prompt 방식의 직접 곱셈과 달리, Prompt-to-Prompt가 attention 확률을 무제한으로 수정하여 분포 외(out-of-distribution) 중간 특징을 생성하고 명확한 아티팩트를 발생시키는 반면, 본 방법은 더 많은 객체를 생성하면서도 품질을 보존한다.

#### (c) Color Guidance (폰트 색상 기반)

사용자가 지정한 RGB 색상 $c_{target}$에 대해, 해당 영역의 디코딩된 이미지가 목표 색상에 가까워지도록 gradient guidance를 적용한다:

$$\nabla_{z_t} \mathcal{L}_{color} = \nabla_{z_t} \| \mathcal{D}(\hat{z}_0) \odot M_k - c_{target} \odot M_k \|_2^2$$

여기서 $\mathcal{D}$는 VAE 디코더, $\hat{z}_0$는 예측된 clean latent, $M_k$는 토큰 $k$에 대응하는 영역 마스크이다. color guidance 가중치 $\lambda$를 조정하면 충실도(fidelity)와 색상 정밀도 사이의 트레이드오프를 제어할 수 있으며, 이미지 충실도는 생성 결과와 평문 프롬프트 사이의 CLIP 스코어로 평가한다.

전체 denoising update:

$$z_{t-1} = z_{t-1}^{plain} - \lambda \nabla_{z_t}\mathcal{L}_{color}$$

#### (d) Region-Based Diffusion (스타일/각주 기반)

각 영역 $R_k$에 대해 개별 프롬프트 $p_k$로 denoising을 수행하고, 영역 밖은 plain-text 생성으로부터 injection:

$$\hat{\epsilon}_t = \sum_{k} M_k \odot \epsilon_\theta(z_t, t, p_k) + (1 - \sum_k M_k) \odot \epsilon_\theta(z_t, t, p_{plain})$$

여기서 $M_k$는 토큰 $k$의 영역 마스크, $p_k$는 스타일이나 각주가 반영된 영역별 프롬프트이다.

## 2.3 모델 구조

본 방법은 **학습 불필요(training-free)** 프레임워크로, 기존 사전학습된 Latent Diffusion Model (Stable Diffusion) 위에서 동작한다.

실험에는 Stable Diffusion V1-5를 사용하며, 토큰 맵 생성 시 첫 번째 인코더와 마지막 디코더 블록을 제외한 모든 블록의 cross-attention 레이어를 사용하고 (고해상도 레이어의 attention 맵이 종종 노이지하기 때문), 초기 denoising 스텝($T > 750$)의 맵은 버린다.

이후 SD-XL과 ANIMAGINE-XL 등 다양한 fine-tuned 모델로도 확장 지원이 이루어졌다.

구조적 핵심:
- **Stage 1 (Token Map 추출):** Plain text → Diffusion forward → Self-attention + Cross-attention → Spectral Clustering → 영역 마스크 $M_k$
- **Stage 2 (Region-Based Denoising):** 각 영역에 해당하는 rich-text 속성 적용 (스타일 프롬프트, 색상 가이던스, 토큰 가중치) → Region-based injection으로 전체 이미지 일관성 유지

## 2.4 성능 향상

정량적으로 Prompt-to-Prompt와 InstructPix2Pix라는 두 가지 강력한 베이스라인과 비교하였다.

- **색상 정확도:** 색상 정확도는 영역과 목표 RGB 값 사이의 평균 L2 거리를 계산하여 평가한다. 세 가지 난이도 수준(기본 색상명, 복잡한 색상명, 임의 RGB 삼중값)에서 모두 본 방법이 우수한 성능을 보임.
- **스타일 제어:** 각 스타일화된 영역과 해당 영역 프롬프트 사이의 CLIP 유사도를 보고하며, 본 방법이 최고의 스타일화 성능을 달성한다.
- **전체 품질:** 두 가지 베이스라인 방법 모두 RGB 값을 올바르게 해석하지 못한다.

## 2.5 한계

목표 색상이 배경으로 번지는(bleed) 현상이 발생할 수 있으며, 임계값(thresholding)이나 침식(erosion) 같은 전처리 단계로 완화할 수 있지만 사례별 조정이 필요한 경우가 많다. 또한 다중 디퓨전 프로세스와 2단계 방법을 사용하므로 계산 비용이 높을 수 있다.

추가적인 한계:
- Cross-attention 기반 영역 마스크의 정밀도에 의존하므로, 토큰 간 의미 중첩이 심한 경우 영역 분리가 부정확할 수 있음
- 사전학습된 디퓨전 모델의 능력에 한정됨 (새로운 개념이나 도메인에 대한 일반화 제한)

---

# 3. 모델의 일반화 성능 향상 가능성

## 3.1 현재의 일반화 능력

본 방법의 핵심 강점은 **training-free** 접근이라는 점이다:

- Stable Diffusion v1-5, Stable Diffusion XL, ANIMAGINE-XL 등 다양한 모델을 지원한다. 이는 사전학습된 디퓨전 모델의 cross-attention 메커니즘이라는 공통 구조를 활용하기 때문에, 동일한 아키텍처를 가진 모든 모델에 즉시 적용 가능하다.
- LoRA 체크포인트도 지원하여, 커뮤니티에서 fine-tune된 다양한 도메인 특화 모델에도 적용 가능하다.

## 3.2 일반화 향상 가능 방향

| 방향 | 설명 |
|------|------|
| **디퓨전 아키텍처 독립성** | 현재 U-Net 기반 cross-attention에 의존하나, DiT(Diffusion Transformer) 등 새 아키텍처로 확장 필요 |
| **토큰 맵 정밀도 향상** | SAM(Segment Anything Model) 등 외부 세그멘테이션 모델과 결합하여 영역 마스크 품질 향상 |
| **다중 모달리티 확장** | 임베디드 이미지 + 텍스처 채우기 외에 오디오, 3D 등 다른 모달리티로의 확장 |
| **대규모 언어 모델(LLM) 연동** | LLM이 자동으로 rich-text 포맷을 생성하여 사용자 의도를 더 정확히 반영 |
| **Adaptive Guidance Scheduling** | $\lambda$ 값을 시간 스텝별로 자동 조정하여 색상 정밀도와 이미지 품질의 트레이드오프 최적화 |

---

# 4. 향후 연구에 미치는 영향과 고려 사항

## 4.1 연구적 영향

1. **새로운 인터페이스 패러다임 제시:** "프롬프트 엔지니어링" 의존도를 줄이고, 워드프로세서와 같은 직관적 UI로 이미지 생성을 제어하는 새로운 패러다임을 열었다. 본 방법은 region-constrained diffusion process를 통한 정밀 리파인먼트를 지원하며, 리치 텍스트 에디터와 자연스럽게 통합 가능한 간결한 UI를 제공한다.

2. **Training-Free 제어의 가능성 확대:** 추가 학습 없이 사전학습 모델의 내부 표현(attention map)만으로 세밀한 제어가 가능함을 증명, 이후 연구에서 inference-time 제어 방법론의 중요성을 부각시켰다.

3. **영역 기반 합성의 발전:** Region-based diffusion 개념은 이후 MultiDiffusion, SpaText, BlobGEN 등 후속 연구에 영향을 미쳤다.

## 4.2 앞으로 연구 시 고려할 점

- **실시간 생성:** 다중 디퓨전 프로세스의 계산 비용 문제를 해결하기 위한 효율적 추론 전략 필요
- **3D/비디오 확장:** 리치 텍스트의 포맷 정보를 3D 장면이나 비디오 생성으로 확장하는 연구
- **사용자 연구(User Study) 강화:** 실제 디자이너/아티스트와의 상호작용 연구를 통한 UI/UX 최적화
- **안전성/윤리:** 정밀한 제어가 가능해질수록 딥페이크 등 악용 가능성에 대한 방어 메커니즘 연구 필요
- **Attention 맵 의존성 탈피:** Cross-attention 맵의 노이즈와 부정확성을 극복하기 위한 대안적 영역 추출 방법 연구

---

# 5. 2020년 이후 관련 최신 연구 비교 분석

| 연구 | 연도 | 핵심 접근 | Rich Text와의 차별점 |
|------|------|-----------|---------------------|
| **DDPM** (Ho et al.) | 2020 | 기초 디퓨전 확률 모델 | Rich Text의 이론적 기반이 되는 denoising framework |
| **LDM / Stable Diffusion** (Rombach et al.) | 2022 | 잠재 공간 디퓨전 | Rich Text가 기반으로 사용하는 backbone 모델 |
| **Prompt-to-Prompt** (Hertz et al.) | 2023 | Cross-attention 제어 기반 편집 | attention 확률을 무제한 수정하여 아티팩트 발생; Rich Text는 softmax 이전 log-space reweighting으로 안정적 |
| **InstructPix2Pix** (Brooks et al.) | 2023 | 지시문 기반 이미지 편집 | 2번째 영역에서 최악의 성능; 첫 번째 지시문이 두 번째 영역 정보를 포함하지 않기 때문 |
| **Attend-and-Excite** (Chefer et al.) | 2023 | Attention 기반 Generative Semantic Nursing(GSN) 메커니즘으로 cross-attention을 정제하여 모든 주체가 정확히 생성되도록 보장 | 전체 이미지 수준 제어; Rich Text는 영역별 세밀한 속성 제어 |
| **ControlNet** (Zhang et al.) | 2023 | 텍스트-이미지 모델에 추가 조건을 삽입하는 신경망으로, 에지 맵, 포즈 등 공간적 조건 제어 | 구조적(spatial) 제어에 특화; Rich Text는 의미적(semantic) 속성 제어에 특화 |
| **GLIGEN** (Li et al.) | 2023 | U-Net 인코더의 self-attention과 cross-attention 사이에 gated self-attention 레이어 추가 | 바운딩 박스 기반 레이아웃 제어; Rich Text는 텍스트 포맷 기반 제어 |
| **IP-Adapter** (Ye et al.) | 2023 | 분리된(decoupled) cross-attention 메커니즘으로 이미지 프롬프트를 추가하며, CLIP 기반 이미지 인코더와 병렬 cross-attention 모듈 활용 | 이미지 프롬프트 기반; Rich Text는 텍스트 포맷 메타데이터 기반 |
| **TextDiffuser** (Chen et al.) | 2023 | 시각적으로 매력적이며 배경과 일관된 텍스트가 포함된 이미지 생성에 초점; Transformer → 레이아웃 → 디퓨전의 2단계 구조 | 이미지 내 "텍스트 렌더링" 문제 해결; Rich Text는 이미지 "속성 제어" 문제 해결 |
| **SDXL** (Podell et al.) | 2024 | 고해상도 잠재 디퓨전 모델 개선 | Rich Text가 SDXL backbone으로도 확장 적용 가능함을 보임 |
| **ARTIST** (2024) | 2024 | 분리된(disentangled) 아키텍처 설계와 학습 전략으로 텍스트 렌더링 능력 향상; 사전학습된 LLM으로 사용자 의도 해석 | 텍스트 구조 학습에 특화된 별도 디퓨전 모델 사용; Rich Text는 단일 모델 + 추론시 제어 |
| **DesignDiffusion** (2025) | 2025 | 프롬프트 향상 및 character localization loss 도입과 Self-Play DPO를 통한 텍스트 렌더링 정확도 향상 | 디자인 이미지에 특화; Rich Text는 범용 이미지 생성/편집에 집중 |
| **ControlNet++** (Li et al.) | 2024 | 효율적 일관성 피드백을 통한 조건 제어 개선 | 구조적 조건 강화; Rich Text의 의미 제어와 상호보완적 |

---

# 참고 자료

1. **Ge, S., Park, T., Zhu, J.-Y., & Huang, J.-B.** (2023). "Expressive Text-to-Image Generation with Rich Text." *ICCV 2023*. — arXiv: [2304.06720](https://arxiv.org/abs/2304.06720)
2. **Ge, S., Park, T., Zhu, J.-Y., & Huang, J.-B.** (2025). "Expressive Image Generation and Editing with Rich Text." *International Journal of Computer Vision*, 133, 4604–4622. — [Springer](https://link.springer.com/article/10.1007/s11263-025-02361-2)
3. **프로젝트 페이지:** [https://rich-text-to-image.github.io/](https://rich-text-to-image.github.io/)
4. **GitHub 코드:** [https://github.com/SongweiGe/rich-text-to-image](https://github.com/SongweiGe/rich-text-to-image)
5. **Hertz, A. et al.** (2023). "Prompt-to-Prompt Image Editing with Cross-Attention Control." *ICLR 2023*.
6. **Brooks, T. et al.** (2023). "InstructPix2Pix: Learning to Follow Image Editing Instructions." *CVPR 2023*.
7. **Zhang, L. et al.** (2023). "Adding Conditional Control to Text-to-Image Diffusion Models." (ControlNet) *ICCV 2023*.
8. **Li, Y. et al.** (2023). "GLIGEN: Open-Set Grounded Text-to-Image Generation." *CVPR 2023*.
9. **Ye, H. et al.** (2023). "IP-Adapter: Text Compatible Image Prompt Adapter for Text-to-Image Diffusion Models." arXiv: 2308.06721.
10. **Chefer, H. et al.** (2023). "Attend-and-Excite: Attention-Based Semantic Guidance for Text-to-Image Diffusion Models." arXiv: 2301.13826.
11. **Chen, J. et al.** (2023). "TextDiffuser: Diffusion Models as Text Painters." arXiv: 2305.10855.
12. **Zheng, K. et al.** (2024). "ARTIST: Improving the Generation of Text-rich Images with Disentangled Diffusion Models." arXiv: 2406.12044.
13. **DesignDiffusion** (2025). "High-Quality Text-to-Design Image Generation with Diffusion Models." arXiv: 2503.01645.
14. **Controllable Generation with T2I Diffusion Models: A Survey.** arXiv: 2403.04279.

> **참고:** 수식의 일부 세부사항(특히 token reweighting의 정확한 log-space 스케일링 계수 유도, region injection 블렌딩 비율 등)은 논문 원문의 수식 표기를 기반으로 재구성한 것이며, 정확한 변수 명명은 원문 PDF를 직접 참조하시길 권장합니다.
