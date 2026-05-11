
# DreamText: High Fidelity Scene Text Synthesis 

> **논문 정보**
> - **저자**: Yibin Wang, Weizhong Zhang, Honghui Xu, Cheng Jin
> - **발표**: CVPR 2025 (pp. 28555–28563)
> - **arXiv**: [2405.14701](https://arxiv.org/abs/2405.14701)
> - **공식 코드**: [GitHub - CodeGoat24/DreamText](https://github.com/CodeGoat24/DreamText)

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

Scene text synthesis는 임의의 이미지에 지정된 텍스트를 렌더링하는 작업이다. 기존 방법들은 이 태스크를 end-to-end 방식으로 정형화하지만, 학습 시 효과적인 **문자 수준(character-level) 가이던스**가 부족하다. 또한 단일 폰트 유형으로 사전 학습된 텍스트 인코더는 실제 응용에서 마주치는 다양한 폰트 스타일에 적응하지 못한다. 결과적으로 이 방법들은 특히 다양한 스타일(polystylistic) 시나리오에서 **문자 왜곡(distortion), 반복(repetition), 누락(absence)** 문제를 겪는다.

### 주요 기여

논문의 주요 기여는 다음과 같다:
1. DreamText는 기존 방법이 겪는 문자 반복, 누락, 왜곡 문제를 효과적으로 완화한다.
2. **휴리스틱 교대 최적화(heuristic alternate optimization) 전략**은 텍스트 인코더와 U-Net의 공동 학습을 통합하여 문자 표현 학습과 문자 어텐션 재추정 간의 공생 관계를 형성한다.
3. **균형 잡힌 감독(balanced supervision) 전략**은 모델을 제약하는 것과 최적 생성 위치 추정의 유연성을 발휘하는 것 사이의 균형을 이룬다.
4. 정성적·정량적 결과 모두에서 최신 방법 대비 우월성을 입증한다.

---

## 2. 해결하고자 하는 문제, 제안 방법, 모델 구조, 성능 향상 및 한계

### 2.1 해결하고자 하는 문제

현재 방법들은 제한된 표현 영역(constrained representation domain)으로 인해 복잡한 장면에서 텍스트를 정확하게 렌더링하는 데 어려움을 겪으며, 모델이 최적의 문자 생성 위치를 추정할 효과적인 가이던스가 부족하여 이상적인 영역에 문자를 합성하는 데도 한계가 있다.

구체적으로 두 가지 핵심 병목이 있다:

- **문제 1 — 문자 수준 어텐션 편향(deflected attention)**: 문자들의 어텐션이 이상적인 생성 영역에 정확하게 집중되지 않으며, 이를 교정하기 위한 추가 제약이 없다.
- **문제 2 — 제한된 폰트 표현 도메인**: 텍스트 인코더가 단일 폰트 유형으로 사전 학습되어 있어 실제 응용에서 만나는 다양한 폰트 스타일에 적응하지 못한다.

---

### 2.2 제안하는 방법 (수식 포함)

#### (A) Diffusion 기반 훈련 재구성

DreamText는 기반 모델로 **Stable Diffusion (v2.0) inpainting** 버전을 사용한다. 표준 Denoising Diffusion Probabilistic Model (DDPM)의 역방향 과정 학습 목표는:

$$\mathcal{L}_{\text{simple}} = \mathbb{E}_{x_0, \epsilon, t}\left[\|\epsilon - \epsilon_\theta(x_t, t, c)\|^2\right]$$

여기서 $x_t$는 시간 $t$에서 노이즈가 추가된 잠재 변수, $\epsilon \sim \mathcal{N}(0, I)$는 노이즈, $c$는 조건(텍스트 임베딩 등), $\epsilon_\theta$는 U-Net 기반 노이즈 예측기이다.

DreamText는 이 표준 학습 과정에 **문자 수준 어텐션 감독**과 **텍스트 영역 강화 손실**을 추가하여 다음과 같이 훈련 목표를 재구성한다:

$$\mathcal{L}_{\text{DreamText}} = \mathcal{L}_{\text{simple}} + \lambda_1 \mathcal{L}_{\text{attn}} + \lambda_2 \mathcal{L}_{\text{region}}$$

> ⚠️ **주의**: $\mathcal{L}\_{\text{attn}}$, $\mathcal{L}_{\text{region}}$의 구체적인 형태 및 $\lambda$ 값은 논문 본문의 세부 기술에 의존하며, 위 형식은 논문의 "additional constraints" 설명에 기반한 구조적 표현임.

#### (B) 하이브리드 최적화 문제 및 교대 최적화 전략

이 변환은 **이산 변수(discrete variables)**와 **연속 변수(continuous variables)** 모두를 포함하는 하이브리드 최적화 문제를 야기한다. 이 문제를 효과적으로 해결하기 위해 **휴리스틱 교대 최적화(heuristic alternate optimization) 전략**을 사용한다.

- 이산 변수: 각 문자에 대한 잠재적 생성 위치(character mask)
- 연속 변수: U-Net 가중치, 텍스트 인코더 파라미터

구체적으로, 각 스텝에서 먼저 크로스-어텐션 맵(cross-attention maps)으로부터 잠재적 문자 생성 위치 정보를 인코딩하여 **잠재 문자 마스크(latent character masks)**를 생성한다. 이 마스크들은 현재 스텝에서 특정 문자의 표현을 업데이트하는 데 사용되며, 이는 다시 생성기(generator)가 다음 스텝에서 문자의 어텐션을 교정할 수 있게 한다.

교대 최적화의 각 반복 구조를 수식으로 나타내면 (개념적 표현):

**Step 1 — Mask 추정 (이산 업데이트)**:

$$M_i^{(k)} = f_{\text{mask}}\left(A_i^{(k)}\right)$$

여기서 $A_i^{(k)}$는 $k$번째 학습 스텝에서 문자 $i$의 크로스-어텐션 맵, $M_i^{(k)}$는 추정된 문자 마스크이다.

**Step 2 — 표현 업데이트 (연속 업데이트)**:

$$e_i^{(k+1)} = g_\phi\left(e_i^{(k)}, M_i^{(k)}\right)$$

여기서 $e_i^{(k)}$는 문자 $i$의 임베딩 표현, $g_\phi$는 텍스트 인코더 업데이트 함수이다.

> ⚠️ 위 Step 1~2의 수식은 논문 내 "encode potential character-generated position information from cross-attention maps into latent character masks" 및 "update the representation of specific characters" 설명을 기반으로 한 **개념적 수식**이며, 논문의 정확한 표기와 다를 수 있다.

#### (C) 텍스트 인코더 및 U-Net 공동 학습

텍스트 인코더와 생성기를 공동으로 학습하여 훈련 데이터셋에 존재하는 다양한 폰트를 포괄적으로 학습·활용하며, 이 공동 학습은 교대 최적화 과정에 자연스럽게 통합되어 문자 임베딩 학습과 문자 어텐션 재추정 사이의 시너지 관계를 촉진한다.

---

### 2.3 모델 구조

DreamText의 **휴리스틱 교대 최적화 전략**은 텍스트 인코더와 U-Net의 공동 학습을 통합하여 문자 표현 학습과 문자 어텐션 재추정 간의 공생 관계를 구축한다.

모델 구조의 주요 구성 요소:

| 구성 요소 | 설명 |
|---|---|
| **Base Model** | Stable Diffusion v2.0 (inpainting 버전) |
| **텍스트 인코더** | 단일 폰트 고정 인코더 → 다양한 폰트 학습을 위해 공동 학습(fine-tuned) |
| **U-Net 백본** | 크로스-어텐션 레이어를 통해 문자 수준 어텐션 맵 추출 |
| **Character Mask Module** | 크로스-어텐션 맵 → 잠재 문자 마스크 변환 |
| **Balanced Supervision** | 텍스트 영역 강화 + 유연한 위치 추정의 균형 |
| **훈련 데이터셋** | LAION-OCR, ICDAR13 |

비교 평가에서는 GAN 기반 및 diffusion 기반 방법을 모두 포함한 여러 최신 베이스라인을 평가하였으며, 구체적으로 MOSTEL, Stable Diffusion-inpainting (v2.0), DiffSTE, TextDiffuser, AnyText, UDiffText와 비교하였다.

---

### 2.4 성능 향상

DreamText는 모든 정량적 지표에서 상당한 우위를 보이며 생성된 장면 텍스트 이미지의 뛰어난 시각적 품질을 입증한다. 특히 이전 최신 모델인 UDiffText와 비교했을 때, DreamText는 **FID(Fréchet Inception Distance) 기준으로 3.66 향상**을 달성하였다.

UDiffText와 AnyText는 DreamText를 제외한 다른 방법들에 비해 이미지 품질은 우수하지만, 특히 텍스트 편집에서 시퀀스 정확도(sequence accuracy) 측면에서는 덜 만족스러운 성능을 보이며, 이는 제한된 표현 도메인과 편향된 어텐션에 기인하여 복잡한 장면에서의 텍스트 렌더링 정확도가 낮아지는 것으로 분석된다.

정성적으로도 DreamText는 다양한 스타일의 장면에서 더 일관적이고 정확한 텍스트를 합성하는 데 있어 다른 베이스라인들을 능가한다.

또한 LAION-OCR 및 SynthText 데이터셋에서의 전체 학습 스텝에 걸친 mIoU 점수 비교에서도 UDiffText, TextDiffuser 대비 우월한 수렴 성능을 보였다.

---

### 2.5 한계

공개된 자료를 바탕으로 다음과 같은 한계가 확인된다:

1. **영어 및 단일 라인(single-line) 제한**: TextDiffuser, UdiffText, DreamText와 같은 최근 방법들은 두 가지 핵심 측면에서 제한적인데, 단일 라인 텍스트 생성만 지원하며 영어에만 국한된다.

2. **하이브리드 최적화 복잡성**: 이 재구성은 불가피하게 이산 변수와 연속 변수 모두를 포함하는 복잡한 하이브리드 최적화 문제를 도입하며, 이를 위해 휴리스틱 교대 최적화 전략을 설계했지만 수렴 보장이나 최적성 이론적 분석은 제한적이다.

3. **폰트 다양성의 훈련 의존성**: 텍스트 인코더와 생성기를 공동 학습하여 훈련 데이터셋 내의 다양한 폰트 스타일을 활용하므로, 훈련 데이터에 없는 폰트에 대한 일반화에는 여전히 제약이 있을 수 있다.

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 현재 일반화 강점

DreamText는 diffusion 학습을 개선하여 문자 수준 어텐션을 안내하고, 텍스트 인코더와 생성기를 공동 학습하여 다양한 폰트 스타일을 처리하는 능력을 강화함으로써 scene text synthesis를 향상시킨다.

이를 통해:
- **단일 폰트 인코더의 한계를 극복**: 훈련 데이터 내 다양한 폰트를 학습하여 폴리스타일 시나리오에서의 일반화 향상
- **어텐션 재추정을 통한 위치 일반화**: 교대 최적화를 통해 모델이 다양한 장면 레이아웃에서도 정확한 문자 위치를 추정하도록 유도

### 3.2 일반화 향상 가능성 분석

#### (i) 다국어(Multilingual) 확장 가능성

TextDiffuser, UdiffText, DreamText와 같은 최근 방법들은 단일 라인 텍스트 생성만 지원하고 영어에만 국한되어 있다는 한계가 있다. 이 한계를 극복하기 위해:
- 다국어 텍스트 인코더(예: ByT5, mT5 등)와의 통합이 유망한 방향
- AnyText의 다국어 확장 철학을 DreamText의 어텐션 교정 메커니즘에 결합 가능

#### (ii) 멀티라인(Multi-line) 텍스트 일반화

현재 단일 라인에 국한된 구조는 멀티라인 텍스트 생성에서 더 강한 문맥 간섭, 잠재적인 마스크 영역 겹침, 정확한 위치 정렬의 어려움을 해결해야 함을 시사한다. 문자 마스크 기반의 교대 최적화를 멀티라인 단위로 확장하면 이 문제를 해결할 수 있다.

#### (iii) 폰트 도메인 외 일반화 (Zero-shot Font Generalization)

DreamText는 다양한 폰트와 같은 추가적인 제어 조건을 도입하여 시각 텍스트의 렌더링 능력을 향상시킨다. 그러나 훈련 데이터에 없는 폰트에 대한 제로샷 일반화를 위해서는:
- **Few-shot 폰트 적응**: CLIP 기반 폰트 임베딩 확장
- **메타-학습 기반 인코더**: 새로운 폰트에 빠르게 적응하는 학습 전략 도입 가능

#### (iv) 모델 구조 측면의 일반화 개선 방향

현재 Stable Diffusion U-Net 기반 구조를 최신 **DiT(Diffusion Transformer)** 구조로 전환하면 더 강한 표현력과 일반화 성능을 기대할 수 있다. 실제로:
확산 기반 scene text synthesis는 빠르게 발전하고 있지만, 기존 방법들은 대개 추가적인 시각적 조건 제어 모듈에 의존하며 다국어 생성을 지원하기 위한 대규모 주석 데이터를 필요로 한다. 이에 복잡한 보조 모듈의 필요성을 재검토하고 diffusion 모델의 내재된 문맥 추론 능력을 활용하는 접근법이 탐색되고 있다.

---

## 4. 관련 최신 연구 비교 분석 (2020년 이후)

| 방법 | 발표 | 접근 방식 | 핵심 특징 | 한계 |
|---|---|---|---|---|
| **MOSTEL** | ~2023 | GAN 기반 스타일 전이 | 참조 이미지 스타일 전이 | 임의 스타일 생성 불가 |
| **TextDiffuser** (Chen et al.) | 2023 | Diffusion + OCR 마스크 | 세그멘테이션 마스크 기반 조건 제어 | 단일 폰트, 단일 라인, 영어 |
| **UDiffText** (Zhao & Lian) | 2023/2024 | Diffusion + 문자 인식 인코더 | 경량 문자 수준 인코더, segmentation map 감독 | 제한된 표현 도메인, 어텐션 편향 |
| **AnyText** (Tuo et al.) | 2024 (ICLR) | Diffusion + 다국어 | GlyphControl 철학 + 다국어 확장 | 시퀀스 정확도 부족 (특히 편집) |
| **DreamText** (Wang et al.) | CVPR 2025 | Diffusion + 교대 최적화 | 문자 어텐션 교정 + 공동 폰트 학습 | 영어 단일 라인, 하이브리드 최적화 복잡도 |
| **AnyText2** (Tuo et al.) | 2024 | Diffusion + 다국어 속성 제어 | WriteNet+AttnX 구조, 라인별 속성 제어 | — |
| **TextFlux** | 2025 | DiT 기반, OCR-free | OCR 인코더 없음, 다국어, 멀티라인 | — |

TextDiffuser, UdiffText, DreamText와 같은 최근 방법들은 단일 라인 텍스트 생성과 영어 제한이라는 두 가지 핵심 측면에서 제약을 보인다.

이에 대응하여 TextFlux는 DiT 기반 프레임워크로 다국어 scene text synthesis를 가능하게 하며, OCR 인코더(시각 텍스트 관련 특징을 추출하기 위해 특별히 사용되는 추가 시각적 조건 모듈)를 제거하는 **OCR-free 모델 아키텍처**를 제안한다.

UDiffText는 경량 문자 수준 텍스트 인코더를 설계 및 학습하여 기존 CLIP 인코더를 대체하며, 문자 수준 세그멘테이션 맵의 감독 하에 로컬 어텐션 제어를 통합하여 diffusion 모델을 파인튜닝한다.

---

## 5. 앞으로의 연구에 미치는 영향 및 고려할 점

### 5.1 앞으로의 연구에 미치는 영향

1. **문자 수준 어텐션 제어의 표준화**
DreamText는 문자들의 어텐션을 이상적인 생성 영역에 더 정밀하게 집중시키고, 추가 제약으로 텍스트 영역의 학습과 문자 표현을 강화하는 접근법을 제시함으로써, 향후 연구에서 문자 수준의 세밀한 제어가 단순한 end-to-end 학습보다 우월함을 보였다.

2. **하이브리드 최적화 프레임워크의 확장**
이산-연속 혼합 최적화 전략은 텍스트 레이아웃 결정(이산)과 이미지 생성 품질(연속)을 동시에 다루는 연구 패러다임을 확립한다.

3. **폰트 인코더의 공동 학습 패러다임**
텍스트 인코더와 생성기를 공동 학습하여 훈련 데이터셋 내 다양한 폰트 스타일을 활용하며, 이 공동 학습이 휴리스틱 교대 최적화 과정에 자연스럽게 통합되어 문자 표현 학습과 문자 어텐션 재추정 간의 시너지적 상호작용을 촉진함을 입증하였다.

4. **후속 연구에서의 직접 참조**
DreamText는 다양한 폰트와 같은 추가적인 제어 조건을 도입하여 시각 텍스트의 렌더링 능력을 향상시키는 방향을 제시했으며, 이후 방법들에 의해 지속적으로 비교 대상으로 인용되고 있다.

### 5.2 앞으로 연구 시 고려할 점

#### ① 다국어 및 멀티라인 확장
DreamText를 포함한 기존 방법들이 단일 라인과 영어에 국한된 한계를 극복하기 위해, 다국어 문자 인코더(예: ByT5, mT5)와 멀티라인 레이아웃 인식 모듈의 통합이 필요하다.

#### ② OCR-free 아키텍처로의 전환 가능성
복잡한 보조 모듈의 필요성을 재검토하고 diffusion 모델의 내재된 문맥 추론 능력을 활용하는 방향이 탐색될 필요가 있다. 이는 DreamText의 OCR 기반 마스크 의존성을 줄이는 방향과 연결된다.

#### ③ 교대 최적화의 이론적 수렴 분석
이산-연속 혼합 최적화의 수렴 보장 및 최적성 분석은 아직 이론적으로 미성숙한 영역으로, 향후 연구에서 엄밀한 이론적 보장이 보완될 필요가 있다.

#### ④ 훈련 데이터 효율성
일부 최신 방법(예: TextFlux)은 경쟁 방법 대비 1%의 훈련 데이터만으로도 강력한 성능을 달성하는 방향을 제시하고 있어, DreamText의 대규모 데이터(LAION-OCR) 의존성을 줄이는 데이터 효율적 학습 연구도 중요하다.

#### ⑤ 최신 생성 모델(DiT, Flow Matching 등)로의 이식
DreamText의 핵심 아이디어인 "문자 수준 어텐션 교정"과 "교대 최적화"를 Flux, SD3 등 최신 Diffusion Transformer 아키텍처에 적용할 때의 설계 조정이 고려되어야 한다.

---

## 📚 참고 자료 (출처)

| # | 출처 | 링크 |
|---|---|---|
| 1 | **arXiv 논문 페이지** (2405.14701) | https://arxiv.org/abs/2405.14701 |
| 2 | **arXiv HTML 전문 (v2)** | https://arxiv.org/html/2405.14701v2 |
| 3 | **arXiv HTML 전문 (최신)** | https://arxiv.org/html/2405.14701 |
| 4 | **CVPR 2025 Open Access** | https://openaccess.thecvf.com/content/CVPR2025/html/Wang_DreamText_High_Fidelity_Scene_Text_Synthesis_CVPR_2025_paper.html |
| 5 | **CVPR 2025 PDF** | https://openaccess.thecvf.com/content/CVPR2025/papers/Wang_DreamText_High_Fidelity_Scene_Text_Synthesis_CVPR_2025_paper.pdf |
| 6 | **IEEE Xplore** | https://ieeexplore.ieee.org/document/11094662/ |
| 7 | **GitHub 공식 구현** | https://github.com/CodeGoat24/DreamText |
| 8 | **프로젝트 페이지** | https://codegoat24.github.io/DreamText/ |
| 9 | **Hugging Face Paper Page** | https://huggingface.co/papers/2405.14701 |
| 10 | **Semantic Scholar** | https://www.semanticscholar.org/paper/DreamText:-High-Fidelity-Scene-Text-Synthesis-Wang-Zhang/efa280f6d3a35194f797c908a4e85ec2871676f3 |
| 11 | **TextFlux 논문** (비교 연구) | https://arxiv.org/html/2505.17778v1 |
| 12 | **UDiffText 논문** (비교 연구) | https://arxiv.org/html/2312.04884v1 |
| 13 | **TripleFDS 논문** (비교 연구) | https://arxiv.org/html/2511.13399v1 |
| 14 | **ResearchGate - DreamText** | https://www.researchgate.net/publication/380847266_High_Fidelity_Scene_Text_Synthesis |
