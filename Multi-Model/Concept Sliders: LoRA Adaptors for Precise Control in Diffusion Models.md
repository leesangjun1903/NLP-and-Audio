# Concept Sliders: LoRA Adaptors for Precise Control in Diffusion Models

> **논문 정보**: Rohit Gandikota, Joanna Materzyńska, Tingrui Zhou, Antonio Torralba, David Bau  
> **발표**: European Conference on Computer Vision (ECCV 2024)  
> **arXiv**: 2311.12092 (2023.11.20)

---

## 1. 핵심 주장 및 주요 기여 (요약)

이 논문은 디퓨전 모델의 이미지 생성에서 속성(attribute)을 정밀하게 제어할 수 있는 해석 가능한(interpretable) **Concept Sliders**를 제안한다. 핵심 접근법은 하나의 개념에 대응하는 **저랭크(low-rank) 파라미터 방향**을 식별하면서, 다른 속성에 대한 간섭을 최소화하는 것이다.

주요 기여는 다음과 같다:

1. **Textual & Visual Concept Sliders**: 슬라이더는 소수의 프롬프트 또는 샘플 이미지를 사용하여 생성되며, 텍스트 기반 또는 시각적 개념 모두에 대해 슬라이더 방향을 만들 수 있다.
2. **Plug-and-Play 합성**: Concept Sliders는 플러그 앤 플레이 방식으로 효율적으로 합성 가능하며 연속적으로 조절하여 이미지 생성을 정밀하게 제어할 수 있다.
3. **우수한 편집 성능**: 기존 편집 기법과의 정량적 비교 실험에서 더 강한 목표 편집 효과와 더 낮은 간섭을 보여준다.
4. **품질 결함 수정**: Stable Diffusion XL의 지속적인 품질 문제(객체 변형, 왜곡된 손 등)를 해결할 수 있다.
5. **StyleGAN 잠재변수 전이**: StyleGAN의 잠재변수를 디퓨전 모델로 전이하여 텍스트로 설명하기 어려운 시각적 개념을 직관적으로 편집할 수 있다.

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

기존의 프롬프트 수정 방식은 많은 속성을 제어할 수 있지만, prompt-seed 민감도로 인해 전체 이미지 구조를 종종 변경한다. PromptToPrompt, Pix2Video와 같은 사후(post-hoc) 기법은 디퓨전을 역전시키고 cross-attention을 수정하여 개념 편집을 가능하게 하지만, 개념당 별도의 추론 패스가 필요하고, 동시 편집이 제한적이며, 이미지별 프롬프트 엔지니어링이 필요하다. 이러한 방법들은 나이 수정 시 인종이 바뀌는 등 개념 얽힘(entanglement) 문제를 야기한다.

### 2.2 제안 방법 (수식 포함)

#### (a) 디퓨전 모델 기초

디퓨전 모델의 순방향 과정(forward process)은 다음과 같다:

$$x_t = \sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1 - \bar{\alpha}_t}\, \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

디퓨전 모델은 랜덤 가우시안 노이즈 $X_T$를 샘플링하고 점진적으로 디노이징하여 이미지 $x_0$를 생성하며, 실제로는 $x_t$가 입력될 때 진정한 노이즈 $\epsilon$을 예측하는 것으로 단순화된다.

#### (b) 개념 슬라이더의 스코어 함수

논문의 핵심 아이디어는 **Classifier-Free Guidance**를 훈련 시간(training time)에 적용하여 개념 방향을 학습하는 것이다. 제안된 수정 스코어 함수(score function)는 다음과 같다:

$$\nabla \log P_{\theta}(X \mid c_t) + \eta \left( \nabla \log P_{\theta}(X \mid c_+) - \nabla \log P_{\theta}(X \mid c_-) \right)$$

이를 재매개변수화(reparameterization)와 Tweedie 공식을 사용해 노이즈 예측 형태로 표현하면:

$$\epsilon_{\theta^*}(X, c_t, t) \leftarrow \epsilon_{\theta}(X, c_t, t) + \eta \left( \epsilon_{\theta}(X, c_+, t) - \epsilon_{\theta}(X, c_-, t) \right)$$

여기서:
- $c_t$: 타겟 개념 (예: "person")
- $c_+$: 강화할 속성 (예: "old person")
- $c_-$: 억제할 속성 (예: "young person")
- $\eta$: 가이던스 스케일
- $\theta$: 사전 훈련된(frozen) 모델 파라미터
- $\theta^*$: 학습된 슬라이더 모델 파라미터

제안된 스코어 함수는 타겟 개념 $c_t$의 분포를 $c_+$의 속성을 더 많이 보이고, $c_-$의 속성을 더 적게 보이도록 이동시킨다.

#### (c) 비얽힘(Disentanglement) 목적 함수

실제로 단일 프롬프트 쌍이 다른 원치 않는 속성과 얽힌 방향을 식별할 수 있다. 따라서 보존 개념 $p \in \mathcal{P}$ (예: 나이 편집 시 인종명)를 사용하여 최적화를 제약한다.

단순히 $P_\theta(c_+ \mid X)$를 증가시키는 대신, 모든 $p$에 대해 $P_\theta((c_+, p) \mid X)$를 증가시키고 $P_\theta((c_-, p) \mid X)$를 감소시키는 것을 목표로 한다.

이를 종합한 최종 학습 목적 함수는:

$$\mathcal{L} = \sum_{p \in \mathcal{P}} \mathbb{E}_{t, \epsilon} \left\| \epsilon_{\theta^*}(x_t, (c_t, p), t) - \left[ \epsilon_{\theta}(x_t, (c_t, p), t) + \eta \left( \epsilon_{\theta}(x_t, (c_+, p), t) - \epsilon_{\theta}(x_t, (c_-, p), t) \right) \right] \right\|^2$$

#### (d) LoRA 저랭크 파라미터화

가중치 업데이트는 LoRA (Low-Rank Adaptation) 형태로 분해된다:

$$W' = W + \alpha \cdot B A$$

여기서 $W \in \mathbb{R}^{d \times k}$는 원래 가중치, $B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times k}$이며 $r \ll \min(d, k)$이다. $\alpha$는 추론 시 슬라이더 강도를 조절하는 스케일링 팩터이다.

### 2.3 모델 구조

Concept Sliders는 사전 훈련된 모델 위에 경량의 플러그 앤 플레이 어댑터를 제공하여, 단일 추론 패스에서 원하는 개념에 대한 정밀하고 연속적인 제어를 가능하게 한다.

- **기본 모델**: Stable Diffusion v1.4 및 Stable Diffusion XL (SDXL)
- **슬라이더 구조**: U-Net의 cross-attention 레이어에 적용되는 LoRA 모듈 (rank 4)
- **훈련**: 주로 SDXL에서 평가되었으며, 500 에폭 훈련
- **합성성**: Concept Sliders는 모듈형이고 합성 가능하며, 50개 이상의 고유 슬라이더를 출력 품질 저하 없이 합성할 수 있다.

두 가지 훈련 모드를 제공한다:
- **Textual Concept Sliders**: 텍스트 프롬프트만으로 훈련
- **Visual Concept Sliders**: 텍스트로 설명할 수 없는 개념을 위해, 아티스트가 6~8쌍의 이미지를 사용하여 생성

### 2.4 성능 향상

기존 편집 기법과의 정량적 비교에서 더 강한 목표 편집과 더 낮은 간섭을 보여준다.

구체적 성능 측면:
- **정밀한 연속 제어**: LoRA 공식에서 도입된 스케일링 팩터가 추론 시 수정되어 편집 강도를 조절하며, 프레임워크 재훈련 없이 편집을 더 강하게 만든다.
- **복합 편집**: 아티스트에게 텍스트, 시각적, GAN 정의 Concept Sliders를 무수히 혼합할 수 있는 세밀한 이미지 제어의 새로운 가능성을 제공한다.
- **결함 수정**: "repair" 슬라이더는 모델이 더 사실적이고 왜곡되지 않은 이미지를 생성하도록 하며, 밀집 배열된 객체의 렌더링 개선, 건축 선 직선화, 복잡한 형태 가장자리의 흐림/왜곡 방지가 가능하다.

### 2.5 한계

LoRA를 사용한 개념 학습으로 인해 파라미터가 추가되고 추론 시간이 증가하며, 이러한 어댑터는 모델 특화(model-specific)이어서 SD v1.5와 SDXL 같은 다른 아키텍처에서는 재훈련이 필요하다.

추가적 한계:
- 자연 장면 이미지(in-the-wild images)에서 테스트할 때 미해결 이슈가 남아 있으며, 통제되지 않은 환경에서 촬영된 이미지는 다양한 조명 조건, 배경, 피사체 포즈를 포함하여 고유한 도전을 제시하며, 일반화 성능의 한계가 Concept Sliders의 실용화를 방해한다.
- Concept Sliders는 텍스트-이미지 생성에서의 간접적 적용으로 인해 실제 이미지 편집 작업에서 성능이 떨어질 수 있다.

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 저랭크 제약의 핵심 역할

각 Concept Slider는 디퓨전 모델의 저랭크 수정이다. 저랭크 제약이 없는 파인튜닝은 정밀도와 생성 이미지 품질을 저하시키는 반면, 저랭크 훈련은 최소 개념 부분공간(minimal concept subspace)을 식별하여 제어된, 고품질의, 비얽힌 편집을 가능하게 한다.

저랭크 제약이 일반화에 기여하는 두 가지 이유:

1. 파라미터 수와 연산의 **효율성**
2. 더 나은 **일반화**와 함께 편집 방향을 정밀하게 포착

### 3.2 비얽힘(Disentanglement)을 통한 일반화

비얽힘 공식은 원치 않는 속성으로부터 편집을 분리하는 데 도움이 된다. 예를 들어, 나이 편집 시 인종이나 성별의 원치 않는 변화를 방지한다.

이는 슬라이더가 학습한 방향이 **특정 입력에 과적합(overfit)하지 않고** 다양한 프롬프트와 시드에 걸쳐 일관되게 작동하도록 보장한다.

### 3.3 합성 가능성(Composability)을 통한 일반화

저랭크 슬라이더 방향의 핵심 장점은 합성 가능성이며, 사용자들이 한 번에 하나의 개념에 제한되지 않고 여러 슬라이더를 결합하여 세밀한 제어를 할 수 있다.

### 3.4 한계 및 개선 방향

기존 Concept Sliders는 비-AIGC 이미지, 특히 실제 환경에서 촬영된 이미지에서 성능이 떨어지는 경우가 있다. 이러한 격차를 해소하기 위해 Beyond Sliders와 같은 후속 연구에서 GAN과 디퓨전 모델을 통합하여 다양한 이미지 카테고리에서의 정교한 이미지 조작을 가능하게 하고, adversarial 방식의 미세 가이던스로 이미지 품질과 사실감을 향상시킨다.

---

## 4. 향후 연구에 미치는 영향 및 고려사항

### 4.1 연구 영향

1. **새로운 패러다임 정립**: Concept Sliders는 디퓨전 모델의 파라미터 공간에서 해석 가능한 **의미적 방향(semantic direction)**을 식별하는 패러다임을 확립했다. 이는 GAN에서의 잠재 공간 탐색을 디퓨전 모델로 확장한 것이다.

2. **후속 연구 촉발**: Concept Sliders 이후 Prompt Sliders (Sridhar & Vasconcelos, 2024), CASteer (Gaintseva et al., 2025), NoiseCLR (Dalva & Yanardag, 2024), Concept Steerers (Kim & Ghadiyaram, 2025) 등의 후속 연구가 발표되었다.

3. **모달리티 확장**: FreeSliders(2025)는 추론 시간에 직접 Concept Slider 업데이트를 추정하여, 이미지, 비디오, 오디오에 걸쳐 플러그 앤 플레이, 모달리티-비의존적 슬라이더를 가능하게 한다.

4. **안전성 연구**: 슬라이더 기법은 디퓨전 모델에서 부적절한 개념을 지우거나(erasing) 완화(moderating)하는 안전성 연구로 자연스럽게 확장된다.

### 4.2 향후 연구 시 고려할 점

1. **아키텍처 일반성**: LoRA 기반 어댑터는 모델 특화적이어서 다른 아키텍처에서 재훈련이 필요하다. FLUX 등 새로운 아키텍처로의 전이 가능성 연구가 필요하다.

2. **실제 이미지(In-the-Wild) 적용**: 일반화 성능의 한계가 실용화를 저해하므로, 비통제 환경의 다양한 이미지에 대한 강건성 향상이 필요하다.

3. **효율성-성능 트레이드오프**: Prompt Sliders와 같은 방법은 LoRA보다 30% 빠르며, 개념 임베딩당 3KB만 필요한 반면 각 LoRA 어댑터는 8922KB 이상이 필요하므로, 더 효율적인 방법 탐색이 필요하다.

4. **다중 속성 얽힘 문제**: 모든 모델에서 어느 정도의 속성 얽힘이 지속되며, 예를 들어 피부톤을 수정하면 머리색이나 조명 같은 상관 특징에 의도치 않게 영향을 줄 수 있는데, 이는 생성 모델 자체의 본질적 속성 결합에서 발생한다.

5. **윤리적 고려사항**: 제어 가능한 생성의 기술적 장벽을 낮추는 것은 창작 및 과학적 워크플로를 민주화할 잠재력이 있지만, 오용과 공정성에 대한 우려도 제기되므로, 표준화되고 투명한 벤치마크와 민감 속성에 대한 보호장치가 필수적이다.

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

| 연구 | 연도 | 핵심 방법 | Concept Sliders 대비 차이점 |
|------|------|-----------|---------------------------|
| **LoRA** (Hu et al.) | 2021 | 저랭크 행렬 분해로 LLM 적응 | 일반적 파인튜닝 기법; 개념별 방향 제어 없음 |
| **ControlNet** (Zhang et al.) | 2023 | 구조적 조건(포즈, 엣지, 깊이)으로 제어 | 구조적 제어에 특화; 의미적 속성의 연속 조절 불가 |
| **IP-Adapter** (Ye et al.) | 2023 | 참조 이미지를 프롬프트로 사용하는 어댑터 | 스타일/콘텐츠 가이던스; 단일 속성의 정밀 슬라이딩 불가 |
| **Prompt Sliders** (Sridhar & Vasconcelos) | 2024 | 텍스트 임베딩 학습을 통한 개념 제어 | 같은 텍스트 인코더를 공유하는 모델 간 일반화 가능; 더 효율적이나 시각적 개념에 제한적 |
| **Beyond Sliders** | 2025 | GAN + 디퓨전 모델 통합, adversarial 훈련 | Age 작업에서 CLIP 점수 22.15 (vs. Concept Sliders 21.92), Chubby 작업에서 LPIPS 0.037 (vs. 0.078)로 시각적 일관성 보존에서 우수 |
| **SliderEdit** | 2025 | 지시 기반 편집 모델에 연속 제어 통합 | FLUX-Kontext, Qwen-Image-Edit 같은 최신 모델과 원활하게 통합; PPS 손실로 더 나은 비얽힘 |
| **FreeSliders** | 2025 | 훈련 없이 추론 시 슬라이더 업데이트 추정 | 이미지, 비디오, 오디오에 걸쳐 아키텍처 비의존적; 추론 비용 증가 대비 일반성·배포 용이성 향상 |
| **CompSlider** (Zhu et al.) | 2025 | 다중 속성 합성 슬라이더 | 이전 방법의 순차 조정이 속성 얽힘을 무시하는 반면, CompSlider는 동시 제어로 더 나은 비얽힘과 독립적 조정 보장 |

---

## 참고자료 및 출처

1. **Gandikota, R., Materzyńska, J., Zhou, T., Torralba, A., Bau, D.** "Concept Sliders: LoRA Adaptors for Precise Control in Diffusion Models." ECCV 2024. [arXiv:2311.12092](https://arxiv.org/abs/2311.12092)
2. **프로젝트 페이지**: [https://sliders.baulab.info/](https://sliders.baulab.info/)
3. **GitHub 저장소**: [https://github.com/rohitgandikota/sliders](https://github.com/rohitgandikota/sliders)
4. **ECCV 2024 공식 PDF**: [https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/05660.pdf](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/05660.pdf)
5. **Springer Nature**: [https://link.springer.com/chapter/10.1007/978-3-031-73661-2_10](https://link.springer.com/chapter/10.1007/978-3-031-73661-2_10)
6. **Sridhar, D., Vasconcelos, N.** "Prompt Sliders for Fine-Grained Control, Editing and Erasing of Concepts in Diffusion Models." ECCV Workshops 2024. [프로젝트 페이지](https://deepaksridhar.github.io/promptsliders.github.io/)
7. **Beyond Sliders**: [arXiv:2509.11213](https://arxiv.org/html/2509.11213)
8. **SliderEdit**: [arXiv:2511.09715](https://arxiv.org/html/2511.09715v1)
9. **FreeSliders**: [arXiv:2511.00103](https://arxiv.org/html/2511.00103v1)
10. **CompSlider**: Zhu et al., ICCV 2025. [PDF](https://openaccess.thecvf.com/content/ICCV2025/papers/Zhu_CompSlider_Compositional_Slider_for_Disentangled_Multiple-Attribute_Image_Generation_ICCV_2025_paper.pdf)
11. **Hu, E.J. et al.** "LoRA: Low-Rank Adaptation of Large Language Models." arXiv:2106.09685 (2021)
12. **Ho, J., Salimans, T.** "Classifier-Free Diffusion Guidance." arXiv:2207.12598 (2022)
13. **Unite.AI 분석**: [https://www.unite.ai/concept-sliders-precise-control-in-diffusion-models-with-lora-adaptors/](https://www.unite.ai/concept-sliders-precise-control-in-diffusion-models-with-lora-adaptors/)
14. **Hugging Face Paper 페이지**: [https://huggingface.co/papers/2311.12092](https://huggingface.co/papers/2311.12092)

---

> **참고**: 본 분석에서 제시된 수식은 논문 원문과 프로젝트 페이지의 공식을 바탕으로 재구성한 것입니다. 세부적인 하이퍼파라미터(예: rank=4, alpha=1, guidance=4, 500 epochs)는 공식 코드 저장소 및 논문에서 확인 가능합니다.
