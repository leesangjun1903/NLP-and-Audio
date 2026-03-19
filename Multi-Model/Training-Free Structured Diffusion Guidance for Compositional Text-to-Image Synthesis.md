# Training-Free Structured Diffusion Guidance for Compositional Text-to-Image Synthesis

## 1. 핵심 주장 및 주요 기여 요약

이 논문은 대규모 확산(diffusion) 기반 Text-to-Image(T2I) 모델이 **속성 바인딩(attribute binding)**과 **구성적 생성(compositional generation)**에서 여전히 취약하다는 문제를 지적하고, **추가 학습 없이(training-free)** 언어의 구조적 표현(constituency tree, scene graph)을 확산 모델의 cross-attention 레이어에 통합하는 **Structured Diffusion Guidance (StructureDiffusion)** 방법을 제안한다.

### 주요 기여:
1. **Training-free 방법론**: 추가 학습 데이터나 파인튜닝 없이 언어 구조 정보를 활용하여 compositional T2I 성능을 향상시키는 직관적이고 효율적인 방법 제안
2. **새로운 벤치마크 제안**: Attribute Binding Contrast set (ABC-6K)과 Concept Conjunction 500 (CC-500) 벤치마크를 제안하여 T2I 모델의 구성적 능력을 정량 평가
3. **심층 분석**: 잘못된 속성 바인딩의 근본 원인(CLIP의 causal attention에 의한 토큰 오염, 부정확한 attention map)을 규명하고, cross-attention의 key/value가 각각 레이아웃/콘텐츠를 제어한다는 성질을 실증적으로 검증

---

## 2. 상세 설명

### 2.1 해결하고자 하는 문제

대규모 T2I 확산 모델(Stable Diffusion, DALL-E 2, Imagen 등)은 고품질 이미지를 생성할 수 있으나, **여러 객체가 포함된 복합 프롬프트**에서 세 가지 핵심 문제가 발생한다:

| 문제 유형 | 설명 | 예시 |
|---------|------|------|
| **Attribute Leakage** | 한 객체의 속성이 다른 객체에 부분적으로 나타남 | "a red car and a white sheep" → 빨간 양 |
| **Interchanged Attributes** | 두 객체의 속성이 서로 교환됨 | "a brown bench in front of a white building" → 흰 벤치, 갈색 건물 |
| **Missing Objects** | 하나 이상의 객체가 누락됨 | "a blue backpack and a brown elephant" → 배낭 누락 |

**근본 원인 분석**:
- **CLIP 텍스트 인코더의 causal attention mask**: 시퀀스 뒤쪽 토큰이 앞쪽 토큰의 의미에 오염됨. 예를 들어 "a yellow apple and red bananas"에서 "banana"의 임베딩이 "yellow"의 의미를 흡수하여 노란 바나나를 생성
- **부정확한 attention map**: 특정 토큰의 attention이 의도하지 않은 공간 영역에도 활성화됨

### 2.2 제안하는 방법 (수식 포함)

#### Background: Cross-Attention in Stable Diffusion

Stable Diffusion의 UNet 내 cross-attention 레이어에서 2D feature map $\mathcal{X}^t$가 query로, CLIP 텍스트 임베딩 $\mathcal{W}_p$가 key/value로 투영된다:

$$Q^t = f_Q(\mathcal{X}^t) \in \mathbb{R}^{(n, h \times w, d)}$$

$$K_p = f_K(\mathcal{W}_p) \in \mathbb{R}^{(n, l, d)}, \quad V_p = f_V(\mathcal{W}_p) \in \mathbb{R}^{(n, l, d)}$$

Attention map은 다음과 같이 계산된다:

$$M^t = f_M(Q^t, K_p) = \text{Softmax}\left(\frac{Q^t K_p^T}{\sqrt{d}}\right), \quad M^t \in \mathbb{R}^{(n, h \times w, l)} $$

여기서 $n$은 attention head 수, $d$는 feature dimension, $l$은 시퀀스 길이, $h \times w$는 공간 해상도이다.

**핵심 관찰**: Attention map $M^t$는 **토큰-영역 대응(token-region association)**을 제공하며, key는 **공간 레이아웃**, value는 **의미적 콘텐츠**를 각각 제어한다.

#### Structured Diffusion Guidance

**Step 1: 구조적 표현 추출**

파서 $\xi(\cdot)$를 사용하여 프롬프트 $\mathcal{P}$에서 계층적 개념 집합을 추출한다:

$$\mathcal{C} = \{c_1, c_2, \ldots, c_k\}$$

Constituency tree에서는 모든 계층의 noun phrase(NP)를, scene graph에서는 객체와 관계를 text segment로 추출한다.

**Step 2: 개별 인코딩 및 정렬**

각 NP를 개별적으로 CLIP 인코더에 통과시킨다:

$$\mathbb{W} = [\mathcal{W}_p, \mathcal{W}_1, \mathcal{W}_2, \ldots, \mathcal{W}_k], \quad \mathcal{W}_i = \text{CLIP}_{\text{text}}(c_i), \quad i = 1, \ldots, k $$

각 $\mathcal{W}_i$는 전체 프롬프트 시퀀스 $\mathcal{W}_p$와 **재정렬(re-alignment)**되어 $\overline{\mathcal{W}_i}$를 생성한다. 이는 $\langle\text{bos}\rangle$와 $\langle\text{pad}\rangle$ 사이의 임베딩을 $\mathcal{W}_p$의 해당 위치에 삽입하는 방식이다.

**Step 3: 다중 Value를 통한 구조적 가이던스 (기본 방법)**

Key는 전체 프롬프트에서만 계산하여 레이아웃을 유지하고, value만 다중으로 사용한다:

$$\mathbb{V} = [f_V(\mathcal{W}_p), f_V(\overline{\mathcal{W}_1}), \ldots, f_V(\overline{\mathcal{W}_k})] = [V_p, V_1, \ldots, V_k] $$

$$O^t = \frac{1}{(k+1)} \sum_{i} (M^t V_i), \quad i = p, 1, 2, \ldots, k $$

이 방식은 $M^t$가 여전히 $Q^t, K_p$로부터 계산되므로 **레이아웃을 변경하지 않으면서** 각 NP의 의미를 해당 공간 영역에 더 정확하게 매핑한다.

**Step 4: 다중 Key를 통한 변형 (missing object 대응)**

객체 누락 문제가 심한 concept conjunction의 경우, key도 다중으로 계산하는 변형을 사용한다:

$$\mathbb{K} = \{f_K(\overline{\mathcal{W}_i})\}, \quad \mathbb{M}^t = \{f_M(Q^t, K_i)\}, \quad i = p, 1, 2, \ldots, k $$

$$O^t = \frac{1}{(k+1)} \sum_{i} (M_i^t V_k), \quad i = p, 1, 2, \ldots, k $$

이 변형은 각 NP별 독립적인 attention map을 생성하여 누락된 객체의 공간 영역을 복원한다.

### 2.3 모델 구조

StructureDiffusion은 **Stable Diffusion v1.4** 위에 구축되며, 기존 모델의 어떤 파라미터도 수정하지 않는다:

```
[프롬프트] → [Parser ξ] → [NP 추출: C = {c1, ..., ck}]
                              ↓
[각 NP] → [CLIP_text] → [W1, ..., Wk] → [재정렬] → [fV로 Value 생성]
                              ↓
[전체 프롬프트] → [CLIP_text] → [Wp] → [fK로 Key 생성] → [Attention Map Mt]
                              ↓
[Mt × 다중 Vi의 평균] → [Ot] → [후속 레이어]
```

**알고리즘 요약 (Algorithm 1)**:
1. 파서로 개념 집합 $\mathcal{C}$ 추출
2. 각 개념과 전체 프롬프트를 CLIP으로 인코딩
3. 각 diffusion step $t = T, \ldots, 1$에서:
   - 각 cross-attention 레이어에서 $Q^t, K_p$로 $M^t$ 계산
   - 다중 $V_i$와 $M^t$를 곱하여 평균 → $O^t$ 생성
4. 최종 $z^0$을 디코더에 통과시켜 이미지 생성

### 2.4 성능 향상

#### 정량적 결과

**ABC-6K (Attribute Binding)**:

| 비교 대상 | Alignment Win↑ | Alignment Lose↓ | Fidelity Win↑ | Fidelity Lose↓ |
|---------|---------------|----------------|--------------|---------------|
| vs. Stable Diffusion | **42.2%** | 35.6% | **48.3%** | 39.1% |

- Alignment에서 약 **6.6%p 우위**, Fidelity에서 약 **9.2%p 우위**

**CC-500 (Concept Conjunction)**:

| 비교 대상 | Alignment Win↑ | Fidelity Win↑ |
|---------|---------------|--------------|
| vs. Stable Diffusion | **31.8%** (vs 27.7%) | **37.8%** (vs 30.6%) |
| vs. Composable Diffusion | **46.5%** (vs 30.1%) | **61.4%** (vs 19.8%) |

**세부 인간 평가 (CC-500)**:
- 두 객체 모두 생성: Stable Diffusion 34.5% → **StructureDiffusion 38.0%** (+3.5%p)
- 두 객체 + 올바른 색상: 19.2% → **22.7%** (+3.5%p)

**일반 프롬프트 (MSCOCO 10K)**:

| 지표 | Stable Diffusion | StructureDiffusion |
|------|-----------------|-------------------|
| IS ↑ | 39.9 | **40.9** |
| FID ↓ | 18.0 | **17.9** |
| R-Precision ↑ | 72.2 | **72.3** |

→ 일반 프롬프트에서도 이미지 품질/다양성 저하 없이 유지

#### 정성적 결과
- **색상 누출 방지**: "a red bird and a green apple" 등에서 색상이 올바른 객체에만 적용
- **누락 객체 복원**: "a blue backpack and a brown elephant" 등에서 누락 객체 생성
- **세밀한 디테일 향상**: 색상 외에도 형태, 재질(shape, material) 등 다양한 속성에서 개선 (Fig. 15)

### 2.5 한계

1. **외부 파서 의존성**: 파싱 함수의 품질에 성능이 좌우되며, 잘못된 파싱은 오히려 성능 저하 가능
2. **스타일 기술 문제**: "in Van Gogh style" 같은 스타일 설명이 별도 NP로 분류될 경우, 이미지 공간에 그라운딩이 어려움 (다만 실험적으로 스타일에는 부정적 영향 없음을 확인)
3. **유사 이미지 생성 빈도**: StructureDiffusion이 Stable Diffusion과 매우 유사한 이미지를 생성하는 경우가 상당수 존재 (평가 시 20% 필터링 필요)
4. **Cross-attention 계산 비용 증가**: NP 수에 비례하여 계산량 증가 (다만 padding 토큰이 대부분이므로 실질적 오버헤드는 제한적)
5. **공간 정보의 명시적 제어 부재**: 바운딩 박스나 좌표 등 명시적 레이아웃 입력을 활용하지 않음

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 일반 프롬프트에서의 품질 유지

StructureDiffusion의 핵심 강점은 **compositional 프롬프트에서 성능을 향상시키면서도 일반 프롬프트에서의 이미지 품질과 다양성을 유지**한다는 점이다. MSCOCO 10K에서 IS, FID, R-Precision이 Stable Diffusion과 동등한 수준을 보여, 방법론이 기존 성능을 훼손하지 않는 **일반화된 개선**임을 입증한다.

### 3.2 다양한 구조적 표현으로의 확장

이 방법은 constituency tree에 국한되지 않고 **scene graph 파서**로도 확장 가능하다:
- Scene graph 파서 사용 시 MSCOCO에서 IS 39.2, FID 17.9, R-Precision 72.0으로 품질 유지
- ABC-6K에서 34.2%-32.9%-32.9% (Win-Lose-Tie)로 Stable Diffusion 대비 개선
- CC-500에서는 constituency parser와 동일한 text span이 추출되어 동일 결과

### 3.3 색상 외 속성으로의 일반화

실험 결과(Fig. 15), StructureDiffusion은 **색상뿐 아니라 형태(shape), 크기(size), 재질(material)** 등 다양한 속성 바인딩에서도 개선을 보인다:
- "a wooden building with a stone bench" vs "a stone building with a wooden bench" → 재질 속성 올바르게 바인딩
- "a red sphere and a blue cube" → 형태와 색상 동시 바인딩

### 3.4 스타일 프롬프트와의 호환성

Fig. 16에서 "an astronaut riding a horse"에 다양한 스타일 기술(fantasy anime, portrait, sharp focus, kyoto animation style 등)을 추가한 실험에서 **StructureDiffusion이 스타일 생성에 부정적 영향을 미치지 않음**을 확인하였다.

### 3.5 일반화 성능 향상을 위한 핵심 메커니즘

1. **Value의 분리 인코딩**: 각 NP를 독립적으로 인코딩함으로써 CLIP의 causal attention에 의한 토큰 간 의미 오염을 방지
2. **Key 유지를 통한 레이아웃 보존**: 전체 프롬프트의 key를 유지하여 공간 레이아웃의 안정성 확보
3. **Re-alignment 기법**: padding 임베딩이 전체 프롬프트의 high-level semantics를 인코딩하고 있다는 발견(Fig. 7)에 기반하여, text span의 인코딩을 전체 시퀀스와 정렬함으로써 이미지 품질 유지

### 3.6 일반화의 한계와 개선 방향

- 파서의 품질이 일반화 성능의 상한을 결정하므로, **학습 기반 파서**로 교체 시 추가 개선 가능
- 3개 이상의 복잡한 객체 조합이나 공간 관계(위/아래/옆 등)에 대한 일반화는 추가 연구 필요
- 명시적 공간 정보(좌표, 바운딩 박스)를 활용하는 방향으로 확장 시 더 강건한 일반화 가능

---

## 4. 향후 연구에 미치는 영향 및 고려할 점

### 4.1 연구적 영향

1. **Training-free 패러다임의 확산**: 이 논문은 대규모 T2I 모델의 내부 메커니즘(cross-attention)을 이해하고 조작함으로써 추가 학습 없이 성능을 개선할 수 있음을 보여주었다. 이는 이후 **Attend-and-Excite (Chefer et al., 2023)**, **SynGen (Rassin et al., 2024)** 등 유사한 training-free 접근법의 활발한 연구를 촉발시켰다.

2. **Cross-attention의 해석 가능성**: Key가 레이아웃을, Value가 콘텐츠를 제어한다는 분석은 이미지 편집(Prompt-to-Prompt, Hertz et al., 2022)에서의 발견을 compositional generation으로 확장한 것으로, 확산 모델의 **해석 가능한 제어(interpretable control)** 연구의 기반이 되었다.

3. **벤치마크 기여**: ABC-6K와 CC-500은 T2I 모델의 compositional 능력을 체계적으로 평가하는 표준 벤치마크로 활용되고 있으며, 이후 **T2I-CompBench (Huang et al., 2023)** 등 더 포괄적인 벤치마크 개발에 영감을 주었다.

### 4.2 향후 연구 시 고려할 점

1. **명시적 레이아웃 제어와의 결합**: StructureDiffusion은 암묵적(implicit) 방식으로 공간 정보를 활용하므로, 바운딩 박스나 세그멘테이션 마스크 등 **명시적 레이아웃 정보**와 결합하면 더 정밀한 제어가 가능할 것이다.

2. **더 강력한 텍스트 인코더 활용**: CLIP의 causal attention이 속성 오염의 원인 중 하나이므로, T5와 같은 **양방향 인코더** 또는 LLM 기반 인코더를 사용하는 모델(Imagen, SDXL 등)에서의 적용 및 효과 검증이 필요하다.

3. **복잡한 관계 처리**: 현재 방법은 주로 속성-객체 바인딩에 초점을 맞추고 있으며, **공간 관계**(위/아래/옆), **수량**(counting), **행동**(action) 등 더 복잡한 compositional 요소에 대한 확장 연구가 필요하다.

4. **파서 없는 구조 추출**: 외부 파서에 대한 의존성을 제거하기 위해, LLM을 활용한 자동 구조 추출이나 학습 가능한 구조 추론 모듈 연구가 유망하다.

5. **평가 지표 개선**: GLIP과 인간 평가 간의 일치도가 50% 미만으로 매우 낮아, compositional generation을 위한 **더 신뢰할 수 있는 자동 평가 지표** 개발이 시급하다.

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

| 연구 | 연도 | 핵심 접근법 | Training-Free | StructureDiffusion과의 비교 |
|-----|------|----------|:---:|------------------------|
| **Composable Diffusion** (Liu et al., 2022) | 2022 | 개별 concept의 score를 합산하는 병렬 diffusion | ✓ | 과포화/부자연스러운 텍스처 문제; StructureDiffusion이 fidelity에서 61.4% 승률로 압도 |
| **Prompt-to-Prompt** (Hertz et al., 2022) | 2022 | Cross-attention map 조작을 통한 이미지 편집 | ✓ | 이미지 편집에 초점; StructureDiffusion은 compositional 생성에 특화 |
| **Attend-and-Excite** (Chefer et al., 2023) | 2023 | Attention map의 최소 활성화를 최대화하는 latent 최적화 | ✓ | 객체 누락 문제에 더 직접적으로 대응하나, latent space에서의 gradient 계산 필요로 추론 속도 저하 |
| **SynGen** (Rassin et al., 2024) | 2024 | Dependency parsing 기반 attention map 재구성 | ✓ | 유사한 언어 구조 활용 철학이나, attention map 자체를 직접 수정하여 더 공격적인 제어 수행 |
| **GLIGEN** (Li et al., 2023) | 2023 | Grounded generation with bounding box input | ✗ | 명시적 레이아웃 입력 필요; 추가 학습 필요하지만 공간 제어가 더 정밀 |
| **T2I-CompBench** (Huang et al., 2023) | 2023 | Compositional T2I 종합 벤치마크 | - | ABC-6K/CC-500의 확장판으로 볼 수 있으며, 색상/형태/텍스처/공간/비공간 관계 등을 포괄적으로 평가 |
| **DALL-E 3** (Betker et al., 2023) | 2023 | 상세 캡션 재작성을 통한 text-image alignment 향상 | ✗ | 대규모 재학습 기반; 캡션 품질 개선으로 compositional 문제를 데이터 측면에서 해결 |
| **SDXL** (Podell et al., 2023) | 2023 | 더 큰 UNet, 듀얼 텍스트 인코더, refiner 모델 | ✗ | 모델 스케일 확대로 일부 compositional 문제 완화하나 근본적 해결은 아님 |
| **RPG** (Yang et al., 2024) | 2024 | LLM을 활용한 region-aware planning + compositional generation | ✓ | LLM이 레이아웃과 속성 할당을 계획하여 더 복잡한 scene에 대응 가능 |

### 핵심 트렌드 분석

1. **Training-free 방법의 확산**: StructureDiffusion의 성공 이후, 모델 재학습 없이 추론 시점에서 compositional 능력을 개선하는 연구가 활발해짐
2. **언어 구조 활용의 다양화**: Constituency tree → Scene graph → Dependency parsing → LLM 기반 계획으로 구조 추출 방법이 진화
3. **평가 체계의 고도화**: 단순 FID/IS에서 compositional-specific 벤치마크(T2I-CompBench, DVMP 등)로 발전
4. **하이브리드 접근**: Training-free와 학습 기반 방법의 결합, 또는 LLM과 diffusion 모델의 협업이 차세대 방향으로 부상

---

## 참고자료

1. **Feng, W., He, X., Fu, T.-J., Jampani, V., Akula, A., Narayana, P., Basu, S., Wang, X. E., & Wang, W. Y.** (2023). "Training-Free Structured Diffusion Guidance for Compositional Text-to-Image Synthesis." *ICLR 2023*. arXiv:2212.05032.
2. **Hertz, A., Mokady, R., Tenenbaum, J., Aberman, K., Pritch, Y., & Cohen-Or, D.** (2022). "Prompt-to-Prompt Image Editing with Cross Attention Control." arXiv:2208.01626.
3. **Liu, N., Li, S., Du, Y., Torralba, A., & Tenenbaum, J. B.** (2022). "Compositional Visual Generation with Composable Diffusion Models." arXiv:2206.01714.
4. **Rombach, R., Blattmann, A., Lorenz, D., Esser, P., & Ommer, B.** (2022). "High-Resolution Image Synthesis with Latent Diffusion Models." *CVPR 2022*.
5. **Chefer, H., Alaluf, Y., Vinker, Y., Wolf, L., & Cohen-Or, D.** (2023). "Attend-and-Excite: Attention-Based Semantic Guidance for Text-to-Image Diffusion Models." *ACM SIGGRAPH 2023*.
6. **Rassin, R., Hirsch, E., Glickman, D., Ravfogel, S., Goldberg, Y., & Chechik, G.** (2024). "Linguistic Binding in Diffusion Models: Enhancing Attribute Correspondence through Attention Map Alignment." *NeurIPS 2024*.
7. **Li, Y., Liu, H., Wu, Q., Mu, F., Yang, J., Gao, J., Li, C., & Lee, Y. J.** (2023). "GLIGEN: Open-Set Grounded Text-to-Image Generation." *CVPR 2023*.
8. **Huang, K., Sun, K., Xie, E., Li, Z., & Liu, X.** (2023). "T2I-CompBench: A Comprehensive Benchmark for Open-world Compositional Text-to-Image Generation." *NeurIPS 2023*.
9. **Saharia, C., Chan, W., Saxena, S., et al.** (2022). "Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding." *NeurIPS 2022*. arXiv:2205.11487.
10. **Ho, J. & Salimans, T.** (2022). "Classifier-Free Diffusion Guidance." arXiv:2207.12598.
11. **Ramesh, A., Dhariwal, P., Nichol, A., Chu, C., & Chen, M.** (2022). "Hierarchical Text-Conditional Image Generation with CLIP Latents." arXiv:2204.06125.
12. **Yang, Z., Liu, D., Wang, C., Yang, J., & Tao, D.** (2022). "Modeling Image Composition for Complex Scene Generation." *CVPR 2022*.
