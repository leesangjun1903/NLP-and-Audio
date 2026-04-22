
# Training-Free Consistent Text-to-Image Generation

---

## 1. 핵심 주장 및 주요 기여 요약

### 🔑 핵심 주장

텍스트-이미지 생성 모델은 자연어를 통해 이미지 생성 과정을 안내할 수 있는 높은 수준의 창의적 자유를 제공하지만, 다양한 프롬프트에 걸쳐 동일한 피사체(subject)를 일관성 있게 묘사하는 것은 여전히 어려운 과제이다.

이에 대해 본 논문은 **ConsiStory**를 제안한다 — 사전학습된 모델의 내부 활성화(internal activations)를 공유함으로써 일관된 피사체 생성을 가능케 하는 **학습 없는(training-free)** 접근법이다.

### ✅ 주요 기여

논문의 주요 기여는 세 가지로 요약된다:
1. 다양한 프롬프트에 걸쳐 피사체 일관성을 달성하는 **학습 없는 방법** 제시
2. Extended Attention 적용 시 발생하는 **레이아웃 붕괴(layout collapse)** 문제를 해결하는 새로운 기법 개발
3. 일관성 평가를 위한 **새로운 벤치마크 데이터셋** 공유

ConsiStory는 단 하나의 최적화 단계도 필요로 하지 않으면서 피사체 일관성과 텍스트 정렬에서 최신 기술(state-of-the-art) 수준의 성능을 보여주며, 멀티 피사체 시나리오로 자연스럽게 확장되고 일반 객체에 대한 학습 없는 개인화(training-free personalization)도 가능하다.

---

## 2. 해결 문제 / 제안 방법 / 모델 구조 / 성능 및 한계

### 2.1 해결하고자 하는 문제 (Problem Statement)

기존 접근법들은 모델을 파인튜닝하여 특정 피사체를 나타내는 새 단어를 학습시키거나, 이미지 컨디셔닝을 추가하는 방식을 취한다. 이러한 방법들은 피사체별로 긴 최적화 시간이나 대규모 사전학습이 필요하다. 뿐만 아니라 생성 이미지를 텍스트 프롬프트에 맞게 정렬하는 데 어려움을 겪고, 다중 피사체 묘사에도 한계가 있다.

즉, 기존 방법들은 다음 세 가지 트레이드오프를 동시에 해결하지 못했다:

| 문제 | 기존 방법의 한계 |
|---|---|
| 피사체 일관성 | Fine-tuning 필요 (Textual Inversion, DreamBooth) |
| 텍스트 정렬 | Encoder-based 방법(IP-Adapter 등)이 외형 과적합 |
| 다중 피사체 | 대부분의 방법이 단일 피사체에만 특화 |

---

### 2.2 제안하는 방법 및 수식

ConsiStory는 세 단계로 작동한다: (1) **피사체 주도 확장 자기 어텐션(SDSA)** — 교차 어텐션 맵을 레이어와 타임스텝에 걸쳐 집계하여 생성된 이미지에서 피사체를 국소화하며, 각 이미지가 다른 프레임에 존재하는 주요 피사체의 패치에 어텐션할 수 있도록 한다.

#### 📐 (1) Subject Mask 생성

Cross-attention 맵 $A^{(l,t)}$를 레이어 $l$, 타임스텝 $t$에 걸쳐 집계하여 피사체 마스크를 생성한다:

$$M_i = \text{Binarize}\left(\sum_{l,t} A_i^{(l,t)}[\text{subject token}]\right)$$

여기서 $M_i$는 이미지 $I_i$에서의 피사체 영역을 나타내는 이진 마스크이다.

#### 📐 (2) Subject-Driven Self-Attention (SDSA)

SDSA는 자기 어텐션 레이어를 확장하여, 생성 이미지 $I_i$의 Query가 다른 이미지 $I_j$ ($j \neq i$)의 Key에 접근할 수 있게 하며, 이는 피사체 마스크 $M_j$로 제한된다.

표준 Self-Attention 연산:

$$\text{Attn}(Q_i, K_i, V_i) = \text{softmax}\left(\frac{Q_i K_i^\top}{\sqrt{d}}\right)V_i$$

이를 확장한 SDSA:

$$\tilde{K}_i = \text{concat}(K_i,\; \{K_j \odot M_j\}_{j \neq i})$$
$$\tilde{V}_i = \text{concat}(V_i,\; \{V_j \odot M_j\}_{j \neq i})$$

$$\text{SDSA}(Q_i, \tilde{K}_i, \tilde{V}_i) = \text{softmax}\left(\frac{Q_i \tilde{K}_i^\top}{\sqrt{d}}\right)\tilde{V}_i$$

#### 📐 (3) Layout Diversity 강화

다양성 증진을 위해 두 가지 기법을 사용한다: (1) dropout으로 SDSA를 약화시키고, (2) 비일관 샘플링 단계의 vanilla Query 특징과 블렌딩하여 $Q^*$를 생성한다.

혼합된 Query는 다음과 같이 정의된다:

$$Q^* = \lambda \cdot Q_{\text{consistent}} + (1 - \lambda) \cdot Q_{\text{vanilla}}$$

여기서 $\lambda \in [0, 1]$는 일관성과 다양성 간의 균형을 조정하는 파라미터이다.

#### 📐 (4) Feature Injection (DIFT 기반)

피사체의 정체성을 이미지 전반에 걸쳐 더욱 세밀하게 보강하기 위해, 배치 내 특징 블렌딩 메커니즘을 도입한다. 각 이미지 쌍 사이의 패치 대응 맵을 추출하고, 이 맵에 기반하여 이미지 간 특징을 주입한다.

패치 대응 맵 $C_{ij}$를 DIFT(Diffusion Features)로 추출하고:

$$\hat{F}_i = \alpha \cdot F_i + (1 - \alpha) \sum_{j \neq i} C_{ij} \cdot F_j$$

특징 주입은 타임스텝 $\alpha = 0.8$에서 적용되며, 유사도 점수가 Otsu 방법으로 자동 설정된 임계값 이상인 패치에만 주입을 수행한다.

---

### 2.3 모델 구조

주어진 프롬프트 집합에 대해, 매 생성 단계마다 각 생성 이미지 $I_i$에서 피사체를 국소화한다. 현재 생성 단계까지의 교차 어텐션 맵을 활용하여 피사체 마스크 $M_i$를 생성하고, U-Net 디코더의 표준 자기 어텐션 레이어를 피사체 주도 자기 어텐션 레이어(Subject Driven Self-Attention)로 대체하여 피사체 인스턴스 간 정보를 공유한다. 또한 추가적인 세밀화를 위해 Feature Injection을 적용한다.

```
[입력 프롬프트 배치: P₁, P₂, ..., Pₙ]
          ↓
[노이즈로부터 병렬 생성 시작]
          ↓
[Cross-Attention 맵 집계 → Subject Mask M_i 생성]
          ↓
[U-Net Decoder Self-Attention → SDSA로 대체]
   - K, V를 마스크된 영역에서 공유
   - Dropout + Query 블렌딩으로 다양성 유지
          ↓
[DIFT 패치 대응 기반 Feature Injection]
          ↓
[일관성 있는 이미지 집합: I₁, I₂, ..., Iₙ]
```

Subject-Driven Self-Attention은 U-Net의 디코더 레이어에서 모든 타임스텝에 걸쳐 적용된다.

---

### 2.4 성능 향상

#### 속도 비교

런타임 분석에서 ConsiStory는 H100 GPU에서 앵커 2개 생성 및 새 프롬프트 기반 이미지 1개 생성에 32초로 가장 빠른 결과를 달성했으며, 이는 Avrahami et al. (2023b)의 최신 접근법(H100에서 약 13분 소요)보다 약 25배 빠르다.

또한 LoRA-DB(4.5분)보다 8~14배, TI(7.5분)보다 훨씬 빠르다.

#### 정량적 평가

CLIP 및 DreamSim 점수를 이용한 실험적 평가와 사용자 연구에서 프롬프트 정렬 및 다중 피사체 시나리오 확장성 면에서 우수한 성능을 확인했다.

ConsiStory(녹색)는 피사체 일관성과 텍스트 유사성 간 최적의 균형을 달성했다. ELITE, IP-Adapter와 같은 인코더 기반 방법은 시각적 외형에 과적합되는 반면, LoRA-DB, TI 등 최적화 기반 방법은 ConsiStory만큼 높은 피사체 일관성을 보이지 못한다.

#### Ablation Study

SDSA, Feature Injection, Self-Attention Dropout, Query Injection 등 다양한 컴포넌트를 ablation한 결과, SDSA나 FI를 제거하면 피사체 일관성이 감소하고, Dropout 및 Query Injection 제거 시 텍스트 유사성이 감소하는 것을 확인했다.

---

### 2.5 한계점 (Limitations)

모델은 피사체의 정체성을 정의하는 앵커 프롬프트가 단순할수록 더 잘 작동하며, 이후 프롬프트는 더 복잡해도 된다는 경향이 있다.

추가적인 알려진 한계:
- SDSA만 단독으로 적용하면 레이아웃 다양성이 감소하는 문제가 있으며, 이를 해결하기 위해 레이아웃 다양성 기법(비일관 샘플링 단계의 Q 특징 혼합, 추론 시간 드롭아웃)이 필요하다.
- ConsiStory는 분류기 자유 안내(classifier-free guidance)의 조건부 브랜치와 비조건부 브랜치 모두에 자기 어텐션 메커니즘을 적용하는데, 이는 효과적인 정체성 전달을 상당히 방해할 수 있다는 후속 연구의 지적이 있다.
- Textual Inversion, PhotoMaker, ConsiStory, StoryDiffusion 등의 방법은 드래곤(dragon)과 같은 객체에 대한 정체성 일관성 유지에 어려움이 있다는 비교 연구 결과도 있다.

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 일반화의 핵심 메커니즘

ConsiStory는 사전학습된 T2I 모델을 튜닝하거나 개인화하지 않으며, 대신 크로스 프레임 특징 공유(cross-frame feature sharing)를 활용하여 생성 중 피사체 일관성을 촉진한다.

이 설계 철학 자체가 일반화의 핵심이다. **어떤 특정 피사체에도 과적합되지 않고**, 사전학습 모델이 학습한 방대한 지식을 그대로 활용하기 때문이다.

### 3.2 멀티 피사체 시나리오로의 확장

ConsiStory는 반복되는 피사체가 있는 입력 프롬프트 집합을 동일한 피사체 정체성을 유지하고 텍스트를 따르는 이미지 시리즈로 변환하며, 다중 피사체에 대한 일관된 정체성 유지도 가능하다. 중요하게도, ConsiStory는 어떠한 최적화나 사전학습도 포함하지 않는다.

### 3.3 ControlNet과의 통합을 통한 일반화

ConsiStory는 ControlNet과 통합되어 포즈 제어(pose control)가 있는 일관된 캐릭터 생성이 가능하다. 또한 피사체 당 실제 이미지 2장을 편집-친화적 역전(inversion)으로 반전시켜 앵커로 활용하며, 이를 학습 없는 개인화(training-free personalization)에 사용한다.

### 3.4 후속 연구들에서의 일반화 확장

StorySync와 같은 후속 연구는 ConsiStory의 설계 철학을 발전시켜 **모델 불가지론적(model-agnostic)** 접근을 채택하며, SDXL, Kandinsky 3, FLUX.1-schnell 등 다양한 최신 확산 모델과 추가 학습 없이 통합 가능함을 입증했다.

---

## 4. 향후 연구에 미치는 영향 및 고려할 점

### 4.1 연구에 미치는 영향

| 영향 영역 | 내용 |
|---|---|
| **패러다임 전환** | "학습 없는" 일관성 생성 패러다임을 확립하여 후속 연구의 기준점(baseline)을 제시 |
| **영상/비디오 확장** | 이미지 간 일관성 기법이 비디오 생성으로 자연스럽게 확장 가능함을 보여줌 |
| **스토리 생성 AI** | 시각 스토리텔링, 만화, 동화책 생성 등 응용 분야를 크게 확장 |
| **개인화 연구** | 학습 없는 개인화(training-free personalization) 연구의 방향성을 제시 |

대규모 T2I 확산 모델은 사용자가 텍스트로부터 상상력 넘치는 장면을 만들 수 있게 하지만, 확률론적 특성으로 인해 다양한 프롬프트에서 시각적으로 일관된 피사체를 묘사하는 데 어려움이 있다. 이러한 일관성은 책과 이야기 삽화, 가상 자산 디자인, 그래픽 노블 및 합성 데이터 생성 등 다양한 응용에 매우 중요하다.

이에 응답하여 후속 연구인 Infinite-Story는 학습 없는 프레임워크를 더욱 발전시켜 기존 가장 빠른 일관성 T2I 모델보다 6배 이상 빠른 추론(이미지당 1.72초)을 달성하며 실세계 적용 가능성을 높였다.

### 4.2 향후 연구 시 고려할 점

**① 배치 의존성 극복**
ConsiStory는 여러 이미지를 배치로 처리해야 하므로, 단일 이미지 생성 파이프라인에 직접 통합하기 어렵다. 단일 이미지에서도 일관성을 달성하는 연구가 필요하다.

이에 대해 1Prompt1Story는 "내재적 컨텍스트 일관성"에서 영감을 받아, 모든 프롬프트를 T2I 확산 모델의 단일 입력으로 연결하는 새로운 학습 없는 방법을 제안했다.

**② 텍스트 정렬과 일관성 간의 트레이드오프**
SDSA만 단독으로 적용하면 레이아웃 다양성이 감소하는 문제가 있어, 피사체 일관성을 높일수록 텍스트 프롬프트 준수도가 낮아지는 근본적인 트레이드오프가 발생한다. 이 균형을 개선하는 연구가 필요하다.

**③ Classifier-Free Guidance에서의 일관성**
ConsiStory는 조건부와 비조건부 브랜치 모두에 메커니즘을 적용하는데, 이는 효과적인 정체성 전달을 저해할 수 있다는 지적이 있다. 조건부 브랜치에만 선택적으로 적용하거나 각 브랜치에 맞게 메커니즘을 조정하는 연구가 필요하다.

**④ 비디오 생성으로의 확장**
비디오 생성을 위한 ConsiStory 적응에는 여러 도전이 존재하며, 단순히 이미지 기반 알고리즘을 비디오 생성에 적용하는 것은 운동(motion) 저하 및 샷 간 동기화 문제를 야기할 수 있다.

**⑤ 더 나은 평가 지표 개발**
일관성 T2I 생성에서 정체성 불일치(identity inconsistency)와 스타일 불일치(style inconsistency)라는 두 가지 핵심 도전을 측정하는 더 정교한 지표가 필요하다.

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

| 방법 | 년도 | 학습 필요 여부 | 다중 피사체 | 텍스트 정렬 | 속도 |
|---|---|---|---|---|---|
| **Textual Inversion** | 2022 | 피사체별 최적화 | 제한적 | 보통 | 느림(~7.5분) |
| **DreamBooth (LoRA)** | 2022 | 파인튜닝 필요 | 제한적 | 보통 | 느림(~4.5분) |
| **IP-Adapter** | 2023 | 대규모 사전학습 | 가능 | 낮음 | 빠름(~8초) |
| **MasaCtrl** | 2023 | 불필요 | 제한적 | 보통 | 빠름 |
| **ConsiStory** | 2024 | **불필요** | **가능** | **높음** | **빠름(32초)** |
| **1Prompt1Story** | 2025 | 불필요 | 가능 | 높음 | 빠름 |
| **Infinite-Story** | 2026 | 불필요 | 가능 | 높음 | 매우 빠름(1.72초/이미지) |

비교 결과, 일부 방법(TI)은 일관성 유지에 실패하거나(IP-Adapter) 프롬프트를 따르지 못하며, 다른 방법들(DB-LoRA)은 일관성 유지와 텍스트 준수 사이에서 교대로 어려움을 겪는다.

2026년에 AAAI에 발표된 Infinite-Story는 스케일별 자기회귀 모델 기반으로 정체성 일관성과 스타일 일관성이라는 두 핵심 도전을 해결하며, **Identity Prompt Replacement**, **Adaptive Style Injection**, **Synchronized Guidance Adaptation** 세 가지 기법을 도입해 ConsiStory의 한계를 보완했다.

---

## 📚 참고 자료 및 출처

1. **Tewel, Y. et al. (2024).** "Training-Free Consistent Text-to-Image Generation." *ACM Transactions on Graphics (SIGGRAPH 2024).*
   - arXiv: https://arxiv.org/abs/2402.03286
   - ACM DL: https://dl.acm.org/doi/10.1145/3658157
   - Project page: https://research.nvidia.com/labs/par/consistory/
   - GitHub: https://github.com/NVlabs/consistory

2. **Park, J. et al. (2026).** "Infinite-Story: A Training-Free Consistent Text-to-Image Generation." *Proceedings of the AAAI Conference on Artificial Intelligence, 40(10), 8278–8286.*
   - https://ojs.aaai.org/index.php/AAAI/article/view/37776

3. **StorySync: Training-Free Subject Consistency in Text-to-Image Generation via Region Harmonization.** (2025)
   - arXiv: https://arxiv.org/html/2508.03735v1

4. **"One-Prompt-One-Story: Free-Lunch Consistent Text-to-Image Generation Using a Single Prompt."** (2025)
   - arXiv: https://arxiv.org/abs/2501.13554
   - Hugging Face: https://huggingface.co/papers/2501.13554

5. **"Multi-Shot Character Consistency for Text-to-Video Generation."** (2024)
   - arXiv: https://arxiv.org/html/2412.07750v1

6. **Semantic Scholar — ConsiStory 인용 네트워크:**
   - https://www.semanticscholar.org/paper/39ba6d541d94132b816938e7e16b1e8fd49c2fd9

7. **Tang, L. et al. (2023).** "Emergent Correspondence from Image Diffusion (DIFT)." NeurIPS 2023.

8. **Cao, M. et al. (2023).** "MasaCtrl: Tuning-Free Mutual Self-Attention Control." ICCV 2023.
