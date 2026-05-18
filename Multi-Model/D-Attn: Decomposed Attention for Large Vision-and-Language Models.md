
# D-Attn: Decomposed Attention for Large Vision-and-Language Models 

> **논문 정보**
> - **제목:** D-Attn: Decomposed Attention for Large Vision-and-Language Models
> - **저자:** Chia-Wen Kuo, Sijie Zhu, Fan Chen, Xiaohui Shen, Longyin Wen (ByteDance)
> - **arXiv:** [2502.01906](https://arxiv.org/abs/2502.01906) (2025.02.04 제출, 2025.08.15 업데이트)
> - **학회:** ICCV 2025
> - **코드:** [github.com/bytedance/DecomposedAttention](https://github.com/bytedance/DecomposedAttention)

---

## 1. 핵심 주장 및 주요 기여 요약

### 🎯 핵심 주장

기존의 Large Vision-and-Language Models(LVLMs)는 시각 토큰과 텍스트 토큰을 LLM의 동질적(homogeneous) 입력으로 처리해 왔으나, 이 두 입력은 근본적으로 다르다: 시각 입력은 다차원적이고 문맥이 풍부하며, 보통 CLIP 같은 모델로 사전 인코딩된 반면, 텍스트 입력은 이러한 구조를 갖지 않는다.

이러한 제약된 아키텍처는 시각 토큰을 처리하는 설계 공간을 제한하며, 이는 최적이 아닌 성능과 비효율성으로 이어질 수 있다.

### ✅ 주요 기여 (4가지)

D-Attn은 시각 입력을 보다 효율적이고 효과적으로 처리하기 위해 설계된 프레임워크로, **(1)** 1D Causal Self-Attention을 V2V Self-Attn, T2V Cross-Attn, T2T Self-Attn 세 구성 요소로 분해하고, **(2)** V2V Self-Attn을 대각화하여 계산 복잡도를 줄이며, **(3)** T2V Cross-Attn에서 편향된 위치 인코딩을 제거하고, **(4)** α-weighting 전략으로 시각·텍스트 정보를 병합한다.

광범위한 실험 및 분석을 통해 D-Attn은 여러 이미지 벤치마크에서 상당한 성능 향상을 보이면서, 동시에 계산 비용도 크게 절감함(예: **5배 빠른 속도**)을 검증하였다.

---

## 2. 해결하고자 하는 문제, 제안 방법, 모델 구조, 성능 및 한계

### 🔴 2-1. 해결하고자 하는 문제

기존 LVLM들은 시각 및 텍스트 토큰을 LLM의 단일 동질 입력으로 연결(concatenation)함으로써 사전 학습된 언어 능력을 최대한 보존해 왔다. 그러나 이러한 제한된 아키텍처는 시각 토큰 처리를 위한 설계 공간을 좁히며, 잠재적으로 최적 이하의 성능과 비효율성을 초래한다.

Chameleon, Molmo, Phi-3.5-Vision 등 기존 연구들은 decoder-only LLM 아키텍처에서 인과적 self-attention을 사용하여 시각·텍스트 토큰을 동등하게 처리하며 LLM의 사전 학습 능력을 최대한 보존하여 우수한 성능을 냈지만, 사전 학습된 LLM의 언어 능력을 저하시킬 수 있는 맞춤형 설계를 적용하는 데 유연성이 제한된다.

구체적으로 해결해야 할 두 가지 핵심 문제는:

1. **위치 인코딩 편향 (Positional Encoding Bias):** 시각 토큰과 텍스트 토큰을 1D 시퀀스로 연결할 때 바람직하지 않은 위치 편향이 발생하며, D-Attn은 T2V Cross-Attn 내에서 회전/상대적 위치 인코딩(RoPE)을 제거하는 방식으로 이를 해결한다. 이 수정은 제안하는 어텐션 분해 프레임워크 없이는 기존 LVLM에 쉽게 적용하기 어렵다.

2. **V2V 어텐션의 이차 복잡도 ( $O(|V|^2)$ ):** V2V 어텐션을 대각화(diagonalize)하여 $|V|$개의 시각 토큰에 대한 계산 복잡도를 $O(|V|^2)$에서 $O(|V|)$로 줄이면서 성능은 저하시키지 않는다.

---

### 🔵 2-2. 제안하는 방법 (수식 포함)

#### Step 1: Causal Self-Attention 분해

D-Attn은 LVLM에서의 Causal Self-Attention 메커니즘을 **(1) visual-to-visual self-attention (V2V Self-Attn), (2) textual-to-visual cross-attention (T2V Cross-Attn), (3) textual-to-textual self-attention (T2T Self-Attn)** 세 가지로 분해한다.

전체 입력 시퀀스를 시각 토큰 집합 $V = \{v_1, \dots, v_m\}$과 텍스트 토큰 집합 $T = \{t_1, \dots, t_n\}$으로 표현하면, 표준 Causal Self-Attention은 다음과 같다:

$$
\text{Attn}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\!\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}}\right)\mathbf{V}
$$

시각 토큰 $v$는 오직 다른 시각 토큰 $V$에만 어텐션하며, 텍스트 토큰의 어텐션 출력은 T2V Cross-Attn과 T2T Self-Attn 간의 상호작용을 나타낸다.

이를 블록 행렬 형태로 분리하면:

**시각 토큰 출력 (V2V Self-Attn):**

$$
\bar{v} = \text{Attn}^{V2V}(Q_V, K_V, V_V) = \text{softmax}\!\left(\frac{Q_V K_V^\top}{\sqrt{d_k}}\right) V_V
$$

**텍스트 토큰 출력 ($\alpha$-weighting을 통한 T2V XA + T2T SA 병합):**

$$
\bar{t} = \alpha \cdot \text{Attn}^{T2V}(Q_T, K_V, V_V) + (1-\alpha) \cdot \text{Attn}^{T2T}(Q_T, K_T, V_T)
$$

여기서 $\alpha$는 두 어텐션 구성 요소의 상대적 기여를 조절하는 가중치이다.

#### Step 2: V2V 대각화 (Diagonal-Attn)

각 시각 임베딩은 이미 다른 시각 임베딩에 대한 문맥 정보를 인코딩하고 있으므로, LVLM 내에서 이 정보를 다시 학습하는 것은 중복(redundant)이다.

이를 반영하여 V2V 어텐션 행렬을 대각 행렬로 근사화:

$$
\text{Attn}^{V2V}_{diag} = \text{diag}(a_1, a_2, \dots, a_m)
$$

이는 Softmax 계산을 생략하고, 각 시각 토큰이 자기 자신에만 어텐션하도록 하여:

$$
\mathcal{O}(|V|^2) \;\longrightarrow\; \mathcal{O}(|V|)
$$

V2V Diagonal-Attn은 특히 고해상도 이미지나 긴 비디오 입력처럼 시각 토큰 수 $|V|$가 클 때 특히 유용하며, 실험을 통해 full attention과 유사한 성능을 유지하면서도 상당한 계산 절감을 달성함을 보였다.

#### Step 3: T2V 위치 인코딩 디바이어싱

T2V Cross-Attn에서 위치 편향을 제거(debiasing)함으로써, 편향된 위치 인코딩을 가진 모델 대비 여러 이미지 벤치마크에서 일관된 성능 향상을 달성하였다. 이 수정은 기존 LVLM에서는 쉽게 구현할 수 없지만, 제안하는 어텐션 분해를 통해서는 비교적 간단히 구현된다.

#### Step 4: $\alpha$-Weighting 전략

$\alpha$-weighting 전략은 T2V Cross-Attn의 시각 정보와 T2T Self-Attn의 텍스트 정보를 병합하며, 이 접근법은 LVLM의 내재적 어텐션 연산과 해석학적으로 동등하여 추가 학습 파라미터 없이 최소한의 아키텍처 변경만 도입한다. 이를 통해 사전 학습된 LLM은 경쟁력 있는 다운스트림 성능을 위한 완전한 능력을 유지한다.

---

### 🟢 2-3. 모델 구조

D-Attn 모델은 LLaVA의 아키텍처를 기반으로 구축되었으며, 사전 학습된 SigLIP 시각 인코더와 무작위 초기화된 2-레이어 MLP 어댑터로 구성된다.

D-Attn을 구현하기 위해 LLM 내의 decoder layer 및 self-attention 메커니즘만 수정한다.

구조 요약:

| 구성 요소 | 설명 |
|---|---|
| **Vision Encoder** | SigLIP (사전 학습된 시각 인코더) |
| **Adapter** | 2-layer MLP (무작위 초기화) |
| **LLM Backbone** | 사전 학습된 LLM (decoder-only) |
| **어텐션 수정 범위** | Decoder layer의 Self-Attention만 수정 |
| **V2V Attn** | Diagonal-Attn (O(V^2)) |
| **T2V Attn** | Cross-Attn (debiased RoPE) |
| **T2T Attn** | 기존 Self-Attn 유지 |
| **병합 전략** | $\alpha$ -weighting |

---

### 🟡 2-4. 성능 향상

광범위한 실험과 엄밀한 분석을 통해 D-Attn은 S-Attn(Standard Attention) 대응 모델 대비 일관되게 우수한 성능을 보이며, 성능 향상과 실질적인 계산 절감 모두를 제공한다. 이 연구의 기여는 시각 및 텍스트 입력을 더 높은 유연성으로 별도 처리하는 것의 중요성을 강조하며, 보다 효율적이고 효과적인 LVLM으로 가는 길을 열어 준다.

D-Attn 모델은 OCR 및 문서 관련 작업에서도 강력한 성능을 보인다.

주요 성능 수치 (논문 보고 기준):

| 개선 항목 | 결과 |
|---|-----------|
| 추론 속도 | **최대 5배** 향상 |
| V2V 복잡도 감소 | O(V^2) -> O(V) |
| 다중 이미지 벤치마크 | S-Attn 대비 일관된 향상 |

---

### 🔴 2-5. 한계

논문 내에서 명시적으로 서술된 한계는 다음과 같이 정리할 수 있다:

1. X-Attn 아키텍처는 IDEFICS-2에서 decoder-only S-Attn 아키텍처보다 성능이 낮음이 보고되어 폐기되었다. X-Attn은 시각 토큰을 유연하게 처리할 수 있지만 성능이 열등하다는 한계가 있으며, D-Attn은 이를 극복하고자 한다.

2. **V2V Diagonal Attn의 근사:** 대각화를 통해 V2V 어텐션을 근사하는 방식은 일부 시각 토큰 간 상호 관계(global context)를 희생할 수 있으며, 매우 복잡한 장면 이해 태스크에서는 미세한 성능 손실 가능성이 존재한다.

3. **비디오 등 다른 모달리티 확장성:** 현재 주요 평가는 이미지 중심이며, 비디오나 오디오 등 다양한 모달리티에 대한 일반화 성능은 추가 검증이 필요하다.

4. **$\alpha$ 값의 결정:** $\alpha$-weighting의 최적값이 태스크나 도메인에 따라 달라질 수 있으며, 동적으로 학습하는 메커니즘이 필요할 수 있다.

---

## 3. 모델의 일반화 성능 향상 가능성

### 3-1. 위치 편향 제거를 통한 일반화

기존 LVLM에서의 **Attention Shift** 문제: 텍스트 어텐션이 시각 토큰 시퀀스의 나중 부분에 집중하려는 경향이 있으며, 이는 LLM 내 단방향 정보 흐름과 토큰 간 상대적 위치 관계라는 두 가지 요인에 기인하며, 텍스트 프롬프트가 시퀀스 후반부에 위치한 시각 토큰에 우선적으로 집중하도록 만든다.

D-Attn의 T2V Cross-Attn에서 RoPE를 제거함으로써 이 편향을 원천적으로 차단하여, **특정 이미지 레이아웃이나 해상도에 편향되지 않은** 일반화된 시각 이해를 가능하게 한다.

### 3-2. 사전 학습 LLM 능력 보존

D-Attn은 Causal Self-Attention 메커니즘에 아키텍처적·운영적 변경을 도입하지 않으며, 이는 사전 학습된 LLM의 능력을 보존하는 데 결정적으로 중요하여 우수한 시각 이해 성능으로 이어진다.

즉, LLM의 광범위한 언어 일반화 능력을 그대로 유지하면서 시각 처리만 개선하는 방식은 **도메인 일반화(domain generalization)** 측면에서 매우 유리하다.

### 3-3. 고해상도 및 비디오 입력에서의 확장성

V2V Diagonal-Attn은 고해상도 이미지나 긴 비디오 입력에서 시각 토큰 수 $|V|$가 커질수록 특히 유용하며, full attention과 유사한 성능을 유지하면서 상당한 계산 절감을 달성한다.

이는 실제 응용 환경에서 더 다양한 입력 해상도와 모달리티에 대한 일반화를 가능하게 한다.

### 3-4. 프레임워크의 범용성

D-Attn은 텍스트-텍스트 어텐션에 영향을 주지 않으면서 시각 토큰 연산을 수정할 수 있게 해 주는 LVLMs를 위한 보다 유연한 어텐션 아키텍처로, **다양한 LVLM 백본에 범용적으로 적용**할 수 있는 일반적인 프레임워크이다.

---

## 4. 앞으로의 연구에 미치는 영향 및 고려할 점

### 4-1. 연구에 미치는 영향

#### (1) 멀티모달 토큰 처리 패러다임의 전환

D-Attn은 LVLM 내에서 시각·텍스트 임베딩을 다르게 처리하도록 설계된 새로운 범용 프레임워크로, V2V Self-Attn 대각화를 통해 계산 복잡도를 줄이고, T2V Cross-Attn 디바이어싱을 통해 모델 성능을 개선하며, $\alpha$-weighting으로 사전 학습된 LLM의 능력을 최소한의 변경으로 보존한다.

이는 기존의 "시각·텍스트 동질 처리(homogeneous processing)" 패러다임에서 **"모달리티 인식 이종 처리(modality-aware heterogeneous processing)"** 패러다임으로의 전환을 촉진한다.

#### (2) 효율적 LVLM 설계에 대한 방향성 제시

원래의 self-attention을 시각-시각, 시각-텍스트, 텍스트-텍스트 어텐션의 세 부분으로 분해하고, V2V Self-Attn 대각화를 통해 $O(|V|^2)$에서 $O(|V|)$로 복잡도를 줄이는 광범위한 실험 결과는, 시각·텍스트 입력을 더 높은 유연성으로 별도 처리하는 것의 중요성을 강조하며 보다 효율적이고 효과적인 LVLM을 향한 길을 열어 준다.

#### (3) 관련 후속 연구와의 연결 - DeAR (2026)

태스크 특화 지식이 모델의 핵심 일반화 능력을 저하시키고 태스크 적응과 제로샷 일반화 보존 사이의 트레이드오프를 야기할 수 있다. 이를 해결하기 위해 DeAR(2026)은 레이어 중심 관점을 탈피하여 어텐션 헤드 역할을 분해함으로써 세밀한 VLM 적응을 달성하는 프레임워크를 제안하며, 이 기능적 특화는 레이어 간이 아니라 심층 레이어의 개별 어텐션 헤드 수준에서 발생한다고 주장한다.

이처럼 D-Attn의 "어텐션 분해" 아이디어는 이후 연구들에 직접적인 영향을 주고 있다.

---

### 4-2. 2020년 이후 관련 최신 연구 비교 분석

| 연구 | 핵심 방법 | D-Attn과의 관계 |
|---|---|---|
| **LLaVA (2023)** | 시각·텍스트 토큰 단순 연결 + Causal SA | D-Attn의 기반 아키텍처 |
| **IDEFICS-2 (2024)** | Cross-Attention vs Self-Attention 비교 | X-Attn의 성능 열등성 문제 제기 → D-Attn이 이를 극복 |
| **FastV (2024)** | 시각 토큰 중요도 기반 가지치기(pruning) | LLM 깊은 레이어에서 시각 토큰이 텍스트보다 훨씬 적은 어텐션을 받는 "비효율적 시각 어텐션" 현상을 발견하고, 레이어 2 이후 어텐션 점수 기반으로 하위 R%의 시각 토큰을 가지치기하는 방법을 제안한다. D-Attn은 근본 원인을 구조적으로 해결 |
| **FasterVLM (2024)** | CLS 토큰 기반 시각 토큰 중요도 평가 | LLM 내 텍스트-시각 어텐션이 실제 시각 토큰의 중요도와 잘 맞지 않아 높은 축소 비율에서 심각한 성능 저하를 초래한다는 문제를 지적한다. D-Attn의 디바이어싱과 직접 연결 |
| **MHA2MLA-VLM (2026)** | Multi-Head Latent Attention으로 KV 캐시 압축 | 모달리티 적응형 partial-RoPE 전략과 모달리티 분리 저랭크 근사 방법을 사용하여 시각·텍스트 KV 공간을 독립적으로 압축한다. D-Attn의 모달리티 분리 처리 방향과 일맥상통 |
| **DeAR (2026)** | 어텐션 헤드 역할 분해를 통한 세밀한 VLM 적응 | Concept Entropy를 통해 어텐션 헤드를 Attribute, Generalization, Mixed 역할로 분류하고, Role-Based Attention Mask 메커니즘으로 정보 흐름을 제어한다. D-Attn의 어텐션 분해 사상의 발전 |

---

### 4-3. 앞으로 연구 시 고려할 점

1. **다양한 모달리티로의 확장:**
   V2V Diagonal-Attn은 고해상도 이미지나 긴 비디오 입력에서 시각 토큰 수 $|V|$가 클 때 특히 유용하다는 점에서, **비디오 LVLM, 오디오-비주얼 모델** 등 다양한 모달리티에서의 성능 검증이 필요하다.

2. **$\alpha$ 값의 동적 최적화:**
   현재 $\alpha$-weighting은 정적(static)이거나 제한적으로 학습된다. 태스크 또는 입력에 따라 $\alpha$를 동적으로 조정하는 **Adaptive $\alpha$-weighting** 메커니즘 연구가 중요하다.

3. **대각화 근사의 이론적 분석:**
   V2V Diagonal-Attn은 시각 토큰 간 전역 문맥을 희생하는 근사이므로, **어떤 조건에서 이 근사가 유효한지** (예: 밀집 장면 이해, 소수 객체 장면 등)에 대한 이론적·실험적 분석이 필요하다.

4. **다른 위치 인코딩 방식과의 호환성:**
   RoPE 외에 ALiBi, NoPE 등 다양한 위치 인코딩 스킴에서도 디바이어싱 효과가 동일하게 적용되는지 검토해야 한다.

5. $\alpha$-weighting 접근법은 LVLM의 내재적 어텐션 연산과 해석학적으로 동등하며, 사전 학습된 LLM이 경쟁력 있는 성능을 위한 완전한 능력을 유지하도록 보장한다는 점에서, **다양한 LLM 백본(Llama, Mistral, Qwen 등)에서의 범용성 검증**이 중요한 향후 연구 방향이다.

---

## 📚 참고 자료 (출처)

| 번호 | 제목 / 출처 |
|---|---|
| 1 | **D-Attn: Decomposed Attention for Large Vision-and-Language Models** — arXiv:2502.01906, Chia-Wen Kuo et al. (ByteDance), ICCV 2025. https://arxiv.org/abs/2502.01906 |
| 2 | **D-Attn ICCV 2025 Open Access Paper** — openaccess.thecvf.com/content/ICCV2025/papers/Kuo_D-Attn... |
| 3 | **D-Attn HTML (arXiv)** — https://arxiv.org/html/2502.01906 |
| 4 | **D-Attn v1 HTML (arXiv)** — https://arxiv.org/html/2502.01906v1 |
| 5 | **GitHub: bytedance/DecomposedAttention** — https://github.com/bytedance/DecomposedAttention |
| 6 | **DeAR: Fine-Grained VLM Adaptation by Decomposing Attention Head Roles** — arXiv:2603.01111, Yiming Ma et al. (2026). https://arxiv.org/abs/2603.01111 |
| 7 | **[CLS] Attention is All You Need for Training-Free Visual Token Pruning (FasterVLM)** — arXiv:2412.01818, Qizhe Zhang et al. (2024). https://arxiv.org/html/2412.01818v1 |
| 8 | **MHA2MLA-VLM: Enabling DeepSeek's Economical Multi-Head Latent Attention across Vision-Language Models** — arXiv:2601.11464 (2026). https://arxiv.org/abs/2601.11464 |
| 9 | **Large Vision–Language Models Get Lost in Attention** — arXiv:2605.05668 (2025). https://arxiv.org/html/2605.05668v1 |
