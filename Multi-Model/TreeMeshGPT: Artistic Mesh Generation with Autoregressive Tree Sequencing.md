# TreeMeshGPT: Artistic Mesh Generation with Autoregressive Tree Sequencing

논문: **Lionar, S., Liang, J., Lee, G. H. (2025). TreeMeshGPT: Artistic Mesh Generation with Autoregressive Tree Sequencing. CVPR 2025, pp. 26608-26617.**

---

## 1. 핵심 주장 및 주요 기여 요약

TreeMeshGPT는 **점군(point cloud) 조건 하에 고품질 아티스틱 메쉬를 생성하는 자기회귀 Transformer**로, 기존의 next-token prediction 패러다임을 **Autoregressive Tree Sequencing**으로 대체합니다.

**3가지 주요 기여:**

1. **Autoregressive Tree Sequencing**: 다음 입력 토큰을 동적으로 성장하는 트리 구조에서 검색하여, 메쉬가 마지막 생성 면(face)으로부터 국소적으로 확장되도록 유도. 면 1개당 **2개의 토큰**만 사용 (naive 9 토큰 대비 약 22% 압축률).
2. **고용량·고충실도 생성**: 7비트 양자화로 최대 **5,500 face**, 9비트 모델은 최대 **11,000 face**까지 생성 가능 (기존 MeshAnything 800, MeshAnythingV2 1,600, EdgeRunner 4,000 대비 향상).
3. **법선 방향 일관성(normal orientation consistency)**: half-edge 자료구조의 반시계 방향 강제 덕분에 flipped normal 문제가 거의 발생하지 않음.

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

기존 자기회귀 메쉬 생성 방법들은 다음 한계를 가집니다.

- **MeshAnything**: 한 face를 9개의 잠재 토큰으로 표현 → 시퀀스 길이가 길어 800 face 이하로 제한.
- **MeshAnythingV2 / EdgeRunner**: triangle adjacency 활용으로 압축률 향상 (각각 1,600 / 4,000 face 지원), 그러나 여전히 복잡한 객체 표현에 부족.
- **공통 문제**: gap, 누락된 부품, **flipped normals** 등 결함 발생.

저자들은 **"토큰화 효율성"** 과 **"메쉬 품질(특히 manifold 및 normal 일관성)"** 이라는 두 축을 동시에 개선하고자 합니다.

### 2.2 제안 방법: Autoregressive Tree Sequencing

#### (a) 입력-출력 정의

입력은 **방향 있는 메쉬 엣지(directed edge)** 의 시퀀스로 정의됩니다.

$$I = \{(v_1^n, v_2^n)\}_{n=1}^{N} \in \mathbb{R}^{N \times 6}$$

각 단계에서 Transformer 디코더는 두 가지 중 하나를 예측합니다.
- 새로운 정점 $v_3^n$을 생성하여 입력 엣지 $(v_1^n, v_2^n)$과 함께 삼각형을 형성, 또는
- $\texttt{[STOP]}$ 라벨을 출력하여 해당 엣지에서 더 이상 확장하지 않음.

#### (b) Half-edge + DFS 기반 시퀀스 구성

새로운 삼각형이 형성되면, 두 개의 새 엣지 $(v_3^n, v_2^n)$ 와 $(v_1^n, v_3^n)$ 가 동적 스택 $S$에 푸시됩니다.

$$S := S \odot (v_1^n, v_3^n) \quad \text{and} \quad S := S \odot (v_3^n, v_2^n)$$

여기서 $\odot$는 스택의 top에 엣지를 푸시하는 연산입니다. 다음 단계 입력은 스택의 top에서 pop:

$$I_{n+1} = (v_1^{n+1}, v_2^{n+1}) := \text{top}(S), \quad S := S \setminus \text{top}(S)$$

이 방향성은 잠재적 인접 면(potential next adjacent face) 위에서 **반시계 방향(counter-clockwise)** 으로 강제되며, 이것이 normal 일관성을 보장하는 핵심 기제입니다.

최종 메쉬는 다음과 같이 구성됩니다.

$$\mathcal{M} = \bigcup_{n=1}^{N}(v_1^n, v_2^n, v_3^n)$$

(단, $I_n \notin \{\texttt{[SOS]}, \texttt{[SOS2]}\}$ 그리고 $o_n \notin \{\texttt{[STOP]}, \texttt{[EOS]}\}$ 인 단계만 해당.)

#### (c) 점군 조건 인코딩

8,192개 점 $X \in \mathbb{R}^{8192 \times 3}$를 cross-attention으로 잠재 코드 $Z$로 변환:

$$Z = \text{CrossAtt}\bigl(Q, \text{PosEmbed}(X)\bigr), \quad Q \in \mathbb{R}^{2048 \times C}, \; Z \in \mathbb{R}^{2048 \times L}$$

이 잠재 코드는 디코더의 첫 $\texttt{[SOS]}$ 토큰 앞에 prepend됩니다.

#### (d) 계층적 MLP 헤드 (Hierarchical MLP Heads)

기존 방법은 정점 $(x,y,z)$를 3개의 별도 토큰으로 예측하지만, 본 논문은 **단일 시퀀스 스텝 내에서 계층적으로** 예측합니다.

$$x \sim p(x \mid \text{prev}) = g_{\theta_1}(\mathbf{c})$$

$$y \sim p(y \mid x, \text{prev}) = g_{\theta_2}\bigl(E_x(x), \mathbf{c}\bigr)$$

$$z \sim p(z \mid y, x, \text{prev}) = g_{\theta_3}\bigl(E_y(y), E_x(x), \mathbf{c}\bigr)$$

여기서 $\mathbf{c} \in \mathbb{R}^{d}$는 디코더의 잠재 코드, $E_*$는 양자화된 좌표의 학습 가능한 임베딩입니다. (실험에서는 $z$ → $y$ → $x$ 순서로 예측하며, $\texttt{[STOP]}$과 $\texttt{[EOS]}$ 라벨은 $z$축의 클래스에 추가.)

#### (e) 학습 목적함수

전체 우도:

$$\prod_{n=1}^{N} P(o_n \mid I_{\leq n}; \theta)$$

손실 함수 (좌표축별 cross-entropy 합):

$$\mathcal{L} = \mathcal{L}_{CE}(O_x, \hat{O}_x) + \mathcal{L}_{CE}(O_y, \hat{O}_y) + \mathcal{L}_{CE}(O_z, \hat{O}_z)$$

#### (f) 평가 지표 (Normal Consistency)

면 중심 $c_i^s = \frac{v_{i_1}^s + v_{i_2}^s + v_{i_3}^s}{3}$ 에 대해 가장 가까운 reference 면 $j = \arg\min_{k \in M_r} d(c_i^s, F_k^r)$를 찾고, 코사인 유사도

$$\text{Sim}_{i \to j}(n^s, n^r) = \frac{n_i^s \cdot n_j^r}{\|n_i^s\|\,\|n_j^r\|}$$

를 양방향 평균하여:

$$NC = \frac{1}{2|M_s|}\sum_{i \in M_s} \text{Sim}_{i \to j}(n^s, n^r) + \frac{1}{2|M_r|}\sum_{k \in M_r} \text{Sim}_{k \to l}(n^r, n^s)$$

부호를 무시한 절댓값 버전:

$$|NC| = \frac{1}{2|M_s|}\sum_{i \in M_s} |\text{Sim}_{i \to j}(n^s, n^r)| + \frac{1}{2|M_r|}\sum_{k \in M_r} |\text{Sim}_{k \to l}(n^r, n^s)|$$

### 2.3 모델 구조 및 학습

| 항목 | 값 |
|---|---|
| Transformer 디코더 | 24 layer, 16 head, hidden dim 1024 |
| 위치 인코딩 | sinusoidal positional encoding |
| Attention | causal self-attention (FlexAttention 구현) |
| 양자화 | 7-bit (128 클래스), 보조 모델은 9-bit |
| 점군 조건 | 8,192 points → 2,048 latent tokens (cross-attention) |
| 옵티마이저 | AdamW, lr $=10^{-4}$, $\beta_1=0.9$, $\beta_2=0.99$ |
| 학습 | 8× A100-80GB, 5일 (7-bit) / 25일 (9-bit), batch=128 |
| 데이터 | Objaverse 75,000개 manifold 메쉬 |
| 샘플링 | top-k=5, 스택 길이 별 가변 temperature (1.0 → 0.4 → 0.2) |

### 2.4 성능 향상

**Objaverse 평가셋(200 샘플):**

| Model | CD ↓ | NC ↑ | \|NC\| ↑ |
|---|---|---|---|
| MeshAnything | 0.0115 | 0.223 | 0.853 |
| MeshAnythingV2 | 0.0102 | 0.167 | 0.843 |
| **TreeMeshGPT** | **0.0070** | **0.798** | **0.880** |

**GSO 데이터셋(실제 3D 스캔):**

| Model | CD ↓ | NC ↑ | \|NC\| ↑ |
|---|---|---|---|
| MeshAnything | 0.0105 | 0.453 | 0.869 |
| MeshAnythingV2 | 0.0116 | 0.327 | 0.865 |
| **TreeMeshGPT** | **0.0077** | **0.842** | **0.897** |

**토큰화 효과 통제 실험(≤500 face, 동일 학습조건):** TreeMeshGPT의 시퀀스 길이는 $2N_f + 2N_c$ ($N_f$=face 수, $N_c$=connected component 수)로, naive($9N_f$), VQ-VAE($6N_f$), AMT($\pm 4N_f$) 대비 가장 짧고, CD=0.0100으로 가장 우수합니다.

### 2.5 한계

- 시퀀스가 길어질수록 **성공률(success rate) 감소** (이전 방법들과 동일한 failure mode).
- 최적의 메쉬 토폴로지(아티스트 수준의 edge flow)를 강제하는 데에는 여전히 한계.
- Manifold 및 flipped normal 없는 메쉬만 학습 가능 → 데이터 필터링으로 학습 데이터가 75,000개로 제한.
- 9-bit 모델 학습에 25일 소요 → 자원 비용이 큼.

---

## 3. 일반화 성능 향상 가능성 (중점 분석)

### 3.1 논문에서 직접 입증된 일반화 능력

논문은 **GSO(Google Scanned Objects)** 데이터셋에서 일반화 성능을 명시적으로 검증합니다. GSO는 실제 3D 스캐너로 촬영된 가정용 물품 데이터셋으로, 학습 분포(Objaverse의 정제된 manifold 메쉬)와 명백히 다른 분포(out-of-distribution)에 해당합니다. 그럼에도 불구하고 TreeMeshGPT는 모든 지표에서 baseline을 상회합니다(CD=0.0077, NC=0.842).

또한 **Luma AI Genie text-to-3D 출력 → decimation → point cloud sampling → TreeMeshGPT**의 다단계 파이프라인을 통해 텍스트 프롬프트("knight helmet with horns", "cyberpunk car" 등)로부터 아티스틱 메쉬를 생성할 수 있음을 보여줍니다. 이는 학습 데이터에 없던 입력 도메인에 대해서도 모델이 작동함을 시사합니다.

### 3.2 일반화에 기여하는 구조적 요인

1. **국소적 예측(localized prediction)**: 매 스텝에서 모델은 "현재 엣지의 반대편 정점"이라는 매우 국소적인 결정만 내리면 됩니다. 이는 BFS 대비 DFS가 더 좋은 perplexity를 보인 ablation 결과와도 일치하며, **국소 의존성이 강한 학습 신호**가 더 일반화 가능한 inductive bias를 제공함을 의미합니다.

2. **강한 점군 조건(strong conditioning)**: 2,048개의 잠재 토큰으로 점군을 인코딩하여 디코더에 prepend함으로써, 생성 과정이 입력 형상에 **타이트하게 종속**됩니다. 이는 unseen 도메인의 점군이라도 형상 정보가 있으면 그에 맞춰 생성 가능하게 합니다.

3. **반시계 방향 강제(half-edge orientation)**: 학습된 normal 방향이 데이터에 우연히 의존하지 않고 **자료구조 수준에서 강제**되므로, 도메인이 바뀌어도 normal 일관성이 깨지지 않습니다. GSO에서 NC=0.842가 MeshAnythingV2의 0.327보다 압도적으로 높은 이유가 여기에 있습니다.

4. **계층적 MLP 헤드**: $z \to y \to x$ 순차 의존을 명시적으로 모델링하므로, 서로 독립적으로 좌표를 샘플링할 때 발생하는 노이즈가 줄어듭니다. ablation에서 simultaneous 예측은 CD=0.0114로 성능이 크게 떨어지는 것이 확인됩니다.

### 3.3 일반화 한계 및 향후 개선 여지

- **시퀀스 길이 의존적 실패**: 5,500 face 근처에서 success rate 감소는, attention의 long-range 의존성 학습 한계 때문일 가능성이 큽니다. Meshtron이 hourglass 구조 + sliding window inference로 64K face까지 확장한 것을 보면, 본 논문의 트리 시퀀싱과 효율적 architecture를 결합할 여지가 있습니다.
- **Manifold 가정**: half-edge 구조는 비-manifold 메쉬(예: T-junction, non-orientable surface)를 다룰 수 없어, 실제 산업 메쉬의 많은 부분이 학습/생성에서 제외됩니다.
- **Connected component 처리**: 새 컴포넌트마다 $\texttt{[SOS]}$, $\texttt{[SOS2]}$ 토큰이 필요하여, 컴포넌트가 매우 많은 메쉬(예: 식물, 머리카락)에서 토큰 효율이 떨어질 수 있습니다.

---

## 4. 향후 연구에 미치는 영향과 고려할 점

### 4.1 영향

1. **Tree-based sequencing 패러다임의 등장**: 1차원 시퀀스로 강제 직렬화하던 기존 패러다임에서 벗어나, **메쉬의 그래프적 본질**(인접성)을 시퀀스에 자연스럽게 반영하는 방향성을 제시했습니다. 이는 후속 연구(예: 2025년 7월 "Auto-Regressive Mesh Generation as Weaving Silk" arxiv:2507.02477에서 TreeMeshGPT를 비교군으로 다룸)에 직접 영향을 주고 있습니다.

2. **Normal 일관성을 자료구조로 강제하는 설계**: ML 모델의 약점(normal flipping)을 학습이 아닌 **구조적 제약(half-edge)** 으로 해결한 사례로, 이후 메쉬 생성 모델 설계의 표준 베스트 프랙티스가 될 가능성이 큽니다.

3. **계층적 좌표 예측**: 다른 자기회귀 3D 생성 모델에도 쉽게 이식 가능한 일반적 기법으로, 토큰 효율과 샘플링 안정성의 trade-off를 해결합니다.

### 4.2 향후 연구 시 고려할 점

1. **Non-manifold 메쉬 처리**: half-edge 자료구조에 의존하는 한, 실제 산업 메쉬의 상당수를 학습에서 제외해야 합니다. 일반화된 그래프 구조(예: dual graph, simplicial complex) 기반의 시퀀싱 연구가 필요합니다.

2. **시퀀스 길이의 효율적 확장**: TreeMeshGPT의 5,500 face는 BPT(>8K)나 Meshtron(64K)과 비교하면 여전히 작습니다. **트리 시퀀싱의 효율성 + sliding window/hourglass 구조의 확장성**을 결합한 하이브리드 연구가 유망합니다.

3. **토폴로지 품질 강제**: 저자도 한계로 인정한 부분으로, "아티스트의 edge flow"는 단순한 connectivity가 아니라 의미적 흐름(symmetry, curvature alignment)을 따릅니다. 이를 강제하려면 추가적인 손실(예: 곡률-인지 손실, 대칭성 손실) 또는 multi-stage 정제(refinement)가 필요할 것입니다.

4. **다양한 조건 입력**: 현재는 점군 조건에 국한되어 있으며, 이미지/텍스트로부터의 직접 조건은 외부 모델(Luma Genie)에 의존합니다. End-to-end 멀티모달 조건 통합이 다음 단계의 자연스러운 방향입니다.

5. **재현성 및 자원 효율**: 25일 / 8× A100-80GB의 학습 비용은 학계 재현이 어렵습니다. 더 효율적인 architecture(예: linear attention, Mamba-style SSM)와의 결합 연구가 필요합니다.

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

| 연도 | 모델 | 토큰화 / 핵심 아이디어 | 압축률 | 최대 face 수 | 좌표 해상도 | 비고 |
|---|---|---|---|---|---|---|
| 2020 | **PolyGen** (Nash et al., ICML) | 정점-면 분리, naive 좌표 직렬화 | — | 매우 작음(수백) | 8-bit | 첫 자기회귀 메쉬 생성 |
| 2024.06 | **MeshGPT** (Siddiqui et al., CVPR) | VQ-VAE + 자기회귀 디코더 | $6 N_f$ | ~800 | 7~8-bit | 압축에 VQ-VAE 사용 |
| 2024.06 | **MeshAnything** (Chen et al.) | naive face tokenization (9 토큰/face) | $9 N_f$ | ~800 | 7-bit | 점군 조건 도입 |
| 2024.08 | **MeshAnythingV2** (Chen et al.) | Adjacent Mesh Tokenization (AMT) | $\sim 4 N_f$ | ~1,600 | 7-bit | 인접성 활용 |
| 2024.09 | **EdgeRunner** (Tang et al.) | Auto-encoder + edge traversal | $\sim 4 N_f$ | ~4,000 | 7-bit | 압축 토큰 사용 |
| 2024.11 | **BPT** (Weng et al.) | Blocked & Patchified Tokenization | ~75% 감소 | >8,000 | 7-bit | 블록 인덱싱 + 패치 집약 |
| 2024.12 | **Meshtron** (Hao et al., NVIDIA) | Hourglass + sliding window inference | 길이 truncation | **64,000** | **10-bit (1024)** | 50% 메모리 ↓, 2.5× throughput ↑ |
| **2025.03** | **TreeMeshGPT** (Lionar et al., CVPR) | **Tree sequencing (DFS + half-edge)** | $2N_f + 2N_c$ (~22%) | 5,500 (7-bit) / 11,000 (9-bit) | 7~9-bit | **Normal 일관성, 국소 예측** |

### 비교 분석 요약

- **압축률 측면**: TreeMeshGPT의 face당 2 토큰은 BPT의 약 25% 수준에 비견되는 매우 공격적인 압축이며, MeshAnythingV2/EdgeRunner의 약 절반(2배 압축)입니다. BPT는 블록 단위 인덱싱과 패치 집약을 통해 시퀀스 길이를 약 75% 줄여 8K face 이상을 다룹니다.

- **face 수용량 측면**: Meshtron은 hourglass 구조와 sliding window 추론으로 최대 64K face를 1024-level 좌표 해상도로 생성합니다. TreeMeshGPT는 face 수에서는 Meshtron/BPT에 뒤지지만, **normal 일관성**과 **시퀀스의 국소성**에서 차별점이 있습니다.

- **품질 vs 규모의 trade-off**: Meshtron과 BPT가 "scale-up"에 집중한다면, TreeMeshGPT는 "structural correctness(manifold + normal 일관성)"에 집중합니다. 이는 직접 산업 활용(rigging, editing)을 고려할 때 중요한 차별점입니다.

- **후속 영향**: 2025년 7월에 발표된 "Auto-Regressive Mesh Generation as Weaving Silk"는 BPT, TreeMeshGPT, MeshAnythingV2 등의 비-경계 정점이 두 번 인코딩되는 redundancy 문제를 지적하고 새로운 토큰화 알고리즘을 제안합니다. 이는 TreeMeshGPT가 이미 후속 연구의 baseline으로 자리잡았음을 보여줍니다.

---

## 참고 자료

1. **본 논문**: Lionar, S., Liang, J., Lee, G. H. *TreeMeshGPT: Artistic Mesh Generation with Autoregressive Tree Sequencing*. CVPR 2025, pp. 26608-26617. arXiv:2503.11629.
2. **공식 코드 저장소**: https://github.com/sail-sg/TreeMeshGPT
3. **CVPR Open Access**: https://openaccess.thecvf.com/content/CVPR2025/html/Lionar_TreeMeshGPT_Artistic_Mesh_Generation_with_Autoregressive_Tree_Sequencing_CVPR_2025_paper.html
4. **MeshAnything**: Chen et al. *MeshAnything: Artist-Created Mesh Generation with Autoregressive Transformers*. arXiv:2406.10163 (2024).
5. **MeshAnythingV2**: Chen et al. *MeshAnything V2: Artist-Created Mesh Generation with Adjacent Mesh Tokenization*. arXiv:2408.02555 (2024).
6. **EdgeRunner**: Tang et al. *EdgeRunner: Auto-regressive Auto-encoder for Artistic Mesh Generation*. arXiv:2409.18114 (2024).
7. **BPT**: Weng et al. *Scaling Mesh Generation via Compressive Tokenization*. arXiv:2411.07025 (2024).
8. **Meshtron**: Hao et al. *Meshtron: High-Fidelity, Artist-Like 3D Mesh Generation at Scale*. arXiv:2412.09548 (2024). NVIDIA Research.
9. **MeshGPT**: Siddiqui et al. *MeshGPT: Generating Triangle Meshes with Decoder-Only Transformers*. CVPR 2024.
10. **PolyGen**: Nash et al. *PolyGen: An Autoregressive Generative Model of 3D Meshes*. ICML 2020.
11. **후속 연구 (TreeMeshGPT 비교)**: *Auto-Regressive Mesh Generation as Weaving Silk*. arXiv:2507.02477 (2025).

---

**참고**: 본 분석은 제공된 PDF 본문과 검색을 통해 확인된 공개 정보에 근거합니다. 표의 face 수용량 수치는 각 논문이 자체 발표한 숫자이며, 학습 데이터·아키텍처 규모가 다르므로 단순 비교에는 주의가 필요합니다. 또한 2025년 5월 현재 mesh generation 분야는 빠르게 발전 중이므로, 가장 최신 SOTA는 본문 시점 이후 변경되었을 가능성이 있습니다.
