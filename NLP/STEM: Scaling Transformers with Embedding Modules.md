# STEM: Scaling Transformers with Embedding Modules 

---

## 📌 참고 자료

- **주요 논문**: Sadhukhan et al. (2026). *STEM: Scaling Transformers with Embedding Modules*. arXiv:2601.10639v1 [cs.LG]. https://arxiv.org/abs/2601.10639
- **관련 논문들** (논문 내 인용 기준):
  - Shazeer et al. (2017). *Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer*. arXiv:1701.06538
  - Fedus et al. (2022). *Switch Transformers*. arXiv:2101.03961
  - Roller et al. (2021). *Hash Layers for Large Sparse Models*. arXiv:2106.04426
  - Geva et al. (2021). *Transformer Feed-Forward Layers Are Key-Value Memories*. arXiv:2012.14913
  - Meng et al. (2022). *Locating and Editing Factual Associations in GPT*. arXiv:2202.05262
  - dos Santos et al. (2023). *Memory Augmented Language Models through Mixture of Word Experts (MoWE)*. arXiv:2311.10768
  - Google DeepMind (2024). *Gemma 3n (Per Layer Embeddings)*. https://ai.google.dev/gemma/docs/gemma-3n
  - Lample et al. (2019). *Large Memory Layers with Product Keys (PKM)*. arXiv:1907.05242
  - Berges et al. (2024). *Memory Layers at Scale*. arXiv:2412.09764
  - Dai et al. (2024). *DeepSeekMoE*. arXiv:2401.06066
  - Kaplan et al. (2020). *Scaling Laws for Neural Language Models*. arXiv:2001.08361
  - Hoffmann et al. (2022). *Training Compute-Optimal Large Language Models*. arXiv:2203.15556
  - Liu et al. (2024). *MobileLLM*. arXiv:2402.14905
  - He (2024). *Mixture of a Million Experts*. arXiv:2407.04153
  - Boix-Adsera & Rigollet (2025). *The Power of Fine-Grained Experts*. arXiv:2505.06839
  - Huben et al. (2024). *Sparse Autoencoders*. ICLR 2024
  - Huang et al. (2025). *Ultra-Sparse Memory Network*. arXiv:2411.12364

---

## 1. 핵심 주장 및 주요 기여 (요약)

STEM(**S**caling **T**ransformers with **E**mbedding **M**odules)은 Transformer의 **FFN(Feed-Forward Network) up-projection 행렬**을 **토큰 인덱스 기반의 레이어-로컬 임베딩 테이블**로 대체하는 정적(static) 희소 아키텍처입니다.

### 핵심 주장

| 항목 | 내용 |
|------|------|
| **훈련 안정성** | MoE의 load imbalance/loss spike 문제 없이 극도의 희소성에서도 안정적 훈련 |
| **성능 향상** | 밀집 베이스라인 대비 최대 ~3–4% 정확도 향상 (지식/추론 태스크에서 ~9–10%) |
| **효율성** | FFN 파라미터의 약 1/3 제거, per-token FLOPs 및 파라미터 접근 비용 절감 |
| **해석 가능성** | 토큰 인덱스 기반 임베딩으로 지식 편집(knowledge editing) 및 주입(injection) 가능 |
| **장문맥 성능** | 시퀀스 길이가 늘어날수록 더 많은 파라미터가 활성화되는 test-time capacity scaling |

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

기존 **Mixture-of-Experts(MoE)** 기반의 세밀한(fine-grained) 희소 아키텍처는 다음과 같은 문제를 내포합니다:

1. **훈련 불안정성**: 비균일 라우팅으로 인한 loss spike 및 일부 expert의 under-training
2. **부하 불균형**: load-balancing 보조 손실이 주 목표함수와 간섭
3. **통신 오버헤드**: expert 병렬화 시 all-to-all 통신 비용이 expert 수에 비례하여 증가
4. **커널 비효율성**: micro-expert 크기가 작아 dense linear-algebra 커널의 GPU 점유율 저하
5. **해석 불가능성**: 각 micro-expert의 역할을 이해하기 어려움

---

### 2.2 제안 방법 및 수식

#### 기존 SwiGLU FFN

레이어 $\ell$에서 입력 $\mathbf{x}\_\ell \in \mathbb{R}^d$, 게이트 투영 $\mathbf{W}^{(g)}\_\ell \in \mathbb{R}^{d_\text{ff} \times d}$, up-projection $\mathbf{W}^{(u)}\_\ell \in \mathbb{R}^{d_\text{ff} \times d}$, down-projection $\mathbf{W}^{(d)}\_\ell \in \mathbb{R}^{d \times d_\text{ff}}$에 대해:

$$\mathbf{y}_\ell = \mathbf{W}^{(d)}_\ell \left( \text{SiLU}\!\left(\mathbf{W}^{(g)}_\ell \mathbf{x}_\ell\right) \odot \left(\mathbf{W}^{(u)}_\ell \mathbf{x}_\ell\right) \right) $$

#### 기존 MoE FFN

$K$개 expert $\{f_{\ell,k}\}$와 라우터 $r_\ell(\mathbf{x}\_\ell)$를 통해 top- $r$ expert 집합 $\mathcal{T}\_\ell(\mathbf{x}_\ell)$를 선택:

$$\mathbf{y}_\ell = \sum_{k \in \mathcal{T}_\ell(\mathbf{x}_\ell)} \pi_{\ell,k}(\mathbf{x}_\ell)\, f_{\ell,k}(\mathbf{x}_\ell) $$

#### Hash-layer MoE (토큰 ID 기반 정적 라우팅)

$$\mathbf{y}_\ell = \sum_{k \in \text{hash}(t)} f_{\ell,k}(\mathbf{x}_\ell) $$

#### ✅ STEM의 핵심 수식

레이어 $\ell$에서 임베딩 테이블 $\mathbf{U}\_\ell \in \mathbb{R}^{V \times d_\text{ff}}$ (vocabulary size $V$), 토큰 ID $t$에 해당하는 행 $\mathbf{U}\_\ell[t] \in \mathbb{R}^{d_\text{ff}}$를 이용하여:

$$\boxed{\mathbf{y}_\ell = \mathbf{W}^{(d)}_\ell \left( \text{SiLU}\!\left(\mathbf{W}^{(g)}_\ell \mathbf{x}_\ell\right) \odot\, \mathbf{U}_\ell[t] \right)} $$

**핵심 변경**: $\mathbf{W}^{(u)}\_\ell \mathbf{x}\_\ell$ (행렬-벡터 곱, 문맥 의존적) → $\mathbf{U}_\ell[t]$ (토큰 인덱스 기반 직접 조회, 문맥 독립적)

#### STEM†(하이브리드 변형)

up-projection을 보존하되 임베딩을 덧붙이는 방식 (성능 향상 없음으로 ablation에서 기각):

$$\mathbf{y}_\ell = \mathbf{W}^{(d)}_\ell \left( \text{SiLU}\!\left(\mathbf{W}^{(g)}_\ell \mathbf{x}_\ell\right) \odot \left(\mathbf{W}^{(u)}_\ell \mathbf{x}_\ell + \mathbf{U}_\ell[t]\right) \right) $$

#### FFN의 Key-Value Memory 관점

$$\mathbf{y} = \sum_{i=1}^{d_\text{ff}} \underbrace{\phi(\langle \mathbf{k}_i, \mathbf{x} \rangle)}_{\text{addressing weight } \alpha_i(\mathbf{x})} \underbrace{\mathbf{v}_i}_{\text{value}}$$

여기서 $\mathbf{k}_i$는 $\mathbf{W}^{(u)}$의 $i$번째 행(key), $\mathbf{v}_i$는 $\mathbf{W}^{(d)}$의 $i$번째 열(value). STEM은 이 key를 토큰별 임베딩으로 대체하여 더 직접적이고 분리된 정보 저장을 실현합니다.

SwiGLU의 gated 버전:

```math
\mathbf{y} = \mathbf{W}^{(d)}\!\left((\mathbf{W}^{(u)}\mathbf{x}) \odot \sigma(\mathbf{W}^g \mathbf{x})\right) = \sum_{i=1}^{d_\text{ff}} \left\{ \underbrace{\langle \mathbf{k}_i, \mathbf{x} \rangle}_{\text{content}} \cdot \underbrace{\sigma(\langle \tilde{\mathbf{k}}_i, \mathbf{x} \rangle)}_{\text{gate}} \right\} \mathbf{v}_i
```

---

### 2.3 모델 구조

```
[STEM Layer Architecture]

Input xℓ
  ├──→ Wg (Gate Projection, dense, 공유)
  │      └──→ SiLU(·)           ← 문맥 의존적 게이팅 (유지)
  │                    ↘
  └──→ Uℓ[t] (CPU 임베딩 테이블에서 조회) ← up-projection 대체
               ↘         ↙
              Element-wise ⊙
                    ↓
              Wd (Down Projection, dense, 공유)
                    ↓
               Output yℓ
```

**구조적 특징:**
- **Gate projection** ($\mathbf{W}^{(g)}_\ell$): 유지 (문맥 의존적 변조를 위해 필수)
- **Up projection** ( $\mathbf{W}^{(u)}\_\ell$): → 임베딩 테이블 $\mathbf{U}\_\ell \in \mathbb{R}^{V \times d_\text{ff}}$로 대체
- **Down projection** ($\mathbf{W}^{(d)}_\ell$): 유지 (forward path 파괴 방지)
- 임베딩 테이블: **CPU 메모리 오프로드**, GPU로 비동기 prefetch

**배치 전략**: FFN 레이어의 1/3, 1/2, 전체를 STEM으로 대체하는 세 가지 설정 실험

---

### 2.4 효율성 분석

#### 훈련 FLOPs

$$F^{\text{base}}_{\text{train}} = B\!\left(4Ld^2 + 2L^2d + 3Ld\,d_\text{ff}\right)$$

$$F^{\text{stem}}_{\text{train}} = B\!\left(\underbrace{4Ld^2 + 2L^2d}_{\text{Attention}} + \underbrace{2Ld\,d_\text{ff}}_{\text{FFN}}\right)$$

$$\Delta F_{\text{train}} = F^{\text{base}}_{\text{train}} - F^{\text{stem}}_{\text{train}} = BLd\,d_\text{ff}$$

$$\text{saving fraction} = \frac{\Delta F_{\text{train}}}{F^{\text{base}}_{\text{train}}} = \frac{d_\text{ff}}{4d + 2L + 3d_\text{ff}}$$

**Qwen2.5 계열 적용 시 절약률**: 1.5B → 21.7%, 3B → 22.8%, 7B → 23.9%, 14B → 19.7%, 32B → 24.8%

#### 디코딩 메모리 접근 비용

$$M^{\text{base}}_{\text{dec}} = B\!\left(4d^2 + 2Ld + 3d\,d_\text{ff}\right)$$

$$M^{\text{stem}}_{\text{dec}} = B\!\left(\underbrace{2Ld}_{\text{KV cache}} + \underbrace{4d^2 + 2d\,d_\text{ff}}_{\text{projection params}}\right)$$

$$\Delta M_{\text{dec}} = Bd\,d_\text{ff}, \quad \text{saving fraction} = \frac{d_\text{ff}}{4d + 2L + 3d_\text{ff}}$$

#### Context-length Adaptive Parameter Usage

$$\text{Params}^{\text{STEM}}_{\text{act}}(L) = |\mathcal{S}|\, d_\text{ff}\, L_{\text{uniq}}$$

여기서 $|\mathcal{S}|$는 STEM 적용 레이어 수, $L_{\text{uniq}}$는 시퀀스 내 고유 토큰 수 (Heaps 법칙에 따라 $L$에 대해 준선형 증가).

---

### 2.5 성능 향상

#### 350M 사전훈련 결과 (주요 수치)

| 모델 | Avg | ARC-C | OBQA | GFLOPs | ROI |
|------|-----|-------|------|--------|-----|
| Dense Baseline | 49.72 | 30.55 | 34.80 | 0.74 | 1x |
| Hash-MoE | 50.58 | 36.33 | 39.26 | 0.74 | 1.02x |
| STEM (1/3) | 50.90 | 32.68 | 33.00 | 0.70 | 1.08x |
| STEM (1/2) | 54.20 | 40.00 | 46.68 | 0.67 | 1.20x |
| STEM (full) | 53.43 | 39.61 | 44.53 | 0.60 | 1.33x |

#### 1B 중간훈련 결과

| 모델 | Avg | GSM8K | MMLU |
|------|-----|-------|------|
| Dense Baseline | 57.50 | 44.2 | 29.92 |
| STEM (1/3) | 58.49 | 46.4 | 32.38 |

#### 장문맥 성능 (Needle-in-a-Haystack)
- 8k context: STEM이 Dense 대비 **+8.4%**
- 32k context: STEM이 Dense 대비 **+13%** (차이가 더 벌어짐)

#### Training ROI 정의

$$\text{Training ROI} = \frac{\text{Model Accuracy (Avg)}}{\text{Total Training FLOPs}}$$

---

### 2.6 한계

논문에서 명시되거나 구조적으로 추론 가능한 한계:

1. **문맥 독립성 (Context Agnosticity)**: up-projection이 토큰 ID에만 의존하므로, 동일한 토큰이라도 다른 문맥에서 동일한 임베딩 벡터를 사용 → 표현력(expressivity) 제한 가능성
2. **CPU-GPU 통신 오버헤드**: 훈련 시 임베딩 그래디언트를 CPU로 전송해야 하므로 통신 비용이 2배
3. **Zipfian 분포의 영향**: 고빈도 토큰의 임베딩이 과도하게 학습되고 저빈도 토큰은 under-training 가능성 잔존
4. **자동회귀 디코딩 시 prefetch 제한**: 다음 토큰 ID를 현재 forward pass가 완료되기 전에는 알 수 없어 prefetch 효율 감소
5. **훈련 구현의 미완성**: 논문 자체적으로 "fully optimized training implementation을 다음 버전에 제공할 것"이라 명시
6. **규모 제한**: 350M, 1B 규모에서만 검증; 더 큰 모델(7B+)에서의 효과는 이론적 절약률만 제시

---

## 3. 일반화 성능 향상 가능성 🔬

STEM의 일반화 성능 향상은 다음 메커니즘들로 설명됩니다:

### 3.1 임베딩 공간의 대규모 각도 분산 (Large Angular Spread)

STEM 임베딩들은 훈련 후 **낮은 쌍별 코사인 유사도**를 보입니다 (P95 |cos| ≈ 0.026~0.033). 이는 표준 FFN의 up-projection 출력 공간과 대조적입니다.

$$\text{간섭 감소: } \cos(\mathbf{U}_\ell[t_i], \mathbf{U}_\ell[t_j]) \approx 0 \quad \forall\, i \neq j$$

이러한 **quasi-orthogonality**는 Donoho & Elad (2003), Tropp (2004)의 희소 표현 이론에 의하면 메모리 슬롯 간 cross-talk를 줄이고, **고정 폭에서 효과적인 정보 저장 용량을 증가**시킵니다.

**수학적 해석**: Key-Value Memory 관점에서, 각 토큰의 임베딩이 서로 직교에 가까울수록 addressing 과정에서 원하는 슬롯만 선택적으로 활성화되어 **정보 검색의 정밀도**가 향상됩니다.

### 3.2 파라메트릭 메모리 용량 확대와 일반화

| 메커니즘 | 설명 |
|---------|------|
| **Superposition 탈피** | 표준 FFN은 저차원 주소 공간에 다수 개념을 superposition으로 인코딩 → STEM은 토큰별 독립 벡터로 간섭 최소화 |
| **Knowledge slot 증가** | 어휘 크기 $V$에 비례하는 독립 슬롯 → 더 많은 사실적 지식 저장 가능 |
| **Context-adaptive modulation** | Gate projection이 여전히 문맥을 활용하여 토큰 임베딩을 동적으로 변조 → 문맥 적응력 보존 |

### 3.3 Test-time Capacity Scaling을 통한 일반화

$$\text{Params}^{\text{STEM}}_{\text{act}}(L) = |\mathcal{S}|\, d_\text{ff}\, L_{\text{uniq}}(L)$$

시퀀스 길이 $L$이 증가할수록 $L_{\text{uniq}}$도 증가(Heaps' Law), 더 많은 파라미터가 활성화됩니다. 이는:
- **MoE와의 차이**: MoE는 expert 수가 고정되어 있어 capacity가 포화 → STEM은 새로운 토큰이 등장할 때마다 새 파라미터 활성화
- **장문맥 일반화**: 다문서 RAG, Chain-of-Thought 등 장문 태스크에서 추가 용량을 일정 per-token compute로 제공

### 3.4 지식 집약 태스크에서의 일반화

| 태스크 유형 | 성능 향상 | 이유 |
|-----------|---------|------|
| ARC-Challenge | ~9% 향상 | 지식 검색 정밀도 향상 |
| OpenBookQA | ~10% 향상 | 토큰별 사실 지식 저장 최적화 |
| GSM8K | +2.2pt | 수학적 추론에서 지식 검색 개선 |
| MMLU | +2.46pt | 다영역 지식 포괄 능력 향상 |
| BBH | 24.87→27.55 | 다단계 추론에서의 문맥+지식 통합 |
| LongBench Multi-hop | 전 범위에서 향상 | 장문맥 지식 추론 |

### 3.5 해석 가능한 지식 편집의 일반화 시사점

토큰별 임베딩의 교체만으로 모델 출력을 조종할 수 있다는 것은, STEM이 **사실적 지식을 토큰 단위로 모듈화**하여 저장함을 의미합니다. 이는 일반화 관점에서:

- **지식 갱신의 국소성**: 특정 사실 변경 시 관련 토큰 임베딩만 수정 → 불필요한 catastrophic forgetting 방지 가능성
- **분포 외(Out-of-Distribution) 대응**: 새로운 개체에 대한 지식을 임베딩 추가로 주입 가능

---

## 4. 최신 연구 비교 분석 (2020년 이후)

### 4.1 희소 아키텍처 계보

```
Switch Transformer (2022)
    ↓ 세밀화
DeepSeekMoE (2024) / DBRX (2024)
    ↓ 극세밀화
MoWE (2023) / Mixture of a Million Experts (2024)
    ↓ 정적 라우팅
Hash Layer MoE (2021)
    ↓ 레이어별 임베딩
PLE / Gemma-3n (2024)
    ↓ 완전 대체 + 효율 최적화
STEM (2026) ← 현재 논문
```

### 4.2 상세 비교표

| 방법 | 라우팅 방식 | 통신 오버헤드 | 훈련 안정성 | 해석 가능성 | 파라미터 절감 | FLOPs |
|------|-----------|------------|-----------|-----------|------------|-------|
| **Switch Transformer** (Fedus et al., 2022) | 학습된 라우터 (top-1) | All-to-all | 중간 (loss spike) | 낮음 | 없음 | Dense 동일 |
| **DeepSeekMoE** (Dai et al., 2024) | 학습된 라우터 (fine-grained) | All-to-all | 낮음 | 낮음 | 없음 | Dense 동일 |
| **Hash Layer MoE** (Roller et al., 2021) | 고정 해시 (token ID) | All-to-all | 높음 | 중간 | 없음 | Dense 동일 |
| **MoWE** (dos Santos et al., 2023) | 고정 (word ID) | All-to-all | 낮음 (Zipfian) | 중간 | 없음 | Dense 동일 |
| **PKM** (Lample et al., 2019) | Product key 검색 | 없음 | 중간 | 낮음 | 있음 | 증가 |
| **Memory Layers at Scale** (Berges et al., 2024) | Top-k 검색 | 낮음 | 중간 | 낮음 | 있음 | 증가 |
| **PLE/Gemma-3n** (Google DeepMind, 2024) | 토큰 ID 기반 | CPU prefetch | 높음 | 중간 | 없음 (추가) | 증가 |
| **STEM** (이 논문, 2026) | 토큰 ID 기반 | CPU prefetch | **높음** | **높음** | **~1/3 FFN** | **감소** |

### 4.3 PLE vs. STEM 상세 비교

| 특성 | PLE (Gemma-3n) | STEM |
|------|----------------|------|
| 기존 FFN 유지 여부 | ✅ 유지 (추가 컴포넌트) | ❌ up-projection 대체 |
| 임베딩 차원 | 낮음 (e.g., 256 vs FFN 16384) | **FFN과 동일** ($d_\text{ff}$) |
| FLOPs 변화 | 증가 | **감소** |
| 파라미터 변화 | 증가 | 총량 증가, active 감소 |
| 설계 목적 | 소형 기기용 용량 보완 | **범용 스케일링** |

### 4.4 Ultra-Sparse Memory Network (Huang et al., 2025)와의 관계

PKM의 under-training 문제를 해결하는 아키텍처로, STEM과 상호 보완적입니다. STEM은 **정적 토큰 인덱싱**으로 훈련 안정성을 보장하는 반면, Ultra-Sparse Memory는 **동적 검색**에 의존합니다.

---

## 5. 향후 연구에 미치는 영향 및 고려사항

### 5.1 연구에 미치는 영향

#### 🔷 아키텍처 설계 패러다임 전환

STEM은 **파라미터 용량과 per-token compute를 분리**하는 새로운 설계 원칙을 제시합니다. 이는 "더 큰 모델 = 더 많은 연산"이라는 기존 공식에 도전하며, 다음 방향을 열어줍니다:

- **Mixture of STEM Experts**: 각 MoE expert의 FFN을 STEM FFN으로 대체하는 하이브리드 설계
- **선택적 레이어 희소화**: 지식 집약 레이어만 STEM으로 교체하는 adaptive 전략
- **다중 모달 확장**: 비전/오디오 토큰에 대한 임베딩 모듈 설계 가능성

#### 🔷 해석 가능성(Interpretability) 연구의 새 방향

STEM의 **토큰-임베딩 직접 대응**은 기존 해석 가능성 연구와 차별화됩니다:

- **Sparse Autoencoder** (Huben et al., 2024)와 비교: SAE는 사후 분석(post-hoc)이지만 STEM은 구조 자체가 해석 가능
- **ROME/MEMIT** (Meng et al., 2022)와 비교: ROME은 가중치 직접 수정이지만 STEM은 임베딩 교체만으로 동등한 효과
- **Causal Tracing 없이도** 지식의 위치를 레이어-토큰 단위로 특정 가능

#### 🔷 지식 편집 연구

STEM의 임베딩 교체 기반 지식 편집은:
- **입력 텍스트 수정 불필요**: 외부 개입 없이 모델 내부에서 지식 조작
- **가역성(Reversibility)**: 임베딩을 원복하면 원래 동작으로 복귀
- **다중 엔티티 동시 편집** 가능성

#### 🔷 시스템 및 효율성 연구

- **CPU Offloading + Prefetching** 패턴의 정교화: LFU 캐시 80%+ hit rate → 캐시 정책 최적화 연구
- **이종 메모리(HBM+DRAM) 최적화**: 향후 CXL 메모리와의 통합 가능성
- **분산 훈련에서의 임베딩 병렬화**: FSDP/TP와의 조합 최적화

---

### 5.2 향후 연구 시 고려할 점

#### ⚠️ 문맥 독립성 문제

up-projection이 토큰 ID에만 의존하므로, **동형이의어(polysemy)** 처리에 취약할 수 있습니다. 예를 들어 "bank(강둑)"과 "bank(은행)"가 동일한 임베딩을 공유합니다.

**연구 방향**: Context-conditioned STEM embedding — 예를 들어 attention 출력이나 이전 레이어 hidden state를 조건으로 임베딩을 보정하는 메커니즘 탐구

#### ⚠️ Zipfian 분포와 저빈도 토큰 under-training

자연어에서 토큰 빈도는 Zipf 분포를 따르므로, 희귀 토큰 임베딩은 훈련 데이터에 충분히 노출되지 않을 수 있습니다.

**연구 방향**:
- 저빈도 토큰 임베딩에 대한 정규화 기법
- 유사 빈도 토큰 간 임베딩 공유 또는 초기화 전략
- Few-shot/zero-shot 시나리오에서의 희귀 토큰 처리

#### ⚠️ 더 큰 규모에서의 검증 필요

현재 논문은 350M과 1B 규모에서만 실험. FLOPs 절약 비율이 이론적으로는 7B~32B에서 더 크지만 (22.8%~24.8%), 실제 성능 향상 여부는 미검증입니다.

**연구 방향**: 7B, 13B, 70B 규모에서의 STEM 효과 체계적 검증

#### ⚠️ 훈련 효율 구현 완성

논문은 "fully optimized training implementation을 다음 버전에 제공"이라고 명시. 현재 CPU offloading 중 그래디언트 write-back 최적화가 미완성입니다.

**연구 방향**: PyTorch FSDP/DeepSpeed와의 완전한 통합, 비동기 optimizer update 구현

#### ⚠️ 지식 편집의 정교화

현재 지식 편집은 단순 벡터 교체 수준입니다. 토크나이제이션 길이 불일치 처리(padding, copying, subset selection, averaging)가 휴리스틱에 의존합니다.

**연구 방향**:
- 편집 성공률의 자동 측정 지표 개발
- 여러 언어에서의 동일 엔티티 처리
- 편집 후 collateral damage(다른 사실에의 영향) 측정

#### ⚠️ 다중 모달 및 특수 도메인 적용

STEM은 텍스트 토큰의 어휘 기반 인덱싱에 의존합니다. 연속적인 값을 갖는 비전/오디오 입력에는 직접 적용이 어렵습니다.

**연구 방향**: 비전 언어 모델(VLM)에서의 image patch → 클러스터 인덱스 변환 후 STEM 적용

#### ⚠️ 일반화 이론적 보장

STEM의 일반화 성능 향상이 경험적으로는 확인되었으나, 이론적 설명이 부분적입니다. 

**연구 방향**: 
- Large angular spread와 일반화 오류 간의 PAC-Bayes 또는 Rademacher complexity 기반 이론적 연결
- Superposition 이론(Toy Models of Superposition, Anthropic)과 STEM의 관계 형식화

---

## 요약

STEM은 Transformer FFN의 up-projection을 토큰 인덱스 기반 레이어-로컬 임베딩 테이블로 대체함으로써, **파라미터 용량 확대, 훈련 안정성, 해석 가능성, 효율성**을 동시에 달성하는 실용적이고 혁신적인 아키텍처입니다. 특히 지식 집약 태스크에서의 일반화 향상과 장문맥에서의 test-time capacity scaling은 향후 LLM 스케일링 연구에 중요한 기여를 할 것으로 판단됩니다. 그러나 문맥 독립성 한계, 더 큰 규모에서의 검증 필요성, 훈련 구현의 완성도 등은 후속 연구에서 반드시 다루어야 할 과제입니다.
