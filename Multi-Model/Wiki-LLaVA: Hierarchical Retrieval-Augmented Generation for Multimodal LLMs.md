# Wiki-LLaVA: Hierarchical Retrieval-Augmented Generation for Multimodal LLMs

---

## 1. 핵심 주장과 주요 기여 요약

### 핵심 주장

Wiki-LLaVA는 **Multimodal LLM(MLLM)이 외부 지식 없이는 답변하기 어려운 세부적이고 전문적인 질문에 대응할 수 없다는 한계**를 극복하기 위해, Wikipedia 기반 외부 지식 베이스에서 계층적(hierarchical) 검색을 수행하여 관련 정보를 LLM의 입력 컨텍스트에 주입하는 방식을 제안합니다.

### 주요 기여

| 기여 항목 | 설명 |
|---|---|
| **최초의 RAG 기반 MLLM** | Retrieval-Augmented Generation을 MLLM에 적용한 최초의 연구 |
| **계층적 검색 파이프라인** | CLIP 기반 문서 검색 + Contriever 기반 패시지 검색의 2단계 구조 |
| **데이터 혼합 학습** | 외부 지식 필요 데이터와 일반 VQA 데이터를 혼합하여 일반화 성능 유지 |
| **광범위한 실험 검증** | Encyclopedic-VQA, InfoSeek 두 벤치마크에서 성능 검증 |

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

현재 MLLM들은 다음과 같은 한계를 가집니다:

1. **Long-tail 정보 부족**: 학습 데이터에 희귀 정보가 적어 특정 엔티티에 대한 세부 지식이 파라미터에 인코딩되지 않음
2. **파라미터 내 지식의 한계**: 모델 파라미터만으로는 고도로 특정적인 사실 기반 질문에 답하기 어려움
3. **기존 벤치마크의 도전**: Encyclopedic-VQA, InfoSeek 등에서 외부 지식 없는 모델들은 낮은 성능을 보임

---

### 2.2 제안하는 방법 (수식 포함)

#### (1) 표준 MLLM의 생성 확률 모델링

표준 MLLM은 다음과 같이 조건부 확률을 모델링합니다:

$$p(w_t | I, w_0, w_1, \ldots, w_{t-1}, \theta)$$

여기서 $\theta$는 모델 파라미터, $I$는 입력 이미지, $w_0, \ldots, w_{t-1}$은 텍스트 프롬프트입니다.

#### (2) 외부 지식 증강 후 생성 확률 (핵심 수식)

Wiki-LLaVA는 외부 메모리에서 검색된 토큰을 추가하여 다음과 같이 확장합니다:

$$p\!\left(w_t \,\Big|\, \overbrace{v_o, v_1, \ldots, v_N}^{\text{Visual tokens}},\ \underbrace{w_0, w_1, \ldots, w_{t-1}}_{\text{System + user prompt}},\ \overbrace{e_0, e_1, \ldots, e_{\tau}}^{\text{External memory tokens}}\right) $$

여기서 $e_0, \ldots, e_\tau$는 외부 메모리에서 검색된 추가 토큰을 나타냅니다.

#### (3) 1단계: 문서(엔티티) 검색 — CLIP 기반

입력 이미지 $I$와 Wikipedia 문서 제목 $t_i$ 간의 유사도를 CLIP의 시각 인코더 $E_v$와 텍스트 인코더 $E_t$를 이용해 계산합니다:

$$\text{sim}(I_i, t_i) = E_v(I) \cdot E_t(t_i)^T $$

상위 $k$개의 관련 문서를 반환합니다. 근사 $k$-최근접이웃(approximate kNN) 검색에는 **Faiss 라이브러리**와 **HNSW 인덱스(32 links/vertex)**를 사용합니다.

#### (4) 2단계: 패시지 검색 — Contriever 기반

각 문서 $d_i = [c_{i_0}, \ldots, c_{i_T}]$ (600자 단위 청크)에서, **Contriever 아키텍처**를 사용하여 청크 임베딩과 질문 임베딩 간의 내적(inner product)으로 유사도를 계산하고, 상위 $n$개의 패시지를 선택합니다.

전체적으로 $k \times n$개의 패시지가 최종 컨텍스트로 활용됩니다.

#### (5) 최종 프롬프트 구성

```
<IMAGE>\nGiven the following context:\n
<R1>\n<R2>\<R3>\n <QUESTION>
Give a short answer. ASSISTANT:     (3)
```

---

### 2.3 모델 구조

```
[입력 이미지 I]
      │
      ▼
[CLIP ViT-L/14@336 (시각 인코더 Ev)]
      │                    │
      ▼                    ▼
[MLP Adapter]    [1단계: CLIP 기반 엔티티 검색]
      │                    │ (상위 k개 문서)
[Visual Tokens]            ▼
      │          [2단계: Contriever 기반 패시지 검색]
      │                    │ (각 문서에서 n개 패시지)
      └──────┬─────────────┘
             │
      [컨텍스트 통합: 이미지 토큰 + 검색 패시지 + 질문]
             │
      [LLM (Vicuna-7B)]
             │
      [최종 답변 생성]
```

**구성 컴포넌트:**
- **시각 인코더**: CLIP ViT-L/14@336
- **LLM 백본**: Vicuna-7B
- **어댑터**: MLP (CLIP 피처 → 밀집 임베딩 토큰)
- **외부 지식 베이스**: Wikipedia (Encyclopedic-VQA: 2M 문서, InfoSeek: 100k 엔티티)
- **1단계 검색**: CLIP 기반 kNN 검색
- **2단계 검색**: Contriever 기반 패시지 검색
- **파인튜닝**: LoRA (Low-Rank Adaptation), 배치 크기 512

---

### 2.4 성능 향상

#### 엔티티 검색 성능 (Table 1)

| Dataset | KB 크기 | R@1 | R@10 | R@20 | R@50 |
|---|---|---|---|---|---|
| Encyclopedic-VQA | 2M | 3.3 | 9.9 | 13.2 | 17.5 |
| InfoSeek | 100k | 36.9 | 66.1 | 71.9 | 78.4 |

→ KB 크기가 클수록 검색 정확도가 낮아지는 경향 확인

#### VQA 정확도 (Table 2 요약)

| 모델 | Enc-VQA (All) | InfoSeek (All) |
|---|---|---|
| BLIP-2 (zero-shot) | 12.4 | 12.5 |
| InstructBLIP (zero-shot) | 12.0 | 8.1 |
| LLaVA-1.5 (zero-shot) | 16.9 | 9.5 |
| LLaVA-1.5 (fine-tuned, no KB) | 28.5 | 17.9 |
| Wiki-LLaVA (k=1, n=1, CLIP) | 26.4 | 25.5 |
| Wiki-LLaVA (k=1, n=3, CLIP) | 20.3 | **28.9** |
| Wiki-LLaVA (k=1, n=2, Oracle) | **40.2** | 47.8 |
| Wiki-LLaVA (k=1, n=3, Oracle) | 38.6 | **51.5** |

**주요 관찰:**
- Oracle 엔티티 사용 시 CLIP 대비 약 **+13.8%p (Enc-VQA)**, **+22.6%p (InfoSeek)** 향상
- InfoSeek에서는 $n$ 증가 시 성능 향상 (n=1→n=3: +3.4%p)
- Enc-VQA에서는 CLIP 검색의 낮은 정확도(R@1: 3.3%)로 노이즈 패시지 유입 → 성능 저하

---

### 2.5 한계

1. **대규모 KB에서의 검색 정확도 저하**: 2M 크기의 KB에서 CLIP R@1이 3.3%에 불과
2. **노이즈 패시지 문제**: 잘못된 엔티티 검색 시 무관한 텍스트가 LLM에 주입되어 성능 저하
3. **컨텍스트 길이 제한**: Vicuna의 2,048 토큰 제한으로 $k$와 $n$의 조합에 제약
4. **MLLM 성능 저하**: 파인튜닝 후 일부 일반 벤치마크(MME 등)에서 성능 감소
5. **멀티모달 패시지 미활용**: 현재 텍스트 패시지만 활용하며 이미지 등 멀티모달 문서 활용은 미완

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 데이터 혼합(Data Mixing) 전략

외부 지식 학습이 기존 능력을 저하시키는 catastrophic forgetting 문제를 완화하기 위해, Wiki-LLaVA는 **LLaVA-Instruct 데이터셋(158k 샘플)과 외부 지식 데이터를 미니배치 내에서 2:1 비율로 혼합**합니다.

$$\text{Fine-tuning data} = \alpha \cdot \mathcal{D}_{\text{ext}} + (1-\alpha) \cdot \mathcal{D}_{\text{LLaVA-Instruct}}$$

**Table 3 결과:**
- InfoSeek에서 LLaVA-Instruct 혼합 시 **+1.9%p** 향상
- Encyclopedic-VQA에서는 개선 없으나 성능 저하도 없음

### 3.2 일반 벤치마크 성능 보존 (Table 4)

| Fine-tuning 설정 | MME Cogn | MME Perc | MMMU | MMB | POPE F1 |
|---|---|---|---|---|---|
| 없음 (원본) | 355.7 | 1513.3 | 35.1 | 86.9 | 85.8 |
| Enc-VQA only | 200.7 | 802.8 | 36.6 | 72.9 | 63.4 |
| Enc-VQA + LLaVA-Instruct | 290.0 | 1170.1 | 36.6 | 87.2 | **86.6** |
| InfoSeek + LLaVA-Instruct | 341.3 | 1438.9 | 35.6 | 85.8 | 84.2 |

→ **LLaVA-Instruct 혼합이 일반화 성능 보존에 핵심적**임을 입증

### 3.3 Unseen 엔티티/질문에 대한 일반화

InfoSeek 벤치마크는 훈련에 없는 엔티티(Unseen-E)와 질문(Unseen-Q)을 별도 평가합니다:

| 모델 | Unseen-Q | Unseen-E | All |
|---|---|---|---|
| LLaVA-1.5 (zero-shot) | 9.6 | 9.4 | 9.5 |
| LLaVA-1.5 (fine-tuned) | 19.4 | 16.7 | 17.9 |
| Wiki-LLaVA (k=1, n=3) | **30.1** | **27.8** | **28.9** |

→ **외부 KB 활용이 unseen 엔티티/질문에 대한 일반화를 크게 향상시킴**

### 3.4 도메인 적응성 (Domain Adaptability)

RAG 방식의 본질적 장점으로, **사전학습 파라미터를 수정하지 않고 KB만 교체**하면 다른 도메인에 적응 가능합니다. 예를 들어 의료 Wikipedia, 법률 문서 등으로 KB를 교체하면 해당 도메인에서의 일반화 성능을 기대할 수 있습니다.

---

## 4. 2020년 이후 최신 연구 비교 분석

### 4.1 관련 연구 계보

| 연구 | 연도 | 방법 | 비교 포인트 |
|---|---|---|---|
| **REALM** (Guu et al., ICML 2020) | 2020 | 텍스트 LM + latent knowledge retriever | 순수 텍스트 RAG의 시초 |
| **Flamingo** (Alayrac et al., NeurIPS 2022) | 2022 | Cross-attention으로 시각 특징 LLM 통합 | 멀티모달 통합, 검색 없음 |
| **BLIP-2** (Li et al., arXiv 2023) | 2023 | Q-Former로 이미지-텍스트 융합 | 검색 미사용, zero-shot |
| **REVEAL** (Hu et al., CVPR 2023) | 2023 | 멀티소스 멀티모달 지식 메모리 | 멀티모달 RAG이나 MLLM 아님 |
| **LLaVA-1.5** (Liu et al., arXiv 2023) | 2023 | Instruction tuning 기반 MLLM | Wiki-LLaVA의 베이스라인 |
| **InstructBLIP** (Dai et al., arXiv 2023) | 2023 | Instruction tuning + Q-Former | 검색 미사용 |
| **SnapNTell** (Qiu et al., arXiv 2024) | 2024 | RAG + Multimodal LLM (엔티티 중심) | Wiki-LLaVA와 가장 유사한 동시기 연구 |
| **Wiki-LLaVA** (Caffagni et al., 2024) | 2024 | 계층적 RAG + MLLM (LLaVA 기반) | **본 논문** |

### 4.2 핵심 차별점

```
텍스트 RAG (REALM, WebGPT)
        ↓ 확장
멀티모달 RAG (REVEAL)
        ↓ + MLLM backbone
Wiki-LLaVA ← 최초의 MLLM + 계층적 RAG 통합
```

**Wiki-LLaVA vs REVEAL (CVPR 2023):**
- REVEAL: 멀티소스 지식 메모리를 사전학습에 통합 → 구조 변경 필요
- Wiki-LLaVA: MLLM 구조를 변경하지 않고 입력 컨텍스트 증강 → 더 유연함

**Wiki-LLaVA vs SnapNTell (arXiv 2024):**
- 두 연구 모두 엔티티 기반 RAG + MLLM 방식
- Wiki-LLaVA는 계층적 2단계 검색 강조
- SnapNTell은 엔티티 중심 VQA에 특화

---

## 5. 향후 연구에 미치는 영향과 고려할 점

### 5.1 향후 연구에 미치는 영향

#### (1) RAG-MLLM 패러다임의 확립
Wiki-LLaVA는 **MLLM에 RAG를 적용하는 연구 방향의 선구자적 역할**을 합니다. 이후 연구들이 더 정교한 검색 메커니즘, 더 큰 KB, 멀티모달 패시지 검색 등을 탐구하는 데 기반이 됩니다.

#### (2) 지식 집약적 멀티모달 벤치마크 연구 촉진
Encyclopedic-VQA, InfoSeek에서의 실험은 **외부 지식이 필요한 VQA 연구**의 표준 평가 프로토콜을 제시합니다.

#### (3) 데이터 혼합 전략의 일반화
LLaVA-Instruct와의 데이터 혼합 방식은 **도메인 특화 학습 시 일반 능력 보존**을 위한 실용적 레시피로 활용 가능합니다.

#### (4) 파라미터 효율적 적응(PEFT) + RAG의 시너지
LoRA를 활용한 효율적 파인튜닝과 RAG의 결합은 **경량화된 지식 증강 모델** 연구의 방향을 제시합니다.

---

### 5.2 앞으로 연구 시 고려할 점

#### (1) 검색 모듈 개선
- **CLIP의 한계**: 대규모 KB(2M)에서 R@1 = 3.3%는 실용성이 낮음. 더 강력한 비주얼-텍스트 검색 모델(예: BLIP-2 피처, DINOv2 등) 탐구 필요
- **멀티모달 쿼리 검색**: 이미지+질문의 멀티모달 쿼리로 검색 정확도 향상 가능

$$\text{sim}(I, q, t_i) = f(E_v(I), E_t(q)) \cdot E_t(t_i)^T$$

#### (2) 노이즈 패시지 필터링
잘못된 엔티티에서 검색된 패시지가 MLLM 성능을 저하시키므로, **신뢰도 기반 패시지 필터링** 또는 **MLLM이 패시지의 관련성을 스스로 판단하는 메커니즘** 연구 필요

#### (3) 멀티모달 패시지 활용
현재 텍스트 패시지만 활용하지만, Wikipedia의 이미지, 표, 인포그래픽 등 **멀티모달 문서 청크**를 활용하면 더 풍부한 컨텍스트 제공 가능

#### (4) 컨텍스트 길이 한계 극복
Vicuna의 2,048 토큰 제한을 극복하기 위해:
- **긴 컨텍스트 LLM** 활용 (예: LLaMA-3, Gemini 등의 128k+ 컨텍스트)
- **압축된 패시지 표현** (예: RAG-Token 방식)

#### (5) 동적 검색 전략
현재는 고정된 $k$, $n$ 값을 사용하지만, **질문 복잡도에 따라 검색 깊이를 동적으로 조절**하는 적응적 검색 전략 연구 가치 있음

```math
k^*, n^* = \arg\max_{k,n} \text{Relevance}(q, \mathcal{D}, k, n)
```

#### (6) Hallucination 억제
검색된 패시지가 부정확할 경우 오히려 hallucination을 증폭시킬 수 있으므로, **검색-생성 일관성 검증 메커니즘** 연구 필요

#### (7) Two-hop 및 다단계 추론
논문에서 two-hop 질문을 실험에서 제외했는데, **다단계 검색과 추론이 결합된 아키텍처** 설계가 향후 중요한 과제

---

## 참고자료 (출처)

본 답변은 다음 자료를 기반으로 작성되었습니다:

1. **주 논문**: Caffagni, D., Cocchi, F., Moratelli, N., Sarto, S., Cornia, M., Baraldi, L., & Cucchiara, R. (2024). *Wiki-LLaVA: Hierarchical Retrieval-Augmented Generation for Multimodal LLMs*. arXiv:2404.15406v2 [cs.CV]. (제공된 PDF 원문)

2. **참조 논문들** (논문 내 인용):
   - Liu et al. (2023). *Visual Instruction Tuning* (LLaVA). NeurIPS 2023.
   - Liu et al. (2023). *Improved Baselines with Visual Instruction Tuning* (LLaVA-1.5). arXiv:2310.03744.
   - Li et al. (2023). *BLIP-2*. arXiv:2301.12597.
   - Dai et al. (2023). *InstructBLIP*. arXiv:2305.06500.
   - Hu et al. (2023). *REVEAL: Retrieval-Augmented Visual-Language Pre-Training*. CVPR 2023.
   - Izacard et al. (2021). *Unsupervised Dense Information Retrieval with Contrastive Learning* (Contriever). arXiv:2112.09118.
   - Guu et al. (2020). *Retrieval Augmented Language Model Pre-Training* (REALM). ICML 2020.
   - Radford et al. (2021). *Learning Transferable Visual Models from Natural Language Supervision* (CLIP). ICML 2021.
   - Mensink et al. (2023). *Encyclopedic VQA*. ICCV 2023.
   - Chen et al. (2023). *InfoSeek*. EMNLP 2023.
   - Hu et al. (2021). *LoRA: Low-Rank Adaptation of Large Language Models*. arXiv:2106.09685.
   - Qiu et al. (2024). *SnapNTell*. arXiv:2403.04735.
   - Alayrac et al. (2022). *Flamingo*. NeurIPS 2022.

> **주의**: 본 답변에서 Table 4의 일부 수치 및 비교 분석은 제공된 PDF 원문에 직접 기재된 내용을 기반으로 하며, 논문에 명시되지 않은 사항에 대해서는 추론을 삼가고 원문 내용만을 기술하였습니다.
