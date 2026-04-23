# MA-LMM: Memory-Augmented Large Multimodal Model for Long-Term Video Understanding

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

MA-LMM은 기존 Large Multimodal Model(LMM)이 갖는 **컨텍스트 길이 제한**과 **GPU 메모리 한계**를 해결하여, 장시간 비디오(Long-Term Video)를 효율적으로 이해하기 위한 모델이다. 핵심 아이디어는 모든 프레임을 동시에 처리하는 대신, **온라인(순차적) 방식**으로 프레임을 처리하면서 과거 정보를 **장기 메모리 뱅크(Long-Term Memory Bank)**에 저장·참조하는 것이다.

### 주요 기여

| 기여 | 설명 |
|------|------|
| 장기 메모리 뱅크 설계 | Visual Memory Bank + Query Memory Bank로 구성된 플러그앤플레이 모듈 |
| GPU 메모리 효율화 | 온라인 처리로 입력 토큰 수를 $N \times T$에서 $N$으로 대폭 감소 |
| Memory Bank Compression (MBC) | 코사인 유사도 기반 인접 프레임 병합으로 메모리 길이 고정 |
| 다양한 태스크에서 SOTA | Long-video understanding, Video QA, Video Captioning에서 최고 성능 달성 |
| Off-the-shelf 적용 가능 | 기존 멀티모달 LLM에 재훈련 없이 통합 가능 |

---

## 2. 상세 분석

### 2-1. 해결하고자 하는 문제

기존 LLM 기반 멀티모달 모델(Video-LLaMA, VideoChat 등)의 한계:

1. **컨텍스트 길이 제한**: LLaMA의 최대 컨텍스트 길이는 2048토큰이며, LLaVA는 이미지당 256토큰, BLIP-2는 32토큰 사용. 긴 비디오의 경우 수천~수만 토큰이 필요하여 처리 불가.
2. **GPU 메모리 폭발**: 프레임 수에 비례하여 GPU 메모리가 급증.
3. **단순 해결책의 한계**: 평균 풀링(Average Pooling)은 시간적 모델링 부재, 추가 Video Q-Former는 파라미터 증가 및 온라인 처리 불가.

---

### 2-2. 제안 방법 (수식 포함)

#### (1) 시각 특징 추출 (Visual Feature Extraction)

$T$개의 비디오 프레임을 사전학습된 Visual Encoder(ViT-G/14, EVA-CLIP)에 입력하여 시각 특징을 추출한다.

$$V = [v_1, v_2, \ldots, v_T], \quad v_t \in \mathbb{R}^{P \times C}$$

여기서 $P$는 프레임당 패치 수, $C$는 채널 차원이다. 이후 시간적 순서 정보를 주입한다:

$$f_t = v_t + PE(t), \quad f_t \in \mathbb{R}^{P \times C} $$

#### (2) 장기 시간 모델링 (Long-Term Temporal Modeling)

**Visual Memory Bank**

현재 타임스텝 $t$까지의 시각 특징을 축적:

$$F_t = \text{Concat}[f_1, f_2, \ldots, f_t], \quad F_t \in \mathbb{R}^{tP \times C}$$

Q-Former의 Cross-Attention에서 Key, Value로 활용:

$$Q = z_t W_Q, \quad K = F_t W_K, \quad V = F_t W_V $$

$$O = \text{Attn}(Q, K, V) = \text{Softmax}\left(\frac{QK^T}{\sqrt{C}}\right)V $$

**Query Memory Bank**

각 타임스텝의 학습된 쿼리를 축적:

$$Z_t = \text{Concat}[z_1, z_2, \ldots, z_t], \quad Z_t \in \mathbb{R}^{tN \times C}$$

Self-Attention에서 Key, Value로 활용:

$$Q = z_t W_Q, \quad K = Z_t W_K, \quad V = Z_t W_V $$

Visual Memory Bank는 **정적(static)** 원시 시각 특징을 저장하는 반면, Query Memory Bank는 Q-Former의 학습 과정에서 진화하는 **동적(dynamic)** 표현을 저장한다.

#### (3) Memory Bank Compression (MBC)

메모리 뱅크 길이가 임계값 $M$을 초과할 경우 압축 수행.

**Step 1**: 공간 위치 $i$에서 인접 프레임 간 코사인 유사도 계산:

$$s_t^i = \cos(f_t^i, f_{t+1}^i), \quad t \in [1, M], \quad i \in [1, P] $$

**Step 2**: 시간 축에서 가장 유사도가 높은(가장 중복된) 위치 선택:

$$k = \text{argmax}_t(s_t^i) $$

**Step 3**: 선택된 토큰 평균으로 메모리 길이를 1 감소:

$$\hat{f}_k^i = (f_k^i + f_{k+1}^i) / 2 $$

이 방식은 FIFO(선입선출)와 달리 **초기 정보를 보존**하면서 중복을 제거한다.

#### (4) 텍스트 디코딩 (Text Decoding)

훈련 목표: 표준 크로스 엔트로피 손실

$$\mathcal{L} = -\frac{1}{S}\sum_{i=1}^{S} \log P(w_i | w_{ < i}, V) $$

여기서 $V$는 입력 비디오, $w_i$는 $i$번째 정답 텍스트 토큰이다. Q-Former만 파인튜닝하고, Visual Encoder와 LLM은 동결(frozen)한다.

---

### 2-3. 모델 구조

```
[입력 비디오 프레임 (순차 처리)]
        ↓
[Frozen Visual Encoder: EVA-CLIP ViT-G/14]
        ↓ f_t (+ Position Embedding)
[Visual Memory Bank: F_t = [f_1, ..., f_t]]
        ↓ (Cross-Attention의 K, V)
[Trainable Q-Former (InstructBLIP 초기화)]
  - Cross-Attention ← Visual Memory Bank
  - Self-Attention  ← Query Memory Bank
        ↓ (32 tokens)
[Frozen LLM: Vicuna-7B]
        ↓
[텍스트 출력]
```

**주요 구현 세부사항:**
- Visual Encoder: ViT-G/14 (EVA-CLIP), 고정(frozen)
- Q-Former: InstructBLIP의 사전학습 가중치 초기화, 파인튜닝
- LLM: Vicuna-7B 또는 FlanT5-XL, 고정(frozen)
- 학습 프레임워크: 4× A100 GPU
- 메모리 뱅크 길이: 장기 비디오 태스크 20, 비디오 캡셔닝 40

---

### 2-4. 성능 향상

#### Long-Term Video Understanding (LVU 데이터셋)

| 모델 | 평균 Top-1 정확도 |
|------|-----------------|
| S5 [Wang et al., 2023] | 59.2% |
| **MA-LMM (Ours)** | **63.0%** (+3.8%p) |

#### Breakfast / COIN 데이터셋

| 모델 | Breakfast | COIN |
|------|-----------|------|
| S5 | 90.7% | 90.8% |
| **MA-LMM** | **93.0%** (+2.3%p) | **93.2%** (+2.4%p) |

#### Video QA (MSRVTT / MSVD)

| 모델 | MSRVTT | MSVD |
|------|--------|------|
| Video-LLaMA | 46.5% | 58.3% |
| **MA-LMM** | **48.5%** | **60.6%** |

#### 다양한 압축 방법 비교 (100프레임, LVU 기준)

| 방법 | #Token | GPU(GB) | LVU |
|------|--------|---------|-----|
| Concat | 1920 | 49.2 | 62.6% |
| Avg Pool | 32 | 21.2 | 57.6% |
| FIFO | 32 | 19.1 | 61.3% |
| **MBC (Ours)** | **32** | **19.1** | **63.0%** |

---

### 2-5. 한계

1. **처리 시간 증가**: 온라인(순차적) 처리 방식으로 인해 GPU 메모리는 절감되나, 처리 시간이 프레임 수에 비례하여 선형적으로 증가.
2. **이미지 기반 사전학습**: Visual Encoder와 Q-Former가 이미지-텍스트 데이터로 사전학습되어, 단기 시간 동역학(short-term dynamics) 포착에 한계.
3. **대규모 비디오-텍스트 사전학습 부재**: VideoCoCa처럼 HowTo100M 등 대규모 비디오-텍스트 데이터셋으로 사전학습하지 않아 일부 태스크(ActivityNet-QA)에서 성능 열위.
4. **초장시간 비디오**: 수십 분~수 시간의 극도로 긴 비디오에 대한 처리 전략(계층적 분할 등)이 미완성.
5. **Audio 미지원**: 비디오의 오디오 정보를 활용하지 않음.

---

## 3. 모델의 일반화 성능 향상 가능성

### 3-1. Off-the-Shelf 적용을 통한 범용성

MA-LMM의 메모리 뱅크는 **재훈련 없이** 기존 멀티모달 LLM에 삽입 가능하다. 논문의 Table 7에서 확인되듯, 베이스라인(InstructBLIP) 대비 메모리 뱅크 추가만으로:

| 데이터셋 | 메모리 뱅크 없음 | 메모리 뱅크 있음 | 향상 |
|---------|--------------|--------------|------|
| ActivityNet | 29.9% | 37.2% | +7.3%p |
| LVU | 23.6% | 32.8% | +9.2%p |

이는 다양한 아키텍처에 즉각 적용 가능한 **범용 모듈**로서의 높은 일반화 가능성을 보여준다.

### 3-2. 다양한 LLM 백본 지원

Table 9에서 FlanT5-XL(인코더-디코더 구조)과 Vicuna-7B(디코더 전용 구조) 모두에서 효과적임이 검증되었다. 이는 특정 LLM 구조에 종속되지 않음을 의미한다.

### 3-3. 멀티태스크 일반화

Short-video(MSRVTT: 10~15초), Long-video(LVU: 1~3분, Breakfast: ~2.7분), Online prediction(EpicKitchens-100) 등 다양한 길이와 태스크에서 일관된 성능 향상을 보여 **태스크 일반화 능력**이 우수하다.

### 3-4. 일반화 향상을 위한 미래 방향 (논문 내 제시)

- **Video/CLIP 기반 인코더 교체**: 이미지 기반 ViT를 비디오 인코더로 대체하면 단기 시간 동역학 포착 능력 향상.
- **대규모 비디오-텍스트 사전학습**: HowTo100M 등 대규모 비디오 데이터로 사전학습 시 일반화 성능 대폭 향상 예상.
- **고급 LLM 통합**: GPT-4급 모델 통합 시 복잡한 비디오 추론 능력 향상.
- **계층적 처리**: 초장시간 비디오를 세그먼트로 나누어 처리 후 세그먼트 간 관계 모델링.

### 3-5. Token-Level 압축의 일반화 기여

Table 10에서 토큰 레벨 압축(63.0%, 93.0%, 93.2%)이 프레임 레벨 압축(61.8%, 86.5%, 91.1%)보다 일관되게 우수함이 확인된다. 세밀한 공간 정보 보존이 다양한 도메인에서의 일반화를 돕는다.

---

## 4. 2020년 이후 관련 최신 연구 비교 분석

| 모델 | 연도 | 시간 모델링 방식 | 장기 비디오 지원 | GPU 효율성 | 온라인 처리 |
|------|------|----------------|--------------|-----------|-----------|
| **Flamingo** (Alayrac et al., NeurIPS 2022) | 2022 | Cross-attention + 프레임 연결 | 제한적 | 낮음 | ✗ |
| **BLIP-2** (Li et al., arXiv 2023) | 2023 | Q-Former (단일 프레임) | ✗ | 중간 | ✗ |
| **Video-LLaMA** (Zhang et al., arXiv 2023) | 2023 | Video Q-Former (추가) | 제한적 | 낮음 | ✗ |
| **Video-ChatGPT** (Maaz et al., arXiv 2023) | 2023 | 시공간 평균 풀링 | ✗ | 중간 | ✗ |
| **ViS4mer** (Islam & Bertasius, ECCV 2022) | 2022 | State-space model (S4) | ✓ | 높음 | ✗ |
| **S5** (Wang et al., CVPR 2023) | 2023 | Selective SSM | ✓ | 높음 | ✗ |
| **MeMViT** (Wu et al., CVPR 2022) | 2022 | Memory-augmented ViT | ✓ | 높음 | ✓ |
| **TESTA** (Ren et al., EMNLP 2023) | 2023 | 공간-시간 토큰 집계 | ✓ | 중간 | ✗ |
| **MovieChat** (Song et al., CVPR 2024) | 2024 | 단기/장기 메모리 + Video Q-Former | ✓ | 중간 | ✗ |
| **Chat-UniVi** (Jin et al., CVPR 2024) | 2024 | 토큰 병합 + 시간축 연결 | 부분적 | 중간 | ✗ |
| **MA-LMM** (He et al., 2024) | 2024 | Visual+Query 메모리 뱅크 + MBC | ✓ | **매우 높음** | **✓** |

### 핵심 차별점 분석

**vs. Video-LLaMA**: Video-LLaMA는 추가 Video Q-Former를 처음부터 학습시켜 파라미터 수 증가 및 GPU 메모리 과부하 발생. MA-LMM은 이미지 Q-Former를 파인튜닝하여 효율적.

**vs. MeMViT**: MeMViT은 학습 가능한 풀링 연산자로 메모리를 압축하여 추가 파라미터 필요. MA-LMM의 MBC는 코사인 유사도 기반의 파라미터 무추가(parameter-free) 압축.

**vs. MovieChat**: MovieChat은 단기 메모리와 장기 메모리를 분리하고 Video Q-Former로 전역 시간 상호작용을 모델링. MA-LMM은 인과적 자기 어텐션(causal self-attention)으로 온라인 처리를 자연스럽게 지원.

**vs. TESTA/Chat-UniVi**: 두 모델은 오프라인 처리만 지원. MA-LMM은 온라인 처리 지원으로 실시간 응용 가능.

---

## 5. 앞으로의 연구에 미치는 영향과 고려할 점

### 5-1. 앞으로의 연구에 미치는 영향

**① 온라인 멀티모달 처리 패러다임 제시**
대부분의 비디오 LMM이 오프라인(일괄 처리) 방식인 반면, MA-LMM은 온라인 처리의 실용성을 증명하였다. 이는 로보틱스, AR/VR, 실시간 스트리밍 등 시간 민감 응용 분야의 연구를 촉진할 것이다.

**② 플러그앤플레이 메모리 모듈의 표준화 가능성**
메모리 뱅크를 off-the-shelf 모듈로 제공하는 방식은, 향후 더 강력한 LMM(GPT-4V, LLaMA-3 등)에도 손쉽게 통합 가능한 표준 컴포넌트로 발전할 수 있다.

**③ 토큰 효율성 연구 촉진**
MBC의 코사인 유사도 기반 압축은 Token Merging(ToMe) 아이디어를 비디오의 시간 축으로 확장한 것으로, 비디오 토큰 효율화 연구의 새로운 방향을 제시한다.

**④ 장기 비디오 이해 벤치마크의 중요성 부각**
LVU, Breakfast, COIN 외에도 더 어렵고 다양한 장기 비디오 벤치마크 개발의 필요성을 제기한다.

### 5-2. 앞으로 연구 시 고려할 점

**① 비디오-텍스트 대규모 사전학습**

현재 MA-LMM은 이미지-텍스트 데이터로만 사전학습된 가중치를 사용한다. HowTo100M, VideoCC3M, Kinetics 등 대규모 비디오-텍스트 데이터로의 사전학습이 일반화 성능 향상에 결정적일 것이다.

**② 비디오 인코더 강화**

현재의 이미지 기반 ViT-G/14 대신 비디오 특화 인코더(VideoMAE, InternVideo 등)를 사용하면 단기 모션 동역학 포착 능력이 향상될 것이다.

**③ 계층적 메모리 구조**

수 시간 길이의 영화, 드라마 등 초장시간 비디오에 대응하기 위한 계층적(hierarchical) 메모리 설계가 필요하다. 예를 들어, 세그먼트 내 메모리와 세그먼트 간 메모리를 분리 관리하는 방식이 고려될 수 있다.

**④ 적응적 메모리 관리(Adaptive Memory Management)**

현재 메모리 뱅크 길이 $M$은 고정된 하이퍼파라미터이다. 비디오 내용의 복잡도에 따라 메모리 길이를 동적으로 조절하는 적응적 방식이 성능을 더 향상시킬 수 있다.

**⑤ 멀티모달 입력 확장**

오디오, 자막(subtitle), 메타데이터 등 추가 모달리티를 메모리 뱅크에 통합하면 이해 능력이 향상될 것이다. Video-LLaMA가 오디오를 통합한 것처럼, MA-LMM도 이를 고려해야 한다.

**⑥ 긴 컨텍스트 LLM 활용**

LLaMA-3, Mistral 등 컨텍스트 길이가 확장된 최신 LLM과의 통합 시, 메모리 뱅크의 설계를 재검토할 필요가 있다. 컨텍스트 제한이 완화됨에 따라 메모리 뱅크와 LLM 컨텍스트의 역할 분담을 재정의해야 한다.

**⑦ 인과성(Causality)과 미래 예측**

현재 온라인 처리는 과거 정보만 참조하는 인과적 구조이다. 향후 연구에서는 다음 행동 예측, 비디오 이상 탐지 등 미래 지향적 추론 능력을 강화하는 방향이 유망하다.

**⑧ 공정한 비교를 위한 표준 벤치마크 필요**

다양한 모델들이 서로 다른 사전학습 데이터, 모델 크기, 평가 프로토콜을 사용하므로, 공정한 비교를 위한 표준화된 장기 비디오 이해 벤치마크의 구축이 시급하다.

---

## 참고 자료

**주 논문:**
- Bo He, Hengduo Li, Young Kyun Jang, et al. **"MA-LMM: Memory-Augmented Large Multimodal Model for Long-Term Video Understanding"**, arXiv:2404.05726v2, 2024.

**논문 내 인용 주요 참고문헌:**
- Junnan Li et al. **"BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models"**, arXiv:2301.12597, 2023.
- Wenliang Dai et al. **"InstructBLIP: Towards General-Purpose Vision-Language Models with Instruction Tuning"**, 2023.
- Hang Zhang et al. **"Video-LLaMA: An Instruction-tuned Audio-Visual Language Model for Video Understanding"**, arXiv:2306.02858, 2023.
- Chao-Yuan Wu et al. **"MeMViT: Memory-Augmented Multiscale Vision Transformer for Efficient Long-Term Video Recognition"**, CVPR 2022.
- Jue Wang et al. **"S5: Selective Structured State-Spaces for Long-Form Video Understanding"**, CVPR 2023.
- Daniel Bolya et al. **"Token Merging: Your ViT but Faster"**, arXiv:2210.09461, 2022.
- Shuhuai Ren et al. **"TESTA: Temporal-Spatial Token Aggregation for Long-form Video-Language Understanding"**, EMNLP 2023.
- Enxin Song et al. **"MovieChat: From Dense Token to Sparse Memory for Long Video Understanding"**, CVPR 2024.
- Peng Jin et al. **"Chat-UniVi: Unified Visual Representation Empowers Large Language Models with Image and Video Understanding"**, CVPR 2024.
- Md Mohaiminul Islam & Gedas Bertasius. **"ViS4mer: Long Movie Clip Classification with State-Space Video Models"**, ECCV 2022.
- Chao-Yuan Wu & Philipp Krahenbuhl. **"Towards Long-Form Video Understanding"**, CVPR 2021.
