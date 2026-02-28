# FastVLM: Efficient Vision Encoding for Vision Language Models

---

## 1. 핵심 주장 및 주요 기여 요약

**핵심 주장:** Vision Language Model(VLM)에서 고해상도 이미지를 처리할 때, 기존 ViT 기반 비전 인코더는 토큰 수 증가와 인코딩 지연(latency)으로 인해 비효율적이다. FastVLM은 새로운 하이브리드 비전 인코더 **FastViTHD**를 도입하여, 해상도-지연-정확도 간의 최적 트레이드오프를 달성한다.

**주요 기여:**
1. **하이브리드 비전 백본의 우수성 입증:** Convolution + Transformer 구조가 순수 ViT 대비 VLM에서 더 효율적임을 실증
2. **FastViTHD 아키텍처 설계:** 5단계 구조(기존 4단계 대비 추가 다운샘플링 스테이지)를 통해 ViT-L/14 대비 $16\times$ 적은 토큰, $3.2\times$ 빠른 TTFT(Time-To-First-Token) 달성
3. **체계적 효율성 분석:** 해상도, 비전 인코더 지연, 토큰 수, LLM 크기 간의 Pareto 최적 곡선을 실제 하드웨어(M1 MacBook Pro)에서 실측
4. LLaVA-OneVision(1152×1152) 대비 동일 0.5B LLM에서 **85× 빠른 TTFT**, **3.4× 작은 비전 인코더**로 SeedBench, MMMU, DocVQA에서 더 나은 성능 달성

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

VLM의 성능 향상을 위해서는 고해상도 이미지 입력이 필수적이나, 이는 두 가지 병목을 야기한다:

1. **비전 인코더 지연 증가:** ViT의 self-attention 복잡도는 토큰 수 $N$에 대해 $O(N^2)$이며, 해상도가 올라갈수록 토큰 수가 급증
2. **LLM 프리필링 시간 증가:** 시각 토큰이 많아지면 LLM이 처리해야 할 컨텍스트가 길어져 TTFT가 증가

TTFT(Time-To-First-Token)는 다음과 같이 정의된다:

$$\text{TTFT} = T_{\text{vision}} + T_{\text{prefill}}$$

여기서 $T_{\text{vision}}$은 비전 인코더의 추론 지연, $T_{\text{prefill}}$은 LLM의 프리필링 시간(시각 토큰 + 텍스트 토큰에 대한 forward pass 시간)이다.

기존 방법의 한계:
- **ViT 기반 인코더** (SigLIP-SO400M, ViT-L/14): patch size 14로 $336 \times 336$ 입력 시 $576$개 토큰 생성, 고해상도에서는 수천 개의 토큰 발생
- **타일링(AnyRes) 전략**: 이미지를 분할하여 독립 처리하지만, 여러 번의 인코더 추론이 필요하고 "의미적 단절(semantic breaks)"로 인해 성능 저하
- **토큰 프루닝**: 후처리 방식으로 토큰을 줄이지만, 정보 손실이 발생하며 추가 모듈 필요

### 2.2 제안하는 방법

#### (A) FastViT를 VLM 비전 인코더로 활용

FastViT(MobileCLIP의 MCi2 인코더, 35.7M 파라미터)는 하이브리드 아키텍처로, 계층적(hierarchical) 다운샘플링 구조를 통해 ViT 대비 토큰을 대폭 줄인다. 입력 해상도 $H \times W$에 대해:

- **ViT-L/14**: 패치 크기 $p = 14$일 때 토큰 수 $= \frac{H}{p} \times \frac{W}{p}$

$$N_{\text{ViT}} = \left(\frac{H}{14}\right) \times \left(\frac{W}{14}\right)$$

- **FastViT**: 전체 다운샘플링 비율이 $32\times$이므로:

$$N_{\text{FastViT}} = \left(\frac{H}{32}\right) \times \left(\frac{W}{32}\right)$$

따라서 동일 해상도에서 FastViT는 ViT 대비 약 $\left(\frac{32}{14}\right)^2 \approx 5.2\times$ 적은 토큰을 생성한다.

**다중 스케일 특징(Multi-Scale Features):** 계층적 아키텍처의 각 스테이지에서 추출한 특징을 학습 가능한 풀링(Depthwise Convolution)으로 다운샘플링한 후 채널 방향으로 결합하여 LLM에 전달한다. 이는 다양한 세밀도의 정보를 통합한다.

#### (B) FastViTHD: 고해상도 VLM을 위한 새로운 인코더

기존 4단계 하이브리드 구조의 한계를 극복하기 위해 **5단계** 아키텍처를 설계:

| 구성 요소 | Stage 1 | Stage 2 | Stage 3 | Stage 4 | Stage 5 |
|---------|---------|---------|---------|---------|---------|
| **블록 유형** | RepMixer | RepMixer | RepMixer | Self-Attention | Self-Attention |
| **깊이** | 2 | 12 | 24 | 4 | 2 |
| **임베딩 차원** | 96 | 192 | 384 | 768 | 1536 |

- 총 다운샘플링 비율: $64\times$ (Stem $4\times$ + 각 스테이지 간 패치 임베딩 $2\times$ × 4회)
- Self-attention은 $64\times$ 다운샘플링된 텐서에서 수행되므로 계산 비용이 크게 감소
- 총 파라미터: **125.1M** (ViT-L/14의 304M 대비 $2.4\times$ 작음)

토큰 수 비교 (동일 해상도 $H \times W$):

$$N_{\text{FastViTHD}} = \left(\frac{H}{64}\right) \times \left(\frac{W}{64}\right)$$

이는 ViT-L/14 대비 $\left(\frac{64}{14}\right)^2 \approx 20.9\times$, FastViT 대비 $\left(\frac{64}{32}\right)^2 = 4\times$ 적은 토큰을 생성한다. 예를 들어 해상도 $336 \times 336$에서:
- ViT-L/14: $576$ 토큰
- FastViTHD: $\left(\frac{336}{64}\right)^2 \approx 25$ 토큰 (실제 $1024 \times 1024$에서 $256$ 토큰 생성)

#### (C) CLIP 사전학습

FastViTHD는 MobileCLIP [83]의 학습 설정을 따라 **DataCompDR-1B** 데이터셋으로 CLIP 사전학습된다. 38개 멀티모달 제로샷 태스크에서 ViT-L/14(304M)과 동등한 성능을 **125M** 파라미터와 $6.9\times$ 빠른 속도로 달성한다 (Table 3).

#### (D) VLM 학습 파이프라인

LLaVA-1.5 [53] 프레임워크를 따르며, 최적 성능을 위해 4단계 학습을 수행:

| 단계 | 데이터 | 학습 대상 | 학습률 |
|------|------|---------|------|
| Stage 1 | LLaVA-558K 정렬 데이터 | Projector만 | $10^{-3}$ |
| Stage 1.5 | Recap-CC3M + CC12M (15M) | 전체 모델 | $2 \times 10^{-5}$ |
| Stage 2 | 1.1M~12.5M 시각 지시 튜닝 | 전체 모델 | $2 \times 10^{-5}$ |
| Stage 3 | MammothVL 10.6M (CoT 추론) | 전체 모델 | $2 \times 10^{-5}$ |

### 2.3 모델 구조

FastVLM의 전체 구조는 세 가지 구성요소로 이루어진다 (Figure 2):

1. **FastViTHD 비전 인코더:**
   - Convolutional Stem → 3개 RepMixer 스테이지 → 2개 Self-Attention 스테이지
   - 각 스테이지 사이에 stride-2 패치 임베딩으로 다운샘플링
   - 다중 스케일 특징: Stage 2, 3, 4의 특징을 학습 가능한 DWConv 풀링으로 최종 해상도에 맞춰 결합

2. **Connector (Projection Layer):**
   - 다중 스케일 특징을 채널 방향으로 결합(concatenation) 후 MLP 프로젝션
   - 비전 토큰을 LLM의 임베딩 공간으로 사상

3. **LLM 디코더:**
   - Qwen2 (0.5B/1.5B/7B) 또는 Vicuna-7B
   - 시각 토큰과 텍스트 토큰을 입력으로 받아 자기회귀적으로 응답 생성

### 2.4 성능 향상

#### 주요 벤치마크 결과 (Table 6, 7):

**LLaVA-1.5 설정 (Vicuna-7B) 비교:**

| 모델 | 해상도 | 토큰 수 | 인코더 크기 | TTFT(ms) | Avg-5 |
|------|--------|--------|----------|---------|-------|
| LLaVA-1.5 (ViT-L/14) | 336 | 576 | 304M | 1297 | 66.1 |
| ConvLLaVA (ConvNeXT-L) | 1536 | 576 | 200M | 2740 | 70.2 |
| **FastVLM** | **1024** | **256** | **125M** | **577** | **68.5~73.5** |

**0.5B LLM (Qwen2-0.5B) 비교:**

| 모델 | 해상도 | 토큰 수 | TTFT(ms) | SeedBench | DocVQA | MMMU |
|------|--------|--------|---------|-----------|--------|------|
| LLaVA-OneVision | 1152 | 7290 | 14124 | 65.5 | 70.0 | 31.4 |
| **FastVLM (R4)** | **1024** | **256** | **166** | **69.2** | **70.4** | **32.9** |

→ **85× 빠른 TTFT**, 주요 벤치마크에서 동등 이상의 성능

**토큰 프루닝 대비 (Table 5):**

FastViTHD는 $256 \times 256$ 해상도에서 단 **16개** 토큰으로도 Matryoshka M³ (9토큰), MQT (16토큰) 등 토큰 프루닝 기법보다 우수한 성능:

| 방법 | 토큰 수 | GQA | TextVQA | POPE |
|------|--------|-----|---------|------|
| ViT-L/14 M³ | 9 | 58.0 | - | 83.4 |
| ViT-L/14 MQT | 16 | 57.6 | - | 80.8 |
| **FastViTHD** | **16** | **60.6** | **53.1** | **82.3** |

**Pareto 최적성 (Figure 4):**
동일 런타임 예산에서 FastViTHD는 FastViT 대비 Avg-5 메트릭에서 **+2.5 포인트** 향상, 동일 성능을 **3× 빠르게** 달성한다.

### 2.5 한계

논문에서 직접적으로 명시된 한계와 분석에서 도출 가능한 한계를 종합하면:

1. **극고해상도에서의 정적 해상도 한계:** 1536×1536 이상에서는 메모리 대역폭 제한으로 정적 해상도보다 AnyRes(동적 해상도)가 더 효과적 (Figure 6)
2. **텍스트가 매우 작은 이미지:** DocVQA, ChartQA 등에서 텍스트가 너무 작거나 정밀한 정렬이 필요한 경우 실패 (Section E, Table 15-16)
3. **지식 기반 벤치마크 한계:** MMMU 같은 도메인 지식이 필요한 태스크에서는 비전 인코더보다 LLM 크기가 성능을 좌우 (Table 14)
4. **CLIP 사전학습 의존성:** FastViTHD의 성능은 CLIP 사전학습의 품질에 크게 의존하며, DataCompDR-1B 외의 데이터로의 확장은 미탐구
5. **비디오/다중 이미지 미지원:** 논문은 단일 이미지 입력에 초점을 맞추며, 비디오나 다중 이미지 시나리오에 대한 확장은 다루지 않음

---

## 3. 모델의 일반화 성능 향상 가능성

FastVLM의 일반화 성능 향상과 관련하여 논문에서 도출할 수 있는 핵심 내용은 다음과 같다:

### 3.1 해상도 스케일링을 통한 일반화

FastViTHD는 하이브리드 아키텍처의 합성곱 구성요소 덕분에 **네이티브 해상도 스케일링**이 가능하다. ViT와 달리 위치 임베딩의 보간 없이도 다양한 해상도에 적응 가능하며, 이는 다양한 해상도의 실세계 이미지에 대한 일반화를 용이하게 한다.

논문의 실험 결과에서:
- 해상도 $256 \rightarrow 512 \rightarrow 768 \rightarrow 1024$로 스케일링하면서 TextVQA와 DocVQA 같은 텍스트 집약 벤치마크에서 **지속적인 성능 향상** 관찰 (Table 4)
- 특히 텍스트 인식이 필요한 태스크에서 해상도 스케일링의 효과가 더 두드러짐: DocVQA에서 $256$ 해상도의 17.4에서 $1024$ 해상도의 35.6으로 **약 2배 향상**

### 3.2 데이터 스케일링과 일반화

논문은 사전학습 및 시각 지시 튜닝 데이터의 규모 확장에 따른 성능 향상을 체계적으로 보여준다 (Table 6, R19→R21→R29→R30):

| 데이터 규모 (PT+IT) | GQA | TextVQA | DocVQA | MMMU |
|-----------------|-----|---------|--------|------|
| 0.5M+0.6M (R19) | 62.4 | 62.9 | 32.9 | 34.9 |
| 0.5M+1.1M (R20) | 63.2 | 67.5 | 57.3 | 36.9 |
| 15M+1.1M (R21) | 65.0 | 69.4 | 65.5 | 37.0 |
| 15M+12.5M (R41) | 65.2 | 73.4 | 82.7 | 47.3 |

**핵심 발견:** FastViTHD는 ViT-L/14나 ViT-H보다 작은 모델임에도 불구하고, 데이터 스케일링에서 **유사한 확장(scaling) 추세**를 보인다. 이는 하이브리드 아키텍처가 작은 모델 크기에서도 충분한 표현 용량을 가짐을 시사한다.

### 3.3 LLM 디코더와의 상호작용을 통한 일반화

비전 인코더-LLM 디코더 간 상호작용 분석 (Section 3.2.1)에서 중요한 일반화 통찰이 도출된다:

- **소형 LLM + 고해상도** 조합은 차선적(suboptimal): 소형 LLM이 많은 시각 토큰을 효과적으로 활용할 수 없음
- **Pareto 최적 곡선**은 다양한 크기의 LLM으로 구성됨: 런타임 예산에 따라 최적의 (해상도, LLM 크기) 쌍이 다름
- Vicuna-7B → Qwen2-7B로 교체 시 MMVet, MMMU 등에서 **유의미한 개선** (R21 vs R22): 이는 비전 인코더의 일반화된 시각 표현이 더 강력한 LLM과 결합될 때 시너지가 발생함을 보여줌

### 3.4 토큰 효율성과 일반화의 관계

FastViTHD의 핵심적 일반화 이점은 **토큰 효율성**에서 비롯된다:

$$\text{토큰 품질} \propto \frac{\text{입력 해상도에서 추출한 정보량}}{\text{토큰 수}}$$

FastViTHD는 $1024 \times 1024$ 해상도에서 256개 토큰만으로 ViT-L/14의 $336 \times 336$ 해상도 576개 토큰보다 높은 성능을 달성한다 (Table 5). 이는 각 토큰이 더 풍부한 시각 정보를 담고 있으며, LLM이 적은 토큰으로도 효과적으로 추론할 수 있음을 의미한다.

### 3.5 다중 스케일 특징의 일반화 기여

다양한 스케일의 특징을 결합하면 세부 텍스처(저수준 특징)와 의미적 이해(고수준 특징)를 동시에 활용할 수 있어, **다양한 유형의 시각적 질문**에 대한 일반화가 향상된다 (Table 2):
- DWConv 풀링을 사용한 다중 스케일: GQA 63.0, POPE 86.8, SeedBench 67.4 (단일 스케일 대비 일관된 개선)

### 3.6 일반화 향상을 위한 추가 가능성

1. **더 큰 CLIP 사전학습 데이터셋** 활용: 현재 DataCompDR-1B 외에 LAION-5B 등으로 확장 시 제로샷 일반화 향상 기대
2. **도메인 특화 해상도 적응:** 의료, 위성 이미지 등 특정 도메인에서 해상도 조절을 통한 전이 학습
3. **다중 이미지/비디오로의 확장:** 효율적인 토큰 생성 특성은 다중 프레임 처리에서 더 큰 이점 제공 가능
4. **Chain-of-Thought 추론과의 결합:** Stage 3에서 MammothVL [30] CoT 데이터를 활용한 미세 조정이 MMMU, DocVQA에서 추가 개선을 보여줌 (R41→R42)

---

## 4. 향후 연구에 미치는 영향 및 고려사항

### 4.1 연구에 미치는 영향

1. **비전 인코더 설계 패러다임의 전환:** VLM에서 순수 ViT가 아닌 하이브리드 아키텍처가 더 효율적임을 실증적으로 보여줌. 이는 후속 연구에서 비전 인코더 선택 시 하이브리드 구조를 우선적으로 고려하도록 유도할 것이다.

2. **토큰 프루닝 vs 아키텍처 설계의 논쟁:** 후처리 방식의 토큰 프루닝보다 아키텍처 수준에서의 토큰 감소가 더 효과적임을 보여줌. 이는 "깨끗한 설계(clean design)"가 복잡한 후처리보다 우수할 수 있다는 메시지를 전달한다.

3. **온디바이스 VLM의 실현 가능성:** M1 MacBook Pro에서의 실측 벤치마크를 통해 모바일/엣지 디바이스에서의 VLM 배포 가능성을 구체적으로 제시. 이는 온디바이스 AI 연구를 가속화할 것이다.

4. **Pareto 최적 분석 프레임워크:** (해상도, LLM 크기, 토큰 수) 3차원 공간에서의 체계적 최적화 분석 방법론은 향후 VLM 설계에 표준적인 평가 프레임워크로 자리 잡을 수 있다.

5. **데이터 효율적 학습:** 기존 방법 대비 적은 학습 데이터로 동등 이상의 성능을 달성함으로써, 데이터 효율적인 VLM 학습 방법론에 대한 관심을 촉진할 것이다.

### 4.2 향후 연구 시 고려할 점

1. **다운샘플링 비율과 정보 손실의 균형:** FastViTHD의 $64\times$ 다운샘플링은 효율적이지만, 극도로 세밀한 시각 정보(예: 작은 텍스트, 미세한 패턴)에서 정보 손실이 발생할 수 있다. 적응적 다운샘플링 비율이나 영역별 차별적 해상도 처리 방법이 필요하다.

2. **비디오 및 다중 이미지 확장:** FastViTHD의 효율적 토큰 생성은 비디오 VLM에서 큰 이점이 될 수 있으나, 시간축 정보 통합 전략이 추가로 필요하다.

3. **사전학습 전략의 최적화:** CLIP 사전학습 외에 DINOv2, MAE 등 다양한 자기지도 학습 목적함수와의 조합이 일반화 성능을 더 향상시킬 수 있다.

4. **동적 토큰 할당:** 이미지의 복잡도에 따라 토큰 수를 동적으로 조절하는 메커니즘이 효율성과 정확도의 트레이드오프를 더 개선할 수 있다.

5. **양자화 및 경량화와의 결합:** FastViTHD의 하이브리드 구조는 양자화(quantization)와의 호환성이 높을 수 있으며, INT8/INT4 양자화를 통한 추가 가속 가능성이 있다.

6. **평가 벤치마크의 다양화:** 현재 대부분의 평가가 영어 기반이며, 다국어/다문화 시각 이해 벤치마크에서의 일반화 성능 검증이 필요하다.

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

### 5.1 비전 인코더 관점

| 연구 | 연도 | 비전 인코더 | 핵심 접근 | FastVLM과의 차이 |
|------|------|----------|---------|--------------|
| **CLIP** (Radford et al.) [69] | 2021 | ViT | 대규모 이미지-텍스트 대비학습 | FastVLM의 사전학습 기반이지만, ViT 대신 하이브리드 구조 채택 |
| **SigLIP** (Zhai et al.) [94] | 2023 | ViT-SO400M | Sigmoid 손실 기반 사전학습 | 430M → 125M으로 3.4× 축소, 동등 이상 성능 |
| **EVA-CLIP** (Sun et al.) [75] | 2023 | ViT | 개선된 CLIP 학습 기법 | 여전히 순수 ViT 기반으로 고해상도 비효율 |
| **ConvNeXT** (Liu et al.) [57] | 2022 | 순수 CNN | 현대적 CNN 설계 | FastVLM은 CNN+Transformer 하이브리드로, ConvNeXT-L 대비 1.6× 작고 2.3× 빠름 |
| **FastViT** (Vasu et al.) [82] | 2023 | 하이브리드 CNN-Transformer | RepMixer 구조 | FastVLM의 기반이나, 4단계→5단계로 확장하여 토큰 4× 감소 |
| **ViTamin** (Chen et al.) [12] | 2024 | 하이브리드 | VLM 시대를 위한 확장 가능 비전 모델 | FastViTHD가 2.7× 작고 5.6× 빠르면서 더 높은 검색 성능 |
| **MobileCLIP** (Vasu et al.) [83] | 2024 | FastViT 기반 | 다중모달 강화 학습 | FastViTHD의 사전학습 설정 기반, 고해상도 VLM에 최적화된 아키텍처로 확장 |

### 5.2 VLM 아키텍처 관점

| 연구 | 연도 | 핵심 기여 | FastVLM과의 비교 |
|------|------|---------|--------------|
| **LLaVA-1.5** (Liu et al.) [53] | 2023 | VLM 학습 파이프라인 표준화 | FastVLM의 학습 프레임워크 기반. 동일 설정에서 3.2× 빠른 TTFT |
| **LLaVA-OneVision** (Li et al.) [45] | 2024 | 다양한 시각 태스크 전이 | 동일 0.5B LLM에서 85× 빠른 TTFT로 SeedBench, MMMU, DocVQA 우위 |
| **MM1** (McKinzie et al.) [66] | 2024 | 대규모 VLM 사전학습 분석 | 3000M PT 데이터 vs FastVLM 15M, 유사 성능이나 5× 적은 토큰 |
| **ConvLLaVA** (Ge et al.) [28] | 2024 | 순수 CNN 인코더 VLM | FastVLM이 TextVQA +8.4%, DocVQA +12.5% 우위, 22% 빠름 |
| **Cambrian-1** (Tong et al.) [78] | 2024 | 다중 비전 인코더 앙상블 | 단일 인코더로 7.9× 빠르면서 동등 이상 성능 |
| **InternVL** (Chen et al.) [15, 16] | 2023-24 | 대규모 비전 기반 모델 | 36개 타일 사용 vs FastVLM 4개 타일(2×2), 훨씬 효율적 |
| **SmolVLM2** (Marafioti et al.) [62] | 2025 | 소형 효율적 VLM | FastVLM이 4.3~8.2× 적은 토큰으로 DocVQA, InfoVQA 우위 |
| **FlorenceVL** (Chen et al.) [11] | 2024 | 생성적 비전 인코더 | FastVLM이 2.3× 적은 토큰, 6.2× 작은 인코더로 TextVQA, DocVQA 우위 |

### 5.3 토큰 효율화 관점

| 연구 | 연도 | 방법 | FastVLM과의 비교 |
|------|------|------|--------------|
| **LLaVA-PruMerge** (Shang et al.) [70] | 2024 | 적응적 토큰 축소 | 40토큰에서 FastViTHD 16토큰보다 낮은 성능 (Table 5) |
| **Matryoshka M³** (Cai et al.) [7] | 2024 | 중첩 토큰 샘플링 | 36토큰에서 GQA 60.3 vs FastViTHD 64토큰 63.0 |
| **MQT** (Hu et al.) [32] | 2024 | Matryoshka 쿼리 트랜스포머 | 256토큰 GQA 61.6 vs FastViTHD 256토큰 63.1 |
| **VisionZip** (Yang et al.) [87] | 2024 | 시각 토큰 압축 | 64토큰 GQA 57.0 vs FastViTHD 64토큰 63.0 |
| **DynamicLLaVA** (Huang et al.) [33] | 2024 | 동적 컨텍스트 희소화 | 115토큰 GQA 61.4 vs FastViTHD 64토큰 63.0 |

**핵심 비교 통찰:** 후처리 기반 토큰 프루닝은 원래 ViT가 생성한 전체 토큰에서 정보를 선별하지만, FastViTHD는 아키텍처 수준에서 계층적으로 정보를 압축하여 각 토큰의 품질을 보장한다. 이는 동일 토큰 수에서 일관되게 더 높은 성능으로 이어진다.

---

## 참고자료

1. Vasu, P.K.A., Faghri, F., Li, C.-L., Koc, C., True, N., Antony, A., Santhanam, G., Gabriel, J., Grasch, P., Tuzel, O., Pouransari, H. "FastVLM: Efficient Vision Encoding for Vision Language Models." *arXiv:2412.13303v2*, 2025. (본 논문)
2. Liu, H., Li, C., Li, Y., Lee, Y.J. "Improved Baselines with Visual Instruction Tuning (LLaVA-1.5)." 2023. [53]
3. Radford, A., et al. "Learning Transferable Visual Models from Natural Language Supervision (CLIP)." ICML, 2021. [69]
4. Vasu, P.K.A., et al. "FastViT: A Fast Hybrid Vision Transformer Using Structural Reparameterization." ICCV, 2023. [82]
5. Vasu, P.K.A., et al. "MobileCLIP: Fast Image-Text Models Through Multi-Modal Reinforced Training." CVPR, 2024. [83]
6. Li, B., et al. "LLaVA-OneVision: Easy Visual Task Transfer." arXiv:2408.03326, 2024. [45]
7. McKinzie, B., et al. "MM1: Methods, Analysis & Insights from Multimodal LLM Pretraining." 2024. [66]
8. Ge, C., et al. "ConvLLaVA: Hierarchical Backbones as Visual Encoder for Large Multimodal Models." 2024. [28]
9. Tong, S., et al. "Cambrian-1: A Fully Open, Vision-Centric Exploration of Multimodal LLMs." 2024. [78]
10. Zhai, X., et al. "Sigmoid Loss for Language Image Pre-training (SigLIP)." ICCV, 2023. [94]
11. Chen, J., et al. "ViTamin: Designing Scalable Vision Models in the Vision-Language Era." CVPR, 2024. [12]
12. Cai, M., et al. "Matryoshka Multimodal Models." arXiv:2405.17430, 2024. [7]
13. Hu, W., et al. "Matryoshka Query Transformer for Large Vision-Language Models." 2024. [32]
14. Shang, Y., et al. "LLaVA-PruMerge: Adaptive Token Reduction for Efficient Large Multimodal Models." arXiv:2403.15388, 2024. [70]
15. Yang, S., et al. "VisionZip: Longer is Better but Not Necessary in Vision Language Models." 2024. [87]
16. Marafioti, A., et al. "SmolVLM: Redefining Small and Efficient Multimodal Models." arXiv:2504.05299, 2025. [62]
17. Chen, J., et al. "Florence-VL: Enhancing Vision-Language Models with Generative Vision Encoder and Depth-Breadth Fusion." arXiv:2412.04424, 2024. [11]
18. Yang, A., et al. "Qwen2 Technical Report." arXiv:2407.10671, 2024. [86]
19. Guo, J., et al. "MammothVL: Eliciting Multimodal Reasoning with Instruction Tuning at Scale." arXiv:2412.05237, 2024. [30]
20. GitHub Repository: https://github.com/apple/ml-fastvlm
