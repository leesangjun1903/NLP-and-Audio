# Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

이 논문은 현재 LLM의 희소성(Sparsity) 설계가 **Mixture-of-Experts(MoE)** 라는 단일 축에만 의존하고 있다는 문제를 지적한다. 언어 모델링은 본질적으로 두 가지 이질적인 하위 작업으로 구성된다:

1. **합성적 추론(Compositional Reasoning)**: 깊고 동적인 계산을 요구
2. **지식 검색(Knowledge Retrieval)**: 명명된 개체, 관용 표현 등 정적이고 반복적인 패턴

현재 Transformer는 두 번째 작업을 처리하기 위한 **기본 지식 룩업 프리미티브(native knowledge lookup primitive)가 없으므로**, 정적 패턴을 동적 계산을 통해 시뮬레이션해야 하는 비효율이 발생한다. 이를 해결하기 위해 **조건부 메모리(Conditional Memory)** 를 새로운 희소성 축으로 제안한다.

### 주요 기여

| 기여 | 내용 |
|------|------|
| **Engram 모듈 제안** | $N$-gram 임베딩을 현대화한 $O(1)$ 룩업 기반 조건부 메모리 모듈 |
| **희소성 할당 문제 정식화** | MoE와 Engram 간 파라미터 배분 최적화 문제 수립 |
| **U자형 스케일링 법칙 발견** | 두 모듈 간 최적 할당 비율의 존재 실증 |
| **대규모 사전학습 검증** | 27B 파라미터 규모에서 iso-parameter/iso-FLOPs MoE 대비 우월한 성능 |
| **시스템 효율성 설계** | 결정론적 주소 지정을 통한 호스트 메모리 오프로드 및 프리페칭 |

---

## 2. 문제 해결 방법, 모델 구조, 성능 향상 및 한계

### 2.1 해결하고자 하는 문제

기존 Transformer에서 "Diana, Princess of Wales"와 같은 다중 토큰 개체를 처리하기 위해 여러 레이어의 Attention과 FFN이 소모된다(Ghandeharioun et al., 2024). 이 과정은 실질적으로 **정적 룩업 테이블의 비용이 큰 런타임 재구성**에 해당하며, 고차원 추론에 활용되어야 할 네트워크 깊이를 낭비한다.

### 2.2 제안 방법 및 수식

#### 2.2.1 토크나이저 압축 (Tokenizer Compression)

원시 토큰 ID $x_t$를 정규화된 표준 ID로 변환하는 전사 함수:

$$\mathcal{P}: V \rightarrow V', \quad x'_t = \mathcal{P}(x_t)$$

이를 통해 "Apple"과 "apple"과 같이 의미적으로 동치인 토큰을 동일한 ID로 매핑한다. 128k 토크나이저 기준 **23% 어휘 크기 감소**를 달성한다.

suffix $N$-gram은 다음과 같이 구성된다:

$$g_{t,n} = (x'_{t-n+1}, \ldots, x'_t)$$

#### 2.2.2 멀티헤드 해싱 (Multi-Head Hashing)

$N$-gram 조합 공간은 직접 파라미터화가 불가능하므로, $K$개의 독립적인 해시 헤드를 사용한다. 각 헤드 $k$는 $n$-gram $g_{t,n}$을 소수 크기 $M_{n,k}$의 임베딩 테이블 $\mathbf{E}_{n,k}$의 인덱스로 매핑한다:

$$z_{t,n,k} \triangleq \varphi_{n,k}(g_{t,n}), \quad \mathbf{e}_{t,n,k} = \mathbf{E}_{n,k}[z_{t,n,k}] \tag{1}$$

모든 검색된 임베딩을 연결하여 최종 메모리 벡터 $\mathbf{e}\_t \in \mathbb{R}^{d_\text{mem}}$를 구성한다:

$$\mathbf{e}_t \triangleq \biggl\|_{n=2}^{N} \biggl\|_{k=1}^{K} \mathbf{e}_{t,n,k} \tag{2}$$

#### 2.2.3 컨텍스트 인식 게이팅 (Context-aware Gating)

검색된 임베딩 $\mathbf{e}_t$는 정적이므로 문맥 적응성이 부족하고 해시 충돌이나 다의어 문제가 발생할 수 있다. 이를 해결하기 위해 현재 은닉 상태 $\mathbf{h}_t$를 동적 Query로, 검색된 메모리 $\mathbf{e}_t$를 Key/Value의 소스로 사용하는 게이팅 메커니즘을 적용한다:

$$\mathbf{k}_t = \mathbf{W}_K \mathbf{e}_t, \quad \mathbf{v}_t = \mathbf{W}_V \mathbf{e}_t \tag{3}$$

스칼라 게이트 $\alpha_t \in (0, 1)$는 다음과 같이 계산된다 (기울기 안정성을 위해 RMSNorm 적용):

$$\alpha_t = \sigma\left(\frac{\text{RMSNorm}(\mathbf{h}_t)^\top \text{RMSNorm}(\mathbf{k}_t)}{\sqrt{d}}\right) \tag{4}$$

게이팅된 출력 $\tilde{\mathbf{v}}_t = \alpha_t \cdot \mathbf{v}_t$에 경량 인과적 합성곱을 적용하여 최종 출력 $\mathbf{Y}$를 계산한다:

$$\mathbf{Y} = \text{SiLU}\left(\text{Conv1D}(\text{RMSNorm}(\tilde{\mathbf{V}}))\right) + \tilde{\mathbf{V}} \tag{5}$$

Engram 모듈은 잔차 연결을 통해 백본에 통합된다:

$$\mathbf{H}^{(\ell)} \leftarrow \mathbf{H}^{(\ell)} + \mathbf{Y}$$

#### 2.2.4 멀티브랜치 아키텍처 통합

$M$개의 병렬 브랜치를 가진 다중 브랜치 아키텍처(Manifold-Constrained Hyper-Connections, mHC)에서 $m$번째 브랜치의 게이팅 신호는 다음과 같이 계산된다:

$$\alpha_t^{(m)} = \sigma\left(\frac{\text{RMSNorm}(\mathbf{h}_t^{(m)})^\top \text{RMSNorm}(\mathbf{W}_K^{(m)} \mathbf{e}_t)}{\sqrt{d}}\right) \tag{6}$$

단일 $\mathbf{W}_V$와 $M$개의 $\mathbf{W}_K^{(m)}$을 FP8 행렬 곱셈으로 융합하여 GPU 활용도를 극대화한다.

#### 2.2.5 희소성 할당 문제 (Sparsity Allocation)

세 가지 파라미터 지표를 정의한다:

- $P_\text{tot}$: 전체 학습 가능 파라미터 수
- $P_\text{act}$: 토큰당 활성화 파라미터 수 (FLOPs 결정)
- $P_\text{sparse} \triangleq P_\text{tot} - P_\text{act}$: 비활성 파라미터 예산

할당 비율 $\rho \in [0, 1]$을 MoE 전문가 용량에 할당되는 비활성 파라미터의 비율로 정의한다:

$$P_\text{MoE}^{(\text{sparse})} = \rho \cdot P_\text{sparse}, \quad P_\text{Engram} = (1 - \rho) \cdot P_\text{sparse} \tag{7}$$

- $\rho = 1$: 순수 MoE 모델
- $\rho < 1$: 일부 파라미터를 Engram에 재할당

#### 2.2.6 CKA 기반 표현 정렬 분석

두 모델 레이어의 표현을 비교하기 위해 Centered Kernel Alignment(CKA)를 사용한다:

$$\text{CKA}(K, L) = \frac{\text{HSIC}(K, L)}{\sqrt{\text{HSIC}(K, K) \cdot \text{HSIC}(L, L)}} \tag{8}$$

여기서 $K = XX^\top$, $L = YY^\top$이며 HSIC는 Hilbert-Schmidt 독립성 기준이다.

Engram 레이어 $j$에 대응하는 MoE의 "유효 깊이"를 정량화하는 소프트 정렬 인덱스:

$$a_j = \frac{\sum_{i \in \mathcal{I}_j} S_{i,j} \cdot i}{\sum_{i \in \mathcal{I}_j} S_{i,j}}, \quad \text{where } \mathcal{I}_j = \text{argtop}_k^i(S_{i,j}) \tag{9}$$

### 2.3 모델 구조

**실험에 사용된 4가지 모델:**

| 모델 | 전체 파라미터 | 활성 파라미터 | 구성 |
|------|--------------|--------------|------|
| Dense-4B | 4.1B | 3.8B | 표준 밀집 FFN |
| MoE-27B | 26.7B | 3.8B | 72개 라우팅 전문가 + 2개 공유 전문가 (top-6) |
| Engram-27B | 26.7B | 3.8B | 55개 라우팅 전문가 + 5.7B Engram 메모리 ($\rho = 74.3\%$) |
| Engram-40B | 39.5B | 3.8B | 55개 라우팅 전문가 + 18.5B Engram 메모리 |

**공통 아키텍처 세부사항:**
- 30개 블록, 은닉 크기 2560
- Multi-head Latent Attention (MLA, 32헤드)
- mHC ($M = 4$, 확장 비율 4)
- Engram: 레이어 2, 15에 삽입, $N$-gram $\{2, 3\}$, 8개 헤드, $d_\text{mem} = 1280$
- 옵티마이저: 백본은 Muon, Engram 임베딩은 Adam (학습률 5× 증폭, weight decay 없음)
- 컨볼루션 파라미터 제로 초기화 (훈련 초기 항등 매핑 보장)

### 2.4 성능 향상

**핵심 벤치마크 결과 (Engram-27B vs MoE-27B, 262B 토큰 훈련):**

| 카테고리 | 벤치마크 | MoE-27B | Engram-27B | 향상 |
|---------|---------|---------|-----------|------|
| 지식 | MMLU | 57.4 | 60.4 | **+3.0** |
| 지식 | CMMLU | 57.9 | 61.9 | **+4.0** |
| 지식 | MMLU-Pro | 28.3 | 30.1 | **+1.8** |
| 추론 | BBH | 50.9 | 55.9 | **+5.0** |
| 추론 | ARC-Challenge | 70.1 | 73.8 | **+3.7** |
| 추론 | DROP | 55.7 | 59.0 | **+3.3** |
| 코드 | HumanEval | 37.8 | 40.8 | **+3.0** |
| 수학 | MATH | 28.3 | 30.7 | **+2.4** |
| 수학 | GSM8K | 58.4 | 60.6 | **+2.2** |

**장문맥 성능 (Iso-Loss 설정, 46k vs 50k):**

| 벤치마크 | MoE-27B | Engram-27B | 향상 |
|---------|---------|-----------|------|
| Multi-Query NIAH | 84.2 | 97.0 | **+12.8** |
| Variable Tracking | 77.0 | 87.2 | **+10.2** |

**시스템 효율성:**
- 100B 파라미터 Engram 테이블을 호스트 메모리에 오프로드 시 처리량 손실 < 3% (최대 2.8%)

### 2.5 한계

1. **고차원 N-gram의 한계**: 4-gram 이상은 고정 메모리 예산 하에서 2/3-gram 패턴의 용량을 희석시켜 성능이 저하될 수 있다. 단, 더 큰 메모리 스케일에서는 유익해질 가능성이 있다.

2. **훈련-추론 불일치**: 추론 시 Engram 출력을 완전히 억제하면 훈련-추론 불일치가 발생하여 복잡한 혼합 능력 과제에서 노이즈가 생긴다.

3. **정적 패턴의 제약**: N-gram 기반 정적 룩업은 고정된 로컬 패턴만 캡처할 수 있으며, 동적이고 전역적인 의존성에는 제한적이다.

4. **Engram-40B의 미포화**: 262B 토큰 훈련 예산 하에서 Engram-40B의 확장된 메모리 용량이 아직 완전히 포화되지 않아, 일부 태스크에서 Engram-27B 대비 우위를 보이지 못한다.

5. **대규모 임베딩 테이블의 훈련 비용**: 훈련 중 임베딩 테이블을 여러 GPU에 분산 저장하고 All-to-All 통신이 필요하여 통신 오버헤드가 발생한다.

6. **해시 충돌 문제**: 다중 해시 헤드로 완화되지만 완전히 제거되지 않으며, 폴리세미(다의성)로 인한 노이즈도 잔존한다.

---

## 3. 일반화 성능 향상 가능성

### 3.1 예상 외 일반화 향상의 발견

가장 주목할 만한 발견은 Engram이 **지식 집약적 태스크를 넘어** 일반 추론, 코드, 수학 등 다양한 도메인에서 더 큰 향상을 보인다는 것이다:

$$\underbrace{\text{BBH} (+5.0)}_{\text{일반 추론}} > \underbrace{\text{MMLU} (+3.0)}_{\text{지식}} \approx \underbrace{\text{HumanEval} (+3.0)}_{\text{코드}}$$

이는 단순한 지식 저장 효과가 아닌, **네트워크 구조적 개선**에 의한 일반화 향상임을 시사한다.

### 3.2 메커니즘: 유효 깊이 증가 (Effective Depth Increase)

**LogitLens 분석**: Engram 모델은 MoE 베이스라인 대비 초기 레이어에서 체계적으로 낮은 KL 발산을 보인다. 이는 모델이 훨씬 빠르게 예측 가능한 표현에 도달함을 의미한다.

**CKA 분석**: Engram-27B의 레이어 5에서 형성된 표현이 MoE 베이스라인의 레이어 12 표현과 가장 유사하다($a_j > j$). 즉:

> **Engram은 초기 레이어의 정적 패턴 재구성 부담을 해소함으로써, 네트워크의 유효 깊이를 증가시키는 효과를 낸다.**

이 유효 깊이 증가가 일반 추론 능력 향상의 직접적 원인이다.

### 3.3 주의 용량 해방 (Attention Capacity Liberation)

로컬 의존성을 Engram 룩업에 위임함으로써, **어텐션 메커니즘이 전역 컨텍스트에 집중**할 수 있게 된다. 이는 장문맥 성능의 대폭 향상으로 직결된다:

- Multi-Query NIAH: $84.2 \rightarrow 97.0$ (Iso-FLOPs 설정)
- Variable Tracking: $77.0 \rightarrow 89.0$

### 3.4 기능적 이분법 (Functional Dichotomy)

민감도 분석(Engram 출력 완전 억제 실험)은 명확한 기능적 이분법을 보여준다:

| 태스크 유형 | 잔류 성능 | 해석 |
|------------|----------|------|
| 사실적 지식 (TriviaQA) | **29%** | Engram이 주요 지식 저장소 역할 |
| 읽기 이해 (C3) | **93%** | 백본 어텐션이 주요 역할 |

이는 Engram이 백본과 **구조적으로 보완적인 역할 분담**을 학습함을 보여주며, 각 컴포넌트가 전문화된 기능을 담당함으로써 전체적인 일반화 성능이 향상된다.

### 3.5 Zipfian 분포 활용

자연어 $N$-gram은 Zipfian 분포를 따르므로, 자주 등장하는 패턴(전체 메모리 접근의 대부분)을 빠른 저장 계층(GPU HBM, 호스트 DRAM)에 캐시할 수 있다. 이는 메모리 계층 구조를 최대한 활용하여 대규모 확장을 가능하게 한다.

---

## 4. 앞으로의 연구에 미치는 영향 및 고려사항

### 4.1 연구에 미치는 영향

#### 4.1.1 새로운 희소성 분류 체계

이 논문은 LLM의 희소성 설계에 대한 새로운 분류 체계를 제시한다:

| 희소성 유형 | 메커니즘 | 대표 방법 | 목적 |
|------------|---------|----------|------|
| 조건부 계산 (Conditional Computation) | 동적 라우팅 | MoE | 동적 논리 처리 |
| **조건부 메모리 (Conditional Memory)** | **정적 룩업** | **Engram** | **정적 지식 검색** |

이 이분법은 향후 아키텍처 설계의 새로운 기준점이 될 수 있다.

#### 4.1.2 스케일링 법칙의 새로운 차원

U자형 스케일링 법칙과 무한 메모리 체제에서의 로그-선형 스케일링은 기존 Chinchilla 스케일링 법칙을 보완하는 새로운 차원을 제시한다. 향후 연구에서 최적 컴퓨팅-메모리 할당에 대한 통합 스케일링 법칙 도출이 중요한 과제가 될 것이다.

#### 4.1.3 인프라 인식 설계 원칙

Engram의 결정론적 주소 지정 기반 프리페칭 전략은 **알고리즘-시스템 공동 설계(Algorithm-System Co-design)** 의 중요성을 강조한다. 이는 향후 모델 설계에서 하드웨어 메모리 계층 구조를 1등급 설계 원칙으로 고려해야 함을 시사한다.

#### 4.1.4 해석 가능성 연구에의 기여

Engram의 게이팅 메커니즘 시각화는 모델이 명명된 개체와 관용구를 어떻게 처리하는지 직접적으로 보여준다. 이는 메커니즘 해석 가능성(Mechanistic Interpretability) 연구에 새로운 도구와 관점을 제공한다.

### 4.2 향후 연구 시 고려할 점

#### 4.2.1 최적 할당 비율의 스케일 의존성 검증

현재 실험에서 최적 $\rho \approx 75\%-80\%$가 5.7B~9.9B 파라미터 규모에서 안정적임을 보였으나, **100B 이상의 대규모 모델**에서도 동일한 최적점이 유지되는지 검증이 필요하다. 또한 다양한 희소성 비율 $P_\text{tot}/P_\text{act}$에서의 최적 $\rho$ 변화를 체계적으로 분석해야 한다.

#### 4.2.2 동적 메모리와의 통합

현재 Engram은 **정적** 임베딩 테이블을 사용하지만, 외부 지식 베이스나 동적으로 업데이트 가능한 메모리(RETRO, REALM 방식)와의 통합 가능성을 탐색해야 한다. 이를 통해 지식 업데이트 없이 재훈련을 피할 수 있다.

#### 4.2.3 다국어 및 도메인 특화 환경에서의 검증

현재 실험은 주로 영어와 중국어 중심의 일반 도메인 사전학습에 집중되어 있다. 코드 전용 모델, 생물의학 특화 모델, 저자원 언어 환경 등 특수 도메인에서 Engram의 효과를 검증하고, $N$-gram 패턴의 분포 특성이 다른 언어에서 어떻게 달라지는지 연구가 필요하다.

#### 4.2.4 사후 학습(Post-training) 단계에서의 효과

RLHF, SFT, DPO 등 사후 학습 단계에서 Engram의 역할 변화를 연구해야 한다. 특히 지식 집약적 태스크에서 Engram이 학습된 사실 지식을 정렬(alignment) 과정에서 어떻게 유지하거나 변형하는지가 중요한 연구 주제이다.

#### 4.2.5 N-gram 범위 및 해시 충돌 최적화

현재 $N = \{2, 3\}$으로 설정된 N-gram 범위가 최적인지, 그리고 고정 예산 하에서 4-gram 이상의 고차 N-gram이 더 큰 메모리 스케일에서 어떻게 기여하는지 체계적으로 연구해야 한다. 또한 해시 충돌을 더 효과적으로 줄이는 새로운 해싱 전략(예: Learned Hashing)의 탐색도 중요하다.

#### 4.2.6 모델 편집 및 지식 업데이트

Engram의 임베딩 테이블이 사실 지식의 주요 저장소임이 밝혀진 만큼, ROME/MEMIT과 같은 **모델 편집** 기술을 Engram 테이블에 직접 적용하는 방식을 탐색할 수 있다. 이는 전체 모델 재훈련 없이 특정 사실을 업데이트하는 효율적인 방법이 될 수 있다.

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

| 연구 | 연도 | 방법 | MoE 호환 | Iso-param/FLOPs 비교 | 시스템 설계 | 주요 한계 |
|------|------|------|---------|---------------------|-----------|----------|
| **Switch Transformer** (Fedus et al.) | 2022 | MoE (희소 라우팅) | - | ✓ | 기본 | 동적 라우팅 오버헤드 |
| **RETRO** (Borgeaud et al.) | 2022 | 비모수 검색 증강 | ✗ | △ | 복잡 | 추론 시 검색 비용 |
| **PKM/PEER** (Lample et al.; He) | 2019/2024 | 파라미터 키-값 메모리 | △ | △ | 미고려 | 동적 키 계산 필요 |
| **DeepSeekMoE** (Dai et al.) | 2024 | 세분화 전문가 분리 | - | ✓ | ✓ | 조건부 메모리 없음 |
| **OverEncoding** (Huang et al.) | 2025 | N-gram 임베딩 평균 | △ | ✗ | ✗ (레이어 0만) | MoE에 효과 미미 |
| **SCONE** (Yu et al.) | 2025 | f-gram 보조 모델 | △ | ✗ | ✗ | 추가 FLOPs 필요 |
| **BLT** (Pagnoni et al.) | 2025 | 바이트 수준 해시 N-gram | △ | △ | △ | 토큰 수준 미적용 |
| **UltraMem** (Huang et al.) | 2025 | 초희소 키-값 메모리 | △ | △ | 미고려 | 동적 키 계산 |
| **Memory+** (Berges et al.) | 2025 | 대규모 메모리 레이어 | △ | △ | 미고려 | 동적 메모리 접근 |
| **SuperBPE** (Liu et al.) | 2025 | 다중어 표현 슈퍼토큰 | ✗ | ✗ | ✗ | 토크나이저 변경 필요 |
| **Engram (본 논문)** | 2026 | 조건부 메모리 + 결정론적 룩업 | ✓ | ✓ | ✓ (프리페치) | 정적 패턴만 처리 |

**핵심 차별점 요약:**

1. **공정 비교 프로토콜**: Engram만이 엄격한 iso-parameter + iso-FLOPs 조건 하에서 MoE 대비 우위를 입증한다.

2. **시스템 공동 설계**: 기존 연구들이 레이어 0에 임베딩을 배치하여 통신과 계산이 직렬화되는 것과 달리, Engram은 중간 레이어 삽입으로 통신-계산 오버랩을 실현한다.

3. **스케일링 검증**: 27B~40B 파라미터 규모에서 실제 사전학습을 통해 효과를 검증한 반면, 대부분의 경쟁 연구는 소규모 실험에 그친다.

4. **조건부 메모리 프레임워크**: Engram은 N-gram 임베딩을 단순한 외부 보조 도구가 아닌 희소성의 독립적인 축으로 정식화한 최초의 연구이다.

---

## 참고자료

**본 논문:**
- Cheng, X., et al. "Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models." arXiv:2601.07372v1 [cs.CL], January 12, 2026. GitHub: https://github.com/deepseek-ai/Engram

**논문 내 주요 인용 문헌:**
- Dai, D., et al. "DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models." arXiv:2401.06066, 2024.
- Fedus, W., et al. "Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity." JMLR, 2022.
- Borgeaud, S., et al. "Improving Language Models by Retrieving from Trillions of Tokens." ICML, 2022.
- He, X. O. "Mixture of a Million Experts." arXiv:2407.04153, 2024.
- Huang, H., et al. "Over-tokenized Transformer: Vocabulary is Generally Worth Scaling." ICML 2025.
- Yu, D., et al. "Scaling Embedding Layers in Language Models (SCONE)." arXiv:2502.01637, 2025.
- Pagnoni, A., et al. "Byte Latent Transformer: Patches Scale Better than Tokens." ACL 2025.
- Berges, V., et al. "Memory Layers at Scale." ICML 2025.
- Liu, A., et al. "DeepSeek-V3 Technical Report." arXiv:2412.19437, 2024.
- Ghandeharioun, A., et al. "Patchscopes: A Unifying Framework for Inspecting Hidden Representations of Language Models." ICML, 2024.
- Kornblith, S., et al. "Similarity of Neural Network Representations Revisited." ICML, 2019.
- Xie, Z., et al. "mHC: Manifold-Constrained Hyper-Connections." arXiv:2512.24880, 2025.
- Hendrycks, D., et al. "Measuring Massive Multitask Language Understanding (MMLU)." ICLR 2021.
- nostalgebraist. "Interpreting GPT: the Logit Lens." LessWrong, 2020.
