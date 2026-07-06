# Byte Latent Transformer: Patches Scale Better Than Tokens

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

**BLT(Byte Latent Transformer)**는 고정 어휘 기반 토크나이저 없이 원시 바이트(raw bytes)를 직접 학습하는 새로운 LLM 아키텍처로, 처음으로 대규모에서 토크나이저 기반 모델(Llama 3)과 동등한 성능을 달성하면서 추론 효율성과 강건성을 크게 향상시켰다는 것이 핵심 주장입니다.

### 주요 기여

| 기여 항목 | 내용 |
|-----------|------|
| **새 아키텍처** | 엔트로피 기반 동적 패칭을 통한 계산 자원의 효율적 배분 |
| **스케일링 동등성** | 최대 8B 파라미터, 4T 바이트에서 Llama 3와 FLOP 대비 동등 성능 |
| **새로운 스케일링 축** | 동일 추론 비용에서 모델 크기 + 패치 크기를 동시에 증가 가능 |
| **강건성 향상** | 노이즈 입력, 저자원 언어, 철자·음운론 태스크에서 월등한 성능 |
| **추론 효율** | 최대 50% 추론 FLOP 절감 |

---

## 2. 해결하고자 하는 문제, 제안 방법, 모델 구조, 성능, 한계

### 2.1 해결하고자 하는 문제

기존 LLM의 **토크나이제이션(tokenization)**은 다음과 같은 구조적 문제를 가집니다:

1. **도메인/모달리티 민감성**: BPE는 특정 도메인에 편향된 압축 방식을 강제합니다.
2. **노이즈에 취약**: 토큰 경계가 고정되어 있어 노이즈 입력에 민감합니다.
3. **철자/음운 지식 부재**: 토큰 내부 바이트 정보에 직접 접근 불가합니다.
4. **다국어 불평등**: 저자원 언어에서 토큰당 더 많은 바이트를 소비합니다.
5. **계산 비효율**: 모든 토큰에 동일한 계산량을 배분하는 비효율이 존재합니다.
6. **스케일링 제약**: 어휘 크기와 추론 비용 사이의 트레이드오프가 고정됩니다.

직접 바이트 레벨 모델링은 긴 시퀀스로 인한 계산 비용 문제가 있었으며, 이를 해결하는 것이 BLT의 핵심 과제였습니다.

---

### 2.2 제안하는 방법 (수식 포함)

#### 2.2.1 패칭 함수 (Patching Function)

바이트 시퀀스 $\boldsymbol{x} = \{x_i \mid i = 1, \ldots, n\}$을 패치 시퀀스 $\boldsymbol{p} = \{p_j \mid j = 1, \ldots, m\}$ ($m < n$)으로 분할합니다. 각 $x_i$를 $\{0, 1\}$로 매핑하며, $1$은 새 패치의 시작을 나타냅니다.

#### 2.2.2 엔트로피 패칭 (핵심 방법)

소형 바이트 레벨 언어모델 $p_e$를 사용해 다음 바이트의 엔트로피를 계산합니다:

$$H(x_i) = \sum_{v \in \mathcal{V}} p_e(x_i = v \mid \boldsymbol{x}_{<i}) \log p_e(x_i = v \mid \boldsymbol{x}_{<i}) $$

패치 경계 결정 방법은 두 가지입니다:

$$\text{Global Constraint:} \quad H(x_t) > \theta_g$$

$$\text{Approx. Monotonic Constraint:} \quad H(x_t) - H(x_{t-1}) > \theta_r$$

**직관**: 엔트로피가 높은 지점(= 예측하기 어려운 바이트)에서 새 패치를 시작하여, 어려운 예측에 더 많은 계산 자원을 배분합니다.

#### 2.2.3 점진적 패칭 조건 (Incremental Patching)

생성 시 미래 바이트에 접근할 수 없으므로:

$$f_p(\boldsymbol{x}_{<i}) = f_p(\boldsymbol{x})_{<i}$$

이 조건을 만족해야 하며, BPE는 동일 접두사가 후속 시퀀스에 따라 다르게 토크나이즈될 수 있어 이 조건을 위반합니다.

#### 2.2.4 해시 n-gram 임베딩

각 바이트 위치 $i$에서 바이트 n-gram을 구성합니다:

$$g_{i,n} = \{b_{i-n+1}, \ldots, b_i\} $$

증강된 임베딩:

$$e_i = x_i + \sum_{n=3,\ldots,8} E_n^{\text{hash}}(\text{Hash}(g_{i,n})) $$

$$\text{Hash}(g_{i,n}) = \text{RollPolyHash}(g_{i,n}) \% |E_n^{\text{hash}}| $$

롤링 다항식 해시:

$$\text{Hash}(g_{i,n}) = \sum_{j=1}^{n} b_{i-j+1} a^{j-1} $$

($a$는 10자리 소수)

#### 2.2.5 인코더 크로스-어텐션

패치 $p_j$에 해당하는 바이트 시퀀스 $f_{\text{bytes}}(p_j)$에 대해:

$$P_{0,j} = \mathcal{E}_C(f_{\text{bytes}}(p_j)), \quad f \text{ is a pooling function} $$

$$P_l = P_{l-1} + W_o\left(\text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V\right) $$

$$\text{where } Q_j = W_q(P_{l-1,j}),\; K_i = W_k(h_{l-1,i}),\; V_i = W_v(h_{l-1,i}) $$

$$h_l = \text{Encoder-Transformer-Layer}_l(h_{l-1}) $$

#### 2.2.6 디코더 크로스-어텐션

$$D_0 = h_{l_\mathcal{E}} $$

$$B_l = D_{l-1} + W_o\left(\text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V\right) $$

$$\text{where } Q_i = W_q(d_{l-1,i}),\; K_i = W_k(\mathcal{D}_C(o_j)),\; V_i = W_v(\mathcal{D}_C(o_j)) $$

$$D_l = \text{Decoder-Transformer-layer}_l(B_l) $$

#### 2.2.7 BLT FLOP 계산

$$\text{FL}_{\text{BLT}} = \text{Transf. FL}(h_\mathcal{G}, l_\mathcal{G}, m = n_{ctx}/n_p, V=0)/n_p $$
$$+ \text{Transf. FL}(h_\mathcal{E}, l_\mathcal{E}, m = w_\mathcal{E}, V=0) $$
$$+ \text{Transf. FL}(h_\mathcal{D}, l_\mathcal{D}, m = w_\mathcal{D}, V=256) $$
$$+ \text{Cross Attn. FL}(h_\mathcal{E}, l_\mathcal{E}, m=n_p, r=n_p/k) \times k/n_p $$
$$+ \text{Cross Attn. FL}(h_\mathcal{D}, l_\mathcal{D}, m=k, r=k/n_p) $$

#### 2.2.8 Bits-Per-Byte (BPB) 평가 지표

$$\text{BPB}(x) = \frac{\mathcal{L}_{CE}(x)}{\ln(2) \cdot n_{\text{bytes}}} $$

---

### 2.3 모델 구조

BLT는 세 모듈로 구성됩니다:

```
바이트 입력
    ↓
[Local Encoder] (경량, lE << lG 레이어)
 - 바이트 임베딩 (256 × hE 행렬)
 - 해시 n-gram 임베딩 (n=3~8)
 - 트랜스포머 레이어 (로컬 인과 어텐션, 윈도우 wE)
 - 크로스-어텐션으로 패치 표현 생성
    ↓
[Latent Global Transformer] (대형, lG 레이어)
 - 패치 표현에 대한 자기회귀 트랜스포머
 - 블록-인과 어텐션 마스크
 - 계산의 대부분을 차지
    ↓
[Local Decoder] (경량, lD << lG 레이어)
 - 패치 표현을 바이트로 복원
 - 크로스-어텐션 (패치가 키/값, 바이트가 쿼리)
    ↓
바이트 출력
```

**모델 규모별 하이퍼파라미터 (Table 10)**:

| 모델 | $l_\mathcal{E}$ | $h_\mathcal{E}$ | $l_\mathcal{G}$ | $h_\mathcal{G}$ | 전역 파라미터 | $l_\mathcal{D}$ |
|------|------|------|------|------|------|------|
| 400M | 1 | 768 | 24 | 1280 | 470M | 7 |
| 1B | 1 | 1024 | 25 | 2048 | 1B | 9 |
| 8B | 1 | 1280 | 32 | 4096 | 6.4B | 6 |

---

### 2.4 성능 향상

#### 스케일링 성능 (Table 1, 8B 모델)

| 태스크 | Llama 3 (1T 토큰) | BLT-Space (6T 바이트) | BLT-Entropy (4.5T 바이트) |
|--------|------|------|------|
| Arc-E | 77.6 | 75.4 | **79.6** |
| Arc-C | **53.3** | 49.8 | 52.1 |
| HellaSwag | 79.1 | 79.6 | **80.6** |
| PIQA | 80.7 | **81.1** | 80.6 |
| MMLU | **58.1** | 54.8 | 57.4 |
| MBPP | 40.2 | 37.6 | **41.8** |
| HumanEval | 31.1 | 27.4 | **35.4** |
| **평균** | 60.0 | 58.0 | **61.1** |

#### 강건성 성능 (Table 3, 노이즈 데이터)

| 태스크 | Llama 3 (1T) | Llama 3.1 (16T) | BLT (1T) |
|--------|------|------|------|
| HellaSwag 노이즈 평균 | 56.9 | 64.3 | **64.3** |
| CUTE 전체 | 27.5 | 20.0 | **54.1** |
| Spelling | 1.1 | - | **99.9** |
| Contains Char | 0.0 | 0.0 | **55.9** |

#### 추론 FLOP 효율

- 패치 크기 8 사용 시 Llama 3 대비 **~50% 추론 FLOP 절감**
- 동일 추론 비용에서 BLT 모델이 **1.6~1.7배 큰 글로벌 트랜스포머** 사용 가능

---

### 2.5 한계

1. **스케일링 법칙 미최적화**: BPE용으로 계산된 Llama 3의 스케일링 법칙을 그대로 사용 → BLT 고유의 최적 (데이터, 파라미터) 비율 미탐색
2. **엔트로피 모델 독립 학습**: 패칭 모델이 별도로 학습되며 엔드-투-엔드 학습이 아님
3. **벽시계 시간 효율**: 이론적 FLOP 동등성은 달성했으나 실제 벽시계 시간은 토크나이저 기반 모델과 동일하지 않을 수 있음
4. **엔트로피 드리프트**: 반복적 콘텐츠(MMLU 등)에서 엔트로피 드리프트가 발생하여 패칭이 비균일해짐
5. **1B 이하 실험 한정**: 많은 실험이 1B 파라미터까지만 수행되었으며, 더 큰 스케일에서 최적 아키텍처 선택이 달라질 수 있음

---

## 3. 모델의 일반화 성능 향상 가능성

BLT가 일반화 성능을 향상시키는 메커니즘은 다음과 같이 분류됩니다:

### 3.1 노이즈 입력에 대한 강건성 (Robustness to Noisy Inputs)

BLT는 토크나이저 기반 모델과 달리 원시 바이트를 직접 처리하기 때문에, 토큰 경계 재구성에 취약하지 않습니다. Table 3에서 다섯 가지 노이즈 전략(AntSpeak, Drop, RandomCase, Repeat, UpperCase)에 대해 Llama 3(1T) 대비 **평균 8점** 높은 성능을 보였고, 16배 많은 데이터로 학습된 Llama 3.1(16T)과도 동등한 수준을 달성했습니다. 이는 **데이터 확장으로는 얻기 어려운 구조적 강건성**임을 시사합니다.

### 3.2 저자원 언어 및 다국어 일반화

Table 4에서 FLORES-101 벤치마크 결과, 27개 언어의 번역에서:

- **→ 영어 번역**: Llama 3(12.1) 대비 BLT(14.0), **+1.9점**
- **영어 → 번역**: Llama 3(5.9) 대비 BLT(6.4), **+0.5점**
- 특히 아르메니아어, 벵골어, 조지아어, 크메르어 등 저자원 언어에서 대폭 개선

이는 BLT가 **토큰 어휘에 종속되지 않는** 구조 덕분에 새로운 언어의 바이트 시퀀스를 더 자연스럽게 처리한 결과입니다.

### 3.3 서브워드 수준 지식 (Orthographic & Phonological Generalization)

CUTE 벤치마크에서 BLT는 Llama 3(1T) 대비 **+25점 이상** 향상을 보였으며:

- **Spelling 태스크**: 1.1% → **99.9%**
- **Substitute Char**: 0.4% → **48.7%**
- **Semantic Similarity**: 65% → **90.5%**

이처럼 BLT는 **철자 조작, 음운 인식, 의미적 유사성** 등 fine-grained 언어 지식에서 탁월한 일반화를 보입니다. 이는 토크나이저가 바이트 수준 정보에 대한 접근을 차단하는 반면, BLT는 이를 직접 모델링하기 때문입니다.

### 3.4 동적 계산 배분에 의한 일반화

$$\text{어려운 바이트 (높은 } H(x_i)\text{)} \Rightarrow \text{짧은 패치} \Rightarrow \text{더 많은 계산 배분}$$

$$\text{쉬운 바이트 (낮은 } H(x_i)\text{)} \Rightarrow \text{긴 패치} \Rightarrow \text{적은 계산 배분}$$

이 메커니즘은 **데이터 복잡도에 적응적**으로 계산을 배분하여, 어려운 부분(예: 새 문장의 첫 단어, 수식)에 더 많은 모델 용량을 할당합니다. 이는 분포 외 데이터(out-of-distribution)에서도 더 나은 일반화를 가능하게 합니다.

### 3.5 Llama 3.1 초기화를 통한 전이 학습 가능성

Table 5에서 Llama 3.1 가중치로 전역 트랜스포머를 초기화한 BLT는 MMLU에서 63.7%를 달성하여, 1T 토큰으로 학습한 BLT-Entropy(57.4%)를 크게 상회합니다. 이는 **기존 사전학습 지식을 보존하면서 바이트 레벨 강건성을 획득**하는 유망한 방향을 제시합니다.

### 3.6 롱테일 데이터 일반화

고정 어휘 토크나이저는 희귀 바이트 시퀀스(예: 특수 기호, 새로운 언어, 코드 패턴)를 비효율적으로 처리하지만, BLT는 **어휘 크기 제한 없이** 모든 바이트 시퀀스를 동일한 방식으로 처리합니다. 이는 롱테일 데이터에 대한 본질적인 일반화 이점을 제공합니다.

---

## 4. 최신 연구 비교 분석 (2020년 이후)

### 4.1 주요 관련 연구와의 비교표

| 연구 | 연도 | 방법 | 주요 특징 | BLT와의 차이 |
|------|------|------|-----------|--------------|
| **ByT5** (Xue et al.) | 2022 | 바이트-투-바이트 | 패칭 없음, 노이즈 강건성 | BLT는 패칭으로 효율성 해결 |
| **MegaByte** (Yu et al.) | 2023 | 고정 스트라이드 패칭 | 1B 스케일 가능 | BLT: 동적 패칭 > 정적 패칭 |
| **SpaceByte** (Slagle) | 2024 | 공백 기반 패칭 | 로컬 인코더 추가 | BLT: 더 발전된 아키텍처 필요 |
| **MambaByte** (Wang et al.) | 2024 | Mamba 아키텍처 | 패칭 없음, 350M 스케일 | BLT: 8B 스케일, 더 높은 성능 |
| **Hourglass Transformer** (Nawrot et al.) | 2022 | 정적 다운/업샘플링 | 150M 스케일 | BLT: 동적 패칭, 더 큰 스케일 |
| **Efficient Transformers with Dynamic Token Pooling** (Nawrot et al.) | 2023 | 동적 토큰 풀링 | 40M 스케일 | BLT: 훨씬 큰 스케일 달성 |
| **CANINE** (Clark et al.) | 2022 | 문자 레벨 인코더 | 150M, 다국어 | BLT: 더 큰 스케일, 생성 가능 |
| **CharacterBERT** (El Boukkouri et al.) | 2020 | 문자 컨볼루션 | 인코더 전용 | BLT: 완전 생성 모델 |
| **Llama 3** (Dubey et al.) | 2024 | BPE 토크나이저 | SOTA 토큰 기반 | BLT의 직접 비교 대상 |

### 4.2 세부 비교 분석

#### ByT5 vs BLT
- **공통점**: 바이트 레벨 직접 처리, 노이즈 강건성
- **차이점**: ByT5는 패칭이 없어 모든 바이트에 동일 계산 → **대규모에서 계산 비효율**. BLT는 동적 패칭으로 이 문제를 해결하여 처음으로 BPE 모델과 동등한 성능을 대규모에서 달성

#### MegaByte vs BLT
- **공통점**: 계층적 구조, 전역/로컬 모델 분리
- **차이점**: MegaByte의 고정 스트라이드 패칭은 계산 복잡도와 데이터 복잡도 미상관. BLT의 엔트로피 기반 동적 패칭은 이 문제를 해결하며, n-gram 임베딩과 크로스-어텐션 추가로 BPE 성능에 도달

#### MambaByte vs BLT
- **공통점**: 토크나이저 없음
- **차이점**: MambaByte는 Mamba 선형 어텐션으로 바이트 시퀀스 처리. 350M 파라미터에서는 경쟁력 있으나, **대규모 스케일에서의 검증이 없음**. BLT는 8B 파라미터까지 검증됨

#### SpaceByte vs BLT
- **공통점**: 공백 기반 패칭 아이디어 공유
- **차이점**: SpaceByte의 단순 규칙 기반 패칭은 Llama 3에 미치지 못함. BLT는 여기에 엔트로피 기반 동적 패칭 + n-gram 임베딩 + 크로스-어텐션을 추가하여 격차를 해소

---

## 5. 앞으로의 연구에 미치는 영향 및 고려 사항

### 5.1 앞으로의 연구에 미치는 영향

#### (1) LLM 아키텍처 패러다임 전환 가능성
BLT는 **"고정 어휘 토크나이저가 필수"라는 암묵적 전제를 깨뜨렸습니다.** 8B 파라미터 이상의 대규모에서 토크나이저 없이 BPE 모델과 동등한 성능을 처음으로 달성함으로써, 향후 LLM 연구에서 토크나이저 없는 아키텍처가 하나의 주류 방향으로 자리잡을 가능성이 높습니다.

#### (2) 새로운 스케일링 차원 제시
기존 스케일링 법칙(Chinchilla 등)은 (모델 크기, 데이터 크기)의 2차원 공간에서 최적화를 논했습니다. BLT는 여기에 **패치 크기**라는 새로운 스케일링 축을 추가합니다:

$$\text{동일 추론 FLOP} \Rightarrow \text{패치 크기} \uparrow \Rightarrow \text{모델 크기} \uparrow$$

이 발견은 향후 스케일링 법칙 연구에서 패치 크기를 변수로 포함해야 함을 시사합니다.

#### (3) 다국어 및 저자원 언어 모델링 연구 촉진
토크나이저 기반 모델의 다국어 불평등 문제를 구조적으로 해결함으로써, **진정한 범언어적 모델** 연구에 새로운 방향을 제시합니다. 특히 저자원 언어, 비표준 스크립트, 방언 처리에서의 응용이 기대됩니다.

#### (4) 강건한 AI 시스템 연구
노이즈 입력에 대한 BLT의 구조적 강건성은 **적대적 공격(adversarial attack)** 및 **노이즈 환경** 연구에 중요한 기준점을 제공합니다.

#### (5) 멀티모달 확장 가능성
바이트 레벨 처리는 텍스트뿐만 아니라 이미지, 오디오, 비디오 등 **모든 데이터를 바이트 스트림으로 통합**하는 진정한 멀티모달 모델로의 확장 가능성을 엽니다. 이미지 픽셀, 오디오 파형 등을 동일한 아키텍처로 처리할 수 있는 잠재력이 있습니다.

#### (6) "Byte-ifying" 기존 모델의 새로운 워크플로우
Llama 3.1 가중치 초기화 실험(Table 5)은 **기존에 학습된 대규모 모델을 토크나이저 없는 모델로 변환**하는 실용적 방법론을 제시합니다. 이는 처음부터 재학습하는 비용 없이 BLT의 이점을 활용하는 방향으로 연구가 진행될 수 있습니다.

---

### 5.2 앞으로 연구 시 고려할 점

#### (1) BLT 전용 스케일링 법칙 도출 필요
현재 논문은 Llama 3용으로 설계된 BPE 스케일링 법칙을 그대로 적용했습니다. **BLT 고유의 최적 (모델 크기, 데이터 크기, 패치 크기) 삼각 관계**를 새롭게 도출해야 합니다. 이는 더 큰 스케일에서 BLT의 잠재력을 충분히 활용하기 위해 필수적입니다.

#### (2) 엔드-투-엔드 패칭 모델 학습
현재 엔트로피 모델은 별도로 학습됩니다. 패칭 경계 결정을 **전체 모델과 함께 엔드-투-엔드로 학습**하면 더 최적화된 패칭이 가능할 수 있으며, 추가적인 일반화 이점을 가져올 수 있습니다.

#### (3) 벽시계 시간 최적화
이론적 FLOP 동등성과 실제 학습/추론 속도 사이의 간극을 줄이는 **하드웨어 최적화 및 커스텀 커널 개발**이 필요합니다. BLT의 동적 패칭은 배치 처리를 복잡하게 만들므로, FlexAttention 이상의 최적화가 요구됩니다.

#### (4) 엔트로피 드리프트 해결
구조화된 데이터(MMLU 등)에서 반복 패턴이 지나치게 큰 패치를 형성하는 문제를 더 근본적으로 해결하는 방법이 필요합니다. 현재의 컨텍스트 리셋 방법보다 **더 정교한 엔트로피 컨텍스트 관리** 기법을 연구해야 합니다.

#### (5) 더 큰 스케일(70B+) 검증
현재 최대 8B 파라미터까지 검증되었으나, 실제 상용 모델(70B, 100B+)에서도 같은 스케일링 이점이 유지되는지 검증이 필요합니다. 논문이 시사하듯 더 큰 패치 크기가 더 큰 스케일에서 더 유리할 수 있습니다.

#### (6) 파인튜닝 및 RLHF와의 호환성
사전학습에서의 성능이 확인되었으나, **지시 따르기(instruction following), RLHF, 강화학습 기반 파인튜닝**과의 호환성 및 성능 유지 여부를 연구해야 합니다.

#### (7) 패치 크기의 태스크별 적응
현재 패치 크기는 훈련 데이터 믹스 기준으로 설정됩니다. **추론 시 태스크에 따라 동적으로 패치 크기 임계값을 조정**하는 방법(논문에서도 임계값을 0.6→0.1로 조정 시 성능 향상이 보고됨)의 체계적 연구가 필요합니다.

#### (8) 코드, 수학 특화 최적화
코드와 수학은 엔트로피 분포가 자연어와 다릅니다. **도메인별 엔트로피 모델** 또는 **도메인 적응형 패칭** 전략이 성능을 더욱 향상시킬 수 있습니다. 현재 HumanEval에서 BLT-Entropy(35.4%)가 Llama 3(31.1%)보다 높지만, 더 큰 격차를 만들 여지가 있습니다.

#### (9) 멀티모달 확장 연구
바이트 레벨 처리의 장점을 텍스트 이외의 모달리티에 확장하는 연구가 유망합니다. 이미지 바이트, 오디오 PCM 데이터 등을 동일한 BLT 프레임워크로 처리하는 **진정한 바이트-레벨 멀티모달 모델**의 가능성을 탐색해야 합니다.

#### (10) 비교 실험의 공정성 확보
BLT와 BPE 모델의 비교 시 **컨텍스트 길이, 학습 데이터 바이트 수, FLOP 예산** 등을 엄밀하게 제어해야 합니다. 논문도 이 점을 강조하고 있으나, 향후 연구에서도 동일한 수준의 엄밀성이 요구됩니다.

---

## 참고자료

**주요 참고 논문 (본 논문 PDF에서 인용)**:

1. **Pagnoni et al. (2024)** - "Byte Latent Transformer: Patches Scale Better Than Tokens" (arXiv:2412.09871v1) — *본 논문*
2. **Xue et al. (2022)** - "ByT5: Towards a Token-Free Future with Pre-trained Byte-to-Byte Models" — *Transactions of the Association for Computational Linguistics*
3. **Yu et al. (2023)** - "MegaByte: Predicting Million-Byte Sequences with Multiscale Transformers" — *NeurIPS 2023*
4. **Wang et al. (2024)** - "MambaByte: Token-free Selective State Space Model" — *arXiv*
5. **Slagle (2024)** - "SpaceByte: Towards Deleting Tokenization from Large Language Modeling" — *arXiv*
6. **Nawrot et al. (2022)** - "Hierarchical Transformers Are More Efficient Language Models" — *NAACL*
7. **Nawrot et al. (2023)** - "Efficient Transformers with Dynamic Token Pooling" — *ACL*
8. **Clark et al. (2022)** - "CANINE: Pre-training an Efficient Tokenization-Free Encoder" — *TACL*
9. **El Boukkouri et al. (2020)** - "CharacterBERT: Reconciling ELMo and BERT" — *COLING*
10. **Dubey et al. (2024)** - "The Llama 3 Herd of Models" — *arXiv*
11. **Hoffmann et al. (2022)** - "Training Compute-Optimal Large Language Models (Chinchilla)" — *NeurIPS*
12. **Jaegle et al. (2021)** - "Perceiver: General Perception with Iterative Attention" — *ICML*
13. **Gu and Dao (2023)** - "Mamba: Linear-Time Sequence Modeling with Selective State Spaces" — *arXiv*
14. **Lester et al. (2024)** - "Training LLMs over Neurally Compressed Text" — *arXiv*
15. **Goyal et al. (2022)** - "The FLORES-101 Evaluation Benchmark" — *TACL*
16. **Edman et al. (2024)** - "CUTE: Measuring LLMs' Understanding of Their Tokens" — *arXiv*
17. **Touvron et al. (2023)** - "Llama 2: Open Foundation and Fine-tuned Chat Models" — *arXiv*
18. **Dao et al. (2022)** - "FlashAttention: Fast and Memory-Efficient Exact Attention" — *NeurIPS*
19. **Su et al. (2021)** - "RoFormer: Enhanced Transformer with Rotary Position Embedding" — *arXiv*
20. **Kaplan et al. (2020)** - "Scaling Laws for Neural Language Models" — *arXiv*

**코드 공개**: https://github.com/facebookresearch/blt
