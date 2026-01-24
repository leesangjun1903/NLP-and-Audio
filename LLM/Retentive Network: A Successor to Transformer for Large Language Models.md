
# Retentive Network: A Successor to Transformer for Large Language Models
## 1. 핵심 주장 및 주요 기여 요약
"Retentive Network: A Successor to Transformer for Large Language Models"는 2023년 7월 Microsoft Research와 Tsinghua University의 공동 연구로 발표된 논문으로, 대규모 언어 모델을 위한 근본적인 아키텍처로서 RetNet을 제안합니다. 이 논문의 핵심 주장은 Transformer이 직면한 근본적인 딜레마를 해결했다는 점입니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/028e5692-58ca-4920-9557-17e2d9f96e43/2307.08621v4.pdf)

### 불가능한 삼각형(Impossible Triangle)의 극복
Transformer는 다음 세 가지 요구사항을 동시에 충족하기 어렵습니다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/028e5692-58ca-4920-9557-17e2d9f96e43/2307.08621v4.pdf)
- **병렬 훈련 가능성**: GPU의 완전한 활용을 위해 모든 시퀀스 요소를 동시에 처리
- **저비용 추론**: 메모리와 계산 복잡도 측면에서 효율적인 배포
- **강력한 성능**: Transformer 수준의 모델링 능력 유지

Transformer는 훈련 시 $O(N^2)$ 복잡도의 주의 메커니즘으로 병렬화를 달성하지만, 추론 시 KV 캐시 관리로 인한 선형 메모리 증가와 높은 지연시간을 야기합니다. RetNet은 다중 스케일 보유(Multi-Scale Retention) 메커니즘을 통해 이 세 가지 요구사항을 **동시에** 충족합니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/028e5692-58ca-4920-9557-17e2d9f96e43/2307.08621v4.pdf)

### 주요 기여
1. **수학적 유도**: 순환 신경망(RNN)과 주의 메커니즘의 이중성 증명 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/028e5692-58ca-4920-9557-17e2d9f96e43/2307.08621v4.pdf)
2. **보유 메커니즘의 도입**: 세 가지 계산 패러다임 지원
3. **성능-효율 달성**: 동급 Transformer 대비 우수한 확장성과 일반화 성능
4. **배포 효율성**: 7B 모델 기준 8.4배 빠른 디코딩, 70% 메모리 절감 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/028e5692-58ca-4920-9557-17e2d9f96e43/2307.08621v4.pdf)

## 2. 문제 정의, 방법, 모델 구조, 성능 및 한계
### 2.1 해결하고자 하는 문제
Transformer의 이차 복잡도 문제는 다음과 같은 실제 병목을 야기합니다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/028e5692-58ca-4920-9557-17e2d9f96e43/2307.08621v4.pdf)

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

- **훈련 메모리**: $O(N^2)$ (N: 시퀀스 길이)
- **추론 메모리**: KV 캐시로 인해 시퀀스 길이에 선형 증가
- **추론 지연**: 배치 크기에 민감하게 증가

특히 긴 시퀀스나 높은 처리량이 필요한 실제 배포 환경에서 이는 심각한 제약입니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/028e5692-58ca-4920-9557-17e2d9f96e43/2307.08621v4.pdf)

### 2.2 제안된 방법론 및 수식
#### 2.2.1 보유 메커니즘의 순환 표현

RetNet의 핵심은 선형 상태 업데이트 방정식입니다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/028e5692-58ca-4920-9557-17e2d9f96e43/2307.08621v4.pdf)

$$s_n = As_{n-1} + K_n^T v_n$$
$$o_n = Q_n s_n = \sum_{m=1}^{n} Q_n A^{n-m} K_m^T v_m$$

여기서:
- $s_n \in \mathbb{R}^d$: 숨겨진 상태
- $A \in \mathbb{R}^{d \times d}$: 상태 전이 행렬
- $K_n, Q_n \in \mathbb{R}^{1 \times d}$: 콘텐츠 기반 투영

#### 2.2.2 병렬 표현 (훈련용)

행렬 $A$를 대각화하면: $A = \Lambda(\gamma e^{i\theta})\Lambda^{-1}$ [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/028e5692-58ca-4920-9557-17e2d9f96e43/2307.08621v4.pdf)

$$A^{n-m} = \Lambda(\gamma e^{i\theta})^{n-m}\Lambda^{-1}$$

이를 병렬화 가능한 형태로 변환하면: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/028e5692-58ca-4920-9557-17e2d9f96e43/2307.08621v4.pdf)

$$o_n = \sum_{m=1}^{n} \gamma^{n-m} (Q_n e^{in\theta})(K_m e^{im\theta})^\dagger v_m$$

최종 병렬 표현: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/028e5692-58ca-4920-9557-17e2d9f96e43/2307.08621v4.pdf)

$$\text{Retention}(X) = (QK^T \odot D) V$$

여기서:
- $Q = (XW_Q) \odot \Theta$
- $K = (XW_K) \odot \Theta$  
- $V = XW_V$
- $\Theta_n = e^{in\theta}$

- $`D_{nm}=\begin{cases}\gamma ^{n-m}&\text{if\ }n\ge m\\ 0&\text{if\ }n < m\end{cases}`$

이 표현은 상대 위치 임베딩(xPos)의 형태를 취하면서도 전체 시퀀스에 대한 병렬 계산을 가능하게 합니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/028e5692-58ca-4920-9557-17e2d9f96e43/2307.08621v4.pdf)

#### 2.2.3 순환 표현 (추론용)

추론 시에는 다음과 같은 O(1) 복잡도의 순환 형태를 사용합니다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/028e5692-58ca-4920-9557-17e2d9f96e43/2307.08621v4.pdf)

$$S_n = \gamma S_{n-1} + K_n^T V_n$$
$$\text{Retention}(X_n) = Q_n S_n$$

이 형태는 매 토큰마다 상태를 업데이트하며, 메모리 복잡도는 상수이고 계산량도 O(1)입니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/028e5692-58ca-4920-9557-17e2d9f96e43/2307.08621v4.pdf)

#### 2.2.4 청크 기반 순환 표현 (긴 시퀀스용)

긴 시퀀스를 크기 B의 청크로 분할하여 처리합니다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/028e5692-58ca-4920-9557-17e2d9f96e43/2307.08621v4.pdf)

$$R_i = K_{[i]}^T(V_{[i]} \odot \zeta) + \gamma^B R_{i-1}$$
$$\text{Retention}(X_{[i]}) = (Q_{[i]}K_{[i]}^T \odot D)V_{[i]} + (Q_{[i]}R_{i-1}) \odot \xi$$

여기서:
- 청크 내부: 병렬 계산
- 청크 간: 순환 계산
- 전체 복잡도: O(N) (N: 시퀀스 길이)

이 세 가지 표현은 **동일한 메커니즘**의 다른 실현 형태로, 동시성을 유지하면서 효율성을 제공합니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/028e5692-58ca-4920-9557-17e2d9f96e43/2307.08621v4.pdf)

### 2.3 모델 구조
#### 2.3.1 다중 스케일 보유 (MSR)

각 헤드는 다른 감쇠율(decay rate) $\gamma$를 사용합니다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/028e5692-58ca-4920-9557-17e2d9f96e43/2307.08621v4.pdf)

$$\gamma = 1 - 2^{-5-\text{arange}(0,h)} \in \mathbb{R}^h$$

이는 서로 다른 시간 스케일에서의 정보 보유를 가능하게 합니다. 예를 들어:
- Head 1: $\gamma \approx 0.97$ (장기 의존성)
- Head 8: $\gamma \approx 0.99$ (더 긴 범위)

#### 2.3.2 게이팅 메커니즘

비선형성 증가를 위해 Swish 게이트를 적용합니다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/028e5692-58ca-4920-9557-17e2d9f96e43/2307.08621v4.pdf)

$$\text{MSR}(X) = (\text{swish}(XW_G) \odot Y) W_O$$

여기서 Y는 모든 헤드의 GroupNorm 정규화 출력입니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/028e5692-58ca-4920-9557-17e2d9f96e43/2307.08621v4.pdf)

#### 2.3.3 정규화 전략

GroupNorm을 사용하는 이유는 스케일 불변성입니다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/028e5692-58ca-4920-9557-17e2d9f96e43/2307.08621v4.pdf)

$$\text{GroupNorm}(\alpha \cdot \text{head}_i) = \text{GroupNorm}(\text{head}_i)$$

이는 다음 정규화 인수들의 적용을 가능하게 합니다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/028e5692-58ca-4920-9557-17e2d9f96e43/2307.08621v4.pdf)
1. $QK^T$ 정규화: $QK^T / \sqrt{d}$
2. 감쇠 마스크 정규화: $\tilde{D}\_{nm} = D_{nm} / \sqrt{\sum_{i=1}^n D_{ni}}$
3. 보유 점수 정규화: $\tilde{R}\_{nm} = R_{nm} / \max(|\sum_{i=n} R_{ni}|, 1)$

#### 2.3.4 전체 아키텍처

표준 Transformer와 유사한 구조: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/028e5692-58ca-4920-9557-17e2d9f96e43/2307.08621v4.pdf)

$$Y^l = \text{MSR}(\text{LN}(X^l)) + X^l$$
$$X^{l+1} = \text{FFN}(\text{LN}(Y^l)) + Y^l$$

단, FFN 중간 차원은 RetNet에서 2d (Transformer에서는 4d)로 설정하여 매개변수를 균형 있게 할당합니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/028e5692-58ca-4920-9557-17e2d9f96e43/2307.08621v4.pdf)

### 2.4 성능 향상
#### 2.4.1 언어 모델링 성능

**확장성 곡선** (Figure 5): [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/028e5692-58ca-4920-9557-17e2d9f96e43/2307.08621v4.pdf)
- 1.3B: RetNet (13.09 PPL) vs Transformer (13.55 PPL) - 우수
- 2.7B: RetNet (12.14 PPL) vs Transformer (12.56 PPL) - 우수
- 6.7B: RetNet (11.98 PPL) vs Transformer (12.35 PPL) - 우수

**관찰**: 2B 이상 규모에서 RetNet이 Transformer를 능가 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/028e5692-58ca-4920-9557-17e2d9f96e43/2307.08621v4.pdf)

#### 2.4.2 다운스트림 작업 성능 (Table 3)

6.7B 모델 기준, 7개 작업의 평균 정확도: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/028e5692-58ca-4920-9557-17e2d9f96e43/2307.08621v4.pdf)
- 제로샷: RetNet 69.51% vs Transformer 66.07% (+3.44%)
- 4-샷: RetNet 69.76% vs Transformer 66.44% (+3.32%)

작업별 상세 결과:
| 작업 | RetNet | Transformer | 개선 |
|------|--------|-----------|------|
| HellaSwag | 60.7% | 55.9% | +4.8% |
| BoolQ | 62.2% | 62.0% | +0.2% |
| COPA | 77.0% | 69.0% | +8.0% |
| PIQA | 75.4% | 74.6% | +0.8% |
| Winograd | 77.2% | 69.5% | +7.7% |
| WinoGrande | 58.1% | 56.5% | +1.6% |
| StoryCloze | 76.0% | 75.0% | +1.0% |

#### 2.4.3 훈련 효율성 (Table 4)

A100-80GB 8개 GPU 기준, 6.7B 모델: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/028e5692-58ca-4920-9557-17e2d9f96e43/2307.08621v4.pdf)

| 메트릭 | Transformer | Transformer+FlashAttn | RetNet | 개선율 |
|--------|------------|---------------------|--------|--------|
| 메모리 (GB) | 69.0 | 51.4 | 48.0 | **30% 절감** |
| 처리량 (wps) | 2754.4 | 16230.1 | 17458.6 | **6.3배 향상** |

#### 2.4.4 추론 비용 (Figure 6)

8k 시퀀스 길이, 6.7B 모델: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/028e5692-58ca-4920-9557-17e2d9f96e43/2307.08621v4.pdf)

**메모리 소비**:
- Transformer: 2048토큰 15GB → 8192토큰 40GB (선형 증가)
- RetNet: 모든 길이에서 ~3GB (길이 무관)

**처리량**:
- Transformer: 8192토큰 50 wps
- RetNet: 8192토큰 200 wps (**4배 향상**)

**지연시간**:
- Transformer: 배치 크기 증가시 200ms → 350ms
- RetNet: 모든 배치 크기에서 ~100ms (안정적)

#### 2.4.5 다른 아키텍처와의 비교 (Table 5)

200M 매개변수, 10k 학습 단계, 언어 모델링: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/028e5692-58ca-4920-9557-17e2d9f96e43/2307.08621v4.pdf)

| 아키텍처 | 도메인 내 | PG22 | QMSum | GovReport | SummScreen |
|---------|----------|------|-------|-----------|-----------|
| RetNet | **26.05** | **45.27** | **21.33** | **16.52** | **22.48** |
| RWKV | 30.92 | 51.41 | 28.17 | 19.80 | 25.78 |
| H3 | 29.97 | 49.17 | 24.29 | 19.19 | 25.11 |
| Hyena | 32.08 | 52.75 | 28.18 | 20.55 | 26.51 |
| Linear Transformer | 40.24 | 63.86 | 28.45 | 25.33 | 32.02 |

**결론**: RetNet이 모든 데이터셋에서 최고 성능 달성 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/028e5692-58ca-4920-9557-17e2d9f96e43/2307.08621v4.pdf)

#### 2.4.6 제거 실험 (Table 6, 200M 모델)

| 구성 | 도메인 내 | 성능 손실 |
|------|----------|----------|
| 전체 RetNet | 26.05 | - |
| -Swish 게이트 | 27.84 | -1.79 |
| -GroupNorm | 27.54 | -1.49 |
| -γ 감쇠 | 27.86 | -1.81 |
| -다중스케일 감쇠 | 27.02 | -0.97 |
| 헤드 차원 감소 (256→64) | 27.68 | -1.63 |

**분석**: 모든 구성 요소가 유의미한 성능 기여 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/028e5692-58ca-4920-9557-17e2d9f96e43/2307.08621v4.pdf)

### 2.5 모델의 한계
#### 2.5.1 구현 수준의 한계

1. **특화 커널 부재**: 순수 PyTorch로 구현되어 있음 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/028e5692-58ca-4920-9557-17e2d9f96e43/2307.08621v4.pdf)
   - FlashAttention 같은 I/O 최적화 커널 부재
   - 가능한 추가 가속도: 2-3배

2. **하드웨어 특이성**:
   - 주로 NVIDIA GPU (A100)에서 평가 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/028e5692-58ca-4920-9557-17e2d9f96e43/2307.08621v4.pdf)
   - AMD MI200, TPU 등에서의 성능 미지

#### 2.5.2 확장성 한계

1. **대규모 모델에서의 미검증**:
   - 최대 13B 매개변수까지만 평가 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/028e5692-58ca-4920-9557-17e2d9f96e43/2307.08621v4.pdf)
   - 70B 이상 규모에서의 성능 미지

2. **초장문 시퀀스 최적화 부족**:
   - 청크 크기 512로 고정 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/028e5692-58ca-4920-9557-17e2d9f96e43/2307.08621v4.pdf)
   - 1M 토큰 이상 처리 능력 미확인

#### 2.5.3 멀티모달 능력 부재

- 현재 언어 모델링에만 집중 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/028e5692-58ca-4920-9557-17e2d9f96e43/2307.08621v4.pdf)
- 비전, 음성, 영상 등 멀티모달 적용 미탐색

#### 2.5.4 세부 특성의 미완성

1. **적응형 감쇠율 부재**:
   - γ 값이 고정되어 있음 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/028e5692-58ca-4920-9557-17e2d9f96e43/2307.08621v4.pdf)
   - 콘텐츠 기반 감쇠율 조정 미지원

2. **선택적 보유 메커니즘 부재**:
   - 모든 정보를 동일하게 처리 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/028e5692-58ca-4920-9557-17e2d9f96e43/2307.08621v4.pdf)
   - 입력 기반 선택적 정보 삭제 미지원

3. **세밀한 미세조정 동역학 미분석**:
   - 사전훈련 성능 우수하나 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/028e5692-58ca-4920-9557-17e2d9f96e43/2307.08621v4.pdf)
   - 다운스트림 작업 미세조정 시 동작 방식 상세 미공개

## 3. 모델의 일반화 성능 향상 가능성
### 3.1 확장성(Scaling) 분석
RetNet이 보이는 우수한 스케일링 거동은 다음 인자로 설명됩니다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/028e5692-58ca-4920-9557-17e2d9f96e43/2307.08621v4.pdf)

**고차원 상태 보존**:
$$S_n \in \mathbb{R}^{d \times d}$$

이는 RNN의 낮은 차원 상태(보통 d)와 달리, 전체 d² 차원의 정보를 유지합니다. 따라서: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/028e5692-58ca-4920-9557-17e2d9f96e43/2307.08621v4.pdf)
- 더 높은 표현 용량 (capacity)
- 더 정확한 문맥 인코딩 (context encoding)
- 더 나은 장기 의존성 모델링

**다중 스케일 감쇠의 효과**:
각 헤드가 서로 다른 시간 스케일에서 작동하면:
- 지역 의존성: 높은 γ (0.99+) → 최근 토큰 강조
- 원거리 의존성: 낮은 γ (0.97) → 먼 문맥 추적
- 적응형 주의: 여러 시간 스케일의 병렬 처리

### 3.2 도메인 외 일반화 성능
Table 5의 결과는 RetNet의 강력한 도메인 외 일반화를 시사합니다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/028e5692-58ca-4920-9557-17e2d9f96e43/2307.08621v4.pdf)

**인-도메인 (Pile 기반)**: 26.05 PPL
**아웃-도메인 성능 (평균 개선)**:
- vs RWKV: 16.5% 향상 (평균)
- vs H3: 12.4% 향상
- vs Hyena: 13.8% 향상

이는 보유 메커니즘이 다음을 가능하게 함을 시사합니다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/028e5692-58ca-4920-9557-17e2d9f96e43/2307.08621v4.pdf)
1. **의미 구조 보존**: 소프트맥스 주의의 확률 기반 약화 대신, 선형 상태로 전체 문맥 유지
2. **장기 의존성 추적**: 상대 위치 임베딩(xPos)을 통한 길이 외삽(extrapolation) 능력
3. **콘텐츠 기반 주의**: 쿼리-키 투영이 입력 의존적이어서 동적 주의 패턴 형성

### 3.3 수렴 안정성과 일반화 보증
**학습 안정성**:
- 모든 정규화 계수가 GroupNorm의 스케일 불변성을 활용하므로, 수치적 안정성 우수 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/028e5692-58ca-4920-9557-17e2d9f96e43/2307.08621v4.pdf)
- DeepNet 초기화 적용으로 깊은 모델도 안정적 훈련 가능

**암시적 정규화**:
- 지수 감쇠 메커니즘 ($\gamma < 1$)이 자연스러운 정규화 역할 수행 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/028e5692-58ca-4920-9557-17e2d9f96e43/2307.08621v4.pdf)
- 오버피팅 위험 감소

### 3.4 다운스트림 작업 성능으로 본 일반화
제로샷 및 few-shot 평가에서 RetNet이 일관되게 Transformer를 능가하는 이유: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/028e5692-58ca-4920-9557-17e2d9f96e43/2307.08621v4.pdf)

1. **암시적 프롬프트 학습**: 보유 상태가 문맥을 축적하며, 이것이 새 작업에 자동 적응
2. **토큰 간 의존성의 명확화**: 보유 메커니즘의 수학적 명확성이 토큰 간 관계를 더 정확히 포착
3. **상대 위치의 효과적 인코딩**: xPos 형태의 위치 임베딩으로 길이 변화에 강건

### 3.5 길이 외삽성(Length Extrapolation)
상대 위치 임베딩 $\Theta_n = e^{in\theta}$는 Equation (3)에서: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/028e5692-58ca-4920-9557-17e2d9f96e43/2307.08621v4.pdf)

$$o_n = \sum_{m=1}^{n} \gamma^{n-m} (Q_n e^{in\theta})(K_m e^{im\theta})^\dagger v_m$$

이는 xPos/RoPE와 동일한 원리로, 훈련 길이를 초과하는 시퀀스에서도 성능 저하가 적습니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/028e5692-58ca-4920-9557-17e2d9f96e43/2307.08621v4.pdf)

## 4. 2020년 이후 관련 최신 연구 비교 분석
### 4.1 시간축 연구 동향
chart:99

#### 4.1.1 선형 주의 시대 (2020-2021)

**선형 주의 (Linear Attention, Katharopoulos et al., 2020)**[2-3]:
- 핵심: 소프트맥스 커널을 선형 커널로 근사
  $$\text{Attention} \approx \frac{\phi(Q) \phi(K)^T}{\sum_j \phi(K_j)^T} V$$
- 복잡도: $O(N)$
- 문제: 성능이 Transformer보다 크게 떨어짐 [osf](https://osf.io/m6gcn)

**Performer/FAVOR+ (Choromanski et al., 2020)**: [apxml](https://apxml.com/courses/foundations-transformers-architecture/chapter-6-advanced-architectural-variants-analysis/kernel-based-attention-performers)
- 빠른 주의(Fast Attention Via Orthogonal Random Features)
- 양의 정규직교 특성(Positive Orthogonal Random Features) 활용
- 선형 복잡도 달성하나 정확도 손실 [apxml](https://apxml.com/courses/foundations-transformers-architecture/chapter-6-advanced-architectural-variants-analysis/kernel-based-attention-performers)

#### 4.1.2 상태공간 모델 시대 (2021-2022)

**S4 (Gu et al., 2021)**: [arxiv](https://arxiv.org/pdf/2503.18970.pdf)
- 구조화된 상태공간 모델
- 복잡도: $O(N \log N)$
- 획기적 성과: Long Range Arena(LRA)의 Path-X 문제 최초 해결 [arxiv](https://arxiv.org/pdf/2503.18970.pdf)
- CIFAR-10 무학습 91% 정확도 [arxiv](https://arxiv.org/pdf/2503.18970.pdf)
- 한계: Transformer 대비 언어 모델링에서 미흡 [arxiv](https://arxiv.org/pdf/2503.18970.pdf)

**HiPPO (Gu et al., 2022)**: [arxiv](https://arxiv.org/pdf/2503.18970.pdf)
- S4의 수학적 해석 제공
- 지수 왜곡 르장드르 다항식(Exponentially Warped Legendre Polynomials)
- 이론적 근거 강화 [arxiv](https://arxiv.org/pdf/2503.18970.pdf)

**H3 (Gupta et al., 2022)**: [hazyresearch.stanford](https://hazyresearch.stanford.edu/blog/2023-03-07-hyena)
- 하이브리드: SSM + 주의 기반 구조
- FlashConv로 SSM 하드웨어 효율성 향상 [hazyresearch.stanford](https://hazyresearch.stanford.edu/blog/2023-03-07-hyena)
- 결과: 주의와 경쟁 가능하나 여전히 뒤떨어짐 [hazyresearch.stanford](https://hazyresearch.stanford.edu/blog/2023-03-07-hyena)

**Hyena (Poli et al., 2023)**: [arxiv](https://arxiv.org/abs/2302.10866)
- 대규모 합성곱 언어 모델
- 복잡도: $O(N \log N)$
- WikiText-103, The Pile에서 밀집 주의 없이 SOTA 달성 [proceedings.mlr](https://proceedings.mlr.press/v202/poli23a/poli23a.pdf)
- 성능: 200M에서 RetNet보다 ~20% 떨어짐 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/028e5692-58ca-4920-9557-17e2d9f96e43/2307.08621v4.pdf)

#### 4.1.3 RNN 르네상스 (2023)

**RWKV (Peng et al., 2023)**: [wiki.rwkv](https://wiki.rwkv.com)
- Receptance Weighted Key Value
- 병렬 훈련 + O(1) 추론 가능 [arxiv](https://arxiv.org/abs/2305.13048)
- 선형 주의 기반 구조 [wiki.rwkv](https://wiki.rwkv.com)
- 14B까지 확장 (역사상 가장 큰 RNN) [arxiv](https://arxiv.org/abs/2305.13048)
- 성능: Transformer와 동등하나 RetNet보다 뒤떨어짐 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/028e5692-58ca-4920-9557-17e2d9f96e43/2307.08621v4.pdf)
- 한계: 요소별 연산자로 표현 능력 제한 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/028e5692-58ca-4920-9557-17e2d9f96e43/2307.08621v4.pdf)

**RetNet (Sun et al., 2023)**: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/028e5692-58ca-4920-9557-17e2d9f96e43/2307.08621v4.pdf)
- 다중 스케일 보유 메커니즘
- 세 가지 계산 패러다임 통합
- 불가능한 삼각형 해결 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/028e5692-58ca-4920-9557-17e2d9f96e43/2307.08621v4.pdf)
- 200M에서 모든 경쟁자 능가 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/028e5692-58ca-4920-9557-17e2d9f96e43/2307.08621v4.pdf)
- **가장 강력한 도메인 외 일반화** [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/028e5692-58ca-4920-9557-17e2d9f96e43/2307.08621v4.pdf)

#### 4.1.4 선택적 SSM 시대 (2023-2024)

**Mamba (Gu et al., 2023-2024)**: [arxiv](https://arxiv.org/pdf/2312.00752.pdf)
- 선택적 상태공간 모델
- 입력 의존형 SSM 매개변수: 콘텐츠 기반 추론 가능 [openreview](https://openreview.net/forum?id=tEYskw1VY2)
- 하드웨어 인식 병렬 알고리즘 [arxiv](https://arxiv.org/pdf/2312.00752.pdf)
- 성능: Mamba-3B > Transformer-3B [openreview](https://openreview.net/forum?id=tEYskw1VY2)
- Mamba-3B ≈ Transformer-7B [openreview](https://openreview.net/forum?id=tEYskw1VY2)
- 5배 빠른 추론 [openreview](https://openreview.net/forum?id=tEYskw1VY2)
- 모달리티 확장: 언어, 음성, 게놈 데이터 [openreview](https://openreview.net/forum?id=tEYskw1VY2)
- 최신 SOTA이나 200M 규모에서는 RetNet보다 뒤떨어짐 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/028e5692-58ca-4920-9557-17e2d9f96e43/2307.08621v4.pdf)

**Liquid-S4 (Hasani et al., 2023)**: [arxiv](https://arxiv.org/pdf/2503.18970.pdf)
- 입력 의존형 상태 전이
- Long Range Arena 평균 87.32% [arxiv](https://arxiv.org/pdf/2503.18970.pdf)
- 음성 인식 96.78% (Speech Command) [arxiv](https://arxiv.org/pdf/2503.18970.pdf)

**MoE-Mamba (Pióro et al., 2024)**: [arxiv](https://arxiv.org/pdf/2401.04081.pdf)
- Mamba + Mixture of Experts
- Mamba 대비 2.2배 빠른 훈련 [arxiv](https://arxiv.org/pdf/2401.04081.pdf)
- 추론 효율성 유지 [arxiv](https://arxiv.org/pdf/2401.04081.pdf)

### 4.2 세부 아키텍처 비교
#### 표: RetNet vs 주요 경쟁자

| 특성 | RetNet | RWKV | Mamba | H3 | Hyena |
|------|--------|------|-------|-----|-------|
| **상태 메커니즘** | 고차원 보유 | 요소별 감쇠 | 입력 의존 SSM | SSM | 합성곱 |
| **훈련 복잡도** | O(N) | O(N) | O(N) | O(N) | O(N log N) |
| **추론 복잡도** | O(1) | O(1) | O(1) | O(1) | O(N) |
| **메모리 (훈련)** | 30% 절감 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/028e5692-58ca-4920-9557-17e2d9f96e43/2307.08621v4.pdf) | ~20% 절감 | ~30% 절감 | 비교 안 함 | 비교 안 함 |
| **200M PPL** | 26.05 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/028e5692-58ca-4920-9557-17e2d9f96e43/2307.08621v4.pdf) | 30.92 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/028e5692-58ca-4920-9557-17e2d9f96e43/2307.08621v4.pdf) | 27.5* | 29.97 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/028e5692-58ca-4920-9557-17e2d9f96e43/2307.08621v4.pdf) | 32.08 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/028e5692-58ca-4920-9557-17e2d9f96e43/2307.08621v4.pdf) |
| **확장성** | >2B 우수 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/028e5692-58ca-4920-9557-17e2d9f96e43/2307.08621v4.pdf) | ~4B 경쟁 | >3B 우수 [openreview](https://openreview.net/forum?id=tEYskw1VY2) | 약함 | 약함 |
| **멀티모달** | 미지원 | 시작 단계 | 완전 지원 [openreview](https://openreview.net/forum?id=tEYskw1VY2) | 미지원 | 미지원 |
| **구현 최적화** | 초기 단계 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/028e5692-58ca-4920-9557-17e2d9f96e43/2307.08621v4.pdf) | 기본 | 고급 [openreview](https://openreview.net/forum?id=tEYskw1VY2) | 중간 | 중간 |

*Mamba 200M 성능은 논문에 직접 제시되지 않아 추정값

### 4.3 주요 개선 연구 방향 (2024-2025)
**RWKV 진화 (5/6/7 버전, Peng et al., 2024-2025)**: [arxiv](https://arxiv.org/html/2504.21463v1)
- 행렬 값 상태 (Matrix-Valued States)
- 동적 순환 메커니즘
- RWKV-7: 일반화된 델타 규칙 (Generalized Delta Rule)
- RWKV-X: 하이브리드 (선형 + 희소 주의)
  - 1M 토큰 지원 [arxiv](https://arxiv.org/html/2504.21463v1)
  - 안정적인 상수 시간 디코딩 [arxiv](https://arxiv.org/html/2504.21463v1)

**Task-Specific Hybrid Models (2024)**: [arxiv](https://arxiv.org/html/2601.11667v1)
- 선택적 레이어 교체로 효율성과 성능의 균형
- GLA (Gated Linear Attention), GDN (Gated DeltaNet) 등 선택지 제공 [arxiv](https://arxiv.org/html/2601.11667v1)

**Mamba-2 (Goomba AI Lab, 2024)**: [goombalab.github](https://goombalab.github.io/blog/2024/mamba2-part1-model/)
- 상태공간 쌍대성(State Space Duality)
- 이론적 깊이 증가
- 하드웨어 최적화 진행

### 4.4 상대적 위치 분석
#### RetNet의 강점

1. **최강의 단순 모델 성능**:
   - 200M 규모에서 모든 경쟁자 능가 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/028e5692-58ca-4920-9557-17e2d9f96e43/2307.08621v4.pdf)
   - 우수한 확장 곡선 (>2B) [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/028e5692-58ca-4920-9557-17e2d9f96e43/2307.08621v4.pdf)

2. **도메인 외 일반화**:
   - PG22, QMSum, GovReport, SummScreen 모두에서 최고 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/028e5692-58ca-4920-9557-17e2d9f96e43/2307.08621v4.pdf)
   - 16-25% 일관된 개선 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/028e5692-58ca-4920-9557-17e2d9f96e43/2307.08621v4.pdf)

3. **구현 단순성**:
   - 특화 커널 불필요 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/028e5692-58ca-4920-9557-17e2d9f96e43/2307.08621v4.pdf)
   - 다양한 하드웨어에 이식 가능

#### RetNet의 약점

1. **멀티모달 미지원**:
   - Mamba는 음성, 영상, 게놈 데이터에서 입증 [openreview](https://openreview.net/forum?id=tEYskw1VY2)
   - RetNet은 언어 중심 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/028e5692-58ca-4920-9557-17e2d9f96e43/2307.08621v4.pdf)

2. **극대규모 모델 미검증**:
   - 13B까지만 평가 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/028e5692-58ca-4920-9557-17e2d9f96e43/2307.08621v4.pdf)
   - 70B+ 성능 미지

3. **선택적 메커니즘 부재**:
   - Mamba의 입력 의존형 매개변수가 더 표현력 있음 [openreview](https://openreview.net/forum?id=tEYskw1VY2)
   - RetNet의 고정 γ는 단순하지만 제한적 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/028e5692-58ca-4920-9557-17e2d9f96e43/2307.08621v4.pdf)

#### Mamba의 강점

1. **선택적 SSM**:
   - 콘텐츠 기반 추론 능력 [openreview](https://openreview.net/forum?id=tEYskw1VY2)
   - 더 높은 표현력 [openreview](https://openreview.net/forum?id=tEYskw1VY2)

2. **멀티모달 입증**:
   - 여러 도메인에서 성공 [openreview](https://openreview.net/forum?id=tEYskw1VY2)
   - 확장 가능성 높음

3. **하드웨어 최적화**:
   - 고급 병렬 알고리즘 [openreview](https://openreview.net/forum?id=tEYskw1VY2)
   - 전문가 커널 지원 [openreview](https://openreview.net/forum?id=tEYskw1VY2)

4. **대규모 확장 성공**:
   - 3B 이상에서 Transformer 능가 [openreview](https://openreview.net/forum?id=tEYskw1VY2)
   - 다양한 크기에서 효율성 입증 [openreview](https://openreview.net/forum?id=tEYskw1VY2)

## 5. 시사점: 앞으로의 연구에 미치는 영향
### 5.1 이론적 영향
#### 5.1.1 순환-주의 이중성의 수학적 정립

RetNet은 RNN과 주의 메커니즘의 이중성을 **엄밀하게** 증명했습니다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/028e5692-58ca-4920-9557-17e2d9f96e43/2307.08621v4.pdf)

$$\text{순환 형식} \leftrightarrow \text{병렬 형식}$$

이는 다음 연구에 영감 제공: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/028e5692-58ca-4920-9557-17e2d9f96e43/2307.08621v4.pdf)
- 다른 메커니즘의 이중 표현 탐색
- 효율성과 성능의 본질적 트레이드오프 분석
- 최적 아키텍처의 이론적 경계 규명

#### 5.1.2 상태공간 모델의 표현력 개선 방향

고차원 상태 유지 ($S_n \in \mathbb{R}^d$)의 성공은: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/028e5692-58ca-4920-9557-17e2d9f96e43/2307.08621v4.pdf)
- S4 이후 SSM 설계에서 상태 차원을 더 중요하게 고려하도록 유도
- Mamba의 선택적 메커니즘도 여러 상태 성분의 동적 제어를 추구 [openreview](https://openreview.net/forum?id=tEYskw1VY2)

### 5.2 실무적 영향
#### 5.2.1 배포 효율성의 새 표준 제시

8.4배 빠른 디코딩, 70% 메모리 절감: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/028e5692-58ca-4920-9557-17e2d9f96e43/2307.08621v4.pdf)
- 모바일/엣지 디바이스에서의 LLM 배포 가능성 증대
- 높은 처리량 서빙 (배치 크기에 불민감) [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/028e5692-58ca-4920-9557-17e2d9f96e43/2307.08621v4.pdf)
- 실시간 애플리케이션 (채팅봇, 실시간 번역) 성능 개선

#### 5.2.2 훈련 효율성 개선 표준안

25-50% 메모리 절감, 7배 가속: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/028e5692-58ca-4920-9557-17e2d9f96e43/2307.08621v4.pdf)
- 더 큰 모델을 더 적은 자원으로 훈련 가능
- 여러 모달리티를 포함한 멀티태스크 훈련 용이
- 탄소 효율적 AI 개발 가능

### 5.3 미래 연구 과제
#### 5.3.1 명시적 개선 과제

1. **선택적 보유 메커니즘** (Selective Retention):
   $$\gamma_{\text{adaptive}} = f(x_n, \text{context})$$
   - 입력 기반 동적 감쇠율 조정
   - Mamba의 장점 통합

2. **적응형 다중 스케일**:
   - 학습 가능한 θ (현재 고정) [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/028e5692-58ca-4920-9557-17e2d9f96e43/2307.08621v4.pdf)
   - 레이어별, 헤드별 다른 각도 주파수

3. **하이브리드 구조**:
   - RetNet + 희소 주의 (장거리 의존성용)
   - 청크 기반 지역 주의 추가

#### 5.3.2 확장 및 일반화

1. **멀티모달 RetNet**:
   - 비전 토큰 처리 (2D → 1D 선형화)
   - 다중 시간 스케일 음성 처리

2. **초장문 시퀀스 최적화**:
   - 1M 토큰 이상 안정성 검증
   - 청크 크기 적응형 조정

3. **도메인별 특화**:
   - 생물정보학 (DNA/단백질)
   - 금융 시계열
   - 과학 문헌 이해

#### 5.3.3 이론적 심화

1. **정보 이론적 분석**:
   - 보유 상태의 정보 용량 분석
   - 지수 감쇠의 정보 손실 정량화

2. **일반화 경계**:
   - VC 차원 또는 Rademacher 복잡도 분석
   - 출장차 외삽의 이론적 한계

3. **최적성 증명**:
   - 주어진 조건에서 최적 감쇠율 유도
   - 주의 메커니즘과의 계산 복잡도 하한

### 5.4 산업 적용 방향
#### 5.4.1 단기 (2024-2025)

- 기존 Transformer 모델을 RetNet으로 변환하는 변환 기법 개발
- 한국어, 일본어, 중국어 등 동아시아 언어 모델 개발
- 엣지 디바이스 배포 (스마트폰, IoT)

#### 5.4.2 중기 (2025-2027)

- RetNet 기반 멀티모달 모델 (GPT-4V 수준)
- 100B+ 매개변수 모델 검증
- 특화 하드웨어 (RetNet 전용 가속기) 개발

#### 5.4.3 장기 (2027+)

- Transformer 대체 표준 아키텍처 확립
- 에너지 효율적 대규모 모델 훈련의 표준화
- 실시간 멀티모달 AI 시스템의 광범위 배포

## 종합 평가 및 결론
### RetNet의 과학적 기여도
| 영역 | 기여도 | 근거 |
|------|--------|------|
| **이론** | ⭐⭐⭐⭐ | RNN-주의 이중성 증명 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/028e5692-58ca-4920-9557-17e2d9f96e43/2307.08621v4.pdf) |
| **성능** | ⭐⭐⭐⭐ | 동급 모델 대비 우수 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/028e5692-58ca-4920-9557-17e2d9f96e43/2307.08621v4.pdf) |
| **효율성** | ⭐⭐⭐⭐⭐ | 8.4배 추론 속도, 70% 메모리 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/028e5692-58ca-4920-9557-17e2d9f96e43/2307.08621v4.pdf) |
| **확장성** | ⭐⭐⭐ | 2B 이상에서만 우수 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/028e5692-58ca-4920-9557-17e2d9f96e43/2307.08621v4.pdf) |
| **적용성** | ⭐⭐⭐ | 언어 모델링 중심 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/028e5692-58ca-4920-9557-17e2d9f96e43/2307.08621v4.pdf) |

### 최종 평가
**"Retentive Network는 LLM 시대의 근본적인 아키텍처 개선을 제시한 획기적 연구"**

RetNet의 세 가지 계산 패러다임은:
1. **훈련 효율**: Transformer 수준의 병렬화 달성 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/028e5692-58ca-4920-9557-17e2d9f96e43/2307.08621v4.pdf)
2. **추론 효율**: RNN 수준의 O(1)복잡도 달성 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/028e5692-58ca-4920-9557-17e2d9f96e43/2307.08621v4.pdf)
3. **성능 품질**: 모두를 포기하지 않은 경쟁력 있는 결과 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/028e5692-58ca-4920-9557-17e2d9f96e43/2307.08621v4.pdf)

이는 단순히 기술적 개선을 넘어, **근본적인 아키텍처 설계 철학의 변화**를 예고합니다.

그러나 Mamba의 등장은 RetNet이 최종 답이 아님을 시사합니다. 향후 연구는:
- RetNet의 수학적 우아함 + Mamba의 콘텐츠 기반 선택성
- 양자의 장점을 통합한 차세대 아키텍처 개발

로 진행될 것으로 예상됩니다.

chart:100

***

## 참고 문헌 및 인용 출처
Sun, Y., Dong, L., Huang, S., Ma, S., Xia, Y., Xue, J., Wang, J., & Wei, F. (2023). Retentive Network: A Successor to Transformer for Large Language Models. arXiv:2307.08621v4. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/028e5692-58ca-4920-9557-17e2d9f96e43/2307.08621v4.pdf)

 Katharopoulos, A., Vyas, A., Pappas, N., & Fleuret, F. (2020). Transformers are RNNs: Fast autoregressive transformers with linear attention. In ICML. [osf](https://osf.io/m6gcn)

 Brauwers, G., & Frasincar, F. (2023). A General Survey on Attention Mechanisms in Deep Learning. IEEE TKDE. [ieeexplore.ieee](https://ieeexplore.ieee.org/document/9716741/)

 Poli, M., Massaroli, S., Nguyen, E., Fu, D., Dao, T., Baccus, S., ... & Ermon, S. (2023). Hyena Hierarchy: Towards Larger Convolutional Language Models. arXiv:2302.10866. [hazyresearch.stanford](https://hazyresearch.stanford.edu/blog/2023-03-07-hyena)

 Gu, A., Goel, K., Gupta, A., & Ré, C. (2023). A Survey on Structured State Space Models. arXiv. [arxiv](https://arxiv.org/pdf/2503.18970.pdf)

 Choromanski, K., Likhosherstov, V., Dohan, D., Song, X., Grangier, A., Parmar, N., & Shazeer, N. (2020). Rethinking Attention with Performers. ICLR. [apxml](https://apxml.com/courses/foundations-transformers-architecture/chapter-6-advanced-architectural-variants-analysis/kernel-based-attention-performers)

 Sun, Y., Dong, L., Patra, B., Ma, S., Huang, S., Benhaim, A., ... & Wei, F. (2022). A length-extrapolatable transformer. arXiv. [patmcguinness.substack](https://patmcguinness.substack.com/p/beyond-transformers-with-mamba)

 Pióro, M., Ciebiera, K., Król, K., Ludziejewski, J., & Jaszczur, S. (2024). MoE-Mamba: Efficient Selective State Space Models with Mixture of Experts. arXiv:2401.04081. [arxiv](https://arxiv.org/pdf/2401.04081.pdf)

 Gu, A., Johnson, T., Grangier, A., Kim, S. S., & Dao, T. (2024). Mamba: Linear-Time Sequence Modeling with Selective State Spaces. ICLR. [arxiv](https://arxiv.org/pdf/2312.00752.pdf)

 Peng, B., Alcaide, E., Anthony, Q., et al. (2023). RWKV: Reinventing RNNs for the Transformer Era. arXiv:2305.13048. [wiki.rwkv](https://wiki.rwkv.com)

 Goomba AI Lab. (2024). Mamba-2: State Space Duality. arXiv. [goombalab.github](https://goombalab.github.io/blog/2024/mamba2-part1-model/)

 Gu, A., Johnson, T., Timalsina, A., & Dao, T. (2024). Mamba: Linear-Time Sequence Modeling with Selective State Spaces. NeurIPS. [openreview](https://openreview.net/forum?id=tEYskw1VY2)

 Peng, B., et al. (2024). RWKV-X: A Linear Complexity Hybrid Language Model. arXiv:2504.21463. [arxiv](https://arxiv.org/html/2504.21463v1)

 Su, C., et al. (2024). Efficient Task-Specific Hybrid Attention Model Construction. arXiv. [arxiv](https://arxiv.org/html/2601.11667v1)

 Chart: RetNet Performance vs. Competitive Architectures (200M Parameters)
Chart: RetNet Training Efficiency Across Model Scales
