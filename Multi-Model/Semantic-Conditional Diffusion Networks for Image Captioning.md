
# Semantic-Conditional Diffusion Networks for Image Captioning

## 요약

**SCD-Net (Semantic-Conditional Diffusion Networks)**은 2022년 12월 발표된 논문으로, 이미지 캡셔닝 작업에 확산 모델(diffusion models)을 적용하는 새로운 패러다임을 제시합니다. 이 연구는 기존 자동회귀(autoregressive) 방식의 느린 추론 속도와 비자동회귀(non-autoregressive) 방식의 단어 반복 및 누락 문제를 동시에 해결하기 위해, 의미론적 조건화를 통한 확산 프로세스와 교사 모델 기반의 강화학습 전략을 제안합니다. COCO 데이터셋에서 CIDEr 131.6을 달성하여 동일 구조의 자동회귀 모델(CIDEr 131.2)을 능가하는 성과를 거두었습니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/124df008-9157-452a-922c-7d79bd75fee5/2212.03099v1.pdf)

***

## 1. 핵심 주장 및 주요 기여

### 1.1 문제 인식

SCD-Net은 세 가지 기술적 병목을 식별합니다. 첫째, 자동회귀 모델은 순차적 토큰 생성으로 인한 이차 계산 복잡도($O(n^2)$ w.r.t. 문장 길이)를 갖습니다. 둘째, 이를 개선한 비자동회귀 모델들(Masked Non-autoregressive Image Captioning, Object-Oriented Non-Autoregressive)은 병렬 처리의 이점을 취하나, 단어 반복 또는 누락 문제가 심각하여 기본 자동회귀 Transformer 대비 성능이 크게 떨어집니다. 셋째, 최근 등장한 Bit Diffusion은 이산 단어를 이진 비트로 변환하여 연속 확산 모델을 적용하나, 여전히 단어 반복/누락을 해결하지 못합니다. [dl.acm](https://dl.acm.org/doi/10.1145/3731715.3733389)

### 1.2 주요 기여

**첫 번째 기여: 의미론적 조건부 확산 프로세스 설계**

기존 확산 모델이 순수 노이즈로부터 시작하여 문장을 생성하는 반면, SCD-Net은 입력 이미지로부터 교차모달 검색 모델을 통해 의미론적으로 관련된 문장들을 검색합니다. 이 의미 사전(semantic prior)을 확산 프로세스의 각 역진 단계에 조건으로 통합함으로써 생성된 단어들이 이미지의 의미론적 내용과 더 강하게 정렬되도록 유도합니다. [arxiv](https://arxiv.org/abs/2506.03067)

**두 번째 기여: 안내된 자기비판 수열 학습(Guided Self-Critical Sequence Training)**

표준 자기비판 수열 학습(SCST)을 비자동회귀 모델에 직접 적용하면, 저품질 샘플로부터의 랜덤 샘플링이 보상 신호를 왜곡하여 성능이 악화됩니다. SCD-Net은 자동회귀 Transformer 교사 모델의 고품질 예측을 일종의 앵커로 사용하여, 강화학습 샘플링에서 이 예측이 반드시 포함되도록 하는 new guided sampling strategy를 제안합니다. 이는 문장 수준의 보상(CIDEr 점수)에 기반한 최적화를 비자동회귀 모델에도 적용 가능하게 만듭니다. [arxiv](https://arxiv.org/abs/2506.03067)

**세 번째 기여: 캐스케이딩 Diffusion Transformer 구조**

이미지 생성 분야의 cascaded diffusion 영감을 받아, SCD-Net은 여러 개의 Diffusion Transformer를 순차적으로 적층합니다. 첫 번째 단계는 의미 조건화를 적용하여 초기 문장을 생성하고, 후속 단계들은 이전 단계의 예측을 추가 조건으로 사용하여 점진적으로 문장의 품질을 개선합니다. [ojs.aaai](https://ojs.aaai.org/index.php/AAAI/article/view/32221)

***

## 2. 해결하는 문제 및 기술적 혁신

### 2.1 단어 반복 및 누락 문제의 근본 원인

비자동회귀 모델이 모든 위치의 토큰을 동시에 독립적으로 생성할 때, 각 위치마다 어떤 단어를 생성할지는 (1) 이미지 특성과 (2) 각 위치의 임베딩 초기값에만 의존합니다. 이미지가 여러 객체를 포함할 경우, 각 위치의 모델은 다른 위치와의 상호작용 없이 유사한 객체를 반복 생성할 수 있습니다. SCD-Net은 의미 사전으로부터 풍부한 의미 정보를 제공함으로써 이를 제약합니다.

### 2.2 시각-언어 정렬의 다층성

기존 확산 모델은 $\mathcal{N}(0, I)$의 순수 가우시안 노이즈로부터 문장을 생성하는데, 이는 이미지 콘텐츠와 무관한 토큰 공간에서 시작합니다. SCD-Net의 의미 사전은 이미지 관련 문장들의 분포로부터 샘플링된 정보를 제공하여, 확산 역진 과정이 이미지-관련 토큰 공간 내에서 움직이도록 제약합니다.

***

## 3. 제안 방법론

### 3.1 이산 데이터를 위한 확산 프로세스 정식화

**전진 프로세스 (Forward Process)**

$$x_t = \sqrt{\text{sigmoid}(-\gamma(t'))}x_0 + \sqrt{\text{sigmoid}(\gamma(t'))}\epsilon$$

여기서 $x_0 \in \mathbb{R}^{n \times N_s}$는 문장을 나타내는 이진 비트의 연속 표현입니다. $n = \lceil \log_2 W \rceil$는 어휘 크기 $W$를 인코딩하기 위한 비트 수, $N_s$는 문장의 최대 길이입니다. 시간 변수 $t' = t/T$는 정규화되고, $\gamma(t')$는 단조증가 함수(예: Affine variance schedules)입니다. [dl.acm](https://dl.acm.org/doi/10.1145/3748653)

**역진 프로세스 (Reverse Process)**

$$\alpha_s = \sqrt{\text{sigmoid}(-\gamma(s'))}, \quad \alpha_t = \sqrt{\text{sigmoid}(-\gamma(t'))}$$
$$\sigma_s = \sqrt{\text{sigmoid}(\gamma(s'))}, \quad c = -\text{expm1}(\gamma(s') - \gamma(t'))$$
$$u(x_t; s', t') = \alpha_s\left(\frac{x_t(1-c)}{\alpha_t} + cf(x_t, \gamma(t'), V)\right)$$
$$x_{t-1} = u(x_t; s', t') + \sigma(s', t')\epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

손실 함수는 L2 회귀입니다:
$$L_{\text{bit}} = \mathbb{E}_{t \sim U(0,T), \epsilon \sim \mathcal{N}(0,I)} \|f(x_t, \gamma(t'), V) - x_0\|^2$$

### 3.2 Diffusion Transformer 아키텍처

**시각 인코더 (Visual Encoder)**

$$V_{i+1} = \text{FFN}(\text{norm}(V_i + \text{MultiHead}(V_i, V_i, V_i)))$$

여기서 $V_0 = V = \{v_1, \ldots, v_K\}$는 Faster R-CNN으로부터 추출된 $K$개 객체의 2048차원 특성들입니다. 각 인코더 블록은 self-attention과 feed-forward layer로 구성되며, 모든 시간 스텝 $t$에 대해 공유됩니다. [ieeexplore.ieee](https://ieeexplore.ieee.org/document/11094818/)

**문장 디코더 (Sentence Decoder - 양방향)**

$$\tilde{h}_i = \text{norm}(h_i + \text{MultiHead}(h_i, h_i, h_i))$$
$$h_{i+1} = \text{FFN}(\text{norm}(\tilde{h}_i + \text{MultiHead}(\tilde{h}_i, \tilde{V}, \tilde{V})))$$

자동회귀 디코더와 달리 마스킹이 없는 양방향 self-attention을 사용하여, 모든 위치이 모든 다른 위치의 정보에 접근할 수 있습니다. 이는 병렬 처리를 가능하게 합니다.

**단어 확률 분포 및 비트 변환**

$$p_i = \text{softmax}(W^T h_i^{N_t})$$
$$b_i = \sum_{c=1}^{W} p_i^c B_c$$

여기서 $W \in \mathbb{R}^{512 \times W}$는 학습 가능한 가중치 행렬, $B_c \in \{0,1\}^n$은 단어 $c$의 이진 표현입니다. 확률 분포를 비트 공간에 직접 투영함으로써 기울기 정보가 비트 수준까지 전파됩니다.

**결합 손실 함수**

$$L = L_{\text{XE}} + L_{\text{bit}}$$

두 가지 손실의 결합을 통해 초기 학습의 수렴을 가속화합니다.

### 3.3 의미론적 조건화 (Semantic Conditioning)

**의미 검색 및 특성 추출**

주어진 이미지에 대해, 교차모달 검색 모델(예: CLIP 기반)을 사용하여 학습 데이터의 문장 풀에서 의미론적으로 관련된 $m$개의 문장들을 검색합니다. 이들을 토큰 시퀀스 $s_r$로 변환하고, 각 시간 스텝 $t$에서:

$$z_x = \text{FC}(\text{Concat}(x_t, \tilde{x}_0)) + \phi(\gamma(t'))$$
$$z_r = \text{FC}(s_r)$$

여기서 $\phi$는 시간 임베딩을 나타내는 다층 퍼셉트론, $\tilde{x}_0$는 현재 스텝에서의 예측된 노이즈 제거 결과입니다.

**의미 Transformer (Semantic Transformer)**

$$W_0 = [z_x, z_r], \quad W_{i+1} = \text{FFN}(\text{norm}(W_i + \text{MultiHead}(W_i, W_i, W_i)))$$

의미 Transformer는 $N_p$개의 블록을 가지며 ($N_p = 3$), $W_{i+1}$의 출력 중 처음 $|z_x|$ 부분인 $W_i^x$를 "강화된 의미 조건 잠재 상태"로 추출하여 문장 디코더의 입력으로 전달합니다. [ieeexplore.ieee](https://ieeexplore.ieee.org/document/11299447/)

### 3.4 캐스케이딩 확산 Transformer

```math
F(x_t, \gamma(t'), V) = f_M \circ f_{M-1} \circ \cdots \circ f_1(x_t, \gamma(t'), V)
```

여기서 $M$은 캐스케이딩 단계 수입니다. 각 단계 $f_i$는 위와 동일한 Diffusion Transformer이나, $i \geq 2$일 때:

```math
z_{x}=\text{FC}(\text{concat}(x_{t},\tilde{x}_{0},x_{i-1},\phi ,\gamma ,t))
```

즉, 이전 단계의 예측 $x_{i-1, 0}$ 을 추가 조건으로 포함하여 점진적 개선을 유도합니다. 추론 시 각 타임스텝에서 모든 단계의 예측을 fusion합니다.

### 3.5 안내된 자기비판 수열 학습 (Guided SCST)

**표준 자기비판 수열 학습의 문제점**

표준 SCST는 다음과 같이 정의됩니다:

$$L_R(\theta) = -\mathbb{E}_{y_{1:N_s} \sim p_\theta}[R(y_{1:N_s})]$$

여기서 $R$은 보상 함수(예: CIDEr). 그러나 비자동회귀 모델에서는 각 위치의 토큰이 독립적으로 샘플링되어, 일부 저품질 샘플들이 보상 신호를 왜곡할 수 있습니다.

**안내된 샘플링 전략**

$$\nabla_\theta L_R(\theta) \approx -\frac{1}{N_y}\sum_{j=0}^{N_y} (R(y'^{s,j}_{1:N_s}) - R(\hat{y}_{1:N_s}))\nabla_\theta \log p_\theta(y'^{s,j}_{1:N_s})$$

여기서 $\{y'^{s,j}\_{1:N_s}\}\_{j=0}^{N_y}$는 샘플 문장들의 집합이고, $j=0$일 때는 자동회귀 Transformer 교사 모델의 예측 $S_{\text{tea}}$를 고정합니다. 이를 통해 고품질 보상이 충분히 큰 가중치를 받습니다.

**적응형 교사 업데이트**

학습 중반부터, 만약 Diffusion Transformer의 예측 문장 $S'$의 CIDEr 점수가 교사 예측 $S_{\text{tea}}$를 초과하면, $S_{\text{tea}}$ 대신 $S'$를 사용합니다. 이는 모델의 점진적 독립성 증대를 가능하게 합니다.

***

## 4. 실험 결과

### 4.1 성능 비교 (COCO Karpathy 분할)

| 모델 | 최적화 방식 | B@4 | M | R | CIDEr | S |
|------|-----------|-----|---|---|-------|-----|
| Transformer (자동회귀) | XE | 34.0 | 27.6 | 56.2 | 113.3 | 21.0 |
| M²Transformer (자동회귀) | XE | - | - | - | - | - |
| M²Transformer (자동회귀) | CIDEr | 39.1 | 29.2 | 58.6 | 131.2 | 22.6 |
| SCD-Net (비자동회귀) | XE | 37.3 | 28.1 | 58.0 | 118.0 | 21.6 |
| **SCD-Net (비자동회귀)** | **CIDEr (Guided SCST)** | **39.4** | **29.2** | **59.1** | **131.6** | **23.0** |

**핵심 결과**: SCD-Net은 CIDEr 최적화 후 M²Transformer를 0.4 포인트 초과하는 131.6을 달성합니다. 이는 확산 기반 비자동회귀 모델이 처음으로 최신 자동회귀 모델을 능가했음을 의미합니다. [oarjst](https://oarjst.com/node/710)

### 4.2 온라인 평가 (공식 테스트 셋)

| 모델 | 단계 | B@4 (c5) | B@4 (c40) | CIDEr (c5) | CIDEr (c40) |
|------|------|---------|----------|-----------|-----------|
| Up-Down (AR) | 앙상블 | 36.9 | 68.5 | 117.9 | 120.5 |
| GCN-LSTM (AR) | 앙상블 | 38.7 | 69.7 | 125.3 | 126.5 |
| **SCD-Net (NAR)** | **단일** | **38.1** | **69.4** | **126.2** | **129.2** |

단일 모델로 앙상블 모델 수준의 성능을 달성했습니다.

### 4.3 절제 연구 (Ablation Study)

| 구성 요소 | Base | Semantic | RL | Cascade | B@4 | CIDEr | SPICE |
|----------|------|----------|----|---------|----|--------|--------|
| 기본 확산 | ✓ | | | | 35.9 | 114.5 | 20.7 |
| +의미 조건화 | ✓ | ✓ | | | 36.4 | 116.2 | 21.2 |
| +표준 SCST | ✓ | ✓ | SCST | | 34.6 | 120.8 | 21.5 |
| +안내된 SCST | ✓ | ✓ | GSCST | | 38.5 | 128.6 | 22.9 |
| +캐스케이딩 | ✓ | ✓ | GSCST | ✓ | **39.4** | **131.6** | **23.0** |

각 컴포넌트의 기여도:
- 의미 조건화: +1.7 CIDEr 포인트
- 표준 SCST: -5.8 CIDEr 포인트 (악화)
- 안도된 SCST: +12.4 CIDEr 포인트 (표준 SCST 대비)
- 캐스케이딩: +3.0 CIDEr 포인트

**분석**: 의미 조건화 단독으로도 기본 확산 모델보다 효과적이며, 안내된 SCST가 가장 큰 성능 향상을 제공합니다. 캐스케이딩은 추가적 개선을 가져오나 수렴 이득은 제한적입니다.

### 4.4 하이퍼파라미터 분석

**Transformer 블록 수 영향**:
- 3개 (최적): CIDEr 128.6
- 4개: CIDEr 128.6
- 5개: CIDEr 128.1 (감소)
- 6개: CIDEr 128.1 (감소)

**해석**: 5개 이상의 블록에서는 불필요한 컨텍스트 정보 마이닝으로 인해 오버파라미터화 현상이 발생합니다.

**캐스케이딩 Transformer 수 영향**:
- 1단계: CIDEr 128.6
- 2단계: CIDEr 131.6 (최적)
- 3단계: CIDEr 131.7 (증가 미미)

**해석**: 2단계 이후 수렴이 정체되며, 추가 단계의 계산 비용 대비 이득이 낮습니다.

### 4.5 정성적 분석

논문에서 제시된 예제 분석:

**예1: 단어 누락 문제**
- 기본 Transformer: "a black and white cat laying on a bed" (누락: "blanket")
- Bit Diffusion: "a cat laying under a blanket on a **blanket**" (반복)
- **SCD-Net: "a cat laying under a blanket on a bed"** (정확)

의미 조건화가 "blanket"이라는 중요한 객체를 포함하도록 유도하면서도, 중복 생성을 방지합니다.

***

## 5. 모델 일반화 성능 분석

### 5.1 크로스도메인 일반화

**COCO Karpathy (오프라인) vs 공식 테스트 셋 (온라인)**:
- 오프라인 CIDEr: 131.6
- 온라인 CIDEr (c5): 126.2 (차이: 5.4 포인트)

이 간극은 테스트 셋의 데이터 분포 변화 및 평가 메트릭 계산 방식의 차이에 기인합니다. 상대적 성능 순위는 유지됩니다.

### 5.2 비자동회귀 모델로서의 구조적 장점

**병렬성**: 모든 토큰이 동시에 생성되므로 최악의 경우 O(1) (inference step 제외)
**견고성**: 초기 토큰 오류가 후속 예측에 영향 없음
**유연성**: 마스킹을 통해 조건부 생성 가능

### 5.3 교사 학습 효과

표준 SCST 적용 시 성능이 악화되는 현상은 비자동회귀 특성 때문입니다. 각 위치의 독립적 샘플링은 보상이 개별 단어에 균등하게 배분되어야 함을 의미하는데, 이는 단어 반복 문제와 직결됩니다. Guided SCST는 고품질 샘플의 보상을 우선적으로 고려함으로써 이를 해결합니다.

***

## 6. 한계 및 개선 기회

### 6.1 기술적 한계

1. **캐스케이딩 계산 오버헤드**: M=2일 때 추론은 기본 모델의 약 2배 연산량 필요. M>2에서 수렴 이득이 미미.

2. **교사 모델 의존성**: Guided SCST의 효과는 교사 모델의 품질에 크게 의존. 약한 교사는 부정적 영향 가능.

3. **의미 검색 품질**: 교차모달 검색 모델의 정확성이 성능에 직결. 도메인 변화 시 성능 저하 가능.

4. **초기화 의존성**: 이진 비트 표현의 초기 확률 분포가 학습에 영향. 특정 초기화에서 불안정성 가능.

### 6.2 미해결 문제

1. **다국어 일반화**: 실험은 영어에 국한. 언어 특성 차이에 따른 성능 변화 미지.

2. **아웃-오브-도메인 성능**: COCO 외 데이터셋(Flickr30K 등)에 대한 미세 조정 없이의 성능은 미측정.

3. **제로샷 학습**: 학습 데이터 없는 새로운 도메인에의 직접 적용 불가.

***

## 7. 2020년 이후 최신 연구와의 비교

### 7.1 진화 경로

```
2022년 8월: Bit Diffusion
  └─ 이진 비트 변환 + Self-Conditioning
    └─ 성능: CIDEr 115.0 (기본 자동회귀보다 약함)
    
2022년 12월: SCD-Net ← 본 논문
  ├─ 의미 조건화 추가
  ├─ Guided SCST 도입
  └─ 성능: CIDEr 131.6 (자동회귀 초과) ★ BREAKTHROUGH
  
2024년 6월: LaDiC (Wang et al.)
  ├─ Split BERT 기반 전용 잠재 공간
  ├─ Back&Refine 기법
  └─ 성능: CIDEr 126.2 (효율성 중심, SCD-Net보다 낮지만 빠름)
  
2025년 7월: POSCD-Net (Liu et al.)
  ├─ 구문론적(POS) 조건화로 확장
  ├─ 제어 가능한 생성
  └─ 성능: 미발표 (제어성 우선)
```

### 7.2 기술적 비교

| 특성 | Bit Diffusion | SCD-Net | LaDiC | POSCD-Net |
|------|---|---|---|---|
| **기본 기법** | 이진 비트 | 의미 조건화 | 잠재 공간 | POS 조건화 |
| **캐스케이딩** | 없음 | 있음 (M=2) | 있음 (모듈식) | 이중 구조 |
| **강화학습** | 없음 | Guided SCST | 없음 | Classifier-Free Guidance |
| **성능 (CIDEr)** | 115.0 | 131.6 | 126.2 | 미발표 |
| **추론 효율** | 빠름 | 중간 | 매우 빠름 | 중간 |
| **제어성** | 없음 | 없음 | 낮음 | 높음 |

**핵심 관찰**: SCD-Net은 성능에서 여전히 최고 수준(131.6)이나, LaDiC는 더 빠른 수렴(5스텝)과 효율성을 우선하는 트렌드 시작. 최신 모델들(POSCD-Net, MirrorDiff 등)은 성능보다 제어성, 도메인 특화, 효율성을 추구.

### 7.3 SCD-Net의 학계 영향

**인용 수**: 159회 (2022년 논문 기준으로 매우 높음)

**직접 영향**:
- POSCD-Net: SCD-Net의 의미 조건화 개념을 구문 조건화로 확장
- MirrorDiff: SCD-Net의 의미 검증 개념을 시각 재생성으로 개선
- 후속 연구들의 기준선 역할

**간접 영향**:
- 확산 모델의 캡셔닝 적용 가능성 입증
- 비자동회귀 모델에 강화학습 통합의 선례
- 조건부 확산의 일반적 패러다임 제시

***

## 8. 향후 연구 방향

### 8.1 단기 개선 (1-2년)

1. **효율성 개선**
   - 적응형 캐스케이딩: 이미지 난이도에 따라 단계 수 동적 결정
   - 조기 종료: BLEU 점수가 수렴하면 추가 단계 생략

2. **교사 모델 제거**
   - 자가-증류(self-distillation): 초기 모델이 후속 단계의 교사 역할
   - 메모리 뱅크 기반 고품질 샘플 유지

3. **도메인 특화**
   - 의료 이미지 캡셔닝
   - 위성 이미지 캡셔닝 (지오-공간 정보 통합)

### 8.2 중기 연구 방향 (2-3년)

1. **다중 조건화**
   - SCD-Net: 단일 의미 조건화
   - 향후: 스타일(formal/casual), 톤(positive/negative), 길이 등 다중 조건

2. **크로스도메인 일반화**
   - COCO 사전학습 → Flickr30K/Nocaps 적응
   - Domain-agnostic semantic retrieval

3. **다국어 통합**
   - 다국어 교사 모델 학습
   - 크로스링귀스틱 의미 조건화

### 8.3 장기 연구 전망 (3년+)

1. **통합 멀티모달 확산**
   - 이미지-텍스트-비디오 통합 모델
   - 최근 DAC (Diffusion-based Audio Captioning) 등 멀티모달 확산 부상

2. **제로샷 적응**
   - Few-shot 학습으로 새 도메인 빠른 적응
   - Prompt 기반 조건화

3. **해석 가능성**
   - 의미 조건화 과정의 시각화
   - 어떤 의미 정보가 생성에 기여하는지 분석

***

## 9. 결론 및 평가

### 9.1 과학적 기여의 평가

**원내 혁신성**: SCD-Net은 확산 모델이 이미지 캡셔닝에서 자동회귀 모델을 능가할 수 있음을 처음으로 입증했습니다. 이는 세 가지 핵심 혁신의 결합입니다:

1. **의미론적 조건화**: 비자동회귀 모델의 단어 누락 문제를 구조적으로 해결
2. **Guided SCST**: 문장 수준 최적화를 비자동회귀 모델에 적용 가능하게 함
3. **캐스케이딩**: 점진적 개선을 통해 최종 품질 향상

**성능의 완전성**: CIDEr 131.6은 당시 최신 자동회귀 모델(M²Transformer, CIDEr 131.2)과 통계적으로 유의미한 수준에서 경쟁합니다.

**후속 영향**: 2024-2025년 최신 연구들이 SCD-Net의 개념을 확장하면서, 이 논문의 핵심 아이디어(의미 조건화, 강화학습 통합)는 확산 기반 생성 모델의 표준 요소가 되었습니다.

### 9.2 실무적 의의

1. **비자동회귀 모델의 재평가**: 병렬 처리의 속도 이점과 높은 품질을 동시에 달성 가능함 증명
2. **강화학습의 새로운 적용**: SCST를 비자동회귀 설정에 효과적으로 적용하는 방법론
3. **다중 디코딩 전략의 통일**: 캐스케이딩을 통한 일관된 품질 향상 프레임워크

### 9.3 한계 인식

- **캐스케이딩의 계산 비용**: 실제 배포 환경에서는 교사 모델 없는 경량 버전 필요
- **도메인 의존성**: 의미 검색 모델의 질에 따른 성능 변동
- **미세 조정 요구**: 새로운 도메인에 대한 적응 필요

### 9.4 학계의 평가

본 논문은 CVPR 2023에 수록되었으며, 3년 내 159회 인용을 기록했습니다. 이는:
- 이미지 캡셔닝 커뮤니티의 적극적 반응
- 확산 모델 연구자들의 관심
- 방법론의 재현성과 강건성 입증

***

## 최종 평가

**SCD-Net은 확산 모델 기반 이미지 캡셔닝의 획기적 작업입니다**. 의미론적 조건화, 안내된 자기비판 수열 학습, 캐스케이딩 구조의 삼중 혁신을 통해, 비자동회귀 방식으로 처음 자동회귀 모델을 능가하는 성과를 달성했습니다. 

다만 교사 모델 의존성, 계산 오버헤드, 도메인 제한성은 실제 배포 시 고려해야 할 사항입니다. 이후 연구들(LaDiC, POSCD-Net)은 이러한 한계를 인식하고 효율성과 제어성을 우선하는 방향으로 발전했으나, **성능 측면에서 SCD-Net은 여전히 확산 기반 이미지 캡셔닝의 최고봉**으로 평가됩니다.

***

## 참고 문헌
<span style="display:none">[^1_10][^1_11][^1_12][^1_13][^1_14][^1_15][^1_16][^1_17][^1_18][^1_19][^1_20][^1_21][^1_22][^1_23][^1_24][^1_25][^1_26][^1_27][^1_28][^1_29][^1_30][^1_31][^1_32][^1_33][^1_34][^1_35][^1_36][^1_37][^1_38][^1_39][^1_40][^1_41][^1_42][^1_43][^1_44][^1_45][^1_46][^1_47][^1_48][^1_49][^1_50][^1_51][^1_52][^1_53][^1_54][^1_55][^1_56][^1_57][^1_58][^1_59][^1_60][^1_61][^1_62][^1_63][^1_64][^1_65][^1_66][^1_67][^1_68][^1_69][^1_70][^1_71][^1_72][^1_73][^1_74][^1_75][^1_76][^1_77][^1_78][^1_79][^1_80][^1_81][^1_9]</span>

<div align="center">⁂</div>

[^1_1]: 2212.03099v1.pdf

[^1_2]: https://dl.acm.org/doi/10.1145/3731715.3733389

[^1_3]: https://arxiv.org/abs/2506.03067

[^1_4]: https://ojs.aaai.org/index.php/AAAI/article/view/32221

[^1_5]: https://dl.acm.org/doi/10.1145/3748653

[^1_6]: https://ieeexplore.ieee.org/document/11094818/

[^1_7]: https://ieeexplore.ieee.org/document/11299447/

[^1_8]: https://oarjst.com/node/710

[^1_9]: https://ieeexplore.ieee.org/document/11242797/

[^1_10]: https://dl.acm.org/doi/10.1145/3746027.3755156

[^1_11]: https://ieeexplore.ieee.org/document/11297974/

[^1_12]: https://arxiv.org/pdf/2305.12144.pdf

[^1_13]: https://dl.acm.org/doi/pdf/10.1145/3607827.3616839

[^1_14]: https://arxiv.org/pdf/2305.01855.pdf

[^1_15]: https://arxiv.org/pdf/2311.14920.pdf

[^1_16]: https://arxiv.org/html/2412.01115

[^1_17]: https://arxiv.org/abs/2212.03099

[^1_18]: https://arxiv.org/pdf/2210.04559.pdf

[^1_19]: http://arxiv.org/pdf/2404.10763.pdf

[^1_20]: https://www.sciencedirect.com/science/article/abs/pii/S0952197624014465

[^1_21]: https://openaccess.thecvf.com/content/CVPR2023/papers/Luo_Semantic-Conditional_Diffusion_Networks_for_Image_Captioning_CVPR_2023_paper.pdf

[^1_22]: https://aclanthology.org/2024.lrec-main.1214.pdf

[^1_23]: https://openaccess.thecvf.com/content/CVPR2025/papers/Lee_Diffusion_Bridge_Leveraging_Diffusion_Model_to_Reduce_the_Modality_Gap_CVPR_2025_paper.pdf

[^1_24]: https://www.emergentmind.com/topics/diffusion-vision-language-models-dvlms

[^1_25]: https://www.ijcai.org/proceedings/2020/0107.pdf

[^1_26]: https://www.fujipress.jp/jaciii/jc/jacii002900061417/

[^1_27]: https://kimjy99.github.io/논문리뷰/scdnet/

[^1_28]: https://arxiv.org/pdf/2211.16769.pdf

[^1_29]: https://dl.acm.org/doi/10.1145/3607827.3616839

[^1_30]: https://aclanthology.org/2024.lrec-main.1214/

[^1_31]: https://arxiv.org/pdf/2503.24379.pdf

[^1_32]: https://liner.com/review/semanticconditional-diffusion-networks-for-image-captioning

[^1_33]: https://ieeexplore.ieee.org/document/11036655/

[^1_34]: https://arxiv.org/pdf/2510.09586.pdf

[^1_35]: https://arxiv.org/pdf/2506.21900.pdf

[^1_36]: https://arxiv.org/pdf/2506.24044.pdf

[^1_37]: https://arxiv.org/html/2405.14857v1

[^1_38]: https://pubmed.ncbi.nlm.nih.gov/37930907/

[^1_39]: https://arxiv.org/pdf/2509.04162.pdf

[^1_40]: https://arxiv.org/html/2503.19012v1

[^1_41]: https://openaccess.thecvf.com/content/CVPR2022/papers/Fei_DeeCap_Dynamic_Early_Exiting_for_Efficient_Image_Captioning_CVPR_2022_paper.pdf

[^1_42]: https://arxiv.org/html/2506.21900v1

[^1_43]: https://arxiv.org/pdf/2212.03099.pdf

[^1_44]: https://arxiv.org/pdf/2512.10038.pdf

[^1_45]: https://0mini.tistory.com/181

[^1_46]: https://arxiv.org/abs/2208.04202

[^1_47]: https://www.ssrn.com/abstract=4120043

[^1_48]: https://link.springer.com/10.1007/s00371-022-02517-y

[^1_49]: https://link.springer.com/10.1007/s00371-022-02645-5

[^1_50]: https://www.mdpi.com/2227-7390/10/17/3038

[^1_51]: https://ieeexplore.ieee.org/document/10205410/

[^1_52]: https://arxiv.org/abs/2204.06125

[^1_53]: https://www.mdpi.com/1099-4300/24/2/287

[^1_54]: https://opg.optica.org/abstract.cfm?URI=oe-30-3-4249

[^1_55]: https://ieeexplore.ieee.org/document/10484137/

[^1_56]: https://arxiv.org/abs/2211.11694

[^1_57]: https://arxiv.org/html/2501.00437v1

[^1_58]: https://aclanthology.org/2024.naacl-long.373/

[^1_59]: https://openaccess.thecvf.com/content/CVPR2024/papers/Qu_A_Conditional_Denoising_Diffusion_Probabilistic_Model_for_Point_Cloud_Upsampling_CVPR_2024_paper.pdf

[^1_60]: https://openreview.net/forum?id=ZDI2lAG0RL

[^1_61]: http://theaitalks.org/talks/2022/0929/

[^1_62]: https://aclanthology.org/2024.naacl-long.373.pdf

[^1_63]: https://dl.acm.org/doi/abs/10.1145/3748653

[^1_64]: https://www.semanticscholar.org/paper/Analog-Bits:-Generating-Discrete-Data-using-Models-Chen-Zhang/b64537bdf7a103aa01972ba06ea24a9c08f7cd74

[^1_65]: https://openreview.net/attachment?id=ZDI2lAG0RL\&name=pdf

[^1_66]: https://kimjy99.github.io/논문리뷰/controlnet/

[^1_67]: https://openaccess.thecvf.com/content/CVPR2022/papers/Wu_DIFNet_Boosting_Visual_Information_Flow_for_Image_Captioning_CVPR_2022_paper.pdf

[^1_68]: https://arxiv.org/abs/2404.10763

[^1_69]: https://openaccess.thecvf.com/content/CVPR2025/papers/Cohen_Conditional_Balance_Improving_Multi-Conditioning_Trade-Offs_in_Image_Generation_CVPR_2025_paper.pdf

[^1_70]: https://arxiv.org/html/2311.14920v1

[^1_71]: https://www.semanticscholar.org/paper/LaDiC:-Are-Diffusion-Models-Really-Inferior-to-for-Wang-Ren/daf1ebccda482ede25044faad26f4f479ffe89e3

[^1_72]: https://arxiv.org/html/2504.10240v1

[^1_73]: https://arxiv.org/pdf/2505.22613.pdf

[^1_74]: https://pdfs.semanticscholar.org/9fe3/3c91d41faeed7b2f888e7945c494dac9331a.pdf

[^1_75]: https://www.arxiv.org/pdf/2409.09401.pdf

[^1_76]: https://www.semanticscholar.org/paper/Syntactic-Conditional-Diffusion-Networks-for-Image-Liu-Yang/ab7f41b054240e0f7df08586c808823c8a342316

[^1_77]: https://arxiv.org/pdf/2208.04202.pdf

[^1_78]: https://www.semanticscholar.org/paper/Diverse-Image-Captioning-via-Conditional-and-Dual-Xu-Liu/4baeba8a654bff62888ada10bf335ed1db7a7b62

[^1_79]: https://arxiv.org/html/2404.10763v1

[^1_80]: https://arxiv.org/html/2409.09401v2

[^1_81]: https://arxiv.org/html/2503.11482v1
