# Explicit Visual Prompting for Low-Level Structure Segmentations

---

## 1. Executive Summary (10문장 이내)

본 논문은 위조 탐지(Forgery Detection), 그림자 탐지(Shadow Detection), 초점 흐림 탐지(Defocus Blur Detection), 위장 객체 탐지(Camouflaged Object Detection)라는 네 가지 저수준 구조 분할(Low-Level Structure Segmentation) 과제를 하나의 통합 프레임워크로 해결하고자 한다.  
기존에는 각 과제마다 도메인 특화 모델을 별도 설계하는 방식이 지배적이었으나, 이는 대규모 데이터셋 부재와 모델 저장 비효율이라는 문제를 수반했다.  
저자들은 NLP의 사전학습(Pre-training) 후 프롬프트 튜닝(Prompt Tuning) 패러다임에서 영감을 받아, 동결된(Frozen) 대형 비전 트랜스포머를 소수의 학습 가능한 파라미터로 적응시키는 **Explicit Visual Prompting (EVP)**을 제안한다.  
EVP의 핵심 아이디어는 각 이미지 자체의 명시적 시각 콘텐츠, 즉 동결 패치 임베딩(Frozen Patch Embedding) 특징과 고주파 성분(High-Frequency Components, HFC)을 프롬프트로 활용하는 것이다.  
이는 데이터셋 전체에 걸쳐 암묵적으로 공유되는 임베딩을 사용하는 기존 VPT(Visual Prompt Tuning)와 근본적으로 구별된다.  
백본은 완전히 동결되고, 각 태스크당 전체 파라미터의 5.7%에 해당하는 추가 학습 가능 파라미터만으로 태스크 적응이 이루어진다.  
9개 데이터셋, 4개 태스크에 걸친 실험에서 EVP는 동등한 파라미터 수 조건 하에 VPT, AdaptFormer 등 다른 파라미터 효율적 튜닝(Parameter-Efficient Tuning) 방법을 유의미하게 상회한다.  
또한 태스크별 전문 솔루션과 비교해도 5개 데이터셋에서 최고 성능(SOTA)을 달성한다.  
본 연구는 시각 프롬프팅 분야에서 명시적 이미지 콘텐츠 기반 프롬프트 설계의 유효성을 처음으로 체계적으로 검증한 연구로서 의의를 갖는다.

### 1-1. 연구의 목적과 필요성

| 구분 | 내용 |
|------|------|
| **배경** | 이미지 편집 기술의 발전으로 사실적 위조 이미지 생성이 용이해졌으며, 위조 탐지·그림자 탐지·초점 흐림 탐지·위장 객체 탐지는 모두 저수준 구조적 단서에 의존함 (p.1) |
| **문제점** | 각 탐지 과제가 도메인 특화 아키텍처로 개별 해결되어 왔으며, 대규모 주석 데이터 부족이 성능 향상의 병목 (p.1) |
| **필요성** | 하나의 통합 모델로 여러 저수준 분할 과제를 효율적으로 처리하고, 사전학습 지식을 최소 파라미터로 활용하는 방법론 필요 |
| **목적** | 동결된 비전 트랜스포머를 기반으로 명시적 시각 프롬프팅을 통해 여러 저수준 분할 과제를 단일 프레임워크로 통합 해결 |

> 💡 **저수준 구조(Low-Level Structure)**: 색상, 질감, 주파수, 노이즈 등 픽셀 수준의 물리적 특성을 의미함. 객체 의미(Semantics)보다 이미지의 물리적 속성에 집중.

---

## 2. 핵심 주장과 근거 표

| 핵심 주장 | 근거 | 위치 |
|-----------|------|------|
| 단일 통합 모델로 4가지 저수준 분할 과제 해결 가능 | 9개 데이터셋 실험에서 태스크별 SOTA와 동등 또는 초과 성능 | Table 2-5, p.5-6 |
| 명시적 시각 프롬프트(패치 임베딩 + HFC)가 암묵적 프롬프트보다 효과적 | 동일 파라미터 수에서 VPT, AdaptFormer 대비 우수한 성능 | Table 6, p.5 |
| 고주파 성분(HFC)이 저수준 과제에 핵심적인 프롬프트 역할 | HFC 제거 시 성능 하락, HFC가 가우시안 필터·원본 이미지 등 대안 대비 일관된 우수성 | Table 7, Table B11, p.7, p.14 |
| 패치 임베딩 특징 튜닝이 분포 이동(Distribution Shift) 보정에 효과적 | Fpe 제거 시 성능 하락, 특히 그림자·위조 탐지에서 두드러짐 | Table 7, p.7 |
| 파라미터 효율성: 5.7%의 추가 파라미터로 완전 파인튜닝에 근접한 성능 | Full-tuning 64M vs EVP(r=4) 3.70M, 3/4 데이터셋에서 Full-tuning 초과 | Table 6, p.7 |
| EVP는 SegFormer뿐만 아니라 ViT 등 다른 아키텍처에도 일반화 가능 | SETR(plain ViT) 기반 실험에서도 타 튜닝 방법 대비 우수 | Table 10, p.8 |
| 모든 스테이지에 프롬프팅을 적용할수록 성능 향상 | Stage1→Stage1,2,3,4 순차 추가 시 단조 성능 향상 | Table 8, p.7 |

### 2-1. 상세 설명

#### 해결하고자 하는 문제

1. **도메인 분절 문제**: 위조·그림자·흐림·위장 탐지가 각각 별개 모델로 해결되어 지식 공유 불가
2. **데이터 부족 문제**: 각 과제의 주석 데이터가 소규모이어서 완전 파인튜닝 시 과적합 위험
3. **기존 시각 프롬프팅의 한계**: VPT 등 기존 방법은 데이터셋 수준의 암묵적(implicit) 임베딩으로, 이미지 개별 콘텐츠를 활용하지 못함 (p.2)

---

#### 제안하는 방법 (수식 포함)

**① 고주파 성분(HFC) 추출** (p.3, Figure 2)

이미지 $I \in \mathbb{R}^{H \times W}$에 대해:

$$z = \text{fft}(I)$$

> 💡 **FFT (Fast Fourier Transform)**: 이미지를 공간 도메인에서 주파수 도메인으로 변환하는 알고리즘. 저주파 성분은 이미지의 큰 구조(배경, 전체 밝기)를, 고주파 성분은 엣지·노이즈·미세 텍스처를 담음.

저주파 성분을 스펙트럼 중앙 $(\frac{H}{2}, \frac{W}{2})$으로 이동시킨 후, 마스크 비율 $\tau$에 따라 HFC 마스크 $\mathbf{M}_h \in \{0,1\}^{H \times W}$ 생성:

$$\mathbf{M}_h^{i,j}(\tau) = \begin{cases} 0, & \dfrac{4\left|(i - \frac{H}{2})(j - \frac{W}{2})\right|}{HW} \leq \tau \\ 1, & \text{otherwise} \end{cases} $$

- $i, j$: 스펙트럼 내 픽셀 좌표
- $\tau$: 마스크 처리되는 영역의 표면 비율 (저주파 영역 제거 비율)
- $\mathbf{M}_h^{i,j} = 0$: 저주파 영역(중심부) → 제거
- $\mathbf{M}_h^{i,j} = 1$: 고주파 영역 → 보존

$$I_{hfc} = \text{ifft}(z \cdot \mathbf{M}_h(\tau)) $$

마찬가지로 LFC 마스크:

$$\mathbf{M}_l^{i,j}(\tau) = \begin{cases} 0, & \dfrac{HW - 4\left|(i - \frac{H}{2})(j - \frac{W}{2})\right|}{HW} \leq \tau \\ 1, & \text{otherwise} \end{cases} $$

$$I_{lfc} = \text{ifft}(z \cdot \mathbf{M}_l(\tau)) $$

---

**② 패치 임베딩 튜닝 (Patch Embedding Tune)** (p.4, Eq.5)

동결 SegFormer의 패치 임베딩 출력을 $I^p$라 할 때, 학습 가능한 선형 레이어 $\mathbf{L}_{pe}$로 차원 축소:

$$F_{pe} = \mathbf{L}_{pe}(I^p), \quad c = \frac{C_{seg}}{r} $$

- $I^p$: 패치 임베딩 출력 특징
- $C_{seg}$: SegFormer 원래 임베딩 차원
- $r$: 스케일 팩터(파라미터 수 제어, 클수록 파라미터 감소)
- $c$: 축소된 차원
- $F_{pe} \in \mathbb{R}^c$: 패치 임베딩 튜닝 특징

> 💡 **분포 이동(Distribution Shift)**: 사전학습 데이터(ImageNet)와 타겟 데이터(저수준 탐지 데이터)의 통계적 분포 차이. 패치 임베딩 튜닝은 이 차이를 좁히는 역할을 함.

---

**③ HFC 튜닝 (HFC Tune)** (p.4, Eq.6)

$I_{hfc}$를 SegFormer 동일 방식으로 패치 $I^p_{hfc} \in \mathbb{R}^C$ ($C = h \times w \times 3$)로 분할 후 선형 투영:

$$F_{hfc} = \mathbf{L}_{hfc}(I^p_{hfc}) $$

- $I^p_{hfc}$: HFC 이미지의 패치
- $\mathbf{L}_{hfc}$: 학습 가능한 선형 레이어
- $F_{hfc} \in \mathbb{R}^c$: HFC 튜닝 특징

---

**④ Adaptor** (p.4, Eq.7)

$i$번째 트랜스포머 레이어에 삽입되는 Adaptor:

$$P^i = \text{MLP}_{up}\left(\text{GELU}\left(\text{MLP}^i_{tune}(F_{pe} + F_{hfc})\right)\right) $$

- $F_{pe} + F_{hfc}$: 패치 임베딩 튜닝 특징과 HFC 튜닝 특징의 합산
- $\text{MLP}^i_{tune}$: 각 Adaptor별 비공유(unshared) 선형 레이어 → 레이어마다 다른 프롬프트 생성
- $\text{GELU}(\cdot)$: Gaussian Error Linear Unit 활성화 함수
- $\text{MLP}_{up}$: 모든 Adaptor 공유 업프로젝션 레이어 → 트랜스포머 특징 차원으로 복원
- $P^i$: $i$번째 트랜스포머 레이어에 추가되는 프롬프트

> 💡 **GELU (Gaussian Error Linear Unit)**: $\text{GELU}(x) = x \cdot \Phi(x)$로 정의되는 활성화 함수. ReLU보다 부드러운 비선형성을 가져 트랜스포머에서 널리 사용됨.

> 💡 **Adaptor**: 대형 사전학습 모델의 각 레이어 사이에 삽입되는 소형 병목(bottleneck) 모듈. 원래 파라미터는 동결하고, Adaptor의 파라미터만 학습하여 효율적 적응 달성.

---

#### 모델 구조 (Figure 3, p.4)

```
입력 이미지
    ├─ [동결] Patch Embedding → Embedding Tune (학습 가능 Lpe) → Fpe
    └─ HFC Extraction → HFC Tune (학습 가능 Lhfc) → Fhfc
         ↓
    Adaptor_1: MLP^1_tune(Fpe + Fhfc) → GELU → MLP_up(공유) → P^1
         ↓ (+P^1)
    [동결] Transformer Layer 1
         ↓
    Adaptor_2 → P^2
         ↓ (+P^2)
    [동결] Transformer Layer 2
         ⋮ (Stage 1~4)
         ↓
    [학습 가능] Decoder → 분할 마스크 출력
```

- **동결**: 백본 전체 트랜스포머 블록, 원래 패치 임베딩
- **학습 가능**: $\mathbf{L}\_{pe}$, $\mathbf{L}\_{hfc}$, $\text{MLP}^i_{tune}$ (각 레이어별), $\text{MLP}_{up}$ (공유), 디코더

---

#### 성능 향상

| 비교 대상 | 주요 성능 향상 결과 | 위치 |
|-----------|-------------------|------|
| VPT-Deep vs EVP(r=16) (동등 파라미터) | Shadow BER: 1.73→1.67, Forgery F1: .588→.602, AUC: .847→.857 | Table 6 |
| AdaptFormer vs EVP(r=16) | Shadow BER: 1.85→1.67, Defocus Fβ: .912→.924 | Table 6 |
| Full-tuning(64M) vs EVP(r=4, 3.70M) | Defocus Fβ: .935→.928, Shadow BER: **2.42→1.35** (대폭 개선) | Table 6 |
| 태스크별 SOTA vs EVP | 5/9 데이터셋에서 SOTA 달성 | Table 2-5 |

#### 한계

1. **그림자 탐지 SBU 데이터셋**: BER 4.31로 FDRNet(3.04)에 비해 성능 열위 (Table 3, p.5)
2. **위장 객체 탐지 CHAMELEON**: FBNet(.888)에 비해 $S_\alpha$ .871로 소폭 열위 (Table 5)
3. **SETR 기반 실험**: SegFormer 기반 대비 전반적 절대 성능이 낮음 (Table 10, p.8)
4. **태스크 범위 제한**: 저수준 구조 분할 4가지로 한정, 고수준 의미 분할이나 인스턴스 분할 등으로의 확장 미검증
5. **단일 GPU 실험**: NVIDIA Titan V 12G 단일 GPU 기반으로 대규모 분산 학습 효율 미검증

---

## 3. 각 주장에 페이지/Figure/Table 번호 표시

| 주장 | 근거 위치 |
|------|-----------|
| EVP 통합 프레임워크 개요 | Figure 1 (p.1), Abstract (p.1) |
| HFC 추출 방법 (Eq. 1-4) | Figure 2 (p.3), Section 3.1 (p.3) |
| EVP 아키텍처 (Eq. 5-7) | Figure 3 (p.4), Section 3.2 (p.4) |
| 데이터셋 요약 | Table 1 (p.4) |
| 초점 흐림 탐지 SOTA 비교 | Table 2 (p.5) |
| 그림자 탐지 SOTA 비교 | Table 3 (p.5) |
| 위조 탐지 SOTA 비교 | Table 4 (p.5) |
| 위장 객체 탐지 SOTA 비교 | Table 5 (p.5) |
| 효율적 튜닝 방법 비교 | Table 6 (p.5) |
| 시각적 결과 비교 | Figure 4 (p.6) |
| 아키텍처 Ablation | Table 7 (p.6), Figure 5 (p.8) |
| 튜닝 스테이지 Ablation | Table 8 (p.7) |
| 스케일 팩터 r Ablation | Table 9 (p.7) |
| ViT(SETR) 일반화 실험 | Table 10 (p.8) |
| HFC 대안 비교 | Table B11 (p.14) |
| HFC vs LFC, 마스크 비율 | Table B12 (p.14) |

---

## 4. 저자 보고 결과 vs. 해석 분리

### 연구 주제
- **저자 직접 보고**: "We consider the generic problem of detecting low-level structures in images" (p.1 Abstract)
- **해석**: 저자들은 기존에 분절되어 있던 4가지 저수준 분할 과제를 시각 프롬프팅 관점에서 재해석함으로써, NLP의 파운데이션 모델 패러다임을 컴퓨터 비전의 저수준 과제에 성공적으로 이식했다고 볼 수 있음.

### 방법

**저자 직접 보고:**
- "our key insight is to enforce the tunable parameters focusing on the explicit visual content from each individual image" (p.1)
- HFC를 활용하는 이유: "the pre-trained visual recognition model is learned to be invariant to these features via data augmentation" (p.2)

**해석:** 사전학습 모델이 데이터 증강 과정에서 HFC에 불변(invariant)하도록 학습되므로, HFC 정보가 모델 내부에 제대로 표현되지 않는다. EVP는 이 "표현 공백"을 명시적으로 보완하는 설계 철학을 가짐. 이는 단순한 추가 특징 입력이 아니라, 사전학습 모델의 인식론적 맹점(blind spot)을 타겟으로 한 설계임.

### 결과

**저자 직접 보고 (Table 6, p.5):**
- EVP(r=4): Shadow BER=1.35, Defocus $F_\beta$=.928, Forgery F1=.636/AUC=.862, Camouflaged $S_\alpha$=.846
- Full-tuning(64M): Shadow BER=2.42, Defocus $F_\beta$=.935
- 저자 언급: "EVP (r=4) outperforms full-tuning on 3 of 4 datasets"

**해석:**
- Shadow 탐지에서 EVP가 Full-tuning 대비 BER 2.42→1.35로 대폭 우수한 것은 주목할 만함. 이는 그림자 탐지가 고주파 경계 정보에 특히 민감하여 HFC 프롬프팅의 효과가 극대화된 것으로 해석됨.
- 반면 Defocus Blur 탐지에서는 Full-tuning(.935)이 EVP(.928)보다 약간 우수한데, 이는 흐림 탐지 자체가 전역적 특징(Global Feature) 변화에 더 의존하므로 전체 파라미터 업데이트의 이점이 남아있는 것으로 보임.
- SBU 그림자 탐지에서 EVP(BER=4.31)가 FDRNet(3.04)에 뒤지는 점은, FDRNet이 밝기 조정 데이터 증강 등 도메인 특화 전략을 활용한 반면 EVP는 범용 방법론을 사용하기 때문으로 해석됨.

---

## 5. 통계적으로 취약한 부분과 비교 불가능한 수치

> ⚠️ **통계적으로 취약한 부분**

| 항목 | 문제점 |
|------|--------|
| **단일 GPU 단일 실험** | 표준편차, 신뢰구간, 다중 실험 반복(multiple runs) 결과가 보고되지 않음. 우연에 의한 성능 변동 가능성 배제 불가 |
| **CHAMELEON 데이터셋** | 테스트 셋이 76장에 불과 (Table 1, p.4). 극소 샘플로 통계적 유의성 낮음 |
| **DUT 데이터셋** | 학습 셋 없이 CUHK로만 학습 후 테스트. 도메인 갭 효과가 성능에 미치는 영향 분리 불가 |
| **IMD20 데이터셋** | 학습 셋 없음. 타 방법과 학습 데이터 조건이 상이할 수 있음 |
| **Mask ratio $\tau$ 민감도** | 최적 $\tau=25\%$가 모든 태스크에 동일하게 적용되나, 태스크별 최적화 결과 미제시 |

> ⚠️ **비교 불가능한 수치**

| 항목 | 문제점 |
|------|--------|
| **위조 탐지 Table 4** | ManTraNet, SPAN, PSCCNet, ObjectFormer는 추가 학습 데이터 사용 (Appendix A, p.13). EVP는 표준 데이터만 사용. 공정 비교 불가 |
| **그림자 탐지 Table 3** | MTMT는 반지도학습(semi-supervised, 미주석 데이터 추가 활용). EVP와 학습 조건 상이 |
| **Table 4 IMD20 F1** | ManTraNet, SPAN, PSCCNet, ObjectFormer의 IMD20 F1 값이 "-"로 미보고. 일부 지표 직접 비교 불가 |
| **Table 5 JCOD** | JCOD의 $F^w_\beta$ 값 미보고("-"). 완전한 지표 비교 불가 |
| **SegFormer vs SETR 비교** | Table 6(SegFormer)과 Table 10(SETR)의 절대 성능 수치는 백본 차이로 인해 직접 비교 불가 |

---

## 6. 문서가 답하지 않는 질문

1. **추론 속도(Inference Speed)**: EVP의 추가 모듈(HFC 추출, Adaptor)이 실시간 추론에 미치는 지연 시간(Latency) 영향이 미보고.

2. **소수샷(Few-Shot) 성능**: 프롬프팅 방법의 핵심 이점으로 언급된 few-shot 일반화 성능에 대한 정량적 실험 부재.

3. **HFC의 최적 주파수 대역**: $\tau=25\%$가 최적임을 보이나, 각 태스크별로 최적 $\tau$가 다를 수 있는지에 대한 태스크별 분석 부재.

4. **다른 사전학습 데이터셋 효과**: ImageNet-1k 외 ImageNet-21k, CLIP 등 더 큰 사전학습 데이터로 백본을 바꿨을 때의 성능 변화 미검증.

5. **교차 태스크 전이(Cross-Task Transfer)**: 하나의 태스크로 학습된 EVP 모듈이 다른 태스크에 얼마나 전이 가능한지 미검증.

6. **비디오(Video) 도메인 적용 가능성**: 정적 이미지에만 실험이 제한되어, 비디오의 시간적 HFC 활용 가능성 미언급.

7. **적대적 공격(Adversarial Attack) 강건성**: 위조 탐지 맥락에서 EVP가 적대적으로 생성된 위조 이미지에 얼마나 강건한지 미검증.

8. **RGB 외 다른 입력 형식**: 깊이(Depth), 적외선(IR) 등 다채널 입력에 대한 HFC 처리 전략 미제시.

---

## 7. 가장 중요한 그림 5개 해석

### Figure 1 (p.1) — EVP 개요

**해석:** 상단은 4가지 저수준 분할 과제(위장 객체, 위조, 그림자, 초점 흐림)가 하나의 동결된 사전학습 트랜스포머 백본에 서로 다른 프롬프팅 모듈을 붙여 처리됨을 보여준다. 하단의 상세 구조도에서 핵심은 **백본은 완전 동결(Frozen, 눈송이 아이콘)**되고, **Embedding Tune과 HFC Tune만 학습 가능(Tunable, 불꽃 아이콘)**이라는 점이다. 이 그림은 EVP의 설계 철학인 "동결 백본 + 명시적 콘텐츠 기반 프롬프트"를 직관적으로 전달하며, 논문의 핵심 아이디어를 한눈에 보여주는 가장 중요한 그림이다.

---

### Figure 2 (p.3) — HFC 추출 과정

**해석:** 입력 이미지 → FFT → 스펙트럼(중앙이 저주파, 주변이 고주파) → 중앙 저주파 영역을 0으로 마스킹 → IFFT → HFC 이미지의 순서를 보여준다. 출력 HFC 이미지는 이미지 내 엣지, 노이즈, 압축 아티팩트 등 고주파 정보만 포함하며, 이것이 위조 흔적·초점 경계·그림자 경계 등 저수준 구조 탐지에 핵심적 단서가 됨을 시각적으로 확인할 수 있다. EVP 방법론의 물리적 근거를 제공하는 그림이다.

---

### Figure 3 (p.4) — EVP 상세 아키텍처

**해석:** EVP의 세 모듈(Embedding Tune, HFC Tune, Adaptor)의 상호작용과 트랜스포머와의 연결 구조를 상세히 보여준다. 특히 $\text{MLP}^i_{tune}$이 각 Adaptor별로 **비공유(unshared)**이고 $\text{MLP}_{up}$이 **공유(shared)**임을 색상으로 구분하여 표시한다. 이 설계가 Ablation(Table 7)에서 최적임이 검증되었으며, 각 트랜스포머 레이어에 레이어별로 다른 프롬프트를 주입하는 방식이 핵심임을 이해하는 데 필수적이다.

---

### Figure 4 (p.6) — 태스크별 시각적 비교

**해석:** 4가지 과제 각각에서 EVP 결과(Ours)를 기존 태스크 특화 방법과 정성적으로 비교한다. 위장 객체 탐지(CAMO)에서 SINet, PFNet 대비 더 정확한 객체 경계 탐지; 위조 탐지(CAISA)에서 ManTraNet, SPAN 대비 위조 영역의 더 깔끔한 분리; 그림자 탐지(ISTD)에서 MTMT, FDRNet 대비 그림자 영역의 더 완전한 포착을 보여준다. 단일 통합 모델이 도메인 특화 모델과 대등 이상임을 직관적으로 증명하는 핵심 정성 근거이다.

---

### Figure 5 (p.8) — Ablation 시각 비교 (ISTD 그림자 탐지)

**해석:** (a) 입력, (b) GT, (c) Full-tuning, (d) 프롬프팅 없음, (e) $F_{pe}$ 제거, (f) $F_{hfc}$ 제거, (g) $\text{MLP}^i_{tune}$ 공유, (h) $\text{MLP}_{up}$ 비공유, (i) EVP 전체의 결과를 나란히 비교한다. (d)에서 프롬프팅 없이 디코더만 튜닝하면 그림자 경계가 매우 부정확해지고, (e)(f)에서 각 성분을 제거하면 성능이 하락함을 시각적으로 확인할 수 있다. (i) EVP 전체가 (c) Full-tuning과 거의 동등한 품질을 보이는 것이 핵심 메시지이다. 각 설계 선택의 기여도를 직관적으로 검증하는 가장 중요한 Ablation 그림이다.

---

## 8. 결론: 시사점, 후속 연구 계획, 추가 방향

### 8-1. 모델의 일반화 성능 향상 가능성

**저자들이 직접 제시한 내용 (p.8, Conclusion):**
- "For future works, we will extend our approach to other related problems and hope it can promote further exploration of visual prompting."

**일반화 성능 관련 논문 내 근거:**

1. **다중 아키텍처 일반화**: SegFormer(계층적 트랜스포머)와 SETR(플레인 ViT) 모두에서 EVP가 타 튜닝 방법 대비 우수함을 Table 6, Table 10에서 확인. 아키텍처 독립적(architecture-agnostic) 특성 시사.

2. **단일 모델의 다중 도메인 적응**: 각 태스크당 $F_{pe}$와 $F_{hfc}$만 교체하면 동일 백본으로 4가지 이질적 과제 처리 가능. 이는 새로운 저수준 과제 추가 시 백본 재학습 없이 최소 파라미터 추가만으로 확장 가능함을 의미.

3. **사전학습 규모 확대 시 기대 효과**: 현재 ImageNet-1k 사전학습 SegFormer-B4 기반이나, 더 큰 사전학습 데이터(ImageNet-21k, 웹크롤링 데이터 등)를 사용하는 백본으로 교체 시 HFC 불변성(invariance)이 더 강해져 EVP의 HFC 프롬프트 효과가 더 두드러질 것으로 기대됨.

4. **한계**: 현재 실험의 4개 과제가 모두 이진 분할(binary segmentation) 위주로, 다중 클래스 저수준 분할이나 비전-언어 멀티모달 환경에서의 일반화는 미검증.

---

### 8-2. 2020년 이후 관련 최신 연구 비교 분석

> ⚠️ **주의**: 아래 비교 분석은 본 논문(arXiv 2023.03 제출) 시점 기준 및 본 논문 내 인용 문헌을 중심으로 서술하며, 2023년 이후 발표된 논문과의 직접 비교는 본 논문에 포함되지 않아 엄밀한 수치 비교가 불가함을 명시합니다.

#### 비교 분석 표

| 연구 | 발표 연도 | 방법 유형 | EVP와 비교 |
|------|-----------|-----------|-----------|
| **VPT** (Jia et al.) [33] | 2022 NeurIPS | 암묵적 토큰 프롬프팅 | EVP는 동일 파라미터에서 VPT 대비 Shadow BER 1.73→1.67, Forgery AUC .847→.857 개선 (Table 6) |
| **AdaptFormer** (Chen et al.) [4] | 2022 NeurIPS | Adapter 기반 | EVP(r=16) vs AdaptFormer: 동등 파라미터에서 전 태스크 소폭 우수 (Table 6) |
| **ObjectFormer** (Wang et al.) [74] | 2022 CVPR | HFC를 보완 신호로 사용한 위조 탐지 특화 | EVP는 도메인 특화 없이 CAISA F1 .579→.636 달성. HFC 활용법은 유사하나 EVP는 프롬프팅 프레임워크로 통합 |
| **FBNet** (Lin et al.) [35] | 2022 ACM MM | 주파수 기반 위장 객체 탐지 특화 | CAMO $S_\alpha$ FBNet .783 vs EVP .846. EVP 대폭 우수 |
| **TransForensics** (Hao et al.) [22] | 2021 ICCV | 비전 트랜스포머 기반 위조 탐지 | IMD20 AUC .848 vs EVP .807. 도메인 특화 방법이 우수. EVP는 범용 방법으로의 트레이드오프 |
| **FDRNet** (Zhu et al.) [90] | 2021 ICCV | 특징 분해 기반 그림자 탐지 | SBU BER FDRNet 3.04 vs EVP 4.31. 도메인 특화 방법이 우수 |

#### EVP가 앞으로의 연구에 미치는 영향

1. **저수준 과제의 프롬프팅 패러다임 개척**: 기존 시각 프롬프팅 연구(VPT 등)가 고수준 인식(recognition)에 집중했다면, EVP는 저수준 분할로 확장 가능함을 최초로 체계적으로 증명. 이후 연구들이 저수준 과제에서 PEFT(Parameter-Efficient Fine-Tuning) 방법론을 탐구하는 기반 제공.

2. **HFC의 범용 프롬프트 가능성 제시**: HFC가 단순 추가 입력이 아닌, 사전학습 모델의 "맹점"을 보완하는 이론적 프레임워크를 제시함으로써, 주파수 도메인 분석을 PEFT에 통합하는 새 연구 방향 제시.

3. **Foundation Model의 저수준 과제 적용 촉진**: SAM(Segment Anything Model, 2023), DINOv2(2023) 등 대형 비전 파운데이션 모델의 등장과 맞물려, EVP 스타일의 경량 프롬프팅으로 이들 모델을 저수준 과제에 적응시키는 후속 연구 촉진 예상.

#### 앞으로 연구 시 고려할 점

1. **더 강력한 파운데이션 모델과의 결합**: SAM, DINO, MAE 등 자기지도학습(Self-Supervised Learning) 기반 백본에 EVP를 적용하면 더 강력한 일반화 성능 기대.

2. **태스크 간 프롬프트 공유(Prompt Sharing) 전략**: 현재 EVP는 태스크별 독립 프롬프트를 사용하나, 과제 간 공통 저수준 특성(HFC 등)을 공유 프롬프트로 학습하면 파라미터 효율 추가 향상 가능.

3. **HFC의 학습 가능한 마스크 설계**: 현재 $\tau$는 고정 하이퍼파라미터인데, 이미지 내용에 따라 동적으로 마스크를 생성하는 어텐션 기반 적응형 HFC 추출기 설계가 유망.

4. **비지도/제로샷 저수준 탐지**: 프롬프팅의 핵심 장점인 few-shot 일반화를 저수준 과제에서 정량적으로 검증하는 연구 필요.

5. **비디오 및 3D 도메인 확장**: 비디오의 시간적 HFC(프레임 간 움직임 노이즈 등) 또는 3D 포인트 클라우드의 주파수 도메인 분석으로 EVP 확장 가능.

6. **경쟁 방법과의 공정 비교 프로토콜 표준화**: 위조 탐지에서 추가 학습 데이터 사용 여부에 따른 성능 차이가 큰 만큼, 향후 연구에서 공정한 비교 조건(학습 데이터, 백본 규모)의 명시적 표준화 필요.

---

## 참고 자료

- **본 논문**: Liu, W., Shen, X., Pun, C.-M., & Cun, X. (2023). *Explicit Visual Prompting for Low-Level Structure Segmentations*. arXiv:2303.10883v2.
- **VPT**: Jia, M., et al. (2022). *Visual Prompt Tuning*. arXiv:2203.12119. NeurIPS 2022.
- **AdaptFormer**: Chen, S., et al. (2022). *AdaptFormer: Adapting Vision Transformers for Scalable Visual Recognition*. NeurIPS 2022.
- **SegFormer**: Xie, E., et al. (2021). *SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers*. NeurIPS 2021.
- **GPT-3 (프롬프팅 원류)**: Brown, T., et al. (2020). *Language Models are Few-Shot Learners*. NeurIPS 2020.
- **ObjectFormer**: Wang, J., et al. (2022). *ObjectFormer for Image Manipulation Detection and Localization*. CVPR 2022.
- **ViT**: Dosovitskiy, A., et al. (2021). *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale*. ICLR 2021.
- **SETR**: Zheng, S., et al. (2021). *Rethinking Semantic Segmentation from a Sequence-to-Sequence Perspective with Transformers*. CVPR 2021.
- **GitHub 코드**: https://github.com/NiFangBaAGe/Explicit-Visual-Prompt
