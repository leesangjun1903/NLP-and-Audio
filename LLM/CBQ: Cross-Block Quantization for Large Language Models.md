# CBQ: Cross-Block Quantization for Large Language Models

## 1. 핵심 주장 및 주요 기여  
CBQ는 초저(ultra-low) 비트 양자화 환경에서 대형 언어 모델(LLM)이 겪는 성능 저하의 근본 원인으로 **모델 파라미터 수 증가에 따른 intra-layer 및 inter-layer 의존성 강화**를 지목하고, 이를 효과적으로 보정하기 위해 다음 세 가지 핵심 기법을 제안한다.  
1. **Cross-Block Dependency (CBD)**: 인접 블록 간 장거리 의존성을 유지하며 슬라이딩 윈도우 방식으로 블록을 중첩 최적화.  
2. **LoRA-Rounding**: 저순위 분해(low-rank decomposition)를 활용해 weight rounding 보상 행렬(∆W)을 학습, intra-layer 의존성 보존 및 계산 효율성 확보.  
3. **Coarse-to-Fine Preprocessing (CFP)**: 사분위수 기반 통계 기법으로 가중치·활성화 양쪽의 이상치(outlier)를 정확 검출·처리.  

이 세 요소를 통합해 W4A4, W4A8, W2A16 같은 극저 비트 환경에서도 기존 PTQ 기법 대비 1–2% 이상의 성능 향상을 달성했다.

***

## 2. 문제 정의, 제안 방법 및 모델 구조  

### 2.1 해결하고자 하는 문제  
- 초저 비트(2∼4비트) 양자화 시 ε(quantization perturbation) 크기가 커지며 Hessian 행렬 H의 비대각 원소(인접 파라미터 간 의존성)가 지배적으로 증가.  
- 전통적 layer-wise 또는 block-wise 재구성 최적화는 이러한 **intra-layer**, **inter-layer** 의존성을 포착하지 못해 성능 저하 발생.  

### 2.2 제안 방법  
1) Cross-Block Reconstruction  
   - 슬라이딩 윈도우 내 블록 i…k를 동시에 최적화:  

$$
       \min_{S^W_{i,k}, S^X_{i,k}, \Delta^W_{i,k}} 
       E\bigl(T_{i,k}(W_{i,k}, X_{i,k}),\; T_{i,k}(Q(W_{i,k}) + \Delta^W_{i,k},\, Q(X_{i,k}))\bigr)
     $$  
   
   - 손실 함수 E는 L2 거리와 KLD 합성:

$$
       E(h_1,h_2)=\lVert h_1-h_2\rVert_2^2 + D_{KL}\bigl(\sigma(h_1),\sigma(h_2)\bigr)
     $$

2) LoRA-Rounding  
   - 보상 행렬 $∆W = Clip(σ(V)(ζ−γ)+γ,0,1)$을 직접 학습하는 대신 저순위 행렬 A₁∈ℝ^{d×r}, A₂∈ℝ^{r×k}로 분해:  

$$
       V = A_1 A_2,\quad r\ll\min(d,k)
     $$  
   
   - 정규화 보상 손실:

$$
       L_{\mathrm{com}} = \sum_{i,j} \bigl(1 - |2\Delta W_{ij} -1|\bigr)^\beta
     $$  

3) Coarse-to-Fine Preprocessing  
   - 1단계: 사분위수(IQR) 기반 임계치 $$T=Q_3+1.5\cdot\mathrm{IQR}$$로 이상치 후보 집합 O 추출  
   - 2단계: O를 intra-set 분산 및 inter-set 거리 M = M_inter −λ₂ M_intra 최소화하여 진짜 이상치 O_outlier 결정  
   - 가중치는 절단(truncate), 활성화는 채널별 스케일링 적용

### 2.3 모델 구조  
- 기존 Transformer 블록 구조 유지  
- 슬라이딩 윈도우(기본 크기 2블록, 오버랩 1)로 연속 블록 최적화  
- 각 윈도우별로 SW, SX(step sizes), LoRA 보상 행렬(A₁, A₂) 동시 업데이트  

***

## 3. 성능 향상 및 한계  

### 3.1 성능 요약  
- **Zero-Shot 정확도**: W4A16 환경에서 OPT-30B, LLAMA-65B 등 전체 모델 대비 1–2%p↑  
- **Perplexity**: C4/WikiText2 기준 W4A16에서 GPTQ 대비 PPL 0.1–0.3p↓  
- **초저 비트(W2A16, W4A4)**: 기존 SOTA 대비 압도적 우위, 성능 감소폭 1%p 이내 유지  
- **효율성**: LLAMA-65B 4비트 양자화에 4.3시간 소요, OmniQuant 대비 약 2× 속도  

### 3.2 한계  
- **GPU 메모리·시간 부담**: 블록 간 중첩 및 LoRA 보상 학습으로 자원 증가  
- **슬라이딩 윈도우 범위 민감도**: 블록 수·오버랩 조합에 따라 성능 기복  
- **활성화 스케일링 의존성**: 일부 채널 왜곡 시 성능 저하 가능성

***

## 4. 일반화 성능 향상 가능성  

- **Cross-Block Dependency**는 블록 간 비국소적 상관관계를 학습하므로, 다양한 Transformer 기반 아키텍처(BERT, ViT 등)에도 적용 가능  
- **LoRA-Rounding** 저순위 보상 기법은 파라미터 수가 큰 모델일수록 상대적 효율↑, GPT-QAT, QAT에도 확장 여지  
- **CFP** 통계적 이상치 처리 전략은 도메인별 데이터 분포 차이에 강건해, 음성·비전 모델 양자화에도 전이 가능  

***

## 5. 향후 연구에의 영향 및 고려 사항  

- **차세대 PTQ 설계**: intra-layer/inter-layer 의존성 모델링이 표준화되어, 더 넓은 범위의 블록 조합(다단계 윈도우) 연구 촉진  
- **하드웨어 친화적 구현**: 중첩 윈도우 최적화 오버헤드를 줄이는 효율적 알고리즘·가속기 지원 필요  
- **동적 비트폭 할당**: 모델 내부 의존성 강도에 따라 비트 할당을 adaptive하게 조절하는 연구  
- **안정성 검증**: CFP 처리로 인한 잠재적 채널 왜곡·스케일 불안정성에 대한 이론적 보장 연구 필요  

CBQ는 초저 비트 PTQ 분야의 새로운 패러다임을 제시하며, 모델 내부 구조적 의존성을 중점적으로 다루는 후속 연구에 핵심 요철을 제공할 것이다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/73371a2e-bf8a-48d8-a8ca-3b3c20e0c423/2312.07950v5.pdf
