# Mamba: Linear-Time Sequence Modeling with Selective State Spaces

**핵심 주장 및 기여**  
Mamba는 선택적 상태공간 모델(selective SSM)을 도입하여 시퀀스 모델이 토큰 간의 *콘텐츠 기반 연산*을 수행하도록 함으로써, 선형 시간 복잡도(𝑂(𝐿))를 유지하면서도 Transformer급 성능을 달성한다. 주요 기여는 다음과 같다.[1]
1. **선택 메커니즘(selection mechanism)** 도입: SSM의 파라미터(Δ, 𝐵, 𝐶)를 입력 의존 함수로 만들어, 입력에 따라 정보를 *선택적으로* 기억하거나 잊도록 함으로써, 이산 및 정보 밀집 데이터에서도 강력한 성능을 발휘.  
2. **하드웨어 최적화 병렬 스캔(scan) 알고리즘** 설계: 선택적(비시정적) SSM의 비효율 문제를 GPU 메모리 계층 구조(SRAM/HBM)를 활용한 커널 퓨전과 재계산(recomputation)으로 해결, 기존 Transformer보다 대규모 시퀀스에서 최대 5× 빠른 추론 속도 달성.[1]
3. **통일된 단일 블록 아키텍처(Mamba)** 제안: H3와 MLP 블록을 결합한 간결한 블록을 반복하여 구성, 선택적 SSM만으로도 다중 모달 언어·오디오·게놈 데이터에서 SOTA 성능을 달성.

***

## 1. 해결 과제  
Transformer는 복잡한 데이터 내의 *밀집한 정보 경로*를 제공하지만, 맥락 길이에 따라 메모리·연산이 이차적 증가(𝐿²)한다. 기존의 SSM 기반 모델들은 선형 또는 준선형 복잡도를 갖지만, 시정적(LTI) 제약 때문에 *콘텐츠 기반 선택(content-based reasoning)*이 불가능해 언어·게놈처럼 불연속·정보 밀집 데이터에서는 성능이 한계에 부딪혔다.[1]

***

## 2. 제안 방법  

### 2.1 선택적 상태공간 모델 (Selective SSM)  
기존 연속-불변 매개변수 (Δ, 𝐴, 𝐵, 𝐶) → 이산 파라미터 (𝐴, 𝐵, 𝐶) 변환 후  

$$
\begin{aligned}
h_t &= A\,h_{t-1} + B\,x_t,\\
y_t &= C\,h_t
\end{aligned}
$$

으로 처리하던 LTI SSM과 달리, Mamba의 S6 블록은 다음과 같이 *입력 의존* 파라미터를 도입한다.[1]

$$
\begin{aligned}
B_t &= s_B(x_t),\quad
C_t = s_C(x_t),\quad
\Delta_t = \tau_\Delta\bigl(\text{Parameter} + s_\Delta(x_t)\bigr),\\
\text{discretize}(\Delta_t,A,B_t)&\to (A_t,B_t),\quad
h_t = A_t\,h_{t-1} + B_t\,x_t,\quad
y_t = C_t\,h_t.
\end{aligned}
$$

여기서 $$s_B, s_C, s_\Delta$$는 작은 선형 투영, $$\tau_\Delta=\mathrm{softplus}$$이다. 이로써 토큰별로 *입력 유지·폐기(gating)*를 가능하게 해, 선택적 복사가 필요한 합성 과제와 유도(head induction) 과제에서 완전 해결 능력 및 맥락 길이 $$>10^6$$까지 완전 일반화 성능을 보인다.[1]

### 2.2 하드웨어-친화적 병렬 스캔  
선택적 SSM은 시불변성을 잃어 전통적 FFT 기반 병렬화 실패. 따라서 GPU HBM→SRAM 데이터 이동을 최소화하는 커널 퓨전으로:  
1) Δ, A, B, C를 SRAM으로 로드.  
2) 이산화·스캔 연산(병렬 prefix scan) 수행.  
3) 출력 $$y$$만 HBM에 기록.  
역전파 시 재계산(recomputation) 적용으로 메모리 사용을 Transformer·FlashAttention 수준으로 유지하면서, 시퀀스 길이 32K 이상에서 FlashAttention 대비 최대 7× 빠른 스캔 성능 달성.[1]

***

## 3. 모델 구조  
Mamba 블록은 **Gated MLP**에 선택적 SSM을 더한 형태로, 두 번 반복하여 Transformer의 MHA+MLP(12𝐷²) 파라미터 규모를 맞춘다(확장율 $$E=2$$). 각 블록 내 주요 연산:  
1. **입력 투영**: $$x\to W_q x,\ W_k x,\ W_v x$$ (확장 차원)  
2. **선택적 SSM**: 위 2.1 절의 S6 연산  
3. **SiLU 활성화** 및 출력 투영  
4. **잔차 연결 + LayerNorm**  

이러한 동질적 블록을 쌓아 깊이 확장만으로 다중 도메인 시퀀스에 보편 적용 가능.

***

## 4. 성능 향상 및 한계  

### 4.1 일반화 성능 향상  
- **합성 과제**: Selective Copying 과제에서 기존 SSM·H3·Hyena는 정확도 30–60%에 그친 반면, S6는 97–99%로 완전 해결. 유도 과제(induction heads)는 훈련 길이 256→테스트 $$10^6$$까지 정확도 100%로 완전 일반화.[1]
- **언어 모델링**: 선형 시간 모델 중 최초로 GPT3 급 레시피(“Transformer++”)와 동등한 스케일링 법칙(perplexity) 달성, 1.4B 파라미터에서 2.8B Transformer 성능에 근접·추월.[1]
- **DNA·오디오**: HG38 유전체에서 길이 최대 $$10^6$$까지 컨텍스트 확장 시 기존 HyenaDNA 성능 악화 대비, Mamba는 길이 증가에 따라 Perplexity가 꾸준히 감소. 음성(waveform)에서도 S4 기반 U-Net 모델보다 긴 시퀀스에서 BPB 지표 개선.[1]

### 4.2 한계 및 고려 사항  
- **연속 vs. 불연속 데이터 스펙트럼**: 오디오처럼 연속 신호에는 LTI 모델이 유리할 수 있어, 선택 메커니즘이 오히려 성능 저하를 초래할 수 있음(내부 블록만 선택적용 등이 필요).  
- **대규모 스케일링**: 7B 이상 초대형 모델에서의 안정성·확장성 검증 필요.  
- **파라미터 초기화·튜닝**: Δ·B·C 파라미터의 투영 차원, 초기화 스케일(실수 vs. 복소수) 등이 도메인별 최적화 요인으로 남음.

***

## 5. 향후 연구 영향 및 고려 사항  
- **다중 모달 통합**: 동일 블록으로 언어·생체신호·비디오 등 다양한 시퀀스 도메인 통합 백본으로 응용 가능.  
- **적응적 선택 메커니즘**: 경계 리셋, 시계열 변화점 탐지 등 강화학습·메타 RL에 자동 선택·리셋 기능 적용.  
- **하드웨어 최적화 확장**: 차세대 TPU나 NPU에 맞춘 메모리 계층 최적화 및 동적 배치 전략 연구.  
- **학습·추론 효율성**: 초장기 시퀀스 학습에서 적절한 sequence length warmup, mixed-precision 학습 안정성 확보 방안 고도화.

Mamba의 **선택적 상태공간** 개념은 시퀀스 모델의 *효율성과 효과성 병립*이라는 오래된 딜레마를 해소하는 중요한 발전으로, 차세대 초장기·대규모 시퀀스 모델 연구에 중대한 이정표를 제시한다.  

***

 Mamba: Linear-Time Sequence Modeling with Selective State Spaces (2024)[1]

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/94b534ef-2323-4f3b-a3f4-d008e7711972/2312.00752v2.pdf)
