# AQLM : Extreme Compression of Large Language Models via Additive Quantization

## 핵심 주장 및 주요 기여  
이 논문은 **Additive Quantization of Language Models (AQLM)**이라는 새로운 양자화 기법을 제안하여, 기존의 2–3비트급 극단적 압축(extreme compression)에서 발생하던 성능 저하를 대폭 완화함으로써, 3비트 이하 영역에서도 **모델 크기 대비 정확도(PPL 및 제로샷 정확도)** 측면에서 **Pareto 최적성**을 달성한 최초의 방법임을 보인다.[1]

주요 기여는 다음과 같다:  
- 멀티코드북 기반의 **Additive Quantization(AQ)** 기법을 LLM에 확장하여, 각 레이어의 입력-출력 활성화 분포를 고려한 **입력 적응형(addaptive)** 코드북 최적화 및 **블록 단위(block-wise)** 공동 미세조정(fine-tuning)을 도입.  
- LLAMA 2 및 Mixtral 계열 모델에서 2–4비트 압축 실험을 통해, 특히 2비트 영역에서 기존 QuIP# 대비 WikiText-2 PPL을 최대 **30% 이상 감소**시키고, 제로샷 정확도에서도 유의미한 향상을 달성.  
- GPU/CPU 가속 커널을 개발하여, 2비트 압축임에도 FP16/FP32 대비 최대 **3× 이상**의 추론 속도 향상을 확인.  

## 해결하고자 하는 문제  
최신 오픈 소스 LLM(예: LLAMA 2, Falcon 등)은 수백억~수조 개의 파라미터로 이루어져 있어, 로컬 디바이스에서의 실행이 어려우며, 특히 2–3비트급 **극단적(post-training) 양자화** 시 정확도 급락과 구현 복잡도가 큰 문제로 작용해 왔다.[1]

## 제안 방법  
### 1. 모델 및 목표 함수  
Transformer의 각 선형 레이어가 $$W\in\mathbb{R}^{d_{\text{out}}\times d_{\text{in}}}$$ 가중치와 입력 활성화 $$X\in\mathbb{R}^{d_{\text{in}}\times n}$$ 을 가질 때, 압축된 가중치 $$\widetilde W$$가 원본 출력과 최소 MSE 차이를 가지도록 최적화한다:  

$$
\min_{\widetilde W}\|WX - \widetilde W X\|_F^2.
$$  

여기서 $$\widetilde W$$는 **M개의 코드북** $$\{C_m\}\_{m=1}^M,C_m\in\mathbb{R}^{g\times 2^B}$$와 **각 그룹별 one-hot 코드** $$b_{i,j,m}$$를 합성한 형태로 표현된다:[1]

$$
\widetilde W_i = \bigoplus_{j=1}^{d_{\text{in}}/g}\sum_{m=1}^M C_m\,b_{i,j,m},\quad
\text{bit-width}=M\cdot B.
$$

### 2. AQLM 알고리즘 구성  
- **Phase 1: Beam Search**  
  MRF 기반의 beam search로 discrete code $$b$$를 MSE 손실식(식 (7) in )에 따라 갱신.[1]
- **Phase 2: Codebook Update**  
  고정된 code 아래에서 codebook $$C_m$$과 per-unit scale $$s$$를 Adam 최적화로 갱신(식 (8) in ).[1]
- **Phase 3: Block-wise Fine-tuning**  
  각 Transformer 블록 단위로 코드를 고정하고, codebook·scale·Norm 파라미터를 재조정하여 블록 출력 오차를 최소화.  

이 세 단계를 층별로 순차 적용한 뒤, 전체 모델에 대해 **엔드투엔드(end-to-end) 지식 증류(KL divergence)** 방식의 미세조정을 추가 수행하여 전역 최적화를 도모한다[Appendix A].[1]

## 모델 구조  
기존 Transformer 아키텍처에 코드북 기반의 **Additive Quantization 레이어**가 결합된 형태이며, 압축된 가중치는 GPU/CPU 친화적 형태로 효율적 인코딩·디코딩 커널을 통해 실행된다.

## 성능 향상  
- **2비트 압축** 시 WikiText-2 PPL:  
  - LLAMA 2 13B: QuIP# 6.06 → AQLM 5.37 (≈12%↓) → AQLM★ 5.22 (end-to-end 미세조정 후)[Table 1, 4].[1]
- **제로샷 평균 정확도**:  
  - LLAMA 2 13B 2비트: QuIP# 57.55% → AQLM 61.80% → AQLM★ 62.67%[Table 1, 4].[1]
- **추론 속도**:  
  - GPU (RTX 3090) gate_proj: FP16 대비 ×1.20–1.31배, 2×8-bit config에서 ×1.82–3.05배[Table 5].[1]
  - CPU (Intel i9): FP32 대비 최대 ×4.07배[Table 5].[1]

## 한계  
- **양자화 시간**: 7B 기준 단일 A100에서 ≈1일, 70B는 10–14일 소요. 블록 튜닝과 beam search의 계산 비용이 상당함.  
- **하이퍼파라미터 민감도**: beam size, codebook 수·크기, fine-tuning 단계별 학습률·스텝 수 등에 따라 성능 편차 존재.  
- **극소형 모델**에서 2비트 이하 압축 시 작은 모델이 더 유리할 수 있는 Pareto 비최적 영역 존재.[1]

## 일반화 성능 향상 가능성  
- **블록 단위 미세조정**은 층 간 상호작용을 고려해 오류 보정, 다양한 다운스트림 태스크로 일반화될 여지가 크다.  
- **코드북 학습**은 데이터 분포 적응형이므로, 특정 도메인이나 멀티모달 입력에 특화한 fine-tuning에도 확장 가능.  
- 대규모 **엔드투엔드 증류** 시나리오에서, 더 다양한 손실(예: 회귀, 순서 예측 등)에 대한 맞춤형 fine-tuning 전략 도입으로 일반화 성능 추가 향상이 기대된다.

## 향후 연구 영향 및 고려 사항  
- **QAT 및 PEFT의 결합**: AQLM의 블록 튜닝을 QAT나 라이트한 PEFT 기법과 결합해, 양자화 시간 단축과 성능 유지 두 마리 토끼를 잡는 연구.  
- **다른 도메인 적용**: 비언어 모델(CV, 음성 합성) 내 대형 파라미터 행렬에도 Multi-Codebook Quantization 적용 가능성 탐색.  
- **하이퍼파라미터 자동화**: Bayesian optimization이나 meta-learning으로 beam size 등 주요 설정을 자동화하여 튜닝 부담 경감.  
- **하드웨어 제약 환경**: 모바일·임베디드 디바이스에서 메모리·연산 제약을 반영한 경량화된 커널 최적화 연구가 필요.  

AQLM은 2–3비트 극한 압축에서 **모델 크기 대비 성능**을 근본적으로 재정의했으며, 이후 LLM 경량화 및 엣지 추론 연구의 핵심 기반이 될 전망이다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/a8494cdb-ea6b-44e4-9c10-9886756ec173/2401.06118v4.pdf
