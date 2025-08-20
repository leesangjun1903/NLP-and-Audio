# QLORA: Efficient Finetuning of Quantized LLMs

## 핵심 주장 및 주요 기여  
**QLORA**는 65B급 대규모 언어 모델도 단일 48 GB GPU에서 4비트 양자화된 상태로 파인튜닝하면서 16비트 정밀도의 성능을 유지할 수 있음을 보인다.[1]
- 4비트 정밀도 저장과 BFloat16 연산을 결합하여 메모리 사용량을 780 GB 이상에서 48 GB 미만으로 대폭 절감  
- 정보 이론적으로 최적화된 **4-bit NormalFloat (NF4)**, **Double Quantization**, **Paged Optimizers** 도입  
- LoRA 어댑터와 결합하여 4비트 베이스 모델로도 16비트 완전 파인튜닝 성능 재현  
- Vicuna 벤치마크에서 ChatGPT 대비 99.3% 성능을 기록한 **Guanaco** 모델군 공개  

## 해결하고자 하는 문제  
대규모 LLM(수십억–수조 파라미터)은 16비트 파인튜닝 시 수백 기가바이트의 GPU 메모리가 필요해 접근성이 극도로 제한된다.  
- 기존 양자화는 추론 전용이며 파인튜닝 중 성능이 급격히 저하  
- 메모리 절감을 위한 파인튜닝 기법(LoRA 등)도 활성화 그라디언트 저장량이 큰 병목  

## 제안하는 방법  
### 1. 4-bit NormalFloat (NF4) 양자화  
- 사전 훈련된 가중치가 대체로 정규분포를 따르는 점을 활용해, 정규분포의 분위수(quantile)를 이용해 k-비트 데이터 타입을 설계  
- **양자화 상수** 없이도 각 비트 구간에 균등 확률 부여  
- 양자화:  

$$
X_{\text{Int}} = \mathrm{round}\bigl(c_{\text{FP32}}\cdot X_{\text{FP32}}\bigr),\quad c_{\text{FP32}} = \tfrac{127}{\max|X_{\text{FP32}}|}
$$  

- NF4에서는 정규분포 $$N(0,1)$$의 $$2^k+1$$ 분위수 $$q_i$$를 계산하여 데이터타입 정의:[1]

$$
q_i=\tfrac12\bigl(Q_{N(0,1)}(\tfrac{i}{2^k+1})+Q_{N(0,1)}(\tfrac{i+1}{2^k+1})\bigr),\quad i=0,\dots,2^k-1
$$  

### 2. Double Quantization  
- 1차 양자화 상수 $$c^{(2)}_{\mathrm{FP32}}$$를 다시 8비트 float로 양자화해 메모리 절감  
- 블록 크기 64 기준으로 파라미터당 0.373 bits 추가 절감  

### 3. Paged Optimizers  
- NVIDIA Unified Memory를 활용해 GPU 메모리 부족 시 자동으로 CPU와 페이지 간 전송  
- 큰 시퀀스/배치에서도 OOM 방지  

### 4. QLORA 알고리즘  
- 가중치는 4-bit NF4로 저장, 연산 시 BFloat16으로 dequantization  
- LoRA 어댑터만 학습, 베이스 가중치는 고정

```math
Y_{\text{BF16}} = X_{\text{BF16}}\cdot\mathrm{doubleDequant}(c^{(1)}_{\mathrm{FP32}},c^{(2)}_{k\text{-bit}},W_{\text{NF4}})+X_{\text{BF16}}L_1L_2
```

## 모델 구조  
- 베이스: LLaMA, OPT, BLOOM, Pythia 등 다양한 아키텍처에서 검증  
- 모든 Transformer 레이어의 선형층에 LoRA 어댑터 삽입  
- NF4 양자화 베이스 + LoRA 어댑터(BFloat16) 조합  

## 성능 향상 및 평가  
- **GLUE**, **Super-NaturalInstructions**: 4-bit QLORA가 16-bit 완전파인튜닝과 동등 성능[1]
- **MMLU 5-shot**: LLaMA 7–65B 모두 NF4+DQ QLORA가 16-bit LoRA 성능 완전히 재현 (평균 53.1% vs 53.0%)[1]
- **Vicuna 챗봇 벤치마크**: Guanaco 65B가 ChatGPT 대비 99.3% 성능, 33B: 97.8%, 13B: 90.4%까지 근접[1]
- 메모리 사용: 65B 모델 41 GB, 33B 21 GB로 대폭 경량화  

## 한계  
- 33B/65B 스케일에서 16-bit 완전파인튜닝 성능 동등 여부 완전 확보 미실시  
- 수학 연산, 이론적 추론 등 일부 작업에서 여전히 오류 발생  
- 평가 벤치마크(챗봇 vs NLU) 간 일관성 부족, 주관적 평가 편향 문제  

## 모델의 일반화 성능 향상 가능성  
- **데이터 품질**이 양보다 중요한 것으로 확인: 소수의 고품질 샘플로도 대규모 데이터셋 대비 동일·우수 성능  
- NF4 양자화로 은닉 분포 정보 보존력 증가 → 다양한 다운스트림 태스크에 일반화 능력 유지  
- 높은 파라미터 수 + 낮은 비트 정밀도 조합이 자원 한계 내 최적의 일반화 구현 가능  

## 향후 연구 영향 및 고려 사항  
- **접근성 대폭 확대**: 단일 48 GB GPU로 65B 파인튜닝 → 소규모 연구실·기업에도 대규모 모델 커스터마이징 가능  
- **프라이버시·엣지 컴퓨팅**: 휴대기기·로컬 환경에서 7B 모델 파인튜닝·배포 가능  
- **벤치마크·평가 개선 필요**: 객관·주관 평가 편향, 벤치마크 간 불일치 연구  
- **더 낮은 비트 정밀도**(3-bit)와 다양한 PEFT 기법 결합 탐색  
- **책임 있는 AI**: 편향·오용 방지 위한 안전장치, 데이터 셋 구성·루프백 평가 중요[1]

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/3d90c585-ada5-4356-8dee-a802cbc5c9b5/2305.14314v1.pdf
