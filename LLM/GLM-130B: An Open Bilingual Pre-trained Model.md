# GLM-130B: An Open Bilingual Pre-trained Model

## 1. 핵심 주장 및 주요 기여
GLM-130B는 영어와 중국어를 모두 지원하는 1300억 매개변수 규모의 오픈소스 사전학습 언어 모델로, 다음과 같은 기여를 제시한다.
- 비침투적(blank infilling) 양방향 주의(attention) 기반 GLM(General Language Model) 아키텍처 적용으로 GPT-3(175B)와 PaLM(540B)을 능가하는 제로샷/파인튜닝 성능 달성.
- DeepNorm 레이어 정규화, 임베딩 그래디언트 축소(α=0.1) 등 훈련 안정성 확보 전략 개발로 100B 규모 모델의 수렴과 학습 붕괴(loss spike) 문제 해결.
- INT4 양자화(quantization)를 통한 추론 메모리 절감(원본 대비 25% 수준) 및 4×RTX 3090, 8×RTX 2080 Ti 환경에서 실용적 추론 가능.

## 2. 문제 정의·제안 기법·모델 구조·성능·한계

### 2.1 해결하고자 하는 문제
1) 100B 규모 언어 모델의 **훈련 불안정성**(loss spike, gradient explosion)  
2) 고비용·고자원 요구사항으로 인한 **인프라 접근성 제약**  
3) 기존 GPT-스타일(unidirectional) 모델의 **영어 편향** 및 **중국어 성능 미흡**

### 2.2 제안 방법
1) **GLM 알고리즘**: bidirectional attention + autoregressive blank infilling  
   - 입력 $$x=[x_1,\dots,x_n]$$에서 연속 스팬 $$\{s_i\}$$을 [MASK] 또는 [gMASK]로 변환  
   - 마스크 부분을 순열 기반 autoregressive 생성  
2) **DeepNorm** (Wang et al. 2022): Post-LN 변형으로 깊은 레이어에서의 그래디언트 스파이크 방지  
   - $$N$$층 모델에서 LayerNorm 계산:  

$$
       \mathrm{DeepNorm}(x)=\mathrm{LayerNorm}\bigl(\alpha\,x + \mathrm{Network}(x)\bigr),\quad \alpha=(2N)^{\tfrac12}.
     $$

3) **Embedding Gradient Shrink (EGS)**: 임베딩 층 그래디언트 스케일 축소  

$$
     \Delta W_{\rm emb}\leftarrow \alpha\,\Delta W_{\rm emb} + (1-\alpha)\,\Delta W_{\rm emb}.\mathrm{detach},\quad \alpha=0.1.
   $$

4) **Mixed-precision**: FP16 연산 + FP32 옵티마이저 상태  
5) **INT4 Weight Quantization**: 주요 선형층(weight-only) 절반 메모리 사용  

$$
     x_q = \mathrm{round}\bigl(x/s\bigr),\quad s=\frac{\max|x|}{2^{b-1}-1},\ b=4.
   $$

### 2.3 모델 구조
- 70개 Transformer 레이어, hidden size 12,288, head 수 96, FFN (GeGLU)  
- 양방향(fully bidirectional) attention, sequence length 2048  
- 3D parallelism: 데이터 × 텐서(4-way) × 파이프라인(8-way)

### 2.4 성능 향상 결과
- **LAMBADA zero-shot**: GLM-130B 80.21% vs. GPT-3 76.2%  
- **MMLU 5-shot**: 44.8% vs. GPT-3 43.9%  
- **BIG-bench-lite zero-shot**: 13.31% vs. GPT-3 4.35%, PaLM 8.05%  
- **CLUE zero-shot**(중국어): 평균 +24.3%p vs. ERNIE Titan 3.0 260B  
- **추론 속도**: A100 단일서버에서 7–8× 가속  
- **메모리 효율**: INT4 양자화로 가중치 메모리 75% 감소, 4×3090/8×2080Ti에서 추론 가능

### 2.5 한계
- **데이터 편향**: 영어·중국어 혼합학습으로 일부 문화적 편향 해소됐으나, 특이 중국어권 편향 가능성  
- **훈련비용**: 96 DGX-A100서버·60일 학습(총 442.4 MWh)로 여전히 높은 탄소 배출  
- **적은 파인튜닝 실험**: 주요 연구는 제로/파인튜닝 없이 평가, 다운스트림 파인튜닝 효율은 추가 검증 필요

## 3. 일반화 성능 향상 관점
- **Bidirectional attention** 으로 컨텍스트 이해·생성 모두 강화 → zero-shot 언어모델링과 QA, 추론에서 대규모 샷 없이도 높은 성능  
- **MIP(Multi-task Instruction Pre-training)** 5% 토큰 투입으로 prompt-generalization↑ (특히 coreference, NLI 계열)  
- **DeepNorm+EGS** 로 수렴 안정화 → 더 많은 토큰(400B) 처리하며 학습 지속 가능  
- **INT4 quantization** 시에도 원본 대비 대규모 성능 저하 불가해 추론 시 일반화 성능 유지

## 4. 향후 연구 영향 및 고려사항
- **개방형 100B급 LLM 활성화**로 학계·산업계 연구 진입장벽 크게 완화  
- **모델 안정화 기법**(DeepNorm, gradient shrink)·**양자화 scaling law** 심층 분석 필요  
- **데이터 다양성·윤리성 평가**: 중국어 문화권 편향·toxicity 경감 메커니즘 추가 연구  
- **친환경 학습**: 저탄소 파라다임, 지식증류·모델 압축으로 CO₂eq 절감 방안 모색  
- **파인튜닝·효율적 전이 학습**: 지표 향상 대비 비용 최적화 기법(Efficient-FT, Prompt Tuning v2)

**결론**: GLM-130B는 100B급 오픈 양방향 모델 학습의 안정화·효율화·실용화를 입증했으며, 향후 LLM 연구의 투명성, 접근성, 지속가능성에 중대한 전환점을 제공할 것이다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/f4f85d50-7cd2-437a-a6d2-33385e1a48ab/2210.02414v2.pdf)
