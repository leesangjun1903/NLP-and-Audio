# Parameter-efficient Fine-tuning of Large-scale Pre-trained Language Models

**핵심 주장 및 주요 기여**  
이 논문은 방대한 파라미터를 가진 사전학습 언어모델(PLM)을 다양한 자연어처리(NLP) 과제에 적응시킬 때, 전체 파라미터를 미세조정(fine-tuning)하는 대신 **극소수의 파라미터(∆Θ)만 업데이트**하는 *delta-tuning* 전략을 제안하고 종합적으로 분석한다. 주요 기여는 다음과 같다.  
1. **통합 분류(criteria)**: delta-tuning 기법을 “추가(addition) 기반”, “명시(specification) 기반”, “재매개변수화(reparameterization) 기반” 세 가지로 체계적으로 분류했다.  
2. **이론 분석**: 저차원 내재차원(subspace optimization)과 최적제어(optimal control) 관점에서 delta-tuning의 이론적 토대를 제시했다.[1]
3. **광범위 실험**: 100여 개의 NLP 과제에서 T5BASE·T5LARGE를 대상으로 prompt-tuning, prefix-tuning, LoRA, adapter, full fine-tuning을 비교하여 성능·수렴·효율·조합 가능성·규모 확장성·전이능력을 평가했다.  

***

## 1. 문제 정의 및 제안 방법

### 1.1 문제 정의  
사전학습된 모델 Θ ∈ ℝⁿ을 주어진 데이터 𝒟에 맞춰 적응할 때, vanilla fine-tuning은 n개의 파라미터를 모두 업데이트하므로 계산·저장 비용이 기하급수적으로 증가한다. Delta-tuning은 Θ′ = Θ + ∆Θ, ∥∆Θ∥₀ ≪ n 을 만족하도록 ∆Θ만 최적화하여 비용을 획기적으로 절감한다.

### 1.2 분류 기준  
-  **추가 기반(addition-based)**: 모델 내부에 adapter 모듈13, prompt token14 등을 *추가*하고 그 파라미터만 업데이트  
-  **명시 기반(specification-based)**: bias term(BitFit) 등 기존 파라미터 중 일부를 *선택*해 업데이트, 나머지는 고정  
-  **재매개변수화(reparameterization-based)**: 파라미터 변화 ΔW를 저차원(rank-r) 행렬 A Bᵀ 형태로 분해(LoRA)하거나, intrinsic subspace로 재매개변수화

### 1.3 대표적 기법 공식  
1) Adapter13  
   입력 h ∈ ℝᵈ에 대해  

$$ \text{Adapter}(h) = h + f(hW_d)\,W_u, $$  
   
   여기서 $$W_d∈ℝ^{d×r}, W_u∈ℝ^{r×d}, f$$은 activation (예: ReLU).  

2) Prefix-tuning40  
   각 층의 키·값에 learnable prefixes $$P_k,P_v∈ℝ^{l×d}$$ 삽입. 기존 어텐션에서  

$$ \mathrm{Attention}(Q,[P_k;K],[P_v;V]). $$  

3) LoRA15  
   사전학습된 self-attention weight W₀에 대해 변화 ΔW를 저랭크로 가정,  

$$ W = W₀ + ΔW = W₀ + A\,Bᵀ,\quad A∈ℝ^{d×r},\,B∈ℝ^{d×r},\,r≪d. $$

***

## 2. 모델 구조 및 학습 효율

### 2.1 구조 비교  
|기법|튜닝 파라미터 비율|주요 구조 변화|
|---|---|---|
|Full fine-tune|100%|모델 전체 업데이트|
|Adapter|0.5–8%|각 레이어에 bottleneck adapter 삽입|
|BitFit|0.38%|bias term만 업데이트|
|LoRA|0.01–0.1%|어텐션 가중치 변화 저랭크 분해|
|Prefix-tune|0.03%|각층 키·값에 prefix 추가|

### 2.2 메모리·연산 효율  
- 소규모 배치(1–8) 시 최대 **75% GPU 메모리 감소**.  
- 백워드 연산 시 업데이트 파라미터 수 감소로 **연산 속도 가속**.  
- Adapter만 전방 연산 경로 연장으로 약간의 지연(latencty) 존재.

***

## 3. 성능 분석 및 일반화

### 3.1 성능 비교  
100개 과제 평균: Full FT(69.3) > LoRA(67.3) ≈ Prefix(65.1) ≈ Adapter(66.8) > Prompt(49.8) .

### 3.2 일반화 성능  
- **일반화 격차(train−dev)**는 delta-tuning이 FT보다 작아 오버핏팅 억제에 유리하며, 특히 Prompt-tuning이 최소 격차를 보인다.  
- **조합 효과**: Prompt⊕Adapter⊕BitFit 시 평균 성능 상승, 단 조합 최적화는 과제·데이터 규모에 따라 달라짐.  

### 3.3 규모 확장 시 일반화  
모델 크기 T5SMALL→T5BASE→T5XXL 증가에 따라 delta-tuning 전반의 수렴·성능 향상 관찰. 대규모 모델일수록 **랜덤 모듈 튜닝**(selective-module)만으로 FT 근접 성능 획득.

***

## 4. 한계 및 미래 연구 과제

1. **조합 최적화 불확실성**: 다양한 delta 기법을 어떻게 자동으로 최적 조합할지 미정.  
2. **수렴 안정성**: Prompt 계열은 소규모 모델·데이터에서 수렴이 느리거나 불안정.  
3. **전이학습 한계**: 과제 유형 간 전이는 제한적이며, 임의 과제 전환 시 성능 저하.  

***

## 5. 미래 영향 및 고려 사항

- **경량화 서비스**: 서버리스·모바일 환경에서 대형 PLM 활용도를 높이고, 파라미터 공유·체크포인트 교환 플랫폼 발전 전망.  
- **자동 구조 탐색**: 메타학습·하이퍼넷을 이용한 delta 구조 자동 생성·조합 연구가 요구됨.  
- **이론적 통합**: 최적제어·서브스페이스 최적화 관점에서 신규 delta-tuning 설계에 이론적 근거 부여 필요.  

***

 AAghajanyan et al. (2021) Intrinsic dimensionality…[1]
 Li & Liang (2021) Prefix-tuning…  
 Fig. 2 GPU memory comparison   
 Table 1 average of all tasks   
 Extended Data Table 3 generalization gap   
 Table 2 combination results (GLUE)   
 Fig. 3 power of scale experiments

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/b51ffb86-d73a-4400-b3d2-c233e5ed09f3/s42256-023-00626-4.pdf)
