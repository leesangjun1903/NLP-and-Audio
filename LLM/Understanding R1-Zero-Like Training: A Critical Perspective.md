# Understanding R1-Zero-Like Training: A Critical Perspective

## 1. 핵심 주장 및 주요 기여  
“Understanding R1-Zero-Like Training: A Critical Perspective” 논문은 R1-Zero 스타일의 **직접 강화학습(RL) 튜닝**이 LLM의 추론 능력을 어떻게 향상시키는지, 그리고 그 과정에서 **사전학습 편향(pretraining bias)** 과 **최적화 편향(optimization bias)** 이 어떻게 작용하는지를 체계적으로 분석한다.  
- 기존 연구들이 간과한 두 가지 축(베이스 모델 특성, RL 최적화 알고리즘)을 분리하여 평가  
- GRPO(Group Relative Policy Optimization)의 **부작용(bias)** 을 규명하고, 이를 제거한 **Dr. GRPO** 알고리즘을 제안  
- 최소한의 연산량으로 7B 모델에서 AIME 2024 43.3% 달성, 새로운 SOTA 수립  

## 2. 해결하고자 하는 문제  
1) **베이스 모델(pretrained base) 편향**  
   - 일부 공개 모델(DeepSeek-V3-Base, Qwen2.5 등)은 이미 수학 추론 능력 및 ‘Aha 모먼트(자기검토)’를 사전습득  
   - 템플릿(prompt template) 사용 여부가 “질문 응답” 행동과 초기 성능에 결정적 영향  
2) **최적화(opt) 편향**  
   - GRPO는 응답 길이 및 샘플 그룹 내 난이도 차이에 따라 잘못된 길이 편향(length bias)과 질문 난이도 편향(difficulty bias)을 유발  
   - 이로 인해 잘못된 응답에 과도한 토큰이 낭비되고, ‘긴 롱코트(long-CoT)’를 “스킬”처럼 오해  

## 3. 제안하는 방법  
### 3.1 Dr. GRPO: Unbiased Group Relative Policy Optimization  
– GRPO의 손실 함수에서 응답 길이(|oᵢ|)와 그룹 내 표준편차(std(R)) 정규화 항 제거  
– PPO surrogate objective (클리핑)과 **Monte Carlo advantage**  
  
$$ \hat A_{i,t} = R(q,o_i) - \mathrm{mean}_j R(q,o_j) $$  
  
⇒ Dr. GRPO 손실:  

$$
    \mathcal{L} = \frac{1}{G}\sum_{i=1}^G \sum_{t=1}^{|o_i|}
    \min\Bigl(r_{i,t}\hat A_{i,t},\,\mathrm{clip}(r_{i,t},1-\epsilon,1+\epsilon)\hat A_{i,t}\Bigr)
  $$  
  
(여기서 $$r_{i,t}=\frac{\pi_\theta(o_{i,t})}{\pi_{\theta_\text{old}}(o_{i,t})}$$)  

### 3.2 모델 구조 및 학습 파이프라인  
- 베이스 모델: Qwen2.5-Math-7B, Qwen2.5-Math-1.5B 등  
- 템플릿: R1 템플릿, Qwen-Math 템플릿, No 템플릿 비교  
- RL 데이터셋: MATH 레벨 3–5 문제, 다양한 크기(2K–57K) question sets 실험  
- Dr. GRPO로 27시간(8×A100) 연산  

## 4. 성능 향상 및 한계  
### 성능 향상  
- **토큰 효율(Token Efficiency)**: Dr. GRPO는 GRPO 대비 잘못된 응답 길이 단축, 전체 평가 벤치마크 평균 약 2–5%p 향상  
- **SOTA 달성**: AIME 2024 43.3%, AMC 62.7%, MATH500 80.0% (Oat-Zero-7B)  
- **템플릿·질문셋 상호작용**: 간단한 o.o.d. 질문셋도 Qwen 템플릿 시 강력한 RL 효과  

### 한계  
- 베이스 모델에 이미 편향된 수학 능력 존재 → RL의 실제 기여 분리 어려움  
- Dr. GRPO는 길이 편향 제거만 해결, 다른 잠재적 편향(보상 설계, 데이터 다양성) 미해결  

## 5. 일반화 성능 향상 관점  
- **도메인 특화 사전학습(Math pretraining)**: Llama3.2-3B에 수학 데이터 추가(pretraining) 시 RL 상한선(ceiling) 대폭 상승  
- **O.o.d. 일반화**: 작은 ASDiv(기본 대수)셋 학습 후 어려운 벤치마크에 2배 성능 향상  
- → 사전학습 단계에서 도메인 지식 다양화가 RL 일반화에 결정적  

## 6. 향후 연구에의 영향 및 고려사항  
- **편향 분석 필수**: RL 튜닝 이전 베이스 모델의 사전습득 역량·편향 정량화  
- **최적화 알고리즘 검증**: 수치 안정성 차원에서 손실 정규화 방법 감시  
- **보상 설계 다양화**: 일관된 보상(normal global advantage) vs. 그룹 내 보상  
- **데이터셋 다양성**: 단순 연산→고난도 고르게 분포하는 o.o.d. 질문 믹스 최적화  
- **토큰 효율과 비용 절감**: Dr. GRPO처럼 불필요한 토큰 낭비 최소화하는 알고리즘 선호  

---  
이 논문은 R1-Zero 스타일 RL 튜닝 연구에 있어 “사전학습 편향”과 “최적화 편향”을 모두 드러내고, **토큰 효율을 개선하면서도 성능 저하 없이 일반화 역량을 유지·강화할 수 있는** Dr. GRPO 기법을 제안함으로써, 향후 LLM 강화학습 연구의 **설계 기준**을 제시한다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/cd27a447-ea0f-44c2-9a71-4dc04e66879e/2503.20783v1.pdf
