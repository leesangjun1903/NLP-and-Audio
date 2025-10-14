# SmolLM2: When Smol Goes Big — Data-Centric Training of a Small Language Model

SmolLM2는 **1.7억**(1.7 B) 매개변수의 소형 언어 모델이지만, **11조 토큰** 이상의 대규모 데이터 중심 훈련을 통해 기존 동급 모델(Qwen2.5-1.5B, Llama3.2-1B)을 상회하는 성능을 달성했다.  
주요 기여:
- **다단계 데이터 혼합**(web, 코드, 수학, instruction-following)과 **수동 재조정**으로 최적의 데이터 배합 탐색  
- **FineMath**, **Stack-Edu**, **SmolTalk** 등 새로운 고품질 도메인별 데이터셋 구축  
- 데이터 중심(on-the-fly) 리밸런싱을 통한 비용-성능 트레이드오프 최적화  
- 2k→8k 토큰 장문(context) 확장 후에도 성능 손실 최소화[1]

# 문제 정의 및 제안 방법

## 해결하고자 하는 문제  
소형 LM은 파라미터 및 컴퓨팅 자원이 제한되어 있어,  
- 훈련 데이터 노이즈 및 품질 민감도↑  
- 수학·코드·추론 성능 저하  
- 대규모 모델 대비 일반화 능력 부족  

## 제안 방법  
SmolLM2는 **4단계 다중 스테이지** 훈련 스케줄과 **온더플라이 데이터 믹싱**을 적용한다:[1]

1. **Stage 1 (0–6T 토큰)**:  
   - Web: FineWeb-Edu: DCLM = 60:40  
   - 코드: StarCoderData 10%  
   - 수학: 제외  

2. **Stage 2 (6–8T)**:  
   - Web 유지(60:40)  
   - 코드↑20%, 수학(OWM) 5%  

3. **Stage 3 (8–10T)**:  
   - Web: FineWeb-Edu:DCLM = 40:60  
   - 코드: Stack-Edu  
   - 수학: OWM + InfiMM-WebMath  

4. **Stage 4 (10–11T, decay)**:  
   - 수학: FineMath₄ + Infi-WebMath₃ 합 14%  
   - 코드: Stack-Edu 24%  
   - Web: DCLM 중심 58%  
   - Cosmopedia v2 4%  

학습률 스케줄: WSD(SGD with Warmup–Stable–Decay)  
Optimizer: AdamW(β₁=0.9,β₂=0.95)  

### 수식  
모델 파라미터 θ 업데이트:  

$$
\theta_{t+1} = \theta_t - \alpha_t \, \nabla_\theta \mathcal{L}(\theta_t)
$$  

여기서 학습률 $$\alpha_t$$는 WSD 스케줄에 따라  

$$
\alpha_t = 
\begin{cases}
\frac{t}{T_{\text{warmup}}}\alpha_{\max}, & t \le T_{\text{warmup}}\\
\alpha_{\max}, & T_{\text{warmup}} < t \le T_{\text{total}}(1-\delta)\\
\alpha_{\max}\left(1 - \frac{t - T_{\text{total}}(1-\delta)}{\delta \, T_{\text{total}}}\right), & t > T_{\text{total}}(1-\delta)
\end{cases}
$$  

($$\delta=0.1$$, $$\alpha_{\max}=5\times10^{-4}$$ , $$T_{\text{warmup}}=2000$$ steps). [1]

## 모델 구조  
LLaMA2 기반 Transformer  
- 레이어: 24  
- 차원: d=2048, FFN=8192  
- 헤드: 32  
- RoPE positional embeddings  
- 토큰 길이: 2k→8k (Stage 4+연장)  

# 성능 향상 및 한계

## 성능 향상  
- **기초 모델**: HellaSwag 68.7 vs. 61.2(Llama3.2) /66.4(Qwen2.5)  
- **수학**: GSM8K 31.1 vs. 7.6/61.7, MATH 11.6 vs. 3.3/34.3  
- **코드**: HumanEval 22.6 vs. 18.9/37.2  
- **일반화**: MMLU-Pro(held-out) 19.4 vs. 11.7/13.7[1]
- **장문**: 8k context 확장 후 성능 저하 거의 없음  

## 한계  
- 수학·코드 최고 성능(대형 모델급) 미달: 토큰 규모 한계  
- 스테이지 간 “loss spike” 관찰(원인 미확인)  
- 온더플라이 리밸런싱은 수동 개입 필요, 자동화 어려움[1]

# 일반화 성능 향상 가능성

- **데이터 다양성**(web, math, code, instruction)과 **고품질 필터링**이 소형 모델의 용량 한계를 넘어서는 일반화 능력을 제공  
- **FineMath**: 고품질 리즈닝 중심 수학 데이터로 MATH 벤치 대폭 상승(6×)[1]
- **Stack-Edu**: 교육용 코드 데이터로 MultiPL-E 24.8→25.6(파이썬)[1]
- **SmolTalk**: 대규모 instruction-following 데이터로 IFEval 56.7점, MT-Bench 6.13점 달성[1]
- 이상 데이터 중심 전략은 용량 제약을 보완하며, **다양한 도메인**으로 일반화 성능을 확장 가능

# 향후 연구 영향 및 고려 사항

SmolLM2는 **데이터 중심 소형 LM 개발**의 청사진을 제시한다.  
- **자동화된 데이터 리밸런싱** 알고리즘 개발  
- **데이터 반복** 최소화 및 확장된 필터링 기법 강화  
- **멀티모달·장문 학습**을 위한 신규 데이터셋 구축  
- **비용 최적화**와 **에너지 효율**을 고려한 소형 모델 설계  

이러한 방향은 연구자들에게 소형 LM의 **접근성**, **실용성**, **확장성**을 동시에 향상시키는 기반을 제공할 것이다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/30813756-a326-4bcc-bdd1-479a0a5f8382/2502.02737v1.pdf)
