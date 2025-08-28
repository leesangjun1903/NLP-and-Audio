# Training Language Models to Follow Instructions with Human Feedback

**핵심 주장 및 주요 기여**  
OpenAI의 InstructGPT 연구는 대규모 언어 모델을 단순히 크기를 키우는 것만으로는 사용자 의도를 충실히 따르도록 정렬(alignment)할 수 없음을 지적하고, 인간 레이블러의 선호도를 보상 신호로 활용하는 강화학습(RLHF: Reinforcement Learning from Human Feedback)을 통해 “도움됨,” “정직함,” “무해함”을 만족하는 모델을 성공적으로 학습시켰다.  
- 1.3B 파라미터 InstructGPT 모델이 175B GPT-3 모델보다 사용자 선호도에서 더 우수함을 보임.  
- RLHF로 정렬(alignment)된 모델이 무작위로 훈련된 대규모 모델보다 진실성·독성 감소 면에서 개선을 달성.  
- 사전학습(pretraining) 혼합 전략(PPO-ptx)으로 RLHF 과정 중 성능 저하를 완화하며, 광범위한 일반화 능력을 확보.  

***

## 1. 해결하고자 하는 문제  
- **언어 모델의 미스얼라인먼트**: 기존 GPT-3 등 대형 모델은 “다음 토큰 예측”만을 목표로 학습되어, 사용자의 실제 의도(“친절하게 답변,” “사실에 기반,” “유해 콘텐츠 회피”)와 충돌  
- **부적절 출력**: 허위 정보 생성(할루시네이션), 독성·편향적 언어 사용, 사용자 지시 미준수  

***

## 2. 제안하는 방법  
1) **Supervised Fine-Tuning (SFT)**  
   - 사람(labeler)이 작성한 바람직한 시연(demonstration) 데이터 약 13k건으로 GPT-3 초기 모델을 지도학습으로 파인튜닝.  
2) **Reward Model (RM) 학습**  
   - 서로 다른 모델 출력(completion) 쌍을 사람 레이블러가 선호도 순위로 라벨링하여 약 33k건의 비교(comparison) 데이터 구축  
   - 보상 모델 $$r_\theta(x,y)$$ 학습:  

```math
       \mathcal{L}(\theta) = -\mathbb{E}_{(x,y_w,y_\ell)\sim D}\Bigl[\log\sigma\bigl(r_\theta(x,y_w)-r_\theta(x,y_\ell)\bigr)\Bigr]
```
     
  여기서 $$y_w$$는 선호되는 출력, $$y_\ell$$는 비선호 출력, $$\sigma$$는 시그모이드 함수.[1]
3) **Reinforcement Learning via PPO**  
   - SFT 모델을 초기 정책으로 사용하고, RM을 보상 함수로 하여 Proximal Policy Optimization(PPO) 적용  
   - 과도한 보상 최적화를 방지하기 위해 per-token KL 벌점 $$\beta\,\mathrm{KL}(\pi_{\text{RL}}\Vert\pi_{\text{SFT}})$$ 도입  
4) **PPO-ptx: 사전학습 혼합**  
   - RL 손실에 사전학습 언어모델 로그우도 보정항을 $$\gamma$$ 계수로 결합하여 alignment tax(성능 저하)를 완화:  

```math
       \mathcal{O}(\phi)=\mathbb{E}_{(x,y)\sim\pi_{\phi}^{\mathrm{RL}}}\bigl[r_\theta(x,y)-\beta\log\frac{\pi_{\phi}^{\mathrm{RL}}(y|x)}{\pi_{\mathrm{SFT}}(y|x)}\bigr]
       +\gamma\,\mathbb{E}_{x\sim D_{\mathrm{pre}}}\bigl[\log\pi_{\phi}^{\mathrm{RL}}(x)\bigr].
```

***

## 3. 모델 구조  
- **기반 모델**: GPT-3 아키텍처(1.3B, 6B, 175B)  
- **Reward Model**: 6B 파라미터 GPT-3 변형, 최종 레이어를 스칼라 보상 출력으로 교체  
- **정책(Value) 모델**: RM 파라미터 재사용, PPO Value 함수 초기화에 활용  

***

## 4. 성능 향상 및 한계  
- **사용자 선호도 평가**: 175B InstructGPT 모델이 175B GPT-3 대비 85% 승률[1]
- **진실성(Truthfulness)**: TruthfulQA 벤치마크에서 GPT-3 대비 진실하고 유익한 답변 비율 2배 증가  
- **독성 감소**: “respectful” 지시 시 RealToxicityPrompts 자동·사람 평가에서 ≈25% 독성 감소  
- **성능 저하(alignment tax) 완화**: 사전학습 혼합(PPO-ptx)으로 SQuAD, DROP 등 public NLP 벤치마크 성능 회복  
- **일반화**: 소수의 비영어 및 코드 질의에도 별도 학습 없이 지시 이행 가능  
- **한계**  
  - 거짓 전제(false premise) 지시 따름, 과도한 답변 회피(hedging), 복잡 제약 이행 실패  
  - 레이블러의 가치·문화 편향에 종속, 고위험 도메인(의료·법률·금융) 적용 시 신중 필요  

***

## 5. 모델 일반화 성능 향상 가능성  
- RLHF가 드물게 관찰된 *비영어*·*코드 요약·질의응답* 과제에 자연정렬을 확장  
- 레이블러 비교 데이터 다변화 및 adversarial 수집으로 거짓 전제·위험한 지시 저항성 향상  
- 제어 코드(control tokens)나 샘플링 시 제약(Steering) 방법과 RLHF 통합으로 steerability 강화  

***

## 6. 향후 연구에 미치는 영향 및 고려 사항  
- **영향**: RLHF가 추후 고급 AI 시스템 정렬(alignment)의 핵심 기법으로 자리매김  
- **고려점**  
  1. **정렬 대상**: 누구의 가치·선호를 반영할지 설계 및 투명성 확보  
  2. **유해 지시 거부**: 사용자의 의도와 상충하는 위험 지시에 대한 거부 메커니즘 구현  
  3. **alignment tax 최소화**: 더욱 다양한 사전학습·동시 최적화 기법 연구  
  4. **책임성과 공정성**: 레이블러·이해관계자 대표성 확충 및 공정한 데이터 수집  
  5. **고위험 도메인 배치**: 의료·법률·재무 등에 적용 시 엄격한 거버넌스 필요  

***

**주석**  
 Ouyang et al., “Training language models to follow instructions with human feedback,” arXiv:2203.02155v1, 2022.[1]

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/a4e55938-d8ed-424b-8a42-d76326b8e3a6/2203.02155v1.pdf)
