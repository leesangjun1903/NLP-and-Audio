# Emergent Abilities of Large Language Models

**핵심 요약:**  
대규모 언어 모델은 매개변수 수 및 계산량이 일정 임계치를 넘어서면 *예측 불가능한 질적 성능 향상*(Emergent Abilities)을 보이며, 이는 소규모 모델의 성능 확장 추세를 단순 외삽으로 예측할 수 없다는 점을 강조한다.[1]

## 1. 해결하고자 하는 문제
- 기존 **스케일링 법칙(scaling laws)** 은 모델 규모가 커질수록 성능이 **연속적·예측 가능**하게 향상된다고 가정해 왔지만, 몇몇 과제에서는 작고 중간 규모 모델의 선형적 성능 상승이 대규모 모델에서 갑작스런 도약(phase transition)을 보이는 비연속적 현상을 설명하지 못함을 지적한다.[1]

## 2. 제안하는 방법
- **Emergent Abilities 정의:**  
  “작은 모델에서 관측되지 않다가, 일정 스케일(계산량 FLOPs 또는 파라미터 수)을 넘으면 비약적으로 성능이 향상되는 능력”으로 규정.
- **측정 절차:**  
  - 가로축: 모델 훈련에 사용된 FLOPs  
  - 세로축: downstream task 성능(정확도, BLEU, Exact Match 등)  
  - 임계 스케일 이전에는 무작위(random) 수준, 이후에는 유의미한 상승을 관측  
- **분석 대상:**  
  - Few-shot prompting (BIG-Bench, MMLU, TruthfulQA 등)  
  - Augmented prompting 전략(chain-of-thought, instruction tuning, scratchpad 등)  
- **수식 예시:**  
  - Emergence 임계점 $$S_c$$ 정의:  

```math
      \forall S < S_c,\quad \mathrm{Perf}(S) \approx \mathrm{Perf}_{\mathrm{random}},
      \quad
      \forall S > S_c,\quad \mathrm{Perf}(S) \gg \mathrm{Perf}_{\mathrm{random}}.
```

[1]

## 3. 모델 구조 및 실험 결과
- **대상 모델:** GPT-3, LaMDA, Gopher, Chinchilla, PaLM 등  
- **Few-shot prompting emergent 사례:**  
  - 3자리 산술 연산: GPT-3 13B 매개변수 이후 급등  
  - MMLU: Dense 모델 70B–280B 매개변수 임계점에서 무작위→50% 이상 도약  
- **Augmented prompting emergent 사례:**  
  - Chain-of-Thought: 100B 매개변수 이상에서만 다중 단계 추론 능력 향상  
  - Instruction Tuning: 68B 매개변수 이상에서만 효과  
  - Scratchpad(중간 단계 출력 학습): 40M 매개변수 이상에서만 실행 가능  
- **성능 향상:**  
  - Emergent 시점 이후, 기존 최고 SOTA 기법보다 수십 퍼센트포인트 높은 정확도 달성  
- **한계:**  
  - Emergence 임계 스케일 예측 불가  
  - 임계점 이하 모델은 해당 능력 전혀 획득 못함  
  - Emergence가 꼭 최종적 성능 정점은 아니며, 추가 스케일링 시에도 불확실성 존재  

## 4. 일반화 성능 향상 관점
- Emergence는 특정 과제 성능 뿐 아니라 **다양한 도메인·언어·멀티태스크** 능력에도 공통적으로 관찰된다.  
- **WikiText103 Perplexity** 등 대체 척도에서도 Emergence가 유사하게 나타나, 모델의 *내재적 일반 언어 이해 능력*이 임계점을 넘을 때 비약적으로 향상됨을 시사.[1]
- 그러나 Retrieval-Augmented, Sparse MoE 등 차별적 아키텍처는 더 낮은 FLOPs에서도 동일한 perplexity를 달성해, 구조적 개선이 Emergence 임계점을 낮출 잠재력 보유.

## 5. 향후 연구 및 고려 사항
- **임계 스케일 예측 기법 개발:** Emergence 발생 규모를 사전 예측할 수 있는 이론·실험 프레임워크 필요  
- **아키텍처·데이터·학습 절차 개선:** Dense Transformer 한계를 넘어 Sparse Expert, 외부 메모리, 목표지향적 데이터 수집 등으로 Emergence 문턱 저감  
- **평가 지표 다양화:** 정확도 기반 지표가 놓치는 개선을 포착할 수 있는 cross-entropy, BLEURT 등 신뢰성 높은 메트릭 확립  
- **Emergent Risk 대응:** 성능뿐 아니라 편향·독성·사기성 콘텐츠 등 *Emergent Risks*도 동시에 모니터링하고 완화 전략 병행  
- **커뮤니티 접근성:** 고사양 자원이 없는 연구자도 Emergent Abilities 연구에 참여할 수 있도록 개방 데이터셋·모델 경량화 연구 장려  

**결론:** Emergent Abilities는 언어 모델이 단순 스케일 확장 이상으로 *질적 도약*을 보이는 현상을 규명하며, 차세대 모델 설계·훈련·평가 패러다임 전환의 핵심 인사이트를 제공한다.[1]

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/1770378e-45d5-40b6-9963-112a7f146306/2206.07682v2.pdf)
