# DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models

## 1. 핵심 주장 및 주요 기여
DeepSeekMath는 7B 규모의 오픈소스 언어 모델을 대상으로  
– **120B 토큰 규모의 고품질 수학 전용 코퍼스**(DeepSeekMath Corpus) 구축  
– **그룹 상대 정책 최적화(GRPO)** 도입을 통한 강화학습 효율화  
를 통해, 외부 툴킷 및 다수 투표 없이도 MATH 벤치마크에서 51.7% 정확도를 달성하며 GPT-4/Gemini-Ultra 수준에 근접한 성능을 보임을 입증한다.

## 2. 해결 과제
1. 공개된 오픈소스 모델의 수학적 추론 성능 한계  
2. 대규모 수학 전용 데이터 수집의 비용·효율 문제  
3. 강화학습(PPO) 적용 시 메모리 및 계산 자원 부담  

## 3. 제안 방법
### 3.1 DeepSeekMath Corpus 구축
– Common Crawl에서 fastText 기반 분류기로 수학 관련 웹페이지 35.5M개(120B 토큰) 선별  
– 도메인·URL 패턴을 통한 반복적 수집·정제  
– 벤치마크 오염 방지를 위해 10-그램 중복 필터링 적용  

### 3.2 모델 구조 및 학습
- **Base 모델**: DeepSeek-Coder-Base-v1.5 7B → 500B 토큰 추가 학습  
  - 56% DeepSeekMath Corpus, 20% GitHub 코드, 10% arXiv, 10% 자연어(CK/EN), 4% AlgebraicStack  
- **Instruction-tuned**: 수학 체인오브토트·프로그램오브토트·툴통합 추론 데이터 776K 예제로 500 step SFT  
- **강화학습**: GRPO로 GSM8K·MATH 질문 144K개, 그룹 크기 G=64, KL 계수 β=0.04, policy LR=1e-6

#### GRPO 핵심 수식
Advantage 계산 없이 그룹 내 보상 평균·표준편차로 정규화한 ˆA를 사용  

```math
J_{GRPO} = \mathbb{E}_{q,\{o_i\}}\frac{1}{G}\sum_{i=1}^G\frac{1}{|o_i|}\sum_{t=1}^{|o_i|}\Big[\frac{\pi_\theta(o_{i,t}|q,o_{i < t})}{\pi_{old}(o_{i,t}|q,o_{i < t})}\hat A_{i,t} - \beta\,D_{KL}(\pi_\theta\|\pi_{ref})\Big]
```

## 4. 성능 향상
| 모델                | MATH (chain-of-thought) | GSM8K (PoT 툴사용) |
|----------------------|----------------------:|-------------------:|
| DeepSeekMath-Base 7B | 36.2%                 | 31.4%              |
| DeepSeekMath-Instruct 7B | 46.8%            | 57.4%              |
| **DeepSeekMath-RL 7B**     | **51.7%**            | **58.8%**          |

- 기존 오픈소스 7B 모델 대비 MATH 절대 +10%p 이상 개선  
- RL 이후 chain-of-thought에서 GPT-4 Code Interpreter 근접 성능  

## 5. 한계 및 일반화 성능
- **형식 수학(기하·정리 증명)**: 삼각형·타원 문제 처리 약함 → 데이터 편향 가능성  
- ** Few-shot 제약**: GPT-4는 few-shot 성능 향상 가능한 반면, DeepSeekMath는 zero/few-shot 유사  
- **Generalization**: Multilingual 코퍼스 덕분에 중국어 벤치마크(CMATH) 성능 +25%p 향상,  
  일반 언어 이해(MMLU+BBH)에서도 +4–5%p 개선  
- **일반화 기회**: GRPO의 그룹 보상 방식은 reward model의 신뢰도·불확실성 반영 시 더 강력해질 여지

## 6. 향후 영향 및 고려사항
- **대규모 도메인 특화 코퍼스**: 웹 크롤링 기반 고품질 데이터 선별 파이프라인이 다른 전문 영역에도 확장 가능  
- **효율적 RL 기법**: GRPO는 value 모형 불필요, 메모리·계산 절감해 대형 모델 RL 실용화 기여  
- **일반화 강화**: reward model의 불확실성 추정 및 트리 탐색 기반 샘플링 도입으로 out-of-distribution 대응력 향상 검토  
- **데이터 편향 주의**: 형식 수학·기하 도메인 추가 수집 및 annotation 강화 필요  

DeepSeekMath는 오픈소스 수학 추론 모델 성능을 한 단계 도약시켰으며, 향후 도메인 특화 학습·효율적 강화학습 연구의 방향타로 작용할 것이다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/e171890e-98b5-43b3-bc89-2249a8f31e74/2402.03300v3.pdf
