# PaLM: Scaling Language Modeling with Pathways

## 1. 핵심 주장 및 주요 기여
PaLM(Pathways Language Model)은 5400억 개의 매개변수를 가진 대규모 Dense Transformer 언어 모델로, Pathways 시스템을 통해 7800억 토큰 규모의 고품질·다양성 있는 웹·도서·위키피디아·코드·소셜미디어 데이터를 단일 에포크로 학습함으로써 다음과 같은 성과를 보인다.

- Few-shot 학습: 29개 영어 자연어 이해·생성 벤치마크 중 28개에서 최첨단 성능 달성.  
- BIG-bench: 150개 이상의 새로운 난제형 언어 과제에서 5-shot 기준으로 평균 인간 성능(평균 인간보다 높거나 유사) 상회.  
- 추론 능력: Chain-of-Thought(자연어 추론 과정 생성) 기법과 결합 시 수학·상식 추론 벤치마크에서 도메인 특화 튜닝 없이도 최첨단 성능 달성.  
- 코드 생성·수정: HumanEval, MBPP, TransCoder, DeepFix 등 코드 과제에서 few-shot 및 fine-tuning 성능 모두 최상위권.  
- 다국어·번역·요약: 영어 외 다국어 번역·요약·질의응답에서 대등하거나 우수한 성능.  
- 효율성: 두 개의 TPU v4 팟(총 6144칩)에서 MFU(Model FLOPs Utilization) 46.2% 달성(기존 대비 대폭 향상).  

## 2. 해결 과제 및 제안 방법

### 2.1 해결 과제  
- Few-shot 학습 성능 한계  
- 대규모 모델 훈련 인프라의 확장성·효율성 문제  
- 긴 추론 과정이 필요한 복합 추론 과제의 해결  

### 2.2 제안 방법  
1. **Pathways 시스템**  
   - 6144개 TPU v4 칩 분산 훈련: 데이터·모델 병렬화, 크로스-팟 그라디언트 통신 최적화  
2. **데이터 혼합 및 정제**  
   - 7800억 토큰: 웹 50%, 웹페이지 27%, 도서 13%, GitHub 코드 5%, 위키·뉴스 5%  
   - 중복 제거·품질 점수 기반 샘플링  
3. **모델 아키텍처**  
   - 118개 레이어, 48개 헤드, d_model=18432, SwiGLU 활성화·Parallel Layer·RoPE 위치 임베딩  
4. **훈련 전략**  
   - Adafactor 옵티마이저, 점진적 배치 증가(512→2048)  
   - Rematerialization 사용→큰 배치에서도 메모리 절약  
5. **추론 기법**  
   - Chain-of-Thought prompting: few-shot 예시로 중간 추론 단계 자연어 표현→최종 답 정확도 대폭 향상  

##### Chain-of-Thought 예시 (GSM8K)  
```
Q: Roger has 5 tennis balls… How many now?  
A: Roger started with 5 balls. 2 cans×3balls=6. 5+6=11. The answer is 11.
```

## 3. 모델 구조 및 수식  
- 디코더 전용 Transformer 블록:  
  $y = x + MLP(LN(x)) + Attention(LN(x))$  
- RoPE(회전 위치 임베딩), Multi-Query Attention, 입력·출력 임베딩 공유  
- Hidden 차원 d_model=18432, 피드포워드 차원 4d_model  

훈련 손실:  

$$
L = -\frac{1}{N}\sum_{i=1}^N \log p(x_i|x_{ < i}) + \lambda_{\mathrm{zloss}}\log^2 Z
$$  

여기서 $$Z$$는 softmax 정규화 상수, $$\lambda_{\mathrm{zloss}}=10^{-4}$$.

## 4. 성능 향상 및 평가 결과

| 과제                            | PaLM-540B | 기타 최첨단 |
|---------------------------------|-----------|-------------|
| 영어 NLP 29개 벤치마크 (few-shot)| 28/29 ↑   | –           |
| BIG-bench (150개 과제,5-shot)   | 인간 평균 상회 | –           |
| 추론 과제 (GSM8K, SVAMP…)        | SOTA      | 도메인별 튜닝 필요 |
| 코드 (HumanEval pass@1=36%)     | pass@100=88% | Codex 12B pass@100=72% |
| 기계 번역 (en→de BLEU=31.8)     | BLEU=31.8 | Supervised BLEU=41.2 |
| 다국어 요약 (MLSum de ROUGE-2)=12.8 | ROUGE-2=12.8 | T5-XXL=36.4  |

- **효율성**: MFU 46.2% (HFU 57.8%), 토큰당 3.28 TFLOP(No-attn)+0.82 TFLOP(attn).  
- **확장성**: 540B→8B까지 로그 선형 성능 향상, 일부 과제에서는 규모 도약(crossover) 현상 관찰.  
- **메모리화 가능성**: 50토큰 연속생성 완전 일치 비율 약 2.4%→로그 선형으로 모델 크기에 비례.

## 5. 일반화 성능 향상 관점  
- **규모의 법칙**: 모델 규모와 학습 토큰 수 동시 확장→few-shot 성능 지속 개선.  
- **불연속적 도약(crossover)**: 62B→540B 확장 시 초기 규모 확장 효과보다 큰 성능 점프 관찰, 새로운 능력 출현 시점 가시화.  
- **Chain-of-Thought**: 언어 생성 유도가 추론 성능에 직접 기여, few-shot 범용성 확대.

## 6. 향후 연구 영향 및 고려사항  
- **효율적 규모 탐색**: 수조 FLOP 훈련 대비 소모 토큰·모델 규모 간 최적점 연구(Chinchilla 사례)  
- **다양한 아키텍처 결합**: sparsity, retrieval, 장문처리(정확도 vs 효율성)  
- **데이터 다양성·품질**: 저자원 언어·비문어체·코드 등 데이터 비율 최적화  
- **안전성·공정성**: 다국어·비영어권 평가, 다운스트림 적용 맥락별 편향·위험성 분석  
- **추론 해석성**: Chain-of-Thought 강화, 모델 내재 추론 메커니즘 해명 연구

PaLM은 적층적 확장 전략과 혁신적 시스템 설계(Pathways)를 통해 few-shot 범용 언어·코드·추론 모델 개발에 중요한 이정표를 제시했다. 이후 연구에서는 효율-정확도 절충 최적화, 새로운 모달리티 결합, 안전·공정성 보강에 주목해야 할 것이다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/db94340c-4597-4491-af4e-8982010a8ad9/2204.02311v5.pdf
