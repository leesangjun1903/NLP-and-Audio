# A Survey on Contextual Embeddings

# 핵심 요약 및 주요 기여  

**주요 주장**  
A Survey on Contextual Embeddings는 전통적인 전역 단어 표현(global embeddings)이 문맥에 따른 단어의 다양한 의미 변이를 포착하지 못하는 한계를 지적하며, 문장 내 각 단어에 대해 주변 문맥을 반영한 **컨텍스추얼 임베딩(contextual embeddings)** 이 NLP 전반에 획기적 성능 향상을 가져왔음을 종합적으로 정리한다.[1]

**주요 기여**  
1. 컨텍스추얼 임베딩의 개념 정의 및 수학적 프레임워크 정리  
2. 주요 모델(ELMo, GPT, BERT, XLNet 등)의 사전학습(pre-training) 방법 비교  
3. 다국어 코퍼스에서의 폴리글롯(pre-training) 기법 고찰  
4. 다운스트림 응용: 특징 추출(feature-based), 파인튜닝(fine-tuning), 어댑터(adapter) 방법  
5. 모델 압축(compression) 및 분석(probing, visualization) 기법 정리  
6. 향후 연구 과제와 도전 과제 제시  

# 논문 상세 분석  

## 1. 해결하고자 하는 문제  
전통적 분산표현(word2vec, GloVe)은 단어 유형(type)마다 단일 벡터를 학습하므로 문맥에 따른 다의어(polysemy)나 의미 변이를 반영할 수 없다.  
컨텍스추얼 임베딩은 입력 시퀀스 $$S=(t_1,\dots,t_N)$$ 전체를 고려하여 각 토큰 $$t_i$$에 대해  

$$
h_{t_i} = f(e_{t_1},e_{t_2},\dots,e_{t_N})
$$  

로 표현하며, 이로써 문맥 의존적 표현을 학습한다.[1]

## 2. 제안 방법 및 모델 구조  

### 2.1 언어 모델 기반 사전학습  
- **기본 언어 모델(LM)**  

$$\displaystyle P(t_1,\dots,t_N)=\prod_{i=1}^{N}P(t_i\mid t_1,\dots,t_{i-1})$$[1]

- **ELMo**  
  양방향 LSTM을 사용해 좌·우 문맥 표현을 얻은 뒤,  

$$
  \mathrm{ELMo}_\mathrm{task}^k = \gamma^\mathrm{task}\sum_{j=0}^{L} s_j^\mathrm{task}h_{k,j}
  $$  
  
  로 계층별 출력을 가중 결합해 다운스트림에 활용.[1]
- **Transformer 계열**  
  -  GPT/GPT2: 단일 방향(decoder-only) Transformer + 언어 모델링  
  -  BERT: 양방향 Transformer encoder + Masked LM + Next Sentence Prediction  
  -  XLNet: 순열 기반 언어 모델링(permutation LM)으로 조건부 독립성 가정과 [MASK] 토큰 문제 해결  
  각 모델은 Transformer 구조를 핵심으로 하며, 다양한 사전학습 목표를 도입해 표현력을 극대화.[1]

### 2.2 다국어 및 폴리글롯 학습  
- **XLM / XLM-R**: 병렬 말뭉치에 MLM·TLM(Translation LM) 결합, 대규모 CommonCrawl 데이터 적용  
- **다국어 BERT**: 100개 언어 위키피디아로 사전학습  
- **공유 vs. 비공유 어휘** 실험을 통해 언어 간 암묵적 구조 전이가 가능함을 보임.[1]

## 3. 성능 향상 및 한계  

### 3.1 성능 향상  
- **GLUE 벤치마크**: BERT로 80.5 → RoBERTa로 더 장시간 학습 시 88대  
- **질의응답/요약**: BART·T5로 SQuAD·CNN/Daily Mail 최고 성능 달성  
- **제로샷/적은 학습**: GPT2가 CoQA에서 라벨 없이 F1 55 달성  
- **다국어 전이**: XLM-R이 여러 언어 자연어 추론에서 동급 또는 초월 성능  

### 3.2 한계  
- **계산 자원**: 대규모 모델은 메모리·연산량 과다  
- **파인튜닝 불안정**: Catastrophic forgetting, 최적화 불안정성  
- **해석 가능성 부족**: 학습된 지식 내재화 범위 불명확  
- **견고성**: 적대적 공격(adversarial triggers), 허위 정보 생성 취약.[1]

## 4. 일반화 성능 향상 가능성  
- **어댑터(Adapter)**: 파라미터 고정·소형 모듈로 다중 과제 적응, 전이 학습 효율화  
- **멀티태스크 학습**: MT-DNN, T5 등으로 GLUE 전반 최적화  
- **모델 압축**: 지식 증류(distillation), 저순위 근사, 양자화로 경량화 후 전이 학습  
- **제어된 파인튜닝**: 계층별 고정·학습률 편차 조절, 정규화로 사전학습 지식 보존  
이들 기법은 도메인·언어 전이에 긍정적 영향을 미치며, 일반화 성능을 꾸준히 개선할 잠재력이 크다.  

# 미래 영향 및 고려 사항  

본 서베이는 컨텍스추얼 임베딩 연구를 체계화하여 다음과 같은 영향력을 가질 것으로 예상된다.  
- **새로운 사전학습 목표 개발**: 양적·질적 효율성 모두 충족하는 목표 설계 필요  
- **표현 해석 및 검증**: 제어 과제를 통한 프로빙, 시각화 확장 및 인과관계 증명  
- **견고성 강화**: 적대적 공격 방어 및 윤리적·사회적 영향 최소화  
- **생성 제어 기술**: 도메인 특화·지식 일관성 보장 생성 기법 연구  
- **경량 모델 전이**: 압축·어댑터 방식으로 실무 적용성 높이기  

향후 연구에서는 위 과제들을 통합하며, **효율·해석성·견고성을 균형 있게 갖춘 컨텍스추얼 임베딩** 개발이 필수적이다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/126d7549-34a6-417a-8085-bd1bc36b15ce/2003.07278v2.pdf)
