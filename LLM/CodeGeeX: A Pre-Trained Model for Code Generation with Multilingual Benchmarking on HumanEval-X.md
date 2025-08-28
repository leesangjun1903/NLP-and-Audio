# CodeGeeX: A Pre-Trained Model for Code Generation with Multilingual Benchmarking on HumanEval-X
# 핵심 요약 및 주요 기여

**CodeGeeX**는 23개 프로그래밍 언어를 아우르는 13억 매개변수 규모의 다국어 코드 생성 모델로, 오픈소스로 공개되어 연구자와 개발자가 자유롭게 활용할 수 있습니다.  
주요 기여는 다음과 같습니다.  
- 대규모 다국어 코드 데이터(8500억 토큰)를 이용한 **13B 파라미터** 모델 사전학습 및 오픈소스 공개  
- Python 외 C/C++/Java/JavaScript/Go로 확장한 **HumanEval-X** 벤치마크(164문제×5언어)를 수작업으로 제작  
- 주요 다국어 모델(Incoder, CodeGen-Multi 등) 대비 **일관된 성능 우위** 입증 (평균 pass@100: 54.76%)  
- VS Code·JetBrains·Cloud Studio 확장플러그인 제공을 통한 **실제 사용성 검증** (사용자 83.4% 생산성 향상)  

# 문제 정의 및 제안 방법

## 해결하고자 하는 문제  
1) 기존 벤치마크(HumanEval, MBPP 등)는 **단일 언어(Python)에** 한정되어 다국어 코드 생성 역량을 평가할 수 없음  
2) 다국어 모델들은 BLEU 기반 문자열 유사도 평가에 의존해 **기능적 정확도**를 평가하지 못함  

## 모델 구조  
- GPT 계열의 **39-layer Transformer 디코더**(hidden size=5120, heads=40)  
- 입력: `[언어 태그] + 코드/문서` 토큰화 → 위치 임베딩 합산 → 순차적 self-attention  
- FastGELU 활성화, LayerNorm, Residual 연결  
- 상단에 **쿼리 레이어(Query Layer)** 추가(다음 토큰 예측용)  

## 학습 목표 및 수식  
- **Autoregressive LM**: 입력 $$x_1,\dots,x_n$$에 대해  

$$
    \mathcal{L} = -\sum_{i=1}^{N-1} y_{i+1} \log P(x_{i+1}\mid x_1,\dots,x_i;\Theta)
  $$  

- **INT8 양자화**($$W_q = \mathrm{Round}(W/\lambda)$$, $$\lambda=\max|W|/(2^{b-1}-1)$$) 후 FasterTransformer로 추론 최적화  

# 성능 향상 및 한계

## 성능 향상  
- **Code Generation**: 평균 pass@100 기준 주요 6개 모델 중 최고(54.76%), CodeGen-Multi-16B 대비 +0.37%p  
- **Code Translation**: XLCoST 벤치마크에서 프로그램 단위 CodeBLEU +4.10p 개선  
- **한계**:  
  - **언어별 불균형**: Python 중심 데이터 분포(26.7%)로 Python 성능이 우수하나, Go 등 소수 언어는 제약 발생  
  - **추론 오류**: 논리적 오류(incorrect logic) 비율 높고, Go 구문 검증 Strictness로 syntax error 다수  
  - **일반화**: 동일 문제 여러 언어에서 일관된 성공률 확보 어려움(언어별 편차 큼)  

# 일반화 성능 향상 관점

- **다국어 예제 조합**을 통한 예제 편향 완화 연구 필요  
- **Chain-of-Thought** 스타일 프롬프트로 추론 단계 명시 → 논리적 오류 감소 가능성  
- **모델 용량 확장** 및 **도메인 특화 파인튜닝**(예: 프로덕션 코드)으로 언어 간 지식 전이 강화  

# 향후 영향 및 고려 사항

CodeGeeX는 **다국어 코드 생성·번역 연구를 위한 공개 인프라**를 제공하며, 후속 연구에서 다음을 고려해야 합니다.  
- **데이터 다양성**: 소수 언어·도메인별 고품질 코드 수집으로 일반화 역량 제고  
- **Few-Shot 학습**: 비용 효율적 프롬프트 기법(Chain-of-Thought 등) 탐색  
- **안정성 검증**: 자동화된 기능 시험(test harness) 확장으로 논리·보안 결함 조기 탐지  
- **생산 환경 적용**: 실서비스 코드 스타일·최적화 요구 반영한 전용 파인튜닝 파이프라인 구축  

이를 통해, CodeGeeX는 다국어·다목적 코드 생성의 **기초 벤치마크**로 자리매김하고, 실용적 AI 개발 환경 조성에 기여할 것입니다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/c01d7d8c-8d5f-49d6-bd24-3ef09cc48899/2303.17568v2.pdf)
