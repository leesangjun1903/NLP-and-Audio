# Measuring Massive Multitask Language Understanding

## 1. 간결 요약
“Measuring Massive Multitask Language Understanding” 논문은  
-  57개의 서로 다른 과목(수학, 과학, 인문, 사회과학 등)에 걸친 다중 선택형 문제 15,908개를 수집하여 대규모 다중 과제 벤치마크를 제안  
-  GPT-3(175B)와 UnifiedQA(11B)를 zero-/few-shot 및 fine-tuning 설정에서 평가하여 기존 소규모 벤치마크 대비 훨씬 낮은 성능을 보임을 입증  
-  벤치마크를 통해 모델의 분야별 지식 격차, 절차적(procedural) vs 선언적(declarative) 지식 문제, 신뢰도 부정확성(calibration error) 등의 주요 한계를 체계적으로 분석  

## 2. 문제 정의
대다수 NLP 벤치마크(GLUE, SuperGLUE, commonsense QA 등)는  
-  특정 분야(언어적 추론, 상식)만 평가하고 빠르게 채워지는 한계  
-  pretrained 모델이 대용량 텍스트에서 얻은 광범위·전문 지식을 평가하지 못함  
따라서 “진정한” 언어 이해 역량을 가늠할 수 있는 다중 도메인, 다양한 난이도의 zero/few-shot 평가 기준이 필요했다.

## 3. 제안 방법
-  벤치마크 구성  
  – 57개 과목 × 100개 이상 시험 예시  
  – 전체 15,908문제를 dev(285), val(1,540), test(14,079)로 분할  
-  평가 설정  
  – Zero-shot: “Answer:” 만 붙여 질문 입력 후 확률 최대 옵션 선택  
  – Few-shot: 과목별 5개 시범 예제 포함  
  – Fine-tuning: UnifiedQA, RoBERTa-base, ALBERT-xxlarge, GPT-2를 dev+val로 학습  

### 수식 및 측정
분류 정확도(Accuracy) = $$\frac{\text{정답 개수}}{\text{전체 문항 수}}$$  
RMS calibration error = $$\sqrt{\frac{1}{N}\sum_{i=1}^N (\text{confidence}_i - \text{accuracy}_i)^2}$$

## 4. 모델 구조 및 학습 방식
-  GPT-3: Transformer 기반 autoregressive 모델, 파라미터 수 2.7B/6.7B/13B/175B  
-  UnifiedQA: T5 텍스트-투-텍스트, 사전학습 후 QA 데이터로 fine-tuning(60M~11B)  
-  기타: RoBERTa-base, ALBERT-xxlarge, GPT-2는 UnifiedQA 형식으로 fine-tuning

## 5. 성능 향상 및 한계
| 모델                         | 전체 평균 정확도 |
|----------------------------|---------------|
| Random baseline            | 25.0%         |
| GPT-3 Small (2.7B, few-shot)   | 25.9%         |
| GPT-3 X-Large (175B, few-shot) | 43.9%         |
| UnifiedQA (11B, fine-tuned)    | 48.9%         |

-  **한계**  
  – **절차적 지식**(계산 중심 STEM)에서 near-random 성능  
  – **사회·윤리 과목**(법·도덕)에서도 매우 낮은 정확도  
  – **신뢰도 불일치**: confidence vs accuracy 차이 최대 24%, RMS error 최대 ~19%  
  – **한 분야도 전문가 수준(≈90%) 미달**: 폭넓지만 얕은 지식 분포

## 6. 일반화 성능 향상 관점
-  **Fine-tuning 효과**: UnifiedQA는 10× 작아도 fine-tune으로 few-shot GPT-3를 능가  
-  **사전학습 데이터 도메인 특화**: 법률 사례, 의학 문헌 등 분야별 데이터 추가 pretraining 시 소폭 개선  
-  **Zero→Few-shot→Fine-tune**: 예시 수 증가 시 정확도 지속 상승(0→5-shot 약 14%p↑)  
-  **Calibration 개선 필요**: 예측 확률과 실제 정확도 일치시키면 일반화 신뢰도 보장

## 7. 향후 연구에 미치는 영향 및 고려 사항
-  **벤치마크 확장**: 멀티모달, 상호작용 과제 포함해 모델의 실세계 지식 적용력 평가  
-  **절차적 능력 강화**: 계산 모듈, symbolic reasoning 결합 연구  
-  **신뢰도 보정**: 온도 조절, 검증 기반 재학습으로 calibration 개선  
-  **데이터 편향·윤리적 측면**: 법·도덕 과제 개선 위한 인간 가치 반영 및 편향 해소  
-  **스케일링 한계**: 파라미터·데이터 확대만으로는 전문 분야 성능 달성 어려움—아키텍처 혁신 필요

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/58298a54-6bd9-4ba5-9330-b8d7c6afffcc/2009.03300v3.pdf)
