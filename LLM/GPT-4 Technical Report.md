# GPT-4 Technical Report

## 핵심 주장과 주요 기여  
OpenAI의 GPT-4 기술 보고서는 **대규모 멀티모달 언어 모델**인 GPT-4의 개발 과정과 성능, 안전·정렬(alignment) 메커니즘을 상세히 기술한다.  
- GPT-4는 이미지와 텍스트를 입력받아 텍스트를 출력하는 트랜스포머 기반 모델로, 변호사 자격시험 등 다수의 전문·학술 벤치마크에서 인간 상위 10% 수준의 성능을 보인다.[1]
- **사전학습(pre-training)** 단계에서는 문서 내 다음 토큰 예측을 수행하며, 이후 **RLHF(Reinforcement Learning from Human Feedback)** 기반 후처리 정렬 과정을 통해 사실성(factuality)과 규범 준수(adherence)를 개선한다.[1]
- 예측 가능한 규모 확장(predictable scaling)을 위해, GPT-4보다 1/1,000~1/10,000 규모의 모델을 활용해 손실(loss) 및 특정 지표를 정확히 예측할 수 있는 인프라와 스케일링 법칙을 제시했다.[2]
- **시스템 카드(systems card)** 형태로 위험(risk) 평가와 완화(mitigation) 전략을 공개하여, 연구·응용 시 안전성과 투명성을 제고한다.[3]

## 1. 해결하고자 하는 문제  
대규모 언어 모델이:
1. 다양한 입력(modalities)에 대응하면서  
2. 전문적·학술적 과제에서 인간 수준의 성능을 달성하고  
3. 허위 생성(hallucination) 및 편향(bias)을 최소화하며  
4. 확장 가능한 인프라 환경에서 예측 가능한 성능 향상을 보장하도록  
개발·배포되는 과정을 기술하는 것이 목표다.

## 2. 제안하는 방법 및 모델 구조  
### 2.1 사전학습 및 토큰 예측  
- GPT-4의 **기본 모델(base model)** 은 거대 인터넷 코퍼스(공개·라이선스 데이터 포함)로 사전학습되며, 다음 토큰 예측(next-token prediction)을 목적으로 한다.[2][1]

### 2.2 후처리 정렬: RLHF  
- 사전학습된 모델은 인간 피드백에 기반한 강화학습(RLHF)을 통해 사용자 의도에 부합하는 응답과 안전한 행동(refusal, 안전 제약 준수)을 학습한다.[1]
- 이 과정에서 안전 관련 프롬프트에 대해 **GPT-4 기반 분류기**가 보상 신호(reward)를 제공하며, 허용/비허용 카테고리 모두에 양·음의 보상 값을 부여해 응답을 학습시킨다.[2]

### 2.3 예측 가능한 스케일링  
- 모델 규모와 학습 컴퓨트(compute) 간 멱법칙(power-law)을 활용해, 소형 모델에서 측정한 손실과 HumanEval 통과율(pass rate) 등을 대규모 GPT-4 수준으로 정확히 예측할 수 있게 한다.[2]
- 예: HumanEval 기준 1/1,000 규모 모델 성능에서 GPT-4 통과율을 추정 가능.

## 3. 성능 향상  
| 벤치마크            | GPT-4 성능 | GPT-3.5 성능 | 최첨단 비교 SOTA |
|--------------------|------------|--------------|------------------|
| MMLU               | 86.4%      | 70.0%        | 75.2%            |
| HellaSwag          | 95.3%      | 85.5%        | 85.6%            |
| AI2 ARC            | 96.3%      | 85.2%        | 85.6%            |
| WinoGrande         | 87.5%      | 81.6%        | 85.6%            |
| HumanEval          | 67.0%      | 48.1%        | 65.8%            |
| DROP (F1)          | 80.9       | 64.1         | 88.4             |
| 변호사시험(시뮬)       | 상위 10%      | 하위 10%       | –                |  
*모든 벤치마크는 GPT-4의 사전학습 후 RLHF 적용 모델 기준.*[2]

## 4. 한계  
- **환각과 오류**: GPT-4는 이전 모델 대비 환각률을 줄였으나 여전히 사실 오류와 논리적 추론 오류가 발생하며, 고위험(high-stakes) 응용에서는 인간 검토·추가 검증이 필요하다.[2]
- **데이터 투명성 부족**: 사전학습 데이터의 구체적 출처·구성은 경쟁·보안 이슈로 대부분 비공개여서, 편향·대표성 문제를 완전 검증하기 어렵다.[3]
- **불확실성 추정 미흡**: 출력의 신뢰도(confidence)·불확실성(uncertainty)에 대한 정량적 표시는 제한적이어서, 사용자가 다양한 과제에 따라 응답 신뢰도를 판단하기 어려운 상황이다.[3]
- **멀티모달 한계**: 이미지 입력 처리 시 간헐적 오류 및 추론 제약이 존재하며, 텍스트와 이미지 정보를 결합해 사용하는 고난도 과제에는 추가 개선이 필요하다.[1]

## 5. 모델의 일반화 성능  
- **다언어 MMLU**: 26개 언어 중 24개 언어에서 영어 기준 GPT-3.5 성능을 능가해, 저자원 언어에서도 강력한 일반화 능력을 보여준다.[2]
- **스케일링 기반 예측**: 소형 모델 실험 결과를 대규모 성능으로 예측하는 스케일링 법칙이 실제 GPT-4 성능과 일치해, 모델 확장 시 일반화 거동을 예측 가능하다.[2]
- **도메인 간 전이**: 변호사·의사 시험, 대학 입시 문제 등 다양한 전문 분야와 학술 평가에서 일관된 고성능을 나타내, 과제별 튜닝 없이도 일반화된 추론 능력을 입증한다.[1]

## 6. 향후 연구 영향 및 고려사항  
### 6.1 연구 영향  
- **멀티모달·대규모 모델** 개발 표준 제시: 텍스트·이미지 통합 처리와 확장 예측 인프라는 차기 모델 아키텍처 및 학습 파이프라인 설계의 핵심 지침이 된다.  
- **정렬·안전 프레임워크** 확장: RLHF 기반 후처리 보상 구조와 시스템 카드를 통한 위험 완화 방안은 안전한 AI 개발 및 규제 가이드라인 수립에 기여한다.  
- **투명한 성능 예측**: 스케일링 법칙을 활용한 성능 예측은 대규모 모델 연구의 비용·시간 절감뿐 아니라, 안전성 평가·리스크 분석에도 활용 가능하다.

### 6.2 향후 연구 시 고려할 점  
- **데이터·편향 투명성 강화**: 사전학습 데이터 셋 구성, 라벨링 메커니즘, RLHF 라벨러 demographics 등 공개를 확대해, 편향 검증과 공정성 보장을 강화해야 한다.  
- **불확실성 정량화**: 응답 신뢰도 및 불확실성 추정을 모델 내부 지표로 제공해, 고위험 응용 시 안전한 배포 전략을 확립해야 한다.  
- **멀티모달 추론 심화**: 이미지와 텍스트 정보의 결합 추론 능력을 높이고, 복합적 시각–언어 과제를 위한 구조적 개선을 모색할 필요가 있다.  
- **다양한 도메인 검증**: 의료·법률·금융 등 실제 고위험 분야에서의 사용자 피드백·추가 벤치마크를 통해, 모델 일반화 및 안전성 보장을 지속 검증해야 한다.

***

GPT-4 기술 보고서는 **고성능·고신뢰 AI** 개발에 필요한 아키텍처, 학습·정렬 파이프라인, 안전 평가 프레임워크, 확장 예측 법칙을 제시하며, 차세대 대규모 언어 모델 연구의 이정표를 세웠다. 미래 연구는 *데이터 투명성*, *불확실성 정량화*, *멀티모달 추론* 및 *실제 도메인 검증*을 통해 이 기준을 더욱 발전시켜야 할 것이다.[1][2]

[1] https://www.semanticscholar.org/paper/163b4d6a79a5b19af88b8585456363340d9efd04
[2] https://openai.com/index/gpt-4-research/
[3] https://pmc.ncbi.nlm.nih.gov/articles/PMC10795998/
[4] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/5a70659b-54fa-4f3e-b498-94b173023222/2303.08774v6.pdf
[5] https://dx.plos.org/10.1371/journal.pdig.0000417
[6] https://arxiv.org/abs/2405.00732
[7] https://www.semanticscholar.org/paper/b5051fbfbbb9bbb4f73898e3e287208cd9726dd6
[8] https://arxiv.org/abs/2412.08905
[9] https://www.jmir.org/2024/1/e52758
[10] https://arxiv.org/abs/2403.17297
[11] https://arxiv.org/abs/2404.07612
[12] https://arxiv.org/abs/2310.18498
[13] http://medrxiv.org/lookup/doi/10.1101/2024.03.22.24304745
[14] https://www.tandfonline.com/doi/pdf/10.1080/27660400.2023.2260300?needAccess=true
[15] https://arxiv.org/pdf/2305.03195.pdf
[16] https://arxiv.org/pdf/2310.17526.pdf
[17] https://arxiv.org/pdf/2311.15732.pdf
[18] https://arxiv.org/pdf/2306.13906.pdf
[19] https://arxiv.org/pdf/2306.09525.pdf
[20] http://arxiv.org/pdf/2401.08396.pdf
[21] https://arxiv.org/pdf/2310.11458.pdf
[22] https://academic.oup.com/jamiaopen/article/doi/10.1093/jamiaopen/ooae060/7705527
[23] https://arxiv.org/pdf/2303.13375.pdf
[24] https://aclanthology.org/2023.emnlp-main.395.pdf
[25] https://arxiv.org/abs/2303.08774
[26] https://www.gpters.org/llm-service/post/gpt4-technical-report-SoHeOZHMqGein4I
[27] https://velog.io/@ttunes2024/GPT-4-Technical-Report-Review
[28] https://mj9245.tistory.com/39
[29] https://velog.io/@nakyung-kim/GPT-4-Technical-Report-%EC%A0%95%EB%A6%AC-%ED%98%84%EC%9E%AC-%EC%83%81%ED%99%A9-%EB%A6%AC%ED%8F%AC%ED%8A%B8-%EB%82%B4%EC%9A%A9-%EC%A0%95%EB%A6%AC-%EC%9D%BD%EC%9C%BC%EB%A9%B4%EC%84%9C-%EC%9E%88%EC%97%88%EB%8D%98-QA
[30] http://ui.adsabs.harvard.edu/abs/2023arXiv230308774O/abstract
[31] https://www.semanticscholar.org/paper/GPT-4-Technical-Report-Achiam-Adler/163b4d6a79a5b19af88b8585456363340d9efd04
[32] http://arxiv.org/pdf/2303.08774.pdf
[33] https://modulabs.co.kr/blog/gpt4-technical-report
[34] https://databoom.tistory.com/entry/%EB%85%BC%EB%AC%B8-%EB%A6%AC%EB%B7%B0-GPT-4
[35] https://chanmuzi.tistory.com/190
[36] https://inspirehep.net/literature/2798025
