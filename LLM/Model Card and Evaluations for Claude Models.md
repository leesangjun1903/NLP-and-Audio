# Model Card and Evaluations for Claude Models

## 1. 핵심 주장과 주요 기여
Anthropic의 “Model Card and Evaluations for Claude Models” 보고서는 Claude 2 모델의 설계 목표, 안전성·정렬(alignment) 평가, 및 능력 평가를 체계적으로 정리했다.  
- **주요 기여**  
  -  Claude 2의 **도움(Helpfulness)**, **정직성(Honesty)**, **무해성(Harmlessness)** 개선을 위한 RLHF와 헌법적 AI(Constitutional AI) 기법 적용  
  -  대규모 인간 피드백 및 외부 레드팀(red teaming)으로부터 얻은 평가 결과 공개  
  -  장문 컨텍스트(최대 200K 토큰)와 다중 언어(200개 이상 언어) 번역 성능 등 새로운 기능 평가  

## 2. 문제 정의, 제안 방법, 모델 구조, 성능 및 한계
### 문제 정의
기존 LLM들은 뛰어난 생성 능력에도 불구하고, 사실 오류(환각), 편향, 해로운 출력 가능성(jailbreak)에 취약하다. 본 보고서는 이러한 한계를 완화하면서도 대화·번역·장문 처리 능력을 개선하려는 목표를 가진다.

### 제안 방법
1) **훈련 데이터 및 구성**  
   - 대규모 비공개·공개 웹 데이터, 라이선스 데이터, 사용자 제공 데이터 및 인간 평가 데이터를 혼합  
   - 데이터 컷오프: 2023년 초  
2) **헌법적 AI(Constitutional AI)**  
   - 모델에 “헌법(Constitution)” 원칙 집합을 내장하고, 유해·비윤리적 질문 필터링  
3) **RLHF 체계**  
   - 인간 선호도 기반의 Elo 스코어를 활용해 **도움·정직·무해** 지표 최적화  
   - 레드팀 공격 시나리오를 포함한 유해성 평가 및 보강  
4) **수식적 평가**  
   - Elo score 비교:  

```math
       \text{Elo}_{2} - \text{Elo}_{1.3} > 0 \quad (\text{helpfulness}, \text{honesty})
```
   
   - BBQ 편향 점수:  

```math
       \text{bias\_score} = \frac{\text{aligned non-Unknown outputs}}{\text{all non-Unknown outputs}}
```
   
   - TruthfulQA 및 HHH 평가 지표 활용  

### 모델 구조
- Transformer 기반, 최대 200K 토큰 컨텍스트 처리 가능한 대규모 언어 모델  
- 파인튜닝: supervised + RLHF + Constitutional AI

### 성능 향상
| 평가 과제            | Claude 1.3 | Claude 2     |
|----------------------|-----------:|-------------:|
| 도움(Elo)           | 기준값       | +0.02 상승   |
| 정직(Elo)           | 기준값       | +0.03 상승   |
| BBQ 편향 점수        | 0.15        | 0.12 감소    |
| GRE Verbal (%tile)   | –           | 95th percentile |
| FLORES200 BLEU       | 평균 35     | 평균 42      |
| USMLE Step 1         | 68.9%       | –            |

### 한계
- 여전히 사실 오류(환각) 및 편향 가능성 존재  
- 고비용의 대규모 컨텍스트 처리  
- 낮은 자원 언어에서 변동이 큰 번역 성능  
- 모델 복잡도 증가에 따른 예측 불가능성

## 3. 일반화 성능 향상 가능성
- **컨텍스트 확대**: 200K 토큰 지원으로 장문 이해·추론 능력 강화  
- **비영어 데이터 비율 증가**: 훈련 데이터 중 약 10% 비영어, 추가 확대 시 저자원 언어 일반화 개선 여지  
- **헌법적 AI 강화**: 다양한 윤리적·문화적 맥락을 반영한 헌법 원칙 추가로 출력을 더욱 정제 가능  
- **메타학습 및 적응**: 소량의 추가 피드백으로 새로운 도메인·사용자 스타일에 빠르게 적응할 수 있는 Few-shot 학습 활용  

## 4. 연구 영향 및 향후 고려 사항
향후 LLM 연구는 **안전성**과 **성능** 간 균형을 맞추는 방향으로 나아가며, CLAUDE 보고서는 이를 위한 구체적 평가 지표 및 프레임워크를 제시했다.  
- **영향**:  
  -  RLHF와 Constitutional AI 결합 사례로, 대규모 모델의 윤리·안전 평가 표준 정립  
  -  장문 컨텍스트 처리 연구 활성화  
- **향후 고려점**:  
  -  지속적 편향·환각 모니터링 및 교정 알고리즘 개발  
  -  저자원 언어 일반화 위한 데이터·피드백 확보  
  -  계산 비용 절감을 위한 경량화·효율화 기법 연구  
  -  다문화·다윤리 환경에서 보편적 헌법 원칙의 확장 가능성 검토

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/3394cf7c-1682-42e1-8018-a9380677077b/Model-Card-Claude-2.pdf)
