# OPT: Open Pre-trained Transformer Language Models

## 1. 핵심 주장 및 주요 기여 (간결 요약)  
OPT 시리즈는 125M에서 175B 파라미터까지 스케일별로 공개된 디코더 전용 Transformer 모델로, GPT-3 급 성능을 재현하면서도  
-  GPT-3 대비 1/7 수준의 탄소 배출량으로 개발  
-  전체 학습 로그와 코드(메타시크)를 투명 공개  
-  125M–66B 모델은 즉시 오픈, 175B 모델은 연구용 비상업 라이선스로 제공  
라는 점에서 ‘대규모 언어모델의 재현성 및 책임 있는 공개’를 실현했다.

## 2. 문제 정의, 제안 방법, 모델 구조, 성능 및 한계

### 2.1 해결하고자 한 문제  
-  제한된 연구 접근성: GPT-3 계열 모델은 API만 일부 공개되어 내부 구조 및 학습 로그 분석이 어려움  
-  높은 개발 비용·환경 비용: 수백만 GPU 시간과 막대한 전력 소모  
-  재현성 및 투명성 부족

### 2.2 제안 방법  
-  **모델 스위트 설계**: 125M–175B 파라미터 디코더 전용 Transformer 8종  
-  **효율적 학습**  
  – Fully Sharded Data Parallel + Megatron-LM Tensor Parallel 활용  
  – Mixed-precision(AdamW+동적 loss scaling)  
  – 학습률 스케줄링: warm-up 후 선형 감쇠, 중간에 안정화 위해 LR 조정 (그림1 참조)  
-  **대규모 코퍼스**  
  – RoBERTa, Pile(선별), PushShift Reddit 통합, 중복 제거(minhash LSH, Jaccard ≥0.95)  
  – 총 1,800억 토큰  
-  **수식**  
  – AdamW 최적화:  

$$ 
      \theta_{t+1} = \theta_t - \eta \Bigl(\frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \theta_t\Bigr)
    $$  
 
  – LayerNorm 초기화: 표준편차 $$\sigma = 0.006$$, 출력층은 $$\sigma \times 1/\sqrt{2L}$$  

### 2.3 모델 구조  
| 모델 크기 | 레이어 수 L | 헤드 수 H | 임베딩 차원 d<sub>model</sub> | 배치(토큰) | LR<sub>peak</sub> |
|---------|----------|---------|----------------------|-----------|---------------|
| 125M    | 12       | 12      | 768                  | 0.5M      | 6e-4          |
| 350M    | 24       | 16      | 1024                 | 0.5M      | 3e-4          |
| …       | …        | …       | …                    | …         | …             |
| 175B    | 96       | 96      | 12288                | 2M        | 1.2e-4        |

### 2.4 성능 향상  
-  **Zero/Few-shot NLP**: GPT-3 대비 평균 정확도 동등 수준(±1%) 유지[Brown et al. 2020]  
-  **대화 모델링**: ConvAI2 등 퍼플렉서티·UF1 유사하거나 상회(비지도 설정)[Roller et al. 2021]  
-  **책임 AI**  
  – 혐오 발언 탐지 F1: Davinci 대비 +8–40% 개선  
  – CrowS-Pairs, StereoSet, RTP 등에서 전반 동등하거나 일부 지표 나쁨 → Reddit 데이터 영향  
-  **환경 효율**: CO₂eq 약 75톤(추정), GPT-3 500톤 대비 1/7  

### 2.5 한계  
-  **편향·독성**: Pile·Reddit 비중으로 인한 차별적·독성 생성 경향  
-  **반복·환각**: 지시문 수행 방식 미숙, 반복 루프, 허위 사실 생성  
-  **인스트럭션 이해 부족**: GPT-3 대비 명령어 직접 수행 취약

## 3. 일반화 성능 향상 가능성  
-  대규모·다양한 코퍼스(소설·위키·소셜미디어)로 훈련하여 다영역 언어 패턴 학습  
-  Mixed-precision·샤딩 기법으로 효율적 자원 활용 → 더 큰 모델·데이터에 접근 용이  
-  Retrieval-augmentation 결합 시 factuality·일반화 강화 여지  
-  중간 LR 조정·그래디언트 클리핑 전략은 안정적 학습에 기여  
→ 다양한 규모 및 도메인 전이 학습(pre-train→fine-tune)에서 추가 이득 예상

## 4. 향후 연구 영향 및 고려 사항  
-  **연구 영향**  
  – ‘재현 가능한 초대형 모델’ 패러다임 제시: 전 세계 연구자들이 동일 실험 재현·비교 가능  
  – 책임 있는 공개 모델 개발 표준 수립 촉진  
  – 환경 비용 투명화: 실제 개발 로그 공유로 지속 가능성 연구 강화  
-  **연구 시 고려점**  
  – **데이터 편향·독성 저감**: 데이터 선별·후처리, unlikelihood 학습, 디버깅 도구 활용  
  – **명령 이해력 개선**: Instruct-tuning, RLHF, 체계적 프롬프트 설계  
  – **환각 방지**: Retrieval-augmented generation, fact-checking 모듈 통합  
  – **탄소 비용 절감**: 효율적 아키텍처·하드웨어, 재생에너지 사용 고려  

---  
위와 같은 기여와 한계를 바탕으로, OPT는 대규모 언어모델 연구의 민주화 및 책임성을 높이는 동시에, 후속 연구를 위한 풍부한 출발점을 제공한다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/a9aea916-9ed0-4f86-8da5-d21c65bc2688/2205.01068v4.pdf)
