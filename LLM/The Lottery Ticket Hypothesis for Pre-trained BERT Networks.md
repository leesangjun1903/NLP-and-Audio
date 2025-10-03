# The Lottery Ticket Hypothesis for Pre-trained BERT Networks

# 핵심 요약 및 주요 기여  
**주장:** 사전 학습된 BERT 네트워크에도 초기화 시점에서 학습·전이 가능한 희소(subnetwork) “당첨 티켓(winning ticket)”이 존재한다는 점을 제시한다.[1]
**기여:**  
- 사전 학습(pre-trained) 초기화 시점($$t_0$$)에서 40–90% 희소도 수준의 matching subnetworks를 발견.  
- MLM(마스크 언어 모델링) 기반 희소 네트워크는 모든 다운스트림 작업에 전이 가능한 범용(universal) subnetworks임을 입증.  
- 반복적 크기 축소(Iterative Magnitude Pruning, IMP)에서 되감기(rewinding)를 수행할 필요 없이 초기화 시점에서 직접 winning tickets를 찾을 수 있음을 확인.  
- 전통적 표준 프루닝 방법과 성능 견주어 동등하거나 우수함을 보임.  

# 문제 정의  
사전 학습된 대규모 BERT 모델은 수억 개 매개변수로 구성되어 비용이 크고 배포가 어려움.  
- **해결 과제:** BERT를 대체할 수 있는, 초기화부터 단독으로(full model과 동등하게) 학습 가능한 작은 subnetworks를 발견하여 비용·에너지 소모를 절감.  
- **Lottery Ticket Hypothesis:** 원 논문에서 제안된, 훈련 가능하고(full accuracy), 전이 가능한(transferrable) 경량 subnetworks 존재 가설.  

# 제안 방법  
1. **Subnetworks 정의**  
   - 모델 $$f(x;\theta)$$에 대해 마스크 $$m \in \{0,1\}^d$$를 적용한 $$f(x; \theta \odot m)$$.  
   - *Matching subnetwork:* 훈련 알고리즘 $$A_{T,t}$$ 수행 후 성능이 원본과 동등 이상.  
   - *Winning ticket:* 초기화 값 $$\theta_0$$와 mask $$m$$ 모두 중요, 즉 $$\mathcal{T}(A_{T,t}(f(x;\theta_0 \odot m))) \ge \mathcal{T}(A_{T,t}(f(x;\theta_0)))$$.  
   - *Universal subnetwork:* 여러 작업 $$\{T_i\}$$에서 모두 matching인 단일 $$m$$.  

2. **Iterative Magnitude Pruning (IMP) 알고리즘**  
   - (1) 사전 학습된 초기화 $$\theta_0$$에서 fine-tune하여 $$\theta_t$$ 획득  
   - (2) 가중치 절댓값 기준 하위 10% pruning → mask 업데이트  
   - (3) 되감기(rewind) 없이 초기화 $$\theta_0$$로 재설정  
   - (4) (2)-(3) 반복하여 목표 희소도 $$s$$ 달성  
   - **수식:**  

$$ m^{(k+1)} = \text{TopK}(|\theta^{(k)}|, (1-s)\cdot d), \quad \theta^{(k+1)} = \theta_0 $$  
     
  여기서 TopK는 절댓값 기준 상위 K개의 가중치를 남기는 연산.  

4. **전이 실험**  
   - 다운스트림 작업별로 희소도 70% IMP subnetworks를 확보  
   - 이를 다른 작업에 fine-tune하여 전이 성능 평가  
   - MLM 기반 subnetworks는 모든 작업에 대해 full BERT 성능 이내 유지 → 범용성 확인.[1]

# 모델 구조 및 성능  
- **모델:** BERTBASE (12-layer Transformer, 110M 파라미터) + 태스크별 분류 헤드  
- **데이터셋:** 9개 GLUE 작업 + SQuAD v1.1 + MLM 사전 학습 태스크  
- **성능 요약:**  
  - **40–90% 희소도**에서 full BERT와 동등 성능 달성  
  - 임의(pruning/random reinit) 대비 15–21%p 성능 우수  
  - 되감기 없이 초기화만으로 winning tickets 발견 (기존 연구 대비 혁신적)  
  - 표준 프루닝과 비교 시 동등하거나 일부 태스크에서 최대 3%p 우수  

# 제한점  
- 다운스트림 데이터셋 크기 작을수록 과적합 위험  
- 표준 프루닝 대비 일부 소형 태스크에서 성능 저하 관측  
- 하드웨어 종속적 속도 개선 효과 및 컴퓨팅 리소스 절감량 미검증  

# 일반화 성능 향상 가능성  
- **범용 subnetworks:** MLM IMP mask는 모든 다운스트림 작업에 일관된 성능 보장 → 초기화 시점 희소화만으로 강력한 전이 가능.[1]
- **데이터셋 크기 의존성:** 큰 태스크(MLM, MNLI, SQuAD)에서 subnetworks 전이 성능 우수. 이는 모델 초기화가 풍부한 표현을 담고 있을 때 희소화가 더욱 일반화에 유리함을 시사.  
- **되감기 불필요성:** 중간 훈련 단계 되감기 없이 초기화에서 바로 subnetworks 발굴 → 일반화 성능 보존 및 추가 계산 단계 제거.  

# 향후 연구 영향 및 고려 사항  
- **2차 사전 학습 파이프라인:** 초기 사전 학습 후 IMP 기반 희소화 → 효율적 경량 BERT 구축 가능성  
- **하드웨어 최적화:** XNNPACK 등 희소 행렬 최적화 라이브러리와 결합하여 실제 훈련·추론 속도 개선 연구 필요  
- **다중 태스크 Pruning:** MLM+다운스트림 동시 IMP 실험에서 범용성 추가 향상 관찰되지 않음 → multi-task 정보 활용 방법 모색  
- **안정성·강건성 평가:** 분포 이동, adversarial 환경에서 범용 subnetworks 일반화 강인성 검증  
- **모델 구조 확장:** 대형 Transformer, GPT 계열에서도 초기화 희소화 가능 여부 및 한계 탐색  

이 연구는 대규모 사전 학습 모델의 **컴퓨팅·에너지 효율화**를 위한 새로운 패러다임을 제시하며, 향후 효율적 모델 배포 및 확장 연구에 중대한 기여를 할 것으로 기대된다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/5088eb4d-9fb1-48c7-8e73-659dafd916e0/2007.12223v2.pdf)
