# Free Process Rewards without Process Labels

## 1. 핵심 주장 및 주요 기여  
**핵심 주장**  
- 기존 Process Reward Model(PRM)은 중간 단계별 보상 레이블(step labels) 수집에 막대한 비용을 소비하지만, 결과 레벨(labels on entire responses) 데이터만으로도 동등하거나 더 우수한 PRM을 학습할 수 있다는 **“Implicit PRM”** 개념을 제안한다.  
- 결과 보상을 로지트 비유(log-likelihood ratio) 형태로 파라미터화하면, 같은 학습 과정 중에 중간 단계별 보상을 자동으로 추출할 수 있으며(Prop 3.1), 추가적인 레이블링 비용 없이 과정 보상을 얻는다.

**주요 기여**  
1. 이론적 증명  
   - 보상을 $$r_\theta(y)=\beta\log\frac{\pi_\theta(y)}{\pi_{\mathrm{ref}}(y)}$$로 파라미터화하면, 응답 전체에 대한 로그비가 중간 토큰까지의 기대 보상($$Q$$ 함수)과 일치함을 보임.  
2. 다양한 손실 함수 적용 가능성  
   - DPO, KTO, NCA 뿐 아니라 일반적인 교차엔트로피(CE) 손실에도 동일한 방식으로 적용 가능함을 입증.  
3. 실험적 검증  
   - MATH 데이터셋(best-of-N 평가)에서 Math-Shepherd 같은 강력한 MCTS 기반 PRM 대비 최대 38.8× 적은 FLOPs로 동등하거나 더 높은 정확도를 달성.  
4. 다중 해석  
   - 다수결(majority voting) 결합, 지시문/응답 규모 확장, 레퍼런스 모델 제거 등 다양한 실험을 통해 기법의 실용성을 다각도로 평가.

## 2. 문제 정의, 제안 방법, 모델 구조, 성능 및 한계

### 2.1 해결하고자 하는 문제  
- **과정 보상(PRM) 학습의 높은 비용**: MCTS나 RL 기반 자동 주석 기법으로 단계별 레이블을 생성하는데 모델 플롭스(FLOPs)가 크게 증가.  
- **희소 보상의 비효율성**: 전체 응답 단위 보상(ORM)만으로는 추론 시 응답 재순위화나 RL 안정성에서 한계.

### 2.2 제안 방법  
1. **보상 파라미터화**  
   - Outcome Reward Model(ORM)을 학습할 때 보상을  

$$
       r_\theta(y) \;=\;\beta \log \frac{\pi_\theta(y)}{\pi_{\mathrm{ref}}(y)}
     $$  
     
으로 정의.  

2. **Implicit PRM 추출**  
   - 응답 $$y=(y_1,\dots,y_T)$$의 t번째 토큰까지에 대한 과정 보상은  

$$
       Q_t(y_{ < t},y_t)
       =\sum_{i=1}^t \beta \log \frac{\pi_\theta(y_i\mid y_{ < i})}{\pi_{\mathrm{ref}}(y_i\mid y_{ < i})}
     $$  
     
  로 계산하며, 단계 보상은 $$r_t=Q_t-Q_{t-1}$$으로 정의(Prop 3.1).  

3. **다양한 손실 함수**  
   - DPO, KTO, NCA, 그리고 cross-entropy(CE) 등 여러 알고리즘으로 학습 가능하며, CE 손실의 경우 unpaired/imbalanced 데이터에도 유연하게 작동:  

```math
       \mathcal{L}_{CE}
       =l\log\sigma\bigl(\Delta\ell\bigr)+(1-l)\log\bigl(1-\sigma(\Delta\ell)\bigr),
       \quad \Delta\ell=\beta\log\frac{\pi_\theta(y)}{\pi_{\mathrm{ref}}(y)},
``` 
     
  여기서 $$l$$은 응답의 정답 여부 레이블.

### 2.3 모델 구조 및 학습  
- **정책 모델 $$\pi_\theta$$**: Llama-3.1-8B-Instruct 기반, β=0.05 설정.  
- **레퍼런스 모델 $$\pi_{\mathrm{ref}}$$**: 동일 아키텍처의 사전 학습 모델 또는 RLHF 학습 완료 모델.  
- **학습 데이터**: UltraInteract 수학 지시문 33K개 × 최대 8 솔루션.  
- **실험 설정**: BoN(best-of-N) 재순위화 on MATH-500, 세 가지 생성 모델(Mistral-7B, Llama-3.1-8B, 70B).

### 2.4 성능 향상  
- **개발 비용**: Math-Shepherd 대비 FLOPs 38.8× 절감.  
- **BoN 정확도**: DPO variant 평균 50.4%, CE variant 48.4%로 기존 PRM/ORM 및 AutoPSV 대비 우수(Avg. ≈46–49%).  
- **다수결 결합**: KTO, CE variant가 다수결과 결합 시 성능 대폭 향상.  
- **확장성**: 지시문·응답 수 확대 시 지속적 성능 상승. CE는 극단적 데이터 부족(1개 응답) 상황에서도 개선.  
- **Inference 효율화**: 레퍼런스 모델 생략 시에도 성능 저하 미미—추론 비용 절감 가능.

### 2.5 한계  
- **레퍼런스 모델 의존성**: 학습 과정에는 필요하지만, 성능에는 크게 기여하지 않아 제거 가능.  
- **레이블 노이즈**: MCTS 기반 보상과 일부 실험에서 step-label 노이즈 존재.  
- **정책 성능과 PRM 성능 간 괴리**: PRM으로 뛰어나게 재순위화해도 직접 문제 해결(policy) 성능 향상으로 직결되지 않음.

## 3. 모델의 일반화 성능 향상 가능성  
- **지시문(Task) 다변화 실험**에서는 수학 외 일반 언어 및 코딩 지시문 추가 시 오히려 성능 저하. → **하위 도메인 적합성 중요**.  
- **응답 다양성**은 중복 제거 후에도 성능 유지, 반복 솔루션이 학습에 기여함.  
- **데이터 규모 확대** 시 지시문보다 응답 수 확대가 더 큰 영향.  
- **Alternate Loss(CE)**: 언페어(unpaired)·인밸런스 상황에서도 데이터 효율적, 실제 애플리케이션에서 일반화 잠재력 높음.

## 4. 향후 연구에 미치는 영향 및 고려 사항  
**영향**  
- PRM 구축 비용 장벽 대폭 완화: 다양한 분야의 고난도 단계별 평가(task decomposition) 영역에 즉시 적용 가능.  
- CE 등 비쌍(pairwise) 손실 함수의 실용성 제고: 실제 데이터가 부족한 상황에서도 PRM 강화를 가능케 함.  
- RLHF 이후 모델의 implicit Q 함수 활용 연구 촉진: off-the-shelf 보상 모델 탐구 확산.

**고려 사항**  
1. **도메인 적합성**: 지시문과 응답이 목표 과제와 관련 있어야 일반화 성능 극대화.  
2. **데이터 노이즈 관리**: 자동 step-label 주석의 한계를 극복할 수 있는 견고한 노이즈 완화 기법 필요.  
3. **정책-PRM 간 상호작용**: PRM 향상이 곧바로 정책 성능 향상으로 이어지지 않으므로, 두 역할 간 트레이드오프를 해소하는 추가 연구 요구.  
4. **Inference 효율화**: 레퍼런스 모델 제거 및 경량화 전략을 통한 실시간 응용 가능성 강화.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/e6d66265-e07f-4f18-8bff-c741ed2fb433/2412.01981v1.pdf
