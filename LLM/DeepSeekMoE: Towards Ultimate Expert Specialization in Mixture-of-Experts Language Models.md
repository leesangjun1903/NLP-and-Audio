# DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models

**핵심 주장**  
DeepSeekMoE는 Mixture-of-Experts(MoE) 아키텍처에서 **전문가 특화(expert specialization)**를 극대화해, 동일한 파라미터 예산 및 계산 비용 하에서 기존 GShard 대비 성능을 크게 향상시킨다.[1]

**주요 기여**  
1. **미세 전문가 분할(Fine-Grained Expert Segmentation)**: FFN 숨김 차원을 $$1/m$$로 축소한 뒤 전문가 수를 $$m$$배로 늘려, 라우팅 가능한 전문가 조합의 수를 기하급수적으로 증가시킨다(예: $$N=16, K=2$$ 일 때 기존 $$\binom{16}{2}=120$$에서 $$\binom{64}{8}\approx4.4\times10^9$$으로 확장).[1]
2. **공유 전문가 격리(Shared Expert Isolation)**: $$K_s$$명의 전문가를 항상 활성화되는 공유 전문가로 분리해 공통 지식을 집중시키고, 나머지 전문가 간의 중복 학습을 방지한다.[1]
3. **대규모 모델링**:  
   - **2B 규모**: DeepSeekMoE 2B는 GShard 2.9B(파라미터 1.5×)와 동등 성능, 동일 파라미터 수의 완전 연결(덴스) 모델 상한선에 근접하는 성능 달성.[1]
   - **16B 규모**: DeepSeekMoE 16B는 LLaMA2 7B 대비 약 40% 계산량으로 동등 성능을 확보하며, Open LLM Leaderboard 상 동일 활성화 파라미터 모델 대비 우수.[1]
   - **145B 예비 연구**: GShard 137B 대비 일관된 우위, DeepSeek 67B(덴스) 대비 28.5%(또는 18.2%) 계산량으로 동등 성능 잠재.[1]
4. **정렬(SFT) 적용**: DeepSeekMoE Chat 16B는 LLaMA2 SFT 7B 및 DeepSeek Chat 7B와 동등 성능을 달성, MoE 모델의 정렬 가능성을 입증.[1]
5. **공개 릴리스**: DeepSeekMoE 16B 체크포인트 공개, 단일 40GB GPU에서 배포·추론 가능.  

***

## 1. 해결하고자 하는 문제  
전통적인 MoE(예: GShard)는 토큰당 상위 $$K$$ 전문가만 활성화함으로써 전문가 간 지식 **하이브리디티(hybridity)**와 **중복성(redundancy)** 문제가 발생해, 전문가의 **전문화(specialization)**를 저해한다.[1]
- **지식 하이브리디티**: 제한된 전문가 수(N=8~16)로 인해 단일 전문가가 다양한 지식을 무작위로 흡수.  
- **지식 중복성**: 서로 다른 전문가가 공통 지식을 중복 학습해 파라미터 효율성 저하.  

***

## 2. 제안 방법 및 모델 구조  

### 2.1 일반 MoE 개요  
Transformer FFN을 $$N$$개 전문가로 대체하고, 각 토큰에 대해 상위 $$K$$ 전문가를 선택해 처리.[1]

$$
h_t^l = \sum_{i=1}^N g_{i,t}\,\mathrm{FFN}_i(u_t^l) + u_t^l
\quad,\quad
g_{i,t} = 
\begin{cases}
s_{i,t}, & s_{i,t}\in \mathrm{TopK}(\{s_{j,t}\})\\
0, & \text{otherwise}
\end{cases}
$$

$$
s_{i,t} = \mathrm{Softmax}_i(u_t^l{}^\top e_i^l)
$$

### 2.2 미세 전문가 분할  
- 전문가 수를 $$mN$$으로 늘리고, 활성화 전문가 수를 $$mK$$로 확장해 동일 연산량 유지.  
- 출력은  

$$
h_t^l = \sum_{i=1}^{mN} g_{i,t}\,\mathrm{FFN}_i(u_t^l) + u_t^l
\quad,\quad
g_{i,t}\in\mathrm{Top}(mK)
$$

- 조합 가능성 대폭 증가해 특화도 향상.[1]

### 2.3 공유 전문가 격리  
- $$K_s$$명의 공유 전문가를 항상 활성화, 나머지 전문가 활성화 수를 $$mK - K_s$$로 감소:  

$$
h_t^l = \sum_{i=1}^{K_s}\mathrm{FFN}_i(u_t^l)\;+\;\sum_{i=K_s+1}^{mN}g_{i,t}\,\mathrm{FFN}_i(u_t^l)+u_t^l
$$

- 공통 지식은 공유 전문가에 집중, 나머지 전문가의 중복 제거.[1]

### 2.4 부하 균형 정규화  
- **전문가 레벨**:  

$$
L_{\mathrm{ExpBal}} = \alpha_1\frac{1}{N'}\sum_{i=1}^{N'}f_iP_i
\quad,\quad
f_i=\frac{N'}{K'T}\sum_t \mathbb{1}(\text{토큰}_t\rightarrow i)
$$

- **장치 레벨**:  

$$
L_{\mathrm{DevBal}} = \alpha_2\frac{1}{D}\sum_{d=1}^D f'_d P'_d
$$

$$\alpha_1\ll \alpha_2$$로 부하 불균형 완화 및 성능 저하 방지.[1]

***

## 3. 성능 향상과 일반화 한계  

### 3.1 성능 비교  
- **2B 모델**: Dense 상한선 근접, GShard×1.5와 동등.[1]
- **16B 모델**: LLaMA2 7B 대비 40% 계산량으로 동등 혹은 상회.[1]
- **145B 예비**: DeepSeek 67B 대비 28.5% 계산량으로 동등, GShard 137B 대비 우수.[1]

### 3.2 전문가 특화 분석  
- **중복성 감소**: 주요 라우팅 전문가 비활성화 시 성능 저하율이 GShard 대비 큼 → 높은 특화도.[1]
- **공유 전문가 중요성**: 공유 전문가 비활성화 시 Pile 손실 1.808→2.414 급등 → 비가역적 역할.[1]
- **지식 획득 효율성**: 활성화 전문가 수 절반(3 vs 7)으로도 GShard 동등 성능 → 조합 유연성에 따른 정확도 향상.[1]

### 3.3 한계  
- **Attention 파라미터 부족**: 다중선택(MC) 과제(MMLU, CEval)에서 경쟁 모델 대비 성능 뒤처짐 → 주로 FFN 기반인 MoE 특성.[1]
- **병렬화 오버헤드**: 과도한 세분화 시 작은 전문가에서의 효율 저하, 대규모에서 균형 요인 중요성 대두.[1]

***

## 4. 일반화 성능 향상 가능성  
- **조합 유연성 증가**: 미세 분할로 토큰별 전문가 조합 선택 폭 확대 → 새로운 도메인 지식 습득 시 특화 전문가의 역할 강화.  
- **공유 전문가 비율 조정**: 실험적으로 1:3 비율이 상용 최적, 도메인별 최적 공유·라우팅 비율 탐색으로 범용성 제고.  
- **부하 균형 최적화**: 장치 간 및 전문가 간 부하 손실 가중치 $$\alpha_1,\alpha_2$$ 튜닝을 통해 대규모 분산 환경에서도 일반화 성능 안정화.  

***

## 5. 향후 연구 영향 및 고려 사항  
- **고도화된 특화**: finer granularity 및 dynamic $$K_s$$ 탐색으로 전문가 특화 극대화 연구 촉진.  
- **범용 MoE 아키텍처**: NLP 외 비전·멀티모달에 적용 시, 공유·세분화 메커니즘 일반화.  
- **효율적 배포**: 단일 GPU 배포가 가능한 경량 MoE 전략 확장, 엣지 환경에서 MoE 모델 활용 가능성.  
- **정렬 및 파인튜닝**: SFT·Instr-Tuning 방법론과 MoE의 결합 연구로 응답 품질·안전성 향상.  
- **윤리·책임**: 수많은 전문가로 인한 편향 감지도구 개발, 공유 전문가의 공통 지식이 잠재적 편향원인지 검증 필요.  

DeepSeekMoE는 MoE 모델이 지닌 잠재적 상한선을 현실화하며, 대규모·고효율 언어 모델 연구에 새로운 기준을 제시한다.[1]

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/8cb0c44d-0cc4-474d-94d6-394eeeab536a/2401.06066v1.pdf)
