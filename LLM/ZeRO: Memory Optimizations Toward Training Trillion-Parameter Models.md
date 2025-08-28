# ZeRO: Memory Optimizations Toward Training Trillion-Parameter Models

## 1. 핵심 주장 및 주요 기여  
ZeRO는 초대형 딥러닝 모델(수십억~수조 매개변수)의 학습을 가능하게 하는 메모리 최적화 기법이다.  
주요 기여:  
- **모델 상태(Model States)**(optimizer 상태, gradient, parameters) 중복 제거를 통해 디바이스당 메모리 사용량을 선형적으로 감소시키는 세 단계(Optimizer State Partitioning, Gradient Partitioning, Parameter Partitioning) 제안  
- **잔여 상태(Residual States)**(activations, 임시 버퍼, 메모리 단편화) 최적화를 위한 Activation Partitioning, 상수 크기 버퍼(Constant-size Buffers), 동적 메모리 재정렬 기법 도입  
- 복잡한 모델 병렬화 없이도 거대 모델을 데이터 병렬만으로 효율적으로 학습 가능  
- 1000여 개 GPU 클러스터에서 1조 매개변수 모델의 학습 가능성을 이론 및 실험으로 입증  

## 2. 상세 설명  
### 2.1 해결하고자 하는 문제  
기존 데이터 병렬(Data Parallelism)은 매개변수·옵티마이저 상태를 모든 장치에 복제하여 메모리 비효율이 크며, 모델 병렬(Model Parallelism)은 통신량 증가와 구현 복잡성 문제를 갖는다. 또한 activations, 임시 버퍼, 메모리 단편화로 인한 추가 메모리 병목이 발생한다.  

### 2.2 제안 방법  
#### 2.2.1 ZeRO-DP: 모델 상태 최적화  
1) Optimizer State Partitioning (Pos)  
   – 옵티마이저 상태(모멘텀·분산치)를 $$N_d$$ 개 파티션으로 분할.  
   – 디바이스당 저장 메모리: $$4\Psi + \frac{K\Psi}{N_d}$$ → 최대 4배 절감  
2) Gradient Partitioning (Pg)  
   – gradient를 대응 파라미터 파티션으로 Reduce-Scatter.  
   – gradient 메모리: $$\frac{2\Psi}{N_d}$$ → 누적하면 최대 8배 절감  
3) Parameter Partitioning (Pp)  
   – 매개변수를 파티션별로 저장, 순전파·역전파 시 파이프라인된 Broadcast 실시.  
   – 전체 메모리: $$\frac{16\Psi}{N_d}$$ → $$N_d$$ 배 절감, 통신량 최대 1.5배  

#### 2.2.2 ZeRO-R: 잔여 상태 최적화  
- Activation Partitioning (Pa): MP 시 복제되는 활성화 출력을 파티션해 저장, 역전파 시 All-Gather로 재구성  
- Constant-size Buffers (CB): 임시 버퍼 크기를 모델 크기와 무관하게 고정하여 메모리 증폭 방지  
- Memory Defragmentation (MD): 장·단기 객체 수명 차이를 이용해 활성화 체크포인트 및 gradient를 미리 예약된 연속 버퍼에 배치  

### 2.3 모델 구조  
ZeRO는 기존 PyTorch DataParallel 또는 Megatron-LM 모델 구조에 삽입만 하면 되며, 모델 코드 변경이 필요 없다.  

### 2.4 성능 향상  
- 400 V100 GPU에서 100B 매개변수 모델 학습 시 15 PFLOPS 집계 처리량(하드웨어 피크의 30%) 달성  
- Megatron-LM 대비 최대 10배 속도 향상, 8배 큰 모델(170B) 학습 가능  
- 64→400 GPU 구간에서 수퍼리니어(speedup >2×) 확장성 관찰  

### 2.5 한계  
- 통신량 1.5× 증가(Pp 단계) 시 네트워크 대역폭 제약 발생 가능  
- Activation offload to CPU(Pa+cpu) 활용 시 PCIe 대역폭에 따른 성능 저하  
- 매우 큰 배치 크기 사용 시 학습 수렴 지연 리스크  

## 3. 모델의 일반화 성능 향상 가능성  
- ZeRO는 메모리 한계로 포기되던 대형 배치 크기를 지원하므로, *대용량 배치 학습*에 따른 *수렴 안정성* 및 *일반화 성능* 개선 여지  
- 특히 Transformer 기반 대형 모델에서 배치 크기 증가가 학습 곡선의 평탄화에 기여할 수 있어, *일반화 오류 감소* 기대  

## 4. 영향 및 향후 고려 사항  
- **영향**: 초대형(수십억~수조 매개변수) 모델 연구를 촉진, 모델 병렬 복잡도 제거로 연구 생산성·재현성 향상  
- **고려 사항**:  
  -  네트워크 아키텍처 계획 시 ZeRO 통신 비용을 반영해야 함  
  -  Activation offload 전략 시 CPU↔GPU 대역폭 한계를 면밀히 평가  
  -  대규모 배치가 학습 수렴에 미치는 영향에 대한 이론·실험적 연구 필요  

---  

ZeRO는 온디바이스 메모리 한계를 획기적으로 확장함으로써 미래의 트릴리언 파라미터 급 모델 학습 인프라 설계에 핵심적 기반을 제공한다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/0f020024-6dfb-44a1-a6f6-543ef5c3102f/1910.02054v3.pdf)
