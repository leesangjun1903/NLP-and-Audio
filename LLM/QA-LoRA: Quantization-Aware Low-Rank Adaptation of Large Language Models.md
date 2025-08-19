# QA-LoRA: Quantization-Aware Low-Rank Adaptation of Large Language Models

## 1. 핵심 주장 및 주요 기여  
**QA-LoRA**는 대형 언어 모델(LLM)의 파인튜닝과 배포 과정에서 발생하는 계산·메모리 부담을 동시에 해결하기 위해 고안된 기법이다.  
- **주장**: 양자화(Quantization)와 로우랭크 어댑테이션(LoRA)을 결합함으로써, 파인튜닝 중과 추론 시 모델의 가중치를 저비트 정수(INT2–4) 형태로 유지하면서도 기존 대비 정확도를 유지하거나 향상시킬 수 있다.[1]
- **기여**:  
  1. 그룹 단위 양자화(group-wise quantization)와 그룹별 LoRA 파라미터 공유를 통해 양자화 자유도와 어댑테이션 자유도의 균형을 맞춤.  
  2. 파인튜닝 시 INT4 이하의 비트로 학습하면서 FP16 수준의 정확도를 확보.  
  3. 학습 후에도 고비트 재변환 없이 INT4 모델을 직접 배포하여 추론 속도를 2배 이상 가속.  

## 2. 해결 과제와 제안 방법  

### 2.1 해결 과제  
- **메모리·속도 병목**: LLaMA-65B 등 대형 모델은 파인튜닝과 추론 시 막대한 GPU 메모리와 연산을 요구.  
- **양자화 후 성능 저하**: 저비트로 포스트 훈련 양자화(Post-Training Quantization, PTQ) 시 성능 손실이 크고, LoRA 결과를 병합하면 FP16으로 복귀되어 연산 이득이 사라짐.  

### 2.2 제안 방법  
#### 그룹별 양자화 및 어댑테이션  
- 입력 차원 $$D_{\text{in}}$$을 $$L$$개의 그룹으로 분할하여 각 그룹에 독립적 스케일 $$\alpha_{l,j}$$, 제로점 $$\beta_{l,j}$$를 계산.  
- LoRA 어댑터 행렬 $$A\in\mathbb{R}^{L\times D_{\text{int}}}$$, $$B\in\mathbb{R}^{D_{\text{out}}\times D_{\text{int}}}$$로 설계하여 그룹 내 파라미터를 공유함으로써 어댑테이션 자유도를 $$L\times D_{\text{int}}+D_{\text{int}}\times D_{\text{out}}$$으로 축소.  
- **전달 식**:  

$$
  \widetilde W = \bigl[\alpha_{l,j}\hat w_{i,j} + \beta_{l,j}\bigr],\quad
  y = x^\top \widetilde W + s(QA(x))^\top A\,B^\top,
  $$  
  
  여기서 $$QA$$는 길이 $$\frac{D_{\text{in}}}{L}$$로 평균 풀링하는 연산.  
- **병합 규칙**:  

$$
  \beta'\_{l,j} = \beta_{l,j} - s\frac{(B\,A^\top)\_{l,j}}{\alpha_{l,j}},
  $$  
  
  로 제로점을 업데이트하여 양자화 특성을 유지한 채 어댑터를 통합.[1]

## 3. 모델 구조 및 성능 향상  

| 모델    | 비트폭 | MMLU 5-shot (INT4) | MMLU 5-shot (INT2) | 추론 속도 |
|---------|-------:|-------------------:|-------------------:|----------:|
| LLaMA-7B | INT4   | 39.4% → **44.9%**   | 27.5% → **30.3%**   | ×2        |
| LLaMA-13B| INT4   | 48.0% → **49.2%**   | 30.9% → **36.9%**   | ×1.8      |

- QA-LoRA는 QLoRA+PTQ 대비 평균 2–5%p 정확도 향상, INT2 환경에서 최대 15%p 극적 개선.  
- 파인튜닝 시 GPU 메모리 사용량과 단계별 학습 시간이 절반 수준으로 감소.[1]

## 4. 한계 및 일반화 성능 향상 가능성  
- **한계**:  
  - 그룹 크기 $$L$$ 선택이 모델 성능과 계산 효율 간 균형을 크게 좌우하며, 최적값 탐색 비용이 발생.  
  - 극저비트(INT2)에서는 여전히 일부 과제에서 성능 변동성 존재.  
- **일반화 향상**:  
  - 그룹 단위 분할은 미세조정 데이터 분포의 지역적 특징을 포착하기 용이하므로, 새로운 도메인·태스크로 전이 학습 시에도 우수한 성능 기대.  
  - FLAN v2, Alpaca 외 Self-instruct, Longform 등 다양한 지침 학습 데이터에서 일관된 성능 개선 관찰됨.[1]

## 5. 향후 연구 영향 및 고려 사항  
- **영향**: 그룹화 기반 양자화와 어댑테이션의 균형 개념은 LLM 경량화·고효율화 연구에 새로운 패러다임을 제시.  
- **고려 사항**:  
  - 동적 그룹 크기 조정 기법을 도입해 데이터 특성에 따라 $$L$$을 학습하는 방향.  
  - 활성화(activations) 양자화 및 Mixed-Precision 적용으로 전체 추론 파이프라인 최적화.  
  - 비영어·멀티모달 모델로 확장하여 범용성 검증.  

***

 QA-LoRA: Quantization-Aware Low-Rank Adaptation of Large Language Models (arXiv:2309.14717v2)[1]

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/2d789d74-48aa-4187-a726-a6546c3036b3/2309.14717v2.pdf
