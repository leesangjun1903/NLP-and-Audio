# DuQuant: Distributing Outliers via Dual Transformation Makes Stronger Quantized LLMs

## 1. 핵심 주장 및 주요 기여  
DuQuant은 대형 언어 모델(LLM)―특히 4비트(weight-activation) 양자화 환경에서—정확도 저하를 일으키는 두 가지 유형의 활성화(outlier) 문제(“Normal Outliers”와 “Massive Outliers”)를 효과적으로 완화하기 위해 **블록 단위 회전(block-diagonal rotation)**과 **지그재그 채널 순열(zigzag permutation)** 의 두 가지 변환을 도입한다. 이를 통해  
- 활성화 및 가중치 모두에서 이상치를 고르게 분산시키고  
- 불필요한 GPTQ 알고리즘 없이도 저비트 양자화에서 SOTA 성능을 달성  

## 2. 해결 과제, 제안 방법, 모델 구조, 성능 향상, 한계  

### 문제 정의  
- Normal Outliers: 모든 토큰에 걸쳐 비교적 큰 값을 지니는 활성화 차원  
- Massive Outliers: 특정 소수 토큰에 국한되어 극도로 큰 값을 가진 활성화 (e.g. FFN down-projection 입력에서 수천 단위)  
기존 SmoothQuant, OmniQuant 등은 Massive Outliers를 효과적으로 처리하지 못해 4비트 양자화 시 성능 저하 발생.  

### 제안 방법 (DuQuant)  
1. **Smooth Technique**  

$$XW = (X\Lambda)\,(\Lambda^{-1}W)$$  
   
$$\Lambda_j = \frac{\max_i|X_{ij}|^\alpha}{\max_i|W_{ij}|^{1-\alpha}}$$ 로 활성화 난이도를 가중치로 이동  

2. **Block-Diagonal Rotation**  

$$
     \hat R = \mathrm{BlockDiag}\bigl(\hat R_{b_1},...,\hat R_{b_K}\bigr),\quad  
     \hat R_{b_i}\in\mathbb R^{2n\times2n}
   $$  
   
   - 이상치가 가장 큰 차원 $$d^{(1)}=\arg\max_j\max_i|X_{ij}|$$ 을 우선 축으로 스위칭(E matrix)  
   - 잔여 차원은 랜덤 직교행렬 $$Q'$$ 로 회전  
   - 탐욕적(greedy)으로 최대 활성화 크기를 최소화하는 $$N$$단계 반복  
3. **Zigzag Permutation**  
   - 채널별 최대 활성화 $$O_{(1)}\ge O_{(2)}\ge\cdots$$ 순서로 블록 간 교대로 배분  
   - 분산 $$\mathrm{Var}([M_{b_1},\dots,M_{b_K}])$$ 최소화  
4. **두 번째 Rotation**  
   - 재균형된 각 블록에 추가 회전 적용  

최종 변환:  

$$
  Y
  = XW
  = \bigl[(X\Lambda)\,\hat R^{(1)}P\,\hat R^{(2)}\bigr]
    \;\bigl[\hat R^{(2)\top}P^\top\hat R^{(1)\top}(\Lambda^{-1}W)\bigr]
$$

### 성능 향상  
- **퍼플렉서티(W4A4)**: LLaMA2-7B WikiText2에서 8.48→6.28로 대폭 개선  
- **Zero-shot QA 평균정확도**: LLaMA2-7B에서 44.52%→60.57%로 ≈16%p 상승  
- **MMLU**: Vicuna-13B zero-shot에서 22.82%→50.94%로 28%p↑  
- **추론 가속**: pre-fill 2.08×, 메모리 3.50× 절감  
- **일반화**: LLaMA3, Mistral, Phi2 등의 모델에서도 일관된 개선 확인  

### 한계  
- **캘리브레이션 데이터 선택**: 무작위 샘플(128개)만 사용, 최적화된 데이터 선택 전략 미구축  
- **회전 블록 크기 및 반복 횟수**: 고정값(128, N=256) 설정, 자동 튜닝 방안 미탐색  

## 3. 모델의 일반화 성능 향상  
- DuQuant은 다양한 아키텍처(LLaMA1/2/3, Vicuna, Mistral, Phi2)에 적용 가능  
- Massive Outliers가 공통적으로 나타나는 FFN down-projection 구조에 특히 강건  
- 4-bit, 6-bit 양자화 환경 모두에서 FP16 대비 성능 저하를 최소화하며 높은 전이 학습 성능 유지  

## 4. 향후 연구에 미치는 영향 및 고려 사항  
- **영향**: 저비트 양자화 기술 연구에 “Dual Transformation” 패러다임 제시, 양자화-추론 연속적 최적화 가능성  
- **고려점**  
  - 캘리브레이션 데이터 효율성: 민감도 낮은 DuQuant 기반 무데이터 양자화 연구  
  - 블록 크기·순열 스케줄 자동화: 모델별 최적 하이퍼파라미터 탐색  
  - 하드웨어 통합: 회전·순열 연산을 통합한 전용 저비트 가속기 설계  

DuQuant은 양자화된 LLM의 배포 효율성과 성능을 동시에 끌어올리는 실용적 솔루션으로, 차후 경량화·가속화 연구의 핵심 기법으로 자리매김할 전망이다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/52a5b347-a059-44db-9247-07d94def3c68/2406.01721v3.pdf
