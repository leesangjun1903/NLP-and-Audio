
# SeqDiffuSeq: Text Diffusion with Encoder-Decoder Transformers

## **1. 핵심 주장 및 주요 기여 요약**

**SeqDiffuSeq**는 이산적(discrete)인 텍스트 데이터를 연속적(continuous)인 확산 모델(Diffusion Model)에 효과적으로 적용하기 위해 제안된 **Encoder-Decoder 기반의 텍스트 생성 모델**입니다.
이 논문의 핵심 주장은 **"적응형 노이즈 스케줄(Adaptive Noise Schedule)"**과 **"셀프 컨디셔닝(Self-Conditioning)"** 기법을 통해 기존 텍스트 확산 모델의 품질과 속도 한계를 극복할 수 있다는 것입니다.

**주요 기여:**
1.  **아키텍처 혁신:** 기존의 Encoder-only 방식(DiffuSeq)에서 벗어나, Seq2Seq 작업에 최적화된 **Encoder-Decoder Transformer** 구조를 도입하여 입력 처리 효율성과 생성 성능을 동시에 향상시켰습니다.
2.  **적응형 노이즈 스케줄:** 토큰별 난이도에 따라 노이즈 레벨을 동적으로 조절하여 모델의 학습 및 생성 능력을 최적화했습니다.
3.  **성능 입증:** 기계 번역, 요약, 대화 생성 등 5개 과제에서 기존 확산 모델(DiffuSeq)을 능가하고, AR(자기회귀) 모델과 경쟁할 수 있는 성능을 입증했습니다.

***

## **2. 상세 분석: 문제 정의, 방법론, 구조 및 성능**

### **2.1. 해결하고자 하는 문제**
텍스트는 이미지와 달리 불연속적(discrete)인 데이터입니다. 기존 연구들(DiffusionLM, DiffuSeq)은 이를 연속 공간인 임베딩(embedding) 공간에 매핑하여 확산 모델을 적용했으나, 다음과 같은 한계가 있었습니다.
*   **구조적 비효율성:** DiffuSeq 등은 Encoder-only 구조를 사용하여 입력(Source)과 출력(Target)을 하나의 시퀀스로 결합해 처리하므로 연산 효율이 떨어짐.
*   **고정된 노이즈 스케줄:** 모든 토큰에 동일한 노이즈 스케줄을 적용하여, 토큰마다 다른 복원 난이도를 반영하지 못함.
*   **정보 손실:** 이전 단계의 예측 정보를 다음 단계에서 충분히 활용하지 못함.

### **2.2. 제안하는 방법 및 수식**

SeqDiffuSeq는 **연속 확산 프레임워크(Continuous Diffusion Framework)**를 기반으로 합니다.

*   **확산 과정 (Forward Process):**
    텍스트 임베딩 $z_0$에 점진적으로 가우시안 노이즈를 주입합니다. 시간 $t$에서의 잠재 변수 $z_t$는 다음과 같이 정의됩니다.

$$ q(z_t|z_{t-1}) = \mathcal{N}(z_t; \sqrt{\alpha_t}z_{t-1}, (1-\alpha_t)I) $$

$$ q(z_t|z_0) = \mathcal{N}(z_t; \sqrt{\bar{\alpha}_t}z_0, (1-\bar{\alpha}_t)I) $$

*   **역확산 과정 (Reverse Process):**
    노이즈가 섞인 $z_t$와 입력 문장 $w_x$를 조건으로 하여 원본 $z_{t-1}$을 복원합니다.

$$ p_\theta(z_{t-1}|z_t, w_x) = \mathcal{N}(z_{t-1}; \tilde{\mu}_\theta(z_t, w_x, t), \tilde{\beta}_t I) $$
    
여기서 모델은 노이즈 자체를 예측하는 대신, $z_t$로부터 원본 임베딩 $z_0$를 직접 예측하는 함수 $f_\theta(z_t, w_x, t)$를 학습합니다.

*   **손실 함수 (Objective):**
    모델은 예측된 $z_0$와 실제 $z_0$ 사이의 거리(MSE)를 최소화하도록 학습됩니다.

$$ \mathcal{L}_{simple} = \mathbb{E}_{q_\phi} \left[ \sum_{t=2}^T || z_{0_\theta}(z_t, w_x, t) - z_0 ||^2 + || z_{0_\theta}(z_1, w_x, 1) - g_\phi(w_y) ||^2 \right] $$

### **2.3. 모델 구조 (Encoder-Decoder Architecture)**
*   **Encoder:** 입력 텍스트($w_x$)를 한 번만 인코딩하여 표현 벡터를 생성합니다. 이는 반복적인 역확산 과정에서 매번 입력을 다시 처리해야 했던 기존 방식보다 연산량을 획기적으로 줄입니다.
*   **Decoder:** 노이즈가 섞인 출력 시퀀스($z_t$)와 인코더의 출력을 받아 디노이징을 수행합니다. Transformer Decoder를 사용하되, 자기회귀(Auto-regressive) 마스킹 없이 전체 시퀀스를 한 번에 처리(Non-autoregressive)합니다.

### **2.4. 성능 향상 및 한계**
*   **성능:** 5개 데이터셋(QQP, Wiki-Auto 등) 실험 결과, **DiffuSeq 대비 BLEU 점수가 크게 향상**되었으며, 추론 속도는 약 **3.6배 가속**되었습니다.
*   **한계:**
    *   **추론 비용:** AR 모델에 비해 여전히 많은 스텝(예: 2000 step)을 거쳐야 하므로 절대적인 연산량은 많습니다.
    *   **다양성 트레이드오프:** 제안된 기법들이 품질(Quality)은 높이지만, 생성 결과의 다양성(Diversity)은 다소 감소시키는 경향이 있습니다.

***

## **3. 모델의 일반화 성능 향상 가능성 (Focus)**

이 논문에서 **"일반화(Generalization)"**는 훈련 데이터에 없는 새로운 문장에 대한 강건한 생성 능력과 토큰별 난이도 차이에 대한 적응력으로 해석할 수 있습니다. 특히 두 가지 핵심 기술이 일반화 성능 향상에 기여합니다.

### **3.1. 적응형 노이즈 스케줄 (Adaptive Noise Schedule)**
기존 모델은 모든 토큰 위치와 시간 단계에 고정된 노이즈 스케줄(예: Cosine, Linear)을 사용했습니다. 그러나 문장 내 토큰마다 예측 난이도는 다릅니다(예: 관사는 쉽고 고유명사는 어려움).

*   **핵심 아이디어:** 모델의 손실(Loss) 값을 기반으로 각 토큰 위치($i$)마다 다른 노이즈 스케줄 $\bar{\alpha}_t^i$를 학습합니다.

$$ \bar{\alpha}_t^i = \mathcal{M}_i(\mathcal{L}_t^i) $$

여기서 $\mathcal{M}_i$는 손실 값 $\mathcal{L}$과 노이즈 레벨 $\alpha$ 사이의 매핑 함수입니다.
*   **일반화 기여:** 어려운 토큰에는 노이즈를 천천히 제거하고, 쉬운 토큰은 빠르게 확정짓게 함으로써, 모델이 **다양한 난이도의 언어 패턴에 유연하게 대응**할 수 있게 합니다. 이는 낯선 문맥이나 복잡한 구문 구조에서도 모델이 무너지지 않고 안정적으로 생성하게 돕습니다.

### **3.2. 셀프 컨디셔닝 (Self-Conditioning)**
*   **핵심 아이디어:** 현재 단계($t$)에서 $z_0$를 예측할 때, 이전 단계($t+1$)에서 예측했던 $z_0$값($\hat{z}_0^{t+1}$)을 추가적인 입력으로 활용합니다.

$$ z_{0_\theta}(z_t, \hat{z}_0^{t+1}, w_x, t) $$

*   **일반화 기여:** 초기 단계의 부정확한 예측을 점진적으로 수정해 나가는 과정을 통해, 모델이 **오류를 스스로 교정(Self-correction)**하는 능력을 갖게 됩니다. 이는 학습하지 않은 패턴이 등장했을 때도 초기 추론의 오류를 완화하며 결과적으로 테스트 셋에서의 일반화 성능(BLEU 점수)을 높이는 결과를 낳았습니다.

***

## **4. 향후 연구 영향 및 고려사항**

### **4.1. 연구에 미치는 영향**
*   **아키텍처의 표준 변화:** 텍스트 확산 모델에서도 Encoder-Decoder 구조가 효율적임을 입증하여, 이후 연구들이 이 구조를 채택하거나(Meta-DiffuB 등) 변형하는 계기가 되었습니다.
*   **비자기회귀(NAR) 생성의 가능성:** 순차적으로 단어를 생성하는 AR 모델의 속도 한계를 극복하고, 병렬 생성(Parallel Generation)이 가능한 확산 모델이 고품질 텍스트 생성의 대안이 될 수 있음을 보여주었습니다.

### **4.2. 향후 연구 시 고려할 점**
*   **추론 단계(Step) 단축:** 2000회에 달하는 디노이징 단계는 실시간 서비스 적용에 걸림돌입니다. 이를 수십 단계로 줄이면서도 품질을 유지하는 **고속 샘플링(Fast Sampling)** 기술(예: DPM-Solver 적용 등) 연구가 필수적입니다.
*   **이산 데이터 최적화:** 임베딩 공간의 연속 확산은 '반올림(Rounding)' 과정에서 정보 손실이 발생할 수 있습니다. **이산 공간 확산(Discrete Diffusion)**이나, 연속-이산 간의 간극을 줄이는 기법에 대한 고려가 필요합니다.
*   **환각(Hallucination) 제어:** 일반화 성능이 높아졌으나, 확산 모델 특유의 제어 불가능한 생성(환각)을 억제하기 위한 제약 조건(Guidance) 연구가 병행되어야 합니다.

***

## **2020년 이후 관련 최신 연구 탐색 (Trend Analysis)**

SeqDiffuSeq 이후 텍스트 확산 모델 연구는 **"속도 개선", "구조 최적화", "품질 향상"**을 중심으로 빠르게 발전하고 있습니다.

| 연구/모델명 | 연도 | 주요 특징 및 SeqDiffuSeq와의 관계 |
| :--- | :--- | :--- |
| **DiffuSeq** | 2022 | SeqDiffuSeq의 직접적인 비교 대상. Encoder-only 구조를 사용하며, SeqDiffuSeq가 이를 Encoder-Decoder로 개선함. |
| **DiffusionBERT** | 2022 | BERT를 활용한 이산 확산 모델. 초기 연구로 분류됨. |
| **DiffuSeq-v2** | 2023 | 연속 공간과 이산 토큰 간의 간극을 줄이기 위해 'Soft absorbing state'를 도입하고, 고속 ODE Solver를 적용하여 **속도를 획기적으로 개선**. |
| **Meta-DiffuB** | 2024 | 문맥을 반영한(Contextualized) 메타 학습 프레임워크를 적용. SeqDiffuSeq보다 **MBR(Minimum Bayes Risk) 디코딩 성능이 우수**하며 다양성과 품질 균형을 개선함. |
| **SSD-LM** | 2023 | **반-자기회귀(Semi-autoregressive)** 방식을 도입. 확산 모델의 병렬성과 AR 모델의 국소적 일관성을 결합하여 긴 텍스트 생성 능력을 향상시킴. |
| **DiNoiSer** | 2023 | 텍스트의 이산적 특성을 고려한 **Counter-discreteness training** 제안. 노이즈 레벨을 적응적으로 조절한다는 점에서 SeqDiffuSeq와 유사한 철학을 공유하지만 방법론이 다름. |

**최신 트렌드 요약:** 
단순한 구조 변경을 넘어, 2024-2025년 연구들은 **대규모 언어 모델(LLM)과 확산 모델의 결합**, 혹은 확산 모델을 활용한 **생각의 사슬(Chain-of-Thought) 계획 능력 향상** 등으로 연구 범위가 확장되고 있습니다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/5ed4de5d-f0ce-4de7-8b76-3451fc9de0ff/2212.10325v5.pdf)
[2](https://arxiv.org/abs/2312.09390)
[3](https://arxiv.org/abs/2407.14985)
[4](https://aclanthology.org/2025.findings-emnlp.217)
[5](https://arxiv.org/abs/2406.16935)
[6](https://pubs.aip.org/pof/article/37/3/035149/3339259/Generalization-capabilities-and-robustness-of)
[7](https://www.mdpi.com/2076-3417/14/10/4005)
[8](https://jivp-eurasipjournals.springeropen.com/articles/10.1186/s13640-024-00656-x)
[9](https://arxiv.org/abs/2412.10146)
[10](https://arxiv.org/abs/2410.18938)
[11](https://arxiv.org/abs/2402.04967)
[12](https://arxiv.org/pdf/2210.08933.pdf)
[13](https://arxiv.org/pdf/2212.10325.pdf)
[14](https://www.aclweb.org/anthology/W18-1004.pdf)
[15](http://arxiv.org/abs/2310.05793v2)
[16](https://aclanthology.org/2023.findings-emnlp.660.pdf)
[17](http://arxiv.org/pdf/2310.09213.pdf)
[18](https://aclanthology.org/2022.emnlp-main.337.pdf)
[19](http://arxiv.org/pdf/2503.06698.pdf)
[20](https://proceedings.neurips.cc/paper_files/paper/2024/file/91d193b65d0b120d29503590827de1ea-Paper-Conference.pdf)
[21](https://summmeer.github.io/uploads/DiffuSeq_poster-v1.pdf)
[22](https://pmc.ncbi.nlm.nih.gov/articles/PMC10909201/)
[23](https://huggingface.co/blog/ProCreations/diffusion-language-model)
[24](https://aclanthology.org/2024.naacl-long.2/)
[25](https://github.com/christianversloot/machine-learning-articles/blob/main/differences-between-autoregressive-autoencoding-and-sequence-to-sequence-models-in-machine-learning.md)
[26](https://peerj.com/articles/cs-1905/)
[27](https://arxiv.org/abs/2410.21357)
[28](https://www.sciencedirect.com/science/article/abs/pii/S0950705125014765)
[29](https://aclanthology.org/2024.naacl-long.2.pdf)
[30](https://www.seangoedecke.com/limitations-of-text-diffusion-models/)
