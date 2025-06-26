
## 핵심 주장과 주요 기여  
“Parameter-Efficient Fine-Tuning Methods for Pretrained Language Models: A Critical Review and Assessment”는 대규모 사전학습 언어모델(PLM)의 전체 파라미터를 재학습하는 비용을 줄이면서 성능 저하 없이 다운스트림 과제에 적응할 수 있는 매개변수 효율적 파인튜닝(PEFT) 기법을 체계적으로 분류·비교하고, 대표 기법을 실험적으로 평가한 최초의 종합 리뷰이다.  
첫째, PEFT 기법을 어딕티브(어댑터·프롬프트), 파셜(바이어스 업데이트·가중치 마스킹), 리파라미터라이즈(로우랭크 분해·LoRA 계열), 하이브리드, 통합 방식의 다섯 카테고리로 분류하고 각 기법의 수식·구조적 차이를 정리했다[1].  
둘째, 11종의 대표 PEFT(어댑터·프롬프트·(IA)³·BitFit·LoRA·AdaLoRA·QLoRA·MAM·ProPETL 등)를 RoBERTa·T5·LLaMA 모델에 적용한 실험을 통해 파라미터·메모리 효율성과 성능(문장 이해·번역·다중지식질의응답)을 정량 비교했다[1].  
셋째, PEFT가 멀티태스크·크로스링구얼 전이·백도어 방어 등 다양한 응용 분야에서 일반화 성능을 높이는 방안을 제시했다[1].

## 해결 문제 및 제안 방법  
기존 PLM의 전체 파인튜닝은 매개변수 수가 수억∼수천억 단위로 늘어남에 따라 연산·메모리 비용이 기하급수적으로 증가하여, 개인·소규모 환경에서 실용적이지 못하다. 이를 해결하기 위해 PEFT는 사전학습 가중치는 고정한 채 소량의 추가 파라미터만 학습하여 비용을 대폭 절감하면서도 본래 모델의 지식 손실을 최소화한다[1].  

제안된 PEFT 분류 및 대표 수식  
- **어댑티브(Adapters)**:  
  – Sequential Adapter: $$X' = \text{ReLU}(XW_{\text{down}})W_{\text{up}} + X$$ (단, $$W_{\text{down}}\in\mathbb{R}^{d\times k},W_{\text{up}}\in\mathbb{R}^{k\times d}$$)[1].  
- **소프트 프롬프트(Prefix-tuning)**:  
  – 입력 임베딩 앞에 학습 가능한 프리픽스 P를 추가하여  
    $$\hat X = [P, X]\in\mathbb{R}^{(l+n)\times d}$$  
    self-attention 시 $$K\leftarrow [\hat P_k,\,K]$$, $$V\leftarrow [\hat P_v,\,V]$$[1].  
- **로우랭크 분해(LoRA)**:  
  – 주어진 가중치 $$W\in\mathbb{R}^{d\times k}$$에 병렬로 두 저차원 행렬을 삽입해  
    $$\Delta W = W_{\text{down}}W_{\text{up}}$$, $$r\ll\{d,k\}$$[1].  
- **기타 기법**: BitFit(오직 바이어스만 업데이트), (IA)$$^{3}$$(키·값·피드포워드 스케일링 벡터 학습), 프리트레인 가중치 마스킹, 델타 가중치 마스킹 등[1].  

## 모델 구조, 성능 향상 및 한계  
RoBERTa-base/large 모델을 GLUE 벤치마크로 평가한 결과, ProPETLAdapter는 전체 파라미터의 1.5%만 학습하면서도 FT 대비 평균 성능을 1.3% (base)·1.65% (large) 향상시켰고, MAM Adapter·LoRA 등도 비슷한 효율을 보였다[1]. T5의 번역 과제(WMT16 En→Ro)에서는 LoRA가 0.39% 파라미터로 BLEU +0.36 개선, (IA)$$^{3}$$가 0.03%로 +0.16 개선을 달성했다[1]. LLaMA-7B/13B의 MMLU에서는 QLoRA가 학습 파라미터를 2.18%로 줄이며 FT 대비 약 2% 성능 저하에 그쳤다[1].  

제약 및 한계로는, 어댑터 기반 기법의 경우 학습 메모리 오버헤드가 커질 수 있고, 프롬프트 기법은 템플릿 길이에 민감해 소규모 데이터에 취약하다. LoRA는 랭크 선택이 성능에 결정적이며, 양자화 기법은 정밀도 손실 위험을 동반한다[1].

## 일반화 성능 향상 관점  
PEFT 모듈의 모듈화된 설계는 멀티태스크 및 크로스-링구얼 전이 시 유연한 조합·재사용을 가능하게 하여 일반화 성능을 크게 강화한다.  
- **멀티태스크 학습**: 프리트레인된 어댑터 조합(AdapterFusion)·소프트 프롬프트 전이(SPOT, ATTEMPT)·LoRA 허브 조합(LoRAHub)을 통해 여러 과제 간 지식 공유·재구성이 가능하다[1].  
- **크로스링구얼 전이**: 언어별 어댑터(MAD-X)·희소 델타 벡터(LT-SFT)·바이링구얼 어댑터(BAD-X) 등이 사전학습 모델을 새로운 언어로 효율 전이하며, zero-shot 성능을 제고한다[1].

## 향후 연구 영향 및 고려 사항  
이 리뷰는 PEFT 연구 전반에 대한 분류·정량 비교·응용 사례를 종합함으로써, 효율적 파인튜닝 기법 개발과 이의 멀티태스크·크로스-모달 전이 활용을 촉진할 것으로 보인다.  
향후 연구에서는 경량 하이브리드 PEFT 구조 탐색, LoRA 기반 가지치기·양자화 기술 심화, PEFT 사용 편의성 제고를 위한 라이브러리 통합, PEFT 기법의 이론적 설명력 강화, 컴퓨터 비전·다중모달 분야 확장 등을 고려해야 한다[1].

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/0599a098-83a3-436c-973e-040abb6ece80/2312.12148v1.pdf

# III. 매개변수 효율적 파인튜닝 방법(Parameter-Efficient Fine-Tuning Methods) 상세 설명

대규모 사전학습 언어모델(PLM)은 수억~수천억 개의 파라미터를 가지며, 모든 파라미터를 재학습(full fine-tuning)하면 계산량과 메모리 부담이 매우 커집니다. 매개변수 효율적 파인튜닝(PEFT)은 **소량의 추가 파라미터**만 학습하거나 **일부 파라미터만 업데이트**함으로써 비용을 절감하면서도 성능 저하를 최소화하는 기법들입니다. PEFT 방법은 크게 다섯 가지 범주로 나눌 수 있습니다[1]:

1. 부가적 파인튜닝(Additive Fine-Tuning)  
2. 부분적 파인튜닝(Partial Fine-Tuning)  
3. 재매개변수화 파인튜닝(Reparameterized Fine-Tuning)  
4. 혼합 파인튜닝(Hybrid Fine-Tuning)  
5. 통합 파인튜닝(Unified Fine-Tuning)  

아래에서는 각 범주별 아이디어와 대표 기법을 **쉽고 이해하기 쉽게** 정리합니다.

## 1. 부가적 파인튜닝(Additive Fine-Tuning)

사전학습된 모델의 기존 파라미터는 그대로 고정하고, **추가적인 작은 모듈**(어댑터, 프롬프트, 스케일링 벡터 등)을 삽입해 이 모듈만 학습하는 방식입니다.

1-1. 어댑터(Adapter-based)  
- Transformer의 각 층(자기어텐션 뒤, FFN 뒤)에 **작은 병목(bottleneck)** 모듈을 삽입  
- 다운(𝑑→𝑘)·ReLU·업(𝑘→𝑑) 구조로, 입력 X에 더해 잔차 학습(residual)이 가능  
- 전체 파라미터의 1–5%만 학습해도 원본 대비 유사한 성능 달성[1]  

1-2. 소프트 프롬프트(Soft Prompt-based)  
- 입력 임베딩 앞이나 어텐션 키·값 앞에 **학습 가능한 벡터**를 삽입  
- 모델 내부 가중치는 고정, 프롬프트(𝐿×𝑑)만 업데이트  
- Few-shot 전이학습에 유리  

1-3. 기타(Other)  
- (IA)³: 어텐션·FFN 활성값을 **벡터로 스케일**[1]  
- Ladder Side-Tuning: 사이드 네트워크에만 파라미터 저장  
- Hadamard Adapter: 어텐션 출력에 요소별 곱 적용  

## 2. 부분적 파인튜닝(Partial Fine-Tuning)

사전학습된 모델 파라미터 중 중요한 일부만 업데이트하거나 마스크를 통해 **불필요한 파라미터를 영(0)으로 고정**하는 기법입니다.

2-1. 바이어스 업데이트(Bias-only)  
- 어텐션·FFN·LayerNorm의 **바이어스 항만** 업데이트 (BitFit)  
- 전체 파라미터의 0.1–0.5%만 학습[1]

2-2. 사전학습 가중치 마스킹(Weight Masking)  
- 가중치 값 크기나 Fisher 정보량 기준으로 상위 k개만 남기고 나머지 고정  
- 마스크된 파라미터는 학습되지 않음

2-3. 델타 가중치 마스킹(Delta Masking)  
- 처음과 끝 학습 가중치 차이(∆W) 상위 k개만 학습 (LT-SFT)  
- 매 스텝마다 ∆W를 계산해 중요 파라미터만 업데이트  

## 3. 재매개변수화 파인튜닝(Reparameterized Fine-Tuning)

∆W를 **저차원·구조적 형태**로 분해해, 원래 가중치는 고정한 채 그 일부만 학습합니다.

3-1. 저차원 분해(Low-Rank Decomposition)  
- Fastfood, Kronecker(∆W=W₁⊗W₂) 등 수학적 변환으로 차원 축소  
- Intrinsic SAID: Fastfood 변환 기반 내재 차원만 학습

3-2. LoRA 계열(LoRA & derivatives)  
- 어텐션 Q/K/V 가중치에 평행 연결(parallel)해 **랭크-r 행렬** 2개만 학습  
- 학습 시 모델 가중치는 고정, ∆W=W_down×W_up만 업데이트→추론 시 병합  
- DyLoRA/AdaLoRA: 동적·적응적 랭크 조정  
- QLoRA/QA-LoRA: 4bit 양자화 모델에 LoRA 적용  
- LoRAPrune: LoRA 가중치에 구조적 가지치기  

## 4. 혼합 파인튜닝(Hybrid Fine-Tuning)

여러 PEFT 기법(어댑터, 프롬프트, LoRA 등)을 **조합**해, 각 기법의 장점은 살리고 단점은 보완합니다.

4-1. 수작업 조합(Manual Combination)  
- MAM Adapter: 병렬 어댑터+prefix tuning 조합  
- Compacter: 하이퍼복소수(PHM) 레이어와 어댑터 결합  
- UniPELT: 어댑터·프롬프트·LoRA를 게이팅으로 결합  

4-2. 자동 구조 탐색(Automatic Combination)  
- AutoPEFT: Bayesian 최적화로 각 층에 넣을 모듈 자동 결정  
- S3Delta-M/S4: 미리 정의된 PEFT 기법 중 **희소성 제약** 아래 최적 조합 검색  

## 5. 통합 파인튜닝(Unified Fine-Tuning)

단일 “프로토타입” 모듈(어댑터·LoRA 등)을 **마스크**로 층별로 분기해 모든 층에서 재사용하면서도, 층별로 다른 부분만 학습하게 만드는 방법입니다.

- AdaMix: MoE처럼 여러 어댑터 중 랜덤 라우팅→일관성 규제→추론 시 평균 병합  
- SparseAdapter: 어댑터 파라미터 전체를 희소 마스크로 제어  
- ProPETL: 하나의 모듈에 **다중 바이너리 마스크** 적용해 층별 차별적 서브네트워크 학습  

이처럼 PEFT 기법들은 **새로운 파라미터 삽입**, **일부만 학습**, **저차원 분해**, **복합 조합**, **모듈 통합** 등 다양한 전략으로 대규모 언어모델의 파인튜닝 부담을 획기적으로 줄이며, 상당수 기법이 풀파인튜닝 대비 동등 또는 더 우수한 성능을 보이고 있습니다[1].  

[1] Xu et al., “Parameter-Efficient Fine-Tuning Methods for Pretrained Language Models: A Critical Review and Assessment,” arXiv:2312.12148, Sec. III.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/0599a098-83a3-436c-973e-040abb6ece80/2312.12148v1.pdf

# III. 매개변수 효율적 파인튜닝 방법(Parameter-Efficient Fine-Tuning Methods)

본 절에서는 사전학습 언어모델(PLM)의 대부분 파라미터를 고정한 채 소량의 파라미터만 학습하여 비용을 크게 줄이면서도 풀 파인튜닝 수준의 성능을 달성하는 주요 PEFT 기법을 다섯 가지 범주로 나누어 상세히 설명한다[1].

## 1. 부가적 파인튜닝(Additive Fine-Tuning)  
추가 모듈을 삽입하고 이 모듈의 파라미터만 학습한다.

-  어댑터 기반(Adapter-based)  
 Transformer 각 층의 Self-Attention 및 FFN 뒤에 작은 병목 모듈(adapter)을 삽입한다. 다운프로젝션(d→k), 비선형 활성화, 업프로젝션(k→d) 구조를 거친 출력이 입력에 더해지는 방식으로, 전체 파라미터의 1–5%만 학습해도 성능 저하가 거의 없다[1].

-  소프트 프롬프트(Soft Prompt-based)  
 입력 임베딩 앞 또는 어텐션 키·값 앞에 학습 가능한 벡터(prefix)를 추가한다. 모델 본체는 고정된 채로 프롬프트(ℓ×d)만 업데이트하며, few-shot 전이학습에 뛰어나다[1].

-  기타(Others)  
 (IA)³: 어텐션과 FFN 활성값에 스케일 벡터를 곱해 학습[1].  
 Ladder Side-Tuning: 사이드 네트워크에만 파라미터 저장·학습[1].  
 Hadamard Adapter: 어텐션 출력에 요소별 곱을 적용한 어댑터[1].

## 2. 부분적 파인튜닝(Partial Fine-Tuning)  
기존 파라미터 중 일부만 업데이트하거나 마스킹으로 나머지를 고정한다.

-  바이어스 업데이트(Bias-only)  
 BitFit: 어텐션·FFN·LayerNorm의 바이어스 항만 업데이트하여 0.1–0.5%만 학습[1].

-  사전학습 가중치 마스킹(Weight Masking)  
 Threshold-Mask: 임계값 이상 가중치만 학습[1].  
 FISH Mask: Fisher 정보량 상위 k개만 업데이트[1].

-  델타 가중치 마스킹(Delta Masking)  
 LT-SFT: 풀 파인튜닝 후 |ΔW|가 큰 상위 k개만 반복 학습[1].  
 Child-Tuning: Task-Driven/F-Free 방식으로 그라디언트 마스킹[1].  
 Diff Pruning: 학습 가능한 마스크로 δ=W+M⊙ΔW를 희소화[1].  
 SAM: 이차 근사해를 이용해 최적의 마스크 선택[1].

## 3. 재매개변수화 파인튜닝(Reparameterized Fine-Tuning)  
ΔW를 저차원·구조적 형태로 분해해 원본 가중치는 고정하고 재구성된 파라미터만 학습한다.

-  저차원 분해(Low-Rank Decomposition)  
 Intrinsic SAID: Fastfood 변환으로 저차원 공간에서 ΔW=F(Wʳ) 학습[1].  
 KronA: Kronecker 곱 분해 ΔW=W↓⊗W↑[1].

-  LoRA 계열(LoRA Derivatives)  
 LoRA: 어텐션 Q/K/V 가중치에 병렬로 랭크-r 행렬 2개 삽입해 ΔW=WdownWup 학습, 추론 시 병합[1].  
 DyLoRA: 매 스텝마다 임의 랭크 b를 선택해 동적 저차원 적응[1].  
 AdaLoRA: SVD로 ΔW=PΛQ, 중요 singular value 동적 유지[1].  
 IncreLoRA: Λ 대각 성분 조정으로 단계적 파라미터 할당[1].  
 Delta-LoRA: 훈련 반복 간 Δ(WdownWup)로 pretrained W 업데이트[1].  
 LoRAPrune: 그룹별 구조적 가지치기용 마스크 적용[1].  
 QLoRA/QA-LoRA/LOFTQ: 4bit NF4 양자화 결합으로 메모리 절감 및 정밀도 유지[1].  

-  LoRA 기반 개선  
 Kernel-mix-lite(qv/qvo): 어텐션 헤드를 커널 관점에서 분리해 LoRA 적용 범위 제어[1].  
 Laplace-LoRA: Laplace 근사로 Bayesian 추론 통합[1].  
 LoRA-FA: Wdown을 고정하고 Wup만 업데이트해 활성화 메모리 절감[1].  
 LoRAHub/MOELoRA/L-LoRA: 다수 LoRA 모듈 결합으로 멀티태스크 전이 강화[1].

## 4. 혼합 파인튜닝(Hybrid Fine-Tuning)  
여러 PEFT 기법을 조합해 장단점을 보완한다.

-  수작업 조합(Manual Combination)  
 MAM Adapter: 병렬 어댑터와 prefix-tuning 결합[1].  
 Compacter: PHM 레이어와 어댑터·저차원 최적화 통합[1].  
 UniPELT: 어댑터·프롬프트·LoRA를 게이트로 융합[1].

-  자동 구조 탐색(Automatic Combination)  
 AutoPEFT: 베이지안 최적화로 각 층에 적용할 PEFT 모듈 자동 결정[1].  
 S3Delta-M: LoRA·Compacter·BitFit·LNFit의 희소 구조 미분 탐색[1].  
 S4: 층을 네 그룹으로 분할하고 그룹별로 A, P, B, L 모듈 할당[1].

## 5. 통합 파인튜닝(Unified Fine-Tuning)  
단일 프로토타입 모듈에 마스크를 통해 층별·태스크별 서브네트워크를 학습한다.

-  AdaMix: 모듈 간 확률적 라우팅 후 일관성 제약으로 하나의 모듈 병합[1].  
-  SparseAdapter: 어댑터·LoRA 파라미터를 네트워크 프루닝으로 통합 희소화[1].  
-  ProPETL: 한 프로토타입 네트워크에 층별 이진 마스크 적용해 서브네트워크별 학습[1].  

이처럼 PEFT는 **추가 모듈 삽입**, **일부 파라미터만 학습**, **저차원 분해**, **복합 조합**, **모듈 통합** 등으로 거대한 PLM 파인튜닝의 연산·메모리 부담을 획기적으로 완화하며, 다수 기법이 풀 파인튜닝과 동등하거나 뛰어난 성능을 보인다[1].

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/0599a098-83a3-436c-973e-040abb6ece80/2312.12148v1.pdf
