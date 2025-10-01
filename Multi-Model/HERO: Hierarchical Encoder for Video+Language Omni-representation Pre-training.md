# HERO: Hierarchical Encoder for Video+Language Omni-representation Pre-training

**주요 요약**  
HERO는 비디오-언어 대규모 사전학습을 위해 설계된 **계층적 멀티모달 인코더**로, 지역적 프레임-자막 융합과 전역적 비디오 문맥 학습을 분리하여 수행한다. 새로운 **영상-자막 정합(Video-Subtitle Matching, VSM)** 및 **프레임 순서 복원(Frame Order Modeling, FOM)** 과제를 도입해 시공간적 정렬 정보를 명시적으로 학습하며, HowTo100M 및 TV 데이터로 공동학습하여 다양한 다운스트림 영상-언어 과제에서 **최신 기법 대비 유의미한 성능 향상**을 입증했다.[1]

## 1. 해결하고자 하는 문제  
기존 영상-언어 사전학습 모델은 대부분 정적 이미지-텍스트에 최적화되어,  
- 프레임과 자막 간의 **시간적 정렬**을 충분히 활용하지 못하고  
- 시퀀스 특성을 반영한 사전학습 과제를 갖지 못하며  
- 학습 도메인이 한정적(주로 요리·설명 영상)에 국한되는 한계가 있다.[1]

## 2. 제안 방법 및 수식  
HERO는 입력 비디오 클립의 프레임 $$v = \{v_i\}\_{i=1}^{N_v}$$ 과 자막 문장 $$s = \{s_i\}_{i=1}^{N_s}$$ 을 다음과 같이 계층적으로 인코딩한다.[1]

  1) **Cross-modal Transformer**: 각 자막 문장 $$s_i$$와 해당 프레임 $$v_{i}$$을 융합해 지역적 멀티모달 임베딩 $$\{V^{\text{cross}}_i, W^{\text{cross}}_i\}$$ 생성.  
  2) **Temporal Transformer**: 모든 $$\{V^{\text{cross}}_i\}$$를 입력받아 전역적 비디오 문맥 $$\ V^{\text{temp}}$$를 학습.  

사전학습 과제:  
- **MLM (Masked Language Modeling)**  

$$ \mathcal{L}_{\mathrm{MLM}} = -\mathbb{E}_{(w,s,v)\sim D}\sum_{m\in M}\log P(w_m\mid w_{\setminus m},\,v)$$  

- **MFM (Masked Frame Modeling)**: 연속적 프레임 특징 복원(MFFR) 또는 NCE 기반 최적화(MNCE)  
- **VSM (Video-Subtitle Matching)**:  
  - 지역 정합: 각 프레임의 시작·종료 인덱스 예측  
  - 전역 정합: 클립-자막 매칭 힌지 손실  
- **FOM (Frame Order Modeling)**: 임의 섞인 프레임 순서 복원  
최종 손실:  

$$
\mathcal{L} = \mathcal{L}_{\mathrm{MLM}} + \mathcal{L}_{\mathrm{MFM}} + \lambda_{1}\mathcal{L}_{\mathrm{VSM}} + \lambda_{2}\mathcal{L}_{\mathrm{FOM}}
$$  

(하이퍼파라미터 $$\lambda_{1,2}$$로 균형 조정).[1]

## 3. 모델 구조  
- **비디오 임베더**: ResNet-101 & SlowFast 추출기 결합 후 FC  
- **텍스트 임베더**: WordPiece 토크나이저 + 포지션 임베딩  
- **Cross-modal Transformer**: 멀티헤드 어텐션 기반  
- **Temporal Transformer**: 전역 순차적 문맥 인코딩  
- **쿼리 인코더**(VSM 전용): 1D 컨볼루션 및 LN 포함 MLP

## 4. 성능 향상 및 한계  
- HowTo100M+TV 공동학습 시 **TVR R@1: 5.13%↑**, TVQA Acc: 74.80%↑ 등 다수 벤치마크에서 SOTA 경신.[1]
- **FOM** 도입으로 시간추론 과제(QA) 성능 개선, **VSM**으로 모멘트 검색 과제 성능 대폭 향상됨.  
- **MFFR**는 MNCE와 경쟁하며 기여 미미.  
- 도메인 차이(요리 vs. TV) 민감도 관찰: TVR은 유사 도메인 학습이 성능에 유리.[1]
- 한계:  
  - 지역 수준(region-level) 특징 비활용으로 세밀한 공간 추론(물체 기반 QA)에는 제약  
  - 컨텍스트가 짧은 캡셔닝(TVC)엔 디코더 사전학습 부재가 성능 저하 요인

## 5. 일반화 성능 향상 가능성  
- **다양한 영상 도메인** 및 **대규모 TV·튜토리얼 영상** 융합으로 일반화 강화  
- **계층적 구조**가 단일 BERT형(flat) 모델 대비 명시적 시공간 정렬 학습 유도.[1]
- 다운스트림 적응 시 Video-QA, Retrieval, Captioning, Inference 등 여러 과제에 **일관된 전이 이점** 확인  
- 모달리티 결합 시기와 정렬 과제 설계가 일반화 성능 향상의 핵심 요소

## 6. 향후 연구 영향 및 고려 사항  
- **연구 영향**:  
  - 시공간 정렬 사전학습 과제(VSM·FOM) 사례 제공  
  - 멀티모달 대규모 비디오 학습 패러다임 선도  
  - 신규 벤치마크(How2R, How2QA)로 다양성 제고  
- **향후 고려점**:  
  - **지역 수준 특징**(객체 탐지 기반) 결합해 세밀한 공간추론 강화  
  - **디코더 사전학습** 또는 다중 과제(multitask) 학습으로 캡셔닝 및 생성 과제 개선  
  - 저자원 도메인(의료, 과학 등) 맞춤형 도메인 적응 연구  
  - **모델 경량화** 및 **실시간 인퍼런스** 최적화

HERO는 계층적 인코딩과 시공간적 정렬 과제 설계를 통해 영상-언어 사전학습의 새로운 기준을 제시하며, **일반화 성능과 다중 과제 전이에 유의미한 진보**를 가져왔다.[1]

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/6d7e251e-cb7d-4a28-a63b-91289ac741dc/2005.00200v2.pdf)
