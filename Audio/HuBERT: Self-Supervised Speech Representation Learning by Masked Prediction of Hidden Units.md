# HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units

## 1. 핵심 주장 및 주요 기여 (간결 요약)
HuBERT는 연속 음성 신호에서 숨겨진 단위(hidden units)를 예측하는 **마스킹 예측(masked prediction)** 방식을 통해 고품질의 셀프-슈퍼바이즈드 음성 표현을 학습한다.  
주요 기여:
- 오프라인 k-평균 클러스터링을 통해 연속 입력에 대한 타깃 레이블을 생성  
- 마스킹된 구간만 예측하도록 학습 손실을 적용하여 음향 모델과 언어 모델 능력을 동시에 학습  
- 반복적 클러스터링을 통한 타깃 정제(target refinement)로 성능 개선  
- 300M~1B 규모의 트랜스포머 모델에서도 Librispeech 및 Libri-light 벤치마크에서 최첨단 성능 달성  

***

## 2. 문제 정의 및 제안 방법

### 2.1 해결하려는 문제
1. **다중 음향 단위**: 한 발화에 여러 음향 단위가 존재  
2. **사전 정의된 음향 단위 부재**: 텍스트 기반 작업과 달리 사전 레이블(lexicon)이 없음  
3. **가변 길이 단위 경계 미정**: 프레임별 명시적 세분화 불가  

### 2.2 HuBERT 방법 개요
1. **숨겨진 단위 발굴 (Offline Clustering)**  
   - 입력 신호 $$X = [x_1, \dots, x_T]$$에 대해 k-평균을 이용해 각 프레임의 클러스터 할당 $$Z = [z_1, \dots, z_T]$$을 생성  
2. **마스킹된 예측 (Masked Prediction)**  
   - 마스킹 인덱스 $$M \subset [T]$$를 랜덤 생성하고 masked input $$\tilde X$$ 구성  
   - 모델 $$f$$는 masked 프레임에 대해서만 클러스터 레이블을 예측  
   - 손실함수:  

$$
       \mathcal{L} = \alpha \sum_{t \in M} -\log p_f(z_t \mid \tilde X, t)
       \quad (\alpha=1.0)
     $$  

3. **반복적 정제 (Iterative Refinement)**  
   - 1차로 학습된 표현으로 재클러스터링하여 더 정교한 타깃 생성  
   - 2회 이상의 반복 후 모델 재학습  

### 2.3 수식 정리
- 마스킹 예측 손실:  

$$
    \mathcal{L}\_\mathrm{mask} = \sum_{t \in M} -\log p_f(z_t \mid \tilde X, t)
  $$

- 전체 손실(마스킹만 적용):  

$$
    \mathcal{L} = \alpha \mathcal{L}\_\mathrm{mask} + (1-\alpha) \mathcal{L}_\mathrm{unmask}, \quad \alpha=1
  $$

***

## 3. 모델 구조
- **CNN 오디오 인코더**  
  - 7개 컨볼루션 레이어, 총 다운샘플링 ×320, 채널 512  
- **BERT-스타일 트랜스포머 인코더**  
  - BASE: 12층, 768 차원 embedding  
  - LARGE: 24층, 1024 차원  
  - X-LARGE: 48층, 1280 차원 (약 1B 파라미터)  
- **프로젝션 및 코드 임베딩**  
  - 각 클러스터 타깃 별로 투영 행렬 적용 후 cosine 유사도로 softmax  

***

## 4. 성능 향상 및 한계

### 4.1 성능 향상
- Librispeech 960h fine-tuning
  - X-LARGE 모델: test-other WER 2.9% (기존 대비 13% 상대 개선)  
- 저자원 시나리오 (10분–100시간)
  - 10분: WER test-clean 4.6%, test-other 6.8%  
  - 1시간: WER test-other 4.8%  
  - 각종 semi-supervised 대비 우수  

### 4.2 한계
- **연산 비용**: 대규모 트랜스포머와 반복 클러스터링으로 자원 요구가 높음  
- **단일 언어·도메인 실험**: 영어 오디오에 한정, 다언어 적용성 추가 검증 필요  
- **타깃 클러스터 의존성**: 초기 k-평균 품질에 민감  

***

## 5. 일반화 성능 향상 가능성
- **마스킹 예측**: 입력의 다양한 프레임 컨텍스트 학습으로 소음·발화자 변이에 강건  
- **반복적 타깃 정제**: 학습이 진전될수록 더 정교해지는 클러스터가 일반화 능력 향상에 기여  
- **모델 크기 확장**: X-LARGE 모델에서도 일관된 성능 향상 관찰  
- **비영어·비음성 태스크**: 같은 masked-prediction 프레임워크를 타 언어 음향 또는 오디오 이벤트 검출에 적용 가능  

***

## 6. 향후 연구 영향 및 고려 사항
- **멀티태스크·다언어 확장**: HuBERT 표현을 여러 언어 및 태스크(CLS, 감정 인식 등)에 전이  
- **효율적 클러스터링**: 더 적은 반복으로 고품질 단위 발굴하는 알고리즘 탐색  
- **Self-Training 결합**: HuBERT 사전 학습 후 pseudo-labeling을 통한 추가 성능 부스트  
- **경량화 모델**: 실시간·임베디드 적용을 위한 소형화·지식 증류 방법 연구 강조  

**결론**: HuBERT는 마스킹 예측과 반복 정제를 결합해 연속 음성 표현 학습의 새 장을 열었으며, 차세대 음성 AI 연구에서 핵심 프레임워크로 자리 잡을 잠재력이 크다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/ab38ddac-1730-4760-bcc2-d3a3e2bf2398/2106.07447v1.pdf)
