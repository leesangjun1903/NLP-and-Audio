# CogLTX: Applying BERT to Long Texts

**핵심 주장 및 주요 기여**  
CogLTX는 BERT의 512토큰 제한을 극복하기 위해 인간의 작업 기억(work­ing memory) 이론을 차용하여 **“핵심 문장”**만을 선별·연결해 추론함으로써, 입력 길이에 상관없이 일정한 메모리 사용으로 장문 이해를 가능케 한다.  
1. 인간의 인지 모델을 본떠, 입력 문서를 동적 블록으로 분할한 뒤, “Judge” BERT를 통해 관련성 점수를 부여해 핵심 블록을 재현(rehearsal)·소거(decay) 과정을 거쳐 순차적·반복적으로 회수(MemRecall)  
2. 핵심 블록을 연결한 짧은 문장열 z를 “Reasoner” BERT에 입력해 문항응답·분류·토큰 태깅 등 다양한 downstream 과제 수행  
3. 정답 스팬이나 지원 문장 주석이 없을 때는 **“중재(intervention)”** 기법으로 각 블록 제거 전후 Reasoner 손실 변화를 측정해 유사 감독 신호 생성  
4. 입력 길이에 독립적인 메모리·시간 복잡도(메모리 O(1), 시간 On) 달성

***

## 1. 해결하고자 하는 문제  
전통적 BERT는 토큰 길이 $$L$$에 대해 시간·공간 복잡도 $$O(L^2)$$를 지녀, 512토큰 이상 입력 시 메모리 초과 혹은 계산 불가 문제가 발생한다.  
- 슬라이딩 윈도우 방식은 장거리 의존성 포착에 한계  
- Transformer 경량화 기법은 커스텀 커널·검증 미흡  

***

## 2. 제안 방법  
### 2.1. 블록 분할 및 핵심 블록 가정  
- 입력 긴 문장 $$x$$을 최대 길이 $$B$$ 블록 $$\{x_0,\dots,x_T\}$$으로 분할  
- **가정:** downstream 과제 수행에 충분한 핵심 블록들의 연결 $$z = [x_{i_0},\dots,x_{i_n}]$$이 존재하며, $$\lvert z\rvert \le L$$  

### 2.2. MemRecall 알고리즘  
1. 초기 $$z$$에 질문 혹은 무작위 스팬 삽입  
2. **Judge** BERT로 각 블록 $$x_i$$ 점수 $$\mathrm{judge}(z,x_i)=\mathrm{sigmoid}(\mathrm{MLP}(\mathrm{BERT}([\mathrm{CLS},z,\mathrm{SEP},x_i,\mathrm{SEP}]))$$ 부여  
3. **Retrieval competition:** 상위 점수 블록을 $$z$$에 추가  
4. **Rehearsal & Decay:** 전체 $$z$$에 대해 재점수화 후 낮은 점수 블록을 제거  
5. 다단계 반복으로 멀티-스텝 추론  

### 2.3. 학습  
- **Judge** 유·무감독 학습  
  - 유감독: 스팬 추출·지원 문장 주석 이용해 교차엔트로피 손실  
  - 무감독: 중재 기반 손실 변화(necessity·sufficiency)로 라벨 생성  
- **Reasoner** 일관성 유지를 위해 MemRecall 추출 입력(z)을 활용해 교차엔트로피 손실로 파인튜닝  

***

## 3. 모델 구조  
- 두 개의 BERT: **Judge**(블록 점수화) + **Reasoner**(실제 과제 수행)  
- Judge와 Reasoner는 독립 파라미터, 동시 최적화  
- 블록 분할은 동적 프로그래밍으로 구현  

***

## 4. 성능 향상 및 한계  
### 4.1. 성능  
- NewsQA: EM +5.6, F1 +3.8 개선 (RoBERTa-large 대비)  
- HotpotQA: BERT 계열 최상위 성능과 비슷한 Joint F1 달성  
- 20NewsGroups: 87.4% 정확도, Glove 초기화·중재 학습으로 개선  
- Alibaba 다중라벨: CogLTXlarge Macro-F1 97.2%  

### 4.2. 메모리·시간 복잡도  
- 메모리: 입력 길이 독립적 (학습 시 고정 메모리)  
- 시간: $$O(n)$$ 멀티스텝 1회 기준, 장문 입력 시 Sliding window, Vanilla BERT보다 우수  

### 4.3. 한계  
- **핵심 문장 가정**이 깨지는 과제(극히 상관없는 퍼센테이지)에서는 성능 저하 가능  
- 블록 경계 전후 **지시·대명사 해소**(coreference) 미반영  
- 효율적 베이지안 추론(q-분포) 도입 필요성  

***

## 5. 일반화 성능 향상 가능성  
- 중재 기반 무감독 학습은 레이블 없는 데이터에도 핵심 블록 인식 학습 유도  
- 멀티-스텝 MemRecall은 복합 질문·다중 문단 종속 관계 학습에 유리  
- Judge BERT가 다양한 도메인 텍스트에서 핵심 정보 선별 역량을 획득하면, Reasoner 일반화 성능 상승 기대  

***

## 6. 향후 연구 방향  
- **핵심 블록 분포 모형화:** 효율적 변분 추론(e.g., normalizing flows) 적용  
- **지시대명사 해결:** 위치 인식 retrieval or coreference 모듈 통합  
- **하이브리드 경량화:** Adapters·LoRA 등 파라미터 효율적 확장과 병합  
- **자기 감독 확장:** 사전학습 단계에서 대규모 장문 코퍼스에 CogLTX 적용  

CogLTX는 **장문 이해**를 위한 강력한 일반 전략으로, 복합 NLP 과제에서 BERT 활용 범위를 크게 확장할 것으로 기대된다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2be7b3b7-6650-4aaf-b780-1da7aa69d26b/NeurIPS-2020-cogltx-applying-bert-to-long-texts-Paper.pdf)
