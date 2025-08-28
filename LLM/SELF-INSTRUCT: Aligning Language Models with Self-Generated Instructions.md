# SELF-INSTRUCT: Aligning Language Models with Self-Generated Instructions

**핵심 주장 및 주요 기여**  
이 논문은 *대규모 언어 모델(LM)*이 스스로 생성한 *instruction–input–output* 데이터를 활용해 사람 손끝 도움 없이 지시 추종 능력을 대폭 향상시킬 수 있음을 보인다. 주요 기여는 다음과 같다:  
1. **SELF-INSTRUCT 프레임워크**: 소량(175개)의 인간 작성 시드(seed) 작업을 시작으로, LM이 스스로 새로운 지시(Instruction)와 대응되는 예시(input–output)를 생성·필터링·추가하는 *반복 부트스트랩* 알고리즘을 제안.  
2. **대규모 합성 데이터**: vanilla GPT-3를 이용해 약 52K개의 지시, 82K개의 인스턴스를 생성.  
3. **모델 파인튜닝 효과**: SELF-INSTRUCT 데이터로 GPT-3를 미세조정한 GPT3SELF-INST 모델이 SUPER-NI 벤치마크에서 +33.1%p 개선을 달성, InstructGPT001 수준에 근접.  
4. **실사용 지시 일반화**: 이메일 작성·코딩·논리 변환 등 252개의 전문가 작성 “실사용 지시”에 대한 인간 평가에서, GPT3SELF-INST가 기존 공개 데이터 기반 파인튜닝 모델을 크게 앞서며 InstructGPT001 대비 오차율 5% 이내 성능.  

***

## 1. 해결 문제  
- **인간 작성 지시 데이터 한계**: 기존 instruction-tuning은 사람의 지시 예시 의존 → 데이터 양·다양성·창의성 부족 → 일반화 한계.  
- **목표**: 최소한의 인간 개입으로 다양한 작업 지시 및 예시 확보 → 언어 모델 자체로 지시 데이터를 생성·확장 → 모델의 *제로샷·광범위 일반화* 능력 제고.  

## 2. 제안 방법  
SELF-INSTRUCT는 다음 네 단계로 구성된다.  

1) **Instruction Generation**  
   - 시드 지시 175개에서 매 단계 8개(in-context 예시) 샘플링 → “Task: …” 템플릿으로 LM에 신규 지시  생성.  

2) **Classification Task Identification**  
   - 생성된 지시가 *분류 과제*인지 판단.  
   - Few-shot 제공된 12개 분류/19개 비분류 지시 예시로 LM에 질문.  

3) **Instance Generation**  
   - 분류 과제: **Output-first** 방식 – 가능한 클래스 레이블 먼저 생성 후, 각 레이블별 입력 생성.  
   - 비분류 과제: **Input-first** 방식 – 필요한 입력 먼저 생성 후, 그에 대응하는 출력 생성.  

4) **Filtering & Postprocessing**  
   - ROUGE-L 중복도 <0.7, 불가능 키워드(“image”, “graph” 등) 제외.  
   - 중복·부적절 예시 제거, 길이·형식 기준 휴리스틱 적용.  

이 과정을 수차례 반복하여 최종 52,445개 지시와 82,439개 인스턴스를 확보(Table 1).  

### 수식화  
지시 $$I_t$$와 입력–출력 쌍 $$\{(X_{t,i},Y_{t,i})\}\_{i=1}^{n_t}$$ 로 구성된 데이터 $$\mathcal{D}=\{(I_t,X_{t,i},Y_{t,i})\}$$.  
모델 파인튜닝은 표준 최대우도 추정:  

$$
\mathcal{L}(\theta) = -\sum_{t,i}\log p_\theta(Y_{t,i}\mid I_t,\,X_{t,i})  
$$  

***

## 3. 모델 구조 및 학습  
- **기저 모델**: GPT-3 “davinci” (175B 파라미터)  
- **파인튜닝**: OpenAI API, prompt–loss weight = 0, 2 epochs  
- **템플릿 다양화**: “Task:”, “Input:”, “Output:” 유·무, 줄바꿈 수 등 포맷 랜덤화로 강건성 향상  

***

## 4. 성능 향상 및 일반화  
### 4.1 SUPER-NI 벤치마크(Zero-Shot)  
| 모델                         | 파라미터 | ROUGE-L |
|------------------------------|---------:|--------:|
| GPT-3 (vanilla)              | 175B     |    6.8  |
| T0                           | 11B      |   33.1  |
| GPT3 + T0 데이터 파인튜닝    | 175B     |   37.9  |
| **GPT3SELF-INST (Ours)**     | 175B     |   39.9  |
| InstructGPT001               | 175B     |   40.8  |
| GPT3 + SUPERNI 데이터 파인튜닝 | 175B   |   49.5  |

- GPT3SELF-INST가 GPT-3 기준 +33.1%p 대폭 성능 향상.  
- InstructGPT001과 동등 수준 근접.  
- SUPERNI 데이터 병합 시 추가 향상 확인.  

### 4.2 사용자 지향 신규 지시(252개) 인간 평가  
- 4단계(A–D) 등급 체계  
- GPT3SELF-INST가 GPT-3 기반 공개 데이터 파인튜닝 모델 대비 월등 우수  
- InstructGPT001 대비 “수용 가능 응답”(A+B) 비율 차이 5%p 이내  

***

## 5. 주요 한계  
- **tail 현상**: 저빈도 작업이나 언어 패턴 학습 미흡 가능성  
- **대형 모델 의존성**: 대규모 LM 필요 → 자원 장벽  
- **편향 증폭 위험**: 모델 고유 사회·문화적 편향 강화 우려  

***

## 6. 향후 연구 시사점  
- *데이터 품질* 개선: InstructGPT003로 출력 재생성→성능 10%p 추가 향상  
- *소형 모델 확대*: 중·소형 모델에서 SELF-INSTRUCT 효과 검증  
- *보상 모델 활용*: RLHF 등 보상 함수 도입해 편향·오류 필터링  
- *멀티모달 확장*: 이미지·음성 지시 데이터로 SELF-INSTRUCT 일반화  

SELF-INSTRUCT는 **“모델 자체 생성 데이터”**로 instruction-tuning의 새로운 패러다임을 제시하며, **자동 확장 가능한 지시 데이터** 확보를 통해 언어 모델의 **제로쇼 및 광범위 일반화** 능력을 획기적으로 개선할 수 있음을 입증했다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/6c397b9e-645a-4cb5-b663-6123f1bca894/2212.10560v2.pdf)
