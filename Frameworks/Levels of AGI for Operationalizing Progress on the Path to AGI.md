# Levels of AGI for Operationalizing Progress on the Path to AGI

## 1. 핵심 주장 및 주요 기여

본 논문은 Google DeepMind 연구진(Morris et al., 2024)이 제안한 **AGI(Artificial General Intelligence) 발전 경로를 체계적으로 측정하기 위한 수준별 분류 프레임워크**를 제시합니다. 핵심 기여는 다음과 같습니다:[1]

**2차원 매트릭스 온톨로지 제안**: AGI를 **성능(Performance)**과 **일반성(Generality)**이라는 두 차원으로 분류하여, 단일 종착점이 아닌 경로 중심의 정의를 제공합니다. 성능은 0~5단계(Emerging, Competent, Expert, Exceptional, Superhuman)로, 일반성은 Narrow와 General로 구분됩니다.[1]

**AGI 정의를 위한 6가지 원칙 정립**:[1]
1. 과정이 아닌 능력에 초점
2. 일반성과 성능 모두 강조
3. 인지 및 메타인지 과제 중심(물리적 과제 제외)
4. 배포가 아닌 잠재력 평가
5. 생태학적 타당성 확보
6. 단일 종착점이 아닌 경로 중심 접근

**자율성 수준(Levels of Autonomy) 도입**: AGI 능력과는 독립적인 6단계 자율성 분류를 제시하여, 배포 시 위험 평가를 위한 인간-AI 상호작용 패러다임을 제공합니다.[1]

## 2. 해결하고자 하는 문제 및 제안 방법

### 문제 정의

AGI에 대한 기존 정의들의 모호성과 일관성 부족으로 인해 다음과 같은 문제가 발생했습니다:[1]
- 모델 간 비교의 어려움
- 위험 평가 및 완화 전략 수립의 난항
- 정책 입안자와 규제 기관의 명확한 기준 부재
- AGI 경로상의 위치 파악 불가

### 제안 방법

논문은 **수학적 수식을 직접 제시하지 않고**, 대신 **개념적 분류 체계와 평가 원칙**을 제안합니다:

**레벨 매트릭스 시스템**:[1]
- **성능 차원**: 각 레벨은 숙련된 성인 인구의 백분위수로 정의됩니다
  - Level 1 (Emerging): 비숙련자 수준
  - Level 2 (Competent): 50백분위수 이상
  - Level 3 (Expert): 90백분위수 이상
  - Level 4 (Exceptional): 99백분위수 이상
  - Level 5 (Superhuman): 100% 초과
  
- **일반성 차원**: 
  - Narrow AI: 명확히 범위가 정해진 작업
  - General AI: 메타인지 능력을 포함한 광범위한 비물리적 작업

**벤치마크 설계 원칙**:[1]
- 언어 지능, 수학적·논리적 추론, 공간 추론, 대인·대내 사회 지능, 새로운 기술 학습 능력, 창의성을 포함
- **메타인지 과제 강조**: (1) 새로운 기술 학습 능력, (2) 도움을 요청할 시점 판단, (3) 마음 이론(Theory of Mind) 관련 사회적 메타인지 능력[1]
- **생태학적 타당성**: 실제 인간이 가치 있게 여기는 작업에 초점
- **살아있는 벤치마크(Living Benchmark)**: 새로운 작업을 지속적으로 생성하고 합의할 수 있는 프레임워크 포함[1]

## 3. 모델 구조 및 일반화 성능 향상 가능성

### 개념적 모델 구조

논문은 특정 신경망 아키텍처를 제안하지 않지만, **AGI 시스템이 갖춰야 할 구조적 요구사항**을 제시합니다:[1]

**메타인지 능력**:[1]
- 새로운 작업 학습을 위한 전략 선택 능력
- 모델 자체 능력의 한계 인식(모델 보정, calibration)
- 사용자 모델링을 통한 정렬(alignment) 지원

**다중 모달 처리**: 텍스트, 이미지, 음성 등 다양한 입력 통합이 일반성 향상의 핵심[2][3]

### 일반화 성능 향상 메커니즘

**전이 학습(Transfer Learning) 중심 접근**:[4][5][6]
- 한 도메인에서 학습한 패턴을 다른 도메인에 적용하는 능력이 일반화의 핵심
- 2025년 연구들은 다중 작업 학습(multitask learning)과 전이 학습이 재훈련 최소화와 함께 AGI 잠재력을 향상시킴을 보고[5]

**메타학습과 적응 능력**:[7][4]
- 고수준 지식의 재사용과 적용이 전문가와 일반가를 구분하는 핵심
- KIX(Knowledge-Interaction-eXecution) 같은 메타인지 프레임워크가 자율적 일반가 행동의 가능성 제시[4]

**추론 능력의 질적 도약**:[8][2]
- GPT-5(2025년 8월 출시)는 추론, 코딩, 다중 학문 작업에서 상당한 개선 보여[2]
- ARC-AGI 벤치마크에서 OpenAI의 o3 모델이 87.5% 달성(이전 시스템 5%에서 도약)[9]

**현재 한계점**:[10][11][1]
- 2023년 9월 기준, ChatGPT와 Bard 같은 프론티어 언어 모델은 일부 작업(짧은 에세이 작성, 간단한 코딩)에서 "Competent" 수준이나, 대부분의 작업(수학, 사실성)에서 여전히 "Emerging" 수준[1]
- 2025년 10월 현재, 전문가들은 AGI 도달 확률이 50%가 되는 시점을 2040-2060년으로 추정[11]

## 4. 성능 향상 및 한계

### 성능 향상 요소

**벤치마크 진화**:[12][13][10]
- ARC-AGI-2(2025년 3월 출시)는 효율성 지표 도입으로 계산 효과성도 평가[10]
- 2024년 ARC Prize 경쟁에서 성능이 33%에서 55.5%로 향상[13][14]
- 주요 접근법: AI 지원 프로그램 합성, 테스트 타임 학습(Test-Time Training), 혼합 방법[13]

**멀티모달 통합**:[3][2]
- 텍스트, 이미지, 오디오를 결합한 AI가 맥락 이해 향상, 새로운 작업에 대한 적응성 개선, 단일 입력 스트림 의존으로 인한 오류 감소[2]
- GPT-4o에서 GPT-5로의 개선: 이미지 지원, 음성 지원, 훨씬 큰 컨텍스트 창, 수학 능력 향상[8]

### 주요 한계점

**일반화의 본질적 어려움**:[15][12][1]
- 완전히 새로운 문제에 대한 적응 능력(유동 지능, fluid intelligence) 측정이 핵심이나, 현재 시스템은 사전 준비된 패턴 검색에 의존[15]
- ARC-AGI 벤치마크의 약 50% 작업이 무차별 대입(brute force) 방법에 취약하여, 일반화 가능한 추론보다 계산 능력 보상[12]

**데이터 및 컴퓨팅 제약**:[16][9]
- 고품질 훈련 데이터 고갈 예상: 2028년까지[9]
- 에너지 소비: AGI 시스템이 2030년까지 미국 전력의 8.4% 소비 가능[9]
- Epoch AI(2024) 보고: 2030년까지 2×10²⁹ FLOP 모델 훈련 가능하나 전력, 칩 제조, 데이터 부족, 처리 지연 등 제약 존재[16]

**메타인지 및 자율 학습**:[6][5][1]
- 인간은 수백만 개의 예제 없이도 학습 가능하나 현재 AI는 대규모 데이터셋 필요[5]
- AGI 정의의 핵심인 자율 학습과 자기 개선 능력이 여전히 부족[17][6]

**정렬(Alignment) 문제**:[18][19][20]
- 정렬된 AGI는 오히려 인간에 의한 재앙적 오용(catastrophic misuse) 위험 증가 가능[18]
- 가치 정렬, 제어, 투명성 확보가 기술적 진보와 함께 필수적[19][20]

## 5. 향후 연구 방향 및 고려사항

### 최신 연구 동향 기반 미래 방향

**새로운 알고리즘 패러다임 필요**:[10][13]
- ARC-AGI-2 결과는 단순 스케일링(scaling)만으로는 불충분함을 보여[10]
- 테스트 타임 적응(test-time adaptation) 알고리즘과 새로운 AI 시스템이 인간 수준의 효율성 달성에 필수[10]

**신경-심볼릭 AI와 인과 학습**:[21][2]
- 규칙 기반 논리와 신경망 결합하는 신경-심볼릭 AI가 추세[2]
- 언어 예측을 넘어 인과 학습, 상징적 추론, 세계 모델링에 초점[21]

**연합 학습과 에너지 효율**:[2]
- 분산 데이터 학습으로 프라이버시 강화
- 저비용 고효율 AI 칩 개발로 더 빠른 계산 가능[2]

**구체화된 AI(Embodied AI)**:[22]
- 물리적 구현이 AGI 필수 요소는 아니지만, 일부 인지 작업의 세계 지식 구축에 필요할 수 있음[1]
- Friston의 능동적 추론(active inference) 원칙과 통합된 프레임워크 제안[22]

### 안전성 및 윤리적 고려사항

**단계별 위험 프로파일**:[23][1]
- Level 1-2 (Emerging/Competent AGI): 오용 위험(부주의, 우발적, 악의적)
- Level 3 (Expert AGI): 경제적 혼란, 대량 실직
- Level 4-5 (Exceptional AGI/ASI): 정렬 실패로 인한 실존적 위험(x-risk)
- Anthropic의 ASL(AI Safety Level) 정책과 연계 필요[1]

**거버넌스와 규제**:[24][23][21]
- 글로벌 AI 거버넌스 필요성 증대[21]
- 투명성, 설명 가능성, 윤리적 프레임워크 구축 강조[24][2]
- AGI-의식(consciousness) 인터페이스, 집단 지능 시스템 같은 신규 연구 프론티어[24]

### 벤치마킹 및 평가

**OECD AI 능력 지표 프레임워크**:[25]
- 인간 능력 도메인 전반에 걸쳐 AI 발전을 체계적으로 비교
- Level 5 성능이 모든 척도에서 달성되면 인간 수준 일반 지능의 가능한 벤치마크[25]

**다학제 협력 필요**:[26][24]
- 신경과학, 인지과학, 윤리학 통합
- 생의학 연구, 나노기술, 에너지 연구, 인지 향상 등 다양한 도메인에 AGI 적용[26]

### 산업 및 사회적 영향

**AGI 타임라인 가속화**:[11][16][9]
- 2,778명의 AI 연구자 대상 최대 규모 설문: 고수준 기계 지능(HLMI) 중간 예측이 2060년에서 2047년으로 13년 앞당겨짐[9]
- Adam Gleave(Center for AI Safety): 2028년 말까지 AGI 점수 95% 달성 확률 50%, 2030년 말까지 80%[8]

**경제적 변혁**:[27][2][1]
- 광범위한 노동 대체 기준 도달 시 지정학적·경제적 우위 획득
- 스마트 그리드, 재생에너지 전환 등 인프라 혁신[27]

**AI 리터러시와 교육**:[21]
- AI 엔지니어링, AI 윤리, AI 해석 가능성이 모든 분야의 핵심 역량으로 부상[21]

## 결론

본 논문은 AGI를 단일 종착점이 아닌 **경로 중심의 다차원 진화**로 재정의함으로써, 연구자와 정책 입안자에게 명확한 소통 언어를 제공합니다. 일반화 성능 향상의 핵심은 **메타인지 능력, 전이 학습, 멀티모달 통합**에 있으며, 현재의 주요 한계는 **진정한 자율 학습 부족, 데이터·컴퓨팅 제약, 정렬 문제**입니다.[4][5][18][9][2][1]

향후 연구는 단순 스케일링을 넘어 **신경-심볼릭 AI, 테스트 타임 적응, 인과 추론** 등 새로운 패러다임으로 전환해야 하며, 기술 진보와 함께 **글로벌 거버넌스, 윤리적 프레임워크, 안전성 연구**를 병행해야 합니다. ARC-AGI-2와 OECD AI 능력 지표 같은 벤치마크가 진정한 일반 지능 측정의 북극성으로 작용하며, 다학제 협력과 산업-학계 연계가 AGI 실현의 핵심 요소입니다.[23][19][12][26][25][24][10][21][2]

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2a170b21-bbbe-4734-8846-cca459c2dfa3/2311.02462v5.pdf)
[2](https://www.sidetool.co/post/ai-and-the-quest-for-agi-where-are-we-now/)
[3](https://www.aimoneysimplified.com/post/latest-advancements-in-artificial-general-intelligence)
[4](https://arxiv.org/pdf/2402.05346.pdf)
[5](https://airesearchlearners.com/artificial-general-intelligence-capabilities-2/)
[6](https://matthopkins.com/technology/what-is-agi-really-and-how-close-are-we-in-2025/)
[7](https://arxiv.org/pdf/2401.06256.pdf)
[8](https://ai-frontiers.org/articles/agis-last-bottlenecks)
[9](https://www.linkedin.com/pulse/agi-tipping-point-why-2025-could-year-everything-changes-jha-1orxf)
[10](https://labs.adaline.ai/p/what-is-the-arc-agi-benchmark-and)
[11](https://etcjournal.com/2025/10/17/status-of-artificial-general-intelligence-agi-october-2025/)
[12](https://arxiv.org/html/2505.11831v1)
[13](https://the-decoder.com/agi-benchmark-arc-remains-unresolved-in-2024-despite-significant-progress/)
[14](https://arxiv.org/abs/2412.04604)
[15](https://arcprize.org/arc-agi)
[16](https://research.aimultiple.com/artificial-general-intelligence-singularity-timing/)
[17](https://www.codagni.com/blog/understanding-agi-what-it-is-and-where-we-stand-in-2025)
[18](https://arxiv.org/pdf/2506.03755.pdf)
[19](https://www.nedcapital.co.uk/artificial-general-intelligence-agi-risks-should-boards-prepare-now/)
[20](https://deepmind.google/blog/taking-a-responsible-path-to-agi/)
[21](https://www.usaii.org/ai-insights/artificial-general-intelligence-challenges-and-opportunities-ahead)
[22](https://arxiv.org/pdf/2402.03824.pdf)
[23](https://arxiv.org/pdf/2412.14186.pdf)
[24](https://pmc.ncbi.nlm.nih.gov/articles/PMC11897388/)
[25](https://www.oecd.org/en/publications/introducing-the-oecd-ai-capability-indicators_be745f04-en/full-report/component-5.html)
[26](https://www.nature.com/articles/s41598-025-92190-7)
[27](https://www.iberdrola.com/about-us/our-innovation-model/general-ai)
[28](https://arxiv.org/pdf/2311.02462.pdf)
[29](https://arxiv.org/pdf/2306.05480.pdf)
[30](http://arxiv.org/pdf/2405.10313.pdf)
[31](https://arxiv.org/pdf/2309.01622.pdf)
[32](https://arxiv.org/pdf/2310.15274.pdf)
[33](https://www.sciencedirect.com/science/article/abs/pii/S1367578825000367)
[34](https://arcprize.org/blog/announcing-arc-agi-2-and-arc-prize-2025)
[35](https://www.fastcompany.com/91358434/these-two-game-changing-breakthroughs-advance-us-towards-artificial-general-intelligence)
[36](https://www.emergentmind.com/topics/abstraction-and-reasoning-corpus-for-artificial-general-intelligence-arc-agi)
[37](https://hai.stanford.edu/ai-index/2025-ai-index-report)
[38](https://www.emerald.com/jtf/article/doi/10.1108/JTF-04-2025-0078/1302704/From-AI-to-Artificial-General-Intelligence-AGI)
[39](https://ieeexplore.ieee.org/iel8/6287639/10820123/11096544.pdf)
[40](https://lewish.io/posts/arc-agi-2025-research-review)
[41](https://arxiv.org/abs/2409.06800v1)
[42](https://arxiv.org/pdf/2303.15935.pdf)
[43](https://arxiv.org/pdf/2401.01001.pdf)
[44](https://arxiv.org/abs/2110.14378)
[45](https://pmc.ncbi.nlm.nih.gov/articles/PMC9163040/)
[46](https://ai-frontiers.org/articles/ai-alignment-cannot-be-top-down)
[47](https://arxiv.org/html/2311.02462v5)
[48](https://alignmentforum.org/posts/PMc65HgRFvBimEpmJ/legible-vs-illegible-ai-safety-problems)
[49](https://www.lesswrong.com/posts/rZQjk7T6dNqD5HKMg/abstract-advice-to-researchers-tackling-the-difficult-core)
[50](https://www.oecd.org/content/dam/oecd/en/publications/reports/2025/11/oecd-ai-capability-indicators-technical-report_d3762d1a/9cdb3dd1-en.pdf)
[51](https://www.linkedin.com/pulse/agi-2025-progress-timeline-safety-helena-ristov-rtdif)
[52](https://openai.com/safety/how-we-think-about-safety-alignment/)
