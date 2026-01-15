
# O1 Replication Journey: A Strategic Progress Report – Part 1

## 1. 핵심 주장과 주요 기여 요약

"O1 Replication Journey"는 상하이 교통대학교와 GAIR의 연구팀이 OpenAI의 획기적인 O1 모델을 투명하게 복제하려는 시도를 기록한 실시간 진행 보고서입니다. 논문의 가장 근본적인 주장은 **"Journey Learning" 패러다임이 기존 "Shortcut Learning"을 근본적으로 초월할 수 있다**는 것입니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/101d1946-0a37-4e13-aac8-b22162f16bec/2410.18982v1.pdf)

논문의 중심 기여는 다음과 같습니다:

1. **새로운 학습 패러다임 제시**: 모델이 최종 정답만이 아니라 시행착오 경로, 반성, 백트래킹을 포함한 **전체 탐색 과정을 학습**하도록 유도합니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/101d1946-0a37-4e13-aac8-b22162f16bec/2410.18982v1.pdf)

2. **데이터 효율성의 극대화**: 단 327개의 훈련 샘플로 기존 방법 대비 8% 이상의 성능 향상을 달성했으며, 추가적인 기법 없이 순수 journey learning만으로 이를 이루었습니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/101d1946-0a37-4e13-aac8-b22162f16bec/2410.18982v1.pdf)

3. **개방 과학의 실천**: 전통적인 학술 논문과 달리 실시간으로 탐색 과정, 실패, 통찰력을 공유하여 AI 연구 커뮤니티의 집단적 비용을 감소시키려 합니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/101d1946-0a37-4e13-aac8-b22162f16bec/2410.18982v1.pdf)

4. **AI 과학 발견의 기초**: 성공과 실패를 포함한 완전한 탐색 과정을 문서화하여 미래의 과학적 발견 능력을 갖춘 AI 시스템 훈련 데이터로 활용할 수 있음을 제시합니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/101d1946-0a37-4e13-aac8-b22162f16bec/2410.18982v1.pdf)

***

## 2. 해결하는 문제, 제안 방법, 모델 구조, 성능 및 한계

### 2.1 핵심 문제 정의

O1 Replication Journey는 세 가지 차원의 문제를 해결하려고 합니다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/101d1946-0a37-4e13-aac8-b22162f16bec/2410.18982v1.pdf)

**기술적 문제:**
- OpenAI's O1 모델의 메커니즘이 베일에 감싸여 있음
- 복잡한 추론 작업에서 일반적인 LLM의 한계
- 모델의 자기 수정 및 반성 능력 부재

**연구 방법론적 문제:**
- 대규모 팀 기반 AI 프로젝트의 정보 고립
- 지연된 성과 공유로 인한 연구자 번아웃
- 개별 기여도 인식의 어려움

**AI 발전 구조적 문제:**
- 투명성 부족이 기술 진전을 방해
- 학습 자료로서의 시행착오 경로의 가치 미인식

### 2.2 제안 방법론

#### A. Shortcut Learning vs. Journey Learning 패러다임 전환 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/101d1946-0a37-4e13-aac8-b22162f16bec/2410.18982v1.pdf)

**Shortcut Learning의 특징:**
- 빠른 결과 지향
- 높은 데이터 의존성
- 제한된 일반화 능력
- 자기 수정 메커니즘 부재
- 인간 유사도: 시험 준비식 교육

**Journey Learning의 특징:**
- 깊은 인과관계 학습
- 강력한 추론 능력
- 지속적 자기 개선
- 강한 일반화 능력
- 높은 해석 가능성
- 인간 유사도: 평생 학습

#### B. 추론 트리 구성 알고리즘 (Reasoning Tree Construction) [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/101d1946-0a37-4e13-aac8-b22162f16bec/2410.18982v1.pdf)

정책 모델 π를 사용하여 추론 트리를 구성합니다:

$$T = (q, \{s_i\}, a)$$

여기서:
- $q$ = 문제 (루트 노드)
- $s_i$ = 추론 단계 (중간 노드)
- $a$ = 최종 답 (리프 노드)

**생성 프로세스:**
1. 루트에서 $w$개의 가능한 첫 단계 생성
2. 각 노드에서 최대 깊이 $D$까지 반복
3. 리워드 모델로 부정확한 단계 제거

$$\text{총 생성 수감소: } \frac{n^{D-1}}{n-1} \rightarrow nK^D$$

여기서 $K$는 각 반복에서 유지되는 최상위 후보의 수입니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/101d1946-0a37-4e13-aac8-b22162f16bec/2410.18982v1.pdf)

#### C. 장 사고 (Long Thought) 파생 과정 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/101d1946-0a37-4e13-aac8-b22162f16bec/2410.18982v1.pdf)

1. **ShortCut 경로 추출**: 정답 리프 노드까지의 최단 경로 추출
2. **DFS 트래버설**: Depth-First Search를 사용한 전체 탐색 경로 생성
3. **제약 적용**: 각 정답 경로 노드는 최대 K번의 시행착오 시도 허용
4. **필터링 및 정제**: GPT-4o를 사용하여 연속성과 일관성 개선

#### D. 단계별 보상 모델 (Process-Level Reward Model) [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/101d1946-0a37-4e13-aac8-b22162f16bec/2410.18982v1.pdf)

단계 수준의 평가로 세밀한 감독을 제공합니다:

$$R_{\text{step}}(s_i | q, s_{ < i}) \in \{0, 1\}$$

**모델 비교:**
- Math-Shepherd: F1 score 0.734~0.855
- **O1-mini: F1 score 0.880 (최우수)**

### 2.3 훈련 파이프라인 및 모델 구조

#### Phase 1: 감독 미세조정 (SFT)

**1단계 - Shortcut Learning:**
- Abel 데이터셋: 120,000 예제
- PRM800K 부분집합: 6,998 예제  
- 1에포크 훈련
- 목적: 응답 형식 학습

**2단계 - Journey Learning:**
- 327개 Long Thought 예제
- 3에포크 훈련
- 목적: 오류 탐지, 반성, 정정, 백트래킹 능력 개발

#### Phase 2: 직접 선호도 학습 (DPO) [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/101d1946-0a37-4e13-aac8-b22162f16bec/2410.18982v1.pdf)

$$\mathcal{L}_{\text{DPO}} = -\log \sigma\left(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right)$$

- 절차:
  - MATH Train 데이터셋 (12,000 예제)
  - 질문당 20개 응답 생성
  - nucleus sampling: top_p=0.95, temperature=0.7
  - 5개 선호도 쌍 생성

**기본 모델:** DeepSeekMath-7B-Base [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/101d1946-0a37-4e13-aac8-b22162f16bec/2410.18982v1.pdf)

### 2.4 성능 향상 결과

| 훈련 설정 | deepseek-sft-abel | deepseek-sft-prm800k | 개선율 |
|-----------|-------------------|----------------------|-------|
| SFT-phase1 (기본) | 0.372 | 0.290 | 기준 |
| SFT-phase2-shortcut | 0.386 | 0.348 | +0.4%, +6.0% |
| **SFT-phase2-journey** | **0.470** | **0.428** | **+8.4%, +8.0%** |
| DPO | 0.472 | 0.440 | +10.3%, +10.0% |

**주목할 점:**
- Journey Learning: 327개 샘플로 8.4%/8.0% 향상
- Shortcut Learning (같은 데이터): 0.4%/6.0% 향상만 달성
- **경로 학습의 가치 명확히 입증** [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/101d1946-0a37-4e13-aac8-b22162f16bec/2410.18982v1.pdf)

### 2.5 인식된 한계

**1. DPO의 제한적 효과**
- Journey Learning 대비 소폭 개선 (2.3%, 1.9%)
- 저자 인정: "초기 탐색 결과"로 향후 개선 필요

**2. 모델 붕괴 위험**
- LLM 자체 생성 데이터 활용 시 분포 축약 현상
- 해결책: 인간 데이터와 합성 데이터 균형 유지 필요 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/101d1946-0a37-4e13-aac8-b22162f16bec/2410.18982v1.pdf)

**3. 확장성 제약**
- 현재 7B 기본 모델, 더 큰 모델 가능성 미탐색
- 전체 O1 성능 달성 불가능 (차원이 다른 규모)

**4. RL 단계 부재**
- Pure RL 없이 SFT+DPO만 활용
- DeepSeek-R1처럼 RL-based 접근 아직 미시도

***

## 3. 모델의 일반화 성능 향상 가능성 (심화 분석)

### 3.1 일반화 성능 향상의 이론적 근거

Journey Learning은 세 가지 메커니즘을 통해 일반화 능력을 향상시킵니다:

#### 1) **다양한 오류 경로 학습을 통한 강건성**

모델이 정답 경로만이 아니라 여러 틀린 경로와 교정 과정을 학습함으로써:

$$P(\text{correct}|\text{new problem}) = \int P(\text{correct}|x, \text{reasoning path}) \cdot P(\text{path}|x) dx$$

- 훈련 분포 외 문제에 대한 적응 능력 증대
- 오류 회복 메커니즘의 일반화
- 새로운 도메인 문제에 대한 견고성

#### 2) **인과 관계 학습**

Shortcut Learning은 **표면적 특성과 답의 상관관계**만 학습하지만, Journey Learning은 **근본적 인과 구조**를 학습합니다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/101d1946-0a37-4e13-aac8-b22162f16bec/2410.18982v1.pdf)

| 측면 | Shortcut | Journey |
|------|----------|---------|
| 학습 깊이 | 표면 특성과 단순 상관관계 | 깊은 인과관계 |
| 추론 능력 | 복잡 추론에서 약함 | 강함 (인간 수준) |
| 자기개선 | 자기 수정 메커니즘 부재 | 지속적 자기 평가 및 개선 |
| 일반화 | 분포 외에서 급격히 악화 | 새로운 상황 처리 가능 |

#### 3) **메타 학습 효과 (Meta-Learning)**

시행착오를 통해 모델은 "어떻게 문제를 푸는가"뿐 아니라 "어떻게 접근하는가"를 학습합니다:

$$f_\theta(\text{new problem}) = \text{Aggregate}(\text{learned strategies from trials and errors})$$

- 학습 과정 자체가 메타-학습 데이터
- 새로운 문제 유형에 대한 빠른 적응
- 도메인 이동 (domain shift)에 강함

### 3.2 실증적 증거: MATH 벤치마크 결과 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/101d1946-0a37-4e13-aac8-b22162f16bec/2410.18982v1.pdf)

327개 Journal Thought 예제를 사용한 성능:

**MATH500 (테스트 세트):**
- DeepSeek-SFT-Abel 기본모델: 37.2%
- + Shortcut Learning (327개): 38.6% (+1.4%)
- + Journey Learning (327개): **47.0% (+9.8%)**

**MATH500 (PRM800K 데이터):**
- 기본모델: 29.0%
- + Shortcut Learning: 34.8% (+5.8%)
- + Journey Learning: **42.8% (+13.8%)**

**해석:**
같은 327개 데이터에서도 시행착오 정보 포함 여부로 8-13% 차이 → **경로 학습의 강력한 일반화 효과** [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/101d1946-0a37-4e13-aac8-b22162f16bec/2410.18982v1.pdf)

### 3.3 일반화 능력 강화 메커니즘

#### 메커니즘 1: 자기 교정 능력

Journey Learning 데이터 예시 (수학 문제 풀이): [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/101d1946-0a37-4e13-aac8-b22162f16bec/2410.18982v1.pdf)

```
【오류 시행】
"2220이 30의 배수인지 확인해보자... 아니다, 
합이 4이므로 3으로 나누어지지 않는다."

【반성】
"잠깐, 계산 실수가 있나? 2+2+2+0 = 6이다. 
6은 3으로 나누어진다."

【교정】
"따라서 2220은 30의 배수이다."
```

이 과정을 통해 모델은:
- 오류 인식 능력 개발
- 자기 검증 메커니즘 형성
- 새로운 문제에서 유사 오류 사전 방지

#### 메커니즘 2: 다양한 접근 전략 습득

논문의 다양한 케이스 분석: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/101d1946-0a37-4e13-aac8-b22162f16bec/2410.18982v1.pdf)

**케이스 1: 대수 문제 (다항식 곱셈)**
- 단계별 전개와 계수 비교

**케이스 2: 나머지 정리 (Remainder Theorem)**
- 다중 조건 활용 및 선형 시스템 풀이

**케이스 3: 수 이론 (30의 배수)**
- 인수분해 조건 (divisibility rules) 활용

모델이 문제 유형에 따라 다양한 전략을 학습 → 새로운 문제에서 최적 전략 선택 능력

#### 메커니즘 3: 시간적 추론 확장

Long Thought는 더 긴 추론 체인을 명시적으로 학습하게 함: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/101d1946-0a37-4e13-aac8-b22162f16bec/2410.18982v1.pdf)

- **토큰 수**: 평균 521행, 약 4,000-18,000 토큰
- **추론 깊이**: 수학 문제의 경우 최대 10단계 이상
- **반성 빈도**: "Wait", "Alternatively" 등 키워드 빈도 증가

이는 모델이 장기적 맥락 유지 능력 개발 → **더 복잡한 문제에 강해짐**

### 3.4 일반화성 향상의 정량적 지표

키워드 분석을 통한 추론 특성: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/101d1946-0a37-4e13-aac8-b22162f16bec/2410.18982v1.pdf)

| 키워드 | 의미 | 복잡도 상관 |
|--------|------|-----------|
| "Therefore" | 결론 도출 | 중간 |
| "Alternatively" | 경로 탐색 | 높음 |
| "Wait" | 자기 수정 | 높음 |
| "Let me compute" | 계산 전환 | 중간 |
| "Consider" | 조건 분석 | 높음 |

**발견**: 복잡한 문제일수록 "Alternatively"와 "Wait" 빈도 증가 → 모델이 자동으로 더 깊은 탐색 활성화 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/101d1946-0a37-4e13-aac8-b22162f16bec/2410.18982v1.pdf)

### 3.5 향후 일반화 성능 개선 경로

저자들이 제시한 향후 계획: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/101d1946-0a37-4e13-aac8-b22162f16bec/2410.18982v1.pdf)

1. **Long Thought Scaling Law 실험**
   - 훈련/추론 시간에 따른 성능 곡선 파악
   - 최적 투자 비율 규명

2. **인간-AI 협업**
   - 327개 → 수천 개 고품질 long thought 확대
   - 도메인 다양성 확충 (수학 외 과학, 기술)

3. **Process-Level RL/DPO 전환**
   - 현재: Outcome-level (최종 답) 기반
   - 향후: Process-level (각 단계) 기반 최적화
   - DeepSeek-R1 스타일 pure RL 가능성

4. **다중 에이전트 확장**
   - 비평자(Critic) 모델과 정책(Policy) 모델 상호작용 강화
   - Debate-based reasoning 도입

***

## 4. 논문의 영향력과 향후 연구 고려사항

### 4.1 이론적 및 실무적 영향

#### A. AI 연구 방법론의 변화

**기존 패러다임 (폐쇄형 장기 프로젝트):**
- 6-12개월 이상의 폐쇄된 팀 작업
- 최종 결과 기반 출판만
- 설패 과정 숨김
- 커뮤니티 피드백 지연

**제안 패러다임 (개방형 실시간 진행):**
- 주간 또는 월간 실시간 업데이트
- 성공과 실패 모두 공유
- 즉시적 커뮤니티 피드백
- 집단 학습 비용 감소 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/101d1946-0a37-4e13-aac8-b22162f16bec/2410.18982v1.pdf)

**영향:** 향후 대규모 AI 프로젝트가 이러한 투명한 접근을 따를 가능성 높음 → AI 연구 민주화 진전

#### B. 학습 패러다임의 근본적 전환

**Shortcut → Journey Learning 전환의 의미:**
- 단순 연관학습에서 **프로세스 학습**으로 전환
- Outcome-only 최적화에서 **Step-wise 감독**으로 전환
- 모델 성능에서 **모델 이해력**으로 평가 기준 이동 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/101d1946-0a37-4e13-aac8-b22162f16bec/2410.18982v1.pdf)

**영향:** 향후 LLM 개발에서 "어떻게 생각하는가"가 "무엇을 답하는가"만큼 중요해짐

#### C. AI-by-AI 과학 발견의 기초

논문이 강조하는 **Walnut Plan**: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/101d1946-0a37-4e13-aac8-b22162f16bec/2410.18982v1.pdf)
- 현재: 인간이 설계한 과학적 절차를 AI가 실행
- 미래: AI가 자체적으로 과학적 사고 과정을 발전시키고 새로운 발견 생성

현재 논문의 완전한 탐색 과정 기록 → 미래 AI의 과학 능력 훈련 데이터

### 4.2 향후 연구 시 고려할 점

#### 1. 데이터 품질과 편향 관리

**현재 상황:**
- 327개 high-quality long thought로 상당한 개선
- 하지만 규모 확대 시 품질 저하 위험

**고려사항:**
- Human-in-the-loop 검증 강화
- 자동 품질 점수 메트릭 개발
- 도메인 다양성 확보 (수학 외 분야)

**권장안:** 
```
규모 확대 시 품질 지표 정의:
Q_score = (consistency × completeness × relevance) / redundancy
```

#### 2. 모델 붕괴(Model Collapse) 방지

**문제:** 
LLM 자체 생성 데이터로 반복 훈련 시 다양성 손실 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/101d1946-0a37-4e13-aac8-b22162f16bec/2410.18982v1.pdf)

**해결 전략:**
1. Human-authored 데이터 일정 비율 유지 (>30%)
2. 각 반복마다 샘플 다양성 모니터링
3. Temperature scaling 동적 조정
4. 구역별(domain-specific) 데이터 분리 관리

#### 3. 계산 효율성과 비용 최적화

**현 상황:**
- Journey Learning으로 8% 향상 + DPO로 추가 2% (총 10%)
- 하지만 계산 비용은?

**분석 필요:**
$$\text{ROI} = \frac{\text{성능 향상 (\%)}}{\text{추가 계산 비용 (FLOPs)}}$$

**권장:**
- Inference-time scaling (o1 방식) vs Training-time scaling 비교
- 작은 모델의 journey learning이 큰 모델의 shortcut learning을 이기는지 검증

#### 4. 단계별 보상 모델의 정확성

**발견:**
O1-mini (F1: 0.880) vs Math-Shepherd (F1: 0.734)의 차이가 최종 성능에 미치는 영향 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/101d1946-0a37-4e13-aac8-b22162f16bec/2410.18982v1.pdf)

**고려사항:**
- PRM 정확도와 최종 모델 성능의 상관계수 정량화
- 약한 보상 신호로도 충분한지 탐색
- 도메인별 최적 PRM 선택

#### 5. 일반화 성능 벤치마크 확장

**현 상황:** MATH500 벤치마크만 주 평가

**확장 필요:**
| 벤치마크 | 도메인 | 추론 유형 |
|---------|--------|---------|
| MATH500 | 수학 | 계산 + 논증 |
| AIME | 경쟁수학 | 깊은 문제해결 |
| GSM8K | 상식 수학 | 생활 응용 |
| BIG-Bench | 일반추론 | 다분야 |
| HumanEval | 코딩 | 프로그래밍 |

#### 6. Process-Level RL로의 전환

**현 한계:**
- DPO는 outcome-level (최종 답)만 고려
- 각 단계별 품질 피드백 부족

**개선 방향:**
$$L_{\text{process-RL}} = \sum_{i=1}^{n} R_i(s_i) \log \pi_\theta(a_i | s_{<i})$$

여기서 $R_i$는 단계 $i$의 프로세스 보상

**DeepSeek-R1과의 비교:**
- DeepSeek: Pure RL (SFT 없음), 매우 큰 규모
- O1 Journey: SFT+DPO 하이브리드, 중간 규모
- 향후: 양쪽 접근의 장점 결합 시도

### 4.3 2024년 이후 관련 연구와의 통합 방안

#### 통합 1: STaR와의 결합 (자기 개선 강화)

**STaR (2022)**: 정답 주어졌을 때 rationale 재생성
**Journey Learning**: 전체 시행착오 경로 학습

**통합 방안:**
```
STaR-Journey Hybrid:
1. 정답 없는 경로에서 model이 생성한 rationale로 학습 (Journey)
2. 정답 주어졌을 때 최적 rationale 재생성 (STaR)
3. 둘을 결합하여 DPO 훈련
```

#### 통합 2: DeepSeek-R1의 Pure RL 활용

**DeepSeek-R1**: 결과 기반 보상만으로 RL 학습

**적용 가능성:**
- 현재 DPO의 제한적 효과 (10.3% 향상)
- Pure RL로 더 큰 향상 기대
- 다만, 계산 비용 매우 높음

**현실적 권장:**
- 소규모: Journey Learning + DPO (현 방식 권장)
- 대규모: Journey Learning + Process-Level RL (미래)

#### 통합 3: Inference-Time Scaling 결합

**현 상황**: 훈련 시간 최적화만 수행

**추론 단계 강화:**
- Long Thought 생성 후 Best-of-N 투표
- Process Reward Model로 상위 경로 선택
- 자기 일관성 디코딩

**예상 효과:**
- +추가 5-10% 성능 향상 (비용 증가)
- 특히 어려운 문제 (AIME 등)에서 효과적

***

## 5. 2020년 이후 관련 최신 연구 비교 분석

### 5.1 주요 선행 연구와의 시간적 진화

| 연도 | 연구 | 핵심 기술 | 성능 향상 | Journey와의 관계 |
|------|------|---------|----------|-----------------|
| 2022 | **STaR** | 자기 개선 루프 | 30배 모델 수준 | 개념적 선조 |
| 2022 | **CoT** | 단계별 추론 프롬팅 | 19-35% | 기초 기술 |
| 2023 | **ToT** | 트리 탐색 구조 | 4%→74% (Game of 24) | 구조적 유사 |
| 2024 | **PRM (Lightman)** | 단계별 보상 모델 | 5-15% | 핵심 컴포넌트 |
| 2024 | **Quiet-STaR** | 각 토큰 후 내부 rationale | +3-8% | 암묵적 사고 vs 명시적 |
| 2024 | **DeepSeek-R1** | Pure RL로 추론 발현 | AIME 79.8% | 대규모 RL 버전 |
| 2024 | **Inference Scaling** | 추론 시간 계산 증대 | 4배 효율 | 보완 기술 |
| **2024** | **O1 Journey** | **시행착오 전체 경로 학습** | **+8.4%/+8.0%** | **통합 혁신** |

### 5.2 Journey Learning의 차별 경쟁 우위

| 측면 | STaR | ToT | DeepSeek-R1 | **O1 Journey** |
|------|------|-----|-------------|----------------|
| **데이터 효율** | 중간 | 낮음 | 매우 높음 | **매우 높음** |
| **훈련 비용** | 중간 | 높음 | 극고 | **낮음** |
| **모델 크기** | 소형 | 소형 | 매우 큼 | **중형** |
| **일반화** | 보통 | 보통 | 우수 | **우수** |
| **투명성** | 제한 | 제한 | 제한 | **완전 공개** |
| **커뮤니티 기여** | 낮음 | 낮음 | 낮음 | **높음** |

### 5.3 각 방법의 보완성 분석

**STaR + Journey Learning 결합:**
```
장점: 자기 개선의 반복성 + 시행착오 경로의 명시성
기대 효과: STaR의 반복 >> Journey의 명시적 학습
```

**ToT + Journey Learning 결합:**
```
장점: 구조화된 탐색 + 비구조화된 자기 수정
기대 효과: 더 효율적인 탐색 공간 탐색
```

**RL (DeepSeek-R1) + Journey Learning:**
```
장점: Pure RL의 강력한 신호 + Journey의 데이터 효율
기대 효과: 더 강력한 추론 능력 + 낮은 계산 비용
```

### 5.4 향후 통합 연구 로드맵 (2025-2026)

**Phase 1 (2025년):** Journey Learning 확대
- 327개 → 5,000개 다양한 long thought 확보
- Process-level DPO 도입
- 다중 벤치마크 평가

**Phase 2 (2025년 후반):** RL 통합
- Journey Learning 데이터로 초기화된 정책 모델
- Process-level RL 훈련
- 각 도메인별 특화 모델

**Phase 3 (2026년):** 다중 모드 확장
- 비전-언어 모델에 Journey Learning 적용
- 코드 생성 분야 전문화
- 다언어 지원

***

## 결론

"O1 Replication Journey"는 단순한 모델 복제 시도를 넘어, **AI 연구 방법론, 학습 패러다임, 과학 발견 방식을 근본적으로 재설정하는 작업**입니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/101d1946-0a37-4e13-aac8-b22162f16bec/2410.18982v1.pdf)

### 핵심 성과:
1. **Journey Learning**: 시행착오 경로의 가치를 수학적으로 입증
2. **데이터 효율성**: 327개 샘플로 8% 향상 달성
3. **방법론 혁신**: 폐쇄에서 개방으로, 결과에서 과정으로
4. **미래 기초 마련**: AI-by-AI 과학 발견의 시작점

### 향후 고려사항:
- **기술**: Process-level RL 도입, Inference scaling 결합
- **규모**: 데이터 확대 시 품질-다양성 균형 유지  
- **방법론**: 더 많은 연구팀의 투명한 공개 연구 참여
- **평가**: 다양한 도메인과 다국어 벤치마크 확대

이 논문의 최대 기여는 **"좋은 답을 아는 것"만이 아니라 "좋은 답에 도달하는 과정을 배우는 것"이 진정한 지능**이라는 인간의 직관을 AI에 구현했다는 점입니다. 이는 향후 AI 시스템이 단순한 도구에서 진정한 "학습자"로 진화하는 길을 제시합니다.

***

## 참고: 주요 수식 정리

**1. 추론 트리 생성 복잡도 감소:**
$$\text{Original: } O(n^D), \quad \text{With Pruning: } O(nK^D)$$

**2. Long Thought 파생 (DFS 기반):**
$$\text{LongThought} = \text{Path}(\text{root} \rightarrow \text{leaf}) + \text{Trials}(\text{on correct nodes})$$

**3. DPO Loss 함수:**
$$L_{\text{DPO}} = -\log \sigma\left(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right)$$

**4. Process Reward Model:**

$$R_{\text{step}}(s_i | q, s_{ < i}) \in \text{ (또는 binary)}$$ [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/101d1946-0a37-4e13-aac8-b22162f16bec/2410.18982v1.pdf)

**5. 향후 권장 Process-Level RL:**
$$L_{\text{process-RL}} = \sum_{i=1}^{n} R_i(s_i) \log \pi_\theta(a_i | s_{ < i})$$

<span style="display:none">[^1_10][^1_11][^1_12][^1_13][^1_14][^1_15][^1_16][^1_17][^1_18][^1_19][^1_2][^1_20][^1_21][^1_22][^1_23][^1_24][^1_25][^1_26][^1_27][^1_28][^1_29][^1_3][^1_30][^1_31][^1_32][^1_33][^1_34][^1_35][^1_36][^1_37][^1_38][^1_39][^1_4][^1_40][^1_41][^1_42][^1_43][^1_44][^1_45][^1_46][^1_47][^1_48][^1_49][^1_5][^1_50][^1_51][^1_52][^1_53][^1_54][^1_55][^1_56][^1_57][^1_58][^1_59][^1_6][^1_60][^1_61][^1_62][^1_63][^1_64][^1_65][^1_66][^1_67][^1_68][^1_69][^1_7][^1_70][^1_71][^1_72][^1_73][^1_74][^1_75][^1_76][^1_77][^1_78][^1_79][^1_8][^1_80][^1_81][^1_82][^1_83][^1_84][^1_85][^1_86][^1_9]</span>

<div align="center">⁂</div>

[^1_1]: 2410.18982v1.pdf

[^1_2]: https://arxiv.org/abs/2404.03361

[^1_3]: https://dl.acm.org/doi/10.1145/3726010.3726041

[^1_4]: https://ieeexplore.ieee.org/document/11117035/

[^1_5]: https://arxiv.org/abs/2410.00151

[^1_6]: https://ieeexplore.ieee.org/document/11117006/

[^1_7]: https://aclanthology.org/2024.semeval-1.196

[^1_8]: https://www.semanticscholar.org/paper/9851db4cab76eed072355ec6d9d91ec187b3b13a

[^1_9]: https://ieeexplore.ieee.org/document/11116958/

[^1_10]: https://arxiv.org/abs/2403.14982

[^1_11]: https://arxiv.org/abs/2404.01961

[^1_12]: https://arxiv.org/abs/2410.05695

[^1_13]: https://arxiv.org/html/2503.05179v1

[^1_14]: http://arxiv.org/pdf/2309.15402.pdf

[^1_15]: https://arxiv.org/pdf/2412.10827.pdf

[^1_16]: http://arxiv.org/pdf/2406.06580.pdf

[^1_17]: https://arxiv.org/abs/2402.00559

[^1_18]: https://arxiv.org/html/2403.14312

[^1_19]: http://arxiv.org/abs/2504.03234

[^1_20]: https://arxiv.org/abs/2402.10200

[^1_21]: https://aclanthology.org/2025.findings-emnlp.253.pdf

[^1_22]: https://magazine.sebastianraschka.com/p/state-of-llm-reasoning-and-inference-scaling

[^1_23]: https://learnprompting.org/docs/intermediate/chain_of_thought

[^1_24]: https://qwenlm.github.io/blog/qwen2.5-math-prm/

[^1_25]: https://arxiv.org/abs/2408.03314

[^1_26]: https://arxiv.org/abs/2201.11903

[^1_27]: https://huggingface.co/papers/2501.07301

[^1_28]: https://cmu-l3.github.io/neurips2024-inference-tutorial/

[^1_29]: https://aclanthology.org/2024.findings-acl.876/

[^1_30]: https://openreview.net/forum?id=3Sxby0hH1q

[^1_31]: https://research.ibm.com/blog/inference-scaling-reasoning-ai-model

[^1_32]: https://kimjy99.github.io/논문리뷰/cot-decoding/

[^1_33]: https://arxiv.org/pdf/2501.07301.pdf

[^1_34]: https://discuss.pytorch.kr/t/deep-research-test-time-compute-test-time-scaling/6153

[^1_35]: https://arxiv.org/abs/2406.09136

[^1_36]: https://arxiv.org/pdf/2505.14391.pdf

[^1_37]: https://arxiv.org/html/2506.12928v1

[^1_38]: https://arxiv.org/abs/2402.07754

[^1_39]: https://arxiv.org/pdf/2505.14674.pdf

[^1_40]: https://arxiv.org/html/2504.00294v1

[^1_41]: https://arxiv.org/pdf/2504.16828.pdf

[^1_42]: https://arxiv.org/pdf/2504.02495.pdf

[^1_43]: https://arxiv.org/abs/2412.06769

[^1_44]: https://arxiv.org/abs/2412.15904

[^1_45]: https://arxiv.org/abs/2510.09599

[^1_46]: https://arxiv.org/abs/2410.23912

[^1_47]: https://arxiv.org/abs/2403.09629

[^1_48]: https://arxiv.org/abs/2407.10040

[^1_49]: https://arxiv.org/abs/2412.16260

[^1_50]: https://dl.acm.org/doi/10.1145/3703323.3704277

[^1_51]: https://arxiv.org/abs/2402.06457

[^1_52]: https://arxiv.org/abs/2412.17256

[^1_53]: https://www.putrapublisher.org/ojs/index.php/jipsi/article/view/637

[^1_54]: https://www.tandfonline.com/doi/full/10.1080/17425964.2024.2440374

[^1_55]: https://arxiv.org/abs/2404.03683

[^1_56]: https://arxiv.org/pdf/2203.14465.pdf

[^1_57]: https://arxiv.org/html/2407.10040

[^1_58]: https://arxiv.org/pdf/2410.23912.pdf

[^1_59]: https://arxiv.org/pdf/2502.13550.pdf

[^1_60]: https://arxiv.org/pdf/2503.04625.pdf

[^1_61]: https://arxiv.org/pdf/2412.17256.pdf

[^1_62]: http://arxiv.org/pdf/2403.09629.pdf

[^1_63]: http://arxiv.org/pdf/2501.04519.pdf

[^1_64]: https://research.google/pubs/star-self-taught-reasoner-bootstrapping-reasoning-with-reasoning/

[^1_65]: https://www.promptingguide.ai/techniques/tot

[^1_66]: https://www.nature.com/articles/s41586-025-09422-z

[^1_67]: https://openreview.net/pdf?id=_3ELRdg2sgI

[^1_68]: http://www.diva-portal.org/smash/get/diva2:1909318/FULLTEXT01.pdf

[^1_69]: https://github.com/deepseek-ai/DeepSeek-R1

[^1_70]: https://arxiv.org/abs/2203.14465

[^1_71]: https://arxiv.org/abs/2305.10601

[^1_72]: https://arxiv.org/html/2501.12948v1

[^1_73]: https://aclanthology.org/2024.findings-naacl.78/

[^1_74]: https://huggingface.co/papers/2501.12948

[^1_75]: https://www.youtube.com/watch?v=rQLZgEgmQCc

[^1_76]: https://arxiv.org/abs/2409.11527

[^1_77]: https://kimjy99.github.io/논문리뷰/deepseek-r1/

[^1_78]: https://arxiv.org/html/2512.02456v1

[^1_79]: https://arxiv.org/pdf/2501.12948.pdf

[^1_80]: https://arxiv.org/abs/2410.17820

[^1_81]: https://arxiv.org/abs/2501.12948

[^1_82]: https://arxiv.org/abs/2407.03687

[^1_83]: https://arxiv.org/html/2503.10573v1

[^1_84]: https://arxiv.org/abs/2412.09078

[^1_85]: https://arxiv.org/abs/2503.19633

[^1_86]: https://aitopics.org/doc/conferences:223CBA23
