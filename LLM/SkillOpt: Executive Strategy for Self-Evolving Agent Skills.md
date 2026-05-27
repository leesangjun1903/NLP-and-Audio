# SkillOpt: Executive Strategy for Self-Evolving Agent Skills

---

## 1. 핵심 주장과 주요 기여 요약

### 핵심 주장

SkillOpt은 **에이전트 스킬(skill)을 딥러닝 최적화와 동일한 원칙으로 훈련 가능한 외부 상태(external state)로 취급**해야 한다고 주장합니다. 기존의 수동 작성, 원샷(one-shot) 생성, 또는 느슨한 자가 수정 방식은 피드백 하에서 안정적으로 개선되지 않으며, 마치 딥러닝 옵티마이저처럼 동작하지 않는다는 문제를 지적합니다.

> "We argue the skill should instead be *trained* as the external state of a frozen agent, with the same discipline that makes weight-space optimization reproducible."

### 주요 기여 (3가지)

| 기여 | 설명 |
|------|------|
| **방법론 제안** | 에이전트 스킬 학습을 외부 자연어 상태에 대한 최적화로 공식화하고, add/delete/replace 편집, 텍스트 학습률, 검증 게이트, 거절 편집 버퍼, epoch-wise slow/meta update를 갖춘 SkillOpt 도입 |
| **광범위한 실증 연구** | 6개 벤치마크, 7개 대상 모델, 3개 실행 하네스에서 52/52 셀 최고 성능 달성 |
| **전이성(transferability) 검증** | 최적화된 스킬 아티팩트가 모델 스케일, 실행 환경, 인접 벤치마크에 걸쳐 전이 가능함을 입증 |

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

**현존하는 에이전트 스킬 방식의 한계:**

1. **수동 작성(hand-crafted skills)**: 브리틀(brittle)하고 도메인 변화에 취약
2. **원샷 LLM 생성(one-shot LLM skills)**: 롤아웃 피드백 후 수정 불가
3. **느슨한 자가 수정(loosely controlled self-revision)**: 불안정하고 재현 불가능

핵심 공백: *"if skills are the adaptation layer, how should they be optimized?"*

또한, 폐쇄형 프론티어 모델(closed frontier models)은 가중치 업데이트가 불가능하고, 오픈 모델은 파인튜닝 비용이 높아 **가중치 변경 없이 적응할 수 있는 메커니즘이 필요**합니다.

---

### 2.2 제안하는 방법 (수식 포함)

#### 2.2.1 문제 정의

하네스 $h$, 작업 $x$, 스킬 $s$, 동결된 대상 모델 $M$에 대해 실행은 다음을 생성합니다:

$$(\tau(s), r(s)) = h(M, x, s), \quad r(s) \in [0, 1] \tag{1}$$

훈련 분할 $D_{\text{tr}}$, 선택 분할 $D_{\text{sel}}$, 테스트 분할 $D_{\text{test}}$가 주어졌을 때, 최적 스킬 선택:

$$s^{\star}_{\text{sel}} = \arg\max_{s \in \mathcal{C}(D_{\text{tr}})} \frac{1}{|D_{\text{sel}}|} \sum_{x \in D_{\text{sel}}} r(s) \tag{2}$$

최종 성능 보고:

$$\text{Test}(s^{\star}_{\text{sel}}) = \frac{1}{|D_{\text{test}}|} \sum_{x \in D_{\text{test}}} r(s^{\star}_{\text{sel}}) \tag{3}$$

#### 2.2.2 SkillOpt 최적화 루프 (Algorithm 1 기반)

```
입력: 동결 모델 M, 옵티마이저 모델 O, 하네스 h
      분할 D_train, D_sel, D_test, 초기 스킬 s0
      에폭 E, 편집 예산 스케줄 Lt, 롤아웃 배치 크기 B
      누적 계수 A, 반성 미니배치 크기 Bm
```

전체 루프:

$$s_{\text{cur}} \leftarrow s_0, \quad s_{\text{best}} \leftarrow s_0, \quad \mathcal{C} \leftarrow \emptyset, \quad \mathcal{B} \leftarrow [\ ], \quad m_{\text{meta}} \leftarrow \emptyset$$

각 에폭 $e$의 각 스텝에서:
1. $A$개의 롤아웃 배치 수집: $h(M, x, s_{\text{cur}})$ 실행
2. 실패/성공 분리 → 미니배치 $B_m$으로 분할
3. 옵티마이저 $O$가 실패/성공 각각 분석 → add/delete/replace 편집 제안
4. 편집 병합 (실패 우선) → 상위 $L_t$개 선택
5. 후보 스킬 $\tilde{s}$ 생성
6. **검증 게이트**: $\text{score}\_{\text{cand}} \leftarrow \text{Evaluate}(M, h, \tilde{s}, D_{\text{sel}})$

$$\text{if } \text{score}_{\text{cand}} > \text{score}_{\text{cur}}: \quad s_{\text{cur}} \leftarrow \tilde{s}$$
$$\text{if } \text{score}_{\text{cand}} > \text{score}_{\text{best}}: \quad s_{\text{best}} \leftarrow \tilde{s}$$

거절된 편집은 버퍼 $\mathcal{B}$에 저장 → 이후 반성 호출에 부정적 피드백으로 활용.

#### 2.2.3 딥러닝 아날로지

| 딥러닝 개념 | SkillOpt 대응 개념 |
|-------------|-------------------|
| 파라미터(parameter) | 스킬 문서(skill document) |
| 그래디언트 방향(gradient direction) | 궤적 유도 편집 방향(trajectory-derived edit direction) |
| 학습률(learning rate) | 편집 예산 $L_t$(edit budget) |
| 검증(validation check) | 헬드아웃 선택 게이트(held-out selection gate) |
| 배치/스케줄 | 배치/미니배치/스케줄/게이트 |
| 모멘텀(momentum) | epoch-wise slow/meta update |

#### 2.2.4 텍스트 학습률 및 스케줄

편집 예산 $L_t$는 스텝 $t$에서 적용 가능한 최대 편집 수입니다. SkillOpt는 다음 스케줄을 지원합니다:

- **Constant**: $L_t = L$ (모든 스텝 동일)
- **Cosine**: $L_t = L_{\min} + \frac{1}{2}(L_{\max} - L_{\min})(1 + \cos(\frac{\pi t}{T}))$
- **Linear**: $L_t$가 선형적으로 감소
- **Autonomous**: 옵티마이저가 자율 결정

기본값: $L_t = 4$ (cosine decay, floor $L_t = 2$)

---

### 2.3 모델 구조

SkillOpt의 구조는 두 개의 분리된 모델로 구성됩니다:

```
┌─────────────────────────────────────────────────────────┐
│                    SkillOpt Pipeline                     │
│                                                          │
│  [동결 대상 모델 M]  ←── 스킬 문서 s_cur               │
│       ↓ 롤아웃 실행                                      │
│  [궤적 τ + 점수 r]                                       │
│       ↓                                                  │
│  [옵티마이저 모델 O]                                     │
│   ├── 실패 분석 (analyst_error)                          │
│   ├── 성공 분석 (analyst_success)                        │
│   ├── 실패 병합 (merge_failure)                          │
│   ├── 성공 병합 (merge_success)                          │
│   ├── 최종 병합 (merge_final)                            │
│   └── 순위/선택 (ranking) → 상위 Lt개                   │
│       ↓                                                  │
│  [후보 스킬 s̃]                                           │
│       ↓ 검증 게이트 (D_sel 평가)                         │
│  수락 → best_skill.md / 거절 → 거절 편집 버퍼 B         │
│                                                          │
│  [Epoch-wise Slow/Meta Update]                           │
│   ├── 이전 epoch 스킬 vs 현재 epoch 스킬 비교            │
│   ├── 개선/회귀/지속 실패/안정 성공 분류                 │
│   ├── 보호된 slow-update 필드에 종단 지침 작성           │
│   └── 옵티마이저 메타 스킬 업데이트 (m_meta)             │
└─────────────────────────────────────────────────────────┘
```

**하네스-독립적(Harness-Agnostic) 배포:**
- Direct Chat: 스킬을 시스템 프롬프트에 삽입
- Codex 하네스: `SKILL.md` 파일로 작업 공간에 배치
- Claude Code 하네스: 동일한 작업 공간 계약

배포 결과물: `best_skill.md` (300~2,000 토큰, 추론 시 추가 모델 호출 없음)

---

### 2.4 성능 향상

#### 주요 결과 (Table 1)

**GPT-5.5 Direct Chat 기준:**

| 벤치마크 | No Skill | SkillOpt | 향상 |
|----------|----------|----------|------|
| SearchQA | 77.7 | **87.3** | +9.6 |
| SpreadsheetBench | 41.8 | **80.7** | +38.9 |
| OfficeQA | 33.1 | **72.1** | +39.0 |
| DocVQA | 78.8 | **91.2** | +12.4 |
| LiveMath | 37.6 | **66.9** | +29.3 |
| ALFWorld | 83.6 | **95.5** | +11.9 |
| **평균** | **58.8** | **82.3** | **+23.5** |

**모델별 평균 향상:**

$$\bar{\Delta}_{\text{GPT-5.5}} = +23.5 \text{ pts (direct chat)}, \quad +24.8 \text{ pts (Codex)}, \quad +19.1 \text{ pts (Claude Code)}$$

$$\bar{\Delta}_{\text{avg, 7 models}} \approx +17.6 \text{ pts}$$

#### 에블레이션 결과 (Table 3)

| 구성 요소 | SearchQA | SpreadsheetBench | LiveMath |
|-----------|----------|------------------|----------|
| **전체 (기본값)** | **87.1** | **77.5** | **61.3** |
| 거절 버퍼 없음 | 85.5 | 72.9 | 58.9 |
| 메타 스킬 없음 | 85.1 | 75.7 | 58.1 |
| 메타 스킬 + slow update 없음 | 86.3 | 55.0 | 59.7 |
| 학습률 없음 | 84.6 | 75.7 | 57.3 |

SpreadsheetBench에서 slow/meta update 제거 시 **-22.5점** 하락 (가장 큰 성능 저하).

---

### 2.5 한계

논문 Appendix B에서 명시된 한계:

1. **자동 검증기 의존성**: 최적화 루프가 점수화된 궤적과 헬드아웃 선택 분할에 의존하므로, **주관적이거나 판단 비용이 높은 개방형 도메인에서는 적용이 제한적**
2. **훈련 비용**: 배포 아티팩트는 컴팩트하지만 스킬 훈련 자체는 추가 롤아웃 연산과 옵티마이저 모델 호출이 필요 (일회성 작업에는 비효율적일 수 있음)
3. **단일 스킬 아티팩트**: 다수의 서로 다른 절차가 필요한 **고도로 이질적인 도메인**에서는 단일 스킬이 불충분할 수 있음
4. **훈련 분포 편향**: 최적화된 스킬이 훈련 분포의 특정 발견적 방법(heuristics)을 인코딩할 수 있으므로, 상이한 모델/하네스/작업으로 전이 시 주의 필요

---

## 3. 일반화 성능 향상 가능성 (중점 분석)

SkillOpt의 일반화 성능은 **세 가지 축의 전이 실험(Table 4)**을 통해 검증됩니다.

### 3.1 모델 스케일 전이 (Cross-Model Transfer)

GPT-5.4로 최적화된 스킬을 더 작은 모델에 배포했을 때:

| 소스 모델 | 대상 모델 | 벤치마크 | 베이스라인 | 직접 최적화 | 전이 |
|-----------|-----------|----------|-----------|------------|------|
| GPT-5.4 | GPT-5.4-mini | SpreadsheetBench | 36.1 | 47.5 | 45.5 (+9.4) |
| GPT-5.4 | GPT-5.4-nano | SpreadsheetBench | 23.5 | 42.5 | 26.5 (+3.0) |
| GPT-5.4 | GPT-5.4-mini | LiveMath | 14.7 | 32.8 | 19.2 (+4.5) |
| GPT-5.4 | GPT-5.4-nano | LiveMath | 23.2 | 27.2 | **28.8** (+5.6) |

주목할 점: LiveMath GPT-5.4-nano의 경우, **전이된 스킬(28.8)이 직접 최적화(27.2)를 능가** — 일부 학습된 절차가 모델-독립적(model-agnostic)임을 시사.

GPT-5.4-mini는 직접 최적화 이득의 **82%($= \frac{9.4}{11.4}$)를 전이로 회수**합니다. 모든 전이 행에서 베이스라인(no-skill) 이하로 하락한 경우는 없었습니다.

**일반화 가능성 근거**: 소형 모델일수록 상대적 이득이 더 큼

$$\Delta_{\text{GPT-5.4-nano}} \approx +26.7 \text{ pts} > \Delta_{\text{GPT-5.5}} \approx +23.5 \text{ pts}$$

이는 *"컴팩트한 스킬 아티팩트가 소형 모델이 가중치에 아직 보유하지 못한 절차적 지식을 공급할 수 있다"*는 견해와 일치합니다.

### 3.2 실행 환경 전이 (Cross-Harness Transfer)

| 소스 하네스 | 대상 하네스 | 벤치마크 | 베이스라인 | 전이 |
|------------|------------|----------|-----------|------|
| Codex | Claude Code | SpreadsheetBench | 22.1 | **81.8** (+59.7) |
| Claude Code | Codex | SpreadsheetBench | 27.5 | 71.1 (+43.6) |
| Claude Code | Codex | LiveMath | 35.2 | 48.0 (+12.8) |

SpreadsheetBench Codex→Claude Code 전이(+59.7)는 **in-domain Claude Code SkillOpt(80.4)를 초과(81.8)**합니다. 두 하네스가 서로 다른 tool/file API와 커맨드 표면을 노출함에도 불구하고 긍정적 전이가 발생한 것은, 학습된 규칙이 하네스-특화 명령 레시피가 아닌 **워크북 수준의 절차적 지식**을 인코딩하기 때문입니다 (예: 구조 우선 검사, 수식 인식 검증, 정적 값 구체화).

### 3.3 벤치마크 전이 (Cross-Benchmark Transfer)

OlympiadBench로 최적화된 스킬을 Omni-MATH에 적용:

| 대상 모델 | 베이스라인 | 전이 |
|-----------|-----------|------|
| GPT-5.4 | 56.6 | 60.3 (+3.7) |
| GPT-5.4-mini | 34.8 | 36.6 (+1.8) |
| GPT-5.4-nano | 38.8 | 40.1 (+1.3) |

세 가지 전이 중 가장 엄격한 조건(테스트 인스턴스와 답변 형식 모두 변경)임에도 **모든 스케일에서 균일하게 긍정적**, 이는 스킬이 **재사용 가능한 수학적 절차를 인코딩**하며 벤치마크 특화 포맷팅을 단순 암기하지 않음을 지지합니다.

### 3.4 일반화를 가능하게 하는 메커니즘

Figure 4의 학습된 규칙을 보면:

- **SearchQA**: *"Infer the expected answer type from clue wording, then choose the shortest canonical entity..."* → 특정 질문이나 엔티티를 명시하지 않는 절차적 규칙
- **SpreadsheetBench**: *"Inspect workbook structure and formulas, then write evaluated static values..."* → 도메인 발견적 지식
- **ALFWorld**: *"Keep a horizon-aware visited/frontier ledger, diversify search..."* → 유한 상태 실행 정책

이 규칙들은 **인스턴스 특화가 아닌 절차적(procedural)**이며, "사려깊은 인간 실무자가 해당 벤치마크를 하루 다룬 후 작성할 법한 규칙"과 유사합니다.

### 3.5 검증 게이트가 일반화에 기여하는 방식

Figure 3이 보여주듯, 검증 체크포인트는 헬드아웃 테스트 성능과 추적 가능하게 정렬됩니다:

$$\text{Selection score} \approx \text{Test score (epoch-wise)}$$

이는 검증 게이트가 단순히 선택 분할에 과적합되지 않고, **실제 일반화를 근사**함을 의미합니다. 게이트 엄격성(strict greater than, ties rejected)이 이를 보장합니다.

---

## 4. 관련 최신 연구 비교 분석 (2020년 이후)

### 4.1 프롬프트 자동 최적화 계열

| 연구 | 방법 | SkillOpt와의 차이 |
|------|------|-----------------|
| **TextGrad** (Yuksekgonul et al., 2024, arXiv:2406.07496) | 텍스트를 통한 자동 "미분", 그래디언트 스타일 자연어 프롬프트 최적화 | 영구적 스킬 아티팩트 대신 프롬프트 최적화; 헬드아웃 게이트 없음 |
| **GEPA** (Agrawal et al., 2025, arXiv:2507.19457) | 반성적 프롬프트 진화로 강화학습 능가 | 재사용 가능한 도메인 적응 스킬 아티팩트 대신 단일 프롬프트/시스템 설계 최적화 |
| **OPRO/LLM as Optimizer** (Yang et al., 2023) | LLM을 메타 프롬프트 최적화기로 활용 | 배치 기반 반성, 검증 게이트, 거절 버퍼 없음 |
| **DSPy** (Khattab et al., 2023) | 선언적 LLM 파이프라인 컴파일 | 파이프라인 전체 최적화 vs. 단일 스킬 문서 최적화 |

### 4.2 스킬 구축 및 진화 계열

| 연구 | 방법 | SkillOpt와의 차이 |
|------|------|-----------------|
| **Trace2Skill** (Ni et al., 2026, arXiv:2603.25158) | 궤적에서 도메인 스킬 증류 | 헬드아웃 게이트 없이 원샷 증류; 반복적 편집 없음 |
| **EvoSkill** (Alzubi et al., 2026, arXiv:2603.02766) | 실패 분석을 통한 스킬 폴더 진화 | 경계있는 텍스트 학습률과 거절 편집 메모리 부재 |
| **EvoSkills** (Zhang et al., 2026, arXiv:2604.01687) | 공진화 검증을 통한 자가 진화 스킬 | 공진화 검증기 vs. SkillOpt의 통제된 편집 예산 |
| **SkillForge** (Liu et al., 2026, arXiv:2604.08618) | 클라우드 기술 지원 도메인 특화 자가 진화 스킬 | 도메인 특화 vs. SkillOpt의 하네스-독립 범용 최적화 |
| **SkillFoundry** (Shen et al., 2026, arXiv:2604.03964) | 이종 과학 자원에서 스킬 라이브러리 구축 | 스킬 라이브러리 성장 vs. 단일 컴팩트 스킬 최적화 |
| **AutoSkill** (Yang et al., 2026, arXiv:2603.01145) | 경험 기반 평생 학습을 통한 스킬 자가 진화 | 평생 학습 vs. SkillOpt의 단일 도메인 집중 훈련 |

### 4.3 에이전트 자기 반성 계열

| 연구 | 방법 | SkillOpt와의 차이 |
|------|------|-----------------|
| **Reflexion** (Shinn et al., 2023, NeurIPS) | 언어 에이전트의 언어 강화 학습 | 에피소드별 언어 반성 vs. SkillOpt의 배치 수준 편집 최적화 |
| **Self-Refine** (Madaan et al., 2023, NeurIPS) | 자기 피드백을 통한 반복적 정제 | 개별 출력 정제 vs. 영구적 스킬 문서 최적화 |
| **ABSTRAL** (Song et al., 2026, arXiv:2603.22791) | 반복적 개선을 통한 멀티에이전트 시스템 자동 설계 | 전체 멀티에이전트 설계 vs. 단일 스킬 최적화 |

### 4.4 정량적 비교 요약

GPT-5.5 SpreadsheetBench (Codex 하네스) 기준:

| 방법 | 점수 | SkillOpt 대비 |
|------|------|--------------|
| No Skill | 27.5 | -57.5 |
| Human Skill | 50.7 | -34.3 |
| LLM Skill | 25.0 | -60.0 |
| EvoSkill | 67.5 | -17.5 |
| **SkillOpt** | **85.0** | — |

---

## 5. 앞으로의 연구에 미치는 영향 및 고려사항

### 5.1 앞으로의 연구에 미치는 영향

#### 5.1.1 패러다임 전환: 스킬을 훈련 가능한 객체로

SkillOpt는 에이전트 스킬을 단순한 프롬프트 엔지니어링의 부산물이 아닌, **최적화의 1등 시민(first-class object)**으로 승격시키는 패러다임 전환을 제안합니다. 이는 향후 연구가 스킬 공간에서 전통적인 딥러닝 최적화 기법(정규화, 커리큘럼 학습, 메타 학습 등)을 직접 탐구할 수 있는 기반을 마련합니다.

#### 5.1.2 폐쇄형 모델 적응의 새로운 방향

GPT-4/5 등 가중치에 접근할 수 없는 폐쇄형 모델에 대한 **경량화된 도메인 적응**의 실행 가능한 경로를 제시합니다. 이는 에이전트 배포 비용을 대폭 낮출 수 있으며, 파인튜닝 없이도 전문 도메인 성능을 확보하는 연구를 촉진할 것입니다.

#### 5.1.3 텍스트 공간 최적화의 이론적 기반 구축

현재 논문은 경험적 결과에 집중하지만, **텍스트 공간 최적화의 수렴성, 안정성, 표현력에 관한 이론적 분석**의 필요성을 제기합니다. 이는 자연어 처리와 최적화 이론의 교차점에서 새로운 연구 영역을 열 것입니다.

#### 5.1.4 스킬 라이브러리 생태계

단일 스킬 최적화에서 **스킬 라이브러리(skill library) 공유, 도메인 간 전이, 스킬 구성(composition)** 연구로의 자연스러운 확장이 가능합니다. 논문 자체도 이를 향후 연구 방향으로 제시합니다.

### 5.2 연구 시 고려해야 할 사항

#### 5.2.1 검증 신호(Validation Signal)의 품질

SkillOpt의 핵심 가정은 **신뢰할 수 있는 스칼라 점수**의 존재입니다. 향후 연구는 다음을 고려해야 합니다:
- 개방형/주관적 작업을 위한 **선호도 기반 검증 게이트** (preference-driven gate)
- LLM-as-Judge 방식의 활용 및 그 신뢰성 평가
- 보상 해킹(reward hacking)을 방지하는 메커니즘

#### 5.2.2 훈련 비용 vs. 재사용 빈도 트레이드오프

$$\text{ROI} = \frac{\text{성능 향상} \times \text{재사용 횟수}}{\text{훈련 비용}}$$

SearchQA의 경우 훈련 토큰 비용이 213.8M (포인트당 37.9M)으로 높습니다. 일회성 배포에서는 비효율적일 수 있으므로, **비용 효율적인 훈련 프로토콜** 연구가 필요합니다.

#### 5.2.3 다중 스킬 구성 및 갈등 해결

단일 스킬은 이질적인 도메인에서 불충분할 수 있습니다. 향후 연구는:
- **스킬 간 충돌(conflict) 해결** 메커니즘
- **조건부 스킬 라우팅** (어떤 스킬을 언제 적용할지)
- **스킬 계층화(hierarchical skill organization)** 방법론

을 탐구해야 합니다.

#### 5.2.4 안전성 및 해석가능성

학습된 스킬이 훈련 분포의 편향을 인코딩할 위험이 있습니다. 연구 시:
- **스킬 감사(auditing) 프로토콜** 개발
- 유해한 절차적 규칙 자동 탐지
- 스킬 편집의 **인과적 영향 추적** (`edit_apply_report.json` 활용)

을 고려해야 합니다.

#### 5.2.5 스킬의 자가 증류 (Self-Distillation)

논문이 언급한 방향으로, 최적화된 스킬을 대상 모델 가중치에 다시 증류(distill back)하는 연구는:
- 텍스트 공간 최적화와 파라미터 공간 최적화의 **가교 역할**
- 반복적 자가 개선(iterative self-improvement) 파이프라인 구축
- 작은 모델이 큰 모델의 스킬을 흡수하는 **지식 전이 체계**

로 이어질 수 있습니다.

#### 5.2.6 실세계 배포에서의 분산 시프트

실험은 동일한 분포의 train/selection/test 분할을 사용하지만, 실세계에서는 **분산 시프트(distribution shift)**가 발생합니다. 연구는:
- 시간적 분산 시프트에 강건한 스킬 설계
- 온라인(continual) 스킬 업데이트 메커니즘
- 전이 품질을 사전에 예측하는 지표 개발

을 탐구해야 합니다.

---

## 참고자료 (논문 내 인용 기준)

**주요 참고 논문 (논문 내 직접 인용):**

1. **SkillOpt 논문 원문**: Yang, Y. et al. "SkillOpt: Executive Strategy for Self-Evolving Agent Skills." *arXiv:2605.23904v2*, 2026.
2. **TextGrad**: Yuksekgonul, M. et al. "Textgrad: Automatic 'differentiation' via text." *arXiv:2406.07496*, 2024.
3. **GEPA**: Agrawal, L.A. et al. "Gepa: Reflective prompt evolution can outperform reinforcement learning." *arXiv:2507.19457*, 2025.
4. **Trace2Skill**: Ni, J. et al. "Trace2skill: Distill trajectory-local lessons into transferable agent skills." *arXiv:2603.25158*, 2026.
5. **EvoSkill**: Alzubi, S. et al. "Evoskill: Automated skill discovery for multi-agent systems." *arXiv:2603.02766*, 2026.
6. **Reflexion**: Shinn, N. et al. "Reflexion: Language agents with verbal reinforcement learning." *NeurIPS 36*, 2023.
7. **Self-Refine**: Madaan, A. et al. "Self-refine: Iterative refinement with self-feedback." *NeurIPS 36*, 2023.
8. **DSPy**: Khattab, O. et al. "Dspy: Compiling declarative language model calls into self-improving pipelines." *arXiv:2310.03714*, 2023.
9. **SkillsBench**: Li, X. et al. "Skillsbench: Benchmarking how well agent skills work across diverse tasks." *arXiv:2602.12670*, 2026.
10. **SpreadsheetBench**: Ma, Z. et al. "Spreadsheetbench: Towards challenging real world spreadsheet manipulation." *NeurIPS 37*, 2024.
11. **ALFWorld**: Shridhar, M. et al. "ALFWorld: Aligning text and embodied environments for interactive learning." *ICLR*, 2021.
12. **Omni-MATH**: Gao, B. et al. "Omni-math: A universal olympiad level mathematic benchmark for large language models." *arXiv:2410.07985*, 2024.
13. **LLM as Optimizer (OPRO)**: Yang, C. et al. "Large language models as optimizers." *ICLR*, 2023.
14. **ReAct**: Yao, S. et al. "React: Synergizing reasoning and acting in language models." *arXiv:2210.03629*, 2022.
15. **SWE-agent**: Yang, J. et al. "Swe-agent: Agent-computer interfaces enable automated software engineering." *NeurIPS 37*, 2024.
16. **Omni-MATH**: Gao, B. et al. arXiv:2410.07985, 2024.
17. **OlympiadBench**: He, C. et al. "Olympiadbench: A challenging benchmark for promoting AGI with olympiad-level bilingual multimodal scientific problems." *arXiv:2402.14008*, 2024.

---

> **⚠️ 정확도 고지**: 본 답변은 제공된 논문 PDF(arXiv:2605.23904v2)를 직접 분석한 결과입니다. 논문에 명시적으로 기술된 내용만을 인용하였으며, 불확실한 내용은 추측하지 않았습니다. 2020년 이후 최신 연구 비교는 논문의 Related Work 및 References 섹션에 인용된 문헌을 기준으로 작성되었습니다.
