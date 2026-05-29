
# MMSkills: Towards Multimodal Skills for General Visual Agents 

> **📌 논문 정보**
> - **제목**: MMSkills: Towards Multimodal Skills for General Visual Agents
> - **저자**: Kangning Zhang 외 10인 (Shanghai Jiao Tong University, Xiaohongshu Inc., Southeast University)
> - **arXiv**: [2605.13527](https://arxiv.org/abs/2605.13527) (게재: 2026년 5월 13일)
> - **GitHub**: [DeepExperience/MMSkills](https://github.com/DeepExperience/MMSkills)

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

기존의 재사용 가능한 스킬 패키지들은 텍스트 프롬프트, 실행 코드, 혹은 학습된 루틴 형태로 행동을 인코딩하는 데 그치고 있다. 그러나 시각적 에이전트에게 절차적 지식은 본질적으로 **멀티모달**하다 — 단순히 어떤 조작을 수행할지뿐만 아니라, 관련 상태를 인식하고, 진행 또는 실패의 시각적 증거를 해석하며, 다음 행동을 결정하는 것까지 포함된다.

이에 MMSkills는 **재사용 가능한 멀티모달 절차를 표현(represent), 생성(generate), 활용(use)하는 프레임워크**로서, 런타임 시각적 의사결정을 지원한다.

### 주요 기여 (3가지)

논문은 세 가지 실용적 과제를 해결한다: **(I)** 멀티모달 스킬 패키지가 무엇을 담아야 하는가, **(II)** 공개 인터랙션 경험으로부터 그런 패키지를 어디서 도출할 수 있는가, **(III)** 에이전트가 추론 시 과도한 이미지 컨텍스트나 참조 스크린샷 과의존 없이 멀티모달 증거를 어떻게 참조할 수 있는가.

---

## 2. 해결하고자 하는 문제, 제안 방법, 모델 구조, 성능, 한계

### 2-1. 해결하고자 하는 문제

MMSkills는 스킬을 텍스트, 코드, API, 실행 그래프로 저장하는 기존 방식에서 벗어나, **시각적으로 근거가 마련된(visually grounded) 런타임 상태 집합**을 중심 증거로 삼는 스킬 패키지를 정의한다.

구체적으로는 다음 두 가지 문제가 핵심입니다:

1. **텍스트 스킬의 시각적 불충분성**: 멀티모달 스킬 패키지는 텍스트 절차, 런타임 상태 카드, 멀티뷰 시각 증거를 결합한다. 텍스트만의 안내는 활성 시트 상태를 놓칠 수 있지만, Branch-Loaded MMSkills는 스킬 증거를 실시간 화면과 정렬하여 상태 인식 기반의 안내를 반환한다.

2. **시각적 앵커링(Visual Anchoring) 문제**: 메인 에이전트가 표면적으로 유사한 참조 스크린샷에 시각적으로 고착되어, 현재 환경 대신 스킬 예시를 중심으로 계획을 세울 수 있다. Branch Loading은 스킬 증거에 대한 멀티모달 형태의 **점진적 공개(Progressive Disclosure)** 방식으로 이 문제를 해결한다.

---

### 2-2. 제안하는 방법

MMSkills는 크게 **두 가지 핵심 메커니즘**으로 구성됩니다.

#### ① Trajectory-to-Skill Generator (스킬 생성)

멀티모달 스킬 패키지를 생성하기 위해, **메타 스킬 가이드 파이프라인(meta-skill-guided pipeline)** 기반의 자동화된 Trajectory-to-Skill Generator를 도입한다.

이 Generator는 공개된 비평가 경로(non-evaluation trajectories)를 **워크플로우 그루핑(workflow grouping) → 절차 유도(procedure induction) → 시각적 근거화(visual grounding) → 메타 스킬 가이드 오디팅(meta-skill-guided auditing)** 의 단계를 통해 재사용 가능한 멀티모달 스킬로 변환한다.

생성된 각 스킬 패키지의 구성은 수식으로 아래와 같이 표현할 수 있습니다:

$$
\text{MMSkill} = \{\mathcal{P}, \mathcal{S}_{runtime}, \mathcal{K}_{multiview}\}
$$

- $\mathcal{P}$: 텍스트 절차(Textual Procedure)
- $\mathcal{S}_{runtime}$: 런타임 상태 카드 집합(Runtime State Cards)
- $\mathcal{K}_{multiview}$: 멀티뷰 키프레임(Multi-view Keyframes)

각 MMSkill은 텍스트 절차와 런타임 상태 카드 및 멀티뷰 키프레임을 결합한 **컴팩트하고 상태 조건적인(state-conditioned) 패키지**이다.

---

#### ② Branch-Loaded Multimodal Skill Agent (스킬 사용)

메인 에이전트가 스킬을 고려할 때, **임시 브랜치(temporary branch)** 를 열어 필요한 상태 카드와 키프레임 뷰를 선택하고, 이를 실시간 화면 또는 장면과 정렬한 후, 적용 가능성 판단, 서브골(subgoal), 다음 단계 계획을 포함한 컴팩트한 구조적 안내를 반환한다.

컨텍스트에 이미지가 넘치는 것을 방지하기 위해 **Branch Loading** 메커니즘을 사용한다. 에이전트의 현재 상태가 스킬 적용을 시사하면, 관련 상태 카드와 키프레임을 선택하는 임시 사이드 브랜치를 열고, 실시간 화면과 정렬한 다음 적용 가능성 판단, 로컬 서브골, 단계별 계획이 담긴 컴팩트한 구조화 요약을 메인 추론 스레드에 반환한다. 메인 에이전트는 자체 컨텍스트를 간결하게 유지하면서 해당 요약에 기반해 행동한다.

Branch-Loaded 추론 과정을 수식으로 표현하면:

```math
g_t = f_{\text{branch}}\bigl(\mathcal{S}_{runtime}^*, \mathcal{K}^*, s_t^{live}\bigr)
```

$$
a_t = \pi_{\text{main}}\bigl(o_t, g_t\bigr)
$$

- $s_t^{live}$: 현재 실시간 환경 스크린샷
- $\mathcal{S}\_{runtime}^\*$, $\mathcal{K}^*$: 선택된 상태 카드 및 키프레임
- $g_t$: 브랜치에서 증류된 구조화 안내(Guidance)
- $a_t$: 메인 에이전트의 최종 행동

> **⚠️ 주의**: 위 수식은 논문에서 직접 제시된 공식 표기가 아닌, 논문의 설명을 바탕으로 개념적으로 정형화한 표현입니다. 정확한 수식은 논문 원문을 참고하세요.

---

### 2-3. 모델 구조

MMSkills는 재사용 가능한 멀티모달 절차적 지식을 시각 에이전트가 표현하고, 로드하고, 활용하기 위한 프레임워크이다. 각 스킬은 텍스트 절차 안내, 컴팩트한 상태 카드 메타데이터, 선택적 시각 참조를 결합한다.

각 스킬 디렉토리는 `SKILL.md`, 런타임 상태 카드, 오디트 상태 카드, 시각 키프레임을 포함한다. 전체 구조는 에이전트 통합(`agent_integrations/`), MMSkill 런타임 아키텍처(`mm_agents/`), OSWorld 통합(`osworld_integration/`), 공개 스킬 라이브러리(`skills_library/`), 태스크-스킬 매핑(`task_skill_mappings/`)으로 구성된다.

에이전트는 515개 스킬로 이루어진 Hugging Face 라이브러리를 검색하고, 태스크 관련 패키지만 다운로드한 후 `SKILL.md`, 런타임 상태, 시각 참조를 필요에 따라 읽는다.

---

### 2-4. 성능 향상

소형 모델인 Qwen3-VL-8B-Instruct의 경우 OSWorld 데스크탑 벤치마크에서 성공률이 10.78%에서 **25.40%로 두 배 이상** 증가하였다. Minecraft 시각 에이전트 벤치마크에서도 동일 모델의 성공률이 23.28%에서 **38.79%로** 상승했다.

macOSWorld에서는 Gemini 3 Flash, GLM-5V 등 대형 모델 실행에서도 MMSkills가 성능을 향상시켰으며, VAB-Minecraft에서도 모든 평가 모델에서 성공률과 평균 점수가 일관되게 향상되었다. Super Mario Bros 역시 같은 패턴을 따라 MMSkills 하에서 더 높은 총 성능과 보상을 달성했다.

| 벤치마크 | 모델 | 기준 성능 | MMSkills 성능 |
|---|---|---|---|
| OSWorld | Qwen3-VL-8B | 10.78% | **25.40%** (+135%) |
| VAB-Minecraft | Qwen3-VL-8B | 23.28% | **38.79%** (+67%) |
| macOSWorld | Gemini 3 Flash, GLM-5V | - | 일관된 향상 |
| Super Mario Bros | 전체 모델 | - | 일관된 향상 |

---

### 2-5. 한계

논문은 세 가지 한계를 명시한다: (1) MMSkills의 품질은 스킬 생성에 사용된 **소스 경로의 커버리지**에 의존한다; (2) 스킬 생성과 시각적 근거화 과정에서 **오류가 발생**할 수 있다; (3) Branch Loading은 **추론 비용(inference cost)을 증가**시킨다. 안전-임계(safety-critical) 또는 체화(embodied) 환경으로의 프레임워크 확장은 보다 강력한 검증과 실패 시 온라인 스킬 수정 능력을 필요로 한다.

---

## 3. 모델의 일반화 성능 향상 가능성

이 결과는 MMSkills가 단일 GUI 벤치마크에 특화되지 않음을 보여준다. **동일한 상태 조건적 스킬 포맷**이 반복되는 상태와 행동 전략이 재사용될 수 있는 시각적 근거 기반 게임 환경에서도 도움이 된다.

GUI 및 게임 기반 시각 에이전트 벤치마크에 걸친 실험은 MMSkills가 **최신 대형 모델과 소형 모델 모두를 일관되게 향상**시킴을 보여주며, 이는 외부 멀티모달 절차적 지식이 모델 내부 사전지식을 보완함을 시사한다.

더 강력한 GUI 행동 모델은 더 정확하게 클릭할 수 있지만, 어떤 절차적 상태가 중요한지, 어떤 시각적 단서가 진행을 확인하는지, 어떤 상태에서 스킬을 적용하지 말아야 하는지를 아는 것에서도 여전히 이점을 얻는다. MMSkills는 그 지식을 **컴팩트하고 재사용 가능한 멀티모달 스킬 패키지**로 표현한다.

일반화 가능성의 핵심 근거:

- 연구자들은 **자율주행, 로보틱스, 모바일 앱, 웹 에이전트, 게임** 등 새로운 도메인을 위한 MMSkill 패키지를 제출할 수 있어 커뮤니티 확장성을 갖춘다.

- 잘 구성된 스킬 패키지가 **원시적 모델 스케일을 부분적으로 대체**할 수 있음을 시사하며, 이는 신뢰할 수 있는 자동화를 의미 있게 더 저렴하게 만들 수 있다.

---

## 4. 앞으로의 연구에 미치는 영향 및 고려할 점

### 4-1. 연구에 미치는 영향

| 방향 | 내용 |
|---|---|
| **스킬 표현의 패러다임 전환** | 텍스트 중심 → 멀티모달 절차 지식 단위로의 전환 촉진 |
| **소형 모델의 효율화** | 대형 모델 없이도 멀티모달 스킬 패키지로 성능 보완 가능성 제시 |
| **표준화** | Agent Skills 표준의 시각적 확장으로서, 멀티모달 스킬 표준화 논의 촉진 |
| **도메인 확장** | GUI → 로보틱스, 자율주행, 게임 등 다양한 도메인으로 확장 가능성 |

이 논문이 대표하는 더 깊은 변화는 **신뢰할 수 있는 AI 행동에 대해 분야가 생각하는 방식의 성숙**이다.

최근 서베이는 에이전트 스킬을 점진적 공개(progressive disclosure)를 통해 로드되는 명령어, 코드, 리소스의 이식 가능한 패키지로 정의하며, Generative Agents는 회상·반성·계획을 지원하는 메모리 스트림을 유지하고, MemGPT는 OS 스타일의 메모리 계층구조를 도입하고 있다.

### 4-2. 향후 연구 시 고려할 점

1. **스킬 품질 및 커버리지 보장**
   MMSkills의 품질은 소스 경로의 커버리지에 의존하므로, 다양하고 포괄적인 경로 데이터 확보 전략이 핵심 연구 과제이다.

2. **온라인 스킬 수정 및 검증 메커니즘**
   안전-임계 또는 체화 환경으로의 확장은 더 강력한 검증과 스킬 실패 시 온라인 수정 능력이 필요하다.

3. **추론 비용 최적화**
   Branch Loading의 추론 비용 증가 문제를 해결하기 위한 경량화 연구가 필요합니다.

4. **2020년 이후 관련 연구와의 비교 및 통합**

| 연구 | 방식 | MMSkills와의 차이 |
|---|---|---|
| Generative Agents (Park et al., 2023) | 텍스트 메모리 스트림 | MMSkills는 시각 증거 포함 |
| MemGPT (Packer et al., 2024) | OS 스타일 메모리 페이징 | MMSkills는 멀티모달 Branch Loading으로 확장 |
| VOYAGER (Wang et al., 2023) | 코드 기반 스킬 라이브러리 | MMSkills는 시각적 상태 카드 추가 |
| XSkill (2025) | 시각 근거 태스크 레벨 스킬 + 경험의 듀얼스트림 | 훈련 없이 시각-도구 상호작용으로부터 지식 축적하는 최초의 통합 프레임워크로 상호보완적 |

5. **멀티모달 스킬의 안전성(Safety) 검토**
   스킬 패키지가 잘못된 시각적 상태와 매칭될 경우 발생하는 오류 전파 문제에 대한 연구가 필요합니다.

---

## 📚 참고 자료 및 출처

1. **arXiv 원문 (v1)**: https://arxiv.org/abs/2605.13527
2. **arXiv 원문 (v2)**: https://arxiv.org/abs/2605.13527v2
3. **arXiv HTML 전문**: https://arxiv.org/html/2605.13527
4. **GitHub 공식 저장소**: https://github.com/DeepExperience/MMSkills
5. **TechTimes 분석 기사**: "Visual State Cards in AI Agent Skills More Than Double Small Model Success Rates on Real Desktop Tasks" — https://www.techtimes.com/articles/316809/20260519
6. **관련 논문 (XSkill)**: "XSkill: Continual Learning from Experience and Skills in Multimodal Agents" — https://arxiv.org/pdf/2603.12056
7. **OSWorld 벤치마크**: Xie et al., 2024 (NeurIPS 2024)
8. **VisualAgentBench (VAB-Minecraft)**: Liu et al., 2024
9. **Generative Agents**: Park et al., 2023
10. **MemGPT**: Packer et al., 2024

> **⚠️ 정확도 주의사항**: 본 답변은 공개된 arXiv 초록, HTML 전문, GitHub 저장소, 관련 기사를 기반으로 작성되었습니다. 논문 내 세부 수식 및 ablation study 수치 일부는 원문 전체 접근이 제한되어, 정확한 수식은 반드시 원문 PDF를 통해 확인하시기 바랍니다.
