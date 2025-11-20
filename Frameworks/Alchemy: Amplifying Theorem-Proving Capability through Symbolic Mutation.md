# Alchemy: Amplifying Theorem-Proving Capability through Symbolic Mutation

### 1. 핵심 주장과 주요 기여도

**Alchemy 논문의 핵심 주장**[1]

Alchemy는 신경 정리 증명(Neural Theorem Proving, NTP)의 주요 병목인 **데이터 부족 문제**를 해결하기 위해 기호적 공간에서 직접 형식적 정리를 합성하는 일반적인 프레임워크를 제안합니다. 기존의 자동 형식화(autoformalization) 방식과 달리, Alchemy는 중간 자연언어 번역 단계를 거치지 않고 기존 정리로부터 새로운 정리를 구성합니다.[1]

**주요 기여도**[1]

- **데이터 규모의 획기적 확대**: Mathlib의 정리 수를 약 110,657개에서 **6,326,649개로 약 57배 증가** (rw와 apply 결합 시)
- **최초의 일반적 프레임워크**: Lean 정리 증명기용 기호 공간 데이터 합성의 첫 번째 일반적 프레임워크
- **검증된 성능 향상**: Leandojo 벤치마크에서 4.70% 절대 성능 향상, miniF2F 벤치마크에서 2.47% 향상
- **포괄적 분석**: 합성 데이터 구성과 훈련 패러다임에 대한 체계적 분석으로 강력한 정리 증명기 개발 지침 제공

### 2. 문제 정의, 제안 방법 및 모델 구조

**문제 정의**[1]

형식적 수학 문헌은 일반 텍스트에 비해 매우 제한적입니다. Mathlib은 약 110,000개의 정리만 포함하고 있어 신경 모델의 학습에 충분하지 않으며, 이는 신경 정리 증명의 핵심 과제입니다. 기존의 자동 형식화는 자연언어 문제를 형식적 명제로 번역하고 증명을 샘플링한 후 재훈련하는 노동 집약적 과정입니다.

**핵심 방법론: 기호적 변이**[1]

Alchemy의 핵심은 두 가지 기본 전술(tactic)을 이용한 기호적 조작입니다:

**1) Rewrite (rw) 전술**[1]

주어진 항등식 또는 동치(iff) 규칙을 사용하여 증명 목표에서 항을 치환합니다. 예를 들어:
- 항등식: $$a = b $$
- 동치: $$P \Leftrightarrow Q $$

rw 전술은 이를 통해 등호 관계를 처리하고 목표를 단순화합니다.

**2) Apply (apply) 전술**[1]

함의 규칙 $$P \Rightarrow Q $$를 사용하여 증명 목표와 가정을 변환합니다. 현재 증명 목표 $Q$와 함의 $P \Rightarrow Q$가 있을 때, apply는 목표를 $P$로 변환하여 "P를 증명하면 Q를 증명할 수 있다"는 의미의 사후 조건 축소를 수행합니다.

**알고리즘 1: 호출 가능한 정리 찾기**[1]

```
입력: 후보 명제 s, 잠재적 호출 가능 정리 T_pinv, 명령 템플릿 I
출력: 호출 가능 정리 T_inv

(env, init_state) ← INIT(s)  // Lean 환경 초기화
T_inv ← ∅
for t in T_pinv do
    for i in I do
        inst ← FORMAT(t, i)  // 명령 생성
        next_state ← RUN_TAC(env, init_state, inst)  // 전술 실행
        if VALID(next_state) then
            Add (init_state, next_state, inst) to T_inv
        end if
    end for
end for
```

**명제 생성 프로세스**[1]

알고리즘의 단계:

1. **호출 가능 정리 검색**: 각 후보 정리에 대해 rw와 apply 명령 템플릿을 사용하여 호출 가능 정리를 찾음
2. **명제 변이**: 추상 구문 트리(AST)를 파싱하여 변경할 위치를 특정하고, 다음 증명 상태에서 새로운 가정 또는 결론을 추출
3. **증명 생성**: `have` 전술을 삽입하여 원본 증명과 통합:
   - 가정 변경: 역방향 rw를 사용하여 원본 가정 복원
   - 결론 변경: 원본 결론을 보조 보조정리로 도입

4. **정리 검증**: 생성된 정리를 Lean에 제출하여 정확성 확인

**have 전술의 역할**[1]

증명 생성에서 핵심적입니다:

$$ \text{have } h: Q := \text{by } P \Rightarrow Q; \text{exact } h $$

이는 새로운 가정 $Q$를 현재 증명 상태에 도입하며, 기하 문제에서 보조점을 구성하는 것과 유사하게 증명을 정규화합니다.

### 3. 실험 결과 및 성능 향상

**데이터 합성 결과**[1]

| 전술 | 후보 정리 | 1단계 | 2단계(검증) | 확대 배수 | 전환율 |
|------|---------|--------|-----------|---------|--------|
| rw | 110,657 | 5,081,544 | 2,830,817 | ×25 | 56% |
| apply | 78,871 | 9,483,504 | 3,495,832 | ×44 | 37% |

**Leandojo 벤치마크 결과**[1]

| 방법 | 무작위 분할 | 새로운 전제 분할 | 개선도 |
|------|-----------|----------------|--------|
| Llama3-8b 기본 | 58.22% | 38.52% | - |
| + rw | 59.62 | 42.13 | +3.62% |
| + apply | 58.84 | 41.29 | +2.77% |
| + rw + apply | **59.82** | **43.22** | **+4.70%** |
| deepseek-coder-7b 기본 | 57.70 | 39.24 | - |
| + rw + apply | **60.39** | **43.46** | **+4.22%** |

**연속 사전훈련(CPT)의 효과**[1]

연속 사전훈련은 모든 감독 미세조정 설정에서 긍정적인 영향을 보였으며, 특히 새로운 전제 분할에서 더 큰 개선을 달성했습니다:
- Mathlib-train만 사용 시: 38.52%
- CPT + SFT (rw + apply): 43.22%

### 4. 일반화 성능 향상 가능성 - 핵심 분석

**분포 외(Out-of-Distribution, OOD) 성능**[1]

miniF2F 벤치마크(경쟁 수준의 문제)에서의 성능:

| 데이터 조합 | 정확도 | 개선도 | rw 사용율 | apply 사용율 | norm_num | linarith |
|-----------|-------|--------|---------|------------|----------|----------|
| Mathlib-train | 34.01% | - | 16.10% | 0.00% | 27.12% | 16.95% |
| + rw | 35.24 | +1.23% | 18.75 | 0.78 | 14.84 | 21.88 |
| + apply | 36.07 | +2.06% | 8.87 | 2.42 | 20.16 | 15.63 |
| + rw + apply | **36.48** | **+2.47%** | 12.31 | 0.77 | 26.92 | 16.92 |

**일반화 성능 향상의 핵심 인사이트**[1]

1. **전술 다양성의 중요성**: 단일 전술의 state-tactic 쌍을 늘리면 빠른 포화에 도달합니다(Figure 4). 약 30k 데이터 포인트에서 최적 지점을 보임.

2. **전술 분포의 영향**: 합성 데이터의 전술 분포를 조정하면 경쟁 수준 문제 해결에 도움이 됩니다:
   - miniF2F 증명에서 고급 자동화 전술(simp, omega, linarith, norm_num)이 약 50% 이상 사용
   - 기본 전술(rw, apply)은 약 20% 정도 사용

3. **제한된 OOD 성능 개선**: Leandojo에서 4.70% 향상에 비해 miniF2F에서 2.47% 향상에 그친 이유:
   - 합성 데이터가 주로 기본 전술 중심
   - miniF2F는 고급 자동화 전술 선호
   - 분포 편차로 인한 성능 차이

4. **누적 효과**: 여러 전술의 데이터를 결합하면 긍정적 누적 효과 관찰:
   - rw만: +3.62%
   - apply만: +2.77%
   - rw + apply: +4.70% (단순 합 초과)

**데이터양의 영향 분석**[1]

Figure 4에서 보이듯, 합성 데이터 포인트 수에 따른 성능:
- 0.25 다운샘플(7.5k): +1.3% 개선
- 0.5 다운샘플(15k): +1.8% 개선
- **1.0 (30k): +3.67% 개선 (최적)**
- 50 중복 제거 임계값(500k): +2.41% (포화)
- 중복 제거 없음(3M): 유사 성능

### 5. 모델 한계 및 제약사항

**기술적 한계**[1]

1. **합성 비용**: 
   - rw: 약 14일 (4,096 CPU 코어)
   - apply: 약 7일 (2,048 CPU 코어)
   - O(n²) 복잡도 문제

2. **데이터 다양성 제한**:
   - 두 가지 전술만 사용 (rw, apply)
   - 도메인 지식 활용 부족
   - 형식화 품질 65% 수준의 전환율

3. **아키텍처 제약**:
   - 상향식(bottom-up) 접근이 아닌 하향식(top-down) 방식
   - 기존 정리에 의존하여 다양성 제한
   - 단일 라운드 합성만 실현 (반복 불가능)

4. **단기 vs 장기 성능**:
   - 인-디스트리뷰션(Leandojo): 4.70% 향상
   - 아웃-오브-디스트리뷰션(miniF2F): 2.47% 향상

### 6. 앞으로의 연구에 미치는 영향과 고려사항

**최신 연구 동향(2024-2025)**[2][3][4][5][6]

**1. 경쟁하는 합성 방법들의 출현**

LeanNavigator는 상태 전이 그래프를 탐색하여 4.7M 정리를 생성했으며, Alchemy의 전술 기반 접근과 상이한 철학을 제시합니다. HunyuanProver는 확장 가능한 합성 프레임워크와 가이드된 트리 탐색을 결합합니다.[3][2]

**2. 형식 정렬 및 검증의 중요성**

FormalAlign는 비형식-형식 정렬을 자동으로 평가하는 프레임워크를 제시했으며, Theorem Prover as a Judge (TP-as-a-Judge)는 반복적 형식화를 통해 Lean 실행률을 60%에서 87%로 개선했습니다. 이는 합성 데이터 품질의 중요성을 강조합니다.[5][7]

**3. 신경-기호 통합의 강화**

형식 수학 추론에 대한 최근 위치 논문은 자동 형식화와 강화 학습의 결합 가능성을 제시합니다. AlphaProof의 성공(IMO 은메달)은 이러한 통합의 잠재력을 입증합니다.[6]

**권장 후속 연구 방향**[4][2][3][6][1]

**1. 다양성 강화**
- 추가 전술(simp, omega, linarith, norm_num) 포함
- 고급 전술의 합성으로 OOD 성능 개선
- 도메인 지식 기반 휴리스틱 활용

**2. 효율성 개선**
- Leandojo 인터페이스 경량화 (메모리 사용량 감소)
- 병렬 처리 최적화 (현재 O(n²) 복잡도)
- 실시간 합성 파이프라인 개발

**3. 반복적 합성**
- 합성된 정리를 seed로 활용한 다중 라운드 실행
- 누적 효과 극대화

**4. 기호-신경 하이브리드 접근**
- 증명 탐색에 강화 학습 통합
- 검색 전략 최적화 (현재 best-first search)

**5. 일반화 성능 타겟**
- 고급 전술 중심 합성으로 OOD 성능 4% 이상으로 증대
- 전술 분포 적응형 조절
- 경쟁 수준 문제 해결율 향상

**6. 자동 형식화와의 통합**
- 비형식 수학 자동 형식화의 출력을 seed로 활용
- RAG(Retrieval-Augmented Generation) 데이터베이스로 활용 가능성

### 결론

Alchemy는 신경 정리 증명의 데이터 부족 문제를 기호적 변이를 통해 혁신적으로 해결했습니다. Mathlib의 정리를 57배로 확대하고, Leandojo에서 4.70%의 성능 향상을 달성한 것은 의미 있는 성과입니다. 그러나 OOD 성능 향상의 제한(2.47%)과 단일 전술 중심 접근의 한계는 향후 연구의 중요한 방향입니다.

최신 연구 동향으로 볼 때, 미래의 강력한 정리 증명기는 다음을 통합해야 합니다: (1) 다양한 전술 기반의 합성, (2) 높은 품질의 형식 정렬 검증, (3) 강화 학습과의 결합, (4) 자동 형식화와의 협력. Alchemy는 이러한 통합을 위한 견고한 기초를 제공하며, 형식 수학 분야에서 AI의 역할 확대에 중추적인 기여를 할 것으로 예상됩니다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/18a7bf7b-d621-4db2-9b58-0f2777c8fdcd/2410.15748v2.pdf)
[2](https://arxiv.org/pdf/2503.04772.pdf)
[3](http://arxiv.org/pdf/2412.20735.pdf)
[4](https://openreview.net/forum?id=EeDSMy5Ruj)
[5](https://aclanthology.org/2025.acl-long.1448/)
[6](https://arxiv.org/html/2412.16075v1)
[7](https://openreview.net/forum?id=B5RrIFMqbe)
[8](https://arxiv.org/html/2410.15748)
[9](https://arxiv.org/pdf/2210.12150.pdf)
[10](https://arxiv.org/pdf/2404.12534.pdf)
[11](http://arxiv.org/pdf/2410.06209v4.pdf)
[12](https://arxiv.org/pdf/2406.03847.pdf)
[13](http://arxiv.org/pdf/2101.02602.pdf)
[14](https://arxiv.org/html/2210.12150v5)
[15](https://figshare.com/articles/journal_contribution/The_Lean_Theorem_Prover_system_description_/6492815/1/files/11937416.pdf)
[16](https://openreview.net/forum?id=7NL74jUiMg)
[17](https://www.nature.com/articles/s43246-024-00731-w)
[18](https://arxiv.org/html/2503.19551v3)
[19](https://openreview.net/pdf?id=7NL74jUiMg)
[20](https://arxiv.org/html/2510.24216v1)
[21](https://bluegen.ai/what-is-the-advantage-of-using-synthetic-data-in-machine-learning/)
[22](https://arxiv.org/abs/2410.15748)
[23](https://openreview.net/pdf?id=c4wEKJOjY3)
[24](https://reports.weforum.org/docs/WEF_Synthetic_Data_2025.pdf)
[25](https://www.turing.com/resources/lean-and-symbolic-reasoning-in-llms-for-math-problem-solving)
[26](https://arxiv.org/pdf/2502.03078.pdf)
[27](http://arxiv.org/pdf/2201.12677.pdf)
[28](http://arxiv.org/pdf/2410.16705.pdf)
[29](https://arxiv.org/html/2410.11963v1)
[30](https://arxiv.org/abs/2403.04190)
[31](https://www.mdpi.com/1424-8220/24/1/266/pdf?version=1704185657)
[32](https://www.emergentmind.com/topics/autoformalization)
[33](https://proceedings.mlr.press/v235/qiao24a.html)
[34](https://openreview.net/forum?id=ZbOSRZ0JXH)
[35](https://aclanthology.org/2025.acl-long.1448.pdf)
[36](https://www.emergentmind.com/topics/deepseek-prover-v2-671b)
