# Large Language Models are Zero-Shot Reasoners

**핵심 주장 및 주요 기여**  
이 논문은 “Let’s think step by step”라는 단일 제로-샷(예제 없음) 프롬프트만으로도 대규모 언어 모델(LLM)이 고난도 다중단계 추론 과제를 상당 수준 해결할 수 있음을 보인다.  
주요 기여는 다음과 같다.  
- **Zero-shot-CoT**: 손수 만든 예제 없이, 모든 문제에 동일한 트리거 문구 “Let’s think step by step”를 삽입하여 체인 오브 사고(chain of thought)를 유도  
- 다양한 추론 벤치마크(산술·상징·논리·상식 문제)에서 제로-샷 기준 성능을 대폭 향상  
- 모델 규모 확장 시 제로-샷 추론 성능이 Few-shot-CoT(예제₈개)와 유사한 증가 곡선을 보임  

***

## 1. 해결 문제  
- 기존 LLM은 예제 기반 Few-shot 또는 태스크별 템플릿 방식 ZERO-shot에서 복잡한 다중 단계(system-2) 추론 과제에 취약  
- 특히 별도 예제 없이 지시어만으로 추론 경로를 끌어내는 범용 프롬프트는 미흡

## 2. 제안 방법: Zero-shot Chain of Thought  
1) 1단계: Reasoning Extraction  
   Q: X  
   A: Let’s think step by step.  
   → 모델이 단계별 추론 경로 $$z$$ 생성  
2) 2단계: Answer Extraction  
   “Q: X A: Let’s think step by step. $$z$$ Therefore, the answer (arabic numerals) is”  
   → $$z$$를 참고해 최종 답 $$\hat y$$ 추출 및 정제  
   
수식 표현:  

$$
\text{Prompt}_1 = \texttt{"Q: }x\texttt{ A: Let’s think step by step."}
\quad\xrightarrow{\text{LM}}\quad z
$$  

$$
\text{Prompt}_2 = [\text{Prompt}_1]\ ||\ z\ ||\ \texttt{"Therefore, the answer is"}
\quad\xrightarrow{\text{LM}}\quad \hat y
$$

## 3. 모델 구조  
- 별도 학습·미세조정 없이, 사전학습된 GPT-3 계열(text-davinci-002 등), PaLM(8B·62B·540B) 등 범용 LLM 활용  
- 디코딩은 그리디(확률 0 최적화), PaLM은 TopK=1  

## 4. 성능 향상  
| 벤치마크      | Zero-shot → Zero-shot-CoT(text-davinci-002) |
|---------------|---------------------------------------------|
| MultiArith    | 17.7% → **78.7%**                          |
| GSM8K         | 10.4% → **40.7%**                          |
| AQUA-RAT      | 22.4% → **33.5%**                          |
| SVAMP         | 58.8% → **62.1%**                          |
| Last Letter   | 0.2% → **57.6%**                           |
| Coin Flip     | 12.8% → **91.4%**                          |
| Date Understand| 49.3% → **67.5%**                         |
| Shuffled Objects| 31.3% → **52.4%**                        |

- PaLM 540B: MultiArith 25.5%→66.1%, GSM8K 12.5%→43.0%  
- Self-consistency(샘플 N=40) 결합 시 PaLM 540B GSM8K 43.0%→70.1%  

## 5. 한계 및 오류 분석  
- 소규모 모델(S·M)에서는 효과 미미  
- 일부 commonsense 오류, 불필요·누락 단계 발생  
- 다지선다 문제에서 답지 선택 어려움  

***

## 모델 일반화 성능 향상 가능성  
- **범용 트리거**: “Let’s think step by step” 하나만으로 여러 과제에 적용 가능 → 모델 내재 지식·추론 능력 광범위 활용  
- **스케일 민감도**: 대규모 모델 증대로 성능 곡선 급격 개선 → 추론 능력은 파라미터 수와 밀접  
- **셀프-컨시스턴시 포함** 시 더욱 안정적 성능 확보 → 무작위체인 앙상블로 일반화  

***

## 향후 연구 영향 및 고려 사항  
- **광범위 추론 프롬프트 탐색**: 다른 broad cognitive 능력(비교·창의력 등) 유도할 문구 자동 탐색  
- **소규모·경량 모델 적용**: 효율적 압축·증류 기법 결합해 경량 LLM에도 제로-샷 추론 능력 전이  
- **프롬프트 자동화·최적화**: 템플릿 자동 설계로 도메인 특화 없이 즉시 적용 가능한 범용 추론 지시어 확보  
- **안정성 강화**: 오류 패턴 분석 기반 필터링·보정 모듈 연계  

이 기법은 “최소 노력으로 최대 성능”을 달성하는 제로-샷 추론의 새 기준을 마련하며, 후속 연구에서 LLM의 내재 추론 능력을 보다 정교하게 탐구하고 활용하는 발판이 될 것이다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/0a7b1865-4ca2-44c8-9349-577a3f5fb14e/2205.11916v4.pdf)
