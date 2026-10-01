# Sharpness-Aware Pretraining Mitigates Catastrophic Forgetting

**분석 대상:** Ishaan Watts, Catherine Li, Sachin Goyal, Jacob Mitchell Springer, Aditi Raghunathan의 논문, 업로드된 **arXiv:2605.02105v1, 2026년 5월 4일 버전**입니다. 아래 페이지는 업로드된 PDF의 인쇄 페이지 번호이며, 외부 연구와의 비교는 별도 절에서 구분합니다. :chatgpt-content-reference{index="0"}

## 1. Executive summary — 9문장

1. 이 연구의 목적은 **사전학습 직후 가장 우수한 모델이 아니라, 이후 추가 학습이나 압축을 거친 뒤에도 기존 능력을 잘 유지하는 모델을 만드는 사전학습 방법**을 찾는 것입니다. **[저자 보고: pp. 1–2]** :chatgpt-content-reference{index="1"}
2. 저자들은 가중치를 조금 변경해도 손실이 크게 증가하지 않는 해를 유도하기 위해 **SAM, 높은 최대 학습률, 짧은 학습률 감소 구간**이라는 세 가지 개입을 연구합니다. **[저자 보고: pp. 1–3]** :chatgpt-content-reference{index="2"}
3. 대표적인 60M 모델·192B 토큰 실험에서 SAM은 StarCoder 추가 학습의 검증 손실을 맞췄을 때 사전학습 손실 증가를 약 `+0.5`에서 `+0.1`로 줄여, 저자 기준 **망각 80% 감소**를 보였습니다. **[저자 보고: pp. 4–5, Figure 2]** :chatgpt-content-reference{index="3"} :chatgpt-content-reference{index="4"}
4. 이러한 이점은 사전학습 손실이 비슷하거나 오히려 나쁜 경우에도 나타났으며, 높은 최대 학습률과 짧은 감소 구간에서도 학습과 기존 능력 보존 사이의 절충이 개선되었습니다. **[저자 보고: pp. 4–7, Figures 3–7]** :chatgpt-content-reference{index="5"} :chatgpt-content-reference{index="6"}
5. 이미 4T 토큰으로 학습된 OLMo-2-1B에 50B 토큰의 SAM 중간학습을 적용하면 MetaMath, StackMathQA, Tülu-3 추가 학습 후 망각이 각각 **31%, 22%, 35% 감소**했지만, **MusicPile에서는 개선되지 않았습니다**. **[저자 보고: p. 1 Figure 1; pp. 8, 11]** :chatgpt-content-reference{index="7"} :chatgpt-content-reference{index="8"}
6. 같은 1B 모델의 별도 4비트 양자화 실험에서는 평균 벤치마크 점수가 AdamW의 **38.89에서 SAM의 40.60으로 높아졌으며**, 저자의 공통 기준점 계산으로 성능 하락이 약 **40% 감소**했습니다. **[저자 보고: p. 8; p. 22, Table 8]** :chatgpt-content-reference{index="9"} :chatgpt-content-reference{index="10"}
7. 저자들은 이러한 결과를 추가 학습이 실제로 이동하는 방향의 손실 곡률 감소와 연결하지만, 제시된 이차 근사는 **실험적 설명이지 일반적인 망각 방지 정리가 아닙니다**. **[저자 보고와 해석: pp. 8–11, Figures 10–12]** :chatgpt-content-reference{index="11"} :chatgpt-content-reference{index="12"}
8. 전체 사전학습에 SAM을 적용하면 계산량이 대략 두 배가 되지만, 마지막 10% 구간에만 적용하는 방식은 약 10%의 추가 계산으로 상당한 이점을 회수합니다. **[저자 보고: pp. 7–8, Figure 9]** :chatgpt-content-reference{index="13"}
9. 따라서 가장 타당한 결론은 **“SAM이 모든 형태의 일반화를 개선한다”가 아니라 “사전학습을 후속 변경에 대한 적응성과 능력 보존까지 고려해 최적화할 가치가 있다”**이며, 반복 실험의 불확실성과 새로운 분포에 대한 일반화는 추가 검증이 필요합니다. **[본 분석: pp. 10–11의 주장 및 p. 19의 실험 설정에 근거]** :chatgpt-content-reference{index="14"} :chatgpt-content-reference{index="15"} :chatgpt-content-reference{index="16"}

> **용어 설명**  
> **사전학습:** 폭넓은 데이터로 기본 능력을 학습하는 단계.  
> **망각:** 이후 모델을 변경하면서 이전에 잘하던 작업의 성능이 떨어지는 현상.  
> **SAM:** 현재 가중치뿐 아니라 그 주변에서도 손실이 낮도록 학습하는 방법.  
> **양자화:** 가중치를 더 적은 비트로 표현해 모델 저장·실행 비용을 줄이는 방법.  
> **곡률:** 가중치를 움직였을 때 손실의 기울기가 얼마나 빠르게 변하는지를 나타내는 양.

---

## 2. 연구의 목적·필요성과 핵심 주장

### 2.1 해결하려는 문제

**[저자 보고]** 기존 사전학습 설정은 주로 사전학습 검증 손실이나 기본 모델의 벤치마크 점수를 기준으로 선택됩니다. 그러나 모델은 실제 사용 전에 특정 작업에 대한 추가 학습이나 압축을 거치므로, **변경 전 성능만으로 변경 후 성능을 예측하기 어렵다**는 것이 출발점입니다. **[pp. 1–2]** :chatgpt-content-reference{index="17"} :chatgpt-content-reference{index="18"}

특히 이 논문은 다음 두 현상을 연결합니다.

| 현상 | 의미 | 논문에서의 관계 |
|---|---|---|
| **Catastrophic forgetting, 치명적 망각** | 새로운 작업을 학습하면서 기존 능력이 저하되는 현상 | 추가 학습 전후의 사전학습 손실 또는 벤치마크 성능 차이로 측정 |
| **Catastrophic overtraining, 치명적 과도학습** | 사전학습을 더 오래 해 기본 모델은 좋아지지만, 이후 추가 학습을 거친 모델은 오히려 나빠지는 현상 | 장기 사전학습으로 업데이트 민감도가 커지는 문제로 해석 |

**근거:** §2.1, §3.2, §5; pp. 2, 5–6, 9–10. :chatgpt-content-reference{index="19"} :chatgpt-content-reference{index="20"} :chatgpt-content-reference{index="21"}

> **주의:** 여기서 *overtraining*은 단순히 훈련 데이터에 과적합했다는 뜻이 아닙니다. **사전학습 검증 성능이 개선되는 상황에서도 후속 적응 결과가 나빠질 수 있다**는 더 구체적인 현상입니다.

**[본 해석]** 연구의 필요성은 평가 단위를 “기본 모델 하나”에서 “사전학습 → 추가 학습 → 배포용 변경”으로 확장하는 데 있습니다. 다만 이 논문이 실제로 검증한 것은 주로 **지도 추가 학습과 별도의 양자화**이며, 모든 개발 단계를 연속 수행한 전체 파이프라인은 아닙니다. **[p. 8, §3.5.1; p. 11, §6]** :chatgpt-content-reference{index="22"} :chatgpt-content-reference{index="23"}

### 2.2 핵심 주장과 근거

| 핵심 주장 | 저자가 제시한 근거 | 본 해석 및 주의점 | 위치 |
|---|---|---|---|
| **기본 모델의 손실만으로 후속 모델의 품질을 판단할 수 없다.** | SAM은 사전학습 손실이 비슷하거나 더 나빠도 추가 학습 후 더 나은 절충을 보임 | 사전학습 손실의 유용성을 부정하는 것이 아니라, **충분한 지표가 아니라는 증거** | p. 4, Figure 3; p. 6, Figure 5. :chatgpt-content-reference{index="24"} :chatgpt-content-reference{index="25"} |
| **SAM 사전학습은 같은 추가 학습 수준에서 망각을 줄인다.** | 60M·192B·StarCoder에서 망각 80% 감소, 5개 데이터셋의 절충 곡선 개선 | **동일 토큰 예산 비교**이지 동일 계산량 비교는 아님 | pp. 4–5, Figure 2. :chatgpt-content-reference{index="26"} :chatgpt-content-reference{index="27"} |
| **장기 사전학습일수록 SAM의 상대적 이점이 커진다.** | 60M 모델의 12B–192B 토큰 실험에서 격차 확대 | 관측 범위의 경향이며, 무한히 학습해도 망각하지 않는다는 뜻은 아님 | p. 5, Figure 4. :chatgpt-content-reference{index="28"} |
| **높은 최대 학습률·짧은 감소 구간도 효과가 있다.** | 기본 모델 손실에 최적인 설정과 학습–망각 절충에 최적인 설정이 다름 | 모든 작업·모든 지표에 대해 “클수록/짧을수록 좋다”는 보편 법칙은 아님 | pp. 5–7, Figures 6–7. :chatgpt-content-reference{index="29"} :chatgpt-content-reference{index="30"} |
| **이점은 추가 학습 외의 가중치 변경에도 나타난다.** | 4비트 양자화와 무작위 가중치 교란 후 손실 증가 감소 | 가중치 변경에 대한 강건성이지, 곧바로 새로운 입력 분포에 대한 일반화는 아님 | pp. 6, 8, Figure 8. :chatgpt-content-reference{index="31"} |
| **마지막 단계에만 SAM을 적용해도 유용하다.** | 학습률 감소 구간 10%에만 SAM을 적용해 절충 개선 | 전체 SAM보다 경제적이지만, 최적 적용 시점·길이를 완전히 규명한 것은 아님 | pp. 7–8, Figure 9. :chatgpt-content-reference{index="32"} |
| **1B 모델에서도 일부 효과가 유지된다.** | 수학·지시 데이터 3종과 4비트 양자화에서 개선 | MusicPile은 실패 사례이며, 더 큰 모델 전체로의 확장은 미검증 | p. 1, Figure 1; p. 22, Table 8; p. 11 결론. :chatgpt-content-reference{index="33"} :chatgpt-content-reference{index="34"} |
| **추가 학습 방향의 곡률 감소가 관측 결과와 연결된다.** | SAM 및 높은 사전학습 학습률에서 방향별 곡률 감소 | 곡률과 실제 업데이트 방향이 함께 달라지므로, 곡률만의 독립적 인과효과는 분리되지 않음 | pp. 9–11, Figures 10–12. :chatgpt-content-reference{index="35"} |

> **학습–망각 절충:** 새 작업을 더 잘하게 만드는 것과 기존 능력을 유지하는 것 사이의 균형.  
> **강건성:** 일정한 변경이나 교란을 받아도 성능이 크게 떨어지지 않는 성질.

---

## 3. 모델 구조와 실험 설계

### 3.1 새로운 모델 구조를 제안한 논문은 아니다

**[저자 보고]** 기존 OLMo 계열 언어모델을 사용하고, 주로 **최적화 방법과 학습률 설정**을 바꿉니다. 따라서 핵심 기여는 새로운 네트워크 구성요소가 아니라 **어떤 가중치 상태로 사전학습을 끝낼 것인가**에 있습니다. **[pp. 3, 18–20]** :chatgpt-content-reference{index="36"} :chatgpt-content-reference{index="37"} :chatgpt-content-reference{index="38"}

| 저자 모델 명칭 | 층 수 | Attention head 수 | 은닉 차원 | 최대 문맥 길이 | 학습 설정 |
|---|---:|---:|---:|---:|---|
| OLMo-20M | 8 | 8 | 256 | 1,024 | DCLM 데이터로 처음부터 사전학습 |
| OLMo-60M | 8 | 8 | 528 | 1,024 | 동일 계열의 통제 실험 |
| OLMo-150M | 12 | 12 | 768 | 1,024 | 동일 계열의 통제 실험 |
| OLMo-2-1B | 16 | 16 | 2,048 | 2,048 | 기존 4T 토큰 체크포인트에 50B 토큰 중간학습 |

**구조의 출처:** p. 19, Table 1; p. 20, Table 4. 모든 설정은 SwiGLU를 사용하고, 표에 제시된 attention·residual·embedding dropout은 0입니다. :chatgpt-content-reference{index="39"} :chatgpt-content-reference{index="40"}

> **Attention head:** 문맥의 관계를 여러 관점에서 계산하는 병렬 처리 단위.  
> **은닉 차원:** 모델 내부에서 토큰 하나를 표현하는 벡터의 길이.  
> **SwiGLU:** 어떤 정보를 통과시킬지 조절하는 게이트를 사용하는 활성화 함수.  
> **체크포인트:** 특정 학습 시점의 모델 가중치를 저장한 상태.

**표 해석상의 주의:** 원문의 `Embedding Size = 100352`는 은닉 벡터 차원 100,352를 뜻하는 것으로 읽으면 안 됩니다. Table 1과 Table 4는 `Hidden dimensions`를 별도로 기재하며, 위 표는 그 값을 사용했습니다. :chatgpt-content-reference{index="41"} :chatgpt-content-reference{index="42"}

### 3.2 비교 조건

**소규모 통제 실험.** StarCoder, GSM8K, StackMathQA, Tülu-3, MusicPile에 대해 추가 학습하며, 학습률을 $10^{-6}$부터 $10^{-2}$까지 탐색합니다. 추가 학습은 AdamW로 수행하고, 한 epoch 또는 최대 10M 토큰으로 제한합니다. 따라서 **SAM으로 사전학습한 모델도 추가 학습 단계에서는 기본적으로 AdamW를 사용**합니다. **[pp. 4, 21, Table 7]** :chatgpt-content-reference{index="43"} :chatgpt-content-reference{index="44"}

**1B 실험.** 두 모델 모두 같은 기존 체크포인트에서 시작해 Dolmino 혼합 데이터로 50B 토큰을 추가 학습합니다. 이후 MetaMath는 약 80M 토큰의 1 epoch, 나머지 3개 데이터셋은 50M 토큰으로 추가 학습하며, 학습률을 $2\times10^{-6}$부터 $2\times10^{-4}$까지 탐색합니다. **[pp. 18–19, Tables 1–3]** :chatgpt-content-reference{index="45"} :chatgpt-content-reference{index="46"}

> **Epoch:** 학습 데이터 전체를 한 번 사용하는 단위.  
> **중간학습:** 대규모 사전학습과 작업별 추가 학습 사이에 수행하는 추가적인 학습 단계.

**[본 해석]** 최적화 효과를 비교하기 위해 동일한 모델 계열과 데이터를 사용하고, 추가 학습률을 넓게 탐색한 점은 강점입니다. 반면 사전학습 학습률은 AdamW의 기본 모델 검증 손실에 맞춰 선택한 뒤 SAM에도 재사용했으므로, 이는 **모든 방법을 각자의 후속 성능 목표에 맞춰 동등하게 최적화한 비교**는 아닙니다. **[pp. 20–21, §C.1.2]** :chatgpt-content-reference{index="47"}

---

## 4. 제안 방법과 수식

### 4.1 “얼마나 배웠는가”와 “얼마나 잊었는가”를 함께 측정한다

**[저자 정의를 정리]**

```math
\Delta_{\text{FT}}
=
\theta_{\text{FT}}-\theta_{\text{PT}}
```

```math
F_{\mathcal L}
=
\mathcal L_{\text{PT}}(\theta_{\text{FT}})
-
\mathcal L_{\text{PT}}(\theta_{\text{PT}})
```

여기서 $\theta_{\text{PT}}$는 사전학습 가중치, $\theta_{\text{FT}}$는 추가 학습 후 가중치, $\Delta_{\text{FT}}$는 가중치 이동량입니다. $\mathcal L_{\text{PT}}$는 사전학습 분포의 손실이고, $F_{\mathcal L}$는 그 손실의 증가량으로 표현한 망각입니다. $F_{\mathcal L}$ 기호는 설명을 위해 도입했습니다. **[pp. 2–3, §2.1–2.2]** :chatgpt-content-reference{index="48"} :chatgpt-content-reference{index="49"}

논문은 한 모델만 비교하지 않고, 여러 추가 학습 설정으로 얻은 결과 집합을 비교합니다.

```math
\mathcal T(\theta_{\text{PT}})
=
\left\{
\left(
\mathcal L_{\text{PT}}(\theta_{\text{FT}}),
\mathcal L_{\text{FT}}(\theta_{\text{FT}})
\right)
:
\theta_{\text{FT}}\in\Theta_{\text{FT}}(\theta_{\text{PT}})
\right\}
```

$\mathcal L_{\text{FT}}$는 추가 학습 데이터의 검증 손실이고, $\Theta_{\text{FT}}(\theta_{\text{PT}})$는 서로 다른 추가 학습 설정에서 얻는 모델 집합입니다. 두 손실 모두 낮을수록 좋습니다. **[p. 2, §2.1]** :chatgpt-content-reference{index="50"}

> **Pareto frontier, 파레토 전선:** 한 성능을 더 개선하려면 다른 성능을 희생해야 하는 최선의 결과 경계. 여기서는 “같은 새 작업 성능에서 가장 덜 잊는 모델들”에 해당합니다.

**중요한 구별:** Figure 2 등의 가로축은 대체로 증가량 $F_{\mathcal L}$ 자체가 아니라 **추가 학습 후의 절대 사전학습 손실**입니다. 시작 손실이 다른 모델을 비교할 때는 절대 손실과 망각 증가량을 구별해야 합니다. **[p. 4, 평가 방법]** :chatgpt-content-reference{index="51"}

#### 동일 추가 학습 손실 비교

Appendix C.4에서는 각 체크포인트가 달성한 최소 추가 학습 손실을 구한 뒤, 그중 가장 큰 값을 공통 문턱값으로 사용합니다.

```math
\ell_i^{\min}
=
\min_{\theta'\in\Theta_{\text{FT}}(\theta_i)}
\mathcal L_{\text{FT}}(\theta'),
\qquad
\tau=\max_i \ell_i^{\min}
```

```math
R_i(\tau)
=
\min_{\substack{
\theta'\in\Theta_{\text{FT}}(\theta_i)\\
\mathcal L_{\text{FT}}(\theta')\leq\tau
}}
\mathcal L_{\text{PT}}(\theta')
```

$i$는 사전학습 체크포인트의 번호, $\ell_i^{\min}$은 해당 체크포인트의 최저 추가 학습 손실, $\tau$는 모든 체크포인트가 달성할 수 있도록 정한 공통 기준입니다. $R_i(\tau)$는 그 기준을 만족하는 결과 중 가장 낮은 사전학습 손실이며, 마지막 식은 원문의 선택 절차를 명시적으로 다시 쓴 것입니다. **[p. 22, 원문 식 (8)–(10)]** :chatgpt-content-reference{index="52"}

**[본 해석]** 이는 “덜 배웠기 때문에 덜 잊었다”는 설명을 줄여주는 설계입니다. 그러나 검증 손실이 같다고 실제 문제 해결 정확도, 생성물 품질, 추론 전략까지 같다는 뜻은 아닙니다.

### 4.2 이론적 직관: 가중치 이동량과 이동 방향의 곡률

**[저자 제시: 원문 식 (1)]**

```math
\kappa(u;H)
=
\frac{u^\top Hu}{\lVert u\rVert_2^2},
\qquad
H=\nabla^2\mathcal L_{\text{PT}}(\theta_{\text{PT}})
```

$H$는 사전학습 손실의 Hessian, $u\neq0$는 가중치 공간의 방향 벡터, $\kappa(u;H)$는 그 방향의 정규화된 곡률입니다. $\lVert u\rVert_2$는 벡터의 유클리드 길이입니다. **[p. 2]** :chatgpt-content-reference{index="53"}

> **Hessian, 헤시안:** 손실의 이차 미분을 모은 행렬. 가중치를 어느 방향으로 움직일 때 손실이 빠르게 증가하는지 보여줍니다.  
> **방향별 곡률:** 전체 지형이 얼마나 날카로운지보다, 실제로 이동하는 특정 방향이 얼마나 가파르게 휘는지를 측정한 값.

사전학습 가중치 주변에서 이차 Taylor 근사를 사용하면 다음과 같습니다.

$$
\mathcal L_{\text{PT}}(\theta_{\text{PT}}+\Delta)
\approx
\mathcal L_{\text{PT}}(\theta_{\text{PT}})
+
\nabla\mathcal L_{\text{PT}}(\theta_{\text{PT}})^\top\Delta
+
\frac12\Delta^\top H\Delta
$$

$\Delta$는 추가 학습 또는 양자화에 따른 가중치 변화이고, $\nabla\mathcal L_{\text{PT}}$는 사전학습 손실의 기울기입니다. **[p. 2, 원문 식 (2)]** :chatgpt-content-reference{index="54"}

기울기 항이 충분히 작다고 가정하면,

```math
F_{\mathcal L}
\approx
\frac12\Delta_{\text{FT}}^\top H\Delta_{\text{FT}}
=
\frac12
\lVert\Delta_{\text{FT}}\rVert_2^2
\kappa(\Delta_{\text{FT}};H)
```

즉 망각은 근사적으로 **얼마나 멀리 이동했는가**와 **그 이동 방향이 얼마나 날카로운가**의 곱으로 설명됩니다. **[p. 3, 원문 식 (3)–(4)]** :chatgpt-content-reference{index="55"}

> **Taylor 근사:** 복잡한 함수를 현재 위치 주변에서 간단한 다항식으로 근사하는 방법. 멀리 이동하면 생략한 고차항의 영향이 커질 수 있습니다.

**[저자 인정 한계]** 사전학습 단계에서는 미래의 추가 학습 방향을 알 수 없고, 큰 업데이트에 대해서는 이차 근사의 정확성이 보장되지 않습니다. **[p. 3]** :chatgpt-content-reference{index="56"}

**[본 해석]** 이 수식은 좋은 가설이지만, “곡률을 낮추면 어떤 추가 학습에서도 망각이 줄어든다”는 보장은 아닙니다. 특히 Figure 12의 방향은 각 모델이 실제로 추가 학습한 결과에서 얻으므로, 측정량은 **기본 모델만의 속성이 아니라 기본 모델과 추가 학습 과정의 결합 속성**입니다. 저자도 이를 명시합니다. **[p. 9]** :chatgpt-content-reference{index="57"}

### 4.3 명시적 방법: SAM

**[저자 제시: 원문 식 (5)]**

$$
\min_{\theta}
\max_{\lVert\epsilon\rVert_2\leq\rho}
\mathcal L_{\text{PT}}(\theta+\epsilon)
$$

$\theta$는 학습할 가중치, $\epsilon$은 가중치 교란, $\rho>0$는 허용 교란 반경입니다. 현재 위치의 손실만 낮추는 것이 아니라, **반경 $\rho$ 안에서 가장 불리하게 가중치를 바꿔도 손실이 낮도록** 학습하려는 목적입니다. **[p. 3]** :chatgpt-content-reference{index="58"}

실제 계산에서는 내부 최대화를 다음과 같이 근사합니다.

```math
g_t=\nabla_\theta\mathcal L_{\text{PT}}(\theta_t),
\qquad
\hat\epsilon_t
=
\rho\frac{g_t}{\lVert g_t\rVert_2}
```

```math
\tilde g_t
=
\left.
\nabla_{\vartheta}\mathcal L_{\text{PT}}(\vartheta)
\right|_{\vartheta=\theta_t+\hat\epsilon_t}
```

$t$는 학습 단계, $g_t$는 현재 위치의 기울기, $\hat\epsilon_t$는 손실을 증가시키는 방향으로 만든 교란, $\tilde g_t$는 교란 위치에서 다시 계산한 기울기입니다. 실무에서는 미니배치 손실로 이를 계산합니다. **[p. 17, §A.1.2, Figure 13]** :chatgpt-content-reference{index="59"}

**중요하게도 이 논문의 SAM은 AdamW를 기반 최적화기로 사용합니다.** 따라서 비교는 실질적으로 “기본 AdamW”와 “SAM으로 계산한 기울기를 사용하는 AdamW”입니다. :chatgpt-content-reference{index="60"}

이를 원문의 AdamW 식과 결합하면 업데이트는 다음 형태입니다.

```math
\theta_{t+1}
=
\theta_t
-
\eta_t
\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon_{\text{opt}}}
-
\eta_t\lambda_{\text{WD}}\theta_t
```

$\eta_t$는 학습률, $\hat m_t$와 $\hat v_t$는 $\tilde g_t$를 사용해 계산한 편향 보정된 일차·이차 모멘트, $\epsilon_{\text{opt}}$는 수치 안정화 상수, $\lambda_{\text{WD}}$는 weight decay 계수입니다. 이 식은 §A.1.1과 실제 SAM 기반 최적화기 설명을 합쳐 정리한 것입니다. :chatgpt-content-reference{index="61"} :chatgpt-content-reference{index="62"}

> **미니배치:** 한 번의 업데이트에 사용하는 데이터 묶음.  
> **모멘트:** 최근 기울기와 기울기 제곱의 이동평균. 업데이트 방향과 크기를 안정적으로 조절하는 데 사용됩니다.  
> **Weight decay:** 가중치 크기를 점진적으로 줄이는 정규화 방식.

저자들은 $\rho=0.05$를 사용합니다. SAM은 Hessian을 직접 계산하는 방식이 아니라 기울기를 두 번 평가하는 방식이므로, 단계당 계산량이 대략 두 배가 됩니다. **[pp. 3, 7]** :chatgpt-content-reference{index="63"} :chatgpt-content-reference{index="64"}

### 4.4 암묵적 방법: 높은 학습률과 짧은 감소 구간

**[저자 제시한 동기]** Edge of Stability 현상에서는 경사하강법의 학습률과 최대 곡률이 다음 관계 주변에서 움직일 수 있습니다.

$$
\lambda_{\max}(H)\approx\frac{2}{\eta}
$$

$\lambda_{\max}(H)$는 Hessian의 최대 고유값, $\eta$는 학습률입니다. 높은 학습률이 매우 날카로운 해에 머무르기 어렵게 만든다는 직관입니다. **[p. 3, §2.3.2]** :chatgpt-content-reference{index="65"}

> **최대 고유값:** Hessian이 나타내는 방향별 곡률 중 가장 큰 값.  
> **Edge of Stability:** 학습이 단순한 안정성 조건의 경계 근처에서 진행되는 현상.

**[본 해석]** 이 관계를 AdamW에 그대로 적용되는 정확한 법칙으로 읽으면 안 됩니다. 논문에서도 높은 학습률을 조사하는 동기로 사용하며, 실제 추가 학습 방향의 곡률 감소는 별도 실험으로 확인합니다. **[pp. 8–9]** :chatgpt-content-reference{index="66"} :chatgpt-content-reference{index="67"}

학습률 감소 구간은 WSD 일정으로 조절합니다. 전체 학습 단계를 $N$, 마지막 감소 구간을 $d$라고 하면, 높은 학습률을 오래 유지하다가 마지막 $d$단계에서 줄이는 방식입니다. 해당 실험에서는 최종 학습률을 최대 학습률의 10%까지 내립니다. **[p. 6]** :chatgpt-content-reference{index="68"}

$$
f=\frac{d}{N},
\qquad
\eta_{\text{final}}=0.1\,\eta_{\max}
$$

$f$는 감소 구간의 비율, $\eta_{\max}$는 최대 학습률입니다. 60M·192B·StarCoder 설정에서 학습–망각 절충은 시험한 값 중 ** $f=0.05$ **가 가장 좋았지만, 기본 모델 손실은 ** $f=0.20$ **에서 가장 좋았습니다. **[pp. 6–7, Figure 7]** :chatgpt-content-reference{index="69"}

> **WSD:** 학습률을 올리는 warmup, 일정하게 유지하는 stable, 낮추는 decay의 세 단계로 구성된 일정.  
> **Annealing:** 학습 후반에 학습률을 낮추는 과정.

### 4.5 실용적 방법: 마지막 구간에만 SAM 적용

**[저자 제안]**

```math
\text{Optimizer}(t)
=
\begin{cases}
\text{AdamW}, & t\leq N-d,\\
\text{SAM with AdamW}, & t>N-d.
\end{cases}
```

$t$는 현재 단계이고, 나머지 기호는 앞과 같습니다. 논문의 대표 설정은 마지막 10% 구간만 SAM으로 바꾸는 것입니다. **[pp. 7–8, Figure 9]** :chatgpt-content-reference{index="70"}

**[본 계산: 단계당 SAM 비용을 두 배로 근사]**

```math
\frac{C_{\text{late-SAM}}}{C_{\text{AdamW}}}
\approx
(1-f)+2f
=
1+f
```

$C$는 총 계산량입니다. 따라서 $f=0.1$이면 약 10%의 추가 계산이 필요합니다. 다만 이는 계산량 근사이며, 실제 GPU 시간·통신 비용·메모리 비용을 직접 측정한 수치는 아닙니다.

또한 1B 실험의 50B 토큰은 기존 4T 토큰의 **1.25%**이지만, **새로 수행하는 50B 토큰 중간학습 자체의 비용은 SAM 때문에 대략 두 배**라는 점을 구별해야 합니다. 기존 사전학습 비용을 포함하느냐에 따라 “추가 비용이 작다”의 의미가 달라집니다. **[p. 8의 설정에 대한 본 계산]** :chatgpt-content-reference{index="71"}

### 4.6 보조 실험: 기존 망각 완화법과의 결합

저자들은 EWC를 추가 학습에 적용하는 실험도 수행합니다.

```math
\mathcal L_{\text{EWC}}(\theta)
=
\mathcal L_{\text{new}}(\theta)
+
\lambda_{\text{EWC}}
\sum_i F_i(\theta_i-\theta_i^*)^2
```

$\mathcal L_{\text{new}}$는 새 작업 손실, $\theta_i^*$는 이전 가중치, $F_i$는 해당 가중치의 중요도 추정값, $\lambda_{\text{EWC}}$는 보존 강도입니다. 중요도는 Fisher 정보로 추정합니다. **[p. 32, 원문 식 (11)]** :chatgpt-content-reference{index="72"}

> **EWC:** 이전 작업에 중요한 가중치를 크게 바꾸지 못하도록 벌점을 주는 방법.  
> **Fisher 정보:** 특정 가중치가 모델의 확률 예측에 얼마나 민감하게 영향을 미치는지 나타내는 양.

**[저자 보고]** 60M·192B 모델의 StarCoder와 MusicPile 실험에서는 SAM 사전학습과 EWC를 결합한 결과가 AdamW 사전학습과 EWC의 결합보다 나았습니다. 따라서 두 접근은 적어도 이 설정에서는 상호 보완적입니다. **[pp. 32–33, Figure 42]** :chatgpt-content-reference{index="73"} :chatgpt-content-reference{index="74"}

---

## 5. 성능 수치의 정확한 의미와 재계산

### 5.1 “80% 개선”은 정확도 80% 향상이 아니다

StarCoder의 80%는 **사전학습 손실 증가량의 상대 감소율**입니다.

```math
\text{Reduction}
=
1-\frac{F_{\text{SAM}}}{F_{\text{AdamW}}}
\approx
1-\frac{0.1}{0.5}
=
0.8
```

여기서 $F_{\text{SAM}}$과 $F_{\text{AdamW}}$는 각각의 추가 학습에 따른 사전학습 손실 증가량입니다. 따라서 이 수치를 “코딩 정확도 80% 향상” 또는 “전체 성능 80% 향상”으로 바꾸어 말하면 잘못입니다. **[pp. 4–5, Figure 2]** :chatgpt-content-reference{index="75"} :chatgpt-content-reference{index="76"}

### 5.2 1B 양자화 결과

| 상태 | AdamW 기반 모델 | SAM 기반 모델 |
|---|---:|---:|
| 양자화 전 평균 점수 | 43.2 | 42.9 |
| 4비트 양자화 후 평균 점수 | 38.89 | 40.60 |
| 각자의 시작점 대비 하락폭 — 본 계산 | 4.31점 | 2.30점 |

**출처:** p. 22, Table 8. :chatgpt-content-reference{index="77"}

저자들은 SAM의 시작 점수가 조금 낮다는 점을 고려해, 망각 비교의 기준을 두 모델 모두 **43.2**로 통일합니다. **[p. 8]** :chatgpt-content-reference{index="78"}

따라서 논문의 40%는 다음 계산과 일치합니다.

```math
1-
\frac{43.2-40.60}{43.2-38.89}
=
39.68\%
\approx40\%
```

이 식에서 분자는 공통 기준점 대비 SAM의 하락폭, 분모는 AdamW의 하락폭입니다. **반면 각자의 시작점을 쓰면 약 46.6%가 됩니다.** 후자는 본 재계산이며, 논문이 보고한 40%와는 기준이 다릅니다.

**[본 해석]** 공통 기준점은 SAM에 유리하게 부풀리는 방식이 아닙니다. 오히려 SAM의 낮은 시작 점수까지 불이익으로 포함합니다. 다만 다른 논문의 망각 감소율과 비교하려면 기준점 정의부터 맞춰야 합니다.

### 5.3 평균 개선이 모든 능력의 균등한 개선을 뜻하지 않는다

Table 8의 개별 작업 점수로 계산하면 다음과 같습니다.

| 항목 | 관측 또는 본 계산 |
|---|---|
| 양자화 후 전체 평균 차이 | SAM이 **+1.71점** |
| GSM8K 차이 | 21.60 → 35.20, **+13.60점** |
| GSM8K가 전체 평균 차이에 기여한 양 | 10개 작업 평균에서 **+1.36점**, 전체 차이의 약 **79.5%** |
| DROP 차이 | 31.40 → 30.20, **−1.20점** |
| AGIEval·MMLU-Pro | 양자화 후 점수가 동일 |

**출처 및 계산 근거:** p. 22, Table 8. :chatgpt-content-reference{index="79"}

**[본 해석]** 이는 결과를 무효화하지 않지만, **“일반 능력 전반이 고르게 보존되었다”는 해석은 지나칩니다.** 특히 79.5%는 *최종 평균 점수 차이의 기여도*이지, 전체 망각 감소율이나 통계적 신뢰도를 뜻하지 않습니다.

### 5.4 MusicPile 결과의 원문 불일치

p. 8에는 네 데이터셋 모두에서 더 강건하다는 포괄적 문장이 있지만, **Figure 1은 MusicPile의 개선을 0%로 표시하고, p. 11 결론은 개선되지 않았다고 명시**합니다. 따라서 이 분석에서는 더 구체적인 그림과 결론에 따라 **“1B MusicPile에서는 망각 개선 없음”**으로 판단합니다. :chatgpt-content-reference{index="80"} :chatgpt-content-reference{index="81"}

---

## 6. 가장 중요한 그림과 해석

| 선정한 그림 | 읽는 방법과 관측 | 해석에서 지켜야 할 경계 |
|---|---|---|
| **Figure 2, p. 4** | 가로축은 추가 학습 후 사전학습 손실, 세로축은 추가 학습 검증 손실입니다. 같은 높이에서 더 왼쪽에 있으면 동일한 새 작업 학습 수준에서 기존 능력을 더 잘 보존합니다. SAM의 우위를 가장 직접적으로 보여줍니다. :chatgpt-content-reference{index="82"} | 같은 토큰 예산에서의 결과이며, 모든 점이 통계적으로 구분된다는 뜻은 아닙니다. |
| **Figure 5, p. 6** | 가로축은 오른쪽으로 갈수록 기본 모델 손실이 낮아지는 역방향 축입니다. StarCoder 등에서는 AdamW가 더 좋은 기본 모델에 도달한 뒤 오히려 추가 학습 후 손실이 상승합니다. :chatgpt-content-reference{index="83"} | SAM의 이점이 단순한 조기 종료 효과라는 설명을 약화하지만, 모든 데이터셋에 같은 정도의 역전 현상이 나타나는 것은 아닙니다. |
| **Figures 6–7, p. 7** | 기본 모델 손실에 최적인 학습률·감소 길이와, 추가 학습 후 절충에 좋은 설정이 다릅니다. 평가 목표를 바꾸면 최적 하이퍼파라미터도 바뀐다는 그림입니다. :chatgpt-content-reference{index="84"} | Figure 6(c)의 양자화 손실은 완전히 단조롭지 않습니다. “학습률은 무조건 크게”라는 처방을 뒷받침하지 않습니다. |
| **Figure 9, p. 8** | 학습 후반만 SAM으로 바꿔도 추가 학습·양자화 결과가 개선됩니다. 이미 수행한 사전학습을 전부 반복하지 않는 실용적 개입의 근거입니다. :chatgpt-content-reference{index="85"} | SAM 적용의 최적 시점과 기간 전체를 탐색한 결과는 아닙니다. |
| **Figure 1, p. 1 및 Figure 16, p. 23** | Figure 1은 1B의 대표 결과를 압축해서 보여주고, Figure 16은 추가 학습 손실을 변화시켰을 때의 전체 절충을 보여줍니다. MusicPile 실패를 포함해 함께 읽어야 합니다. :chatgpt-content-reference{index="86"} :chatgpt-content-reference{index="87"} | Figure 1의 선택된 지점만으로 모든 추가 학습 설정에서 우세하다고 결론 내릴 수 없습니다. |
| **Figures 10–12, pp. 10–11** | 실제 손실과 이차 근사 예측을 비교하고, 추가 학습 방향의 곡률을 측정합니다. 작은 추가 학습률에서는 근사가 비교적 잘 맞지만, 큰 업데이트에서는 오차가 커집니다. :chatgpt-content-reference{index="88"} :chatgpt-content-reference{index="89"} :chatgpt-content-reference{index="90"} | 이차 근사가 대체로 위쪽에 있다는 관측은 수학적으로 증명된 상한이 아닙니다. |

**선정 우선순위:** 연구의 경험적 기여는 **Figure 2**, 기본 모델 중심 평가의 문제는 **Figure 5**, 실용성은 **Figure 9**, 규모 확장과 예외는 **Figure 1·16**, 기제 설명의 강점과 한계는 **Figures 10–12**에서 가장 잘 드러납니다.

---

## 7. 통계적으로 취약한 부분과 직접 비교하면 안 되는 수치

아래의 “취약”은 결과가 틀렸다는 뜻이 아니라, **보고된 정보만으로 어느 수준까지 확신할 수 있는가**에 대한 평가입니다.

| 항목 | 확인된 사실 | 평가 |
|---|---|---|
| **반복 시드와 불확실성** | 1B 설정에는 seed 42가 제시되지만, 주요 효과에 대한 반복 시드별 평균·표준편차·신뢰구간은 제시되지 않습니다. **p. 19, Table 1**. :chatgpt-content-reference{index="91"} | **[통계적 취약]** 효과의 방향은 관측되었으나 재학습 시 변동폭과 통계적 유의성은 판단하기 어렵습니다. |
| **많은 실험 수의 의미** | 80회 이상의 사전학습과 3,500회 추가 학습을 보고합니다. **p. 1**. :chatgpt-content-reference{index="92"} | **[주의]** 서로 다른 하이퍼파라미터 실험 수는 같은 조건의 독립 반복 수를 대신하지 않습니다. |
| **최적 지점 선택** | 광범위한 학습률 탐색에서 절충 경계와 문턱값을 선택합니다. 1B의 대표 기준은 “reasonable”한 절충 지점으로 설명됩니다. **pp. 20–22**. :chatgpt-content-reference{index="93"} :chatgpt-content-reference{index="94"} | **[통계적 취약]** 선택용 데이터와 최종 검증의 분리가 충분히 명확하지 않아, 선택된 최적점의 낙관 편향 가능성을 배제하기 어렵습니다. |
| **동일 토큰과 동일 계산량** | 전체 SAM은 같은 토큰 수에서 계산량이 대략 두 배입니다. **p. 7**. :chatgpt-content-reference{index="95"} | **[직접 비교 불가]** 토큰당 효율과 계산량당 효율은 다릅니다. 전체 SAM이 같은 계산 예산에서도 최선인지는 별도 문제입니다. |
| **80%와 31%·40%** | 작은 모델은 손실 증가량, 1B는 벤치마크 점수 하락을 사용합니다. **pp. 4, 8**. :chatgpt-content-reference{index="96"} :chatgpt-content-reference{index="97"} | **[직접 비교 불가]** 서로 다른 척도이므로 평균을 내거나 “규모가 커져 효과가 80%→31%로 줄었다”고 해석하면 안 됩니다. |
| **공식 OLMo와 논문 내 기준 모델** | 원본 OLMo 평균은 43.7, 논문 AdamW는 43.2이며, 연구에서는 중간학습 문맥 길이를 4,096에서 2,048로 변경했습니다. **pp. 18–19, Table 2**. :chatgpt-content-reference{index="98"} :chatgpt-content-reference{index="99"} | **[직접 비교 불가]** 공식 수치와의 차이를 SAM 또는 AdamW의 효과로만 해석할 수 없습니다. |
| **평균 점수의 구성** | Table 8에서 최종 평균 차이 대부분은 GSM8K에 집중되며 DROP은 악화됩니다. :chatgpt-content-reference{index="100"} | **[해석 취약]** 평균 개선이 모든 능력의 개선을 보장하지 않습니다. 작업별 결과와 최악의 하락을 함께 봐야 합니다. |
| **기제 분석 범위** | 주요 Hessian 분석은 60M·StarCoder이며, Figure 12는 대표 추가 학습률 $4\times10^{-4}$를 사용합니다. **p. 9**. :chatgpt-content-reference{index="101"} :chatgpt-content-reference{index="102"} | **[외삽 주의]** 1B의 모든 작업, 다른 구조, 다른 추가 학습 방식의 기제를 직접 검증한 것은 아닙니다. |
| **150M 토큰 예산 표기** | Table 5에는 150M의 예산이 15·30·60·120B로 적혀 있지만, Figure 18과 Figures 29–33에는 240B 결과가 등장합니다. **pp. 20, 24, 28–29**. :chatgpt-content-reference{index="103"} :chatgpt-content-reference{index="104"} :chatgpt-content-reference{index="105"} | **[문서 불일치]** 표의 누락인지 다른 설정인지 v1만으로 확정할 수 없습니다. |

> **신뢰구간:** 반복 표본이나 반복 실험의 불확실성을 반영한 추정 범위.  
> **선택 편향:** 많은 후보 중 가장 좋은 결과를 고르는 과정 때문에 그 결과가 실제보다 좋아 보일 수 있는 현상.  
> **외삽:** 확인한 모델·데이터·규모의 범위를 넘어 결과를 적용하는 것.

**추가 주의:** 양자화는 추가 학습 없이도 성능을 떨어뜨릴 수 있으므로, 논문은 “망각”을 넓게 사용합니다. 양자화 손실을 전통적인 순차 학습 망각과 완전히 동일한 현상으로 취급하기보다는 **모델 변경에 따른 능력 저하라는 공통 관점**으로 읽는 것이 정확합니다. **[pp. 2, 6]** :chatgpt-content-reference{index="106"} :chatgpt-content-reference{index="107"}

---

## 8. 모델의 일반화 성능 향상 가능성

### 8.1 무엇이 실제로 입증되었는가

**[저자 관점]** 논문은 일반화를 단순한 테스트 정확도뿐 아니라 **추가 학습·양자화 등 이후 변경을 거친 뒤에도 성능을 유지하는 능력**까지 넓혀 해석합니다. **[p. 10, §5]** :chatgpt-content-reference{index="108"}

그러나 다음 세 수준은 분리해야 합니다.

| 일반화의 수준 | 이 논문의 증거 | 판단 |
|---|---|---|
| **기본 모델 자체의 검증 성능** | SAM이 비슷하거나 더 나쁜 사전학습 손실을 보이기도 함 | 보편적인 개선을 주장할 수 없음 |
| **추가 학습 후 기존 분포·벤치마크 성능 보존** | 여러 통제 실험과 1B의 일부 작업에서 개선 | 논문이 가장 직접적으로 뒷받침하는 결과 |
| **전혀 새로운 분포·언어·환경에 대한 일반화** | 이를 분리해 평가한 광범위한 실험은 제시되지 않음 | 가능성은 있지만 아직 검증되지 않은 주장 |

첫 두 행의 근거는 Figure 3과 §3.5.1이고, 세 번째 판단은 보고된 평가 범위에 근거합니다. :chatgpt-content-reference{index="109"} :chatgpt-content-reference{index="110"} :chatgpt-content-reference{index="111"}

> **분포 밖 일반화, OOD generalization:** 학습·설정 선택 과정에서 경험하지 않은 종류의 데이터나 환경에서도 잘 작동하는 능력.  
> 특정 도메인으로 추가 학습한 뒤 그 도메인의 검증 데이터를 평가하는 것은, 그 도메인을 전혀 보지 않은 OOD 평가와 다릅니다.

### 8.2 일반화 개선을 기대할 수 있는 이유와 제한

**[본 해석: 긍정적 가능성]** 기존의 폭넓은 능력을 보존하면서 새 작업을 학습할 수 있다면, 전문화 때문에 발생하는 성능 손실을 줄일 수 있습니다. 같은 새 작업 검증 손실에서 사전학습 분포의 손실이 더 낮다는 결과는 이러한 가능성을 뒷받침합니다. **[pp. 4–6, Figures 2–5]** :chatgpt-content-reference{index="112"} :chatgpt-content-reference{index="113"}

**[본 해석: 제한]** 다음 등식은 성립한다고 볼 수 없습니다.

$$
\text{가중치 교란에 대한 강건성}
\;\not\equiv\;
\text{입력 분포 변화에 대한 일반화}
$$

왼쪽은 가중치를 바꿀 때의 안정성이고, 오른쪽은 입력 데이터의 성질이 바뀔 때의 성능입니다. 논문은 전자를 중심으로 측정하므로 후자를 별도로 실험해야 합니다.

또한 평탄함 자체가 일반화를 완전히 설명하지 못한다는 외부 연구가 있습니다. **A Modern Look at the Relationship between Sharpness and Generalization**은 여러 현대적 설정에서 sharpness와 일반화의 상관이 약하거나 반대 방향일 수 있음을 보였고, **Sharpness Minimization Algorithms Do Not Only Minimize Sharpness To Achieve Better Generalization**은 가장 평탄해도 일반화하지 못하는 경우를 이론적으로 분석합니다. :chatgpt-content-reference{index="114"}

**핵심 판단:** 이 논문에서 중요한 것은 막연한 “전체 지형의 평탄함”보다 **실제로 일어나는 후속 업데이트에 대한 민감도**입니다. 따라서 미래의 일반화 연구는 곡률 자체뿐 아니라 어떤 특징을 학습했는지, 어떤 방향으로 업데이트되는지, 어떤 능력이 보존되는지를 함께 측정해야 합니다.

---

## 9. 2020년 이후 관련 최신 연구 비교

아래는 직접 관련성이 높은 연구를 선별한 비교입니다. **본 논문의 실험 결과와 외부 연구의 결과는 다른 모델·데이터·척도에서 얻어졌으므로, 수치상 우열표가 아닙니다.** 문헌 확인 기준일은 2026년 10월 1일이며, 2026년 8월 공개·개정 연구까지 포함합니다.

### 9.1 방법·결과·차별점 비교

| 연구 — 정식 제목 | 해당 저자들이 보고한 핵심 내용 | 대상 논문과의 관계 및 본 해석 |
|---|---|---|
| **Foret 외, 2020 공개·2021 ICLR — Sharpness-Aware Minimization for Efficiently Improving Generalization** | 주변 최악 손실을 최소화하는 SAM을 제안하고 여러 이미지 분류·전이 설정에서 일반화 및 라벨 잡음 강건성 개선을 보고 | 대상 논문은 SAM 자체를 발명한 것이 아니라 **장기 언어모델 사전학습과 후속 망각**에 목적을 맞춰 적용합니다. :chatgpt-content-reference{index="115"} |
| **Liu 외, 2022 공개·2023 ICML — Same Pre-training Loss, Better Downstream: Implicit Bias Matters for Language Models** | 사전학습 손실이 같아도 후속 성능이 다를 수 있고, 일부 설정에서 평탄함이 그 차이를 설명함을 분석 | “사전학습 손실만으로 부족하다”는 주장에 중요한 선행 연구입니다. 대상 논문은 이를 **망각·양자화 및 긴 토큰 예산**으로 확장합니다. :chatgpt-content-reference{index="116"} |
| **Mehta 외, 2023 JMLR — An Empirical Investigation of the Role of Pre-training in Lifelong Learning** | 사전학습과 평탄한 해가 순차 학습의 망각을 완화함을 연구하고, §5.2에서는 SAM으로 평탄한 사전학습 해를 만드는 통제 실험도 수행 | **사전학습 평탄화로 망각을 줄인다는 발상 자체가 최초는 아닙니다.** 차별점은 현대 언어모델의 장기 사전학습, 손실을 맞춘 절충 평가, 압축 강건성, 1B 후반 개입의 결합입니다. :chatgpt-content-reference{index="117"} |
| **Wen·Ma·Li, 2022 공개·2023 개정 — How Does Sharpness-Aware Minimization Minimize Sharpness?** | 실제 SAM의 근사 과정과 확률적·전체 배치 설정이 어떤 sharpness를 줄이는지 구분 | 대상 논문의 “작은 배치 SAM이 평균적 곡률을 줄일 것”이라는 직관을 이해하는 배경입니다. **SAM을 모든 방향의 곡률을 직접 최소화하는 알고리즘으로 단순화하면 안 됩니다.** :chatgpt-content-reference{index="118"} |
| **Andriushchenko 외, 2023 ICML — A Modern Look at the Relationship between Sharpness and Generalization** | 재매개변수화 문제를 고려해도 sharpness가 일반화를 안정적으로 예측하지 못하는 설정들을 보고 | 대상 논문의 결과를 **보편적인 평탄함–일반화 법칙**으로 확장하는 데 대한 중요한 반론입니다. :chatgpt-content-reference{index="119"} |
| **Wen·Li·Ma, 2023 — Sharpness Minimization Algorithms Do Not Only Minimize Sharpness To Achieve Better Generalization** | 평탄함이 일반화를 보장하는 경우, 가장 평탄해도 일반화하지 못하는 경우 등을 이론·실험으로 구분 | 좋은 결과의 원인이 곡률뿐인지, 특징 학습이나 다른 최적화 편향인지 분리해야 한다는 시사점을 줍니다. :chatgpt-content-reference{index="120"} |
| **Springer 외, 2024 — Sharpness-Aware Minimization Enhances Feature Quality via Balanced Learning** | SAM이 잘 학습된 특징을 상대적으로 억제해 다양한 특징의 학습을 균형 있게 만들고, 여러 데이터셋에서 OOD 관련 이점을 보고 | 일반화 개선의 **대안적 기제**를 제공합니다. 대상 논문에서는 이 특징 균형 효과를 직접 측정하지 않았습니다. :chatgpt-content-reference{index="121"} |
| **Springer 외, 2025 ICML — Overtrained Language Models Are Harder to Fine-Tune** | 사전학습을 더 오래 하면 기본 모델은 좋아져도 추가 학습 후 성능이 악화될 수 있고, 파라미터 변경 민감도가 증가한다고 보고 | 대상 논문이 해결하려는 문제를 직접 제기한 선행 연구입니다. 대상 논문은 현상 진단을 넘어 **최적화 개입**을 시험합니다. :chatgpt-content-reference{index="122"} |
| **Zhou 외, 2025 ICLR — Sharpness-Aware Minimization Efficiently Selects Flatter Minima Late in Training** | 학습 마지막의 짧은 SAM 적용만으로 전체 SAM과 비슷한 일반화·평탄함을 얻는 현상을 분석 | 후반 SAM 자체도 선행 아이디어가 있습니다. 대상 논문의 추가 가치는 이를 **언어모델 중간학습과 망각·양자화 평가**로 연결한 것입니다. :chatgpt-content-reference{index="123"} |
| **Catalan-Tatjer 외, 2025 공개·2026 ICLR — Training Dynamics Impact Post-Training Quantization Robustness** | 최대 32B 모델·15T 토큰의 학습 궤적 분석에서 학습률 감소와 다른 설정이 양자화 민감도에 중요하며, 토큰 수만으로 설명하기 어렵다고 보고 | “오래 학습해서 나빠졌다”와 “학습 일정 때문에 민감해졌다”를 분리해야 합니다. 여기서 32B는 분석 범위이지 **32B SAM의 효과 검증**을 뜻하지 않습니다. :chatgpt-content-reference{index="124"} |
| **Han 외, 2026, 5월 개정 — Weight Decay Improves Language Model Plasticity** | 더 큰 사전학습 weight decay가 후속 적응성을 높이며, 기본 모델 성능과 후속 성능 사이의 역전을 보고 | SAM 외에도 간단한 사전학습 설정으로 적응성이 달라집니다. 향후에는 **SAM·학습률·weight decay의 공동 비교**가 필요합니다. :chatgpt-content-reference{index="125"} |
| **Kotha·Liang, 2026 — Replaying pre-training data improves fine-tuning** | 일반 사전학습 데이터를 추가 학습에 섞으면 기존 능력 보존뿐 아니라 목표 작업의 데이터 효율도 개선될 수 있음을 보고 | 대상 논문은 **초기 가중치 상태**, 이 연구는 **추가 학습 데이터 구성**에 개입합니다. 두 방법의 결합은 유망하지만 실제 상호작용은 미검증입니다. :chatgpt-content-reference{index="126"} |
| **Rofin 외, 2026, 8월 개정 — (How) Learning Rates Regulate Catastrophic Overtraining** | 같은 SFT 손실에서도 추가 학습률에 따라 다른 모델이 형성되며, 사전학습 학습률 감소가 곡률과 망각에 연결된다고 분석 | 사전학습 설정만이 아니라 **사전학습과 추가 학습의 학습률을 함께** 분석해야 한다는 근거입니다. :chatgpt-content-reference{index="127"} |
| **Deng·Pang, 2026년 8월 — On the Implicit Flatness Bias of Sharpness-Aware Minimization: A Linear Stability Analysis with Quantitative Hyperparameter Bounds** | 특정 국소 근사·잡음 정렬 가정 아래에서 SAM 반경, 학습률, 배치 크기와 최대 곡률의 관계를 분석하고, Taylor 오차에 따라 반경을 조절하는 방법을 제안 | 고정 $\rho=0.05$보다 **반경·배치·학습률의 공동 설계**를 연구할 이론적 동기입니다. 실험은 CIFAR-100 중심이므로 언어모델 망각 보장으로 읽으면 안 됩니다. :chatgpt-content-reference{index="128"} |

> **암묵적 편향:** 명시적인 목표함수에 적혀 있지 않아도, 최적화 알고리즘이 특정 종류의 해를 선호하는 현상.  
> **재매개변수화:** 모델이 표현하는 함수는 유지하면서 가중치를 표현하는 방식을 바꾸는 것. 이때 측정된 평탄함이 달라질 수 있습니다.  
> **Plasticity, 가소성:** 새로운 데이터나 작업을 학습해 성능을 개선할 수 있는 능력.  
> **Replay:** 이전 학습 분포의 데이터를 다시 섞어 학습하는 방식.

### 9.2 최신 이론이 제안하는 구체적인 고려 사항

Deng·Pang의 2026년 연구는 특정 가정 아래에서 선형적으로 안정한 해가 다음 경계를 만족한다고 보고합니다.

$$
\lambda_{\max}(H)
\leq
\left(
\frac{b\Gamma}{2\rho\eta^2}
\right)^{1/3}
$$

$b$는 배치 크기, $\Gamma$는 기울기 크기의 상한, $\rho$는 SAM 반경, $\eta$는 학습률, $\lambda_{\max}(H)$는 최대 곡률입니다. 이는 **국소 선형화와 기울기 잡음 정렬 등의 가정하에서 제시된 외부 연구 결과**입니다. :chatgpt-content-reference{index="129"}

> **선형 안정성:** 해 주변에서 작은 변화가 시간이 지나며 과도하게 증폭되지 않는 성질.  
> **국소 선형화:** 해 근처의 학습 동작을 일차식으로 근사하는 분석.

**[본 해석]** 이 결과는 대상 논문에 세 가지 질문을 제기합니다. 첫째, 모델 크기와 배치 크기가 바뀌어도 같은 $\rho$가 적절한가; 둘째, 반경을 키워 평탄함을 유도하면서도 국소 근사를 유지할 수 있는가; 셋째, AdamW 기반 SAM에서도 이 관계가 실제 망각 예측에 유효한가입니다. **이 질문의 답은 대상 논문이나 해당 이론만으로 확정할 수 없습니다.**

### 9.3 앞으로의 연구에 미칠 영향

**[본 해석]** 이 논문의 영향은 새로운 “만능 망각 방지법”보다는 다음 평가 관행을 촉진하는 데 있을 가능성이 큽니다.

**기본 모델만 평가하는 관행의 수정.** 사전학습 손실과 함께 대표 추가 학습 이후의 성능 보존을 검증 지표로 삼아야 합니다. 이는 Liu 외, Springer 외, Han 외의 결과와 같은 방향입니다. :chatgpt-content-reference{index="130"}

**토큰 예산과 최적화 일정의 분리.** 장기 사전학습의 문제를 단순히 토큰 수 탓으로 돌리지 말고, 학습률 감소·weight decay·배치·반경이 만든 민감도 변화를 함께 분석해야 합니다. :chatgpt-content-reference{index="131"}

**망각 완화법의 조합 연구.** 사전학습 가중치의 안정성, 추가 학습 데이터 replay, 중요 가중치 보존 규제를 서로 다른 개입으로 분해하고 결합해야 합니다. 다만 EWC 결합 외의 조합 효과는 이 논문의 확인된 결과가 아니라 후속 연구 과제입니다. :chatgpt-content-reference{index="132"} :chatgpt-content-reference{index="133"}

---

## 10. 문서가 답하지 않는 질문

| 질문 | 원문이 답한 범위 | 남아 있는 문제 |
|---|---|---|
| **왜 1B MusicPile에서는 효과가 없는가?** | 실패를 결론에서 명시 | 도메인 차이, 데이터 구조, 업데이트 방향, 필요한 가중치 이동량 중 무엇이 원인인지 불명확. **p. 11**. :chatgpt-content-reference{index="134"} |
| **지도 추가 학습 외에도 효과가 있는가?** | 지도 추가 학습과 PTQ를 연구 | 강화학습, 선호 최적화, adapter 등은 미검증. **p. 11**. :chatgpt-content-reference{index="135"} |
| **어떤 지표로 추가 학습 강건성을 미리 예측할 수 있는가?** | 사전학습 손실만으로 부족함을 보임 | 최적의 검증 지표나 직접적인 사전학습 목적함수는 제시하지 못함. **p. 11**. :chatgpt-content-reference{index="136"} |
| **곡률 감소가 원인인가, 특징 학습 변화가 원인인가?** | 방향별 곡률과 손실 증가의 연결을 제시 | 두 요인의 독립적 인과효과를 분리하지 않음. **pp. 8–9**. :chatgpt-content-reference{index="137"} :chatgpt-content-reference{index="138"} |
| **더 큰 모델·다른 구조·다국어에서도 유효한가?** | OLMo 계열 통제 실험과 1B 중간학습 | 모델 계열과 규모를 넘어선 일반성은 미확인. **pp. 3, 8**. :chatgpt-content-reference{index="139"} :chatgpt-content-reference{index="140"} |
| **동일 계산량에서 가장 좋은 전략은 무엇인가?** | 마지막 구간 SAM의 비용 절감 가능성을 제시 | 더 많은 AdamW 토큰, 더 큰 모델, 다른 일정과의 동일 예산 비교가 필요. **p. 7**. :chatgpt-content-reference{index="141"} |
| **추가 학습을 반복하거나, 추가 학습 후 양자화하면 어떻게 되는가?** | SFT와 양자화를 각각 평가 | 긴 작업 순서와 여러 변경이 누적되는 상황의 안정성은 불명확. **p. 8**. :chatgpt-content-reference{index="142"} |
| **양자화 외의 압축에도 효과가 있는가?** | 양자화를 연구 | 가중치 가지치기 등은 향후 연구로 남김. **p. 2**. :chatgpt-content-reference{index="143"} |

> **선호 최적화:** 어떤 응답을 더 선호하는지에 관한 데이터로 모델을 조정하는 방식.  
> **Adapter:** 전체 가중치를 모두 바꾸는 대신 일부 추가 모듈 등을 학습하는 접근.  
> **가지치기:** 중요도가 낮다고 판단한 가중치나 구성요소를 제거하는 압축 방식.

---

## 11. 결론: 저자의 시사점과 추가 후속 연구 방향

### 11.1 저자들이 제시한 시사점과 연구 과제

**[저자 결론]** 사전학습은 기본 체크포인트만이 아니라 **최종적으로 추가 학습·정렬·압축을 거친 모델**을 목표로 설계해야 합니다. 학습률, 감소 일정, 최적화기, 규모 확장 규칙의 최적값도 이러한 후속 모델을 기준으로 달라질 수 있다는 주장입니다. **[p. 11]** :chatgpt-content-reference{index="144"}

저자들은 후속 방향으로 **데이터 특성에 따른 효과 차이 규명**, **강화학습·선호 최적화·adapter 등으로의 확장**, **후속 강건성을 직접 예측하는 목적함수와 검증 기준 개발**을 제시합니다. 이는 구체적인 실행 일정이 정해진 계획이라기보다 논문이 명시한 연구 과제입니다. **[p. 11]** :chatgpt-content-reference{index="145"}

### 11.2 추가 제안 — 일반화 성능을 검증하기 위한 우선순위

**첫째, 같은 계산 예산과 독립 반복으로 효과를 재검증해야 합니다.**  
전체 SAM, 후반 SAM, 높은 학습률, 짧은 감소 구간, 높은 weight decay를 동일 계산 예산에서 비교하는 실험을 제안합니다. 가능한 범위에서 여러 독립 시드를 사용하고, 설정 선택용 데이터와 최종 평가용 데이터를 분리하며, 특정 문턱값 한 곳뿐 아니라 넓은 학습–망각 절충 구간을 평가해야 합니다. 이 제안의 목적은 “가장 좋은 한 점”의 우위가 아니라 **재현 가능한 계산량당 이득**을 확인하는 것입니다.

**둘째, 곡률과 업데이트 방향을 분리하는 실험이 필요합니다.**  
SAM과 AdamW 모델의 실제 추가 학습 방향을 각각 얻은 뒤, 각 방향을 두 기본 모델에서 교차 평가하는 설계를 제안합니다. 여기에 가중치 이동 거리와 표현 특징의 변화를 함께 측정하면, 개선이 “더 평탄한 지형”에서 오는지 “덜 해로운 방향의 학습”에서 오는지 구분하는 데 도움이 됩니다. 이는 원문이 인정한 방향의 내생성 문제를 직접 다루는 실험입니다. :chatgpt-content-reference{index="146"}

> **내생성:** 설명하려는 결과와 설명 변수 자체가 같은 학습 과정에서 함께 결정되어, 한쪽만의 효과를 분리하기 어려운 상황.

**셋째, 일반화는 ‘기존 능력 보존’과 별도의 평가 축으로 검증해야 합니다.**  
최적화 설정을 고르는 데 사용하지 않은 도메인·언어·작업을 남겨 두고, 추가 학습 후 그 환경에서 평가하는 실험을 제안합니다. 평균 점수 외에도 작업별 최악의 하락, 코드 실행 성공률, 실제 답의 정확성처럼 검증 손실만으로 대체하기 어려운 결과를 포함해야 합니다. 특히 MusicPile처럼 효과가 사라지는 사례를 예외로 제외하기보다 **방법이 작동하는 조건을 밝히는 핵심 실험**으로 다뤄야 합니다.

**넷째, 실제 후속 업데이트 분포를 반영하는 평탄화가 유망합니다.**  
이는 저자가 제시한 새 방법이 아니라, 원문의 Taylor 분석에서 도출한 후속 연구 방향입니다. 업데이트 $\Delta$의 평균을 $\mu$, 공분산을 $\Sigma$라고 하면 이차 근사 아래에서 다음 관계를 얻습니다.

$$
\mathbb E[F_{\mathcal L}]
\approx
g^\top\mu
+
\frac12\mu^\top H\mu
+
\frac12\text{tr}(H\Sigma)
$$

여기서 $g=\nabla\mathcal L_{\text{PT}}(\theta_{\text{PT}})$, $H$는 Hessian, $\mu=\mathbb E[\Delta]$, $\Sigma=\mathbb E[(\Delta-\mu)(\Delta-\mu)^\top]$, $\text{tr}$는 행렬 대각 원소의 합입니다. 이 식은 **국소 이차 근사가 성립한다는 조건하에서의 수학적 도출**이지, 검증된 새로운 학습 알고리즘은 아닙니다.

> **공분산:** 업데이트가 어떤 방향으로 얼마나 변동하는지 나타내는 행렬.  
> **Trace:** 행렬의 대각 원소를 모두 더한 값.

이 관점에서는 모든 방향을 동일하게 다루기보다 **실제로 자주 발생하는 후속 변경 방향의 손실 증가**를 줄이는 목표를 고려할 수 있습니다. 다만 미래 작업을 지나치게 가정하면 범용 사전학습의 장점을 잃을 수 있으므로, 알려진 대리 작업과 완전히 새로운 평가 작업을 분리해야 합니다.

### 최종 평가

이 논문의 가장 설득력 있는 기여는 **“사전학습 손실이 조금 더 낮은 모델”과 “후속 변경을 더 잘 견디는 모델”이 서로 다를 수 있음을 여러 통제 실험과 1B 중간학습에서 연결해 보인 점**입니다. 반면 **보편적 OOD 일반화 향상, 모든 도메인에서의 망각 감소, 곡률 감소만으로 설명되는 인과기제, 동일 계산량에서의 최적성**은 아직 입증되지 않았습니다.

따라서 연구자가 취할 가장 생산적인 방향은 SAM을 무조건 채택하는 것이 아니라, **사전학습 품질·새 작업 학습 능력·기존 능력 보존·새로운 분포의 일반화·총 계산비용을 함께 최적화하는 실험 체계**를 만드는 것입니다. 이는 원문의 시사점을 유지하면서도, 보고된 증거를 넘어서는 주장을 피하는 결론입니다.
