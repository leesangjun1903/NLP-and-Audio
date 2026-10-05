# Diffusion Drafts, AR Verifies: Accelerating Document OCR with Self-Speculative Decoding

## 1. Executive summary

이 연구의 목적은 문서 이미지를 텍스트·표·수식으로 변환할 때 발생하는 **순차적 토큰 생성 병목을 줄이면서 인식 품질을 유지하는 것**이다〔pp.1–2〕. :chatgpt-content-reference{index="1"}  
> **용어 풀이:** 토큰은 모델이 읽고 생성하는 단위로, 단어 전체뿐 아니라 단어 조각, 숫자, HTML 태그의 일부 등이 될 수 있다.

제안 모델 **GravityOCR**는 하나의 모델이 블록 확산으로 여러 토큰의 초안을 동시에 만들고, 자기회귀 방식으로 검증하는 **자기 추측 디코딩**을 사용한다〔pp.4–6, Figures 3–4〕. :chatgpt-content-reference{index="2"} :chatgpt-content-reference{index="3"}  
> **용어 풀이:** 자기회귀, 즉 AR은 앞서 확정한 토큰을 보고 다음 토큰을 예측하는 방식이고, 블록 확산은 가려진 여러 토큰을 한 묶음 안에서 병렬로 복원하는 방식이다.

저자 보고에서 추가 학습 이후 종합평가 점수는 **94.92에서 95.16으로 상승**했지만, 원래 GLM-OCR의 **95.48보다는 0.32점 낮았다**〔p.9, Table 1; p.13, Table 6〕. :chatgpt-content-reference{index="4"} :chatgpt-content-reference{index="5"}  
SGLang 구현에서는 모델 호출당 평균 **9.7개 토큰**을 확정했고, 같은 학습 후 모델의 AR 방식 대비 **영역 이미지의 토큰 생성은 3.94배, 전체 페이지 처리는 1.32배** 빨라졌다〔pp.9–11, Tables 1·3〕. :chatgpt-content-reference{index="6"} :chatgpt-content-reference{index="7"}  
다만 보존 대상은 **원래 GLM-OCR이 아니라 동일한 학습 후 모델의 AR 출력**이며, 실제 저정밀 연산 실험에서 두 방식의 출력 문자열이 완전히 같았던 비율은 **96.6%**였다〔p.21, Appendix B.3〕. :chatgpt-content-reference{index="8"}  
제 해석으로 이 연구의 핵심은 확산 모델의 오류를 없애는 것이 아니라, **초안 오류가 최종 출력 오류 대신 수락 길이 감소와 속도 손실로 나타나게 하는 설계**에 있다〔p.6, Figure 4; p.21, Appendix B〕. :chatgpt-content-reference{index="9"} :chatgpt-content-reference{index="10"}  
새로운 언어·문서 유형·촬영 환경으로의 일반화는 유망한 후속 연구 주제지만, 현재 결과만으로 **일반화 성능 자체가 향상되었다고 결론 내릴 근거는 충분하지 않다**〔p.8, 학습·평가 구성; p.25, 강화학습 데이터 구성〕. :chatgpt-content-reference{index="11"} :chatgpt-content-reference{index="12"}

---

## 2. 핵심 주장과 근거

이 논문에서 **‘검증’은 정답과 대조하는 과정이 아니라, 초안 토큰이 AR 경로의 예측과 일치하는지 확인하는 과정**이다. 따라서 “검증을 통과했다”와 “이미지를 정확하게 읽었다”는 같은 의미가 아니다〔p.6, Figure 4〕. :chatgpt-content-reference{index="13"}

| 핵심 주장 | 저자가 제시한 근거와 위치 | 해석 및 근거의 범위 |
|---|---|---|
| 확산 예측을 병렬로 바로 확정하면 누락·중복이 발생할 수 있다. | `vehicles` 누락과 `to the the` 중복 사례. **p.3, Figure 2**. :chatgpt-content-reference{index="14"} | 실패 메커니즘을 보여주는 사례다. 모든 확산 모델에서 발생하는 빈도를 측정한 결과는 아니다. |
| 별도 초안 모델 없이 병렬 초안과 AR 검증을 결합할 수 있다. | 비전 인코더·언어 디코더·출력 헤드를 공유하며, 추가 구조는 마스크 토큰 임베딩이다. **pp.4–5, Figure 3**. :chatgpt-content-reference{index="15"} :chatgpt-content-reference{index="16"} | 모델 관리 구조는 단순해진다. 그러나 추가 학습이나 임시 상태 저장 비용까지 없어지는 것은 아니다. |
| 자기 추측 디코딩은 같은 모델의 AR 출력을 보존한다. | 정확한 산술에서의 논증과 실제 출력 일치율 96.6%. **p.21, Appendix B.3**. :chatgpt-content-reference{index="17"} | 이론적 보존과 실제 구현의 완전 일치는 구분해야 한다. 원래 GLM-OCR의 출력 보존을 뜻하지도 않는다. |
| 직접 확산 확정보다 품질–병렬성 절충이 유리하다. | 약 10 토큰/호출에서 직접 확산 점수 **92.53**, 자기 추측 **95.16**. **pp.11–13, Figure 5·Table 6**. :chatgpt-content-reference{index="18"} :chatgpt-content-reference{index="19"} | 동일 모델의 디코딩 방식 비교라는 장점이 있다. 다만 두 방식의 호출 수 집계 규칙이 완전히 같지는 않다. |
| 실제 서비스 구현에서도 속도가 향상된다. | SGLang에서 생성 단계 **3.94배**, 영역 요청 전체 **1.74배**, 페이지 전체 **1.32배**. **pp.9–11, Tables 1·3**. :chatgpt-content-reference{index="20"} :chatgpt-content-reference{index="21"} | 서로 다른 측정 범위의 숫자다. “문서 처리가 3.94배 빨라졌다”로 요약하면 부정확하다. |
| 공동 학습의 AR 손실은 검증기의 품질 유지에 중요하다. | 10,000단계 모델에서 AR 손실 포함 시 **95.02**, 제외 시 **93.64**. **p.13, Table 5**. :chatgpt-content-reference{index="22"} | 확산 학습 중 AR 능력 보존을 지지한다. AR만 같은 데이터로 추가 학습한 모델보다 우수하다는 증거는 아니다. |
| AR 경로의 추가 강화학습은 초안 효율을 유지하면서 품질을 개선한다. | 점수 **94.92→95.16**, 토큰/호출 **9.61→9.68**. **p.13, Table 6**. :chatgpt-content-reference{index="23"} | 관찰된 개선은 있지만, 반복 학습의 분산이나 신뢰구간이 없어 통계적 유의성은 판단할 수 없다. |
| 별도 표 인식 벤치마크에서도 성능이 개선된다. | PubTabNet의 표 구조·내용 유사도 **0.803→0.871**. **p.10, Table 2**. :chatgpt-content-reference{index="24"} | 검증 분할에서의 개선이다. PubTabNet 학습 분할도 사용했으므로 완전히 새로운 도메인에 대한 증거는 아니다. |

---

## 3. 연구 목적, 모델 구조, 제안 방법

### 3.1. 해결하려는 문제: 이미지에 답이 있어도 출력 토큰 간 의존성은 남는다

**저자 보고.** 일반적인 자유로운 문장 생성과 달리 OCR의 답은 입력 이미지에 강하게 제약된다. 이 때문에 여러 위치를 동시에 복원하는 확산 방식이 적합해 보인다. 그러나 문자의 읽기 순서, 표의 태그 짝, 수식의 괄호·구분자처럼 **출력 토큰 사이의 관계**는 여전히 중요하다〔pp.1–2〕. :chatgpt-content-reference{index="25"} :chatgpt-content-reference{index="26"}

같은 확산 단계에서 예측되는 위치들은 주변의 가려진 문맥을 함께 보지만, **그 단계에서 다른 위치에 최종적으로 선택될 토큰 값까지 조건으로 사용할 수는 없다**. 각 위치의 예측 확률이 높아도, 한꺼번에 확정한 문자열은 불일치할 수 있다. Figure 2의 누락·중복 사례가 이를 설명한다〔p.3〕. :chatgpt-content-reference{index="27"}

**해석.** 논문의 설계상 전환점은 “확산 예측을 더 확신하게 만들자”가 아니라 **“병렬로 제안하는 일과 최종 출력으로 확정하는 일을 분리하자”**는 것이다.

### 3.2. 모델 구조: 새 OCR 전체 시스템이 아니라 영역 인식기의 변환

전체 처리 흐름은 다음과 같다.

**페이지 → PP-DocLayout-V3로 영역 검출 → 잘라낸 영역의 OCR → 결과 병합·후처리**

GravityOCR는 기존 GLM-OCR의 레이아웃 검출기와 결과 조립 과정을 유지하고, **영역별 OCR 모델만 AR–확산 공동 모델로 변환**한다〔p.7, §4.1〕. :chatgpt-content-reference{index="28"}

비전 인코더, 언어 디코더, 언어 모델 출력 헤드를 두 경로가 공유하며, 별도 초안 네트워크나 보조 예측 헤드를 추가하지 않는다. 외부의 **「GLM-OCR Technical Report」**에 따르면 기반 모델은 약 0.9B 파라미터로, 0.4B 비전 인코더와 0.5B 언어 디코더를 결합한다. 이 규모 정보는 GravityOCR 본문의 새 구조 제안이 아니라 기반 모델 보고서의 설명이다. :chatgpt-content-reference{index="29"} :chatgpt-content-reference{index="30"}

> **용어 풀이:** 비전 인코더는 이미지에서 특징을 추출하고, 언어 디코더는 이를 조건으로 토큰을 생성한다. 출력 헤드는 내부 표현을 각 후보 토큰의 점수로 바꾸며, 임베딩은 토큰을 모델이 처리할 수 있는 수치 벡터로 나타낸 것이다.

**해석.** 따라서 이 연구가 직접 개선하는 부분은 주로 **영역 내부의 인식·생성 과정**이다. 새로운 페이지 배치, 영역 검출 실패, 영역 간 관계 같은 문제까지 자동으로 해결하는 구조는 아니다.

### 3.3. AR 학습과 블록 확산 학습

#### A. 자기회귀 확률과 손실: 원문 식 (1)–(2)

```math
p_{\theta}^{\text{AR}}(\mathbf{y}\mid I,c)
=
\prod_{i=1}^{N}
p_{\theta}^{\text{AR}}
\left(y_i\mid I,c,\mathbf{y}_{ < i}\right)
```

```math
\mathcal{L}_{\text{AR}}
=
-\sum_{i=1}^{N}
\log p_{\theta}^{\text{AR}}
\left(y_i\mid I,c,\mathbf{y}_{ < i}\right)
```

**기호 설명.** $I$는 입력 이미지, $c$는 작업 지시문, $\mathbf{y}=(y_1,\ldots,y_N)$은 정답 토큰열, $N$은 길이, $i$는 위치, $\mathbf{y}\_{ < i}$는 앞선 토큰들이다. $\theta$는 모델 파라미터, $p_{\theta}^{\text{AR}}$는 AR 경로가 부여하는 조건부 확률, $\mathcal{L}_{\text{AR}}$는 정답 토큰 확률이 낮을수록 커지는 손실이다〔p.3〕. :chatgpt-content-reference{index="31"}

> **용어 풀이:** 음의 로그우도 손실은 정답에 높은 확률을 부여하도록 학습시키는 목적함수다. 여기서 ‘인과적’이라는 표현은 현실의 인과관계를 추론한다는 뜻이 아니라, 뒤의 토큰을 보지 않고 앞의 토큰만 사용한다는 뜻이다.

#### B. 블록 확산 손실: 원문 식 (3)

```math
\mathcal{L}_{\text{diff}}
=
-\mathbb{E}_{b,t,\mathcal{M}_{t}^{(b)}}
\left[
w(t)
\sum_{j\in\mathcal{M}_{t}^{(b)}}
\log p_{\theta}^{\text{diff}}
\left(
y_j^{(b)}
\mid
I,c,\mathbf{y}^{( < b)},\mathbf{y}_{t}^{(b)}
\right)
\right]
```

**기호 설명.** $b$는 왼쪽부터 센 블록 번호, $B$는 최대 블록 크기, $j$는 블록 내부 위치다. $\mathbf{y}^{( < b)}$는 앞선 완성 블록들, $t$는 마스킹 수준, $\mathcal{M}\_{t}^{(b)}$는 가린 위치들의 집합, $\mathbf{y}\_{t}^{(b)}$는 해당 위치를 전용 마스크 토큰으로 바꾼 블록이다. $p_{\theta}^{\text{diff}}$는 확산 경로의 예측 확률, $w(t)$는 마스킹 수준별 손실 가중치이며, $\mathbb{E}$는 선택한 블록·마스킹 수준·마스크 위치에 대한 평균을 뜻한다〔p.4〕. :chatgpt-content-reference{index="32"}

현재 블록 안에서는 양방향으로 정보를 참고하지만, 블록 간에는 왼쪽에서 오른쪽으로 진행한다. 중요한 점은 **가리는 대상이 이미지가 아니라 출력 텍스트 토큰**이라는 것이다〔pp.4–5〕. :chatgpt-content-reference{index="33"} :chatgpt-content-reference{index="34"}

> **용어 풀이:** 마스킹은 일부 정답을 빈칸으로 바꾸는 것이고, 잡음 제거 학습은 그 빈칸을 복원하는 학습이다. 이 논문의 마스킹을 흐린 사진·오염된 스캔 등에 대한 이미지 증강과 동일시해서는 안 된다.

#### C. 공동 학습: 원문 식 (4)와 실제 정규화

원문은 두 손실을 다음처럼 결합한다.

```math
\mathcal{L}
=
\mathcal{L}_{\text{AR}}
+
\lambda\mathcal{L}_{\text{diff}}
```

실제 구현에서는 각 손실을 해당 감독 토큰 수로 평균한 뒤 다음처럼 정규화한다.

```math
\mathcal{L}_{\text{impl}}
=
\frac{
\overline{\mathcal{L}}_{\text{AR}}
+
\lambda\overline{\mathcal{L}}_{\text{diff}}
}{
1+\lambda
},
\qquad
\lambda=1
```

**기호 설명.** $\lambda$는 확산 손실의 상대적 가중치다. 윗줄은 토큰별 평균을 뜻하며, $\mathcal{L}_{\text{impl}}$은 실제 구현의 결합 손실이다. 따라서 기본 설정에서는 AR과 확산 목표의 가중치가 각각 0.5이고, 확산 손실의 $w(t)$는 1이다〔pp.5–6〕. :chatgpt-content-reference{index="35"} :chatgpt-content-reference{index="36"}

Figure 3에서는 **깨끗한 응답 하나와 상보적인 마스크 응답 두 개**를 한 번의 순전파로 처리한다. 첫 마스크가 가린 위치를 두 번째 마스크는 보여주고, 첫 마스크가 보여준 위치를 두 번째는 가리므로, 모든 응답 토큰이 두 확산 스트림 중 정확히 하나에서 복원 대상이 된다. 확산 블록은 앞선 깨끗한 블록을 참조할 수 있지만, 현재·미래의 깨끗한 정답 블록은 볼 수 없다〔p.5〕. :chatgpt-content-reference{index="37"}

> **용어 풀이:** 순전파는 입력을 모델에 넣어 출력을 계산하는 한 번의 과정이다. 상보적 마스킹은 두 빈칸 배치가 서로의 빈칸을 보완하도록 만드는 방식이다.

### 3.4. 자기 추측 디코딩: 왜 두 번의 호출로 여러 토큰을 확정할 수 있는가

**저자 보고.** 기본 블록 크기는 $B=32$이며, 한 라운드는 다음 두 번의 호출로 구성된다〔p.6, Figure 4; p.21, Appendix B.2〕. :chatgpt-content-reference{index="38"} :chatgpt-content-reference{index="39"}

**초안 호출.** 마지막 확정 토큰 $x_0$ 뒤에 $B$개의 마스크를 붙인다. $x_0$ 위치에서는 인과적 AR 예측 $a_0$을 얻고, 마스크 위치에서는 초안 $d_1,\ldots,d_B$를 동시에 얻는다. $a_0$은 처음부터 AR 예측이므로 바로 확정한다.

**검증 호출.** 이미 만들어진 $[a_0,d_1,\ldots,d_B]$를 인과적 어텐션으로 한 번에 처리해 AR 예측 $a_1,\ldots,a_{B+1}$을 얻는다. 초안이 입력으로 주어졌기 때문에, 모든 위치의 검증 예측을 한 호출에서 계산할 수 있다.

아래는 원문 알고리즘을 이해하기 쉽게 다시 쓴 식이다.

```math
A=
\max
\left\{
\ell\in\{0,\ldots,B\}:
d_j=a_j
\ \text{for every }1\le j\le\ell
\right\}
```

```math
\Delta\mathbf{y}
=
[a_0,d_1,\ldots,d_A,a_{A+1}]
```

**기호 설명.** $A$는 처음부터 연속해서 일치한 초안 길이, $\ell$은 그 후보 길이, $d_j$는 초안 토큰, $a_j$는 검증기의 해당 예측이다. $\Delta\mathbf{y}$는 이번 라운드에 추가하는 출력이다. 마지막의 $a_{A+1}$은 첫 불일치 위치를 바로잡는 AR 토큰이거나, 초안 전체가 맞았을 때의 추가 토큰이다〔p.21〕. :chatgpt-content-reference{index="40"}

예를 들어 첫 두 초안만 맞으면 $d_1,d_2$를 받아들이고, 세 번째 위치에는 $a_3$을 넣는다. 뒤의 초안은 틀린 앞부분을 조건으로 검증되었을 수 있으므로 버린다. 종료 조건으로 중간에 잘리지 않는 한, 한 라운드는 **2개에서 $B+2$개 토큰**을 확정한다.

검증 과정에서 수락한 토큰의 **인과적 KV 캐시**도 함께 만든다. 거절된 토큰의 상태와 초안의 양방향 상태는 남기지 않으므로, 별도의 캐시 작성 호출이 필요 없다〔p.21〕. :chatgpt-content-reference{index="41"}

> **용어 풀이:** 접두부는 문자열의 처음부터 이어지는 부분이다. KV 캐시는 앞선 토큰에 대해 계산한 어텐션용 정보를 저장해 다음 단계에서 재사용하는 장치다.

#### 토큰/호출과 실제 속도는 다르다

원문 식 (5)는 다음과 같다.

```math
\text{TPF}
=
\frac{
\text{확정한 출력 토큰 수}
}{
\text{모델 순전파 호출 수}
}
```

GravityOCR에서는 초안 호출과 검증 호출을 **각각 한 번**으로 센다. 평균 초안 수락 길이가 17.4이므로, 종료·경계 효과를 생략한 라운드 기준 설명은 다음과 같다.

```math
\text{TPF}
\approx
\frac{17.4+2}{2}
=
9.7
```

여기서 TPF는 *tokens per forward*, 즉 호출당 확정 토큰 수이며, 분자의 2는 $a_0$과 검증기의 추가 토큰이다〔pp.8–9, 식 (5); p.12, Table 4〕. :chatgpt-content-reference{index="42"} :chatgpt-content-reference{index="43"} :chatgpt-content-reference{index="44"}

**해석.** 여러 토큰을 처리하는 호출은 한 토큰을 처리하는 호출보다 비쌀 수 있다. 따라서 **9.7 TPF가 9.7배의 실제 가속을 뜻하지 않는다.**

### 3.5. AR 출력 보존의 정확한 의미

**저자 논증.** 같은 파라미터, 정확한 산술, 일관된 최고확률 토큰 선택 아래에서는 모든 확정 토큰이 AR 경로의 예측이므로, 자기 추측 결과는 단독 AR의 그리디 출력과 같다〔p.21, Appendix B.3〕. :chatgpt-content-reference{index="45"}

> **용어 풀이:** 그리디 디코딩은 매 단계에서 확률이 가장 높은 토큰 하나를 선택하는 방식이다.

그러나 다음 세 가지는 구별해야 한다.

| 구별해야 할 대상 | 이 논문에서의 의미 |
|---|---|
| **같은 학습 후 모델의 AR 출력** | 이론적 보존 대상이다. |
| **원래 GLM-OCR의 출력** | 공동 학습으로 파라미터가 바뀌므로 보존 대상이 아니다. |
| **정답 OCR 문자열** | 검증기가 정답을 확인하는 것이 아니므로 정확성을 보장하지 않는다. |

실제 SGLang·bf16 실험에서는 영어 영역 8,922개 중 **96.6%에서 문자열이 같았다**. 저자들은 나머지 차이를 서로 다른 연산 커널에서 근접한 후보 토큰의 순위가 뒤집히는 현상과 일관된 것으로 설명한다. 첫 불일치 305개 위치를 fp32로 다시 계산했을 때는 228개가 AR 토큰, 77개가 자기 추측 토큰과 일치했다. **전체 시퀀스에 대해 fp32 완전 일치를 입증한 결과는 아니다**〔p.21〕. :chatgpt-content-reference{index="46"}

> **용어 풀이:** bf16과 fp32는 수를 저장·계산하는 정밀도 형식이다. bf16은 더 적은 비트를 사용하므로 계산 효율이 좋지만, 점수가 매우 비슷한 후보의 순위가 fp32와 달라질 수 있다.

또한 실제 보존 검증은 그리디 설정에 대한 것이다. 일반적인 추측 디코딩의 확률분포 보존 이론을 이 구현의 모든 무작위 샘플링 설정에 그대로 확대해서는 안 된다〔p.4, 예비 지식; p.21, 검증 설정〕. :chatgpt-content-reference{index="47"} :chatgpt-content-reference{index="48"}

### 3.6. AR 경로의 강화학습: 확산 경로의 확률 계산 문제를 ‘해결’하지 않고 ‘우회’한다

**저자 보고.** 토큰별 정답 확률만 높이는 학습은 완성된 표의 구조나 수식의 형식적 완전성을 직접 최적화하지 않는다. 이에 저자들은 **GRPO를 AR 경로에 적용**하고, 공유 파라미터를 통해 초안 경로도 함께 바뀌도록 한다〔p.7, §3.3〕. :chatgpt-content-reference{index="49"}

> **용어 풀이:** 강화학습은 생성 결과에 보상을 주고 높은 보상을 받는 출력을 더 자주 만들도록 학습하는 방식이다. GRPO는 같은 입력에서 여러 답을 생성한 뒤, 그 그룹 안에서 상대적으로 좋은 답을 기준으로 모델을 갱신하는 방법이다.

다단계 확산에서는 완성된 출력의 확률을 구하려면 여러 마스크 해제 경로를 고려해야 한다. GravityOCR는 최종 출력을 결정하는 AR 경로의 명시적인 확률 분해를 사용하므로, **확산 경로 전체의 확률을 추정하지 않고도 학습할 수 있다**. 이는 확산 확률 추정 문제 자체를 푼 결과와는 다르다〔p.7; p.15〕. :chatgpt-content-reference{index="50"} :chatgpt-content-reference{index="51"}

#### GRPO의 설명용 수식

논문은 GRPO의 전체 구현 목적함수를 수식으로 제시하지 않는다. 아래는 원래 GRPO의 핵심을 설명하면서 부록 E의 비대칭 제한 범위를 반영한 **요약식**이며, 손실 정규화와 실행 엔진 간 확률 보정까지 포함한 구현 전체의 전사는 아니다. :chatgpt-content-reference{index="52"} :chatgpt-content-reference{index="53"}

$$
\mu_R=\frac{1}{G}\sum_{k=1}^{G}R_k,
\qquad
\widehat{A}_k=\frac{R_k-\mu_R}{\sigma_R}
$$

```math
u_{k,i}(\theta)
=
\frac{
p_{\theta}^{\text{AR}}
(y_{k,i}\mid I,c,\mathbf{y}_{k, < i})
}{
p_{\theta_{\text{old}}}^{\text{AR}}
(y_{k,i}\mid I,c,\mathbf{y}_{k, < i})
}
```

```math
\ell_{k,i}^{\text{policy}}
=
-\min
\left[
u_{k,i}\widehat{A}_k,\,
\text{clip}
\left(u_{k,i},1-\epsilon_{\text{low}},1+\epsilon_{\text{high}}\right)
\widehat{A}_k
\right]
```

**기호 설명.** $G$는 같은 입력에 대한 출력 개수, $k$는 출력 번호, $R_k$는 보상, $\mu_R$와 $\sigma_R$는 그룹 보상의 평균과 표준편차다. $\widehat{A}\_k$는 상대적 우수성을 나타내며 첫 식은 $\sigma_R>0$인 경우를 나타낸다. $i$는 토큰 위치, $\theta_{\text{old}}$는 비교 기준인 이전 정책 파라미터, $u_{k,i}$는 토큰 확률의 비율이다. $\text{clip}$은 값을 지정 범위 안으로 제한하고, $\ell_{k,i}^{\text{policy}}$는 정책 갱신 항이다. 논문 설정은 $\epsilon_{\text{low}}=0.2$, $\epsilon_{\text{high}}=0.28$이다〔p.24〕. :chatgpt-content-reference{index="54"}

추가로 초기 공동 학습 모델의 분포에서 너무 멀어지지 않도록 계수 $10^{-3}$의 KL 벌점을 사용한다. 한 단계에서 입력 24개, 입력당 출력 28개를 생성하며, 최종 보고 모델은 500단계 학습 결과다. 확산 전용 손실은 이 단계에 사용하지 않는다〔p.24〕. :chatgpt-content-reference{index="55"}

> **용어 풀이:** 정책은 모델의 출력 확률분포다. KL 벌점은 현재 모델과 기준 모델의 확률분포 차이가 지나치게 커지는 것을 억제하는 항이며, 완전한 출력 동일성을 보장하는 장치는 아니다.

#### 작업별 보상

핵심 보상식은 다음과 같다〔p.25, Appendix E〕.

```math
r_{\text{text}}
=
1-\text{NED}(\widehat{\mathbf{y}},\mathbf{y})
```

```math
r_{\text{table}}^{(0)}
=
\text{clip}_{[0,1]}
\left[
\left(
0.45\,\text{TEDS}_{s}
+
0.55\,s_{\text{cell}}
\right)
(1-p_{\text{row}})
(1-p_{\text{loop}})
\right]
```

```math
r_{\text{formula}}
=
\text{sim}
\left(
\text{canon}(\widehat{\mathbf{y}}),
\text{canon}(\mathbf{y})
\right)
\cdot 0.3^{v}
```

**기호 설명.** $\widehat{\mathbf{y}}$와 $\mathbf{y}$는 예측과 정답이다. $\text{NED}$는 정규화 편집 거리, $\text{TEDS}\_{s}$는 표 구조만 비교하는 유사도, $s_{\text{cell}}$은 읽기 순서로 연결한 셀 내용의 편집 유사도다. $p_{\text{row}}$는 행 수 오류, $p_{\text{loop}}$는 과도한 반복 행에 대한 벌점이다. $\text{sim}$은 정규화 편집 유사도, $\text{canon}$은 LaTeX 표기를 정규화하는 함수, $v$는 정답에는 없지만 예측에서 실패한 형식 검사 개수다. $r_{\text{table}}^{(0)}$은 추가 보정 전 표 보상이다. :chatgpt-content-reference{index="56"}

> **용어 풀이:** 편집 거리는 한 문자열을 다른 문자열로 바꾸는 데 필요한 삽입·삭제·치환 횟수에 기반한다. TEDS는 표를 트리 구조로 나타내 비교하는 지표이며, 전체 TEDS와 구조만 보는 TEDS-struct는 서로 다르다.

표 보상에는 행·열 수가 맞을 때의 추가 보너스가 있고, 닫히지 않은 `<table>`은 0점을 받는다. 모든 작업 보상에는 지나친 반복 출력을 억제하는 보정도 적용한다. 특히 **수식 보상은 CDM 자체가 아니라 정규화한 LaTeX 문자열의 유사도**라는 점이 중요하다〔p.25〕. :chatgpt-content-reference{index="57"}

---

## 4. 실험 결과: 무엇이 개선되었고, 무엇이 유지되지 않았는가

### 4.1. 학습·평가 구성

공동 학습 데이터는 12.3M 영역 풀에서 구성한 **10.8M 예제**이며, 텍스트·표·수식 비율은 **60/20/20**이다. 대부분 영어이고, 정답은 주로 원래 GLM-OCR의 전사 결과를 사용한다. 표 스트림의 약 39%는 원 데이터셋의 셀 주석을 사용한다〔p.8〕. :chatgpt-content-reference{index="58"}

공동 학습은 16개 H100에서 40,000단계 수행했다. 약 **26B 토큰**이라는 학습량에는 이미지 토큰과 세 응답 스트림이 모두 포함되므로, 이를 26B개의 서로 다른 텍스트 학습 토큰으로 해석하면 안 된다〔p.7; p.20, Table 7〕. :chatgpt-content-reference{index="59"} :chatgpt-content-reference{index="60"}

강화학습 입력 풀은 7,723개이며 **표 92%, 텍스트 6%, 수식 2%**로 훨씬 더 표 중심이다. 후보에서 보상 분산이 0인 입력을 제외하고 어렵고 긴 표 쪽으로 선택했다〔p.25〕. :chatgpt-content-reference{index="61"}

**해석.** 최종 모델을 “균형 잡힌 세 작업에 동일하게 강화학습한 모델”로 설명해서는 안 된다. 표 성능 향상에는 구조 자체뿐 아니라 **표 중심 데이터 선택과 고품질 표 주석**이 기여했을 가능성이 있다.

### 4.2. 품질 지표

OmniDocBench v1.6의 품질 평가는 1,651페이지 전체에서 수행된다. 논문의 **Overall은 텍스트·표·수식의 세 인식 축을 종합하며, 읽기 순서 지표 Order는 포함하지 않는다**〔p.8; p.9, Table 1 설명〕. :chatgpt-content-reference{index="62"} :chatgpt-content-reference{index="63"}

> **용어 풀이:** Overall은 여러 지표를 합친 종합점수이지, 전체 문서가 완벽하게 맞은 비율이 아니다. CDM은 수식을 문자 검출·대응 관점에서 평가하는 지표다.

| 평가 항목 | 원래 GLM-OCR | GravityOCR | 해석 |
|---|---:|---:|---|
| OmniDocBench Overall | 95.48 | 95.16 | **0.32점 하락**했다. |
| OmniDocBench 표 TEDS | 0.934 | 0.928 | 표 품질이 소폭 하락했다. |
| OmniDocBench 수식 CDM | 0.970 | 0.967 | 수식 품질이 소폭 하락했다. |
| PubTabNet TEDS | 0.803 | 0.871 | 해당 검증 분할에서 **0.068 상승**했다. |
| PubTabNet TEDS-struct | 0.858 | 0.916 | 표 구조 점수가 **0.058 상승**했다. |
| UniMER CDM | 0.963 | 0.962 | 거의 비슷하지만 수치상 0.001 낮다. |

위 수치는 **p.9, Table 1 및 p.10, Table 2**의 저자 측정치다. :chatgpt-content-reference{index="64"} :chatgpt-content-reference{index="65"}

논문의 “원래 점수의 99.7% 유지”는 **95.16/95.48이라는 점수 비율**이다. “문자를 99.7% 정확하게 인식한다”거나 “99.7%의 문서가 완벽하다”는 의미가 아니다〔p.2〕. :chatgpt-content-reference{index="66"}

또한 GravityOCR는 Table 1에서 가장 빠르지만 품질 1위는 아니다. Overall은 MinerU2.5-Pro **95.57**, HunyuanOCR-1.5 **95.52**, 원래 GLM-OCR **95.48**보다 낮다〔p.9〕. :chatgpt-content-reference{index="67"}

### 4.3. 세 가지 속도 수치는 반드시 분리해야 한다

| 측정 범위 | 같은 GravityOCR의 AR | 자기 추측 | 저자 보고 가속 | 포함·제외 범위 |
|---|---:|---:|---:|---|
| 영역의 **토큰 생성만** | 777 tok/s | 3,057 tok/s | **3.94배** | 이미지 인코딩과 프롬프트 사전 처리를 제외한다. |
| 영역 **요청 전체** | 486 tok/s | 844 tok/s | **1.74배** | 이미지 인코딩·사전 처리·생성을 포함하지만 레이아웃 검출은 제외한다. |
| **페이지 전체** | 0.554 pages/s | 0.730 pages/s | **1.32배** | 레이아웃·영역 처리·결과 조립을 포함한다. |

출처는 **p.11, Table 3; p.9, Table 1; p.22, Appendix C**다. :chatgpt-content-reference{index="68"} :chatgpt-content-reference{index="69"} :chatgpt-content-reference{index="70"}

> **용어 풀이:** tok/s와 pages/s는 각각 초당 토큰 수와 페이지 수다. 사전 처리, 즉 prefill은 생성에 앞서 이미지·프롬프트의 문맥 상태를 계산하는 과정이다.

**추가 계산.** 원래 GLM-OCR의 페이지 속도 0.571 pages/s를 기준으로 하면 가속은 약 **1.28배**다. 논문의 1.32배는 원래 모델이 아니라 **같은 GravityOCR 체크포인트의 AR 경로**를 기준으로 한 값이다〔p.9, Table 1〕. :chatgpt-content-reference{index="71"}

표 영역에서는 3.36배, 텍스트 영역에서는 1.54배의 영역 요청 가속을 보고하지만, 표의 평균 출력 길이는 872토큰이고 텍스트는 73토큰이다. 부록도 내용 유형별 차이의 상당 부분이 길이 효과와 겹친다고 설명한다. 따라서 “표 문법이 규칙적이어서 빨라진다”는 설명만으로 충분하지 않다〔p.12, Table 4; p.23, Table 8〕. :chatgpt-content-reference{index="72"} :chatgpt-content-reference{index="73"}

---

## 5. 가장 중요한 그림들의 해석

가장 중요한 그림은 **Figure 3의 학습 구조, Figure 4의 검증 절차, Figure 5의 품질–병렬성 관계, Figure 6의 동시 처리 한계**다. Figures 2·7은 각각 문제의 발생 원리와 초안 작성 전략을 보완한다.

| 그림 | 그림이 직접 보여주는 내용 | 해석과 주의점 |
|---|---|---|
| **Figure 2, p.3** | 높은 신뢰도로 병렬 확정한 토큰들이 단어 누락과 중복을 만든다. :chatgpt-content-reference{index="74"} | 개별 토큰의 높은 확률이 완성 문자열의 일관성을 보장하지 않는다는 사례다. 다만 오류 빈도에 대한 통계는 아니다. |
| **Figure 3, p.5** | 깨끗한 AR 응답과 상보적인 두 확산 응답이 모델을 공유한다. 확산 블록은 현재 정답 블록을 볼 수 없다. :chatgpt-content-reference{index="75"} | **세 모델을 학습하는 그림이 아니라, 한 모델에 세 학습 문맥을 제공하는 그림**이다. AR 능력 보존과 병렬 복원을 동시에 학습한다. |
| **Figure 4, p.6** | 한 번의 초안 호출 뒤 한 번의 인과적 검증으로 연속 일치 부분과 다음 AR 토큰을 확정한다. :chatgpt-content-reference{index="76"} | 논문의 핵심이다. 검증은 ‘이미지의 정답 확인’이 아니라 ‘AR 예측과의 일치 확인’이며, 캐시 처리도 가속의 일부다. |
| **Figure 5, p.11** | 직접 확산은 더 많은 토큰을 한 번에 확정할수록 품질이 낮아지는 경향을 보이고, 자기 추측은 높은 품질을 유지한다. :chatgpt-content-reference{index="77"} :chatgpt-content-reference{index="78"} | x축은 실제 시간 속도가 아닌 TPF다. 직접 확산에서는 캐시 작성 호출을 제외하므로 계산량이 완전히 정렬된 비교는 아니다. |
| **Figure 6, p.12** | 동시 처리 수가 늘면 가속이 **1.85배에서 배치 64의 1.13배**로 줄어든다. :chatgpt-content-reference{index="79"} | 저동시성에서 특히 유리하다. 이 그림은 강화학습 전, 검출기로 자른 영역을 사용하므로 최종 모델·정답 영역을 쓴 Table 3의 1.74배와 직접 맞춰서는 안 된다. |
| **Figure 7, p.23** | 초안을 여러 번 정제하면 수락 길이는 늘지만, 호출 비용 때문에 TPF는 낮아진다. :chatgpt-content-reference{index="80"} | “더 정확한 초안”과 “더 빠른 시스템”은 같은 목표가 아니다. 400개 영어 영역에서의 결과이므로 모든 하드웨어·블록 크기에 대한 단일 단계 최적성을 입증하지는 않는다. |

**Figure 1〔p.2〕**의 “가장 빠르면서 상위권 품질”도 **측정한 시스템과 조건 안에서의 주장**으로 읽어야 한다. 품질과 속도는 각각 전체 평가 집합과 영어 속도 평가 부분집합에서 측정되므로, 동일 문서 집합의 품질–시간을 일대일로 대응시킨 그래프는 아니다. :chatgpt-content-reference{index="81"} :chatgpt-content-reference{index="82"} :chatgpt-content-reference{index="83"}

---

## 6. 통계적으로 취약한 부분과 비교 불가능한 수치

| 구분 | 확인된 문제 | 허용되는 결론과 허용되지 않는 결론 |
|---|---|---|
| **통계적 불확실성 미보고** | 주요 표는 학습 시드별 분산·신뢰구간·유의성 검정을 제시하지 않는다. 특히 강화학습 개선은 +0.24점이다. **p.13, Table 6**. :chatgpt-content-reference{index="84"} | “점수가 상승했다”는 가능하지만 “통계적으로 유의하게 개선했다”는 판단할 수 없다. 집계 점수만으로 유의확률을 계산해서는 안 된다. |
| **‘품질 유지’의 통계적 정의 부재** | 원래 모델보다 0.32점 낮지만 허용 가능한 열화 폭을 미리 정한 검정은 없다. **p.9, Table 1**. :chatgpt-content-reference{index="85"} | “수치상 가깝다”는 타당하지만, 정식으로 성능 저하가 없다고 입증한 것은 아니다. |
| **TPF 분모 불일치** | GravityOCR는 초안·검증 호출 모두, MTP·DFlash는 주모델 검증 호출만, 직접 확산은 캐시 작성 호출을 제외한다. **p.22, Appendix C**. :chatgpt-content-reference{index="86"} | **9.7 대 9.9 대 3.7을 동일 계산비용의 효율 순위로 읽을 수 없다.** |
| **실행 환경 혼재** | SGLang, vLLM, PaddleX, native Transformers 등 시스템별 실행 환경이 다르다. **p.22**. :chatgpt-content-reference{index="87"} | 실제 배포 시스템 비교로는 의미가 있지만, 알고리즘 자체의 우열만 분리하지 못한다. GLM의 MTP가 AR보다 느린 결과도 이 맥락에서 읽어야 한다. |
| **속도 평가 범위 혼재** | 생성만, 영역 전체, 페이지 전체, 강화학습 전후, 정답 영역과 검출 영역이 섞여 있다. **p.22**. :chatgpt-content-reference{index="88"} | 서로 다른 표의 가속 비율을 직접 연결하거나 평균해서는 안 된다. |
| **평가 집단 차이** | 품질은 전체 1,651페이지, 속도는 주로 영어 부분집합이며 페이지 비교는 100페이지다. **pp.8–10**. :chatgpt-content-reference{index="89"} :chatgpt-content-reference{index="90"} :chatgpt-content-reference{index="91"} | 다국어 전체에서 같은 가속이 유지된다는 근거가 아니다. |
| **내용·길이의 교란** | 표는 243개이며 평균 출력이 길고, 1,024토큰 이상 구간은 77개다. **p.12, Table 4; p.23, Table 8**. :chatgpt-content-reference{index="92"} :chatgpt-content-reference{index="93"} | 표의 구조적 규칙성과 긴 출력의 효과를 별도로 추정할 수 없다. 긴 문서 전반에 3.68배를 일반화해서는 안 된다. |
| **특정 임계값의 불안정성** | 직접 확산에서 임계값 0.95의 +2.91점은 CDM 0점 수식이 116개에서 37개로 줄어든 현상과 연결된다. **p.24**. :chatgpt-content-reference{index="94"} | 모든 작업·설정에서 균일하게 개선된 결과가 아니다. 실제로 임계값 0.7·0.9에서는 점수가 조금 하락한다. :chatgpt-content-reference{index="95"} |
| **학습 요인의 인과 분리 부족** | 공동 학습, 교사 전사, 원 주석, 표 중심 강화학습이 함께 적용된다. AR 손실 실험은 10,000단계 모델이다. **pp.8·13·25**. :chatgpt-content-reference{index="96"} :chatgpt-content-reference{index="97"} :chatgpt-content-reference{index="98"} | 표 성능 상승을 확산 학습의 일반화 효과라고 단독 귀속할 수 없다. |
| **파라미터 절약과 총비용의 구별** | 별도 초안망은 없지만 전체 모델을 추가 학습하고 세 응답 스트림을 처리한다. **p.20, Table 7**. :chatgpt-content-reference{index="99"} | “추가 초안망 없음”은 “추가 학습 비용·메모리 비용 없음”이 아니다. 총 GPU 시간·에너지 비용의 우위는 확인되지 않는다. |

서로 다른 모델의 tok/s 역시 토큰화 방식과 생성 길이에 영향을 받으므로, **모델 간 비교에서는 pages/s와 품질을 함께 보는 편이 더 해석 가능하다**. 다만 pages/s도 위의 실행 환경과 내부 병렬 처리 차이를 포함한 시스템 수준 수치다〔p.22, 측정 프로토콜에 근거한 해석〕. :chatgpt-content-reference{index="100"}

**제안하는 통계 검증.** 동일 문서의 영역들이 독립이라고 가정하지 말고, 문서 또는 페이지 단위로 두 모델의 결과를 함께 재표집하여 공식 Overall을 다시 계산하는 방식이 적절하다. 여러 독립 학습 시드와 함께, 평균 차이뿐 아니라 작업별 최악 집단·실패율의 신뢰구간을 보고해야 한다.

> **용어 풀이:** 부트스트랩은 관측 자료를 반복해서 재표집하여 결과의 불확실성을 추정하는 방법이다. 여기서는 같은 재표집 문서 집합에서 두 모델을 비교해야 문서 난이도의 영향을 함께 통제할 수 있다.

---

## 7. 2020년 이후 관련 연구와의 비교

### 7.1. 기반 연구: 생성형 OCR와 병렬 생성의 결합까지

| 연구 | 해당 연구의 기여 | GravityOCR와의 관계 |
|---|---|---|
| **PubTabNet/TEDS, ECCV 2020** | 이미지에서 표 구조와 내용을 복원하는 데이터·모델·평가 체계를 제시한다. :chatgpt-content-reference{index="101"} | GravityOCR가 사용하는 표 평가 및 구조 보상의 기반이다. 디코딩 가속 연구는 아니다. |
| **D3PM, 2021** | 이산 상태 공간에서 토큰을 손상시키고 복원하는 확산 모델링을 발전시킨다. :chatgpt-content-reference{index="102"} | 마스크 토큰 복원이라는 이론적 계보에 해당하며, 문서 OCR 검증기를 제안한 연구는 아니다. |
| **Donut, 2022 / Nougat, 2023** | 문서 이미지를 직접 구조화된 출력으로 변환하고, Nougat은 학술 문서의 마크업 복원을 다룬다. :chatgpt-content-reference{index="103"} | GravityOCR의 과제 설정과 연결된다. GravityOCR는 이 흐름에서 출력 생성의 순차 비용을 줄이는 데 집중한다. |
| **Speculative Decoding, ICML 2023** | 먼저 만든 초안을 목표 모델이 병렬 검증하여 목표 생성 과정을 가속한다. :chatgpt-content-reference{index="104"} | 초안–검증 분리라는 핵심 원리의 선행 연구다. GravityOCR는 초안을 병렬 확산으로 만들고 파라미터를 공유한다. |
| **DeepSeekMath/GRPO, 2024** | 같은 입력의 여러 출력에 대한 상대 보상으로 정책을 갱신한다. **§4.1**. :chatgpt-content-reference{index="105"} | GravityOCR가 AR 경로에 적용한 강화학습 방법의 출처다. GRPO 자체가 이 논문의 신규 기여는 아니다. |
| **Block Diffusion / Fast-dLLM v2, 2025** | 블록 간 순차성·블록 내부 병렬성을 결합하고, AR 모델을 블록 확산으로 변환하는 학습법을 발전시킨다. :chatgpt-content-reference{index="106"} | GravityOCR는 블록 구조와 상보적 마스킹 등 이 계열의 요소를 OCR에 적용한다. |

> **용어 풀이:** Donut의 ‘OCR-free’는 문자를 인식하지 않는다는 뜻이 아니라, 별도의 OCR 엔진이 만든 텍스트를 먼저 입력받는 단계를 없앤다는 뜻이다.

### 7.2. 가장 가까운 2025–2026년 연구

아래의 외부 논문 가속 수치는 **각 논문 저자의 자체 조건에서 보고한 값**이다. GravityOCR의 Table 1 재측정치와 섞어 속도 순위를 만들지 않는다.

| 연구 | 방법과 직접 보고된 결과 | GravityOCR 대비 해석 |
|---|---|---|
| **TiDAR, 2025** | 구조화된 어텐션으로 확산 초안과 AR 생성을 통합하며, 단일 순전파에 두 역할을 결합하는 설계를 제시한다. :chatgpt-content-reference{index="107"} | **확산 초안＋AR 확정이라는 일반 아이디어는 선행한다.** GravityOCR의 차별점은 OCR 적용·보상·서비스 평가에 있다. |
| **GLM-OCR, 2026** | 소형 시각–언어 모델, 레이아웃 분석 뒤 영역 인식, 다중 토큰 예측을 결합한다. **초록**. :chatgpt-content-reference{index="108"} | GravityOCR의 직접 기반이다. 따라서 구조뿐 아니라 기존 파이프라인과 사전학습 능력의 기여가 크다. |
| **Fast-dVLM, 2026** | AR 시각–언어 모델을 직접 변환하고, AR·확산 공동 학습과 자기 추측을 지원한다. 선형 방식은 초안·검증 두 호출을 사용한다. 최대 **6.18배**는 SGLang과 FP8까지 결합한 결과다. **Figure 1, §3.3–3.4**. :chatgpt-content-reference{index="109"} | **구조적으로 매우 가까운 선행 연구**다. GravityOCR를 최초의 자기 추측 시각–언어 모델이라고 평가할 수 없다. 6.18배와 3.94배는 측정 범위·정밀도가 달라 직접 비교할 수 없다. |
| **Nemotron-Labs-Diffusion, 2026** | AR·확산·자기 추측의 세 모드를 공유 모델에서 지원한다. 8B 모델은 GB200·SGLang의 SPEED-Bench에서 약 **4배 처리량**을 보고한다. **Figure 1, §6.5**. :chatgpt-content-reference{index="110"} | 공유형 세 모드의 확장 가능성을 보여주는 선행 사례다. 다른 모델 규모·작업·GPU의 수치이므로 GravityOCR의 OCR 가속과 직접 비교할 수 없다. |
| **DODO, 2026 v2** | 전역 확산의 구조 불안정을 블록 단위 OCR 확산으로 완화한다. v2의 **Figure 5**는 AR 기준 약 **5배** 처리량을 보고한다. :chatgpt-content-reference{index="111"} | 확산을 최종 생성기로 사용하는 접근이다. GravityOCR는 확산을 초안기로 제한해 AR 출력을 기준으로 삼는다. DODO의 편집 거리 평가와 GravityOCR의 Overall도 동일 지표가 아니다. |
| **MinerU-Diffusion, 2026** | OCR를 역렌더링으로 보고 블록 확산과 단계적 학습을 사용한다. **Figure 7**에서는 단어 순서를 섞은 입력에 대한 강건성을 분석한다. :chatgpt-content-reference{index="112"} | 평균 정확도 외에 **언어적 개연성보다 이미지에 충실한가**라는 평가 축을 제시한다. 이는 GravityOCR의 일반화 연구에 중요한 비교 기준이다. |
| **DFlash / HunyuanOCR-1.5, 2026** | 별도의 경량 확산 초안기가 목표 모델을 보조한다. HunyuanOCR-1.5는 초안 학습 시 목표 모델을 고정하고, 자체 vLLM 실험에서 **2.14배**를 보고한다. **§3.2, §7.1**. :chatgpt-content-reference{index="113"} | 추가 초안 파라미터가 필요하지만 목표 모델을 바꾸지 않을 수 있다. GravityOCR는 별도 초안망을 없애는 대신, 공동 학습에 따른 원래 모델의 품질 변화를 관리해야 한다. |
| **HSD, 2026** | 영역 초안·병렬 영역 검증에 페이지 수준 검증을 더하는 학습 없는 가속법이며, OmniDocBench v1.5에서 약 **2.78배**를 보고한다. :chatgpt-content-reference{index="114"} | 토큰 수준의 GravityOCR와 달리 페이지 전체 일관성을 다룬다. 두 접근의 결합은 후속 연구 후보지만, 속도 수치는 버전·대상 모델이 달라 직접 비교할 수 없다. |

> **용어 풀이:** FP8은 8비트 부동소수점 형식으로 계산 비용을 줄이는 방법이다. 역렌더링 관점의 OCR는 화면에 그려진 결과에서 그 결과를 만든 텍스트·구조를 거꾸로 복원하는 문제로 OCR를 이해한다.

### 7.3. 이 논문의 신규성과 향후 영향

**해석.** 가장 방어 가능한 신규성 평가는 다음과 같다.

**GravityOCR는 완전히 새로운 AR–확산 패러다임이라기보다, 이미 발전하던 공유형 초안–검증 모델을 문서 OCR에 맞게 학습·보상·캐시 처리·서비스 측정까지 연결한 연구다.** 특히 Fast-dVLM과 Nemotron-Labs-Diffusion은 매우 가까운 선행 구조를 제공한다. :chatgpt-content-reference{index="115"}

그럼에도 OCR에서는 토큰 하나의 누락이 셀 내용이나 수식 구조를 바꿀 수 있으므로, **“병렬 생성 능력”과 “최종 확정 권한”을 분리해 평가하는 실험 설계**가 중요하다. 이 논문의 영향은 확산을 AR의 대체물로만 평가하지 않고, **AR 출력 품질을 기준으로 한 가속 구성요소**로 평가하게 만드는 데 있다고 본다〔p.6, Figure 4; p.11, Figure 5에 근거한 해석〕. :chatgpt-content-reference{index="116"} :chatgpt-content-reference{index="117"}

---

## 8. 일반화 성능: 현재 증거와 추가 연구 방향

### 8.1. ‘인식 일반화’와 ‘가속 일반화’를 분리해야 한다

아래 식은 원문의 출력 보존 논증을 일반화 관점에서 다시 표현한 것이다.

```math
f_{\text{SS}}(I,c;\theta)
=
f_{\text{AR}}(I,c;\theta)
\quad
\not\Rightarrow
\quad
f_{\text{AR}}(I,c;\theta)
=
\mathbf{y}^{*}
```

**기호 설명.** $f_{\text{SS}}$와 $f_{\text{AR}}$는 각각 자기 추측과 AR 방식이 생성하는 출력, $\theta$는 같은 학습 후 모델, $\mathbf{y}^{*}$는 실제 정답이다. 왼쪽 등식은 정확한 산술과 그리디 조건의 알고리즘적 보존이고, 오른쪽은 정답 인식 여부다〔p.21의 논증에 근거한 재표현〕. :chatgpt-content-reference{index="118"}

따라서 두 평가 질문이 따로 필요하다.

| 일반화의 축 | 실제로 물어야 할 질문 |
|---|---|
| **인식 일반화** | 새로운 언어·폰트·문서 도메인·촬영 환경에서도 AR 검증기가 정확하게 읽는가? |
| **가속 일반화** | 새로운 입력에서도 초안이 AR 검증기와 길게 일치하여 가속이 유지되는가? |

**해석.** 낯선 입력에서 초안이 자주 틀려도, 이상적인 검증 아래에서는 그 오류가 추가적인 출력 오류로 이어지지 않을 수 있다. 그러나 **검증기 자체가 틀리면 그대로 틀린 출력을 유지**하며, 초안 수락 길이가 짧아지면 가속이 사라지거나 블록 처리 비용 때문에 느려질 가능성도 있다.

> **용어 풀이:** 분포 밖 일반화, 즉 OOD 일반화는 학습 때와 다른 언어·문서 유형·이미지 조건에서도 성능을 유지하는 능력이다. 같은 데이터셋의 검증 분할에서 잘 작동하는 것과는 구분된다.

### 8.2. 현재 실험이 지지하는 것과 지지하지 않는 것

**지지하는 관찰.** 표·수식 전용 평가에서도 기반 모델의 성능을 대체로 유지하거나 개선했고, AR 손실과 부분 마스킹이 학습에 유용했다. Table 9에서 부분 마스킹은 전부 마스킹하는 학습보다 여섯 체크포인트 모두에서 TPF가 높고, 다섯 체크포인트에서 품질도 높았다〔p.10, Table 2; p.24, Table 9〕. :chatgpt-content-reference{index="119"} :chatgpt-content-reference{index="120"}

**지지하지 않는 확대 해석.** 이것만으로 다국어·새로운 촬영 환경에 대한 강건성이 좋아졌다고 말할 수는 없다. PubTabNet과 UniMER의 학습 자료가 사용되었고, 마스킹은 이미지가 아니라 응답 토큰에 적용되었으며, 속도 측정도 주로 영어다〔pp.5·8–9〕. :chatgpt-content-reference{index="121"} :chatgpt-content-reference{index="122"} :chatgpt-content-reference{index="123"}

교사 모델의 전사를 대량 사용한 점도 중요하다. **해석상**, 이 방식은 교사의 유용한 지식을 전달할 수 있지만 교사의 오인식·언어적 편향도 전달할 수 있다. 논문의 중복 점검은 의미 있는 조치지만, 원래 GLM-OCR의 전체 사전학습 노출이나 문서 템플릿 수준의 유사성까지 독립적으로 검증한 것은 아니다〔p.8의 데이터 설명에 근거한 한계〕. :chatgpt-content-reference{index="124"}

### 8.3. 일반화 연구에서 특히 중요한 두 가지 위험

#### A. AR 검증이 시각적 충실도를 항상 높이는 것은 아니다

MinerU-Diffusion은 영어 문서 112개에서 단어 순서를 섞고 다시 렌더링한 **Semantic Shuffle** 실험을 수행했다. 저자들은 이 조건에서 확산 방식이 AR 비교 모델보다 의미 훼손에 덜 민감했다고 보고한다〔해당 논문 §4.5, Figure 7〕. :chatgpt-content-reference{index="125"}

**해석·가설.** GravityOCR의 AR 검증기가 언어적으로 그럴듯한 문장을 선호한다면, 초안이 이미지에 충실하게 낸 낯선 문자열을 거절할 가능성도 연구해야 한다. 이는 GravityOCR에서 확인된 실패가 아니라, **AR 출력 보존과 시각적 정답 보존이 다르기 때문에 필요한 검증 과제**다. 무작위 제품 코드, 인명, 희귀 용어, 비문법적 원문, 숫자열처럼 문맥 추측이 위험한 입력이 유용한 시험 대상이다.

#### B. 수식 보상은 실제 수식 구조를 완전히 대변하지 않는다

부록 E의 수식 정규화는 중괄호를 제거한다〔p.25〕. :chatgpt-content-reference{index="126"}

**해석상 반례.** 원시 LaTeX `x^{12}`와 `x^12`는 지수의 묶임이 다르지만, 중괄호를 지우면 같은 문자열이 된다. 따라서 문자열 보상이 높다고 해서 수식의 렌더링이나 구조가 동일하다고 보장할 수 없다. 이것은 논문이 보고한 실제 오류 사례가 아니라 **보상 정의에서 도출되는 잠재적 사각지대**다.

후속 연구에서는 문자열 유사도, 렌더링 기반 CDM, 수식 구문 구조를 함께 확인하고, 실제 평가 점수와 보상 사이의 불일치 사례를 공개하는 것이 필요하다.

### 8.4. 제안하는 후속 연구: 우선순위와 검증 설계

| 우선순위 | 제안 | 필요한 검증 |
|---|---|---|
| **1. 진정한 분포 밖 평가** | 한국어·저자원 언어·새 폰트·새 문서 제공처·촬영 왜곡·노후 스캔을 별도 축으로 구성한다. | 원문서·템플릿·제공처 단위로 학습과 시험을 분리하고, 언어별 품질과 수락 길이를 함께 측정한다. |
| **2. 인과적 학습 비교** | 같은 초기 모델·데이터·주석 비율·학습 비용 아래에서 AR만 학습, AR만 학습＋강화학습, 공동 학습, 공동 학습＋강화학습을 비교한다. | 확산 목표가 일반화에 기여하는지, 단순한 추가 데이터·정답 품질의 효과인지 분리한다. |
| **3. 초안–검증기의 접두부 일치 최적화** | 모든 위치의 평균 정확도보다 처음부터 연속해서 맞는 길이를 직접 고려한다. | 전체 수락 길이 분포와 첫 거절 위치를 평가하되 AR 품질 저하도 함께 확인한다. |
| **4. 입력·부하별 블록 크기 선택** | 출력 유형, 예상 길이, 동시 처리 수에 따라 블록 크기를 바꾼다. | 현재의 $B=32$ 밖에서도 학습·검증하고, TPF뿐 아니라 실제 요청 지연시간을 비교한다. |
| **5. 품질 보존의 구현 검증** | bf16·fp32·다른 저정밀 형식 및 연산 커널에서 전체 출력 차이를 평가한다. | 같은 종합점수만 보지 말고 숫자·표 셀·수식의 의미를 바꾸는 차이를 분류한다. |
| **6. 페이지 수준 병목과 비용** | 영역 인식뿐 아니라 레이아웃·시각 인코딩·결과 조립을 함께 최적화하고, HSD식 전역 검증과의 결합을 검토한다. | 전체 페이지의 평균·상위 지연시간, 최대 메모리, 에너지, 학습비 회수 조건을 보고한다. |

#### 왜 ‘처음부터 연속해서 맞는 길이’가 중요한가?

아래는 원문의 접두부 수락 규칙에서 도출되는 **분석식**이며, 논문이 제안한 별도의 학습 손실은 아니다.

```math
\mathbb{E}[A]
=
\sum_{j=1}^{B}
\Pr(d_1=a_1,\ldots,d_j=a_j)
```

**기호 설명.** $A$는 수락 길이, $B$는 초안 길이, $d_j$와 $a_j$는 초안과 검증 토큰이다. 확률과 기대값은 입력·디코딩 라운드의 분포에 대해 취한다. 이 식에는 위치별 오류가 독립이라는 가정이 필요하지 않다.

**해석.** 뒤쪽 토큰을 많이 맞혀도 첫 토큰이 틀리면 그 뒤는 수락되지 않는다. 따라서 초안의 평균 토큰 정확도만 높이는 목적보다 **초기 위치의 일치와 연속 수락**을 반영하는 목적이 실제 가속에 더 직접적일 수 있다. 선행 Nemotron 연구의 초안 정렬 실험 역시 이런 방향을 검토할 근거가 된다. :chatgpt-content-reference{index="127"}

블록 크기도 다음처럼 실제 시간 비용에 맞춰 선택하는 연구를 제안할 수 있다.

```math
B^{*}
=
\underset{B\in\mathcal{B}}{\text{arg max}}
\frac{
\mathbb{E}[A_B]+2
}{
T_{\text{draft}}(B)+T_{\text{verify}}(B)
}
```

**기호 설명.** $\mathcal{B}$는 후보 블록 크기의 집합, $A_B$는 해당 크기에서의 수락 길이, $T_{\text{draft}}$와 $T_{\text{verify}}$는 평균 초안·검증 시간이다. $B^{*}$는 식의 비율을 가장 크게 만드는 크기다. 이는 종료·시각 처리·대기열 효과를 생략한 **생성 단계의 제안 기준**이며, 논문에서 구현·검증된 결과는 아니다.

---

## 9. 문서가 답하지 않는 질문

| 미해결 질문 | 현재 근거만으로 답할 수 없는 이유 |
|---|---|
| 한국어와 다른 저자원 언어에서도 정확도와 가속이 함께 유지되는가? | 언어별 품질·초안 수락·페이지 속도 분해가 없다〔pp.8–9〕. :chatgpt-content-reference{index="128"} :chatgpt-content-reference{index="129"} |
| 표 성능 개선 중 확산 학습, 원 주석, 강화학습이 각각 얼마나 기여하는가? | 동일 데이터·비용의 AR 전용 추가 학습 대조군과 요인별 분리가 충분하지 않다〔pp.8·13·25〕. :chatgpt-content-reference{index="130"} :chatgpt-content-reference{index="131"} :chatgpt-content-reference{index="132"} |
| +0.24점 개선이 독립 재학습에서도 반복되는가? | Table 6에 반복 실험의 분산과 신뢰구간이 없다〔p.13〕. :chatgpt-content-reference{index="133"} |
| 불일치한 3.4%의 출력 차이가 실제 업무상 얼마나 중요한가? | 첫 불일치의 수치 분석은 있지만, 전체 오류의 의미적 심각도별 분석은 없다〔p.21〕. :chatgpt-content-reference{index="134"} |
| $B=32$가 다양한 길이·언어·하드웨어에서 적절한가? | 기본 학습·추론 설정이 32이며, 다양한 블록 크기에 대한 일반적인 최적성은 제시되지 않는다〔p.7〕. :chatgpt-content-reference{index="135"} |
| 확률 샘플링에서도 이 구현이 목표 AR 분포를 보존하는가? | 구체적인 출력 동일성 실험은 그리디 설정이다〔p.21〕. :chatgpt-content-reference{index="136"} |
| 추가 학습비를 고려하면 어느 처리량부터 경제적인가? | 학습 장비·단계 수는 있지만 총 학습 시간, 에너지, 비용 회수 분석이 없다〔p.20, Table 7〕. :chatgpt-content-reference{index="137"} |
| 언어적으로 부자연스럽지만 시각적으로 명확한 문자열에도 충실한가? | AR 보존 논증은 있지만, 시각적 충실도와 언어적 개연성을 분리하는 전용 실험은 보고되지 않는다〔p.21의 보존 범위와 관련됨〕. :chatgpt-content-reference{index="138"} |

---

## 10. 결론

### 저자들이 제시한 시사점

저자들의 결론은 **하나의 모델에 병렬 초안과 AR 검증을 함께 학습시키면, 별도 초안망 없이 OCR 생성 비용을 줄일 수 있다**는 것이다. 또한 AR 경로의 명시적인 확률을 활용하면 확산 생성 경로의 확률 추정 없이 작업별 강화학습을 적용할 수 있으며, 보고된 조건에서는 초안 효율을 유지하면서 품질을 개선했다〔p.15, §6〕. :chatgpt-content-reference{index="139"}

**첨부 v1의 결론에는 다국어 확장, 새로운 모델 계열 적용, 적응형 블록 크기 같은 구체적인 후속 연구 일정이나 계획이 명시되어 있지 않다.** 따라서 앞서 제시한 방향은 저자의 계획이 아니라 이 검토의 연구 제안이다.

### 종합 평가와 향후 연구에서 고려할 점

**가장 강하게 지지되는 결론은 ‘조건부 품질 보존형 OCR 가속’이지, ‘일반화 능력을 획기적으로 높인 새로운 OCR 모델’이 아니다.** 동일한 학습 후 AR 모델을 기준으로 병렬 초안을 안전하게 활용하는 설계와 실제 SGLang 구현은 설득력이 있지만, 원래 모델 대비 품질 변화, 실제 연산에서의 출력 불일치, 고동시성에서 줄어드는 가속, 영어·표 중심 데이터라는 범위를 함께 인정해야 한다〔pp.9·12·21·25〕. :chatgpt-content-reference{index="140"} :chatgpt-content-reference{index="141"} :chatgpt-content-reference{index="142"} :chatgpt-content-reference{index="143"}

일반화 성능을 실제로 높이려면 **검증기가 낯선 문서를 정확히 읽는 능력**과 **초안기가 그 검증기에 효율적으로 일치하는 능력**을 별도로 개선하고 검증해야 한다. 이 두 목표를 분리한 데이터 설계, 대조 실험, 시각적 충실도 평가, 실제 서비스 비용 측정이 이 연구를 다음 단계로 발전시키는 핵심이라고 판단한다.

---

## 참고자료 및 출처

아래는 본문 분석과 외부 비교에 사용한 1차 자료의 제목이다. 외부 모델에 대한 GravityOCR의 재측정 수치는 첫 번째 논문을 출처로 삼았다.

| 자료 | 출처 |
|---|---|
| **Diffusion Drafts, AR Verifies: Accelerating Document OCR with Self-Speculative Decoding** — Kim et al., 2026, v1 | 첨부 PDF 및 arXiv. :chatgpt-content-reference{index="144"} |
| **Image-Based Table Recognition: Data, Model, and Evaluation** — Zhong et al., ECCV 2020 | Springer/ECCV. 초고 공개는 2019년이다. :chatgpt-content-reference{index="145"} |
| **Structured Denoising Diffusion Models in Discrete State-Spaces** — Austin et al., 2021 | 논문 원문, arXiv/NeurIPS. :chatgpt-content-reference{index="146"} |
| **OCR-free Document Understanding Transformer** — Kim et al., ECCV 2022 | 논문 원문, arXiv. :chatgpt-content-reference{index="147"} |
| **Nougat: Neural Optical Understanding for Academic Documents** — Blecher et al., 2023 | 논문 원문, arXiv. :chatgpt-content-reference{index="148"} |
| **Fast Inference from Transformers via Speculative Decoding** — Leviathan et al., ICML 2023 | PMLR. :chatgpt-content-reference{index="149"} |
| **DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models** — Shao et al., 2024 | 논문 원문, arXiv. :chatgpt-content-reference{index="150"} |
| **Block Diffusion: Interpolating Between Autoregressive and Diffusion Language Models** — Arriola et al., 2025 | 논문 원문, arXiv. :chatgpt-content-reference{index="151"} |
| **Fast-dLLM v2: Efficient Block-Diffusion LLM** — Wu et al., 2025 | 논문 원문, arXiv. :chatgpt-content-reference{index="152"} |
| **TiDAR: Think in Diffusion, Talk in Autoregression** — Liu et al., 2025 | 논문 원문, arXiv. :chatgpt-content-reference{index="153"} |
| **GLM-OCR Technical Report** — Duan et al., 2026 | 논문 원문, arXiv. :chatgpt-content-reference{index="154"} |
| **DFlash: Block Diffusion for Flash Speculative Decoding** — Chen et al., 2026 | 논문 원문, arXiv. :chatgpt-content-reference{index="155"} |
| **DODO: Discrete OCR Diffusion Models** — Man et al., 2026, v2 | 논문 원문, arXiv. :chatgpt-content-reference{index="156"} |
| **HSD: Training-Free Acceleration for Document Parsing Vision-Language Models with Hierarchical Speculative Decoding** — Liao et al., 2026 | 논문 원문, arXiv. :chatgpt-content-reference{index="157"} |
| **MinerU-Diffusion: Rethinking Document OCR as Inverse Rendering via Diffusion Decoding** — Dong et al., 2026 | 논문 원문, arXiv. :chatgpt-content-reference{index="158"} |
| **Fast-dVLM: Efficient Block-Diffusion VLM via Direct Conversion from Autoregressive VLM** — Wu et al., 2026 | 논문 원문, arXiv. :chatgpt-content-reference{index="159"} |
| **HunyuanOCR-1.5: Making Lightweight OCR VLMs Faster and Better** — Li et al., 2026 | 논문 원문, arXiv. :chatgpt-content-reference{index="160"} |
| **Nemotron-Labs-Diffusion: A Tri-Mode Language Model Unifying Autoregressive, Diffusion, and Self-Speculation Decoding** — Fu et al., 2026 | 논문 원문, arXiv. :chatgpt-content-reference{index="161"} |
