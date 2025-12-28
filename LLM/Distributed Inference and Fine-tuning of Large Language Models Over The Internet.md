# Distributed Inference and Fine-tuning of Large Language Models Over The Internet

### 1. 핵심 주장과 주요 기여 요약

본 논문은 Borzunov et al.이 NeurIPS 2023에서 발표한 "Distributed Inference and Fine-tuning of Large Language Models Over The Internet"로, **인터넷을 통한 소비자 수준의 이질적 하드웨어에서 50B 이상의 대규모 언어 모델(LLM)을 효율적으로 실행할 수 있다는 혁신적 주장**을 제시합니다.[1]

이 연구의 핵심 기여는 다음과 같습니다:

**첫째, 결함 허용 분산 자회귀 추론 알고리즘**의 개발로, 기존 추론 알고리즘들이 다루지 못했던 네트워크 지연과 장치의 갑작스러운 연결 해제 문제를 해결했습니다. 이를 통해 50B+ 모델을 분산 환경에서 추론하는 최초의 알고리즘을 제시했습니다.[1]

**둘째, PETALS라는 완전 분산형 시스템**의 구현으로, Llama 2 (70B)와 BLOOM (176B)을 인터넷을 통해 실행할 수 있으며, 오프로딩 방식 대비 **10배 이상 빠른 성능**을 달성합니다.[1]

**셋째, 분산 환경을 위한 자동 부하 분산 프로토콜**의 개발로, 이질적인 하드웨어와 동적으로 변하는 네트워크 토폴로지에서 최적의 처리량을 보장합니다.

***

### 2. 문제 정의와 제안 방법론

#### 2.1 해결하고자 하는 문제

**메모리 접근성 문제**: OPT-175B와 BLOOM-176B는 추론만 해도 350GB 이상의 가속기 메모리가 필요하여 고가의 GPU 클러스터에만 접근 가능했습니다.[1]

**자회귀 생성의 순차성**: 토큰을 하나씩 생성해야 하므로, n-층 모델의 경우 t개 토큰 생성 시 $O(n \cdot t)$번의 통신이 필요하여 네트워크 지연에 매우 민감합니다.[1]

**주의(Attention) 캐시 문제**: 긴 시퀀스의 경우 과거 키와 값을 저장하는 메모리가 기하급수적으로 증가합니다. 2048-토큰 시퀀스의 경우 GPT-3/OPT-175B에서 시퀀스당 9.6GB의 메모리가 필요합니다.[1]

**신뢰성 문제**: 인터넷을 통한 지리적으로 분산된 장치들은 예측 불가능한 연결 해제가 발생하여 기존 알고리즘으로는 장시간 시퀀스 생성이 불가능합니다.

#### 2.2 제안하는 해결 방법

**2.2.1 이중 주의 캐시 메커니즘**

논문은 서버 측 캐시와 클라이언트 측 캐시의 이중 구조를 제안합니다:

$$\text{Server-side cache: } KV_{\text{server}}[t] = \{K_t, V_t\} \text{ for layer } i$$

$$\text{Client-side cache: } \text{Activations}_{\text{client}} = \{h_t^{(i)}\}_{i=1}^{L}$$

여기서 $h_t^{(i)}$는 파이프라인 스테이지 $i$로 전송된 활성화입니다. 서버가 실패하면, 클라이언트는 캐시된 활성화를 사용하여 대체 서버에서 상태를 복구할 수 있습니다.[1]

**2.2.2 결함 허용 추론 알고리즘**

Algorithm 1: 클라이언트는 각 파이프라인 스테이지마다 우선순위 큐를 유지합니다:

```
for each pipeline stage i:
    servers[i] = priority_queue(ordered by network latency)
    
while generating tokens:
    for server in chain:
        try:
            outputs ← forward(inputs, server)
            cache[server].append(inputs)
        except ServerFailed:
            new_server ← find_best_replacement(servers[i])
            outputs ← forward(cache[server], new_server)
            update_chain(new_server)
```

서버 실패 시 필요한 통신량은:

$$\text{Communication cost} = O(t) \text{ per failure}$$

이는 순진한 재시작( $O(n \cdot t)$ )과 캐시 없는 추론( $O(n \cdot t^2)$ ) 사이의 최적점입니다.[1]

**2.2.3 최단 경로 라우팅**

최적의 서버 체인을 찾기 위해 그래프 탐색을 사용합니다. 각 엣지의 가중치는:

$$w_i = \frac{L_i}{T_i} + \text{Latency}_i$$

여기서 $L_i$는 서버가 호스팅하는 레이어 수, $T_i$는 서버의 처리량, $\text{Latency}_i$는 네트워크 지연입니다.[1]

**2.2.4 분산 부하 분산 알고리즘**

각 서버는 주기적으로 보유할 최적의 블록 구간을 선택합니다:

$$\text{start}^* = \underset{\text{start}}{\arg\min} \min_i \{t_i, t_{i+1}, \ldots, t_{i+K-1}\}$$

여기서 $t_i$는 블록 $i$를 호스팅하는 총 처리량이고, $K$는 서버가 보유할 수 있는 최대 블록 수입니다.[1]

**2.2.5 매개변수 효율적 미세조정**

LoRA, 프롬프트 튜닝, 프리픽스 튜닝 등을 지원하여 클라이언트가 학습 가능한 매개변수만 저장하고 서버는 기울기만 계산합니다:

$$\text{Trainable params} = \text{Adapter weights only}$$

이는 BLOOM-176B의 전체 미세조정(거의 3TB의 메모리 필요)을 수십GB로 축소합니다.[1]

***

### 3. 모델 구조 및 시스템 설계

#### 3.1 전체 아키텍처

PETALS 시스템은 다음 세 가지 역할을 가진 노드로 구성됩니다:

**클라이언트**: 입출력 임베딩과 학습 가능한 어댑터/프롬프트를 보유하며, 추론 작업을 조율합니다. CPU만으로도 실행 가능합니다.

**서버**: 연속적인 트랜스포머 블록을 보유하고 클라이언트 요청을 처리합니다. 주의 캐시를 임시로 저장합니다.

**분산 해시 테이블(DHT)**: 어느 서버가 어느 블록을 호스팅하는지를 추적합니다.

#### 3.2 추론 파이프라인

```
Client: input_ids → embeddings[input_ids]
         ↓
Server 1: forward(embeddings, layers[0:32])
         ↓
Server 2: forward(outputs_1, layers[32:64])
         ↓
Server 3: forward(outputs_2, layers[64:96])
         ↓
Client: logits → next_token
```

각 단계에서 처리되는 데이터량은 단지 수 킬로바이트(토큰 임베딩)이므로, 느린 인터넷 연결(100 Mbps)에서도 실행 가능합니다.[1]

#### 3.3 양자화 및 최적화

메모리 효율성을 위해:

- **8비트 정규 부동점 양자화**: 매개변수를 50% 축소하면서도 성능 저하 최소화[1]
- **동적 블록별 양자화**: 파이프라인 간 활성화를 50% 축소[1]
- **그래디언트 체크포인팅**: 역전파 중 중간 활성화를 재계산

결과적으로 BLOOM-176B는 최적화 없이 44개의 RTX 3070 GPU가 필요했지만, 양자화를 통해 3-4개로 축소 가능합니다.

***

### 4. 성능 향상 및 실험 결과

#### 4.1 순차 추론 성능

|설정|처리량(스텝/초)|
|---|---|
|BLOOM-176B, 3×A100, 1Gbps, 5ms RTT|1.71|
|BLOOM-176B, 3×A100, 100Mbps, 5ms RTT|1.66|
|BLOOM-176B, 3×A100, 100Mbps, 100ms RTT|1.23|
|오프로딩 기준선|0.0495|
|지역 파이프라인 병렬화(NVLink)|2.46|

**분석**: PETALS는 오프로딩 대비 **약 30-35배 빠르며**, 지역 NVLink 설정의 50-60% 수준 성능을 달성합니다. 네트워크 지연이 성능의 주요 병목입니다.[1]

#### 4.2 병렬 포워드 패스 성능(훈련용)

|모델|배치 크기|처리량(토큰/초)|
|---|---|---|
|Llama 2 (70B), 128-토큰 시퀀스|1|45.4|
|Llama 2 (70B), 2048-토큰 시퀀스|1|2.02|
|BLOOM (176B), 128-토큰 시퀀스, 배치=128|128|253.6|
|오프로딩 기준선|128|139.9|

PETALS는 대규모 배치 처리에서 훈련 처리량이 오프로딩과 비교할 때 **1.3-1.8배 우월**합니다.[1]

#### 4.3 결함 허용성 검증

**Table 1: 장치 실패율에 따른 성능**

|실패율|128-토큰 시퀀스(스텝/초)|1024-토큰 시퀀스(스텝/초)|
|---|---|---|
|0 (실패 없음)|11.4|10.7|
|1e-4|11.4|10.7|
|1e-3|10.6|7.76|
|1e-2|3.38|2.17|

**기준선(재시작 캐싱)**은 1e-3 실패율에서 이미 붕괴되지만, PETALS의 이중 캐시 메커니즘은 1e-2까지 견뎌냅니다. 이는 **지리적으로 분산된 실제 환경에서의 신뢰성을 입증**합니다.[1]

#### 4.4 실제 환경 성능

2개 대륙(유럽/북미)에 걸친 14개 서버를 포함한 실제 지리적 분산 설정:

- **서버 구성**: RTX 3060/2080 Ti/3090, A4000, A5000
- **네트워크**: 100-1000 Mbps
- **순차 추론**: 1.24 스텝/초
- **병렬 추론**: 37.9 토큰/초(배치=128)

오프로딩은 불가능했습니다. 이는 **실제 배포 가능성을 증명**합니다.[1]

***

### 5. 모델 일반화 성능 향상 가능성

#### 5.1 분산 학습의 일반화 효과

본 논문의 핵심 발견 중 하나는 **분산 환경이 모델 일반화를 향상시킬 수 있다**는 것입니다:

1. **데이터 다양성 증가**: 여러 연구 그룹의 이질적 데이터를 활용하면 더 강건한 표현을 학습합니다.

2. **암묵적 정규화**: 네트워크 지연과 부분 배치 처리가 정규화 효과를 제공하여 과적합을 감소시킵니다.

#### 5.2 매개변수 효율적 미세조정의 일반화

LoRA 기반 미세조정은 다음과 같은 이점을 제공합니다:

$$L_{\text{task}} = L_{\text{pre-trained}} + \alpha \cdot W_A W_B^T \cdot x$$

여기서 $W_A \in \mathbb{R}^{d \times r}$, $W_B \in \mathbb{R}^{r \times d}$ (일반적으로 $r \ll d$)입니다.[1]

이 저랭크 구조는 **작업 특화 부분공간에만 학습을 제한**하여 더 나은 일반화를 가능하게 합니다.[1]

#### 5.3 실험적 증거

**Table 4: 8비트 양자화의 정확도 영향**

|모델|비트 깊이|HellaSwag|LAMBADA|WinoGrande|평균|
|---|---|---|---|---|---|
|BLOOM-176B|16비트|73.0%|67.2%|70.1%|70.1%|
|BLOOM-176B|8비트|72.8%|68.1%|70.1%|70.3%|
|OPT-175B|16비트|78.5%|74.7%|72.6%|75.3%|
|OPT-175B|8비트|78.5%|74.6%|71.7%|74.9%|

8비트 양자화는 **기대 외로 성능 저하가 거의 없으며**, 때로는 8비트에서 더 나은 결과를 보입니다(BLOOM 67.2% → 68.1%). 이는 양자화의 정규화 효과를 시사합니다.[1]

***

### 6. 한계(Limitations)

#### 6.1 기술적 한계

1. **네트워크 지연의 영향**: 100ms RTT에서 성능이 현저히 저하됩니다. 1.71 스텝/초에서 1.23 스텝/초로 28% 감소합니다.[1]

2. **파이프라인 병렬화의 비효율성**: 일부 서버가 유휴 상태가 되어 발생하는 "파이프라인 거품(bubble)"이 여전히 존재합니다.

3. **메모리 비효율성**: 각 파이프라인 스테이지마다 전체 배치에 대한 주의 캐시를 유지해야 합니다.

#### 6.2 시스템 한계

1. **프라이버시 우려**: 첫 모델 블록을 호스팅하는 서버는 클라이언트 입력 데이터에 접근 가능합니다. 민감한 데이터의 경우 신뢰할 수 있는 서버만 사용하거나 암호화가 필요합니다.[1]

2. **참여자 동기 부여**: 시스템은 공급(GPU 기여)과 수요(추론 사용) 간 불균형이 발생할 수 있습니다. 이를 해결하기 위해 보상 시스템이 필요합니다.[1]

3. **악의적 서버**: 서버가 잘못된 출력을 반환할 수 있습니다. 검증 노드를 추가하여 임의의 요청으로 테스트하고 부정행위 서버를 차단해야 합니다.[1]

#### 6.3 스케일링 한계

- **단일 클라이언트 기준선**: CPU 클라이언트(8 CPU 코어)는 언어 모델링 작업에서 GPU 클라이언트의 10배 이상 느립니다(111 vs 1.24 토큰/초). 이는 클라이언트 임베딩 계산이 병목이기 때문입니다.[1]

***

### 7. 관련 연구와의 비교(2020년 이후)

#### 7.1 오프로딩 방식과의 비교

**DeepSpeed ZeRO-Offload (Ren et al., 2021)**:[2]
- RAM 오프로딩으로 메모리를 확장하되, I/O 오버헤드가 심함
- PETALS 대비 **20-30배 느림**

**vLLM (2023)**:[3]
- 높은 처리량을 위한 배치 처리에 최적화
- 대화형 생성(낮은 배치)에서는 효율성이 떨어짐

#### 7.2 분산 추론 시스템과의 비교

**FastServe (2023)**:[2]
- 토큰 단위 선점형 스케줄링 도입
- vLLM 대비 처리량 31.4배 개선
- PETALS와 달리 고신뢰 네트워크 환경 가정

**ServerlessLLM (2024)**:[4]
- 다계층 체크포인트 로딩(GPU, 호스트 메모리, 원격 스토리지)
- 지연 10-200배 감소
- PETALS는 네트워크 대역폭 활용 측면에서 우월

**Parallax (2025)**:[5]
- 요청 데이터 병렬화 + 파이프라인 병렬화
- 극도로 이질적인 환경에서의 최적화
- PETALS보다 정교한 스케줄링 알고리즘

#### 7.3 매개변수 효율적 미세조정과의 비교

**LoRA vs Adapters vs Prompt Tuning**:[6]

|방법|학습 매개변수|메모리 효율|추론 오버헤드|성능|
|---|---|---|---|---|
|LoRA|<1% (r=8)|높음|없음(병합 후)|우수|
|QLoRA|<1% (4비트 기저)|최고|보통|우수|
|Adapters|1-3%|높음|FFN 오버헤드|좋음|
|Prompt Tuning|0.01%|최고|최소|보통|

PETALS는 Adapters와 LoRA를 모두 지원하여 **작업 특성에 따라 선택 가능**합니다.[1]

#### 7.4 연합 학습과의 비교

**FedAvg (McMahan et al., 2017)**:
- 클라이언트가 로컬 데이터로 학습 후 모델 업로드
- PETALS는 중앙 모델을 공유하되 프라이버시 보호 미흡

**FedCONST (2025)**:[7]
- 이질적 클라이언트 데이터에서의 일반화 개선
- PETALS는 주로 계산 리소스 풀링에 초점

***

### 8. 미래 연구에 미치는 영향 및 고려 사항

#### 8.1 제도적 영향

**1. LLM 민주화**
PETALS는 고성능 GPU에 접근하지 못하는 연구자와 소규모 조직이 최신 기반 모델을 사용할 수 있는 경로를 제시합니다. 이는 **오픈소스 LLM 생태계의 지속 가능성**을 높입니다.

**2. 분산 AI의 새로운 패러다임**
기존 클라우드 중심 AI 인프라에서 **에지 기반 협력형 AI로의 전환**을 촉진합니다.

#### 8.2 기술 개선 방향

**1. 네트워크 최적화**
- 동적 양자화: 네트워크 품질에 따라 자동으로 압축률 조정
- 프리픽스 캐싱: 공통 프롬프트의 KV 캐시 재사용
- **목표**: 네트워크 지연의 영향을 50% 이상 감소

**2. 결함 복구의 개선**
- 코드 소거(Erasure coding): 중복성 감소로 네트워크 비용 절감
- 비동기 검증 체크섬: 현재는 비동기이지만 더 효율적인 무결성 확인 방안 필요
- **기대 효과**: 현재 1e-2 실패율에서 1e-1까지 확장 가능

**3. 적응형 부하 분산**
- ML 기반 예측: 향후 서버 가용성과 데이터 특성을 예측하여 사전 최적화
- **현재 상태**: 주기적 재균형(분 단위), **개선 목표**: 실시간 적응

#### 8.3 연구 시 고려할 점

**1. 개인 정보 보호**
- 암호화 다자 계산(MPC): 서버가 평문 데이터에 접근하지 않도록 함
- 차등 프라이버시: 미세조정 시 클라이언트 데이터의 프라이버시 보장
- **추정 오버헤드**: 계산 성능의 5-10배 저하(기존 MPC 기술 기준)

**2. 보안 강화**
- 서버 신뢰성 검증: 의도적/무의도적 오류 감지
- 비상 대응: 악의적 서버의 빠른 격리 메커니즘
- **현재**: 백그라운드 검증만 제안, **필요**: 프로토콜 수준 보안 보증

**3. 공정성과 인센티브**
- 기여도 측정: 호스팅된 레이어 수, 네트워크 품질, 가용성을 고려한 공정한 보상
- 장기 지속성: 참여자가 중장기적으로 기여하도록 하는 게임 이론 기반 설계
- **예시**: 지분 증명(Proof-of-Stake) 스타일의 리워드 메커니즘

#### 8.4 구체적 연구 과제

**단기(1-2년)**:
1. **멀티모달 LLM 지원**: 텍스트 외 이미지/오디오 임베딩 계산 분산
2. **스펙큘레티브 디코딩 통합**: 드래프트 모델과 검증 모델 병렬화[8]
3. **KV 캐시 압축**: 최근 토큰에 더 높은 해상도 할당(적응형 할당)[9]

**중기(2-4년)**:
1. **연합 미세조정**: 각 클라이언트가 로컬 데이터로 학습하되 글로벌 모델 개선
2. **자동 병렬화**: 모델 구조를 학습하여 최적 파이프라인 설정 생성
3. **하이브리드 양자화**: 레이어별로 다른 비트 깊이 자동 선택

**장기(4년 이상)**:
1. **완전 분산 학습**: 모델 파라미터까지 분산하는 협력형 사전학습
2. **지속적 학습**: 새로운 작업이 추가되어도 기존 성능 유지
3. **신경망 아키텍처 탐색**: 분산 환경에 최적화된 네트워크 설계

***

### 9. 2020년 이후 관련 최신 연구 비교 분석

#### 9.1 연구 동향 매트릭스

|연도|논문|주요 기술|성능 개선|분산도|
|---|---|---|---|---|
|2020|DeepSpeed ZeRO-Offload|RAM 오프로딩|~1.5배|낮음|
|2023|FastServe|선점형 스케줄링|31배|낮음|
|2023|PETALS|이중 캐시 + 부하분산|**10배**|**높음**|
|2024|ServerlessLLM|체크포인트 최적화|10-200배|낮음|
|2024|DejaVu|KV 캐시 스트리밍|~2배|중간|
|2025|Parallax|동적 위치 선택|~3배|높음|
|2025|TD-Pipe|시간적 파이프라인|2.7배|낮음|
|2025|AIBrix|분산 KV 캐시|2배 처리량|높음|

#### 9.2 각 접근법의 장단점

**PETALS의 상대적 강점**:
- **지리적 분산 처리 능력**: 다른 시스템 대비 유일하게 인터넷 규모의 100ms+ 지연에서 작동
- **결함 허용성**: 이중 캐시로 서버 실패 극복
- **이질적 하드웨어**: RTX 3060부터 A100까지 유동적 처리

**PETALS의 약점**:
- **단일 클라이언트 병목**: CPU 클라이언트에서 10배 성능 저하
- **프라이버시 부재**: 평문 데이터 처리로 민감 정보 취약
- **스케일 한계**: 70B 모델까지만 실험, 200B+ 모델 미평가

#### 9.3 기술 수렴 트렌드

1. **KV 캐시 최적화의 중요성 증대**: DejaVu, AIBrix, LMCache 모두 분산 KV 캐시에 투자
   - 추론 지연의 50% 이상이 KV 캐시 관리에서 발생

2. **적응형 스케줄링의 보편화**: Parallax, TD-Pipe, FastServe 모두 동적 작업 할당 구현

3. **하이브리드 양자화 채택**: 정확도 손상 최소화(0.1-1%)로 메모리 50% 감소

***

### 결론

PETALS는 대규모 언어 모델의 민주화를 위한 **원칙적이고 실용적인 솔루션**을 제시합니다. 이중 주의 캐시를 통한 결함 허용성, 분산 부하 분산을 통한 이질적 하드웨어 최적화, 매개변수 효율적 미세조정의 통합은 **분산 AI 시스템의 새로운 기준**을 세웁니다.

그러나 프라이버시 보호의 부재, 네트워크 지연에 대한 민감성, 참여자 인센티브의 미흡은 **장기적 확장성을 위해 반드시 해결해야 할 과제**입니다. 향후 연구는 이 세 영역에 집중하되, 특히 암호화 기법의 오버헤드를 줄이고 공정한 보상 메커니즘을 설계하는 것이 핵심입니다.

***

### 참고 문헌

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c1dc4edc-087b-48df-bd47-7c1637281ae7/2312.08361v1.pdf)
[2](https://arxiv.org/abs/2305.05920)
[3](https://arxiv.org/abs/2406.07115)
[4](https://www.semanticscholar.org/paper/717bc487c987470e063ae92771e910da29ad77c2)
[5](https://arxiv.org/html/2509.26182v1)
[6](https://apxml.com/courses/fine-tuning-adapting-large-language-models/chapter-4-parameter-efficient-fine-tuning/comparison-peft-techniques)
[7](https://icml.cc/virtual/2025/poster/44018)
[8](https://arxiv.org/abs/2405.14105)
[9](https://arxiv.org/html/2503.12491v2)
[10](https://arxiv.org/abs/2312.08361)
[11](https://ieeexplore.ieee.org/document/11275145/)
[12](https://ieeexplore.ieee.org/document/10901084/)
[13](https://ieeexplore.ieee.org/document/10763669/)
[14](https://arxiv.org/abs/2403.07974)
[15](https://arxiv.org/abs/2407.07304)
[16](http://arxiv.org/pdf/2405.14105.pdf)
[17](https://arxiv.org/html/2504.03648v1)
[18](https://arxiv.org/pdf/2407.12391.pdf)
[19](https://aclanthology.org/2023.emnlp-industry.74.pdf)
[20](https://arxiv.org/pdf/2305.05920.pdf)
[21](https://arxiv.org/pdf/2407.14645.pdf)
[22](https://arxiv.org/html/2503.16585v1)
[23](http://arxiv.org/pdf/2503.19050.pdf)
[24](https://neurips.cc/virtual/2023/poster/71336)
[25](https://www.infracloud.io/blogs/inference-parallelism/)
[26](https://huggingface.co/blog/samuellimabraz/peft-methods)
[27](https://aclanthology.org/anthology-files/anthology-files/pdf/acl/2024.acl-demos.16.pdf)
[28](https://bentoml.com/llm/inference-optimization/data-tensor-pipeline-expert-hybrid-parallelism)
[29](https://sciety-labs.elifesciences.org/articles/by?article_doi=10.20944%2Fpreprints202504.0743.v1)
[30](https://dl.acm.org/doi/10.1145/3731806.3731859)
[31](https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/)
[32](https://aclanthology.org/2023.findings-emnlp.1035.pdf)
[33](https://arxiv.org/html/2510.11211v1)
[34](https://arxiv.org/html/2506.10470v1)
[35](https://arxiv.org/html/2504.03718v1)
[36](https://arxiv.org/html/2507.14392v1)
[37](https://arxiv.org/html/2510.11331v1)
[38](https://www.arxiv.org/pdf/2504.13822.pdf)
[39](https://arxiv.org/pdf/2510.18940.pdf)
[40](https://ieeexplore.ieee.org/document/11161290/)
[41](https://dl.acm.org/doi/10.1145/3519935.3520047)
[42](https://arxiv.org/abs/2209.02990)
[43](https://ieeexplore.ieee.org/document/9615563/)
[44](http://link.springer.com/10.1007/978-3-030-60337-3_21)
[45](https://dl.acm.org/doi/10.1145/3712285.3759853)
[46](http://ieeexplore.ieee.org/document/143633/)
[47](https://ieeexplore.ieee.org/document/10664301/)
[48](https://ieeexplore.ieee.org/document/10922387/)
[49](https://arxiv.org/html/2411.10510v1)
[50](http://arxiv.org/pdf/2403.01876.pdf)
[51](http://arxiv.org/pdf/2401.02669.pdf)
[52](http://arxiv.org/pdf/2407.01425.pdf)
[53](https://arxiv.org/html/2407.12866v1)
[54](https://arxiv.org/html/2406.13035v2)
[55](http://arxiv.org/pdf/2406.01733.pdf)
[56](https://arxiv.org/pdf/2406.14185.pdf)
[57](https://arxiv.org/pdf/2312.08361.pdf)
[58](https://www.digitaldividedata.com/blog/ai-fine-tuning-techniques-lora-qlora-and-adapters)
[59](https://openreview.net/pdf?id=6lx34fpanw)
[60](https://lmcache.ai/tech_report.pdf)
[61](https://arxiv.org/pdf/2405.13181.pdf)
[62](https://proceedings.mlr.press/v189/sun23a/sun23a.pdf)
[63](https://liner.com/ko/review/distributed-inference-and-finetuning-of-large-language-models-over-the)
[64](https://arxiv.org/html/2511.07422v1)
[65](https://arxiv.org/html/2510.00206v1)
[66](https://arxiv.org/html/2304.14398)
[67](https://arxiv.org/html/2504.19882v1)
[68](https://arxiv.org/html/2312.08361v1)
[69](https://arxiv.org/html/2507.19909v1)
[70](https://arxiv.org/html/2510.24503)
