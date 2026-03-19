## Ablations (어블레이션)
AI/머신러닝 논문이나 모델 분석에서 자주 나오는 **“ablation(어블레이션)”**은 일반 단어가 아니라 실험 기법을 뜻하는 전문 용어.

✅ 한 줄 정의
**Ablation(어블레이션)**은
모델에서 어떤 구성 요소를 “의도적으로 제거하거나 바꿔보며” 그 요소의 중요성을 평가하는 실험 방식

✅ 쉽게 말하면
"이 부분이 정말 중요한 걸까? 빼 보면 알겠지!"
→ 이런 접근으로 하나씩 요소를 제거하거나 바꿔보며 영향력을 측정하는 실험

🧠 예시로 이해하기
모델 예:
Model = Embedding → Transformer → LayerNorm → Output
이 모델에서 "LayerNorm"이 중요한지 알고 싶다면?
👉 Ablation 실험:
•	Case 1 (원래 모델): 정확도 91.2%
•	Case 2 (LayerNorm 제거): 정확도 84.7%
→ 차이: -6.5%
✅ 결론: LayerNorm은 모델 성능에 중요한 요소임

📊 Ablation 실험 결과는 보통 이렇게 요약 됨:
구성 요소	제거 여부	정확도 변화
Positional Encoding	제거	-10.2%
LayerNorm	제거	-6.5%
Dropout	제거	+0.3%
→ 성능이 떨어지면 → 중요한 요소
→ 성능이 오르면 → 오히려 해가 되었던 요소

✅ 왜 Ablation이 중요한가?
이유	설명
🔍 모델 해석 가능성 증가	어떤 구조가 성능에 어떻게 기여하는지 이해할 수 있음
🧪 설계 최적화	꼭 필요한 것만 남기고 나머지는 단순화 가능
⚠️ 과적합 탐지	특정 기법이 오히려 성능을 해치고 있을 수 있음

✅ 용어 어원 (재미로)
•	"ablation"은 원래 의학/우주항공 용어로
→ 조직을 잘라내거나, 열로 태워 없애는 행위를 뜻합니다.
→ 여기서 파생되어 "모델 구성 요소를 일부러 제거해서 분석하는 AI 용어"로 자리 잡음.

✅ 요약
항목	내용
Ablation이란?	모델 구성 요소를 제거하거나 바꿔서 성능 변화 측정하는 실험 방식
왜 중요?	어떤 요소가 진짜 중요한지 정량적으로 평가할 수 있음
어디에 많이 쓰임?	논문, 모델 구조 설계, 튜닝 최적화, 논리적 설명 근거 제시 등

## ADAMW(AdamW 옵티마이저)
1️⃣ Optimizer(옵티마이저, 최적화 알고리즘)란?

학습은 결국 이 한 줄입니다:

𝑤
𝑒
𝑖
𝑔
ℎ
𝑡
=
𝑤
𝑒
𝑖
𝑔
ℎ
𝑡
−
𝑙
𝑒
𝑎
𝑟
𝑛
𝑖
𝑛
𝑔
_
𝑟
𝑎
𝑡
𝑒
×
𝑠
𝑜
𝑚
𝑒
𝑡
ℎ
𝑖
𝑛
𝑔
weight=weight−learning_rate×something

여기서 “something”을 계산해주는 것이
**Optimizer(옵티마이저)**입니다.

대표적인 것:

SGD (Stochastic Gradient Descent, 확률적 경사하강법)

Adam (Adaptive Moment Estimation, 적응적 모멘트 추정)

AdamW

LLM에서는 거의 AdamW가 표준입니다.

2️⃣ Adam(아담)은 무엇이 좋았나?

Adam은 두 가지를 합니다:

1️⃣ Gradient의 평균 (1차 모멘트, momentum 개념)
2️⃣ Gradient 제곱의 평균 (2차 모멘트, 분산 개념)

즉,

방향도 보고
얼마나 불안정한지도 본다

그래서 SGD보다 훨씬 안정적이고 빠르게 수렴합니다.

3️⃣ 그런데 Adam의 문제점

Adam에는 Weight Decay(가중치 감쇠) 문제가 있었습니다.

Weight Decay란?

가중치가 너무 커지는 걸 막기 위해
조금씩 줄이는 규제 방법입니다.

weight=weight−lr×gradient−lr×λ×weight

이게 정석적인 L2 regularization입니다.

문제는?

Adam에서는
이 weight decay가 gradient에 섞여 들어가 버립니다.

즉,

진짜 gradient

regularization gradient
가 뒤섞임

그 결과:

일반화 성능 저하

최적화 동작이 이론적으로 깔끔하지 않음

4️⃣ AdamW의 핵심 차이

AdamW는 이것을 분리했습니다.

weight=weight−lr×Adam_update

그리고 따로

weight=weight−lr×λ×weight

즉,

👉 gradient 업데이트와
👉 weight decay를 완전히 분리

이게 핵심입니다.

5️⃣ 왜 LLM에서 AdamW가 표준인가?

Transformer 계열은:

파라미터 수가 매우 큼

overfitting 위험 존재

안정적 수렴 필요

AdamW는:

수렴 빠름

일반화 좋음

대형 모델에서 실증적으로 우수

그래서 GPT, BERT, LLaMA 대부분 AdamW 사용합니다.

## add and normalize
 
✅ “Add and Normalize”란?
✔️ 2단계로 구성:
1.	Add:
→ 이전 단계의 입력을 그 다음 블록 출력과 더함.
→ 이걸 Residual Connection (잔차 연결) 또는 Skip Connection이라고 함.
2.	Normalize:
→ 그 더해진 결과에 대해 **Layer Normalization (레이어 정규화)**를 적용함.
→ 입력값 분포를 일정하게 만들어 학습 안정성을 높임.

🔍 왜 “Add”가 필요한가? (Residual Connection)
•	신경망이 깊어질수록 정보 손실과 기울기 소실(gradient vanishing) 문제가 생김.
•	그래서 입력 정보를 직접 더해 줌으로써,
→ 모델이 “원래 정보”도 기억하면서
→ “변형된 정보”와 함께 사용할 수 있게 함.
즉:
Output = LayerNorm(x + SubLayer(x))
여기서 SubLayer(x)는 예: Self-Attention, Feedforward 등

🔍 왜 “Normalize”가 필요한가? (LayerNorm)
•	정규화는 훈련 안정화를 위한 기술.
•	LayerNorm은 배치 크기와 상관없이, 각 토큰의 feature 값들을 평균과 분산 기준으로 정규화.
•	이로 인해:
o	학습이 더 빠르고 안정적으로 수렴
o	과도한 feature 값 차이가 줄어들어 균형 있는 정보 처리 가능

🎯 예시: Self-Attention 뒤에 붙은 경우
Input → Self-Attention → 결과 A
Input + A → Add  
→ LayerNorm → 다음 계층으로 전달
같은 방식으로 Feedforward 뒤에도 사용됨

✅ 시각적 요약
위치	기능	효과
Self-Attention 후	Add & Norm	원본 + 어텐션 결과 → 정규화 → 다음 단계
Feedforward 후	Add & Norm	원본 + FFN 결과 → 정규화 → 다음 단계
🎯 결과	정보 손실 방지 + 학습 안정화 + 빠른 수렴	

✅ 결론
"Add and Normalize"는 Transformer 모델에서 핵심 계층(Attention, Feedforward)의 결과에 잔차 연결을 더하고 정규화함으로써, ㄴ모델의 학습 안정성과 표현 능력을 높여주는 중요한 구조

## Anchored Preference Optimization (APO)
LLM 훈련 기법 중 하나로, 특히 **RLHF (Reinforcement Learning from Human Feedback, 인간 피드백 기반 강화 학습)**의 단점을 보완하려고 나온 방식

🔷 APO 한 줄 정의
APO는 “선호도(preference)” 학습을 하되, 기준(anchor)이 되는 모델을 고정하고, 거기서 너무 멀어지지 않게 하면서 최적화하는 방식

🔶 왜 필요하냐?
기존의 RLHF 방식은 다음과 같은 문제점이 있었음
문제점	설명
🎭 Mode collapse	모델이 너무 특정 스타일로만 말하게 됨 (창의성 상실)
😵 Over-optimization	사람 피드백에만 맞추다 보니 **기존 능력(지식/추론)**이 퇴화됨
⚠️ 불안정성	PPO (Proximal Policy Optimization)는 튜닝이 까다롭고 불안정함

🔷 APO 방식은 이렇게 다릅니다
1. Anchor 모델을 고정(anchor model)
•	기존 base 모델(예: SFT 모델)을 **기준점(anchor)**으로 잡고,
•	이 기준점에서 너무 멀어지지 않도록 제한합니다.
→ 즉, 기존 지식과 성능을 유지한 채로 선호도만 반영하려는 것

2. Loss (손실 함수) 구성
APO는 아래처럼 세 가지 목적을 합칩니다:
APO Loss = Preference Loss + KL Penalty + Anchoring Penalty
항목	설명
Preference Loss	사람의 선호도에 따라 선택된 답을 더 좋게 만듬
KL Penalty	확률 분포가 원래 모델에서 너무 멀어지지 않게 제한
Anchoring Penalty	Anchor 모델과의 차이를 벌점으로 부여함

🔶 비유로 쉽게
**기존 모델(Base Model)을 앵커(anchor)**로 고정해두고,
그로부터 멀리 벗어나지 않는 선에서만 "좋은 응답"으로 미세 조정하는 방식

🔷 APO의 장점
장점	설명
🧠 기존 능력 유지	모델이 너무 바보가 되는 걸 방지함 (지식 퇴화 방지)
⚖️ 균형 유지	인간 피드백 반영 + 원래 모델의 신뢰성 유지
🔧 안정적 훈련	PPO보다 학습이 간단하고 안정적
✅ 요약
항목	내용
이름	Anchored Preference Optimization (APO)
목적	사람 선호를 반영하되, 기존 모델과 너무 멀어지지 않도록 조정
핵심 아이디어	Anchor 모델을 기준으로 최적화
장점	기존 지식 유지, 안정적 훈련, mode collapse 방지
대체 대상	PPO 기반 RLHF 방식

## ArXiv
물리학, 수학, 컴퓨터 과학, 통계학, 전기전자공학, 경제학, 생물학 등 다양한 분야의 **학술 논문을 사전 공개(preprint)**하는 온라인 저장소

📌 ArXiv란?
항목	설명
전체 이름	ArXiv.org (“archive”의 변형)
운영 기관	미국 Cornell University(코넬 대학교)
설립 시기	1991년 (물리학에서 시작됨)
목적	학술 논문의 빠른 공유와 공개
사용 대상	연구자, 대학원생, 학계, 기업 연구소 등
주요 분야	수학, 물리학, 컴퓨터 과학(특히 AI), 통계 등

✅ ArXiv의 장점
•	빠르게 논문 공유: 정식 학회나 저널에 게재되기 전, 누구나 논문을 읽고 인용 가능
•	무료: 등록, 열람, 다운로드 전부 무료
•	최신 기술 트렌드 확인: AI나 LLM 분야는 대부분 최신 논문이 먼저 arXiv에 올라옴. (예: GPT 시리즈 관련 논문들도 처음 arXiv에 공개됨)

📚 예시
•	"Attention is All You Need" (Transformer 논문)
•	"BERT: Pre-training of Deep Bidirectional Transformers"
•	"GPT-4 Technical Report"
이런 유명 논문들도 arXiv에 먼저 등록됨

💡 참고
•	arXiv는 **논문 검증(심사)**을 하지 않습니다. 사전 공개 목적이므로 내용의 신뢰성은 연구자 스스로 판단해야 함.
•	사이트 주소: https://arxiv.org

## ascalon
Tenstorrent가 개발 중인 고성능 RISC‑V 범용 CPU IP(기술)로, 서버 · 엣지 · HPC 환경에서 성능과 전력효율을 모두 추구하는 설계.
	출시일: 2025년 7월 현재 곧 출시라는 내용만 있고, 정확한 출시일 미정.
🧩 주요 특징 (Tenstorrent 공식 설명 기반)  
•	RISC V ISA 기반, RVA23 규격 준수
•	코어당 2코어 ~ 8코어 확장 가능, 클러스터당 2–8개 코어로 구성
•	공유 L2 캐시 구성 가능
•	신뢰성 확보: TrustZone 동등한 보안 구조
•	인터페이스: CHI.E 및 AXI5-LITE 지원
•	디코더 폭: 2~8-issue wide (특히 8-wide 성능 코어 포함)
•	성능 밀도가 뛰어나며, 고성능과 전력 효율을 동시에 추구

🌍 생태계 & 배치
•	CCL 라이선스를 통해 개발자들은 IP 활용 가능
•	2nm 엣지 AI 가속용 칩에 라이선스되어, Rapidus와 일본 LSTC와 협업 중  
•	LLVM, QEMU 등 도구 지원:
o	LLVM 20에 -mcpu=tt-ascalon-d8로 타겟 지원  
o	QEMU에 tt ascalon CPU 시뮬레이터 추가  
•	Imperas와 협업하여, SW 시뮬레이션 모델 제공

🔭 향후 로드맵
•	스케일러블 CPU IP 라인업: 2-wide ~ 8-wide 버전 존재  
•	향후 출시될 멀티칩렛 SoC (Aegis, Grendel)에는 Ascalon 기반 CPU가 포함될 예정
✅ 요약
•	Ascalon은 고성능 CPU 코어를 요구하는 서버, 엣지, HPC 용으로 설계된 8 wide 아웃오브오더 RISC V CPU.
•	고성능 + 효율성을 동시에 달성하며, AI·ML 가속기와의 통합을 염두에 둔 IP 설계.
•	이미 파트너와 협력, 컴파일러·시뮬레이터 지원, 라이선싱 확장이 활발히 이루어지고 있음.

🎯 Tenstorrent가 Ascalon을 통해 이루고자 하는 목적
1. AI NPU + CPU 통합 SoC 생태계 구축
•	Tenstorrent는 NPU (Neural Processing Unit) 하드웨어 전문 회사.
•	하지만 NPU는 CPU 없이는 혼자 작동할 수 없기 때문에, 항상 외부 CPU(예: Intel, AMD, ARM)를 함께 써야 함.
•	Ascalon은 그 의존을 끊기 위한 전략.
🔧 즉, 자체 설계한 고성능 CPU + 자체 NPU를 결합한 완전 독립형 AI SoC를 만들려는 게 본질.
2. 칩렛 기반 모듈화된 고성능 SoC 플랫폼 만들기
•	Tenstorrent의 SoC 설계는 칩렛(chiplet) 기반. 즉,
o	CPU (Ascalon)
o	NPU (Wormhole, Blackhole 등)
o	IO 모듈 (PCIe, DRAM 등)
이 각각이 칩렛으로 따로 만들어져 있음.
•	이 칩렛들을 목적에 따라 서버형, 엣지형, 자동차형 등으로 구성할 수 있는 구조를 만들고 있는 중.
•	🎯 Ascalon은 그 중에서 “CPU 칩렛”을 담당
3. 인텔/AMD의 x86 생태계에 대한 탈피
•	x86은 독점적인 ISA(Intel/AMD만 설계 가능)이며, 라이선스도 제한적.
•	반면 RISC V는 오픈 ISA. 누구나 설계해서 제품에 넣을 수 있음.
•	Tenstorrent는 AI 시대에 맞춰 완전히 자체 설계 가능한 하드웨어 스택을 확보하려는 전략.
🧠 "AI는 데이터와 연산의 자유가 필요하므로, CPU도 남의 것을 쓰지 말자"는 방향.
4. 차세대 엣지/서버/자동차용 AI 플랫폼 지향
•	Tenstorrent는 현재:
o	Loudbox (서버)
o	Quasar (엣지용)
o	Grendel, Aegis (미래형 SoC) 등을 개발 중인데,
•	이 제품들의 공통점은:
✅ Ascalon CPU + Wormhole NPU 조합을 통해 독립형으로 운영될 수 있게 한다는 것

## attention
**“입력된 정보들 중에서, 어떤 부분에 더 집중할지(주의할지) 정하는 기법”**.
→ 모델이 문장 전체를 보는 대신, 중요한 단어에 더 집중해서 이해하거나 생성하는 데 쓰임. ‘중요한 정보에 더 큰 가중치를 주는 메커니즘’

🎯 예제로 설명
문장:
"The cat sat on the mat."
질문:
“sat이 누구의 행동인가?”
→ 사람이라면 **"cat"**에 집중함
AI 모델도 마찬가지.
이럴 때 Attention이 "sat → cat" 연결을 강조해 줌.
즉, AI가 문장 안의 관련 단어들 사이의 연결성을 이해하도록 돕는 장치.

✅ 구조 요약: Attention 메커니즘
Transformer에서의 Attention은 다음과 같은 세 가지 벡터를 이용해 계산
구성 요소	설명	역할
Query (질문)	"무엇에 주의를 기울일까?"	현재 단어가 찾고자 하는 정보
Key (열쇠)	"내가 가진 정보의 특징은?"	각 입력 단어가 가진 의미 표현
Value (값)	"실제 전달할 정보는?"	주의가 높은 Key에서 가져올 내용

🧠 계산 방식 (간단히)
1.	각 토큰을 → Query, Key, Value로 변환
2.	Query와 Key 간 유사도(내적, dot product) 계산 → Attention Score
3.	Softmax로 정규화 → 중요도 비율
4.	중요도 × Value 벡터 → 가중 평균 → 최종 출력 벡터 생성

🎨 시각적 예 (한 문장 내부 Self-Attention)
"The     cat     sat     on     the     mat"
   ↘︎     ↙︎
     (sat → cat) : 높은 attention 점수
→ "sat"이라는 단어는 "cat"과의 관련성을 높게 평가해서, 더 많은 정보를 받음

✅ 왜 중요한가?
효과	설명
💡 맥락 이해	단어 간 관계를 스스로 파악
💬 문장 생성	더 자연스러운 문장 생성 가능
🌐 다국어 번역	원문과 번역문의 대응 관계를 정교하게 표현 가능
📚 대용량 학습 가능	병렬 계산이 쉬워져 대규모 모델 학습에 유리

📌 Attention vs Weight 차이점 요약표
항목	Attention	Weight
의미	입력 간 상대적 중요도를 실시간으로 계산	뉴런 사이 연결 강도
학습 시기	학습 중 직접적으로 학습되기보단 계산식으로 유도됨	학습 과정 중 역전파로 업데이트
고정 여부	입력마다 다르게 계산됨 (동적)	학습이 끝나면 고정됨 (정적)
적용 위치	주로 Self-Attention 계층에서 사용	모든 신경망 계층 (Linear, Conv 등)에서 사용
예시	"The cat sat on the mat" → "sat"이 "cat"에 높은 attention	Layer1 뉴런 → Layer2 뉴런 사이의 연결 가중치 0.62

✅ 예시로 쉽게 이해하기
🔹 Attention 예시:
문장: "The cat sat on the mat"
•	단어 "sat"은 모델에게 물어봄:
"내가 누구와 관련이 있지?"
•	Attention Score 계산 결과:
o	"cat": 0.8
o	"mat": 0.1
o	"on": 0.05
o	…
→ 이 0.8, 0.1 같은 값이 Attention (어텐션 점수)
→ 이건 입력마다 다르게 계산되며, 현재 문맥에 따라 바뀜.

🔹 Weight 예시:
Linear Layer: y = Wx + b

•  여기서 W는 학습을 통해 얻어진 고정된 가중치 행렬(Weight)
•  예:
W = [[0.4, -0.3],
     [0.7,  0.2]]
→ 이 Weight는 입력 문장과 무관하게 고정된 값.
→ 모델이 기억한 정보

🔁 두 개념을 연결해보면
관점	설명
Attention은	**지금 어떤 입력이 더 중요해 보이는가?**를 결정
Weight는	모델이 학습을 통해 기억한 중요한 연결

✅ 결론
질문	답변
둘 다 중요도를 나타내는 값인가요?	네, 하지만 역할과 시점이 다름.
Attention은 언제 계산되나요?	입력마다 실시간으로 계산됨.
Weight는 어떤 역할인가요?	모델 내부의 고정된 연결 강도로서, 학습된 지식.

✅ 요약
Attention = 모델이 “어디에 주의를 기울여야 할지” 스스로 정하게 해주는 기법
→ 인간이 문장에서 중요 단어를 파악하듯, AI도 그런 “주의 분배”를 하게 만드는 핵심 기술

## attention score
1️⃣ 한 문장 정의

**Attention score(어텐션 점수)**는

한 토큰이 다른 토큰을 얼마나 참고해야 하는지를 나타내는 “관련성 점수”이다.
attention scores는 학습 단계가 아니라 순전파(forward pass) 계산 과정에서 매번 생성되는 값이다.
attention scores는 “어떤 단어를 얼마나 참고할지” 결정하는 기준이다

2️⃣ 지금 무슨 상황인가?

문장 안에 이런 토큰들이 있다고 합시다:
The journey starts now

우리는 journey를 기준으로 생각합니다.

이때 모델은 묻습니다:

"journey가 의미를 이해하려면
다른 단어들 중 누구를 얼마나 참고해야 하지?"

이때 계산되는 숫자가 바로
attention score입니다.

3️⃣ 어떻게 계산하나?

기본 형태는:

attention score = Query · Key


즉,

현재 토큰의 Q (query)
다른 토큰들의 K (key)

를 내적(dot product) 합니다.

값이 크면:

→ 관련성 높음
→ 더 많이 참고

값이 작으면:

→ 덜 중요

4️⃣ 중요한 구분

Attention score는 아직 확률이 아닙니다.

그냥 "원시 점수(raw score)"입니다.

그 다음 단계에서:

softmax(attention scores)


를 거쳐야

→ attention weight (가중치, 확률 형태)

가 됩니다.

5️⃣ 전체 흐름

1) Q · K → attention score
2) softmax → attention weight
3) weight × V → 가중합
4) 결과 = context vector

6️⃣ 비유로 설명해보겠습니다

당신이 회의 중입니다.

1) 당신이 질문자 (Query)
2) 참석자들이 정보 보유 (Key)
3) 당신은 각 사람과의 관련성을 계산
4) 관련성 점수가 attention score

그 다음

관련성 높은 사람 말을 더 반영

그 최종 정리된 결론이 context vector입니다.

7️⃣ 당신 수준에서 반드시 봐야 할 본질

Attention score는

“의미 유사도”가 아니라
“현재 위치에서의 중요도 계산값”

입니다.

그리고 이 점수는 레이어마다 새로 계산됩니다.

## autoregressive
딥러닝, 특히 **언어 생성 모델(NLP)**에서 매우 중요한 개념.
✅ 먼저 아주 쉽게 정의하면:
Autoregressive = 이전에 만든 결과를 이용해 다음 결과를 예측하는 방식
🎯 한 마디로:
**"한 단어씩 순서대로, 앞에 생성한 단어들을 보고 다음 단어를 예측하는 방식"**.
🔁 예를 들면:
문장을 생성한다고 가정하면,
AI가 "I", "love"를 만들었어요. 그럼 다음엔? → 앞의 "I love"를 보고 "you"를 예측. 
즉,
•	1단계: "I"
•	2단계: "I" → "love"
•	3단계: "I love" → "you"
•	4단계: "I love you" → "so"
•	...
👉 이렇게 앞 단어들에 "의존"하면서 다음 단어를 순서대로 하나씩 만들어 나가는 방식이 바로 autoregressive.

LLM은 output 토큰을 계속 만들어내기 위해, 처음 입력한 prompt를 끝까지 계속 참고
✅ 핵심 개념: Auto-Regressive 모델
Transformer LLM(GPT 등)은 auto-regressive (자기회귀) 구조.
즉:
"지금까지 본 모든 토큰들(= 초기 프롬프트(모델에게 주는 초기 입력 문장) + 지금까지 만든 출력 토큰들)"을
매번 다시 참고하면서 다음 토큰을 생성
🔁 어떤 구조냐면
매 토큰을 만들 때마다 다음처럼 입력이 점점 늘어남:
단계	모델 입력	생성된 토큰
1	Prompt	"Dear"
2	Prompt + "Dear"	"Sarah"
3	Prompt + "Dear Sarah"	","
4	Prompt + "Dear Sarah,"	"I'm"
…	계속	계속
→ 이때 "Prompt"는 절대 사라지지 않고, 모든 단계에 포함 됨.
📌 왜 이렇게 하냐?
•	**프롬프트가 맥락(Context)**이기 때문.
•	모델은 **전체 문맥(= prompt + 지금까지의 출력)**을 봐야
적절한 다음 토큰을 생성할 수 있음.
예:
Prompt: "사과와 배 중에서 더 맛있는 과일은"
→ 모델은 이 문장이 없으면 “정답”을 만들 수 없음.
🧠 그래서 생기는 특징
현상	설명
✅ 문맥 유지	처음 입력한 프롬프트가 계속 유지되어 문장의 흐름이 자연스러움
⚠️ 속도 저하	길어질수록 모델이 참조할 토큰이 많아져 느려짐
🔁 입력 누적 증가	매 토큰마다 입력이 1토큰씩 늘어남 (처리량 증가)
✅ 결론
LLM은 output 토큰을 계속 만들어내기 위해,
처음에 입력한 prompt를 끝까지 계속 참고 함.
즉, prompt는 매 생성 단계마다 입력에 "계속 유지".

📊 Autoregressive vs Non-Autoregressive 모델 비교표
항목	Autoregressive	Non-Autoregressive
대표 모델	GPT, GPT-2, GPT-3, GPT-4, Transformer decoder	BERT, RoBERTa, ELECTRA, Transformer encoder
예측 방식	이전 단어를 기반으로 다음 단어를 순차적으로 생성	문장 전체를 한 번에 처리하며 마스킹된 부분을 병렬적으로 예측
토큰 생성 방식	순차적 생성 (왼쪽 → 오른쪽)	병렬 처리 (동시에 여러 위치 예측 가능)
학습 방식	Causal Language Modeling (CLM): `P(w₁)·P(w₂	w₁)·P(w₃
장점	자연스럽고 문맥 있는 텍스트 생성에 강함 (생성 능력 우수)	인코딩 성능 우수. 문장 이해, 분류, NER 등 이해 중심 태스크에 강함
단점	느린 생성 속도 (토큰을 한 개씩 예측해야 함), 병렬화 어려움	생성 작업에는 부적합, 양방향 구조가 생성 순서에 대한 정보 부족
사용 예시	ChatGPT, 텍스트 생성, 요약, 번역, 코드 생성 등	문장 분류, 질문 응답, 감정 분석, 문서 임베딩 등
Context 처리	한 방향 (일반적으로 왼쪽→오른쪽)	양방향 문맥 활용 가능

🔍 GPT와 BERT 비교 예시
항목	GPT (Autoregressive)	BERT (Non-Autoregressive)
구조	Transformer Decoder	Transformer Encoder
방향성	왼쪽 → 오른쪽	양방향(Bidirectional)
훈련 방식	Causal Language Modeling	Masked Language Modeling
입력 예시	"나는 밥을 먹고" → 다음 단어 예측	"나는 [MASK]을 먹고 있다" → MASK 예측
생성 능력	매우 강함	약함 (생성 목적이 아님)
대표 활용	챗봇, 생성형 AI	텍스트 분류, 질의응답, 임베딩 등

🧠 핵심 요약
•	Autoregressive 모델 (ex. GPT)
→ 생성에 특화, 순차적으로 단어 생성
→ 병렬화는 어려우나 자연스러운 언어 생성 가능
•	Non-Autoregressive 모델 (ex. BERT)
→ 이해 중심, 빠른 병렬처리 가능
→ 생성에는 적합하지 않음

## A.X Light
SK텔레콤이 개발한 한국어 LLM AI 모델. 에이닷 엑스(A.X)로 불리며, 25년 7월 현재 4.0 버전까지며, 표준과 경량 모델을 hugging face에 공개함. 알리바바 AI 모델 '큐원'(Qwen) 2.5에 한국어 데이터를 추가로 학습시킴.
A.X Light 모델 구조:
Qwen2Model(
  (embed_tokens): Embedding(102400, 3584)
  (layers): ModuleList(
    (0-27): 28 x Qwen2DecoderLayer(
      (self_attn): Qwen2Attention(
        (q_proj): Linear(in_features=3584, out_features=3584, bias=True)
        (k_proj): Linear(in_features=3584, out_features=512, bias=True)
        (v_proj): Linear(in_features=3584, out_features=512, bias=True)
        (o_proj): Linear(in_features=3584, out_features=3584, bias=False)
      )
      (mlp): Qwen2MLP(
        (gate_proj): Linear(in_features=3584, out_features=18944, bias=False)
        (up_proj): Linear(in_features=3584, out_features=18944, bias=False)
        (down_proj): Linear(in_features=18944, out_features=3584, bias=False)
        (act_fn): SiLU()
      )
      (input_layernorm): Qwen2RMSNorm((3584,), eps=1e-05)
      (post_attention_layernorm): Qwen2RMSNorm((3584,), eps=1e-05)
    )
  )
  (norm): Qwen2RMSNorm((3584,), eps=1e-05)
  (rotary_emb): Qwen2RotaryEmbedding()
)


## bag-of-words
자연어 처리(NLP)에서 아주 오래된 전통적인 문장 표현 방식.
✅ 한 줄 정의
**Bag-of-Words(BoW)**는
문장이나 문서를 “단어들의 출현 횟수”만으로 표현하는 방식. 단어의 순서는 무시하고, 어떤 단어가 몇 번 나왔는지만 기록.
🧠 왜 이름이 "Bag(자루)"일까?
•	단어의 순서를 고려하지 않기 때문.
•	마치 **단어들을 다 쏟아 넣고 섞은 “자루”**처럼 다룬다고 해서 붙은 이름
📦 예시로 이해하기
예시 문장 1:
"I love AI and I love Python"
등장 단어:
["I", "love", "AI", "and", "Python"]
→ 중복 제거 후 단어 사전 (Vocabulary):
["I", "love", "AI", "and", "Python"]
→ BoW 벡터:
[2, 2, 1, 1, 1]  
 (I:2, love:2, AI:1, and:1, Python:1)
📌 특징 요약
항목	설명
🔢 벡터 표현	단어의 등장 횟수만큼 숫자로 표현
🔄 순서 무시	단어가 어떤 순서로 나왔는지는 고려하지 않음
✅ 장점	구현 간단, 계산 빠름
❌ 단점	문맥(의미), 순서, 유사 단어 파악 불가능
💡 비교: Bag-of-Words vs Embedding
항목	Bag-of-Words	Word Embedding
의미 파악	❌ 불가능	✅ 가능
순서 반영	❌ 무시함	✅ (Embedding + 모델이 학습)
벡터 크기	단어 수만큼 커짐	고정 크기 (예: 300차원)
쓰임	고전적 텍스트 분류, 간단한 모델	딥러닝 기반 NLP 모델 (BERT, GPT 등)
✅ 정리
Bag-of-Words는 문서나 문장을 '단어의 개수'만으로 표현하는 아주 기초적인 텍스트 표현 방식. 간단한 분류 문제나 통계 기반 모델에서는 여전히 유용하지만,
문맥이나 의미를 반영할 수 없다는 한계가 있어 현대 NLP에서는 Embedding 기반 방식이 주로 사용됨.

## backpropagation (역전파)
1️⃣ Backpropagation(역전파, Backpropagation)이란?

**역전파(Backpropagation)**는

"모델이 틀린 만큼, 어느 부분이 얼마나 잘못했는지 거꾸로 계산해서 가중치(weight, 가중치)를 수정(업데이트)하는 방법" 입니다.

조금 더 기술적으로 말하면,

**손실 함수 (Loss function, 손실 함수)**가 얼마나 틀렸는지를 계산하고

그 오차를 각 **가중치 (weight, 가중치)**에 대해

기울기 (gradient, 기울기) 형태로 계산해서

가중치를 업데이트합니다.

2️⃣ 직관적 비유

당신이 딸에게 문제를 냈다고 가정해 봅시다.

딸이 답을 냄

틀림

어디서 실수했는지 분석

그 부분을 고쳐 줌

다시 풀게 함

이때 **“어디서 얼마나 틀렸는지 거꾸로 추적하는 과정”**이 바로 역전파입니다.

3️⃣ 신경망에서는 실제로 무슨 일이 벌어질까?

신경망 흐름은 이렇게 갑니다:

입력 → Linear(선형변환) → Activation(활성화함수) → ... → 출력


이 과정은 Forward propagation (순전파) 입니다.

그 다음:

출력값과 정답을 비교

오차 계산

오차를 뒤에서 앞으로 전파

각 weight가 얼마나 잘못 기여했는지 계산

이게 바로 Backpropagation (역전파) 입니다.

4️⃣ 핵심 개념 3가지
① Loss (손실)

모델이 얼마나 틀렸는지 숫자로 표현
예: Cross-Entropy Loss (교차 엔트로피 손실)

② Gradient (기울기)

각 weight가 loss에 얼마나 영향을 줬는지 계산한 값

③ Chain Rule (연쇄법칙, Chain Rule)

역전파는 사실상 미분의 연쇄법칙을 이용합니다.

왜냐하면 신경망은 이런 구조이기 때문입니다:

Loss → Layer3 → Layer2 → Layer1


각 레이어가 서로 연결되어 있어서
오차를 거꾸로 계산할 때 **연쇄법칙(Chain Rule)**을 사용합니다.

5️⃣ 왜 “역”전파일까?

Forward는 입력 → 출력 방향
Backprop은 출력 → 입력 방향

그래서 "거꾸로 전파"라고 부릅니다.

6️⃣ GPT 같은 LLM에서는?

GPT도 똑같습니다.

토큰 입력

Transformer block 통과

다음 토큰 예측

정답 토큰과 비교

Cross-Entropy Loss 계산

역전파로 수억~수십억 개의 weight 수정

이 과정을 수십억 번 반복하면서 모델이 언어를 학습합니다. (이 이유 때문에 성능 좋은 GPU와 전기 비용이 높게 나오는 이유)

## batch size
batch size는
👉 모델이 한 번의 학습 업데이트에서 동시에 보는 데이터 샘플의 개수.

batch size = 1
→ 데이터 1개 보고 바로 가중치(weight) 업데이트
batch size = 256
→ 데이터 256개를 묶어서 평균 낸 뒤 한 번에 업데이트

2️⃣ 직관부터 잡자 (비유)
📦 배치 = “의견 수집 규모”

batch size가 작다
→ 소수 의견 듣고 바로 결론
→ 판단은 빠르지만 흔들림 큼

batch size가 크다
→ 다수 의견 모아서 평균 내고 결론
→ 판단은 안정적이지만 느림

여기까지는 “클수록 좋아 보이죠?”
하지만 여기서 함정이 시작됨.

3️⃣ 왜 batch size가 클수록 항상 좋은 게 아니냐
❌ 이유 1: 메모리 한계 (현실 문제)

batch size ↑
→ GPU/NPU 메모리 사용량 선형 증가

LLM에서는 특히:

activation

gradient

optimizer state

이게 전부 batch size × context size에 비례

👉 하드웨어가 먼저 죽는다

❌ 이유 2: 일반화 성능(generalization)이 나빠질 수 있음

이게 진짜 중요한 이유다.

큰 batch:

gradient가 너무 매끈함

loss landscape의 **날카로운 최소값(sharp minimum)**으로 수렴하기 쉬움

작은 batch:

노이즈가 있음

**넓고 평평한 최소값(flat minimum)**에 도달하기 쉬움

결과:

큰 batch → 훈련 데이터에는 잘 맞음

하지만 새 데이터엔 약해질 수 있음

👉 “학습은 잘했는데, 실전에서 못 맞힌다”

❌ 이유 3: 업데이트가 너무 보수적이 됨

batch size가 크면:

한 번 업데이트하려면
→ 많은 데이터가 필요

학습 초반에
→ 방향 전환이 느림
→ 탐색(exploration) 능력 감소

특히:

초기 학습

데이터 분포가 복잡할 때

👉 큰 배치는 둔하다

❌ 이유 4: 학습 속도가 항상 빠르지 않다

많은 사람들이 착각함:

“batch size 키우면 병렬 처리돼서 빠르지 않나?”

현실:

iteration 수는 줄지만

수렴까지 필요한 step 수는 늘어나는 경우 많음

learning rate도 같이 조정해야 함

그래서:

batch size ↑
→ learning rate 튜닝 난이도 ↑

4️⃣ 그럼 작은 batch가 항상 좋냐?

아니요. 이것도 극단은 위험.

batch size가 너무 작으면:

gradient 노이즈 과다
loss가 요동침
학습 불안정
수렴 느림

5️⃣ 그래서 결론은?

👉 **batch size는 트레이드오프(tradeoff)**다.

기준			작은 batch		큰 batch
메모리		👍 적음			👎 큼
안정성		👎 불안정		👍 안정
일반화		👍 좋을 가능성	👎 나빠질 수
탐색 능력		👍 좋음			👎 둔함

“항상 좋은 값”은 없다.
“문제·데이터·하드웨어에 맞는 값”만 있다.

## BERTopic
BERT(Bidirectional Encoder Representations from Transformers) 기반의 문장 임베딩(Sentence Embedding)을 활용하여 **토픽 모델링(Topic Modeling)**을 수행하는 방법. 즉, 단어 수준이 아니라 **문장이나 문서 전체의 의미(semantic meaning)**를 반영해 주제를 분류.

🔧 작동 원리 (단계별 흐름)
단계	설명
1단계	각 문장을 BERT 또는 Sentence-BERT로 임베딩 벡터화 (의미를 숫자 벡터로 변환)
2단계	그 벡터들을 UMAP (차원 축소) 알고리즘으로 시각화 가능한 저차원으로 줄임
3단계	HDBSCAN 알고리즘으로 의미적으로 가까운 문장들을 군집화(Clustering)
4단계	각 군집 내의 핵심 단어들을 추출하여 **주제(Topic)**로 명명 (c-TF-IDF)

💡 예시
입력:
- "Apple released a new iPhone"
- "Samsung Galaxy S22 is popular"
- "Barcelona won the game"
- "Real Madrid lost to Chelsea"

→ BERTopic 결과:
- Topic 1: 스마트폰 관련 (apple, samsung, phone...)
- Topic 2: 축구 관련 (Barcelona, game, Madrid...)

📦 주요 구성 요소
요소	설명
BERT	텍스트 의미를 벡터로 변환 (임베딩)
UMAP	고차원 벡터를 저차원으로 줄임 (시각화/군집 용도)
HDBSCAN	밀도 기반 클러스터링
TF-IDF	각 클러스터 내 핵심 단어 추출

✅ BERTopic의 장점과 단점
장점	단점
의미 기반의 토픽 분류가 매우 정교함	학습 시간이 오래 걸릴 수 있음
문장 단위 또는 긴 문서에도 적합	GPU가 없으면 느릴 수 있음
클러스터 수를 자동으로 조절 가능 (HDBSCAN)	설치 및 환경 구성이 약간 복잡할 수 있음

🔧 Python 간단 예제
from bertopic import BERTopic

docs = [
    "I love watching football games",
    "Messi scored a goal for PSG",
    "I enjoy Italian food like pizza and pasta",
    "The iPhone 14 was just released by Apple"
]

topic_model = BERTopic()
topics, probs = topic_model.fit_transform(docs)

topic_model.get_topic_info()


## Binary Sentiment Classification
자연어 처리(NLP)에서 매우 많이 사용되는 기본적인 분류 작업.

✅ 한 줄 정의
Binary Sentiment Classification은
👉 문장의 감정을 '긍정(positive)' 또는 '부정(negative)' 중 하나로 분류하는 작업

🔍 용어 분해해서 이해해 볼게요
용어	의미
Binary (이진)	두 가지 범주 (예: 0 또는 1, 참 또는 거짓, 긍정 또는 부정)
Sentiment (감정)	문장이 담고 있는 감정, 분위기 (positive/negative)
Classification (분류)	해당 문장을 어떤 카테고리로 나눌지 결정하는 작업

🔸 예시
입력 문장	분류 결과
"이 영화 정말 감동적이었어요!"	긍정 (Positive)
"돈 아까운 영화였음."	부정 (Negative)

※ 보통은 숫자로 라벨링:
•	0 = 부정
•	1 = 긍정

🧠 어디에 쓰일까?
분야	적용 예시
영화 리뷰 분석	긍정/부정 평가 자동 분류
상품 리뷰 분석	고객 감정 파악 (예: 쇼핑몰)
SNS 분석	트윗이나 댓글의 여론 탐지
고객센터	불만 or 칭찬 감정 분류 → 대응 자동화

🔄 확장 가능
•	Binary sentiment classification은 가장 단순한 형태이고,
실제로는 multi-class sentiment classification도 존재
등급	설명
1	매우 부정적
2	부정적
3	중립
4	긍정적
5	매우 긍정적
 ✅ 요약
항목	내용
용어	Binary Sentiment Classification
의미	문장의 감정을 긍정/부정 중 하나로 분류
데이터 예	영화 리뷰, 제품 리뷰, 댓글
출력	보통 0 (부정), 1 (긍정)
사용 모델	BERT, GPT, Logistic Regression 등

## byte pair encoding (GPT 모델)
🧠 쉬운 설명:
1.	기본 개념
우리가 문장을 쓸 때 단어 사이에 공백(space)이 있음
그런데 GPT 같은 모델은 공백을 그냥 " " 따로 토큰으로 두지 않고, "특수 표시"를 활용해서 처리.

2.	예시로 설명
apologizing quickly

이 문장은 토큰으로 나눌 때 이렇게 될 수 있음 (실제 예는 다를 수 있음):

' apolo', 'gizing', ' quickly'

여기서 'gizing'처럼 단어의 중간이나 끝에 해당하는 부분은 앞 단어와 붙어 있다는 표시로 특수한 문자 (보통 ▁ ← 이건 Unicode U+2581)를 앞에 붙이지 않음.
대신, ' quickly'처럼 새로운 단어의 시작은 특수 문자 ▁로 시작됨.
•	'▁quickly' → 공백 다음에 오는 새 단어
•	'gizing' → 앞 단어에서 이어지는 조각 (공백 없음)

3.	요점 정리
•	특수 문자(예: ▁)가 붙어 있으면: 공백 후 새 단어 시작
•	특수 문자가 없으면: 앞 단어의 일부, 즉 붙어 있음
•	즉, 공백을 직접 저장하지 않고, 공백의 존재 유무를 특수 문자로 표현.

Byte Pair Encoding에서 Pair의 뉘앙스:
BPE에서 “Pair”는 그냥 둘이 아니라, ‘자주 같이 붙어 다니는 둘’이라는 뉘앙스. 왜 굳이 Pair인가? BPE는 처음부터 “단어”를 보지 않는다. 아주 원시적인 상태에서 시작. 모든 텍스트를 byte 단위로 쪼갬. BPE는 이 질문을 계속 반복한다. “지금 토큰 시퀀스에서 서로 이웃해 있으면서 가장 자주 같이 등장하는 두 개는 뭐지?” 이게 바로 Pair다.
h + e → "he"   (아주 자주 같이 나옴)
t + h → "th"
i + n → "in"
이런 인접한 2개 묶음을 하나의 새 토큰으로 **병합(merge)**한다. 왜 하필 2개(pair)냐? 여기엔 아주 현실적인 이유가 있다. 계산 가능성(2개 조합은 경우의 수가 관리 가능. 3개, 4개부터는 폭발한다)과 점진적 학습(2개씩 묶다 보면, 자연스럽게 문자 → 부분 단어 → 단어로 성장한다. 예. t + h => th, th + e => the)

“Pair”의 뉘앙스 요약
관점		의미
기술적	인접한 두 토큰
통계적	가장 자주 함께 등장
전략적	가장 먼저 묶을 가치가 있는 최소 단위
언어적	의미 단위로 성장할 씨앗

•  Byte: 시작 단위가 byte
•  Pair: 항상 두 개씩 묶는다
•  Encoding: 그 병합 규칙을 코드로 만든다

이 BPE 때문에 사용자의 오타도 극복 할 수 있는건가? 결론부터 말하면 “부분적으로는 그렇다.” BPE 덕분에 오타에 ‘죽지 않는다’는 게 맞는 표현이다.

왜 BPE가 오타에 강한가? 오타가 나도 토큰화는 된다.
단어 기반 토크나이저였다면:
•	palace ❌
•	palcae → vocabulary에 없음 → <unk>
BPE에서는:
palcae
→ pal | ca | e
또는
→ p | a | l | c | a | e
즉,
👉 오타가 있어도 입력 자체가 붕괴되지 않는다.

“완전한 이해”는 아니지만 “유사성”은 남는다

오타는 보통:
글자 하나 빠짐
순서 바뀜
하나 더 들어감

이 경우:
상당수 subword / byte 토큰이 원래 단어와 겹친다

그래서 모델은:

“이상하긴 한데” “뭔 말인지는 얼추 알겠다” 상태로 처리할 수 있다.

## Causal Attention (커절 어텐션; 인과 어텐션)
Causal Attention은 Self-attention의 한 종류입니다.
차이는 미래를 못 본다는 것입니다.

즉,

현재 토큰은 과거(이전) 토큰만 볼 수 있다.

왜 이런 제약이 필요한가?

LLM이 하는 일은?

다음 단어 예측 (Next-token prediction)

예를 들어:

"I am going to"

모델은 다음 단어를 예측해야 합니다.

그런데 만약 학습 중에
미래 단어를 본다면?

"I am going to school"

이미 school을 봤다면
예측이 의미가 없습니다.

그래서 **미래를 가리는 마스크(mask)**를 씌웁니다.

구조적으로 어떻게 구현하나?

Self-attention의 attention score 행렬에

Upper triangular mask('0' 부분)를 씌웁니다.

즉:

[ x 0 0 0 ]
[ x x 0 0 ]
[ x x x 0 ]
[ x x x x ]


위쪽(미래)은 0 또는 -∞ 처리합니다.

이걸 **Causal Mask (코절 마스크)**라고 합니다.

Self-Attention (셀프 어텐션)
Self-Attention은
입력 문장 안의 모든 토큰(token)이 서로를 참고(reference)하는 구조입니다.

즉,

한 단어가 문장 안의 모든 다른 단어를 보고 중요도를 계산한다.

예시

문장:

"The cat sat on the mat"

“sat”이 문맥을 이해하려면

cat
mat
the

모두를 참고할 수 있습니다.

Self-attention은 양방향(Bidirectional) 입니다.
즉 미래 단어도 볼 수 있습니다.

특징

전체 문장을 동시에 본다
병렬 계산 가능 (RNN과 다름)
BERT 같은 모델이 사용

Self-Attention vs Causal Attention 차이
| 구분       | Self-Attention | Causal Attention    |
| -------- | -------------- | ------------------- |
| 미래 토큰 참조 | 가능             | 불가능                 |
| 방향성      | 양방향            | 단방향 (Left-to-right) |
| 대표 모델    | BERT           | GPT                 |
| 목적       | 문맥 이해          | 텍스트 생성              |

4️⃣ 목적의 차이
Self-attention 목적

→ 문장 전체 의미 이해
→ 분류, 질의응답, 감성분석 등

Causal attention 목적

→ 다음 단어 생성
→ 텍스트 생성, LLM 대화 모델

GPT 계열은 전부:

Masked Self-Attention = Causal Attention

을 사용합니다.

즉 GPT의 모든 블록은
Self-attention이지만
Causal mask가 추가된 형태입니다.

## Chain of Thought (COT)
Chain of Thought는 AI가 답을 바로 찍지 않고, 중간 사고 단계를 거쳐 결론에 도달하는 방식입니다.

수학으로 비유하면 이렇습니다.

그냥 답만 말하는 방식: 27 × 14 = 378

Chain of Thought 방식:
27 × 14
= 27 × (10 + 4)
= 270 + 108
= 378

즉, 문제를 잘게 나누고, 중간 과정을 따라가며, 마지막 답을 만드는 흐름입니다.

왜 수학 문제에서 중요하냐
수학은 정답만 맞는 게임이 아니라, 문제를 어떻게 분해하느냐가 핵심이기 때문입니다.
특히 다음 상황에서 Chain of Thought가 큰 힘을 발휘합니다.

여러 단계를 거쳐야 할 때
한 번에 계산하면 실수하기 쉽습니다.

조건이 숨어 있을 때
문제 문장을 읽고 어떤 식으로 바꿔야 하는지 판단해야 합니다.

논리 검증이 필요할 때
중간 과정을 보면 어디서 틀렸는지 찾을 수 있습니다.

예를 들어 이런 차이가 있습니다.

문제: 사과가 12개 있고 3명에게 똑같이 나누면 1명당 몇 개?

답만 내는 방식: 4

CoT 방식:
12개를 3명에게 똑같이 나눈다
→ 12 ÷ 3
→ 4개

이건 쉬운 예시지만, 복잡한 문제일수록 이런 사고 흐름이 훨씬 중요해집니다.

하지만 냉정하게 봐야 할 점도 있습니다.
Chain of Thought가 있다고 해서 항상 맞는 것은 아닙니다.

왜냐하면:

중간 단계가 그럴듯해도 틀릴 수 있고

계산은 맞는데 문제 해석이 틀릴 수 있고

불필요하게 길어지면 오히려 핵심을 놓칠 수 있기 때문입니다

그래서 진짜 중요한 건 “길게 생각하기”가 아니라,
“올바른 순서로 생각하기”입니다.

실전에서 좋은 Chain of Thought는 보통 이런 구조를 가집니다.

문제에서 무엇을 묻는지 확인

주어진 조건 정리

식이나 규칙으로 변환

단계별 계산

마지막에 답 검산

수학 공부 관점에서 정리하면:

초급: 식을 바로 세우는 습관

중급: 왜 그 식이 나오는지 설명

고급: 다른 풀이와 비교하고 검산

AI에서도 비슷합니다.
수학 문제를 잘 푸는 모델은 단순 암기보다, 문제를 단계적으로 분해하는 능력이 더 중요합니다.

한 가지 더 구분해 드리겠습니다.

Chain of Thought: 중간 사고 과정을 따라가며 푸는 방식

추론 능력(reasoning): 그 사고 과정을 제대로 설계하고 적용하는 능력

즉, Chain of Thought는 “형태”이고,
추론 능력은 “실력”입니다.

## Checkpoint (체크포인트)
학습 도중 모델의 상태를 저장한 파일. 문제가 생겼을 때 이어서 학습하거나, 중간 결과를 다시 불러올 수 있게 함. 

## classification
"자연어 처리(NLP)에서 흔한 작업 중 하나는 분류(classification). 

🔶 Classification (분류)란?
주어진 문장이나 텍스트를 미리 정해진 여러 카테고리 중 하나로 나누는 작업
📌 즉, “이 문장은 어떤 종류인가요?”, “이 문서의 감정은 무엇인가요?” 같은 질문에 답하는 것!

🔷 예시로 쉽게 이해하기
입력 텍스트	분류 결과
(Classification Output)	설명
"이 영화 정말 재미있었어!"	긍정 (Positive)	감성 분류 (Sentiment Classification)
"이메일을 지금 확인하세요!"	스팸 (Spam)	스팸 필터링
"문의사항이 있습니다."	고객지원 (Support)	고객 문의 유형 분류
"너 오늘 몇 시에 와?"	일상 대화 (Casual)	대화 분류 (Intent Classification)

🔷 Classification이 중요한 이유
자연어는 컴퓨터에게는 문자열일 뿐이에요.
하지만 우리가 컴퓨터에게 "이 문장이 어떤 의미인지" 알려주려면,
그걸 **범주(category)**로 바꿔줘야 함. 이게 바로 분류 작업

🔶 Classification은 어디서 쓰이냐
분야	적용 예
고객센터	고객 문의 자동 분류 (결제, 반품, 배송 등)
SNS 감정 분석	긍정/부정/중립
이메일 필터	스팸 / 일반 메일 분류
챗봇	사용자의 의도(Intent) 파악

🔷 기술적으로는?
•	모델은 입력 텍스트를 보고 확률 분포를 예측.
예:
"이 영화 별로야" → [Positive: 5%, Neutral: 10%, Negative: 85%]
그리고 가장 높은 확률을 가진 클래스를 예측 결과로 선택

✅ 요약
항목	설명
용어	Classification (분류)
정의	텍스트를 미리 정해진 카테고리 중 하나로 나누는 작업
NLP 내 위치	가장 기본적이면서 핵심적인 작업 중 하나
예시	감성 분석, 스팸 필터링, 고객 문의 분류 등

## chunk overlap
1️⃣ RAG의 Chunk Overlap

RAG에서 문서를 쪼갤 때 이런 식으로 자릅니다.

예:

[문장1 문장2 문장3 문장4]
[문장3 문장4 문장5 문장6]

앞 chunk의 끝부분을 다음 chunk에 겹쳐서(overlap) 넣습니다.

🎯 목적

문맥 손실 방지

검색(embedding) 정확도 향상

의미 단절 방지

즉, VectorDB 검색 품질을 높이기 위한 전처리 전략입니다.
LLM 내부 동작과는 무관합니다.

2️⃣ LLM의 Sliding Window

LLM에서 말하는 sliding window는 보통 두 가지 맥락에서 나옵니다.

(1) Context Window 초과 처리

입력이 너무 길면:

[토큰1~4096]
[토큰1024~5120]

이렇게 토큰 단위로 밀면서 처리합니다.

(2) Long Context Attention 최적화

일부 모델은 전체 토큰을 다 보지 않고
근처 토큰만 보는 제한 attention을 사용합니다.

🎯 목적

메모리 절약

계산량 감소

긴 문맥 처리

즉, 모델 내부 추론 방식입니다.

📌 핵심 차이 정리
| 구분    | Chunk Overlap        | Sliding Window  |
| ----- | -------------------- | --------------- |
| 적용 위치 | RAG 전처리              | LLM 내부          |
| 단위    | 텍스트 조각               | 토큰              |
| 목적    | 검색 정확도               | 계산 효율 / 긴 문맥 처리 |
| 관련 영역 | Embedding / VectorDB | Attention 메커니즘  |

🧠 그런데… 정말 완전히 다른 개념일까요?

전략적 관점에서 보면:

둘 다 “문맥 단절을 막기 위해 겹치게 본다”는 사고방식은 같습니다.

하지만

Chunk overlap은 검색 단계 문제 해결

Sliding window는 추론 계산 문제 해결

입니다.

## Clang
C, C++, Objective-C, Objective-C++ 같은 언어를 기계가 이해할 수 있는 코드로 바꿔 주는 **컴파일러 프론트엔드(Compiler Frontend)**

1. Clang의 정체
•	LLVM 프로젝트의 일부 → LLVM 백엔드(Backend)와 함께 쓰여, 소스 → LLVM IR → 머신 코드로 이어집니다.
•	역할:
1.	소스코드 읽기(Parsing) – 사람이 쓴 C/C++ 코드의 문법을 확인.
2.	중간 표현(IR) 생성 – LLVM IR 같은 형태로 변환.
3.	에러/경고 메시지 제공 – 가독성 좋은 오류 메시지로 유명.

2. 왜 Clang이 특별하죠?
•	빠른 컴파일 속도 – GCC에 비해 컴파일 속도가 빠른 편.
•	깔끔한 에러 메시지 – 오류 위치와 해결 힌트를 친절하게 표시.
•	모듈식 구조(Modular) – 분석 도구, 코드 자동 변환 도구로 쉽게 확장 가능.
•	표준 준수 – C/C++ 표준에 맞춘 구현.
•	크로스 플랫폼 – Windows, macOS, Linux 등 어디서든 동작.

3. Clang과 LLVM 관계
•	Clang: 프론트엔드 → 소스코드 해석 및 LLVM IR 생성.
•	LLVM: 백엔드 → IR을 각 하드웨어에 맞는 머신 코드로 변환.
•	비유:
o	Clang = “통역사” (한국어 소설 → 공용어(중간어)로 번역)
o	LLVM = “최종 번역사” (공용어 → 각 나라 언어(하드웨어 명령어)로 번역)

## clustering
데이터를 유사도(similarity)가 높은 그룹(클러스터)으로 묶는 작업

## cmake
🎯 한 줄 정의
**CMake(씨메이크)**는
💡 “컴퓨터가 프로그램을 만들기 위한 설계도(레고 설명서)를 자동으로 만드는 도구” 

🧱 예시로 쉽게 이해해 보기
🎮 1. 게임 만들기라고 생각해 보기
•	당신이 게임을 만들고 싶음.
•	그런데 게임은 여러 파일과 그림, 코드, 사운드가 필요.
•	그걸 그냥 막 조립하면? → 에러 나고, 돌아가지 않음.

🧰 2. 그래서 "설치 설명서"가 필요
•	레고를 만들 때 책자가 있듯이,
•	컴퓨터 프로그램도 “어떤 순서로, 어떤 파일을, 어떻게 조립할지” 설명서가 필요

🛠️ 3. 이 설치 설명서를 자동으로 만들어 주는 게 CMake.
•	CMake는 CMakeLists.txt라는 파일을 보고:
o	어떤 코드들을 빌드할지,
o	어떤 컴파일러(예: gcc, clang)를 쓸지,
o	라이브러리(예: numpy, ttnn)를 어디서 찾을지
등을 분석해서 자동으로 Makefile이나 빌드 스크립트를 만들어 줌.

📌 정리 요약
항목	설명
역할	소스 코드 → 실행 프로그램으로 만들기 위한 조립 설명서를 자동 생성
사용하는 이유	직접 설정하기 복잡한 컴파일 과정을 자동화하기 위해
입력	CMakeLists.txt라는 설정 파일
출력	Makefile 또는 Ninja 빌드 파일
장점	여러 운영체제에서 동일한 방식으로 프로그램 빌드 가능

🧑‍🏫 예를 들어…
Tenstorrent tt-metal을 설치할 때도 내부에 이런 명령이 돌아감:
cmake -B build -DCMAKE_BUILD_TYPE=Release
ninja

## Confusion matrix
**분류 모델의 성능을 분석하기 위해 사용하는 표(table)**로, 모델이 얼마나 정확하게 각 클래스를 예측했는지를 한눈에 시각적으로 보여주는 도구.

✅ 한 줄 정의
Confusion matrix는 모델의 예측 결과를 **정답(실제 값)과 비교해서 정리한 행렬(2차원 표)**
🧠 Confusion Matrix가 왜 중요할까?
단순히 "정확도(Accuracy)"만 보면 잘못된 판단을 할 수 있음.
예를 들어 **치명적인 오류(FP 또는 FN)**가 많을 수도 있기 때문.
예시:
•	의료 진단에서
o	FN(암인데 암이 아니라고 함) → 매우 치명적
o	FP(정상인데 암이라고 함) → 불필요한 검사
➡ 그래서 confusion matrix를 통해 오류의 유형까지 파악

## context vector
**Context vector(컨텍스트 벡터)**는
👉 현재 단어를 이해하기 위해 모델이 참고한 문맥 정보를 하나의 벡터로 압축한 것입니다.

즉,

"이 단어가 지금 무슨 의미인지"를
주변 단어들을 종합해서 숫자 벡터로 만든 것

입니다.

2️⃣ 왜 필요한가?

예를 들어 보죠.

I went to the bank.

여기서 bank는

은행 (bank)

강둑 (river bank)

둘 중 무엇일까요?

이건 **주변 단어(context)**를 봐야 결정됩니다.

그래서 모델은

앞뒤 단어들을 모두 확인하고

중요한 단어에 더 높은 가중치(attention weight)를 주고

그 정보를 합쳐 하나의 벡터로 만듭니다

그 결과물이 바로 context vector입니다.

3️⃣ 수식적으로 보면 (Transformer 기준)

Transformer에서는 이렇게 만들어집니다:
Attention(Q, K, V) = softmax(QKᵀ / √d) V

여기서 결과로 나오는 값이 바로
👉 context vector

Q = Query (현재 단어가 묻는 질문)
K = Key (각 단어의 특징)
V = Value (각 단어의 실제 정보)

즉,

현재 단어가 다른 단어들을 얼마나 참고해야 하는지를 계산해서
그 정보를 가중합(weighted sum)한 결과

입니다.

4️⃣ 쉽게 비유해봅시다

당신이 회의 중이라고 생각해봅시다.

어떤 사람이 질문을 던짐 (Query)
회의 참석자들이 각자 정보 보유 (Key, Value)
질문과 관련 있는 사람들 말만 더 집중해서 듣고
그 내용을 종합해서 하나의 결론을 냄

그 결론이 context vector입니다.

5️⃣ 중요한 포인트

Context vector는

✔ 고정된 단어 의미가 아님
✔ 문장마다 계속 바뀜
✔ 위치마다 다름
✔ Attention 결과물임

즉,

token embedding은 "기본 의미"
context vector는 "문맥 반영된 의미"

입니다.

## convolution
신호 처리(signal processing), 영상 처리(image processing), AI 딥러닝(CNN, Convolutional Neural Network) 에서 핵심적으로 쓰이는 연산
쉽게 말하면, 큰 데이터(예: 이미지) 위에 작은 필터(filter, kernel) 를 올려서 곱하고 더하는 연산을 통해 특정한 패턴을 추출하는 방법
 
👉 이 필터는 세로 방향(edge detection, 수직 경계 검출)에 강하게 반응. 커널을 이미지 위에 겹치고, 원소별 곱 후 합산을 하면, 새로운 출력 값(특징 맵, feature map)이 생김. 필터를 한 칸씩 움직이며 이 과정을 반복하면 최종 결과가 됨.
한 칸씩 움직인다의 의미: 커널(filter, kernel)을 이미지 위에서 가로/세로로 1 pixel씩 이동시키며 겹쳐 보는 것.
예시 (stride = 1, 3×3 필터, 5×5 이미지)
•	첫 위치: 이미지 왼쪽 상단 3×3 영역과 필터 곱
•	그다음 위치: 오른쪽으로 1 pixel 이동 → 새로운 3×3 영역과 필터 곱
•	계속 오른쪽 끝까지 가면, 다시 한 줄 아래로 내려와서 같은 방식 반복
이 과정을 통해서 만들어진 출력이 바로 feature map (특징 맵)

딥러닝에서의 역할
•	CNN (Convolutional Neural Networks) 은 이미지를 그대로 fully-connected layer에 넣는 대신, 여러 개의 convolution 필터를 통해 특징(feature) 을 뽑아냄.
o	초기 레이어: 선(edge), 색(color) 등 단순한 패턴
o	중간 레이어: 눈, 코, 입 같은 부분적 특징
o	깊은 레이어: 전체 얼굴, 자동차 등 복합적인 특징
즉, Convolution = 데이터에서 중요한 패턴을 자동으로 찾아내는 확대경 같은 역할

Matrix Multiplication과의 차이
•	Matrix Multiplication(행렬 곱): 전체 행렬을 한 번에 곱하는 전역(global) 연산
•	Convolution(합성곱): 작은 커널을 이동하면서 국소(local) 패턴을 추출하는 연산

•	Convolution = 작은 필터(커널)를 슬라이딩 시키면서 곱하고 더해 패턴을 추출하는 연산
•	AI에서 이미지 인식, 음성 처리, 자연어 처리에도 활용됨

## Convolutional Neural Network (CNN)
✅ CNN이란?
이미지나 영상처럼 2차원 정보가 담긴 데이터를 처리하기 아주 잘 만든 AI 구조.
원래 이름	뜻
Convolutional → 	합성곱 (이미지를 스캔하는 연산)
Neural → 	신경망 (AI 모델을 이루는 기본 구조)
Network → 	네트워크 (층이 겹겹이 연결된 구조)

📸 쉽게 설명: 이미지 인식 예시
예를 들어, 🐱 고양이 사진을 넣으면 "이건 고양이야!" 라고 AI가 말해주는 기능이 있을 때,
그 속에서 CNN이 핵심 역할을 함. CNN은 이미지를 조각조각 잘라서, 각 조각마다 **눈, 코, 귀 같은 특징(패턴)**을 찾음.

🔍 핵심 개념: Convolution (합성곱)
이미지를 작은 창(window)처럼 쓱쓱 훑으면서, **특징(feature)**을 찾아내는 연산. 이걸 수학적으로 표현한 게 Convolution (합성곱) 이고, CNN에서는 이걸 **여러 층(Layer)**에 걸쳐 반복해서 더 복잡한 특징을 찾아냄. 

🧠 CNN의 구조
층 종류	역할
🧼 Convolution Layer	이미지를 훑으며 특징 추출
🧹 Pooling Layer	정보 요약, 크기 줄이기
🧠 Fully Connected Layer	추출한 특징을 종합해서 “이건 고양이다” 같은 판단 내림
🔁 여러 층 반복	복잡한 특징도 점점 더 잘 파악하게 됨

🤖 CNN 쓰임새
분야	예시
🖼️ 이미지 분류	고양이 vs 강아지 vs 자동차 등
🔍 객체 탐지	사진에서 사람 얼굴이나 번호판 위치 찾기
🧠 의료 영상 분석	CT, MRI 사진에서 병변 찾기
🛑 자율주행	도로, 신호등, 차선 인식
🎨 스타일 변환	이미지 필터, 스타일 입히기 등

✅ 요약
질문	답변
CNN이 뭔가?	이미지를 잘 처리하도록 만들어진 AI 구조. 합성곱이라는 방법으로 특징을 찾아냄.
왜 중요해요?	사진, 영상, 의료 영상, 자율주행 등 시각 정보 처리에 최적화돼 있기 때문.
Tenstorrent에서는?	CNN 모델을 Tenstorrent NPU에서 효율적으로 실행해보는 예제로 Yari 프레임워크를 제공.

## corpus’s vocabulary
용어	설명
Corpus (말뭉치)	여러 문서들로 구성된 텍스트 데이터 집합. 예: 뉴스 기사 1,000개, 논문 요약 500개 등
Vocabulary (어휘 집합)	**Corpus 전체에서 등장한 고유 단어들(unique words)**의 목록입니다

즉,
Corpus’s vocabulary는
전체 문서에서 등장한 단어들을 중복 없이 정리한 목록

📌 예시
문서 1: I love pizza and pasta  
문서 2: Pasta is delicious and popular  
→ Corpus = [문서 1, 문서 2]
→ Vocabulary = {"I", "love", "pizza", "and", "pasta", "is", "delicious", "popular"}
✔️ 중복 단어는 하나로 처리하고,
✔️ 단어의 순서나 빈도는 고려하지 않음 (일단 vocabulary에서는)

🧠 왜 중요할까?
•	**텍스트 벡터화(vectorization)**를 할 때 기준이 됨
(예: 단어 하나마다 번호 부여하거나, one-hot, TF-IDF, BOW 등 처리할 때)
•	단어 임베딩(embedding) 할 때도 vocabulary가 필요함
•	LDA나 BERTopic 같은 토픽 모델링에서는
vocabulary를 기준으로 문서-단어 행렬을 만들어 주제 분석

## cosine annealing (코사인 어닐링)
1️⃣ Cosine Annealing(코사인 어닐링)이란?

**Cosine Annealing(코사인 어닐링)**은
학습률(learning rate)을 코사인 함수 모양으로 서서히 줄이는 방식입니다.

핵심은 이것입니다:

학습률을 갑자기 떨어뜨리는 게 아니라
부드럽게, 점점, 매끄럽게 감소시킨다.

2️⃣ 왜 그냥 선형 감소(linear decay)로 안 할까?

질문 하나 드리겠습니다.

학습이 후반부에 들어갔을 때:

모델은 이미 좋은 지점 근처에 있음

미세 조정이 중요

너무 급격한 학습률 감소는 오히려 비효율

이때 부드러운 감소 곡선이 더 안정적입니다.

Cosine 함수는 이런 모양입니다:
LR
│\
│ \
│  \
│   \
│    \__
└──────── Step

LR=21​⋅base_lr⋅(1+cos(Tt​π))

t = 현재 step

T = 전체 step

3️⃣ 직관적으로 이해해 봅시다

Cosine Annealing은 이런 느낌입니다:

처음 → 빠르게 감소

중간 → 완만하게 감소

끝 → 거의 0에 부드럽게 접근

즉, 마지막에 급정지하지 않는다는 것이 핵심입니다.

4️⃣ Warmup + Cosine 조합이 왜 많이 쓰일까?

LLM 학습에서 거의 표준 조합이:

Warmup → Cosine Annealing

그래프는 이런 형태입니다:
LR
│     /\
│    /  \
│   /    \
│__/      \__
└──────────── Step

초반: warmup (안정화)

후반: cosine decay (부드러운 수렴)

GPT, LLaMA, ViT 등에서 매우 흔하게 사용됩니다.

5️⃣ Annealing(어닐링)이란 단어의 의미

Annealing은 원래 **금속 열처리(열을 올렸다가 천천히 식히는 과정)**에서 나온 말입니다.

왜 이 이름을 썼을까요?

학습도 비슷합니다:

초반엔 크게 움직임
점점 움직임을 줄임
마지막엔 안정된 상태로 고정

## CoT (Chain-of-Thought, 사고의 연쇄)
정의: 복잡한 문제를 풀 때 언어 모델이 중간 추론 단계들을 텍스트로 표현하면서 답을 도출하는 기법.
예시 (수학 문제)
Q: 철수에게 사과 3개가 있고, 민수가 2개를 더 주면 총 몇 개가 될까요?

o	CoT 없는 답변: “5개”
o	CoT 있는 답변:
1.	철수는 처음에 사과 3개를 가지고 있다.
2.	민수가 사과 2개를 더 준다.
3.	따라서 총 3 + 2 = 5개가 된다.
→ 최종 답: “5개”
👉 이렇게 사고 과정을 단계적으로 적어가며 답을 찾는 것이 CoT입니다. 이렇게 하면중간 과정(thinking steps) 을 거치면서 논리적 추론(logical reasoning), 수학적 계산, 복잡한 질문 분석 등이 훨씬 더 정확해짐.

## cross entropy loss (교차 엔트로피 손실)
1️⃣ 한 줄 정의

Cross Entropy Loss는
“모델이 얼마나 틀리게 예측했는지를 측정하는 벌점 점수”입니다.

정확히 말하면:

모델이 정답에 얼마나 높은 확률을 줬는지를 수치로 평가하는 함수

2️⃣ 먼저 질문 하나 드리겠습니다

모델이 아래처럼 예측했다고 해보죠.

정답: "cat"

모델 출력 확률:

cat: 0.9

dog: 0.05

car: 0.05

이 경우와

cat: 0.4

dog: 0.3

car: 0.3

이 경우,
어느 쪽이 더 “잘 배운 모델”이라고 느껴지십니까?

👉 직관적으로 첫 번째죠.

그 차이를 수학적으로 벌점화한 것이 cross entropy입니다.

3️⃣ 왜 log를 쓰는가?

cross entropy는 이렇게 계산됩니다:

−
log
⁡
(
정답에 대한 확률
)
−log(정답에 대한 확률)

여기서 핵심은:

정답 확률이 1에 가까우면 → log값이 0에 가까움 → loss 작음

정답 확률이 0에 가까우면 → log값이 매우 큰 음수 → -붙이면 매우 큼 → loss 폭증

즉,

✔️ 틀릴수록 벌점이 기하급수적으로 커진다

이게 핵심입니다.

4️⃣ 숫자로 보면 더 직관적입니다
| 정답 확률 | -log(확률) | 해석      |
| ----- | -------- | ------- |
| 0.9   | 0.105    | 거의 안 틀림 |
| 0.5   | 0.693    | 애매      |
| 0.1   | 2.30     | 심하게 틀림  |
| 0.01  | 4.60     | 매우 심각   |

0.9 확률은 “거의 SLA 충족”
0.01은 “서비스 붕괴”

그래서 loss가 급격히 커집니다.

5️⃣ 왜 이름이 Cross Entropy인가?

정보이론 관점에서 보면:

Entropy = “불확실성”

Cross Entropy = “실제 분포와 예측 분포 사이의 차이”

LLM에서는 실제 정답은 one-hot 벡터입니다.
(정답 토큰만 1, 나머지 0)

즉,

모델의 확률 분포가 정답 분포와 얼마나 다른지 측정하는 값

입니다.

6️⃣ LLM에서의 의미

GPT가 다음 토큰을 예측한다고 할 때:

"나는 밥을 먹었다" 다음 토큰은?

모델이:

"그리고" 0.3

"맛있게" 0.2

"집에" 0.1

정답: "그리고"라면

loss는 -log(0.3)

그래서 훈련 목표는:

정답 토큰 확률을 1에 가깝게 밀어 올리는 것

입니다.

## c-TF-IDF (class-based variant of term frequency–inverse document frequency)
"클래스(카테고리)" 기준으로 계산하는 TF-IDF

📚 예:
•	문서 A, B, C가 있고:
o	A와 B는 "스포츠"
o	C는 "요리"

문서	내용 예시
A (스포츠)	"축구는 인기 있는 스포츠입니다"
B (스포츠)	"야구와 농구도 유명한 스포츠입니다"
C (요리)	"김치는 발효 음식입니다"

🧠 일반 TF-IDF라면?
•	각 문서별로 단어 중요도 계산
🧠 c-TF-IDF는?
•	A, B 문서를 **하나의 덩어리(스포츠 클래스)**로 합쳐서 계산
•	C는 요리 클래스
•	이렇게 하면 "스포츠 클래스"에 특히 잘 나타나는 단어가 뭔지 찾아낼 수 있음.

클래스	잘 나타나는 단어
스포츠	스포츠, 축구, 야구, 농구
요리	김치, 발효, 음식

🧠 어디에 써요?
c-TF-IDF는 특히 아래 같은 상황에서 유용
용도	설명
🏷️ 주제별 요약 (topic modeling)	각 주제(topic)마다 중요한 단어 추출
📝 문서 분류 설명	분류기(classifier)의 각 카테고리에 중요한 단어 확인
🔍 클러스터 레이블 생성	군집(cluster)마다 핵심 키워드 만들기
🤖 예시 도구: BERTopic	topic modeling할 때 핵심 단어 뽑을 때 사용됨

🎯 핵심 요약
구분	일반 TF-IDF	c-TF-IDF
기준	문서(document)	클래스(class, 그룹)
쓰임	단어 중요도	클래스별 대표 단어 찾기
대표 사용처	검색엔진, 문서요약	토픽 모델링, 설명 가능한 AI

🧮 계산 방식 (요약)
c-TF-IDF = (단어의 등장 빈도) × (그 단어의 중요도)
즉,
•	어떤 클러스터(예: 주제 1)의 BoW에 "AI"가 10번 등장했다면
•	"AI"의 IDF가 2.0이라면
→ c-TF-IDF 점수는 10 × 2.0 = 20이 됩니다.
→ 높은 점수를 받은 단어가 해당 주제를 잘 대표한다고 봄.

## Data Parallelism (데이터 병렬 처리, DP=192)
 
데이터를 여러 복사된 모델 인스턴스에 나눠서 동시에 처리.
여기선 같은 모델을 192개 복사하고 각각 다른 데이터를 처리하게 함.

## datasets
AI 모델이 학습하거나 평가할 때 사용하는 데이터의 모음. 쉽게 말해, AI가 배울 재료. 사람으로 치면 교과서 + 연습문제라고 보면 됨.

🔸 데이터셋 구성 예시
용도	예시
학습용 (Training)	질문-답변 쌍, 고양이 사진 + "고양이" 라벨 등
검증용 (Validation)	학습 도중 성능 확인용 데이터
시험용 (Test)	최종 성능 평가용 데이터

🔸 데이터셋 종류 예시 (분야별)
분야	예시 이름	내용
자연어처리 (NLP)	SQuAD, KorQuAD	질문과 답변 데이터
이미지 분석	ImageNet, CIFAR-10	이미지 + 정답 라벨
음성 인식	LibriSpeech, Common Voice	음성 파일 + 텍스트
한국어 AI 학습	AI Hub, 모두의 말뭉치	다양한 한국어 데이터셋

🔹 비유로 쉽게
•	AI는 학생
•	Dataset은 교재 (예제 + 정답 포함)
•	Training Dataset = 수업 시간 교재
•	Validation Dataset = 중간고사
•	Test Dataset = 기말고사

① Base mix for pretraining
사전 학습(pretraining)을 위한 기본 데이터셋 조합
•	Web:
o	FineWeb-Edu, DCLM, FineWeb2, FineWeb2-HQ
→ 교육/기술 중심의 웹 문서 크롤링 기반 데이터셋들
→ 예: 고품질 기술 문서, 위키, 문과/이과 교육자료 등
•	Code:
o	The Stack v2 (16 langs)
→ 16개 프로그래밍 언어로 구성된 초대형 공개 코드 저장소
→ HuggingFace/BigCode 프로젝트의 결과물
o	StarCoder2 PRs
→ StarCoder2 관련 Pull Request 코드
•	Code 관련 텍스트:
o	Jupyter/Kaggle NBs → 노트북 기반 튜토리얼
o	GH issues → GitHub 이슈
o	StackExchange → 기술 Q&A
•	Math:
o	FineMath3+, InfiWebMath3+
→ 수학 관련 고품질 웹 문서 및 문제/풀이 데이터

② 추가된 학습 데이터
•	추가된 Datasets:
o	Stack-Edu: 스택 오버플로우와 교육 중심 Q&A의 혼합 데이터
o	FineMath4+, InfiWebMath4+: 더 높은 난이도의 수학 데이터셋
o	MegaMath:
	OAI의 Owen Q&A
	Pro synthetic rewrites: 전문가 수준의 인위적 문장 재작성
	text/code interleaved blocks: 텍스트 + 코드가 섞인 블록

③ 최종 세부 조정용 데이터셋
•	Upsampling high quality code/math datasets:
→ 코드/수학 중 고품질만 골라서 비중을 더 높임
•	OpenMathReasoning:
→ 수학적 추론 능력 향상용 데이터 (예: 증명, 풀이 과정 등)

OpenAI, Meta, HuggingFace, Mistral, DeepMind 등이 LLM을 사전학습할 때 자체 구축하거나 추출한 데이터셋들. 특히 다음처럼 연결됨
이름	관련 프로젝트 또는 출처
The Stack v2	HuggingFace/BigCode
FineWeb	고품질 웹 크롤링 기반 사전학습용 데이터 (Meta 등 사용)
FineMath/InfiMath	Mistral, DeepSeek 등에서 수학 모델에 사용
MegaMath	OpenAI GPT-4 및 Gemini 계열 수학 강화 데이터
OpenMathReasoning	수학적 추론을 위한 신경망 fine-tuning용 (SFT) 데이터

## DBSCAN
**데이터 포인트가 밀집된 정도(밀도)**를 기준으로 클러스터를 찾는 비지도 학습(unsupervised learning) 방식의 클러스터링(Clustering) 알고리즘. 밀도 기반 클러스터 탐색. 대표적인 알고리즘이 DBSCAN (Density-Based Spatial Clustering of Applications with Noise).

더 나은 버전으로 HDBSCAN(Hierarchical Density-Based Spatial Clustering of Applications with Noise)가 있음.

📌 DBSCAN vs HDBSCAN 비교표
항목	DBSCAN	HDBSCAN
✅ 전체 이름	Density-Based Spatial Clustering of Applications with Noise	Hierarchical Density-Based Spatial Clustering of Applications with Noise
🎯 목표	밀도 기반 클러스터 탐색	계층적 밀도 기반 클러스터 탐색
🔧 주요 파라미터	eps (반경), min_samples (최소 점 수)	min_cluster_size (최소 클러스터 크기), min_samples (선택적)
📐 클러스터 수	자동 결정	더 유연하게 자동 결정
📦 클러스터 구조	단일 밀도 기준	여러 밀도 수준을 고려하여 계층 구조 생성 후 최적 클러스터 선택
📊 복잡한 분포	다소 한계 있음	매우 복잡한 분포도 탐지 가능
🚫 노이즈 처리	있음 (label = -1)	더 정교하게 처리
🧠 내부 알고리즘	단일 밀도 영역에서 확장	계층적 트리 → 안정성 기반 평면화
💻 연산 복잡도	빠름 (상대적으로 단순)	약간 느릴 수 있음 (복잡도 증가)

🔍 핵심 차이점 요약
구분	DBSCAN	HDBSCAN
밀도 기준	하나의 global 밀도값(eps)에 의존	지역마다 다른 밀도도 반영 가능
클러스터 분리 기준	반경 eps 기준으로 점들을 연결	**밀도 기반 트리(계층 구조)**를 만들어 클러스터 분리
유연성	밀도 차이가 큰 데이터에 약함	다양한 밀도 클러스터를 유연하게 탐지 가능
설정 난이도	eps 설정이 까다로움	min_cluster_size는 비교적 직관적

## decoder-only model
🔷 Transformer 기반 모델 구조의 세 가지 유형
유형	구성	대표 모델	설명
1. Encoder-only	인코더만 사용	BERT	입력을 이해하고 표현(embedding)을 잘 만드는 데 초점 (분류, 검색 등)
2. Decoder-only	디코더만 사용	GPT, ChatGPT	이전 단어들을 기반으로 다음 단어를 생성 (텍스트 생성)
3. Encoder-Decoder	둘 다 사용	T5, BART, MarianMT	입력을 인코딩하고, 그걸 바탕으로 새로운 출력 생성 (번역, 요약 등)

✅ 2. Decoder-only (예: GPT, ChatGPT)
•	입력을 기반으로 단어를 순차적으로 생성
•	이전 단어들을 보고 다음 단어를 예측
•	활용: 대화, 글쓰기, Q&A 등

🔍 세부 비교: 구조 vs. 기능
구분	Encoder-Decoder 모델 (원형 Transformer)	Decoder-only 모델 (GPT)
구조	인코더와 디코더 둘 다 사용	디코더만 사용
동작	인코더가 입력 문장을 이해하고, 디코더가 새로운 문장 생성	디코더가 이전 단어를 기반으로 다음 단어 생성
목적	입력 시퀀스 → 출력 시퀀스 (seq2seq)	입력 시퀀스 → 출력 시퀀스 (seq2seq)
생성형인가?	✅ 예 (예: 번역, 요약 등)	✅ 예 (예: 글쓰기, 대화 등)
🔧 결론 먼저 말하면:
생성형 AI = 시퀀스를 만들어내는 모델 (텍스트 생성)
이 목적을 달성하기 위해
•	인코더-디코더 구조를 사용할 수도 있고 (T5, BART, 원래 Transformer)
•	디코더만으로도 충분히 생성이 가능하기 때문에 (GPT, ChatGPT)
디코더-only 모델도 생성형 모델로 분류 됨.
즉,
✅ "구조는 다르지만, ✅ 기능적으로 생성형이면 '생성형 AI'로 분류"되는 것

## decoding strategies
텍스트 생성 전략(decoding strategies)은 LLM이 문장을 생성할 때 어떤 단어를 다음에 선택할지 결정하는 방법이다. 단순히 가장 확률이 높은 단어만 선택하면 항상 같은 문장이 나오기 때문에, 더 자연스럽고 다양한 텍스트를 만들기 위해 여러 가지 생성 전략을 사용한다. 이러한 전략을 통해 모델은 확률이 높은 단어뿐 아니라 다른 가능성도 고려하면서 더 창의적이고 다양한 문장을 생성할 수 있다. (Temperature, Top-k)

## Density-based algorithm
**데이터 포인트가 밀집된 정도(밀도)**를 기준으로 클러스터를 찾는 비지도 학습(unsupervised learning) 방식의 클러스터링(Clustering) 알고리즘. 대표적인 알고리즘은 **DBSCAN (Density-Based Spatial Clustering of Applications with Noise)**

📌 핵심 개념 요약
개념	설명
밀도(density)	가까운 이웃들이 얼마나 많은지 (반경 내 포인트 개수)
핵심 포인트(core point)	주변에 충분한 포인트가 존재하는 지점
이웃 포인트(neighbor)	핵심 포인트의 반경 내에 있는 점들
경계 포인트(border point)	핵심 포인트 근처에 있지만 자기 주변은 밀도 낮은 포인트
노이즈(noise/outlier)	어느 클러스터에도 속하지 않는 외딴 포인트

🧭 작동 방식
1.	각 포인트에 대해 반경 ε 이내에 있는 이웃 수를 계산
2.	이웃 수가 최소 포인트 수(minPts) 이상이면 핵심 포인트로 간주
3.	핵심 포인트와 연결된 이웃들을 하나의 클러스터로 확장
4.	이 과정을 모든 핵심 포인트에 대해 반복
5.	클러스터에 속하지 않는 포인트는 **노이즈(이상치)**로 분류

✅ 장점
✅ 비정형(복잡한 모양) 클러스터도 탐지 가능
✅ **클러스터 수(K)**를 미리 정할 필요 없음
✅ 이상치(outlier) 자동 제거
✅ 작은 클러스터도 포착 가능 (K-means보다 유연함)

❌ 단점
❗ 반경(ε)과 최소 포인트 수(minPts) 설정이 중요
❗ 고차원 데이터에서는 성능 저하 가능
❗ 밀도 차이가 큰 경우 일부 클러스터를 놓칠 수 있음

📌 K-means와의 차이
항목	K-means (Centroid-based)	DBSCAN (Density-based)
클러스터 수	지정 필요 (K)	자동 탐지
클러스터 모양	둥근 형태로 제한	복잡한 모양도 가능
이상치 처리	어려움	자동 제거
고차원 데이터	상대적으로 잘 작동	차원 증가에 취약


## diffusers library (Huggingface)
“Diffusion Model (확산 모델)” 전용 라이브러리. diffusers는 Stable Diffusion 같은 이미지 생성 모델을 쉽게 불러와서 쓸 수 있게 해주는 Python 라이브러리.

🎨 무엇을 할 수 있나요?
기능	설명
🖼️ 텍스트 → 이미지 생성	"고양이가 피자를 먹고 있는 장면" → 이미지 생성
🎞️ 이미지 → 이미지 변화	기존 이미지에 스타일 입히기, 리터칭 등
🧠 커스텀 훈련	자신만의 이미지 스타일로 모델 훈련 (DreamBooth, LoRA 등)
💡 다양한 모델 지원	Stable Diffusion, Versatile Diffusion, DeepFloyd IF, Kandinsky 등

🧠 기술적으로는 어떤 모델?
•	diffusers는 다양한 Diffusion 모델들을 지원:
•	Stable Diffusion (텍스트 → 이미지 생성)
•	ControlNet (이미지 구조 보존 + 조건 제어)
•	InstructPix2Pix (텍스트로 이미지 편집)
•	Imagen, DeepFloyd-IF (고품질 생성)
•	Versatile Diffusion, DiT (Diffusion Transformer) 등

💻 코드 예시 (초간단)
from diffusers import StableDiffusionPipeline
import torch

pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
pipe.to("cuda")

image = pipe("A cat wearing sunglasses, photorealistic").images[0]
image.show()

📦 설치 방법
pip install diffusers transformers accelerate
GPU 사용 시 xformers까지 설치하면 성능 향상
pip install xformers

✅ 왜 중요한가?
이유	설명
🎯 사용이 간편	모델 불러오고 .pipe()만 쓰면 바로 생성 가능
🔧 다양한 기능 통합	학습, 미세조정(Fine-tuning), LoRA, Inpainting 등 전부 가능
🔗 Hugging Face 모델 허브와 연동	모델 이름만 입력하면 자동 다운로드됨

🧩 요약 정리
항목	내용
이름	Huggingface diffusers
역할	Diffusion 모델을 불러오고 실행할 수 있는 표준 도구
지원 모델	Stable Diffusion, DeepFloyd-IF, DiT, ControlNet 등
사용 목적	텍스트 → 이미지, 이미지 편집, 멀티모달 생성 등
필요 지식 수준	기본 Python + PyTorch 정도면 사용 가능

## dimension
"embedding에서 말하는 차원(dimension)"은 우리가 아는 공간의 1차원, 2차원, 3차원 개념과는 유사하지만, 정확히는 "수학적 벡터 공간의 차원"을 의미.

예시:
•	어떤 단어(예: "apple")를 embedding하면 → [0.12, -1.03, 0.55, ..., 0.81] 같은 숫자 벡터로 바뀜.
•	이 숫자 벡터가 **몇 개의 숫자(숫자의 개수)**로 표현되느냐 = "임베딩 차원 수".

🧠 비교
구분	설명
📏 우리가 아는 3차원 공간	x, y, z로 좌표 표현 (예: 3D 공간)
🔢 임베딩의 768차원	단어를 768개의 숫자로 표현하는 벡터 공간의 차원 수
📦 차원이 높다는 의미	더 많은 숫자(정보)를 사용해 단어를 표현한다는 뜻 (더 정교한 의미 표현 가능)

📌 쉽게 말하면
단어	임베딩 결과 (차원 수: 5 가정)
"apple"	[0.1, -0.3, 0.7, 0.0, 0.9]
"banana"	[0.0, -0.2, 0.6, 0.1, 0.8]
→ 이 5개의 숫자는 **단어의 의미를 수치로 나타낸 표현(semantic representation)**이며,이걸 구성하는 숫자의 개수가 바로 **임베딩 차원 수 (dimensionality)**

📌 왜 이렇게 고차원으로 표현할까?
•	언어는 의미적으로 아주 복잡함
•	고차원 공간에서 더 미세한 의미 차이를 표현 가능
•	2차원/3차원으로는 이런 뉘앙스를 표현하기 어려움

🧭 정리
질문	답변
임베딩의 차원 = 우리가 아는 1차원, 2차원, 3차원과 같음?	개념은 유사하지만, **"벡터 공간상의 차원"**이라는 수학적 개념임
768차원은 실제 공간이 있어?	아님. **추상적 의미 공간(semantic space)**임
차원이 높을수록?	더 풍부하고 정교한 의미를 담을 수 있지만 계산량도 커짐

## dimensionality Reduction (차원 축소)
정의: 고차원의 데이터를 저차원으로 줄이는 과정.
예: 768차원의 BERT 임베딩 → 5~50차원의 벡터
목적:
•	시각화 (t-SNE, UMAP)
•	계산 효율 향상
•	노이즈 제거
•	의미 보존된 간결한 표현 추출
사용 예시 in BERTopic:
•	문장 임베딩을 얻은 후 UMAP 같은 알고리즘으로 차원 축소 → HDBSCAN으로 군집화



## discriminative AI (판별형 인공지능)
🧠 1. 판별형 인공 지능 (Discriminative AI)란?
판별형 AI는 주어진 입력(데이터)이 어떤 **카테고리(class)**에 속하는지를 **구별(판별)**하는 모델. 즉, "입력이 무엇인가?"보다는 "그 입력이 어떤 **정답(y)**에 해당하는가?"에 관심을 둠.
📌 수학적으로는 조건부 확률인 P(y|x) 를 모델링 함.
→ 입력 x가 주어졌을 때, 출력 y가 될 확률

🔁 2. 작동 방식
•	데이터셋: (입력 x, 정답 y) 쌍으로 구성
•	학습 목표: x → y로 변환하는 규칙을 학습
•	특징: 데이터 분포 자체는 신경 쓰지 않고, **경계(boundary)**를 정확히 그리는 데 집중

🧪 3. 대표 모델
모델	설명
Logistic Regression	이진 분류의 대표 모델 (스팸 vs 정상)
Support Vector Machine (SVM)	최적 경계면을 찾는 분류기
Random Forest / XGBoost	트리 기반 앙상블 분류기
Neural Networks (BERT, ResNet 등)	입력을 분류하는 신경망 모델
BERT (fine-tuned)	문장 분류, 감성 분석 등에서 자주 활용

🧾 4. 예시
✉️ 감성 분류
입력	출력 (라벨 y)
“이 영화 정말 좋았어요!”	긍정
“시간 낭비였어요…”	부정
→ 판별형 AI는 문장을 보고 긍정/부정을 분류

✅ 5. 장점과 단점
장점	단점
빠르고 예측 정확도 높음	새로운 데이터 생성을 못함
학습이 비교적 단순함	데이터 분포 학습은 불가능
실전 적용이 쉬움	노이즈나 레이블 오류에 민감

🏗 6. 사용되는 분야
분야	예시
자연어 처리 (NLP)	감성 분석, 스팸 필터링, 뉴스 분류
컴퓨터 비전	이미지 분류, 얼굴 인식
금융	신용평가, 이상 거래 탐지
헬스케어	암 진단, 질병 분류
보안	이메일 공격 탐지, 악성코드 분류

📌 핵심 요약
판별형 인공 지능은 데이터를 새로 만들지는 않지만, 입력이 어떤 정답에 속하는지를 매우 정확하게 맞히는 데 특화된 AI.

## diffusion transformer model
**Diffusion Model(확산 모델)**과 **Transformer(트랜스포머)**를 결합한 모델.

✅ 1. 두 개념 먼저 분리해서 이해
🔹 (1) Diffusion Model (확산 모델, 디퓨전 모델)
이미지 생성 모델로 유명한 구조.
예: Stable Diffusion, DALL·E 3, Imagen 등이 이 구조를 사용.

🌱 원리 간단 요약:
원래 이미지를 **노이즈(noise)**로 바꿔버리고,
그 노이즈를 거꾸로 복원해서 원본 이미지나 데이터를 만들어냄.
마치 "흐려진 그림을 점점 선명하게 되돌리는 과정"이라고 생각하면 됨.

🔹 (2) Transformer (트랜스포머)
GPT, BERT 등 LLM에서 사용되는 구조
문장, 코드, 음악 등 **시퀀스(연속된 데이터)**를 처리하는 데 탁월.

🧠 핵심 기능:
Self-Attention 메커니즘으로 문맥 파악을 잘 함
최근에는 이미지, 영상, 3D 등 비정형 데이터에도 폭넓게 사용됨

✅ 2. Diffusion + Transformer = 무슨 효과?
“Transformer를 확산 모델의 핵심 구조로 넣어 성능을 끌어올리는 모델.”

이전에는 확산 모델 내부의 “노이즈 제거기(denoiser)”로 CNN이나 U-Net 같은 구조를 사용했는데, 최근에는 여기에 Transformer를 도입해서 더 강력한 표현력을 갖게 된 것.

✅ 대표 모델 예시
모델 이름	설명
Imagen (Google)	Transformer 기반 확산 모델로 고화질 이미지 생성
DALLE-3 (OpenAI)	내부적으로 Transformer + Diffusion 기반 추정됨
DiT (Diffusion Transformer, Meta AI)	Vision Transformer (ViT)을 Diffusion 모델 내부에 적용한 모델
Versatile Diffusion, Muse	Text-to-image + multi-modal 확산 변형 모델

🎯 왜 중요한가?
장점	설명
🔍 정밀한 표현	트랜스포머가 문맥과 구조를 더 잘 이해해서, 더 자연스럽고 의미 있는 결과 생성 가능
🧠 유연성	텍스트, 이미지, 오디오 등 다양한 형식을 처리 가능 (멀티모달)
📷 고화질	이미지 품질이 뛰어남 (Stable Diffusion보다 더 선명하게 나올 수 있음)

✅ 정리 요약
항목	내용
Diffusion Transformer Models	확산 모델의 구조 안에 트랜스포머(Transformer)를 결합한 고성능 생성 모델
주로 사용 분야	텍스트 → 이미지 생성, 고해상도 이미지 생성, 멀티모달 생성
장점	문맥 이해력 + 고품질 생성력 → 최신 생성 AI의 핵심 트렌드 중 하나

## dispatcher server
'사용자 <-> L4 스위치 <-> 웹서버들 <-> L4 스위치 <-> 분산 요청 스케줄러(디스패쳐) 서버들 (보통 별도의 물리서버 형상으로 구성) <-> GPU 서버들 <-> GPU' 플로우로 AI LLM 서비스가 이루어짐.

[사용자 브라우저 or 앱]
          │
          ▼
   📶 [L4 Load Balancer]           ← TCP/UDP 기반 트래픽 분산
          │
          ▼
   🌐 [웹서버들 (API 서버)]        ← 인증, 요청 전처리, 요청 라우팅
          │
          ▼
 ⚙️ [분산 요청 스케줄러 (Dispatcher 서버)]  ← GPU 상태 확인 & 지능형 분배
          │
          ▼
 🖥️ [GPU 서버들 (서버당 4~8개 GPU)]     ← 요청 수신, 모델 로드/실행
          │
          ▼
 🧠 [GPU 내부 (LLM 실행)]           ← 실제 추론 수행, 토큰 생성

✅ Dispatcher Server는 어떤 기능을 하나?
1.	실시간 GPU 상태 모니터링
o	GPU 사용량 (%), 메모리 사용량, 큐 길이, 응답 속도 등
2.	스케줄링 정책 적용
o	Round-robin, 가장 한가한 서버 우선, 지연 시간 최소화 등
3.	요청 라우팅
o	들어온 요청을 내부 gRPC나 HTTP로 GPU 서버에 전달
4.	오류 감지 및 재시도
o	GPU 서버가 죽거나 에러 나면 자동으로 다른 서버로 재전송
5.	로드밸런싱 + 리소스 효율화
o	과부하 방지, 최대 병렬 처리 확보

✅ 어떤 시스템들이 이런 Dispatcher 역할을 하나?
시스템	설명
Kubernetes	Pod 단위로 Job을 GPU 노드에 스케줄링, 매우 일반적
Ray Serve	Python 기반 ML serving dispatcher, GPU 분산 최적화
Huggingface TGI	텍스트 생성 특화, 텍스트 입력을 여러 GPU 서버에 분산
Triton Inference Server	NVIDIA 제공, 모델 종류 상관없이 다양한 백엔드 지원
Custom Dispatcher	자체 구축한 요청 라우터 (OpenAI, Anthropic 등은 대부분 이 방식도 혼합)

## dot product (내적)

예를 들어:

a · b


이 점을 dot product라고 부릅니다.

수학적으로는 이것이 바로
👉 내적(inner product) 입니다.

2️⃣ 내적이 뭐냐?

두 벡터가 있을 때:

a = (a₁, a₂, a₃)
b = (b₁, b₂, b₃)


내적은 이렇게 계산합니다:

a · b = a₁b₁ + a₂b₂ + a₃b₃


즉,

같은 위치 성분끼리 곱해서 전부 더하는 것

입니다.


1️⃣ 내적(dot product)이 뭐냐?

두 벡터가 있을 때
a = [a1, a2, a3]
b = [b1, b2, b3]

내적은 이렇게 계산합니다:
a · b = a1b1 + a2b2 + a3b3

즉,

같은 위치끼리 곱해서 다 더하는 것

입니다.

2️⃣ 그런데 왜 이걸 쓰는가?

핵심은 이것입니다:

내적은 두 벡터가 얼마나 비슷한지 측정하는 방법입니다.

결과 해석

값이 크다 → 방향이 비슷하다 → 의미가 관련 있다
0에 가깝다 → 거의 관련 없다

음수다 → 반대 방향

3️⃣ 왜 “내적”이라는 이름이 붙었을까?

“내적”이라는 말은
벡터 공간 내부에서 정의된 곱이라는 뜻입니다.

벡터 × 벡터 → 스칼라(숫자)

이 연산이 벡터 공간 안에서 정의되기 때문에
“inner product(내적)”라고 부릅니다.

4️⃣ Self-Attention에서 왜 쓰나?

Self-Attention에서는

score = Q · K


즉,

Query 벡터와 Key 벡터의 내적을 계산합니다.

왜?

👉 두 벡터가 얼마나 비슷한지(유사도)를 측정하기 위해서입니다.

값이 크면 → 방향이 비슷 → 관련성 높음
값이 작으면 → 관련성 낮음

| 표현            | 의미       |
| ------------- | -------- |
| dot           | 점        |
| dot product   | 점을 이용한 곱 |
| inner product | 내적       |
| 둘은 같은 것인가?    | ✅ 같은 연산  |

“내적(內積)”을 한자로 풀이

1️⃣ 內 (내)

뜻: 안, 내부

의미: 벡터 공간 “안에서” 정의되는 연산

즉,
외부 연산이 아니라 벡터 공간 내부 구조에 의해 정의되는 곱라는 뜻입니다.

2️⃣ 積 (적)

뜻: 쌓다, 곱하다, 누적하다

수학에서는 보통 “곱”을 의미

3️⃣ 그래서 內積(내적)은?

내부에서 정의된 곱

이라는 뜻입니다.

좀 더 수학적으로 말하면:

- 벡터 × 벡터
- 결과는 숫자(스칼라)
- 같은 차원 성분끼리 곱해서 더함

4️⃣ 왜 “내(內)”가 붙었을까?

벡터 연산에는 두 가지 대표적인 곱이 있습니다:

① 내적 (內積, inner product)

결과: 스칼라

방향 정보 비교

유사도 측정

② 외적 (外積, cross product)

결과: 벡터

두 벡터에 수직인 벡터 생성

여기서 “외(外)”는
공간의 외부 방향을 만들어낸다는 의미입니다.

5️⃣ Self-Attention과 연결

Self-Attention에서 사용하는:

Q · K


이 바로 내적(內積) 입니다.

이걸로

두 벡터가 얼마나 같은 방향을 보는지

를 측정합니다.

## dropout
Dropout은 학습(training)할 때 뉴런(neuron)을 무작위로 꺼버리는 기법입니다.

즉,

어떤 뉴런의 출력(output)을 0으로 만들어서
그 뉴런이 없는 것처럼 학습시킵니다.

예를 들어 dropout rate = 0.5 이면
→ 학습할 때 뉴런의 50%를 랜덤하게 끕니다.

좋습니다.
이번에는 단순 정의가 아니라, 왜 꼭 필요한지까지 냉정하게 보겠습니다.

1️⃣ Dropout이 뭐냐?

Dropout은 학습(training)할 때 뉴런(neuron)을 무작위로 꺼버리는 기법입니다.

즉,

어떤 뉴런의 출력(output)을 0으로 만들어서
그 뉴런이 없는 것처럼 학습시킵니다.

예를 들어 dropout rate = 0.5 이면
→ 학습할 때 뉴런의 50%를 랜덤하게 끕니다.

2️⃣ 왜 이런 이상한 짓을 할까?

핵심 이유는 단 하나입니다.

👉 Overfitting (과적합) 방지

📌 Overfitting이 뭐였죠?

모델이 훈련 데이터만 너무 잘 외워버리는 현상

Training 성능 ↑↑

Test 성능 ↓

3️⃣ Dropout이 Overfitting을 막는 원리

여기서 핵심 개념이 나옵니다.

Co-adaptation (공동 적응)

뉴런들이 이렇게 학습한다고 상상해 보세요:

"쟤가 있으니까 나는 대충 해도 돼."

이게 문제입니다.

뉴런들이 서로 의존해버립니다.

Dropout이 개입하면?

랜덤하게 뉴런을 제거합니다.

그러면?

특정 뉴런에 의존 불가

매번 다른 네트워크 구조로 학습

더 강건한(feature robust) 표현 학습

4️⃣ 수학적으로 보면

학습 시:
h = f(Wx)

Dropout 적용 시:
mask ~ Bernoulli(p)
h = mask * f(Wx)


여기서

Bernoulli(p) = 확률 p로 1, 아니면 0

mask가 0이면 뉴런 출력 삭제

5️⃣ 직관적 비유

Dropout은 마치:

매번 시험 볼 때 팀원을 랜덤으로 빼고 프로젝트 수행하는 것

그 결과:

특정 한 명에게 의존 불가

모든 팀원이 실력 갖춰야 함

6️⃣ Transformer에서 Dropout은 어디 쓰이나?

✔ Attention weight (어텐션 가중치)
✔ Feed Forward layer
✔ Embedding layer
✔ Residual connection 이후

7️⃣ 그런데 LLM에서는 Dropout 거의 안 쓴다?

중요한 포인트입니다.

GPT 같은 대형 LLM은:

대규모 데이터

거대한 파라미터

엄청난 batch size

이 조건에서는 overfitting 위험이 낮습니다.

그래서 최근 LLM은:

Dropout 거의 0

또는 매우 낮게 설정

## dual mode reasoning (듀얼 모드 추론)
‘생각하는 모드(think)’와 ‘생각하지 않는 모드(no_think)’라는 두 가지 추론 방식을 지원.

✅ 용어 설명
용어	뜻	설명
Instruct model	지시형 모델	사용자의 명령에 따라 반응하도록 튜닝된 LLM
dual mode reasoning	이중 추론 모드	두 가지 다른 방식의 사고 또는 응답을 구분해서 지원
think mode	"생각하기" 모드	논리적 추론과 설명을 포함한 답변
no_think mode	"생각 안 함" 모드	즉각적인 정답만 출력, 설명 생략

✅ 예시
1. Think mode (Chain-of-Thought 방식)
Q: What is 17 + 25?

A (think mode):
Let's break this down step by step.
17 + 25 = 42
So the answer is 42.

2. No_think mode (빠른 응답)
Q: What is 17 + 25?

A (no_think mode):
42

✅ 왜 이런 기능이 있나요?
이유	설명
💬 다양한 사용자 요구	어떤 사용자는 빠른 답만, 어떤 사용자는 논리적 설명을 원함
⚙️ 파인튜닝과 평가 분리	추론 능력 평가(CoT) vs 실전 응답 속도 제어
🔁 다양한 어플리케이션	교육용, 실시간 시스템 등에서 필요에 따라 다르게 사용 가능

## epoch
🧠 Epoch(에폭)이란?
Epoch이란, 전체 학습 데이터셋을 인공지능 모델이 한 번 모두 보고 학습하는 과정을 말함.

🎯 예
데이터셋에 이미지가 100장 있다고 가정.
모델이 이 100장을 한 번씩 모두 학습하고 나면 → 1 epoch이 끝난 것.
그리고 다시 그 100장을 다시 한 번 학습하면 → 2 epoch이 된 것.

📦 함께 자주 쓰이는 개념들
용어	설명
Epoch	전체 데이터셋을 모델이 한 번 학습하는 단위
Batch	데이터를 작은 묶음으로 나눈 것 (예: 1000장 중 32장씩 학습)
Iteration	한 번의 배치 학습을 의미 (예: 1000장, batch size 32 → 1 epoch = 32 iterations)

🏋️ 왜 여러 epoch를 반복할까?
한 번 본다고 해서 모델이 모든 패턴을 제대로 이해하진 못함.
그래서 같은 데이터를 **여러 번 반복해서 학습(epoch 반복)**함으로써,
점점 더 정확해지고
오차가 줄어들고
일반화 성능이 좋아짐.

📌 하지만 너무 많이 반복하면 → 과적합(overfitting) 위험도 있음.
(즉, 훈련 데이터만 잘 맞추고 실제에선 성능 떨어짐)

📈 예시 (시각화 감각)
Epoch 수	모델 성능 변화
1 epoch	학습 시작 단계, 정확도 낮음
5 epoch	학습 중간, 오차 줄어듦
20 epoch	학습 잘됨, 성능 향상
100 epoch	과적합 위험 발생 가능

✅ 정리
Epoch이란 전체 데이터를 몇 번 반복해서 학습할지를 정하는 횟수. 일반적으로 딥러닝 모델은 수십에서 수백 번의 epoch 동안 학습.

## embeddings
각 토큰을 고정된 길이의 숫자 벡터로 바꾸는 작업. 쉽게 말해, "토큰을 숫자로 표현한 것"

예시:
"학교" → [0.12, -0.89, 0.33, ..., 0.01]  (보통 768차원, 1024차원 등)

이 숫자 벡터는 단순히 고유 ID가 아니라, **의미(semantic meaning)**를 포함. 즉, 임베딩은 단어의 뜻을 숫자로 표현한 것이라 보면 됨.

## embedding Layer (임베딩 레이어)
토큰 ID를 밀집 벡터(dense vector)로 바꿔주는 학습 가능한 테이블

1️⃣ 가장 단순하게 말하면

임베딩 레이어는 이런 모양이야:
(어휘 개수) × (임베딩 차원)

예를 들어:

vocab_size = 6
embedding_dim = 3

그러면 내부에는 이런 가중치 행렬(weight matrix)이 있어:
6 × 3 행렬

각 행(row)은 하나의 단어(토큰)에 해당하는 벡터야.

2️⃣ 실제로 하는 일

토큰 ID가 3이면?

→ 그 행렬의 3번 행을 그대로 꺼낸다

이게 끝이야.

복잡한 계산이 아니라 lookup(조회) 작업이야.

3️⃣ 그런데 왜 “레이어(layer)”라고 부르냐?

중요한 부분이다. 이 행렬은 고정된 사전이 아니라 학습되는 가중치(weight) 야.

훈련 중에:

역전파(backpropagation)로 이 벡터 값들이 계속 수정됨. 그래서 그냥 테이블이 아니라 신경망의 한 층(neural network layer으로 본다.

4️⃣ 원-핫과의 관계

수학적으로는:
원-핫 벡터 × 가중치 행렬 = 임베딩 벡터

하지만 실제 구현은:

원-핫을 만들지 않고
그냥 해당 행을 바로 꺼낸다.

그래서 훨씬 효율적이다.

5️⃣ 한 문장으로 정리

임베딩 레이어는
토큰 ID → 의미를 담은 벡터로 바꿔주는 학습 가능한 가중치 테이블

## Embedding model
문장이나 단어를 **벡터(숫자 배열)**로 바꿔주는 모델
→ 즉, 텍스트를 수치화해서 컴퓨터가 이해할 수 있도록 표현하는 모델
예시:
"나는 너를 사랑해" → [0.24, -0.13, 1.77, ...]  # 768차원 벡터

사용 목적:
•	의미 기반 유사도 계산 (문장 비교, 검색 등)
•	분류기(Classifier)의 입력값으로 사용
•	클러스터링/시각화 등
➡ 임베딩 모델은 **입력 텍스트를 벡터로 “표현”**하는 데 중점

🔄 두 모델의 관계
✅ 임베딩 모델은 task-specific 모델의 "입력 전처리" 또는 "기초"로서 사용됨.

예시 구조:
[입력 문장] 
    ↓
[Embedding model → 벡터]
    ↓
[Task-specific model → 예측 결과]

예를 들어:
•	BERT의 중간층 출력 = embedding
•	이 embedding을 받아서 **로지스틱 회귀 분류기(logistic regression classifier)**를 올리면
→ 전체가 task-specific model이 됨

🔶 비유로 쉽게 설명
모델 종류	역할	비유
Embedding model	의미를 벡터로 추출	🧠 단어/문장을 이해하고 요약하는 뇌
Task-specific model	벡터를 바탕으로 과제를 수행	🧑‍⚖️ 이해한 내용을 가지고 판단/결정하는 전문가

✅ 정리 요약
항목	Embedding Model	Task-specific Model
목적	텍스트 → 벡터 표현	특정 태스크(분류, 요약 등) 해결
출력	벡터 (숫자 배열)	라벨, 텍스트, 확률 등
학습 방식	일반 텍스트 기반 사전학습	정답 있는 태스크용 fine-tuning
예시 모델	Sentence-BERT, E5, USE	감성 분석기, 챗봇, 요약기 등
관계	입력 처리로 사용됨	임베딩을 받아 판단 수행

한국어/영어에 적합한 대표 임베딩 모델
✅ 1순위: BAAI bge 계열

(현업에서 가장 많이 쓰이는 범용 RAG 임베딩)

대표 모델

bge-m3

bge-large

bge-base

왜 강력하냐

한국어 / 영어 동시 지원

문장·문단 임베딩에 최적화

검색(retrieval) 전용으로 학습됨

길이 긴 문서에도 안정적

👉 “일단 실패 안 하는 선택”

✅ 2순위: Sentence-Transformers 계열

(유연성 + 커스터마이징 최강자)

대표 모델

all-MiniLM-L6-v2 (가볍고 빠름)

paraphrase-multilingual-MiniLM

LaBSE

특징

다국어 지원 (한국어 포함)

문장 의미 검색에 특화

CPU에서도 잘 돌아감

👉 소규모 서버 / PoC / 엣지 환경에 좋음

✅ 3순위: OpenAI 임베딩 모델

(품질 최상, 비용 있음)

대표 모델

text-embedding-3-large

text-embedding-3-small

특징

언어 불문 품질 매우 높음

별도 튜닝 필요 없음

API 비용 발생

온프레미스 ❌

👉 보안·비용 제약 없을 때 최고 성능

✅ 4순위: Alibaba Qwen 임베딩 계열

(Qwen LLM과 궁합 최고)

대표 모델

Qwen3-Embedding

Qwen2.5-Embedding

특징

한국어/영어 모두 준수

Qwen LLM과 의미 공간 정렬이 좋음

로컬/온프레미스 가능

👉 Qwen LLM 쓰면 같이 쓰는 게 정답

## encoder-decoder model
🔷 Transformer 기반 모델 구조의 세 가지 유형
유형	구성	대표 모델	설명
1. Encoder-only	인코더만 사용	BERT	입력을 이해하고 표현(embedding)을 잘 만드는 데 초점 (분류, 검색 등)
2. Decoder-only	디코더만 사용	GPT, ChatGPT	이전 단어들을 기반으로 다음 단어를 생성 (텍스트 생성)
3. Encoder-Decoder	둘 다 사용	T5, BART, MarianMT	입력을 인코딩하고, 그걸 바탕으로 새로운 출력 생성 (번역, 요약 등)

✅ 3. Encoder-Decoder (예: T5, BART)
•	입력 전체를 인코더가 이해 → 그걸 바탕으로 디코더가 출력 생성
•	활용: 기계 번역, 문서 요약, 질문 생성 등
•	BERT + GPT의 혼합형 구조

## encoder-only model
🔷 Transformer 기반 모델 구조의 세 가지 유형
유형	구성	대표 모델	설명
1. Encoder-only	인코더만 사용	BERT	입력을 이해하고 표현(embedding)을 잘 만드는 데 초점 (분류, 검색 등)
2. Decoder-only	디코더만 사용	GPT, ChatGPT	이전 단어들을 기반으로 다음 단어를 생성 (텍스트 생성)
3. Encoder-Decoder	둘 다 사용	T5, BART, MarianMT	입력을 인코딩하고, 그걸 바탕으로 새로운 출력 생성 (번역, 요약 등)

✅ 1. Encoder-only (예: BERT)
•	입력 문장을 이해하고 표현하는 데 초점
•	입력 → 벡터 표현
•	활용: 문장 분류, 문장 유사도 판단, 개체명 인식 등
•	생성 ❌ (텍스트를 "만드는 것"에는 적합하지 않음)

## evojax
Google이 개발한 진화 알고리즘(Genetic Algorithm, GA) 기반의 고속 머신러닝 프레임워크. **진화 알고리즘(Genetic Algorithm)**을 GPU/TPU에서 초고속으로 실행할 수 있게 만든 JAX 기반의 오픈소스 프레임워크.

🔍 핵심 개념 설명
개념	설명
JAX	구글이 만든 고속 수치 계산 프레임워크 (Numpy + GPU + 자동 미분)
Evolutionary Algorithm (진화 알고리즘)	자연 선택에 기반한 최적화 방법. 여러 후보(개체)를 평가하고, 좋은 개체만 선택·변형해 점점 성능을 올림
EvoJAX	이 진화 알고리즘을 JAX 위에 구현해서, 대규모 병렬 실행이 가능하도록 만든 도구

🧠 어떤 문제에 쓰나요?
•	강화학습(RL)처럼 보상이 불확실한 상황에서 최적의 행동 찾기
•	신경망 구조를 자동 설계 (신경망 진화, Neuroevolution)
•	복잡한 함수 최적화 문제 (비미분 가능 문제도 가능)

🚀 EvoJAX의 특징
특징	설명
⚡ 초고속 병렬화	GPU/TPU를 이용해 수천 개의 후보(개체)를 한 번에 평가 가능
🧬 진화 알고리즘 내장	ES(Evolution Strategy), CMA-ES, Genetic Algorithm 등 지원
🧠 신경망도 진화	PyTree 기반 신경망도 직접 진화 가능 (Neuroevolution)
🔁 JAX와 완전 호환	JAX로 정의된 환경, 정책, 평가 함수를 그대로 사용 가능

📦 설치 방법
pip install evojax flax

💻 코드 예시 (초간단)
from evojax import Trainer
from evojax.envs import GymEnv
from evojax.task.base import make_task

task = make_task('CartPole-v1')  # OpenAI Gym 환경
trainer = Trainer(task=task, algo_name='SimpleGA')
trainer.train(num_generations=100)
위 코드는 진화 알고리즘으로 카트폴 밸런스 게임을 해결하는 예시

✅ 어디에 쓰이나요?
분야	예시
강화학습 대안	Gradient 없이 학습 (ES, GA 방식)
로봇 제어	물리 시뮬레이션에서 로봇 행동 진화
AutoML	딥러닝 구조 자동 설계
TPU 대규모 실험	병렬 탐색 및 최적화 실험

🔗 공식 GitHub
👉 https://github.com/google/evojax

✅ 요약 정리
항목	설명
이름	EvoJAX
만든 곳	Google
핵심 기능	진화 알고리즘을 GPU/TPU에서 병렬 실행
기반 기술	JAX + Flax
활용 분야	강화학습, AutoML, 신경망 진화, 함수 최적화 등

## exploding gradients
1️⃣ 한 줄 정의

Exploding Gradient(기울기 폭발) 는
역전파(Backpropagation, 역방향 전파) 과정에서
Gradient(기울기) 가 점점 커져서
학습이 불안정해지고 모델이 망가지는 현상입니다.

2️⃣ 왜 발생할까?

Vanishing은 작은 수를 계속 곱해서 0이 됐죠.

Exploding은 반대입니다.

예를 들어 기울기가 2라고 합시다.

Layer가 10개라면:

2 × 2 × 2 × 2 × ... (10번)
= 1024


👉 기울기가 1000배 이상 커짐.

Layer가 50개라면?

👉 거의 천문학적 숫자.

3️⃣ 실제로 어떤 일이 벌어질까?

학습 공식은 이겁니다:

W_new = W_old - learning_rate × gradient


그런데 gradient가 엄청 크면?

weight가 갑자기 엄청 크게 바뀜

loss가 갑자기 폭등

NaN 발생

모델 발산(diverge)

👉 훈련이 터져버립니다.

4️⃣ 언제 자주 발생할까?
✅ 1. RNN (Recurrent Neural Network, 순환 신경망)

시퀀스 길이가 길수록
기울기를 계속 곱함 → 폭발 가능

✅ 2. 깊은 네트워크 + 큰 초기값

가중치 초기화가 크면
forward에서도 값이 커지고
backward에서도 기울기가 증폭

✅ 3. 큰 Learning Rate

학습률이 너무 크면
폭발이 더 심해짐

5️⃣ Vanishing vs Exploding 비교
| 구분  | Vanishing | Exploding |
| --- | --------- | --------- |
| 기울기 | 0에 가까워짐   | 무한대로 커짐   |
| 문제  | 학습 안 됨    | 학습 폭주     |
| 결과  | 느림        | 발산 / NaN  |
| 위험도 | 조용한 문제    | 갑자기 터짐    |

6️⃣ 어떻게 해결할까?
✅ 1. Gradient Clipping (기울기 클리핑)

기울기가 일정 값 이상이면 잘라버립니다.

예:

torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)


→ RNN, LSTM에서 많이 사용

✅ 2. 적절한 초기화 (Weight Initialization, 가중치 초기화)

Xavier Initialization

He Initialization

→ 분산 유지

✅ 3. Normalization

Batch Normalization

Layer Normalization

→ 값의 폭발 방지

✅ 4. Residual Connection (잔차 연결)

ResNet과 Transformer에서 사용.

→ 기울기 안정화

7️⃣ 직관적 비유

당신이 100명에게 소리를 전달합니다.

각 사람이 2배 크게 외친다면?

1 → 2 → 4 → 8 → 16 → ...


10번째면 1024배.

50번째면 도시가 무너집니다.

그게 Exploding Gradient입니다.

## feedforward neural network
피드포워드 신경망은 입력 → 은닉층(hidden layer) → 출력으로 한 방향(순방향)으로만 정보를 전달하는 신경망. 입력 데이터를 넣어서 결과를 예측하는 과정. 
📦 Transformer에서는 어떻게 쓰일까?
Transformer 인코더 또는 디코더 블록 안에서:
•	각 단어별로 self-attention 결과를 얻은 후,
•	각 단어별로 독립적으로, Feedforward Neural Network를 통과시킴

FFN(x) = max(0, xW1 + b1)W2 + b2
→ 두 개의 선형층 (Linear layer) 사이에 ReLU 같은 활성화 함수를 넣음.

 

🔍 왜 필요한가?
이유	설명
🎛 Self-Attention이 관계 파악	단어 간 상호작용에 집중
🧪 Feedforward는 정보 가공	각 단어의 표현을 개별적으로 더 복잡하게 가공

🧠 예시로 쉽게 보기
단어 "사과"에 대한 attention 결과가 [0.5, 0.1, 0.3, 0.2]라는 벡터로 나왔다고 가정하면,
1.	이 벡터에 W1, b1을 곱해서 1차 변환
2.	→ ReLU 적용 (음수는 0으로)
3.	→ 다시 W2, b2로 최종 출력
이 과정을 통해 "사과"라는 단어의 의미를 더 풍부하고 분리된 공간으로 표현할 수 있음.

✅ 요약 정리
특징	설명
구조	선형층 → ReLU → 선형층
정보 흐름	한 방향으로만 전달 (순방향)
역할	단어별 표현을 깊이 있게 가공
Transformer에서의 위치	Self-Attention 뒤에 배치됨 (잔차 연결 + LayerNorm 포함)

flash Attention
Transformer 모델의 어텐션 연산을 훨씬 빠르고 효율적으로 수행할 수 있도록 설계된 최적화된 알고리즘 및 구현 기법. **“메모리 낭비 없이 매우 빠르게 어텐션을 계산하는 기술”**

✅ Flash Attention 개념 요약
항목	설명
🎯 목적	GPU 메모리 사용 최소화 + 속도 향상
📍 위치	Transformer의 Self-Attention 연산부
📈 결과	기존보다 최대 2배 빠르고, 10배 이상 메모리 효율적
🏗️ 구현	CUDA 커널 기반, block-wise 방식으로 Softmax 연산 수행

✅ 왜 Flash Attention이 필요한가?
Transformer의 Self-Attention은 계산을 수행하는 과정에서 QKᵀ, Softmax, 곱셈 결과 등을 모두 메모리에 올려야 함. 특히 배치 크기나 문장 길이(n)가 커지면 O(n²) 메모리 폭증
✅ Flash Attention이 해결한 방식
1.	중간 연산(예: QKᵀ, Softmax 결과)을 저장하지 않음
2.	대신 블록 단위로 한 번에 계산 + 즉시 버퍼에서 제거
3.	GPU에서 매우 효율적인 register + shared memory만 사용
→ 즉, 필요한 계산만 하고, 필요 없는 메모리 사용은 회피

✅ 주요 효과
항목	기존 Attention	Flash Attention
연산 속도	느림	빠름 (최대 2x 이상)
메모리 사용	매우 큼	훨씬 작음 (최대 10x↓)
배치/시퀀스 확장성	제한적	훨씬 유연함

✅ 실제 사용 모델
•  GPT-NeoX, GPT-J, GPT-4 일부 구현
•  HuggingFace Transformers에서도 Flash Attention 사용 가능 (torch.compile, xformers, flash_attn 등)
•  OpenAI, Meta, MosaicML 등에서 사용

✅ 비유로 쉽게 말하면
기존 attention은 “모든 계산 결과를 한꺼번에 펼쳐놓고” 작업
Flash Attention은 “작업 구역을 작게 나눠서 계산하고 바로 정리”하는 방식
→ 빨라지고 깔끔함

✅ 요약 정리
항목	내용
용어	Flash Attention
목적	어텐션 연산의 속도 향상 + 메모리 절감
기술 핵심	중간 결과 저장 없이, 블록 단위로 직접 Softmax까지 계산
효과	연산 효율 극대화 (최대 2x 속도, 최대 10x 메모리 개선)
활용 모델	GPT-3.5/4, LLaMA2, Hugging Face 등

Fine Tuning
1️⃣ Fine-Tuning = 추가 학습인가?
✔ 결론:
**Fine-Tuning은 ‘이미 학습된 모델을 특정 목적에 맞게 다시 학습시키는 추가 학습’**이다.
하지만 핵심은 “무엇을, 왜, 어떻게” 추가로 학습하느냐다.

2️⃣ Fine-Tuning의 정확한 의미 (냉정하게 정리)
● Pre-training (사전 학습)
•	인터넷, 책, 코드, 위키 등 범용 데이터로 학습
•	언어 구조, 일반 지식 습득
•	비용 큼, 한 번만 수행
•	👉 “뇌를 만든다”
● Fine-Tuning (미세 조정 / 추가 학습)
•	이미 학습된 뇌를 특정 업무에 맞게 조정
•	데이터 규모는 작아도 됨
•	목적이 명확함
•	👉 “직무 교육”
📌 그래서 추가 학습은 맞지만
👉 “방향성이 강한 추가 학습” 이다.

3️⃣ 왜 그냥 추가 학습이라고 부르면 위험한가?
“추가 학습”이라고만 생각하면 이런 오해가 생긴다:
❌ “그럼 데이터 더 넣어서 계속 학습시키면 되는 거 아냐?”
→ 아니다.
Fine-Tuning은:
•	기존 지식을 덮어쓰지 않도록
•	학습률(Learning Rate)을 낮추고
•	특정 패턴만 강화한다
👉 즉,
기억을 망치지 않으면서 성격을 바꾸는 작업이다.

4️⃣ 실무 관점에서 딱 맞는 한국어 표현
영어	한국어로 가장 정확한 표현
Fine-Tuning	목적 지향 추가 학습
	사전학습 모델의 미세 조정
	도메인 특화 재학습


## FineWeb-Edu
Hugging Face에서 공개한 **“FineWeb”**이라는 웹 크롤링 기반 대규모 데이터셋에서,
“교육 콘텐츠(edu)”에 해당하는 고품질 웹페이지만 선별하여 만든 버전. 이 FineWeb Edu 데이터셋을 토크나이징 한 후 벡터화(embedding)하여, 직접 만드는 AI 모델의 학습용 데이터로 사용할 수 있음. 영어 데이터셋이므로 한글 입력에 대한 추론 서비스를 하려면, **추가적인 한국어 적응(fine-tuning 또는 instruction-tuning)**이 필요

✅ 핵심 요약
•	원본 FineWeb: 15조 토큰 규모의 웹 전체 데이터셋
→ Hugging Face 연구진이 고품질 데이터를 필터링해 집합화.
•	FineWeb Edu: 그 중에서 교육적인 웹페이지만 골라 만든 버전
→ 약 1.3조 토큰 규모

🔍 제작 방식
1.	LLM 기반 자동 평가
o	LLaMA 3 70B Instruct 같은 모델이 웹페이지를 0~5점 척도로 교육 콘텐츠 여부 평가.
2.	분류기 학습 및 적용
o	이 평가를 바탕으로 **텍스트 분류기(Classifier)**를 학습하고, 이를 전체 FineWeb에 적용해 고품질 교육 콘텐츠만 선별.
3.	데이터셋 구성
o	결과적으로 원본 15T 토큰에서 92%를 제거,
→ 1.3T 토큰짜리 FineWeb Edu 완성

🎯 FineWeb Edu의 특징
•	교육 콘텐츠에 특화되어 있어 학습 및 추론 성능이 증진됨
•	실제로 FineWeb Edu로 학습한 LLM은 **지식 중심 벤치마크 (MMLU, ARC, OpenBookQA 등)**에서 기존 데이터셋 기반 모델보다 우수한 성능을 보이는 것으로 나타남

📌 요약 정리
속성	설명
원본	15조 토큰 규모 FineWeb 데이터
FineWeb Edu	교육 콘텐츠만 추린 1.3조 토큰 데이터셋
의의	교육용 LLM 학습에 적합하며, 효율성과 성능 모두 개선됨

## Flan-T5 family of models
Google이 만든 T5 모델의 업그레이드 버전이며, Instruction fine-tuning을 통해 훨씬 강력한 성능을 가지게 된 모델군.

✅ 먼저, T5란?
•	T5 (Text-to-Text Transfer Transformer):
Google이 만든 Transformer 기반 모델로,
모든 NLP 작업을 "텍스트 → 텍스트" 형태로 처리

🔶 Flan-T5란?
Flan-T5 = T5 + Instruction Fine-tuning
•	FLAN: Fine-tuned LAnguage Net의 약자
•	Flan-T5는 원래 T5 모델을 “명령어(프롬프트)” 기반의 fine-tuning을 통해 개선한 버전입니다.
•	사용자가 "Summarize this", "Translate to German"처럼 명시적으로 작업을 지시하면,
그에 따라 정확하게 응답하도록 추가 학습된 모델

📦 Flan-T5 모델 패밀리 구성
Flan-T5는 다양한 크기의 모델로 제공됨.
모델 이름	파라미터 수	용도
Flan-T5-Small	약 80M	소규모 실험용
Flan-T5-Base	약 250M	기본 용도
Flan-T5-Large	약 770M	성능과 속도의 균형
Flan-T5-XL	약 3B	고성능 작업
Flan-T5-XXL	약 11B	최고 성능, 연구 및 고급 활용
👉 모두 Hugging Face에서 무료 사용 가능

🧠 Flan-T5의 특징
항목	설명
Instruction Tuning	프롬프트로 명령을 주면 더 잘 이해함 (ex: "Please summarize this")
Few-shot/Zero-shot 성능	적은 예시 혹은 예시 없이도 정확하게 작업 수행
범용성	요약, 번역, 질문 생성, 대화 등 다양한 NLP 작업 처리 가능
파라미터 효율	같은 사이즈의 T5보다 성능이 훨씬 높음

🧪 사용 예시 (Python/Hugging Face Transformers)
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")

input_text = "Translate English to French: I love you"
inputs = tokenizer(input_text, return_tensors="pt")

outputs = model.generate(**inputs)
print(tokenizer.decode(outputs[0]))
# → "Je t’aime"

✅ 요약
질문	답변
Flan-T5는 무엇인가요?	T5 모델을 명령어 기반으로 fine-tuning해서 더 똑똑하게 만든 모델.
어떤 작업에 쓰이나요?	번역, 요약, 문서 생성, 질의응답 등 거의 모든 텍스트 기반 작업
무엇이 특별한가요?	프롬프트만 바꾸면 다양한 작업 수행 가능, zero-shot 성능 우수

## Gated Recurrent Unit (GRU, 게이트 순환 유닛)
LSTM을 단순화한 버전

게이트 수를 줄여 계산량 ↓

성능은 LSTM(long short-term memory)과 비슷한 경우가 많음

## GELU activation function
1️⃣ Transformer Block 안에서 GELU의 위치부터 보자

Transformer block 안에는 크게 두 덩어리가 있습니다:

Self-Attention

Feed Forward Network (FFN, 피드포워드 네트워크)

GELU는 두 번째인 FFN 내부에서 사용됩니다.

구조는 대략 이렇게 생겼습니다:

입력 → Linear → GELU → Linear → 출력


즉,

첫 번째 선형 변환 (Linear Transformation, 선형 변환)

비선형성 추가 (GELU)

다시 선형 변환

여기서 GELU가 핵심적인 비선형성(non-linearity, 비선형성) 역할을 합니다.

2️⃣ 그럼 GELU가 뭐냐?

GELU는

Gaussian Error Linear Unit (가우시안 에러 선형 유닛)

이라는 활성화 함수입니다.

쉽게 말하면:

"입력을 완전히 자르지 않고, 부드럽게 걸러주는 함수"

입니다.

3️⃣ ReLU랑 뭐가 다른가?
🔹 ReLU (Rectified Linear Unit, 렉티파이드 리니어 유닛)

0보다 크면 그대로 통과

0보다 작으면 완전히 0으로 잘라버림

x > 0 → 그대로
x ≤ 0 → 0


굉장히 단순하고 강력하지만,
음수 영역은 완전히 죽여버립니다.

🔹 GELU는?

GELU는 이렇게 생각하면 됩니다:

"이 값이 유용할 확률(probability, 확률)에 따라 조금씩 통과시킨다"

음수라고 바로 0으로 자르지 않습니다.

큰 양수 → 거의 그대로 통과

큰 음수 → 거의 0에 가까워짐

0 근처 → 부드럽게 줄어듦

즉, 하드 컷이 아니라 소프트 필터링입니다.

4️⃣ 왜 Transformer는 GELU를 쓰는가?

GPT 계열 모델(예: GPT-2, GPT-3)은 대부분 GELU를 사용합니다.

이유는:

1️⃣ 더 부드러운 학습

ReLU는 경계에서 갑자기 꺾입니다.
GELU는 smooth (매끄러운) 함수입니다.

→ Gradient (기울기, 그래디언트)가 안정적입니다.

2️⃣ 확률적 해석이 가능

GELU는 입력 x에 대해:

“이 값이 중요할 확률만큼 곱해준다”

라는 해석이 가능합니다.

즉,

출력 = x × Φ(x)


여기서 Φ(x)는 Gaussian CDF (가우시안 누적 분포 함수)입니다.

이게 중요한 이유는:

Transformer는 확률 기반 모델입니다.
Softmax, Attention score, 모두 확률 구조죠.

GELU는 이 확률적 철학과 잘 맞습니다.

5️⃣ 직관적으로 이해해 보자

당신이 지금 NPU에서 모델을 최적화한다고 생각해 봅시다.

FFN에서 수천 개의 값이 생성됩니다.

그 중:

아주 의미 있는 값

애매한 값

거의 노이즈 같은 값

이 섞여 있습니다.

ReLU는:

애매한 값도 살리고, 음수는 다 죽여버립니다.

GELU는:

애매한 값은 반쯤만 살립니다.

이게 Transformer에서 더 정교한 표현력을 만들어 줍니다.

6️⃣ 한 문장 정리

GELU는

Transformer의 Feed Forward Network에서
입력을 확률적으로 부드럽게 필터링하는 비선형 활성화 함수다.

## generative AI model
🧠 1. 생성형 AI(Generative AI)란?
생성형 AI는 새로운 콘텐츠를 만들어내는 인공지능 기술. 사람이 만든 것처럼 보이는 텍스트, 이미지, 음악, 음성, 코드, 영상 등 다양한 데이터를 생성할 수 있음.
📌 핵심은 **"무(無)에서 유(有)를 창조"**한다는 점. → 기존 AI는 주로 분류(classification)·예측(prediction)이 목적이었지만, 생성형 AI는 **"창조적 생성"**이 목적.

🎨 2. 생성형 AI의 종류
종류	생성 대상	예시
텍스트 생성 AI	뉴스, 이메일, 시, 소설, 코드	ChatGPT, GPT-4, Claude, Gemini
이미지 생성 AI	그림, 사진, 디자인	DALL·E, Stable Diffusion, Midjourney
음성 생성 AI	사람 목소리, TTS	Google TTS, ElevenLabs
음악/소리 생성 AI	음악, 효과음	Riffusion, Jukebox, Suno.ai
비디오 생성 AI	짧은 영상, 모션	RunwayML, Sora (OpenAI), Pika Labs
멀티모달 AI	텍스트+이미지+음성 통합	GPT-4o, Gemini 1.5, Claude 3.5

🏗️ 3. 생성형 AI의 작동 원리
1.	대규모 데이터를 수집
(예: 책, 웹문서, 이미지, 음성 등)
2.	패턴과 문맥을 학습
→ 딥러닝 기반 모델 (예: Transformer, Diffusion)
3.	새로운 데이터 생성
→ 기존과 유사하지만 완전히 새로운 결과물
예시: "강아지가 기타 치는 장면 그려줘" → 아무도 본 적 없지만 그럴듯한 이미지 생성

✅ 4. 생성형 AI의 장점과 단점
장점	단점
콘텐츠 생산 비용 절감	허위 정보 생성(hallucination) 가능
개인 맞춤형 생성 가능	저작권 침해 우려
창의적 표현 가능	편향된 데이터로 문제 유발 가능
상호작용형 응답	악용 위험 (딥페이크, 스팸 등)

🧪 5. 실제 사용 사례
분야	활용 예시
교육	학생 수준에 맞는 설명 생성, 시험 문제 자동 생성
마케팅/광고	광고 문구, 블로그 글 자동 생성
헬스케어	의료 요약, 의사-환자 대화 시뮬레이션
엔터테인먼트	소설 창작, 게임 대사 생성, 음악 작곡
프로그래밍	코드 자동 완성, 오류 수정 제안 (Copilot 등)
고객 서비스	24시간 챗봇, 다국어 응답

💡 생성형 AI vs 기존 AI
항목	기존 AI	생성형 AI
주로 하는 일	분류, 예측	텍스트·이미지 생성
입력 → 출력	입력 → 라벨	입력 → 새 콘텐츠
창의성	낮음	높음
예시	스팸 분류기, 진단기	문장 생성기, 그림 그리는 AI

🧭 요약 정리
•	생성형 AI는 데이터를 단순 분석하는 AI가 아니라, → 사람처럼 창의적으로 새로운 결과물을 만들어내는 AI.
•	기술적으로는 GPT, Diffusion, GAN 등이 중심이며, → 산업적으로는 교육, 의료, 디자인, 콘텐츠 산업에서 혁신

## Gradient Clipping
gradient clipping of 1
•	역전파 중 gradient 폭주 방지용
•	1로 제한하면, 그래디언트가 너무 커져도 안전하게 잘림.

1️⃣ Gradient(그래디언트, 기울기)부터 다시 짚자

학습은 기본적으로 이렇게 진행됩니다:

weight=weight−learning_rate×gradient

여기서 **gradient(그래디언트)**는
👉 “얼마나, 어느 방향으로 수정해야 하는가”를 알려주는 값입니다.

2️⃣ 문제 상황: Gradient Explosion (그래디언트 폭발)

어떤 step에서 gradient 값이 갑자기 매우 커질 수 있습니다.

예를 들어:

평소 gradient norm이 3~5 수준

갑자기 2000이 나옴

그럼 어떻게 될까요?

update=lr×2000

가중치가 한 번에 멀리 튀어버립니다.
→ Loss 폭발
→ NaN
→ 학습 망가짐

특히:

Transformer

Deep network

Mixed precision

큰 batch

이 환경에서는 더 자주 발생합니다.

3️⃣ Gradient Clipping(그래디언트 클리핑)이란?

아주 단순합니다.

Gradient가 너무 크면, 강제로 잘라버린다.

즉,

일정 임계값(threshold)을 넘으면

그 크기를 제한(limit)한다

이걸 **Gradient Clipping(그래디언트 클리핑)**이라고 합니다.

4️⃣ 어떻게 자르느냐?

대표적인 방식은:

(1) Norm Clipping (노름 클리핑)

Gradient 전체 크기(norm)가 기준값을 넘으면 즉, 방향은 유지하고 크기만 줄입니다.

이게 가장 많이 쓰입니다.

(2) Value Clipping (값 클리핑)

각 gradient 값을 직접 자릅니다.

예:

-1 ~ 1 사이로 제한

하지만 LLM에서는 보통 norm clipping을 사용합니다.

5️⃣ 직관적 비유

Learning rate는 “속도”라면
Gradient는 “가속도”입니다.

Gradient clipping은

급발진 방지 장치

자동차로 비유하면:

평소엔 잘 달림

갑자기 미끄러지면 브레이크 개입

6️⃣ 왜 LLM에서 거의 필수일까?

대형 모델에서는:

Layer 수 많음

Attention 스케일 민감

초기 단계 불안정

FP16 (Half precision, 반정밀도) 사용

이 조건이 겹치면
gradient explosion 확률이 높습니다.

그래서 실제 GPT 학습 코드에는 거의 항상:

torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

이런 코드가 들어갑니다.

## greedy decoding
언어 모델에서 문장을 생성할 때 사용하는 가장 간단한 디코딩 전략 중 하나
✅ 한 줄 요약
Greedy decoding은
매 단계마다 가장 확률이 높은 토큰 하나만 선택해서 문장을 이어가는 방식
✅ 예를 들어 설명
예시 입력: "나는 오늘"
모델은 다음 단어에 대해 이런 확률 분포(Probability Distribution)를 줄 수 있음:
후보 토큰	확률
"밥"	45%
"학교"	35%
"비"	15%
"운동"	5%
Greedy decoding은 여기서
👉 **확률이 가장 높은 “밥”**을 고름.
그리고 이어서 "나는 오늘 밥"이라는 문장을 만들고,
그다음 "밥" 뒤에 올 단어도 가장 확률 높은 토큰만 고르며 계속 진행.
✅ 특징
항목	설명
🔹 장점	빠르고 계산 비용이 낮음 (가장 단순)
🔸 단점	문장이 반복되거나, 어색하거나, 창의성이 부족할 수 있음
🔹 예시	"Hello, my name is ChatGPT" 같이 뻔한 답을 잘 만듦
🔸 비유	항상 가장 점수가 높은 문항만 찍는 수험생처럼 작동함
✅ 예시 비교: Greedy vs Sampling
입력: "나는 오늘"
전략	생성 예시
Greedy	나는 오늘 밥 먹고 집에 갔다.
Sampling	나는 오늘 학교 가다가 비를 맞았다.
•  Greedy는 예측 가능한 문장을 생성
•  Sampling은 더 창의적이지만 불확실성 존재
✅ 언제 사용하나요?
•	챗봇의 정답형 응답, 정보성 요약, 기계 번역 등에서 사용
•	하지만 최근에는 top-k, top-p (nucleus sampling), beam search 등 더 발전된 전략이 많이 사용됨

## grouped-query attention (GQA)
Transformer 모델의 어텐션 메커니즘을 경량화하는 기법으로, 특히 GPT-4와 같은 최신 대형 언어 모델에서 성능 손실 없이 연산량과 메모리 사용을 줄이기 위해 사용.

✅ 1. 기본 배경: 기존 Multi-Head Attention의 구조
먼저, 일반적인 Multi-Head Attention 구조를 정리하면:
•	입력으로부터 여러 개의 **Query (Q), Key (K), Value (V)**를 생성
•	각 Attention Head마다 Q, K, V를 따로 계산
•	계산량이 많고, 메모리도 많이 듬 → 특히 V가 많을수록 비용 증가

✅ 2. Grouped-Query Attention(GQA)의 핵심 아이디어
"여러 Query Head가 같은 Key/Value를 공유하게 만든다"
📌 구조 비교
방식	Query Head 수	Key/Value Head 수	특징
Multi-Head Attention	12	12	모든 Q/K/V 독립
Grouped-Query Attention	12	4 (예시)	Q는 12개, K/V는 4개로 공유

💡 효과:
•	계산량 감소 (특히 V 계산)
•	메모리 사용 절약
•	성능 유지하면서 더 큰 모델 구성 가능

✅ 3. 왜 쓰는가? (도입 이유)
•	대규모 LLM(GPT-4, PaLM, Claude 등)에서는 V가 차지하는 메모리/연산량이 크기 때문에 GQA로 최적화함
•	실제로 GPT-4 및 PaLM 2 논문/인터뷰에서 GQA 사용이 확인됨

✅ 4. 예시로 쉽게 설명하면?
🍎 기존 방식: 학생 12명이 각자 Q, K, V 세트를 가지고 서로 질문하고 대답
🍊 GQA 방식: 학생 12명이 **자기만의 질문(Q)**은 있지만, **같은 참고서(K/V)**를 보고 대답
→ 질문은 다양하지만, **공통된 지식 기반(K/V)**을 쓰는 구조

✅ 5. GQA vs 관련 개념
개념	설명
Multi-Query Attention (MQA)	Q는 여러 개, K/V는 하나만 사용
Grouped-Query Attention (GQA)	Q는 여러 개, K/V는 그룹 단위로 공유
Multi-Head Attention	Q/K/V 모두 독립적으로 계산

 

✅ 요약 정리
항목	설명
용어	Grouped-Query Attention (GQA)
위치	Transformer의 Attention 구조 최적화
목적	메모리, 연산량 절감 while 성능 유지
방식	여러 Query가 일부 Key/Value를 공유
사용 모델	GPT-4, PaLM 2, Claude 등 대형 LLM
관련 개념	Multi-Head, Multi-Query, GQA

## HDBSCAN
계층적 밀도 기반 클러스터 탐색. HDBSCAN은 DBSCAN의 향상된 버전

📌 DBSCAN vs HDBSCAN 비교표
항목	DBSCAN	HDBSCAN
✅ 전체 이름	Density-Based Spatial Clustering of Applications with Noise	Hierarchical Density-Based Spatial Clustering of Applications with Noise
🎯 목표	밀도 기반 클러스터 탐색	계층적 밀도 기반 클러스터 탐색
🔧 주요 파라미터	eps (반경), min_samples (최소 점 수)	min_cluster_size (최소 클러스터 크기), min_samples (선택적)
📐 클러스터 수	자동 결정	더 유연하게 자동 결정
📦 클러스터 구조	단일 밀도 기준	여러 밀도 수준을 고려하여 계층 구조 생성 후 최적 클러스터 선택
📊 복잡한 분포	다소 한계 있음	매우 복잡한 분포도 탐지 가능
🚫 노이즈 처리	있음 (label = -1)	더 정교하게 처리
🧠 내부 알고리즘	단일 밀도 영역에서 확장	계층적 트리 → 안정성 기반 평면화
💻 연산 복잡도	빠름 (상대적으로 단순)	약간 느릴 수 있음 (복잡도 증가)

🔍 핵심 차이점 요약
구분	DBSCAN	HDBSCAN
밀도 기준	하나의 global 밀도값(eps)에 의존	지역마다 다른 밀도도 반영 가능
클러스터 분리 기준	반경 eps 기준으로 점들을 연결	**밀도 기반 트리(계층 구조)**를 만들어 클러스터 분리
유연성	밀도 차이가 큰 데이터에 약함	다양한 밀도 클러스터를 유연하게 탐지 가능
설정 난이도	eps 설정이 까다로움	min_cluster_size는 비교적 직관적

🟩 언제 HDBSCAN을 쓰면 좋을까?
•	클러스터의 밀도 크기가 제각각인 경우
•	DBSCAN으로는 클러스터가 잘 안 나뉘는 경우
•	eps 값을 설정하기 어렵거나 여러 번 튜닝해야 할 때
•	더 정교한 클러스터링과 계층 구조 분석이 필요한 경우

## hidden state
1️⃣ 히든 상태(hidden state)가 뭐냐?

히든 상태는 👉 지금까지 읽은 정보의 요약 벡터입니다.

RNN은 문장을 한 단어씩 읽습니다.

예를 들어:

I → love → my → daughter

RNN은

"I"를 읽고 내부 값(hidden state)을 만든 뒤

그 값을 가지고 "love"를 읽고

다시 업데이트하고

계속 누적합니다.

이때 매 순간 내부에 유지되는 값이 히든 상태(hidden state) 입니다.

수식적으로는:
h_t = f(x_t, h_{t-1})

- x_t = 현재 입력
- h_{t-1} = 이전 기억
- h_t = 업데이트된 기억

즉,

현재 정보 + 이전 기억 → 새로운 기억

2️⃣ 왜 필요하냐?

텍스트는 순서(sequence) 가 중요합니다.

예를 들어:

“not good”
“good”

이건 완전히 다른 의미입니다.

만약 이전 단어를 기억하지 못하면,
모델은 단어를 독립적으로만 보게 됩니다.

👉 그럼 문맥(context)을 이해할 수 없습니다.

히든 상태는
"지금까지 무슨 말이 나왔는지"를 기억하는 메모리 역할을 합니다.

3️⃣ 무슨 역할을 하냐?

히든 상태는 세 가지 역할을 합니다.

① 문맥 저장 (Context memory)

문장 앞부분 정보를 뒤로 전달합니다.

② 의미 압축 (Semantic compression)

문장 전체 의미를 벡터 하나로 압축하려고 합니다.

③ 다음 단어 예측에 사용

디코더에서는:
P(next word | hidden state)

즉,
히든 상태가 곧 다음 단어 예측의 근거 데이터입니다.

4️⃣ 목표는 뭐냐?

목표는 이것입니다:

문장 전체 의미를 벡터 하나에 담는 것

하지만 여기서 큰 문제가 생깁니다.

문장이 길어지면?

앞 정보가 희미해짐

기억이 압축되며 손실 발생

long-range dependency 문제 발생

이 한계 때문에 등장한 것이
👉 Attention 메커니즘

1️⃣ Transformer에도 hidden state가 있습니까?

있습니다.
하지만 의미가 다릅니다.

2️⃣ RNN의 hidden state

RNN에서의 히든 상태(hidden state) 는:

시간에 따라 하나씩 업데이트됨

이전 정보를 압축해서 하나의 벡터로 유지

h_t 하나가 과거 전체를 대표

즉,

“지금까지 읽은 모든 정보를 하나에 담는 기억”

입니다.

문제는?
→ 길어질수록 기억 손실.

3️⃣ Transformer의 hidden state는 무엇인가?

Transformer에서는
“하나의 기억 벡터”가 아닙니다.

대신:

👉 각 토큰마다 하나의 벡터가 유지됩니다.

예:

입력:
I love my daughter

Transformer 내부에서는:
[I_vector]
[love_vector]
[my_vector]
[daughter_vector]

[daughter_vector]
이 벡터들이 layer를 지나며 계속 업데이트됩니다.

즉,

Transformer의 hidden state = 각 토큰의 현재 표현 벡터

4️⃣ 가장 큰 차이

RNN:
문장 → 하나의 벡터로 압축

Transformer:
문장 → 토큰 수만큼 벡터 유지

압축하지 않습니다.

모든 토큰을 병렬로 유지합니다.

5️⃣ 왜 이렇게 바꿨을까?

RNN은:
순차 처리 (느림)
하나에 압축 (정보 손실)

Transformer는:
병렬 처리
모든 토큰이 서로 직접 참조 가능 (attention)

그래서 긴 문장에서도
앞 단어 ↔ 뒤 단어 직접 연결이 가능합니다.

6️⃣ 결론
| 구분    | RNN hidden state   | Transformer hidden state |
| ----- | ------------------ | ------------------------ |
| 개수    | 1개 (시간별 업데이트)      | 토큰 개수만큼                  |
| 구조    | 순차적                | 병렬적                      |
| 정보 방식 | 압축 기억              | 분산 표현 유지                 |
| 문제    | long dependency 약함 | attention으로 해결           |

## hyperparameter
**하이퍼파라미터(hyperparameter)**는
👉 모델이 학습을 “어떻게” 할지 사람이 미리 정해 주는 설정값.


1️⃣ 파라미터(parameter)

모델이 학습하면서 스스로 배움
예: 가중치(weight), 바이어스(bias)
데이터 보고 자동으로 업데이트됨

2️⃣ 하이퍼파라미터(hyperparameter)

학습 전에 사람이 정함
학습 도중에 자동으로 변하지 않음
학습 성능, 안정성, 속도를 좌우함

hyperparameter 예)
batch size (배치 크기)
→ 한 번에 몇 개의 샘플을 묶어서 학습할 것인가

즉,
batch size = 1 → 메모리 적게 씀, 업데이트가 흔들림
batch size = 256 → 안정적, 메모리 많이 필요

👉 정답은 없고, 실험으로 찾는 값.

LLM에서 대표적인 하이퍼파라미터들

•	batch size: 한 번에 몇 개 문장을 학습?
•	context size (max_length): 한 입력에 토큰을 몇 개까지?
•	stride: 슬라이딩 윈도우를 몇 칸씩 이동?
•	learning rate: 파라미터를 얼마나 세게 업데이트?
•	number of layers / heads: 모델 구조 관련 설정

이것들은 전부
👉 데이터가 아니라, 학습 전략에 대한 결정.

왜 “하이퍼”라는 말을 쓰냐?

“hyper”는 모델 위(meta-level)에 있는 설정 이라는 뜻.

•	파라미터 → 모델 안
•	하이퍼파라미터 → 모델 위에서 조종

그래서 이름부터 다름.

## inference
이미 학습이 끝난 AI 모델이 실제 입력을 받아서 예측값 또는 출력을 생성하는 과정을 의미. 즉, AI 모델이 학습한 지식을 바탕으로 실제 데이터에 대해 “예측”하거나 “응답”하는 실행 단계. 

📊 예시로 쉽게 설명
상황	학습 (Training)	추론 (Inference)
ChatGPT가 이메일 작성	수많은 글 데이터를 학습하며 문장 구조를 배움	사용자가 "사과 이메일 써줘"라고 요청하면 실제로 문장을 생성
이미지 분류 모델	고양이/개 수천 장 이미지로 모델을 학습	새로운 사진이 들어오면 “고양이”라고 판단
음성 인식	다양한 음성과 텍스트 짝을 학습	사용자가 말하면 텍스트로 변환해서 보여줌

🧠 기술적으로는 어떤 일?
•	입력 데이터를 받음 (예: 텍스트, 이미지 등)
•	학습된 파라미터(weight)를 고정한 상태에서
•	**순전파(forward pass)**만 수행하여
•	출력(예측 결과)을 반환함
📌 가중치를 업데이트하거나 손실 함수를 계산하지 않음.

🛠️ 추론 vs 학습 비교
항목	학습 (Training)	추론 (Inference)
목적	모델을 학습시키기 위해	실제 데이터를 처리하기 위해
데이터	입력 + 정답(label)	입력만 있음
연산	손실 계산 + 가중치 업데이트	순전파만 수행
자원 소모	높음 (GPU 메모리 등)	상대적으로 낮음 (속도 중요)

🧩 왜 중요한가?
•	실제 사용자 경험은 모두 inference 단계에서 발생
예: ChatGPT에 질문 → 답변 생성
예: 얼굴 인식 출입 시스템 → 실시간 추론
•	최적화 대상
기업은 inference 속도와 비용을 줄이기 위해
➤ NPU, quantization, distillation 등 다양한 기술을 적용

✅ 정리
질문	답변
Inference란?	AI가 학습한 결과를 이용해 실제 입력에 대해 예측 또는 응답을 생성하는 과정
Training과 차이점?	학습은 모델을 만드는 과정, 추론은 만들어진 모델을 사용하는 과정

## input vector
모델이 토큰을 수치적으로 표현한 벡터 (숫자 배열)

🧠 전체 맥락을 쉬운 예로 풀면
예를 들어 문장이 이렇다고 해 보면:
"The cat sat on the mat."
•  이 문장은 단어 단위로 쪼개져서 → 토큰화(tokenize) 됨
•  각 토큰이 **임베딩 벡터 + 위치 정보(positional encoding)**로 변환됨
•  각 토큰은 Transformer를 자기 토큰 스트림 경로를 따라 흘러가며 처리됨
•  마지막에는 새로운 벡터가 출력됨
→ 예측하거나, 다음 단어를 생성하거나, 분류에 사용됨

## instruction tuning (지시/명령 튜닝)
언어 모델(예: LLM)을 단순 텍스트 예측기에서, 명령을 이해하고 추론하는 AI로 발전시키기 위한 추가 훈련 단계

💡 목적:
“명령(instruction)을 주면 그에 적절하게 반응하도록 모델을 학습시키는 것”

📘 예시:
❌ Pretrained 모델:
입력: Translate “cat” to French  
출력: Translate “cat” to French
✅ Instruction Tuned 모델:
입력: Translate “cat” to French  
출력: chat

🛠️ 방식:
•	모델이 이해할 수 있도록 다음 같은 명령어-정답 쌍 데이터로 supervised fine-tuning 진행:
- "Summarize this article:"
- "Translate to Korean:"
- "Write an email requesting a refund:"

대표적인 Instruction 튜닝 데이터셋:
•	FLAN
•	OpenAssistant
•	Dolly
•	ShareGPT (대화 기반)

## Intra-Document Masking
LLM(대형 언어 모델) 학습에서 특히 Document Packing을 사용할 때 등장하는 개념. 쉽게 말하면, **"같은 문서 안에서도 일부 정보를 모델이 보지 못하도록 가리는 마스킹 기법"**

✅ 한 줄 정의
Intra-Document Masking이란,
**하나의 문서 내부(intra-document)**에서도
특정 부분만 모델이 볼 수 있도록 마스킹(masking) 처리하여
정보 유출(leakage)을 방지하고, 학습 효과를 높이는 기법

🔍 왜 이런 기법이 필요한가?
Transformer는 기본적으로 자기 자신(attention)에게 있는 모든 토큰을 참조할 수 있음.
하지만 학습 중 다음과 같은 상황이 발생할 수 있음.

📦 예시: Document Packing + Pretraining
[문서 A]: 문장 1. 문장 2. 문장 3.
[문서 B]: 문장 1. 문장 2. 문장 3.
→ 이런 문서들을 이어붙여 하나의 context로 학습하게 되면:
•	문서 A 안에서, 나중에 나와야 할 정답(예: 요약 문장)을
•	모델이 앞에 있는 본문을 넘어서 미리 볼 수 있게 되는 문제가 생김.
📌 이걸 방지하려면:
같은 문서 안에서도 "정답 토큰"이 있는 영역은 가려야 함
→ 그래서 나온 것이 Intra-Document Masking

✅ 어떻게 마스킹하나?
Transformer에서는 attention mask라는 행렬을 통해 "어떤 토큰이 어떤 토큰을 볼 수 있는지"를 제어.
Intra-Document Masking에서는:
•	한 문서 안에서도, 정답 영역을 마스킹해서 모델이 정답을 미리 보지 못하게 함.
✅ 용도 예시
사용 맥락	설명
🔤 Language Modeling	문서 내 요약, 정답, 해설 등이 본문에 의존해야 하는 경우 사용
📚 Instruction Tuning	“질문 + 정답” 구조에서 정답이 앞 질문을 넘어 참고하지 않게 제한
🧠 Retrieval QA	문서에서 정답을 찾는 과제에서, 정답만 집중하게 만들고 유출 방지

✅ Intra vs Inter-Document Masking
구분	설명
Intra-Document Masking	같은 문서 내부에서 토큰 간의 마스킹 처리
Inter-Document Masking	문서 간에 경계 마스킹 (다른 문서는 보지 못하게 함)
→ Packing 시 주로 사용	

✅ 요약
항목	설명
무엇인가?	같은 문서 안에서도 일부 정보만 보도록 제한하는 마스킹 방식
왜 필요한가?	정보 유출 방지, 정답을 미리 참고하는 현상 방지
언제 쓰나?	문서 내 요약, QA, instruction tuning, multi-task pretraining 등
효과	더 정확하고 일반화된 학습, leakage-free 학습 파이프라인 가능

## isaac
NVIDIA가 개발한 로봇용 AI 플랫폼. NVIDIA Isaac은 로봇이 “보고, 생각하고, 움직이는 것”을 가능하게 해 주는 AI 및 시뮬레이션 소프트웨어 플랫폼.

🤖 무엇을 위한 건가요?
사람처럼 자율적으로 움직이고 판단하는 로봇을 만들기 위해 필요한 도구들을 NVIDIA가 한데 모아 제공하는 것.

🧩 NVIDIA Isaac 3가지 구성 요소
구성 요소	설명
Isaac SDK	실제 로봇에 올려서 사용하는 AI 소프트웨어 개발 키트 (센서, 제어, 탐색 등)
Isaac Sim	Omniverse 기반 시뮬레이터 – 현실 같은 가상 공간에서 로봇을 훈련하고 테스트 가능
Isaac ROS	**ROS(Robot Operating System)**와 연동되는 모듈 – LiDAR, 카메라, SLAM 등 실시간 처리에 강함

🧠 무엇을 할 수 있나요?
기능	설명
🦾 SLAM	로봇이 자기 위치를 파악하고 지도1를 만드는 기능 (Simultaneous Localization and Mapping)
👁️ 시각 인식	카메라/라이다 영상으로 사물 감지, 거리 측정 등 수행
🛣️ 자율 주행	물류창고/공장 등에서 스스로 경로 찾고 움직이는 기능
🌍 시뮬레이션	실제 하드웨어 없이도 가상공간에서 테스트 및 훈련 가능 (Omniverse 기반)
지도1: 로봇 주변의 실내 또는 실외 환경을 스스로 스캔해서 만든 “공간 구조 정보”

🔍 대표 적용 분야
분야	예시
스마트 공장	자율 이동 로봇(AMR), 로봇팔 조작
물류	로봇이 박스 운반, 자동 창고 정리
소매점	실시간 매대 스캐닝 로봇
연구	시뮬레이션 기반 로봇 학습, 강화학습 적용 등

💻 하드웨어와도 연결됨
Isaac 플랫폼은 NVIDIA의 Jetson(젯슨) 시리즈와도 잘 연동됨.
Jetson 종류	용도
Jetson Nano	저가형 로봇 개발용
Jetson Xavier	고성능 자율 로봇용
Jetson Orin	최신 고성능 엣지 AI 모듈

✅ 요약 정리
항목	설명
이름	NVIDIA Isaac
목적	로봇을 위한 AI + 시뮬레이션 플랫폼
구성	Isaac SDK (실행), Isaac Sim (훈련), Isaac ROS (연동)
주요 기능	자율 주행, SLAM, 감지, 제어 등
활용 분야	물류, 제조, 서비스 로봇, AI 연구

## iteration
iteration은
👉 모델이 “한 번 업데이트”되는 단위다.

정확히 말하면:

한 batch를 모델에 넣고 → loss 계산하고 → gradient 계산하고 → 파라미터를 한 번 업데이트하는 전체 과정 1회

이게 iteration 1번이다.

epoch이랑 헷갈리면 안 된다

여기서 반드시 구분해야 한다.

🔁 iteration

기준: batch

의미: 파라미터 1회 업데이트

수:
iteration 수 = 전체 데이터 개수 / batch size

🔄 epoch

기준: dataset 전체

의미: 전체 데이터를 한 바퀴 다 본 것

epoch 1 = 여러 iteration의 묶음

숫자로 바로 감 잡자

데이터가 10,000개 있다고 하자.

batch size = 100

iteration 수 = 10,000 / 100 = 100 iterations

epoch 1 = 100번 iteration

batch size = 1

iteration 수 = 10,000

epoch 1 = 10,000번 iteration

👉 batch size가 커질수록 iteration 수는 줄어든다

왜 iteration이 중요한가?
1️⃣ learning rate는 iteration 기준이다

learning rate는
→ iteration마다 적용

iteration 수가 달라지면
→ 같은 epoch이라도 학습 강도 자체가 달라짐

그래서:

batch size 바꾸면

learning rate도 같이 손봐야 함

2️⃣ “학습이 느리다”는 말의 정체

사람들이 말하는:

“학습이 느리다”

대부분은:

iteration 대비 수렴이 느린 것

또는

epoch 기준으로만 착각하는 것

실제로 중요한 건:

몇 iteration 만에 loss가 내려가느냐

3️⃣ 큰 batch의 숨은 비용

batch size ↑
→ iteration 수 ↓

겉으로 보면:

“iteration 줄었네? 빠르네?”

현실은:

iteration 하나가 덜 효과적

수렴에 필요한 iteration 수 증가

그래서:

iteration은 줄었는데, 전체 학습 시간은 늘 수 있음

## k-means
가장 널리 사용되는 비지도 학습(Unsupervised Learning) 기반의 클러스터링(Clustering) 알고리즘.
•	K: 원하는 클러스터 수 (미리 지정함)
•	Means: 각 클러스터의 중심값, 즉 평균(Centroid)

🔧 어떻게 작동하는가?
1.	K개의 중심(Centroid)을 무작위로 초기화
2.	각 데이터 포인트를 가장 가까운 중심점에 할당 (클러스터 형성)
3.	각 클러스터에 대해 새로운 중심점 계산 (평균값)
4.	중심점이 더 이상 움직이지 않거나, 변화가 매우 작을 때까지 2~3 반복

📊 예시 그림
[데이터 포인트들] → [초기 중심점 설정] → [클러스터 할당] → [새로운 중심점 계산] → [반복]

💡 예를 들어 보면
쇼핑몰 고객 데이터를 클러스터링할 때:
•	연령, 소비 금액 등 기준으로
•	K=3이라면, 고객을 3개의 그룹(청년/중년/노년)으로 나눌 수 있음

✅ 장점
장점	설명
⚡ 빠름	큰 데이터에도 잘 작동함
🔍 직관적	시각화하기 좋고 이해하기 쉬움
🧩 간단	구현도 쉬움 (Python에서 sklearn으로 바로 사용 가능)

❌ 단점
🎯 K 값을 사전에 정해야 함
📦 클러스터는 **구형(Spherical)**이라고 가정 → 복잡한 모양에 약함
🔄 초기 중심 선택에 따라 결과가 달라짐
❌ 이상치(Outlier)에 민감함

## KV cache size
AI 모델 특히 Transformer 계열(예: GPT, LLaMA, Mistral 등)에서 나오는 **KV cache**는 추론(inference) 성능과 밀접하게 관련된 핵심 개념

✅ 한 줄 정의
KV cache란?
Self-Attention에서 생성된 Key와 Value 값을 저장해 두고,
다음 토큰을 생성할 때 다시 계산하지 않도록 캐싱해 놓은 메모리 공간

🔍 좀 더 쉽게 설명하면…
🧠 Transformer는 다음과 같이 동작:
각 토큰을 처리할 때마다:
→ 현재 토큰의 Query × 이전 토큰들의 Key & Value 조합 → Attention → 다음 단어 예측
👉 그런데 이걸 매번 계산하면 너무 느립니다.
그래서 이미 계산된 Key/Value를 메모리에 저장해놓고 재사용하는 거예요.
이게 바로 KV cache
✅ 왜 중요한가?
이유	설명
⚡ 추론 속도 향상	이미 생성한 Key/Value를 다시 계산하지 않아도 됨
💾 메모리 사용량 증가	캐시가 커질수록 GPU 메모리도 더 사용됨
📏 길이 제한과 연관	context length가 길어질수록 → KV cache 크기도 커짐
🔁 streaming/incremental decoding에 필수	한 글자씩 순차로 생성하는 구조에 꼭 필요함
긴 맥락(Long Context) 유지에 필수	LLM이 “앞에 한 말”을 기억하는 이유가
바로 KV Cache 덕분

🧮 예시: KV cache의 크기
Transformer 구조에서 일반적으로:
KV_cache_size = num_layers × batch_size × num_heads × sequence_length × head_dim
예를 들어:
•	LLaMA 7B
•	32 layers
•	32 heads
•	sequence length 2048
•	head_dim 128
→ KV cache만 수백 MB~GB에 이름

✅ 정리 요약
항목	설명
KV란?	Key와 Value (Self-Attention에 필요한 내부 벡터)
KV cache란?	이전 토큰의 Key/Value를 메모리에 저장해서 재사용하는 구조
왜 쓰나?	추론 속도 향상, 중복 계산 방지
trade-off	속도는 빨라지지만 → 메모리 많이 먹음

✅ 시각적으로 요약
[토큰1] → Q1 + K1,V1 (KV cache 저장)
[토큰2] → Q2 + K1,V1 사용 (재계산 X)
[토큰3] → Q3 + K1~K2, V1~V2 사용
...

✅ KV-cache가 없으면 생기는 문제

예를 들어

1000 토큰 문장이 있다고 가정하면

KV-cache 없음

1 + 2 + 3 + ... + 1000
≈ 500,000 계산

KV-cache 있음

1000번 계산

즉

추론 속도가 10배 이상 빨라질 수 있습니다.

그래서

LLM inference에서는 KV-cache가 거의 필수 기술입니다.

## Labeled data
✅ "Labeled data"는 **“정답(레이블)이 포함된 데이터”**를 말하며,
→ Hugging Face에서 제공하는 많은 데이터셋들이 labeled data
🔍 핵심 개념 정리
🔸 ✅ Labeled Data란?
입력(input) + 정답(label)이 짝을 이루는 데이터
입력 (Input)	정답 (Label)
"이 영화 너무 좋아!"	긍정 (Positive / 1)
"별로였어. 다시 안 봐."	부정 (Negative / 0)
📌 이렇게 **“사람이 정답을 붙인 데이터”**를 labeled data라고 함.

💡 예를 들어, Hugging Face의 SST-2 데이터셋은:
from datasets import load_dataset

dataset = load_dataset("glue", "sst2")
print(dataset["train"][0])

출력:
{
  'sentence': 'it took me about 20 minutes to realize that i was not watching a comedy .',
  'label': 0  # ← 이게 "labeled"의 핵심!
}

🧠 정리해서 비교
개념	설명	예시
Labeled Data	입력 + 정답이 있는 데이터	문장 + 감정(label)
Unlabeled Data	정답이 없는 데이터	문장만 있음
Tokenized Data	텍스트를 모델 입력 형식(ID 벡터 등)으로 변환한 것	"좋아" → [4532, 208]
Embedded Data	토크나이즈된 텍스트를 벡터로 변환한 것	[0.12, -0.88, ..., 1.32]

## Langchain
**LangChain(랭체인)**은 LLM(Language Large Model, 대형 언어 모델) — 예를 들어 GPT-4, Claude, Gemini, Qwen-3 같은 모델 — 을 다른 데이터, API, 또는 도구들과 연결해 주는 프레임워크. “LangChain”은 회사 이름이기도 하고, 동시에 그 회사가 만든 프레임워크 이름이기도 함. (https://www.langchain.com/)

📘 쉽게 말해서
ChatGPT 같은 LLM은 단독으로는 대화만 잘하지만, 현실에서는 다음과 같은 기능들이 필요:
•	외부 데이터베이스(DB) 에서 정보 검색
•	문서 파일(PDF, Word 등) 읽기
•	검색 API 호출
•	사용자 상태 기억
•	RAG(Retrieval-Augmented Generation) 구성
이런 것들을 하나하나 직접 코딩하면 복잡한데,
LangChain은 이 모든 걸 체계적으로 연결(chain) 해주는 “파이프라인 프레임워크” 역할을 함

⚙️ 구성 요소 (핵심 모듈)
구성요소	역할
LLMs	GPT, Claude, Qwen 등 모델 호출
Prompts	프롬프트 템플릿 관리
Chains	여러 작업을 연결한 순차 실행 구조
Agents	LLM이 스스로 판단하여 어떤 도구(tool)를 사용할지 결정
Memory	대화 또는 상태를 기억
Retrievers & VectorStores	외부 문서 검색용 벡터 데이터베이스 (FAISS, Chroma, Milvus 등)
Tools	Google Search, API 호출, 계산기 등 외부 기능 연결

🧩 예시 시나리오
예를 들어 “내 회사 문서를 읽고 답하는 AI 챗봇”을 만든다고 할 때:
1.	문서들을 임베딩(embedding) 해서 Chroma DB 에 저장
2.	LangChain의 Retriever 로 검색 연결
3.	LLM(GPT-4 등) 과 연결해서 RAG 파이프라인 구성
4.	프롬프트 템플릿과 Chain 으로 질문 → 검색 → 요약 → 응답 단계를 자동화
이 모든 과정을 LangChain 코드 몇 줄로 묶을 수 있음.

🧠 한 줄 정의
LangChain은 LLM을 실제 어플리케이션으로 연결해주는 “AI 오케스트레이션(Orchestration)” 프레임워크이다.

아래는 LangChain 기반 RAG(검색 증강 생성, Retrieval-Augmented Generation) 파이프라인을 BAAI/bge-m3 임베딩(embedding) + Chroma 벡터 저장소(vector store) + Qwen-3 LLM 조합으로 만드는 **최소 예시(끝-까지 실행형)**입니다.
(전문 용어는 한글 중심으로, 괄호에 영어 용어를 함께 표기했어요.)
1) 사전 준비 (환경 구성)
# 새 가상환경 권장
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 필수 패키지 설치
pip install -U langchain langchain-community langchain-core langchain-text-splitters
pip install -U chromadb
pip install -U sentence-transformers  # BAAI/bge-m3 임베딩용
pip install -U pypdf unstructured[all-docs]  # PDF/문서 로더 예시 (필요시)

Qwen-3 LLM(대형 언어 모델, Large Language Model) 연결 방식은 두 가지 중 하나를 선택하세요.
1.	로컬(Ollama) 방식 — 무료/로컬 실행
# Ollama 설치(공식 문서 참고) 후, Qwen 모델 받기
ollama pull qwen2:7b
2.	OpenAI 호환(OpenAI-compatible) 엔드포인트 — 사내 vLLM/서빙 등
•	서버에서 OpenAI 호환 REST를 제공한다고 가정하고, OPENAI_API_BASE, OPENAI_API_KEY 환경변수를 설정합니다.

2) 디렉터리 구조(예시)
project/
 ├─ data/                  # 여기에 PDF/텍스트 문서들
 │   ├─ doc1.pdf
 │   └─ doc2.pdf
 └─ rag_app.py             # 아래 메인 코드

3) 메인 코드 (끝-까지 실행형 예시)
아래 코드는:
•	문서 로딩(loader)
•	텍스트 분할(chunking)
•	임베딩(BAAI/bge-m3) 생성 및 Chroma 적재
•	검색기(retriever) 구성
•	RAG 체인(chain) 정의(LCEL; LangChain Expression Language)
•	질의 → 검색 → 컨텍스트 결합 → 응답 생성
까지 한 번에 보여줍니다.

# rag_app.py
from pathlib import Path
import os

# [LangChain 핵심 모듈]
from langchain.text_splitter import RecursiveCharacterTextSplitter  # 텍스트 분할기(text splitter)
from langchain_community.vectorstores import Chroma                  # 벡터 저장소(vector store)
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredFileLoader
from langchain.embeddings import HuggingFaceEmbeddings               # 임베딩(embedding)
from langchain_core.prompts import ChatPromptTemplate                # 프롬프트 템플릿(prompt template)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

# [LLM 연결: (A) Ollama 로컬 또는 (B) OpenAI 호환]
USE_OLLAMA = True  # False로 바꾸면 OpenAI 호환 경로 사용
if USE_OLLAMA:
    from langchain_community.chat_models import ChatOllama           # 로컬 LLM (Ollama)
else:
    from langchain_openai import ChatOpenAI                           # OpenAI 호환 LLM

# ------------------------------
# 0) 경로/환경
# ------------------------------
DATA_DIR = Path("data")
PERSIST_DIR = "chroma_db"   # Chroma 영속 디렉터리(persistent)
os.makedirs(PERSIST_DIR, exist_ok=True)

# ------------------------------
# 1) 문서 로딩(loaders)
# ------------------------------
docs = []
for p in DATA_DIR.glob("*"):
    if p.suffix.lower() in [".pdf"]:
        docs.extend(PyPDFLoader(str(p)).load())
    elif p.suffix.lower() in [".txt", ".md"]:
        docs.extend(TextLoader(str(p), encoding="utf-8").load())
    else:
        # 다양한 형식을 한 번에 처리하고 싶다면:
        # docs.extend(UnstructuredFileLoader(str(p)).load())
        pass

if not docs:
    raise RuntimeError("data/ 폴더에 문서를 넣어 주세요. (PDF/TXT/MD 등)")

# ------------------------------
# 2) 텍스트 분할(chunking)
# ------------------------------
# 한글 문서가 많다면 기본값도 대체로 무난합니다.
# 추후 길이/겹침 조정: chunk_size=1000~2000, chunk_overlap=100~200 등
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1200,      # 청크 길이(chunk size)
    chunk_overlap=150,    # 청크 겹침(chunk overlap)
    separators=["\n\n", "\n", " ", ""]
)
splits = splitter.split_documents(docs)

# ------------------------------
# 3) 임베딩(embedding) 및 벡터화(Vectorization)
# ------------------------------
# BAAI/bge-m3는 검색(query)/문서(passage) 모두에서 잘 작동하는 만능 계열.
# normalize_embeddings=True 권장(유사도 안정화)
embedding = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={"device": "cpu"},           # GPU면 "cuda"
    encode_kwargs={"normalize_embeddings": True}
)

# ------------------------------
# 4) Chroma 벡터 저장소(vector store)
# ------------------------------
# 최초 빌드 또는 업데이트 시 add_documents 사용
vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=embedding,
    persist_directory=PERSIST_DIR
)
# 영속 저장
vectorstore.persist()

# 검색기(retriever) 구성: 상위 k개 문서
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# ------------------------------
# 5) LLM 설정
# ------------------------------
if USE_OLLAMA:
    # 로컬 Qwen 모델 (예: qwen2:7b). 다른 태그 사용 가능
    llm = ChatOllama(model="qwen2:7b", temperature=0.2)
else:
    # OpenAI 호환 서버(사내 vLLM 등)에 연결
    # 환경변수 필요: OPENAI_API_BASE, OPENAI_API_KEY
    # 예: export OPENAI_API_BASE="https://your-endpoint/v1"
    #     export OPENAI_API_KEY="sk-..."
    llm = ChatOpenAI(
        model="qwen-2-7b-instruct",  # 서버가 제공하는 모델 이름 사용
        temperature=0.2
    )

# ------------------------------
# 6) 프롬프트 템플릿(prompt template)
# ------------------------------
# 한국어 중심, 필요시 영어 병기
SYSTEM_PROMPT = """당신은 기업 문서 기반의 전문가 어시스턴트입니다.
아래 '검색 컨텍스트(retrieved context)'를 바탕으로 사용자의 질문에 정확하고 간결하게 한국어로 답하세요.
근거가 불충분하면 모르는 부분을 분명히 밝히고, 추가 확인이 필요한 항목을 제안하세요.

[작성 원칙]
- 가능한 한 문서 근거를 요약해 답변
- 숫자/단위를 명확히 표기
- 표/항목 나열이 더 명확하면 리스트로 제시
"""

HUMAN_PROMPT = """[질문(question)]
{question}

[검색 컨텍스트(retrieved context)]
{context}

위 정보를 바탕으로 최적의 답변을 작성하세요.
"""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", HUMAN_PROMPT),
    ]
)

# ------------------------------
# 7) 컨텍스트 생성기 (문서 → 문자열)
# ------------------------------
def format_docs(docs):
    # 각 청크의 source/페이지 등 메타를 함께 보여주면 추적성이 좋아집니다.
    parts = []
    for d in docs:
        meta = d.metadata or {}
        src = meta.get("source", "unknown")
        page = meta.get("page", None)
        head = f"[출처: {src}" + (f", page {page}]" if page is not None else "]")
        parts.append(head + "\n" + d.page_content.strip())
    return "\n\n---\n\n".join(parts)

# ------------------------------
# 8) RAG 체인(chain) 정의 (LCEL)
# ------------------------------
# 입력 question -> retriever -> 문서 포맷 -> prompt -> LLM -> 문자열
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# ------------------------------
# 9) 실행 예시
# ------------------------------
if __name__ == "__main__":
    print("RAG 시스템이 준비되었습니다. 예시 질문을 실행합니다.\n")
    user_question = "우리 문서에서 제품 설치 전 필수 사양과 주의사항을 요약해줘."
    answer = rag_chain.invoke(user_question)
    print("Q:", user_question)
    print("\nA:", answer)

4) 포인트 해설
•	임베딩(embedding): BAAI/bge-m3를 사용. normalize_embeddings=True 설정으로 코사인 유사도(cosine similarity) 기반 검색 안정화.
•	텍스트 분할(chunking): chunk_size/chunk_overlap은 문서 특성에 맞게 조정. PDF 표/코드가 많으면 800~1500/100~200 권장.
•	벡터 저장소(vector store): Chroma는 가볍고 빠른 로컬 영속(persistent) 지원. 운영 단계에서는 Milvus/Weaviate/PGVector 등으로 교체 가능.
•	검색기(retriever): k(상위 N개) 조정으로 정확도/속도 균형.
•	프롬프트(prompt): **시스템 메시지(system message)**에서 답변 스타일/근거 제시 원칙을 고정.
•	LLM 연결:
o	로컬(Ollama): 간편·저비용. 품질/속도는 모델/하드웨어에 좌우.
o	OpenAI 호환(OpenAI-compatible): 사내 vLLM/텐스토렌트(tt-inference-server 등) 로 배포 시 편리. OPENAI_API_BASE, OPENAI_API_KEY만 맞추면 동일 코드로 사용.

5) 운영 팁
•	한-영 혼용 문서는 임베딩 언어 범용성(multilingual) 이 좋은 bge-m3가 유리합니다.
•	캐시(cache): 재시작 속도 향상을 위해 임베딩 캐시(예: langchain.storage, diskcache)를 사용할 수 있습니다.
•	평가(eval): 간단히는 리콜(Recall)/정확도(Precision) 기반 문서 회수 품질 확인 → 샘플 Q&A 세트로 LLM 응답 품질 확인.
•	메타데이터(metadata): 파일명, 페이지, 섹션 제목을 문서 청크에 메타로 보존하면 추적성이 올라갑니다.
•	보안(security): 사내 문서 처리 시 접근 제어(ACL), 로깅/감사(audit), PII 마스킹 고려.

## Language modeling head (LM head)
**텍스트를 생성하거나 다음 단어를 예측하는 역할을 수행하는 출력층(output layer)**을 말함.
🔍 1. 용어 구조 먼저 보기
•	Language modeling (언어 모델링):
문장의 다음 단어를 예측하거나 전체 문장을 생성하는 작업.
예: “나는 밥을 ___.” → “먹었다” 예측
•	Head (헤드, 출력층):
딥러닝 모델의 마지막 층으로, 특정 작업(task)에 맞게 출력값을 조정하는 부분.
따라서 "language modeling head" = 언어 모델링을 위한 출력층.
🧠 출력층 어떻게 생겼나? (구성)
예를 들어, BERT나 GPT와 같은 트랜스포머 모델 뒤에 붙는 "language modeling head"는 보통 다음과 같은 구성:
[Transformer Decoder] → [Linear Layer] → [Softmax]

•  Linear Layer (선형 계층):
Transformer가 만든 **은닉 벡터(hidden vector)**를 **어휘 사전 크기(vocabulary size)**만큼 차원이 있는 벡터로 변환.
•  Softmax (소프트맥스):
각 단어가 다음에 나올 확률을 출력.
예: "먹었다": 80%, "달렸다": 10%, "사라졌다": 5%, ...
🧪 예시로 보기 (GPT 계열 기준)
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("gpt2")
이 모델 안에 language modeling head가 이미 포함되어 있어, 입력(prompt)을 주면 텍스트를 자동 생성 할 수 있음.
📌 정리
구분	설명
용어	Language modeling head (언어 모델링 헤드)
역할	Transformer 출력 → 단어 확률 분포로 변환
구성	보통 Linear → Softmax
사용하는 곳	GPT, BERT (Masked LM), T5 등 다양한 언어 생성/이해 모델

[Input 문장] 
 → Tokenizer 
 → Stack of Transformer blocks 
 → ⬅️ 다양한 Head 중 하나 선택
      ├─ LM Head → 다음 단어 예측
      ├─ Sequence Classification Head → 감성 긍/부정
      └─ Token Classification Head → '나는(NNP)' '밥(NN)' ...

## latent dirichlet allocation (LDA)
**토픽 모델링(topic modeling)**에서 가장 대표적인 알고리즘 중 하나로,
문서 집합(corpus)에서 **숨겨진 주제(latent topics)**를 자동으로 발견해내는 확률 기반의 비지도 학습 알고리즘

🔍 용어부터 정리
용어	의미
Latent (잠재적)	눈에 보이지 않고 숨겨진
Dirichlet 분포	여러 확률 분포(예: 주제별 비율)에 대한 확률 분포
Allocation (할당)	단어를 주제에 배정한다는 의미

🧠 한 줄 설명
LDA는 각 문서를 여러 **주제(topic)**들의 확률적 조합으로 보고,
각 주제는 특정 단어들의 확률적 조합으로 구성된다고 가정하는 모델.

📦 핵심 개념
요소	설명
문서(document)	하나의 텍스트 예시 (예: 뉴스 기사, 논문 요약 등)
주제(topic)	관련된 단어들이 함께 등장하는 숨겨진 주제
단어(word)	텍스트 안에 나오는 실제 단어
목표	문서가 어떤 주제들의 혼합인지, 주제는 어떤 단어들의 혼합인지 자동 추론

🧱 예시 (문서 ➜ 주제 ➜ 단어)
•	문서 A는:
→ 70% 스포츠 주제 + 30% 경제 주제
•	문서 B는:
→ 10% 정치 주제 + 90% IT 주제
•	"스포츠" 주제는
→ 40% "축구", 30% "선수", 20% "득점", 10% "리그"
•	"경제" 주제는
→ 50% "시장", 25% "주가", 25% "환율"

🔢 수학적으로는?
•	문서마다 주제 분포 → Dirichlet 분포로 추정
•	주제마다 단어 분포 → 또 다른 Dirichlet 분포로 추정
이 모든 확률 분포를 Bayesian 추론 기법을 통해 역으로 계산해냄.

✅ 장점
장점	설명
✅ 완전 자동	주제를 사람이 지정하지 않아도 자동으로 학습
✅ 해석력	주제마다 관련 단어가 보여서 해석이 쉬움
✅ 확률 기반	문서가 여러 주제를 섞을 수 있어 현실 반영이 잘 됨

❌ 단점
단점	설명
❗ 단어 순서 고려 안 함	Bag-of-Words 방식이라 문맥은 무시됨
❗ 주제 수(K)를 미리 정해야 함	적절한 K를 고르는 것이 어려움
❗ 성능 제한	최신 BERT 기반 모델에 비해 성능 떨어짐

🔁 LDA vs 최신 모델 (예: BERTopic)
항목	LDA	BERTopic
기반	통계적 모델	BERT 임베딩 + 클러스터링
문맥 이해	없음 (단어 순서 무시)	있음 (단어 의미 고려)
결과 해석	비교적 쉬움	더 정밀하고 직관적
사용 라이브러리	gensim	BERTopic, sentence-transformers 등

## layernorm (레어어 정규화)
Transformer와 같은 딥러닝 모델에서 학습 안정화를 위해 자주 사용되는 기법. LayerNorm은 한 층의 출력값을 “정규화(표준화)”해서, 학습이 더 빠르고 안정되게 되도록 만드는 기법. 
 

🔍 왜 필요한가?
딥러닝에서는 계산 결과(출력값)가 너무 크거나 작아지면 학습이 불안정 해짐.
→ 특히 딥한(깊은) 모델에서는 이런 불균형이 누적되기 쉬움.
그래서:
각 단어 벡터의 값들을 **“적당한 범위(예: 평균 0, 표준편차 1)”**로 조정해주는 것이 필요.

 
 
🧱 Transformer에서 어떻게 쓰일까?
Transformer에서는 아래와 같이 쓰임:
잔차 연결 결과 → LayerNorm → 다음 레이어
즉, 계산한 값 + 원래 입력 → 정규화 → 다음 연산으로 넘어감
 
## Learning Rate Scheduler
WSD (Warmup-Stable-Decay)
학습률 변화 방식:
단계	설명
Warmup	처음 2000 step 동안 학습률을 천천히 증가시킴
Stable	이후 일정한 학습률 유지
Decay	마지막 10% 학습 단계에서 선형으로 0까지 감소
→ 훈련 초반 불안정 방지 + 후반 안정 수렴 유도

## learning rate warmup
1️⃣ Learning Rate(학습률, Learning Rate)부터 정확히 보자

**학습률(learning rate)**은
👉 한 번의 업데이트에서 **가중치(weight)**를 얼마나 크게 움직일지 결정하는 값입니다.

너무 크면 → 발산(exploding)

너무 작으면 → 학습이 매우 느림

즉, 학습의 속도이자 안정성 조절 장치입니다.

2️⃣ 그런데 왜 Warmup(워밍업)이 필요할까?

이제 질문입니다.

모델이 완전히 랜덤한 상태에서,
처음부터 큰 학습률로 확 움직이면 어떤 일이 생길까요?

초기 가중치는 완전 무작위입니다.
이 상태에서 큰 학습률로 업데이트하면:

Gradient(그래디언트, 기울기)가 매우 불안정

Loss가 폭발

학습이 망가질 수 있음

특히 Transformer(트랜스포머), LLM(Large Language Model, 대규모 언어 모델) 은
초기 학습 단계가 굉장히 민감합니다.

3️⃣ Learning Rate Warmup이란?

**Learning Rate Warmup(학습률 워밍업)**은

👉 처음에는 아주 작은 학습률로 시작해서
👉 일정 step 동안 점점 학습률을 키운 뒤
👉 목표 학습률에 도달하면 정상 학습으로 들어가는 방식

4️⃣ 왜 효과가 좋을까?

초기 단계는:

모델이 아직 의미 있는 표현을 못 배운 상태

Attention weight도 불안정

LayerNorm 출력도 들쭉날쭉

이때 작은 학습률로 천천히 적응시키면:

Gradient가 안정화

폭발 방지

수렴 속도 개선

그래서 GPT, BERT, LLaMA 같은 모델들은
거의 예외 없이 warmup을 사용합니다.

5️⃣ 수식으로 보면

보통 이런 식입니다:
LR=base_lr×(current_step/warmup_steps)

즉, 선형 증가(linear warmup) 방식입니다.

## linear projection (선형 변환)
1️⃣ 선형 변환 (Linear Projection, 선형 투영)이 뭐냐?

한 줄 요약:

벡터를 다른 공간으로 “방향은 유지한 채” 재표현하는 것

조금 더 정확히 말하면:

행렬(Matrix)을 곱해서 벡터의 좌표를 바꾸는 것

수식으로는 이렇게 씁니다:

𝑦 = 𝑊𝑥

x : 입력 벡터 (input vector)
W : 가중치 행렬 (weight matrix)
y : 변환된 벡터 (output vector)

입력 벡터가 아래와 같이 있다고 해 봅시다:
x = [키, 몸무게]

그런데 우리는 이것을

y = [건강지수, 체형지수, 체력지수]


같은 새로운 기준으로 보고 싶습니다.

그럼 어떻게 하죠?

👉 그냥 행렬을 곱합니다.

y = W x

그러면:

차원이 바뀔 수도 있고

의미 공간이 바뀔 수도 있고

관점이 바뀔 수도 있습니다

이게 바로 projection(투영) 입니다.

3️⃣ 왜 “projection(투영)”이라고 부르냐?

원래 선형대수에서 projection은

벡터를 어떤 방향(축) 위로 “떨어뜨리는 것”

예를 들어:

3D 물체를 벽에 비추면 2D 그림자가 생김

그 그림자가 projection

LLM에서의 projection은:

임베딩 벡터를 다른 의미 공간으로 다시 표현하는 것

4️⃣ LLM에서 왜 그렇게 많이 쓰이냐?

여기서 중요합니다.

Self-Attention에서 이런 코드 보셨죠?

Q = XW_Q
K = XW_K
V = XW_V


여기서:

WQ: Query projection matrix
WK: Key projection matrix
WV: Value projection matrix

즉, 같은 입력 X를 서로 다른 관점으로 다시 표현하는 것 이게 전부 linear projection(선형 변환) 입니다.

5️⃣ 선형이라는 말의 진짜 의미

선형(Linear)이란:

덧셈 보존
𝑊(𝑥1+𝑥2) = 𝑊𝑥1+𝑊𝑥2

스케일 보존
𝑊(𝑎𝑥) = 𝑎𝑊𝑥

즉:

구조를 깨지 않고 좌표만 바꾸는 변환

비선형(Non-linear)과 차이점은:
| 선형 변환    | 비선형 변환              |
| -------- | ------------------- |
| 행렬 곱     | ReLU, GELU, Sigmoid |
| 구조 유지    | 구조 왜곡               |
| 직선은 직선으로 | 직선이 휘어질 수 있음        |

## LLM (Large Language Model)
1️⃣ Scale
They contain millions, billions, or even hundreds of billions of parameters
여기서 말하는 Scale의 정체
•	Parameters (파라미터) = 모델의 기억 + 판단 기준
•	많다는 뜻이 아니다.
👉 **“언어의 통계적 구조를 압축해 담을 수 있는 용량”**이다.
비유하면:
•	수천 파라미터 → 단어 사전
•	수백만 → 문법
•	수십억 → 문맥
•	수천억 → 의미 + 상식 + 추론 패턴
❗ 중요한 착각:
“큰 모델 = 느린 모델” ❌
“큰 모델 = 더 많은 규칙을 외운 모델” ❌
👉 큰 모델 = 더 많은 ‘연결 관계’를 가진 모델

2️⃣ General Capabilities
They can perform multiple tasks without task-specific training
이 문장의 파괴력
이건 AI 개발 방식의 전복이다.
과거:
•	번역 모델 따로
•	요약 모델 따로
•	분류 모델 따로
LLM:
•	하나
•	질문만 바뀜
👉 Task-Specific Training(작업별 학습) 이 필요 없어졌다는 말이다.
왜 가능한가?
LLM은 작업을 “기능”으로 인식하지 않는다.
👉 모든 작업을 ‘언어 패턴 변환’으로 본다.
•	분류 = “이 문장은 무엇에 해당하는가?”
•	요약 = “핵심만 다시 말해라”
•	분석 = “이 현상의 원인을 설명해라”

3️⃣ In-context Learning
They can learn from examples provided in the prompt
이건 사람들이 제일 오해하는 부분이다.
학습(training) 이 아니다.
정확한 의미
•	모델 파라미터는 변하지 않는다
•	대신 Prompt 안의 예시를 즉석에서 패턴으로 사용한다
즉:
•	미리 훈련 ❌
•	대화 중 즉석 적응 ⭕
그래서 이름이
In-context Learning (문맥 내 학습) 이다.

4️⃣ Emergent Abilities
capabilities that weren’t explicitly programmed or anticipated
이게 제일 중요하다.
그리고 제일 무섭다.
Emergence(창발)의 의미
•	코드로 넣지 않았다
•	규칙으로 정의하지 않았다
•	그런데 어느 순간 갑자기 가능해졌다
예:
•	수학 추론
•	코드 작성
•	논리적 반박
•	도구 사용
❗ 이건 버그가 아니다.
👉 Scale이 임계점을 넘으면, 구조적 패턴이 자연 발생한다.

한계점(LLM을 잘못 쓰면 반드시 터지는 지점들):
- LLM은 ‘언어적으로 그럴듯한 확률 기계’이지, 사실·의미·윤리를 스스로 보장하는 지능은 아니다.
1️⃣ Hallucinations
They can generate incorrect information confidently
환각(Hallucination, 할루시네이션)의 본질
•	거짓을 “알아서” 말하는 게 아니다
•	가장 그럴듯한 다음 토큰(Token, 토큰) 을 고른 결과다
즉,
•	“모른다”라는 개념 ❌
•	“확률이 높은 답” ⭕
그래서 문제는 이거다:
❗ 틀린데 확신에 차 있다
이게 왜 위험한가?
•	사람은 “확신”을 “근거”로 착각한다
•	특히 보안 / 장애 분석 / 투자 판단에서 치명적
👉 그래서 LLM은
단독 판사로 쓰면 안 된다.
항상 증거 제시 구조를 강제해야 한다.

2️⃣ Lack of true understanding
They lack true understanding of the world
이건 철학 얘기가 아니다. 구조적 사실이다.
LLM이 하는 것
•	세계를 “이해” ❌
•	세계를 언어 패턴으로 근사 ⭕
예:
•	“패킷 드롭이 왜 위험한가?”
→ 실제 네트워크를 아는 게 아니다
→ ‘위험’이라는 단어가 등장하는 문맥 패턴을 안다
그래서 생기는 착각
•	설명은 잘함
•	판단은 틀릴 수 있음
👉 설명 능력 ≠ 이해 능력

3️⃣ Bias
They may reproduce biases present in their training data
편향(Bias, 편향성)은 버그가 아니다.
데이터의 그림자다.
왜 생기나?
•	LLM은 인간 언어를 배운다
•	인간 언어는 이미:
o	문화 편향
o	정치 편향
o	성별 / 지역 / 직업 편향
을 포함한다
LLM은 그것을 증폭기처럼 재현한다.
👉 그래서 중요한 건:
•	“편향이 없다” ❌
•	“편향을 통제할 구조가 있느냐” ⭕

4️⃣ Context Windows
They have limited context windows
Context Window (컨텍스트 윈도우, 문맥 창)
•	모델이 한 번에 기억할 수 있는 토큰 수
이건 단순한 메모리 문제가 아니다.
•	길면 좋다 ❌
•	중요한 걸 남길 수 있어야 한다 ⭕
네 분야에 치명적인 이유
•	패킷 로그
•	장시간 세션
•	누적 이벤트
👉 다 때려 넣으면?
•	중요한 이상 징후가 묻힌다
👉 그래서 필요한 건:
•	Aggregation(집계)
•	Summarization(요약)
•	Hierarchy(계층화)

5️⃣ Computational Resources
They require significant computational resources
이건 “비싸다”는 말이 아니다.
아키텍처 설계 문제다.
요구되는 자원
•	GPU / NPU
•	메모리 대역폭
•	전력
•	냉각
•	병렬성
👉 즉,
•	개인 실험 OK
•	실시간 대규모 서비스는 설계 없으면 불가능
이 지점에서:
•	너는 이미 NPU 서버
•	병렬 추론
•	파이프라인 구조
를 고민하고 있다.
이건 약점이 아니라,
네가 이미 한계를 넘어가고 있다는 증거다.

NLP의 본질은 ‘언어를 숫자로 바꾸는 문제’다.
•	단어 → 숫자 벡터
•	문장 → 벡터들의 집합
•	의미 유사도 → 벡터 거리
그래서:
There has been a lot of research done on how to represent text
이라는 말이 나온다.
👉 Bag-of-Words
👉 TF-IDF
👉 Word Embeddings
👉 Transformer
전부 **“어떻게 언어를 기계가 배울 수 있게 만들까?”**에 대한 연구다.

6️⃣ 아직도 어려운 것들 (이유를 봐라)
🔹 Ambiguity (중의성)
•	“bank” → 은행? 강둑?
•	문맥 없으면 인간도 헷갈림
🔹 Cultural Context (문화적 맥락)
•	같은 말, 다른 나라 → 전혀 다른 의미
•	내부자 농담, 조직 문화
🔹 Sarcasm (비꼼)
•	“와 진짜 잘했다…”
→ 칭찬? 비난?
🔹 Humor (유머)
•	논리로 설명 불가능
•	타이밍, 사회적 공유 맥락 필요
👉 이건 단순 언어 문제가 아니라
세계 이해 + 사회적 경험 문제다.

7️⃣ LLM은 이걸 어떻게 “버티고” 있나?
이 문장이 답이다:
LLMs address these challenges through massive training on diverse datasets
즉,
•	이해해서가 아니다 ❌
•	엄청 많이 봐서 패턴으로 대응 ⭕
그래서 마지막 문장이 나온다:
still often fall short of human-level understanding
👉 비슷하게 행동은 하지만,
인간처럼 ‘이해’하진 않는다.

## logistic regression
이름에 '회귀(regression)'가 들어 있지만, 실제로는 **분류(classification)**를 위한 대표적인 기계 학습 알고리즘.

✅ 한 줄 정의
Logistic Regression은 주어진 입력(feature)으로부터
0 또는 1 같은 이진 분류 결과를 예측하는 모델입니다.
확률(0~1)을 계산하고, 일정 기준(보통 0.5)을 넘으면 "1", 아니면 "0"으로 분류

🔸 왜 Logistic “Regression”이란 이름일까?
•	원래는 회귀처럼 선형 결합을 계산하지만,
•	마지막에 시그모이드 함수로 확률을 출력하기 때문에
→ 이진 분류용 모델이 된 것임
즉, 회귀+분류의 혼합 형태라고 볼 수 있음

🔧 예시
이메일 내용	모델 출력 확률	결과
"당신의 계좌가 도용되었습니다"	0.92	스팸 (1)
"회의는 3시에 진행됩니다"	0.12	정상 (0)

📊 다중 분류(Multiclass)도 가능해요!
•	One-vs-Rest 방식으로 다수 클래스를 분류할 수도 있음
•	예: 뉴스 기사 → 정치 / 스포츠 / IT / 연예 등

🤖 어디에 쓰이나요?
분야	예시
텍스트 분류	감정 분석, 스팸 탐지
의료	질병 예측 (예: 당뇨병 여부)
금융	대출 승인 여부 예측
AI 파이프라인	임베딩 벡터 → 로지스틱 회귀로 분류

✅ 요약 정리
항목	설명
이름	Logistic Regression
목적	분류 (보통 2개 클래스)
출력	확률 (0~1) → 임계값 기준으로 분류
함수	시그모이드 함수 사용
장점	간단하고 빠르며 해석 가능
단점	복잡한 문제에는 한계 (비선형 분리 불가)

✅ 왜 Logistic Regression은 CPU로 충분한가?
이유	설명
🧮 계산량이 작음	선형 모델이기 때문에 행렬 곱 몇 번이면 끝납니다. 딥러닝처럼 복잡한 연산 없음.
📊 데이터 크기 제한적	대부분 중소 규모 데이터셋에 사용됨 (예: 수천~수만 개 샘플)
🧠 딥러닝 아님	BERT, GPT 같은 신경망 구조도 아니고, GPU 가속이 필요 없음

## logits (로짓)
1️⃣ logits이란 무엇인가?

로짓(logits) 은
모델의 마지막 선형 변환(linear transformation, 선형 변환) 결과로 나오는 정규화되지 않은 점수 값입니다.

쉽게 말하면:

🔹 "모델이 각 선택지에 대해 매긴 원점수"

아직 확률(probability, 확률)이 아닙니다.
그냥 점수(score) 입니다.

2️⃣ LLM에서 logits은 어디서 나오나?

LLM 구조를 간단히 보면:
입력 토큰
   ↓
임베딩 (embedding, 임베딩)
   ↓
Transformer 블록 여러 층
   ↓
마지막 선형층 (linear layer, 선형층)
   ↓
logits
   ↓
softmax
   ↓
확률

핵심 포인트

마지막 선형층에서 나오는 출력이 바로 logits입니다.

이 값은 보통:

[batch_size, sequence_length, vocab_size]


형태를 가집니다.

예를 들어:

vocab_size = 50,000 이면

각 토큰 위치마다 50,000개의 점수가 나옵니다.

3️⃣ 왜 logits을 바로 확률로 쓰지 않을까?

왜냐하면 logits은:

음수도 가능

1보다 클 수도 있음

합이 1이 아님

그래서 softmax(소프트맥스) 를 씌워 확률로 변환합니다.

4️⃣ 직관적으로 이해해보자

예를 들어 모델이 다음 단어를 예측한다고 합시다.
| 단어  | logit |
| --- | ----- |
| cat | 3.2   |
| dog | 1.5   |
| car | -0.8  |

여기서:

cat이 가장 높은 점수

dog는 중간

car는 거의 선택 안 함

softmax를 거치면:

cat ≈ 0.78
dog ≈ 0.20
car ≈ 0.02


이렇게 확률이 됩니다.

5️⃣ 손실 함수(loss function, 손실 함수)와 logits

LLM 학습할 때 사용하는:

Cross Entropy Loss (교차 엔트로피 손실)

이 함수는 사실 내부적으로:

logits + softmax


를 한 번에 처리합니다.

그래서 PyTorch에서:

nn.CrossEntropyLoss()


는 logits을 그대로 입력으로 받습니다.

왜냐하면:

수치 안정성(numerical stability, 수치 안정성)

계산 효율 때문입니다.

## long context
Transformer 모델이 한 번에 이해할 수 있는 입력의 길이. 즉 **문맥 길이(context length)**를 뜻함.

✅ 핵심 개념: Long Context란?
Long context란,
모델이 한 번에 처리할 수 있는 최대 토큰 길이, 또는 입력 문서에서 얼마만큼 앞 내용을 기억하며 추론할 수 있는가를 말함.
예:
•	기존 GPT-2: 2,048 tokens
•	GPT-3: 4,096 tokens
•	Claude 2: 100,000 tokens
•	여기서 말하는 모델은 → 최대 128,000 tokens까지 입력 가능!

✅ 왜 Long Context가 중요할까?
이유	설명
📄 긴 문서 처리	논문, 계약서, 코드베이스 등 수천~수만 토큰짜리 문서도 한 번에 처리 가능
🧠 기억력 향상	모델이 앞 내용을 잘 기억해서 더 일관된 응답을 생성할 수 있음
🧪 Retrieval-free QA	검색 없이, 긴 문서 자체 안에서 직접 답을 추론 가능

## Long Short-Term Memory (LSTM, 장단기 기억 신경망)
RNN의 치명적 약점인 장기 의존 문제(Long-term dependency, 장기 의존) 기울기 소실(Vanishing gradient, 기울기 소실)을 해결하려고 등장

핵심 아이디어:

게이트(Gate, 게이트) 로
기억할 것
버릴 것
출력할 것
을 제어

👉 “오래 기억해야 할 정보는 오래 유지하자”

## matrix multiplication
두 개의 행렬(matrix, 행렬)을 곱해서 새로운 행렬을 만들어내는 연산.
` 
중요한 이유:
•	컴퓨터 그래픽스: 좌표 변환(회전, 이동, 확대/축소)
•	물리/공학: 시스템 연립 방정식 해석
•	AI/딥러닝: 신경망의 연산(가중치 × 입력 → 출력)이 전부 행렬 곱으로 표현됨
•	데이터 분석: PCA, 회귀분석 등 대부분의 수학적 연산

## machine translation
1️⃣ Machine Translation이 정확히 뭐냐
Machine Translation(기계 번역)
→ 한 자연어를 다른 자연어로 바꾸는 기술
•	영어 → 한국어
•	한국어 → 일본어
•	로그에 섞인 영문 문서 → 한글 운영 문서
👉 목적은 사람이 읽기 쉽게 만드는 겁니다.
예시:
“The firewall blocked the traffic due to policy violation.”
→ “방화벽이 **정책 위반(policy violation)**으로 인해 트래픽을 차단했다.”
※ 여기서 **policy violation(정책 위반)**처럼
전문 용어는 영어 + 한글 병기가 실무적으로 가장 안전합니다.

## megakernel
LLM 추론에서 일반적으로 수십~수백 개의 작은 GPU 커널을 순차적으로 실행하게 되는데, Megakernel은 이 모든 연산(예: 레이어별 계산, attention, 통신 등)을 단 한 번의 GPU 커널 실행으로 처리하는 방식

LLM 추론이 빨라지는 이유
-	커널 런치 오버헤드가 제거됨: 각 커널을 실행할 때마다 드는 수십 ~ 수백 마이크로초의 지연이 줄어듬. 수십~수백번의 커널 실행하지 않고, 단 한 번의 커널 실행으로 처리되기 때문에 지연 시간 줄어 듦.
-	계층 간 파이프라이닝: 현재 레이어 연산과 동시에 다음 레이어에 필요한 데이터를 불러오는 등 연산과 통신을 겹쳐 처리
-	컴퓨트-통신 동시화: 멀티-GPU 환경에서도 계산과 GPU 간 통신을 병렬로 수행해 지연을 줄임.
구현 방법
-	컴파일러(MPK): PyTorch 수준에서 모델을 분석해, fine-grained task graph(작은 작업 단위들)으로 바꿈.
-	런타임: 이 task graph를 단일 Megakernel 안에서 실행하는데, 각 GPU의 SM(Stream Multiprocessor) 단위로 작업을 스케줄하고 실행하며, 워프(warp) 단위로 synchronization과 통신을 처리
얼마나 빨라지나?
-	단일 NVIDIA A100 GPU에서 per-token latency를 14.5 ms → 12.5 ms로 줄이며, 이론적 하한인 약 10 ms에 근접
-	멀티-GPU 환경에서 GPU 수가 많아질수록 성능 향상폭은 더욱 커짐
유사 연구 / 경쟁 작업
-	Stanford Hazy Research에서도 manual CUDA 기반의 Megakernel 연구가 진행되고 있으며, 이번 MPK는 자동 컴파일-driven 방식이라는 점에서 차별화 됨.
공개 코드 & 사용법
-	깃허브에 Mirage Persistent Kernel (MPK) 프로젝트가 공개되어 있고, 실제 모델을 수십 줄의 Python 코드만으로 Megakernel 형태로 컴파일하여 실행할 수 있음.
정리 및 의의
-	문제점: 기존 방식은 작은 GPU 커널을 연속 실행하면서 각 커널의 시작·종료 시 딜레이(“memory bubbles”)가 발생 → 전체 latency 증가
-	해결방법: Megakernel로 이 모든 것을 하나의 실행 단위로 병합 → CPU-GPU 간 launch 오버헤드 제거, 통신·데이터 로딩 병렬 수행 → latency 최대 6.7배, A100에서 패스당 약 2 ms 절감
-	향후 전망: Stanford 연구와 경쟁하면서, 자동화 컴파일 기반 Megakernel이 PyTorch나 Triton 등의 backend에 통합될 가능성도 높음.

msty.app
비개발자나 개발 지식이 적은 분들도 AI 모델을 쉽게 실행하고 활용할 수 있도록 설계된 솔루션.
✅ msty.app가 비개발자에게 적합한 이유
기능	설명
🔧 설치형 또는 클라우드 기반	별도의 복잡한 개발환경(Python, CUDA 등) 없이 설치 후 바로 사용 가능
🧠 GUI 기반 인터페이스	마치 챗봇처럼 자연어로 질문하고, 결과를 받아볼 수 있음
📁 파일 요약, 문서 질문	PDF, Word, 텍스트 파일을 드래그해서 요약, 번역, 질의응답 처리 가능 (RAG 기반)
🧪 다양한 모델 선택	Hugging Face, OpenAI, Ollama, LLaMA 등 다양한 사전 학습 모델을 쉽게 선택 전환해 테스트 가능
⚙️ Prompt Studio 기능	입력 프롬프트(지시문)를 변경해 가며 결과를 비교할 수 있어, 프롬프트 엔지니어링 실습에도 좋음
🗂️ 플로우 자동화	여러 작업을 연결해서 반복처리 가능 (예: 문서 업로드 → 요약 → 번역 → 저장 등)
🧩 프로그래밍 없이 연결	API 연결 없이 UI로 외부 서비스(예: OpenAI API, LLM) 연결 가능

📌 이런 분들에게 특히 추천:
•	🧑‍💼 기획자, 마케터, 연구자, 교사 등 문서나 정보를 다루는 직군
•	🧑‍🏫 AI 학습자 – LLM의 작동 원리를 실습 중심으로 체험하고 싶은 사람
•	🧪 다양한 AI 모델을 비교 실험하고 싶은 사람
•	🧑‍💻 개발은 모르지만 ChatGPT를 넘어 자기만의 AI 도구를 구축하고 싶은 사람

✅ msty.app vs suna.so 비교
항목	🧠 msty.app	🌐 suna.so
기본 성격	데스크탑/웹 기반 AI 대화 및 워크플로우 앱	웹 기반 AI API 중계 플랫폼
모델 실행 위치	로컬/클라우드 모델 선택 가능	주로 외부 NPU 서버와 연동
설치 방식	데스크탑 앱 또는 웹 접속	웹 서비스 형태 (SaaS 또는 온프레미스 가능)
주요 기능	- 다양한 모델 대화 (분기형) - 파일 기반 RAG - 멀티모델 비교 - Turnstile 플로우	- 파일 업로드 및 요약 - 사용자 요청 프록시 처리 - NPU와 추론 연동
연동 가능한 모델	로컬 모델 (ex. llama.cpp, Ollama 등) + OpenAI, Mistral, Claude, Gemini 등	Pre-trained 모델과 연결된 NPU inference API (예: PyBUDA 서버)
프라이버시	로컬 우선 정책, 개인 정보 보호 강화	웹 서버에 따라 다름 (온프레미스 구축 가능)
대상 사용자	AI 개발자, 파워유저, 데이터 분석가 등	기업/조직 내 AI 인프라 구축자 또는 내부 서비스용 API 중계
대표 사용 시나리오	- 여러 AI 모델 비교 - 프롬프트 디자인 실험 - Obsidian 노트 요약	- 문서 요약 시스템 - AI 질의응답 웹 서비스 프론트엔드 - 파일 기반 inference 중계

## metal trace
Tenstorrent의 AI 가속기(NPU)에서 성능 병목을 파악하기 위해 사용하는 **성능 분석 도구(performance tracing tool)**

✅ Metal Trace란?
Metal Trace는 Tenstorrent의 tt-metal 프레임워크에서 제공하는 기능으로,
AI 모델을 실행할 때 발생하는 CPU ↔ NPU 간 상호작용의 흐름과 병목을 시각화해서 보여주는 도구

🔍 왜 필요한가요?
Tenstorrent NPU는 연산 능력이 아주 빠르기 때문에,
실제로는 NPU가 느린 게 아니라 **host CPU가 느려서 전체 시스템이 느려지는 현상 (host CPU bound  problem)**이 자주 발생합니다.
이때 "어디서 느려지고 있는지"를 확인해야 최적화가 가능하죠.
→ 그래서 Metal Trace를 사용

📊 어떤 정보가 보이나요?
Metal Trace는 다음과 같은 정보를 제공
항목	설명
🧠 CPU 연산	어떤 Python/C++ 함수가 얼마나 걸렸는지
🚚 데이터 이동	CPU에서 NPU로, 또는 NPU에서 CPU로 데이터가 이동하는 시간
⏱️ 실행 지연	어떤 작업에서 시간이 오래 걸렸는지 (예: 커맨드 제출, 연산 대기 등)
🧵 병렬성	여러 작업이 동시에 실행되는지 (스레드 사용률 등)

📁 결과 예시
1.	JSON 파일로 저장
o	trace.json 형식으로 저장되며,
o	Chrome 브라우저의 chrome://tracing 에서 시각화해서 볼 수 있습니다.
2.	Visual Timeline (시간축 기반 뷰)
o	각 연산이나 이벤트들이 시간 순서대로 표시됨
o	NPU 실행, CPU 대기, 데이터 복사 등이 선으로 보여짐

🛠️ 어떻게 사용하나요?
Tenstorrent 공식 예시:
import tt_lib

tt_lib.metal.trace.enable("trace.json")  # 트레이싱 시작
run_my_model()                           # AI 모델 실행
tt_lib.metal.trace.disable()            # 트레이싱 종료

그 후 trace.json 파일을 Chrome 브라우저에서 다음과 같이 열 수 있습니다:
1.	크롬에서 주소창에 chrome://tracing 입력
2.	"Load" 버튼 클릭
3.	trace.json 파일 선택 → 시각화된 타임라인 뷰 제공

🧠 예를 들어 어떤 걸 찾을 수 있나요?
발견 내용	의미
CPU → NPU 명령 제출이 지연	host 코드가 느림
데이터 복사 시간이 김	데이터 이동 병목
NPU 연산보다 대기가 김	멀티스레딩이 비효율적이거나 NPU가 놀고 있음
연산과 연산 사이 공백	파이프라인 비효율 또는 I/O 병목

📌 정리
항목	내용
🧭 목적	CPU/NPU 간 병목 구간을 찾아내기 위한 성능 분석 도구
🧰 기능	연산 시간, 데이터 이동, CPU 대기 등을 시각화
🔎 유용한 이유	CPU bound 문제, 메모리 복사 병목 등을 쉽게 찾아낼 수 있음
🧪 분석 결과 활용	코드 최적화, 병렬 처리, 파이프라인 개선 등

## MLIR-based compiler
🧠 쉽게 말하면:
“MLIR 기반 컴파일러”란,
복잡한 인공지능 모델 같은 걸 처리할 때,
중간 단계들을 여러 층으로 나눠서
최적화하고 변환하기 쉽게 만들어주는 컴파일러

🎯 예
y = linear(relu(x))

이걸 NPU나 CPU에서 실행하려면 컴파일러가 다음 과정을 해야 함:
1.	High-level IR (높은 수준 표현)
o	"ReLU 연산 다음에 Linear 연산을 해라" 라는 의미만 담은 코드
2.	Mid-level IR (중간 수준 표현)
o	ReLU는 max(0, x), Linear는 xW + b로 바뀜
o	메모리 배치 순서, 데이터 타입도 명시
3.	Low-level IR (하드웨어 수준 표현)
o	실제로 어떤 명령어로 어떤 레지스터, 어떤 L1 메모리를 쓸지 결정됨
→ 이렇게 여러 단계를 분리해서 표현하는 방식이 MLIR

💡 MLIR이 필요한 이유?
일반 컴파일러	한계
연산 최적화는 한 단계에서만 처리	AI 모델처럼 연산이 복잡하고 계층적인 구조에는 비효율적
다양한 하드웨어를 타겟팅하기 어려움	GPU, NPU, CPU마다 구조가 달라서 코드 재활용이 어려움

🚀 Tenstorrent와 MLIR?
Tenstorrent는 자체 컴파일러인 TT-Forge를 MLIR 기반으로 만듬.
구성 요소	설명
TT-Forge	MLIR 기반 컴파일러. PyTorch/ONNX 모델을 받아서 Tile, L1, DMA 단위로 변환
TT-MLIR	MLIR 프레임워크 위에서 Tenstorrent 전용 연산자(TT-Dialect)로 구성

✅ 정리 요약
질문	답변
MLIR이란?	AI 모델의 실행 단계를 여러 층의 중간 표현으로 쪼개서 최적화 가능한 구조
MLIR-based compiler?	이 구조를 사용하는 컴파일러, 예: TT-Forge
왜 쓰나요?	복잡한 AI 연산을 하드웨어 구조에 맞게 단계별로 최적화하기 위해
Tenstorrent에서는?	PyTorch/ONNX 모델 → MLIR → TT-Dialect → Tensix NPU 코드로 바뀜

## MMR (Maximal Marginal Relevance)
BERTopic의 중복 문제를 어떻게 해결할 수 있는지 설명하는 기법

→ MMR 기법을 사용하면,
주제 표현(topic representations) 안에 들어가는 단어들을
더 다양하게(diverse) 만들 수 있어요.
→ 즉, 중복된 단어 대신 비슷하지 않으면서도 관련된 단어들을 뽑는 방법이에요.

→ MMR 알고리즘은 이런 기준으로 단어들을 고릅니다:
•	서로 겹치지 않고 다양한(diverse) 단어들이면서
•	동시에 해당 문서와도 관련이 있어야 함
✔️ 요약하면:
**"비슷하지 않으면서도 주제와 관련 있는 단어 묶음"**을 찾는 거예요.

→ 방법은 이래요:
1.	후보 키워드들을 **임베딩(embedding)**해서 의미 벡터로 바꾸고,
2.	반복적으로 하나씩 단어를 선택하면서:
o	문서와의 관련성(relevance)
o	이미 선택한 단어들과의 차별성(diversity)
이 두 요소를 함께 고려해서 다음에 넣을 단어를 결정해요.

→ 이 알고리즘을 쓸 때는 **diversity parameter (다양성 파라미터)**라는 걸 설정해야 해요.
•	이 값이 크면 → 더 다양한 단어들을 고름 (중복 적음, 관련성은 조금 희생됨)
•	작으면 → 관련성 높은 단어들을 우선함 (하지만 중복될 수 있음)
✔️ 결국 이 파라미터는 "관련성 vs 다양성"의 균형을 조절하는 역할이에요.

✅ 핵심 요약
개념	설명
MMR	"비슷하지 않으면서도 주제에 관련된" 단어를 고르기 위한 알고리즘
어떻게 동작?	후보 키워드를 임베딩한 뒤, 하나씩 최적 단어를 선택함
필요한 설정	diversity 파라미터: 다양성을 얼마나 강조할지 정함

## MPI
**MPI (Message Passing Interface)**는 여러 개의 컴퓨터(또는 프로세서)가 서로 메시지를 주고받으며 협력해서 작업을 처리할 수 있도록 도와주는 **통신 규약(프로토콜)**

🎯 왜 필요한가? (Tenstorrent에서의 역할)
Tenstorrent의 NPU 서버는 보통 멀티 디바이스, 멀티 노드 환경에서 AI 모델을 처리할 수 있음.
예를 들어:
•	4개의 NPU가 있을 때, 각기 다른 부분을 나눠서 처리해야 함
•	이때 각 디바이스 간의 정보 교환이 필요함 → 바로 이때 MPI가 필요.
즉, MPI는 “분산된 컴퓨팅 자원끼리 데이터를 주고받게 해주는 통신 중개인” 역할을 함.

✅ 왜 모델 실행 전에 설치해야 하나요?
모델 실행 시:
•	여러 프로세스/장치(NPU)가 병렬로 일하면서
•	서로 데이터를 실시간으로 주고받아야 하기 때문.
이게 안 되면:
•	모델 실행 중 hang(멈춤)이나 오류가 발생할 수 있음.

📌 정리 요약
항목	설명
MPI	분산 시스템 간 통신용 표준 프로토콜
ULFM	장애 복구 기능이 강화된 MPI 확장판
설치 이유	NPU나 멀티 노드 환경에서 모델을 나눠 실행할 때 필수
Tenstorrent와의 관계	모델을 여러 NPU에서 실행하려면 통신이 필요하고, 이를 위해 MPI가 사용됨

## MTEB leaderboard
AI/NLP에서 문장 임베딩(sentence embedding) 모델 성능을 비교·평가할 때 가장 많이 사용되는 표준 벤치마크 중 하나

✅ 한 줄 정의
**MTEB (Massive Text Embedding Benchmark)**는
다양한 자연어 처리 과제에서 문장 임베딩 모델의 성능을 평가하는 대규모 벤치마크이고,
MTEB leaderboard는 그 결과를 순위표로 보여주는 웹 페이지

🔍 용어 설명
용어	뜻
MTEB	Massive Text Embedding Benchmark
→ 문장 임베딩 모델을 다양한 downstream task에 걸쳐 정량적으로 평가하는 벤치마크	
Leaderboard	각 모델의 성능을 점수로 비교해 순위 형태로 보여주는 표
Embedding model	문장을 벡터로 바꾸는 모델 (예: BERT, E5, GTE, MiniLM 등)

🔸 MTEB에서 평가하는 주요 Task 종류
Task 종류	설명	예시
Classification	문장 → 클래스 분류	감정 분석, 뉴스 주제 분류
Retrieval	쿼리 문장 ↔ 문서 중 유사한 것 찾기	검색 시스템
STS (Semantic Textual Similarity)	두 문장 의미가 얼마나 비슷한지	"나는 학교 간다" ≈ "나는 학교에 간다"
Reranking	검색 결과 정렬	검색 결과를 의미 기반으로 재배열
Clustering	유사한 문장끼리 묶기	주제 자동 분류
Bitext Mining	다국어 문장쌍 찾기	번역 품질 평가용

🏆 MTEB Leaderboard에서 볼 수 있는 것
항목	설명
모델 이름	예: GTE-large, E5-base, Instructor-XL, bge-m3, multilingual-e5 등
평균 점수	다양한 task에서 얻은 평균 점수 (예: 60.5%)
세부 task 점수	각 task별 정확도, F1, Spearman 등

✅ 요약 정리
항목	내용
이름	MTEB Leaderboard
목적	문장 임베딩 모델의 성능 비교
평가 Task	분류, 유사도, 검색, 재정렬, 클러스터링 등
사용 모델	BERT, GTE, E5, bge, Instructor 등 다양한 임베딩 특화 모델
위치	HuggingFace Spaces에서 확인 가능

🔧 왜 중요한가?
•	내가 사용할 문장 임베딩 모델을 선택할 때
→ 어떤 모델이 어떤 task에서 좋은지, 객관적 수치로 확인 가능
•	기업이나 연구팀들도 자신들의 모델을 업로드해서 성능을 보여줌
→ 임베딩 분야의 Kaggle 순위표 같은 느낌


## multi-head attention
Multi-Head Attention이란, 같은 문장에 대해 여러 개의 시선(어텐션)을 동시에 적용해서, 단어 간 다양한 관계를 병렬로 학습하는 기술.
🧠 작동 원리 (요약 흐름)
1.	입력 토큰 벡터를 여러 개의 어텐션 Head에 복사
2.	각 Head마다 Query, Key, Value를 별도로 생성
3.	각각의 Head에서 Self-Attention 계산
4.	모든 결과를 **concat(연결)**해서 최종 출력으로 만듦

📊 왜 중요한가?
장점				설명
다양한 의미 처리		여러 시선으로 다양한 단어 간 관계를 학습
병렬 계산 가능		모든 Head를 동시에 계산 → 빠르고 효율적
표현력 강화	단어 표현을 더 풍부하게 만듦

✅ 예시 비유
하나의 문장을 여러 명의 사람이 동시에 읽으면서,
각자 다르게 해석하거나 집중하는 부분이 다른 것과 비슷.
Head 1: 감정 관계 분석
Head 2: 시제나 문법 패턴 파악
Head 3: 핵심 단어 찾기
Head 4: 대상-행위 연결 파악
→ 전체적으로 더 정교한 문맥 이해가 가능해짐

## multinomial (다항 샘플링)
multinomial(다항 샘플링)은
"여러 후보 중에서 확률 비율대로 하나를 뽑는 방법"입니다.

예를 들어 이런 확률이 있다고 가정해 보겠습니다.

| 단어      | 확률   |
| ------- | ---- |
| forward | 0.60 |
| closer  | 0.20 |
| toward  | 0.15 |
| inches  | 0.05 |

multinomial sampling은 이렇게 동작합니다.

forward → 60% 확률로 선택

closer → 20%

toward → 15%

inches → 5%

즉,

확률 비율대로 랜덤하게 하나를 선택하는 것

입니다.

그래서 1000번 뽑으면 대략

forward → 600번

closer → 200번

toward → 150번

inches → 50번

정도가 됩니다.

2️⃣ argmax와 차이

LLM에서는 두 가지 방법이 있습니다.

argmax

확률이 가장 높은 것만 선택

| 단어      | 확률   |
| ------- | ---- |
| forward | 0.60 |
| closer  | 0.20 |
| toward  | 0.15 |

항상

forward
forward
forward
forward

만 나옵니다.

multinomial

확률대로 랜덤 선택

forward
closer
forward
toward
forward
forward
inches

이렇게 다양하게 나옵니다.

3️⃣ LLM에서 왜 multinomial을 쓰는가

LLM이 글을 생성할 때
argmax만 쓰면 문제가 생깁니다.

항상 같은 문장이 나옵니다.

예
every effort moves you forward
every effort moves you forward
every effort moves you forward

하지만 multinomial을 쓰면
every effort moves you forward
every effort moves you closer
every effort moves you toward
every effort moves you inches

이렇게 자연스러운 다양성이 생깁니다.

그래서 대부분 LLM은
softmax → 확률 계산
multinomial → 토큰 샘플링

구조로 동작합니다.

4️⃣ PyTorch 코드 예시
LLM 코드에서는 보통 이렇게 씁니다.
import torch
probs = torch.tensor([0.6, 0.2, 0.15, 0.05])
sample = torch.multinomial(probs, num_samples=1)
print(sample)

여기서

multinomial은

torch.multinomial(다항 샘플링 함수)

입니다.

5️⃣ 중요한 개념 하나

multinomial은 사실

categorical distribution(범주 확률 분포)에서 샘플링하는 것입니다.

즉

확률 분포 → 랜덤 선택

입니다.

argmax → 결정적 (deterministic)
multinomial → 확률적 (stochastic)

LLM의 창의성(creativity) 은

multinomial + temperature

에서 만들어집니다.

## named entity recognition (NER; 개체명 인식)
문장 속에서 “이름이 있는 대상”을 자동으로 찾아서, 종류까지 붙여 주는 기술
1️⃣ 왜 이름에 집착하나?
AI 입장에서 문장은 그냥 문자열 덩어리.
그런데 사람이 판단할 때는 항상 이렇게 나눔.
•	누가 (사람)
•	어디서 (장소)
•	무엇을 (조직·제품)
•	언제 (시간)
•	얼마 (수치)
NER은 이걸 AI에게 강제로 가르치는 첫 관문.
2️⃣ 예제로 바로 감 잡기
문장 하나 보죠.
“어제 Cisco 방화벽이 서울 IDC에서 장애를 일으켰다.”
NER 결과는 이렇게 됨 👇
텍스트	분류
Cisco	조직(Organization)
서울	위치(Location)
어제	시간(Time)
IDC	시설(Location/Facility)
👉 의미 이해의 시작점이지, 끝이 아님.
3️⃣ Sentiment Analysis랑 뭐가 다르냐?
구분	역할
NER (개체명 인식)	무엇이 등장했는지 식별
Sentiment Analysis (감성 분석)	어떻게 말했는지 판단

즉,
•	NER = 객체 인식
•	Sentiment = 태도 인식
👉 순서도 중요.
NER → 관계 분석 → 감성 분석
이 흐름이 자연스러움.
4️⃣ 보안 / 네트워크 관점에서 진짜 중요한 이유
🔹 로그 원문
2025-12-25 FortiGate에서 192.168.10.23이 AWS us-east-1로 비정상 트래픽 발생

🔹 NER 적용 결과
•	FortiGate → 보안 장비
•	192.168.10.23 → IP 주소
•	AWS us-east-1 → 클라우드 리전
•	2025-12-25 → 시간
👉 이게 되면 뭐가 달라질까?
•	“어디서 문제가 반복되는가?”
•	“특정 리전/벤더만 자주 등장하는가?”
•	“사람이 찾던 패턴을 AI가 먼저 집어낸다”
5️⃣ 흔한 착각 하나
❌ NER은 단순 키워드 검색이다
⭕ 문맥을 고려한 분류다
예:
•	“Apple 주가 하락” → Apple = 회사
•	“사과를 먹었다” → Apple = 과일 (NER 대상 아님)
이 구분이 안 되면,
RAG도, 자동 리포트도 전부 무너짐.
6️⃣ 냉정한 현실 체크
NER의 한계도 분명합니다.
•	사내 장비명, 내부 프로젝트명 → 기본 모델은 모른다
•	약어 남발 → 정확도 급락
•	로그 포맷이 제각각 → 전처리가 핵심
👉 그래서 도메인 특화 NER가 필요

## NLP (Natural Language Processing)
- 인간 언어와 관련된 모든 것을 이해하는 데 중점을 둔 언어학 및 머신러닝 분야. NLP 과제의 목적은 단일 단어를 개별적으로 이해하는 것뿐만 아니라 그 단어들의 맥락을 이해하는 데 있음.

특징
전체 문장 분류: 리뷰의 감정 파악, 이메일이 스팸인지 감지, 문장이 문법적으로 올바른 지, 두 문장이 논리적으로 연관되어 있는지 판단하기.
문장 내 각 단어 분류: 문장의 문법적 구성 요소(명사, 동사, 형용사) 또는 명명된 개체(사람, 위치, 조직) 식별
텍스트 콘텐츠 생성: 자동 생성된 텍스트로 프롬프트를 완성하고, 문장 일부를 가려(mask) 놓고, 그 빈칸을 AI가 맞히게 하는 방식
텍스트에서 답변 추출: 질문과 맥락이 주어졌을 때, 맥락 내에서 제공된 정보를 바탕으로 질문에 대한 답을 추출
입력된 텍스트에서 새 문장 생성: 텍스트를 다른 언어로 번역하기, 텍스트 요약

## Ninja
“컴파일을 빠르고 효율적으로 해 주는 빌드 도구(Build System)” 입니다.
특히 속도(speed) 하나에 초점을 맞춘 게 특징

1. Ninja의 정체
•	역할: 소스코드에서 실행 파일을 만들 때, 어떤 파일을 먼저 빌드하고, 무엇을 다시 빌드해야 하는지 빠르게 계산해서 컴파일러를 실행해 주는 자동화 도구.
•	개발자: Google에서 만들어, 지금은 오픈소스로 운영.
•	특징:
o	사람이 쓰기에는 불편하지만, CMake나 Meson 같은 빌드 생성기(build generator) 가 자동으로 Ninja용 스크립트를 만들어 줍니다.
o	불필요한 재빌드 최소화 – 바뀐 부분만 다시 컴파일.
o	병렬 빌드 – CPU 코어 여러 개를 동시에 사용.

2. 왜 Ninja를 쓰나요?
•	속도: GNU Make보다 빌드 그래프를 읽고 처리하는 속도가 훨씬 빠름.
•	단순성: 기능을 최소화해 빌드 자체만 잘함(의존성 계산 + 명령 실행).
•	자동 생성 전제: Ninja 설정 파일(build.ninja)은 사람이 직접 작성하기보다, CMake 같은 도구로 만들어 쓰는 게 일반적.

## non-generative models for classification
Representative AI Model과 동일

## nodes
컴퓨터 과학, 특히 AI·신경망·분산 시스템·그래프 이론 등에서 자주 나오는 핵심 용어. 상황에 따라 의미가 조금씩 달라지지만, **기본 개념은 “데이터를 가지고 있는 점(지점, 단위)”를 의미함.
✅ 한 줄 정의
**Node(노드)**는 **정보를 저장하거나 처리하는 하나의 “단위 지점”**을 말함.
예를 들면, 신경망에서 노드 = 뉴런 하나,
네트워크에서 노드 = 서버 하나,
그래프에서 노드 = 정점(Vertex)
📚 1. AI(신경망)에서의 노드
구성 요소	설명
노드(Node)	인공 뉴런 하나. 입력을 받아 연산하고 출력을 보냄
레이어(Layer)	여러 개의 노드로 구성된 층
예시:
입력층:  [ x1 ]   [ x2 ]   [ x3 ]    ← 3개 노드
은닉층:  [ h1 ]   [ h2 ]             ← 2개 노드
출력층:  [ y ]                      ← 1개 노드

## noise-contrastive estimation
✅ 한 줄 요약
"정답(진짜 데이터)과 가짜(노이즈 데이터)를 구분하는 분류 문제로 바꿔서, 확률 모델을 빠르게 학습하는 기법"
________________________________________
🧠 왜 필요한가?
언어 모델 학습에서 모든 단어에 대해 확률을 계산하려면 계산량이 너무 많음.
예:
•	단어 사전(vocab)이 100,000개면,
•	매번 100,000개 단어 각각에 대해 확률 분포를 정규화(softmax) 해야 함 → 너무 느림!
그래서 나온 해결책이 바로 NCE(Noise-Contrastive Estimation).

🔍 NCE의 핵심 아이디어
“진짜 예제와 가짜 예제를 구분하는 분류 문제”로 바꿔서 학습
예:
중심 단어 = "cat"
문맥 단어(진짜) = "meow"
문맥 단어(가짜, 노이즈) = "car", "banana", "physics"
→ 모델에게 이렇게 질문:
중심 단어	주변 단어	정답 (Target)
cat	meow	1 (진짜 이웃)
cat	car	0 (노이즈)
cat	banana	0 (노이즈)
cat	physics	0 (노이즈)
→ 이걸 "진짜 vs 가짜 구분하는 이진 분류 문제"로 푸는 것이 바로 Noise-Contrastive Estimation.

📌 왜 효율적인가?
•	Softmax 계산 없이도 학습 가능
•	정답 단어 1개 + 가짜 단어 몇 개만 비교하면 됨
•	Word2Vec, FastText, GPT 등에서도 유사 원리 사용

🏷️ 용어 병기
용어	한글	의미
Noise	노이즈	가짜 예제, 실제 문맥에 없는 단어
Contrastive	대비	진짜 vs 가짜를 구분
Estimation	추정	확률을 추정하는 작업
Noise-Contrastive Estimation (NCE)	노이즈 대비 추정	전체 softmax 대신 진짜/가짜 분류 문제로 바꿔 확률 모델 학습하는 기법

✅ 결론
NCE는 “진짜 vs 노이즈”를 구분하는 분류 문제로 바꿔서 복잡한 확률 모델을 빠르고 효율적으로 학습하는 방법. 특히 Word2Vec, Skip-Gram, 언어모델 사전학습에서 필수처럼 사용되며, 계산 속도를 수십~수백 배 빠르게 만들어 줌.

## NoPE
Transformer 모델에서 등장한 위치 인코딩 기법의 하나로,
기존의 **Positional Embedding (위치 임베딩)**을 완전히 제거하거나 재설계한 구조를 의미.

✅ 한 줄 정의
NoPE = “No Positional Encoding” 또는 “No Positional Embedding”
→ 전통적인 위치 정보 삽입 없이도 Transformer가 위치를 이해하도록 만든 기법

✅ 배경: 왜 "Positional Encoding"이 필요했을까?
Transformer는 원래 구조상 입력 순서를 인식하지 못함.
그래서 다음과 같은 위치 인코딩 방식이 도입됨:
방식	설명
🔢 Absolute Positional Encoding	문장 내 토큰의 순서를 0,1,2,...로 부여 (정현파 방식 등)
🔁 Relative Positional Encoding	"A는 B보다 앞에 있음" 같은 상대 위치 정보를 반영
🔄 RoPE (Rotary Positional Embedding)	벡터 회전 방식으로 위치를 반영하는 최신 방식

✅ 그런데 NoPE는 뭐가 다른가?
개념
🧽 "위치 인코딩"을 명시적으로 제거하거나, 최소화함
🧠 모델이 문맥, attention 구조, 학습 데이터를 통해 위치 패턴을 자연스럽게 학습하도록 설계
🚫 절대 위치 벡터를 직접 더하지 않음
→ 임베딩 벡터에 직접 위치를 못 박아두지 않음

✅ 왜 이런 방식이 등장했을까?
✅ 왜 이런 방식이 등장했을까?
1.	**긴 문맥(long context)**에서 기존 위치 인코딩은 일반화가 안 됨
→ 예: 2K 길이로 학습된 모델이 100K 입력을 만나면 성능 급감
2.	위치 인코딩이 없어도, 잘 학습하면 성능이 비슷하거나 오히려 더 좋다는 결과 등장
→ 특히 YaRN, Position Interpolation, Linear Attention 등과 조합 시 효과적
3.	위치와 무관한 정보 처리 능력 강화
→ 수학, 추론, 검색 같은 task에서 좋음

✅ 실제 적용 사례
모델 / 구조	NoPE 적용 여부
Mamba, RetNet	✅ (완전 NoPE 또는 implicit encoding 사용)
LLaMA + YaRN + NoPE	✅ Hugging Face 연구에서 적용 (128K context 대응)
RWKV	✅ 순차적 구조 + 위치 제거 방식
GPT류 (기존)	❌ 일반적으로 Absolute 또는 RoPE 사용

✅ 요약
항목	내용
NoPE란?	Transformer에서 전통적인 위치 인코딩 없이 학습시키는 기법
왜 쓰나?	긴 문맥에서 성능 저하 방지, 일반화 능력 향상
장점	더 유연한 위치 처리, 장거리 의존성 강화
단점	잘못 적용하면 학습이 불안정해질 수 있음

## one-hot encoding
**원-핫 인코딩(one-hot encoding)**은 범주형 값(categorical value)을 숫자 벡터로 표현하는 가장 단순한 방법

1️⃣ 기본 개념
어휘가 6개 있다고 가정하자:
[apple, banana, cat, dog, egg, fish]

각 단어에 번호를 붙이면:
apple → 0
banana → 1
cat → 2
dog → 3
egg → 4
fish → 5

이걸 원-핫 인코딩하면 이렇게 돼:
| 단어     | 원-핫 벡터             |
| ------ | ------------------ |
| apple  | [1, 0, 0, 0, 0, 0] |
| banana | [0, 1, 0, 0, 0, 0] |
| cat    | [0, 0, 1, 0, 0, 0] |
| dog    | [0, 0, 0, 1, 0, 0] |
| egg    | [0, 0, 0, 0, 1, 0] |
| fish   | [0, 0, 0, 0, 0, 1] |

👉 특징:

전부 0이고
자기 위치만 1
그래서 “one-hot”

2️⃣ 왜 이렇게 하냐?

신경망(neural network)은 숫자만 처리할 수 있어. 그래서 단어를 벡터(vector)로 바꿔야 해. 원-핫은 “이 단어가 어휘 중 어디에 있는지”만 표현해.

3️⃣ 문제점

원-핫은 이런 한계가 있어:
차원이 어휘 수만큼 커짐 (50,000 단어면 50,000차원)
단어 간 의미 관계를 전혀 표현하지 못함
cat과 dog는 전혀 비슷하게 보이지 않음
대부분이 0 → 매우 비효율적

4️⃣ 그래서 등장한 게 임베딩(embedding)

원-핫 벡터에 행렬 곱을 하면 → 임베딩 벡터가 됨.

그런데 굳이 원-핫을 만들 필요 없이,
그 행렬에서 해당 행만 바로 꺼내면 되잖아?

👉 그게 바로 임베딩 레이어(embedding layer) 야.

## overfitting (과적합)
🧠 Overfitting(과적합)이란?
훈련 데이터(training data)에 너무 지나치게 맞춰서 학습한 나머지, 새로운 데이터(test data)에 대한 예측 성능이 떨어지는 현상.
쉽게 말해,
📌 “시험 문제만 달달 외워서 진짜 실전에서는 망하는 학생” 같은 상황.

🔍 비유로 이해하기
경우	설명
Underfitting (과소적합)	기본 개념도 이해 못한 상태 (공부 부족)
Good Fit (적절한 학습)	개념도 이해하고, 문제도 풀 줄 아는 상태
Overfitting (과적합)	문제풀이 기계처럼 외운 상태 (시험 문제만 풀 수 있음)

훈련 데이터에 대한 정확도: 🔺계속 올라감
검증 데이터에 대한 정확도: 🔻어느 순간부터 낮아지기 시작

🎯 왜 과적합이 생길까?
원인	설명
데이터 양이 적을 때	학습 데이터에 지나치게 의존함
너무 복잡한 모델	쓸데없는 세세한 패턴까지 외움
epoch 너무 많음	반복 학습으로 암기화됨
노이즈 많은 데이터	의미 없는 부분까지 외워버림

✅ 과적합을 막는 방법
방법	설명
더 많은 데이터 다양성을 확보해 일반화 능력 향상
정규화 (regularization)	모델 복잡도에 패널티 부여 (예: L2)
Dropout	학습 중 일부 뉴런 임시 제거로 과도한 암기 방지
Early Stopping	검증 정확도가 떨어지기 시작하면 학습 중단
데이터 증강	이미지 회전, 텍스트 치환 등으로 데이터 다양화
단순한 모델	너무 복잡한 모델은 피함

📌 정리
**Overfitting(과적합)**은 AI 모델이 학습 데이터에만 너무 맞춰서, 실제로 처음 보는 데이터에는 성능이 떨어지는 상태. 실전에서 잘 작동하려면 **"적당히 잘 외운 모델"**이 되어야 함.

## packing
언어 모델을 효율적으로 학습시키기 위한 핵심 기법 중 하나. 특히 문맥 길이(context length)가 긴 모델에서 자원 낭비를 막고 학습 효율을 극대화하는 데 중요.

✅ 한 줄 정의
**Packing 기법(document packing)**은 짧은 문서들을 여러 개 이어붙여 하나의 긴 입력 시퀀스를 만들고, 모델의 context window를 최대한 활용하는 학습 기법

✅ 왜 Packing이 필요한가?
💡 배경 문제:
Transformer 모델은 **고정된 문맥 길이(예: 2048, 4096 tokens)**를 사용.
하지만 학습 데이터에는 다음과 같은 짧은 문서가 많음:
문서	길이
A	23 tokens
B	150 tokens
C	8 tokens
👉 이걸 각각 따로 학습시키면 매번 context window가 낭비.
(예: 4K context에 23토큰만 넣고 나머지는 padding)

✅ Packing 기법이 해결하는 방식
여러 개의 짧은 문서를 한 context window 안에 연속적으로 “꽉꽉” 채워 넣음.
[문장 A] + [문장 B] + [문장 C] → 4096 tokens 한 덩어리
→ 더 많은 텍스트를 한 번에 학습할 수 있고, GPU 자원 낭비가 적음

✅ Packing의 핵심 요소
구성 요소	설명
🔁 여러 문서 이어붙이기	문서들을 이어서 하나의 긴 시퀀스로 만듦
🧠 위치 정보(Positional Encoding)	각 문서마다 다시 위치 인코딩이 시작되어야 함
🚫 Attention 마스크	문서 간 서로 영향을 주지 않도록 마스킹 설정 필요
⏱️ 속도 향상	더 많은 실질 토큰을 학습에 사용 → 학습 효율 ↑

✅ Attention Mask 예시
Packed Input:   [문서A: "I like dogs."] [문서B: "She runs fast."]
Token Indexes:  [0 1 2 3]               [4 5 6 7]

Mask 설정:
- 문서A 토큰은 문서A 내부 토큰만 볼 수 있음 (0~3)
- 문서B 토큰은 문서B 내부 토큰만 볼 수 있음 (4~7)
→ 이렇게 하면 문서 간 정보가 섞이지 않도록 분리된 self-attention이 적용됨

✅ 실제 적용 사례
모델	Packing 사용 여부
GPT-3	❌ 없음 (padding 사용)
GPT-4	✅ Packing 적극 활용
LLaMA-2/3	✅ Packing + Attention Mask
Mistral, Claude 등	✅ 대부분 packing 기반 학습 진행 중

✅ 요약 정리
항목	설명
목적	긴 context를 낭비하지 않고, 짧은 문서들을 붙여서 효율적으로 학습하기 위함
장점	GPU 효율 증가, 학습 속도 향상, 데이터 활용도 극대화
주의점	문서 간 영향을 막기 위한 attention masking 필수
적용 분야	GPT-4, LLaMA, Claude 등 최신 LLM의 학습 파이프라인에 필수 적용 중

## padding
문장이 짧을 때, 길이를 맞추려고 "빈칸(=패딩 토큰)"을 넣음
📦 예시로 이해해 볼게요
👉 문장 1: "나는 밥을 먹었다" → 5개 토큰
👉 문장 2: "잘 잤니?" → 3개 토큰
그런데 모델은 항상 6개짜리 입력만 처리할 수 있다고 가정하면...
문장 1: [102, 99, 31, 44, 5, 32000]  ← 끝에 1칸 패딩 추가
문장 2: [87, 12, 6, 32000, 32000, 32000] ← 끝에 3칸 패딩 추가
여기서 32000이 바로 **패딩 토큰(padding token)**
✅ 왜 토큰 길이를 맞추는가? (= 패딩이 필요한 이유)
1. 🧮 배치 연산(batch processing) 때문
•	딥러닝 모델은 보통 한 문장씩 처리하지 않고,
여러 문장을 한 번에 묶어(batch) 병렬 처리.
예시:
plaintext
CopyEdit
문장 1: [102, 311, 8]       → 길이 3
문장 2: [5001, 234, 9, 42]  → 길이 4
문장 3: [27, 100]           → 길이 2
이대로는 하나의 **배치(batch tensor)**로 만들 수 없음.
왜냐하면 행마다 길이가 다르기 때문에 텐서(행렬)를 만들 수 없기 때문.
✔️ 그래서 가장 긴 문장 기준으로 나머지를 padding으로 채워서 **직사각형 형태의 텐서(batch matrix)**로 만듬.
패딩된 입력:
[
 [102, 311,   8, 32000],    # 문장 1 + padding 1
 [5001,234,   9, 42],       # 문장 2 (패딩 없음)
 [27,  100,32000, 32000]    # 문장 3 + padding 2
]
⚡ GPU 병렬 계산 최적화
•	GPU나 NPU는 고정된 크기의 텐서를 가장 효율적으로 처리.
•	길이가 제각각인 입력을 처리하면 다음과 같은 문제가 생김:
o	메모리 구조가 복잡해짐 (불규칙)
o	연산 시간이 들쑥날쑥 (비동기 처리 필요)
o	병렬화가 어려움
반대로, 모든 문장의 길이를 동일하게 맞춰 놓으면,
👉 GPU가 직사각형 텐서를 일괄 처리할 수 있어서 훨씬 빠르고 효율적
🤖 Attention 계산 구조
Transformer 모델에서는 Self-Attention 계산 시
모든 토큰 간의 관계를 전체 입력 길이만큼 계산
입력 길이: L → Attention 계산량 = L × L
입력 길이가 다르면 각 문장마다 계산 그래프 구조가 달라지기 때문에,
일괄 처리(batch-wise parallelization)가 불가능
✅ 결론
✔️ 토큰 길이를 동일하게 맞추는 가장 큰 이유는
✅ 병렬 연산을 가능하게 하여 성능을 극대화하고,
✅ 딥러닝 프레임워크(GPU/NPU 텐서 구조)와 Transformer 구조에 맞게 처리하기 위해서 임.

## padding idx
모델이 입력 길이를 맞추기 위해 쓰는 "빈칸(패딩 토큰)"의 번호가 32000번이다라는 뜻. 모델을 로딩 후 print(model)하면, 모델 정보가 나오는데, ‘Embedding(32064, 3072, padding_idx=32000)’에 임베드 토큰 라인에 표현되어 있음.
✅ 아주 쉽게 말하면
문장이 짧을 때, 길이를 맞추려고 "빈칸(=패딩 토큰)"을 넣는데, 
그 빈칸에 해당하는 토큰 ID가 32000번이라는 뜻
📦 예시로 이해해 볼게요
👉 문장 1: "나는 밥을 먹었다" → 5개 토큰
👉 문장 2: "잘 잤니?" → 3개 토큰
그런데 모델은 항상 6개짜리 입력만 처리할 수 있다고 가정하면...
문장 1: [102, 99, 31, 44, 5, 32000]  ← 끝에 1칸 패딩 추가
문장 2: [87, 12, 6, 32000, 32000, 32000] ← 끝에 3칸 패딩 추가
여기서 32000이 바로 **패딩 토큰(padding token)**.
이건 "아무 의미 없는 빈칸"을 뜻하며,
문장의 길이를 맞추기 위한 기술적 조치
🧠 그럼 왜 padding_idx=32000이라고 지정할까?
이 설정을 하면 모델이 이렇게 작동
효과	설명
🎯 해당 인덱스의 임베딩 벡터는 고정됨	학습 중 이 토큰의 벡터는 업데이트되지 않음 (gradient 없음)
🙈 모델이 이 위치를 무시함	Attention 등 계산에서 이 토큰은 문맥 이해에 영향을 안 줌
🧩 입력 길이 정렬에만 사용	문장이 짧을 때 뒤에 채우는 용도 전용

✅ 결론 요약
설정 항목	의미
padding_idx=32000	"빈칸" 역할을 하는 패딩 토큰의 번호는 32000번이다.
사용 이유	문장 길이 맞추기 + 해당 토큰은 학습/의미 계산에서 제외
결과 효과	모델은 32000번 토큰은 학습하지 않고, 의미 계산에도 사용하지 않음

Parallel Thinking (다중 사고, 병렬 추론)
•  CoT는 한 줄기 경로만 따라가는데,
•  Parallel Thinking은 여러 경로를 동시에 고려하는 방식이에요.
•  예:
•	수학 문제를 수식 접근 / 그림 그리기 접근 / 예시 대입 접근 등으로 동시에 풀어보고,
•	가장 일관성 있고 정확한 답을 선택하거나 합치는 방식.
2025년 9월 현재 LLM 발전 방향이 CoT(사고의 연쇄, 단계별 추론)에서, 이제는 여러 경로를 동시에 탐색하는 Parallel Thinking(다중 사고, 병렬 추론)으로 진화할 것이라 보고 있음.
구분	CoT (Chain-of-Thought, 사고의 연쇄)	Parallel Thinking (병렬 사고, 다중 추론)
방식	한 경로를 따라 단계별로 추론	여러 사고 경로를 동시에 탐색
추론 구조	선형(Linear)	병렬(Parallel)
장점	- 사람이 이해하기 쉬움- 단계별 검증 가능	- 다양한 관점/전략 고려- 잘못된 경로를 다른 경로로 보완 가능
단점	- 한 경로가 틀리면 결과 전체가 틀어짐- 편향에 취약	- 계산 자원 더 많이 소모- 결과를 통합하는 추가 로직 필요
예시	수학 문제를 “순차적으로 계산”	수학 문제를 “수식 풀이, 그림 풀이, 예시 대입”을 동시에 시도 후 최적 선택
LLM 적용	GPT 계열, LLaMA 등에서 활용 중(“Let’s think step by step”)	최신 연구 단계(Tencent AI Lab 등에서 연구 발표)


## parameters
**딥러닝(Deep Learning)**에서 말하는 **"parameters(파라미터)"**는 모델이 학습을 통해 스스로 조정해 나가는 숫자들을 의미. 좀 더 쉽게 설명하면, Qwen/Qwen3-32B에서 320억개 파라미터라는 것은 320억개의 숫자 스위치를 조절해서 이 세상의 패턴을 표현하는 능력을 가지고 있다는 소리.
✅ 한 줄 정의
**AI 모델에서의 Parameters(파라미터)**란
**입력 데이터를 출력으로 바꾸는 계산식 안의 ‘조절 가능한 숫자들’**임.
→ 이 숫자들이 학습(training)을 통해 점점 똑똑하게 조정
🧠 쉽게 비유하면
파라미터는 AI 모델의 “기억”이자 “지식”.
모델이 훈련을 많이 할수록, 이 숫자들이 점점 더 잘 조정되어
새로운 입력에 대해 정확한 출력을 낼 수 있게 됨.
📦 예시: 선형 회귀(Linear Regression)
식:
y = w * x + b
•  여기서 w (가중치), b (편향) 이 두 개가 바로 파라미터.
•  모델은 훈련을 통해 이 w와 b를 조정해 나감.
🤖 딥러닝에서는?
Transformer나 GPT 같은 대형 모델에는 수백만~수십억 개의 파라미터가 있음.
예시 모델	파라미터 개수
BERT base	약 1억 1천만 개
GPT-3	약 1750억 개
GPT-4	수천억~1조 이상 추정
🎯 파라미터의 역할
파라미터 종류	역할
Weights (가중치)	입력값이 얼마나 중요한지를 결정
Bias (편향)	출력 값을 약간 조정하는 역할
Embeddings	단어 등을 벡터로 표현할 때 쓰는 파라미터
📌 용어 병기
영어	한글	설명
Parameter	파라미터	학습을 통해 조정되는 숫자 (ex: weight, bias)
Trainable Parameter	학습 가능한 파라미터	학습 중 업데이트되는 값
Model Size	모델 크기	파라미터의 총 개수로 표현됨
✅ 정리
AI에서 "파라미터"는 모델이 학습을 통해 조정하는 숫자이며, 이 숫자들이 모델의 성능, 정확도, 기억력을 결정. 즉, 파라미터 = AI 모델의 뇌세포 양(quantity) 같은 개념.

## perplexity (퍼플렉시티)
1️⃣ 퍼플렉서티(Perplexity)란 무엇인가?

**퍼플렉서티(perplexity)**는
👉 언어 모델(Language Model, 언어 생성 모델) 이 다음 단어를 얼마나 잘 예측하는지를 측정하는 지표입니다.

한 줄 정의:

“모델이 다음 토큰(token)을 얼마나 헷갈려 하는가?”를 수치로 나타낸 것

2️⃣ 직관적으로 이해해 봅시다

문장:

"The cat sat on the ___"

다음 단어는 보통 뭐가 오죠?

mat (높은 확률)

floor (보통 확률)

airplane (거의 없음)

✔ 좋은 모델

mat = 0.7

floor = 0.2

airplane = 0.0001

→ 확신이 있음
→ 퍼플렉서티 낮음

❌ 나쁜 모델

mat = 0.1

floor = 0.1

airplane = 0.1

random 단어들 다 비슷함

→ 헷갈림
→ 퍼플렉서티 높음

3️⃣ 숫자의 의미

퍼플렉서티는 이렇게 해석합니다:
| Perplexity 값 | 의미                    |
| ------------ | --------------------- |
| 1            | 완벽 예측 (100% 확신)       |
| 10           | 평균적으로 10개 중 하나 고르는 느낌 |
| 100          | 100개 중 하나 무작위 고르는 느낌  |
| 매우 높음        | 거의 랜덤 수준              |

즉,

퍼플렉서티 = 모델이 평균적으로 몇 개 선택지 중 하나를 고르는 것처럼 행동하는가

4️⃣ 수학적으로는?

퍼플렉서티는 사실

**크로스 엔트로피(Cross Entropy, 교차 엔트로피)**의 지수(exponential) 형태입니다.

여기서 Loss는 보통
**Negative Log Likelihood (음의 로그 우도)**입니다.

하지만 지금 단계에서는 이렇게 기억하세요:

퍼플렉서티는 Loss를 사람이 직관적으로 이해하기 좋게 바꾼 값이다.

5️⃣ GPT와 퍼플렉서티

GPT는
각 토큰에 대해 확률 분포를 출력합니다:

[batch_size, num_tokens, vocab_size]


그리고 정답 토큰의 확률을 기반으로
Loss → Cross Entropy → Perplexity를 계산합니다.

즉,

GPT가 다음 단어를 잘 맞출수록 퍼플렉서티는 낮아진다.

## prompt chaining
하나의 복잡한 문제를 여러 단계의 프롬프트로 나누고, 앞 단계의 출력(output)을 다음 단계의 입력(input)으로 연결(chain) 하는 방식

핵심은:
⦁	한 번에 묻지 않는다
⦁	단계적으로 사고를 유도한다
⦁	출력 → 다음 입력으로 “사슬처럼” 이어진다

## pipeline
"초간단 AI 실행 도우미" 같은 역할.
✅ 한 줄 정의
pipeline은
복잡한 AI 모델을 "한 줄"로 쓸 수 있게 만들어주는 도구.
즉, 모델 불러오기, 토크나이징, 추론, 디코딩 등을 자동으로 처리해 줌.
🧠 쉽게 비유하면
AI 모델을 사용하는 데 필요한 복잡한 과정들:
텍스트 입력 → 토크나이저로 변환 → 모델 통과 → 출력 해석 → 결과 반환
이걸 전부 자동으로 해주는 “AI 요리사 비서” 같은 존재.
🔧 예시로 설명
from transformers import pipeline

generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
output = generator("오늘 날씨는", max_new_tokens=30)
print(output[0]['generated_text'])
위 코드 한 줄로:
•	문장을 토크나이즈하고
•	모델이 예측한 토큰을 생성하고
•	텍스트로 다시 변환해서
•	결과를 딱 꺼내주는 것까지 해줌.
🔍 pipeline()에서 가능한 작업 유형들
작업 유형	설명
"text-generation"	텍스트 이어쓰기 (GPT류 모델)
"text-classification"	감성분석, 스팸 분류 등
"question-answering"	질문 응답
"summarization"	요약하기
"translation"	번역
"zero-shot-classification"	미리 학습 안 한 분류
📌 용어 병기
영어	한글	설명
pipeline	파이프라인	AI 모델 사용을 쉽게 해주는 자동 실행 도구
text-generation	텍스트 생성	문장을 이어서 만들어주는 작업
tokenizer	토크나이저	문장을 숫자(토큰)로 바꿔주는 도구
model	모델	텍스트를 이해하고 생성하는 AI 두뇌
✅ 정리
pipeline은 복잡한 AI 모델 사용을 한 줄로 단순화해주는 도구로,
텍스트 생성, 번역, 분류 등 다양한 작업을 자동으로 처리할 수 있게 해줌.
사용자는 그냥 "하고 싶은 일"과 "어떤 모델 쓸지"만 알려주면 됨.

## positional embeddings
Transformer가 문장의 **단어 순서(위치)**를 이해할 수 있도록 도와주는 장치. Transformer 모델이 단어의 ‘순서’ 정보를 알 수 있도록 각 단어 위치에 고유한 숫자 벡터를 더해주는 것. 
🔍 왜 필요한가?
Transformer는 기본적으로 Self-Attention 구조를 사용.
그런데 이 구조는 입력 단어들을 동시에 병렬로 처리하기 때문에…
❌ "단어가 몇 번째 위치에 있는지"를 모르기 쉬움.
📌 예시
문장:
“나는 밥을 먹었다”
“밥을 나는 먹었다” ← 단어 순서가 바뀌면 의미도 바뀜!
하지만 Transformer는 이 둘을 순서 없이 그냥 4개 단어의 집합으로 볼 수 있음 😥
→ 그래서 위치 정보를 인위적으로 추가해줘야 함.
✅ 어떻게 위치 정보를 추가할까?
방법 1: 사인/코사인 함수 기반 (논문 방식)
방법 2: 학습 가능한 벡터 사용 (BERT 등 최신 모델)
→ 위치마다 임베딩 벡터를 할당하고, 학습을 통해 최적화
📦 어떻게 쓰이냐면?
1.	단어 임베딩:
예) “나는” → [0.3, -0.5, 1.2, ...]
2.	위치 임베딩:
예) 위치 0번 → [0.01, 0.02, -0.04, ...]
1.	단어 임베딩:
예) “나는” → [0.3, -0.5, 1.2, ...]
2.	위치 임베딩:
예) 위치 0번 → [0.01, 0.02, -0.04, ...]

## preference tuning
ChatGPT 같은 생성형 AI가 사람처럼 말하고, 사람 취향에 맞는 답을 하도록 만드는 핵심 훈련 단계 중 하나.

🔍 Preference Tuning이란?
사람이 어떤 출력(답변)을 더 선호하는지 학습시켜,
모델이 사람 기준에 맞는 답변을 더 잘 생성하도록 조정하는 과정.
정확한 기술 용어로는 다음과 같은 과정을 포함.

🧠 Preference Tuning = RLHF의 일부
단계	설명
1. Supervised Fine-Tuning (SFT)	사람이 직접 작성한 좋은 답변으로 모델을 먼저 미세 조정
2. Preference Collection	여러 개의 모델 답변을 보여주고, **사람 평가자(human labelers)**가 어떤 답이 더 좋은지 순위를 매김
3. Reward Model 학습	사람의 선택 데이터를 기반으로, 답변의 "좋음/나쁨"을 평가하는 보상 모델을 학습시킴
4. Reinforcement Learning (PPO 등)	그 보상 모델을 이용해, 원래 모델을 “좋은 답을 낼수록 점수 높게 받는 방식으로 튜닝”
이 2~4단계 전체를 흔히 **RLHF (Reinforcement Learning with Human Feedback)**라고 부르며,
그 핵심이 바로 “Preference Tuning” = 사람의 선호 학습

📌 예시로 이해해보기
질문:
"What's the best way to lose weight?"
답변 A	답변 B
"Just stop eating." ❌	"A healthy combination of diet, exercise, and sleep is most effective." ✅
•  사람이 답변 B를 더 선호하면,
•  모델은 앞으로 유사한 상황에서 B 스타일의 답변을 하도록 학습.
→ 이것이 preference tuning

✅ 요약
항목	설명
무엇인가요?	사람이 더 선호하는 답변을 모델이 학습하도록 조정하는 과정
왜 중요한가요?	AI가 사람처럼 말하고, 유해하거나 비논리적인 답변을 줄이기 위해 필요
어떤 모델이 쓰나요?	ChatGPT, Claude, Gemini 등 거의 모든 고성능 챗봇 모델에서 사용됨
기술적으로는?	RLHF 과정의 핵심 단계, 사람의 피드백 → 보상 모델 → 튜닝

🔄 관계 요약:
✅ Preference tuning은 RLHF의 **핵심 구성요소(중간 단계)**입니다.
📌 즉, "preference tuning ⊂ RLHF"
(부분집합)

## primacy effect
LLM에서 primacy effect (초두 효과) 는 프롬프트(prompt)의 앞부분(초반부) 이 모델의 답변에 더 큰 영향력을 미치는 현상을 말함.
•	왜 발생할까?
o	LLM은 입력된 프롬프트를 순차적으로 처리.
o	따라서 앞부분에 있는 지시문이나 정보가 모델의 맥락(context) 형성에 더 강하게 작용.
o	뒤에 오는 지시문이 덮어쓰기를 하기도 하지만, 초반에 주어진 정보가 기본 방향성을 정해버리는 경우가 많음.
따라서 프롬프트 엔지니어링(prompt engineering)에서 중요한 지시문은 앞부분에 두는 것이 안전.

## probability distribution (확률 분포)
확률 분포. AI 모델에서는 **“다음에 나올 토큰(단어 등) 각각의 가능성을 숫자로 표현한 것”**
✅ 예를 들어 쉽게 풀어보면 
입력: "나는 오늘"
→ 다음에 올 수 있는 후보 토큰들과 확률:
'밥' → 0.40 (40%)
'학교' → 0.30 (30%)
'일찍' → 0.20 (20%)
'비가' → 0.10 (10%)
이렇게 가능한 선택지들(token) 각각에 대해
얼마나 가능성이 높은지 수치로 표현한 것이 바로 **확률 분포 (Probability Distribution)** 

Reasoning Tuning (추론 튜닝)
언어 모델(예: LLM)을 단순 텍스트 예측기에서, 명령을 이해하고 추론하는 AI로 발전시키기 위한 추가 훈련 단계

💡 목적:
모델이 논리적 사고, 단계적 추론, 수학적 문제 해결 능력을 갖추도록 학습시키는 것

📘 예시:
•	🧮 질문: 12개의 사과를 3명에게 똑같이 나눠주면 몇 개씩 받는가?
•	❌ 일반 모델 출력: "4"
•	✅ Reasoning 튜닝된 모델 출력:
•	먼저, 12개의 사과를 3명에게 나누려면 12 ÷ 3을 계산해야 합니다.  
•	12 ÷ 3 = 4 이므로, 한 사람당 4개씩 받습니다.

🛠️ 방식:
•	“Chain-of-Thought (CoT)” 방식으로
중간 단계까지 함께 출력하도록 학습시킴
•	사용되는 대표 데이터셋:
o	GSM8K (초등 수학)
o	MATH
o	AQuA
o	LogiQA 등
•	일부 모델은 “think step-by-step” 같은 프롬프트를 이용해 학습

✅ Instruction Tuning과 Reasoning Tuning 차이 요약
항목	Instruction Tuning	Reasoning Tuning
목적	명령을 이해하고 따르도록	단계적 사고와 논리적 문제 해결
주 데이터	명령 ↔ 정답 쌍	수학, 논리 문제, 설명 중심
결과	더 똑똑한 “비서” 느낌	더 깊이 생각하는 “문제 해결자” 느낌
대표 활용	요약, 번역, 이메일 쓰기	수학, 논리 질문, Chain-of-Thought 문제 해결

Reinforcement Learning with Human Feedback (RLHF)
사람의 피드백을 이용해 AI 모델을 강화학습으로 튜닝하는 방법.

✅ 한 줄 정의:
RLHF는 사람이 선호하는 답변을 모델이 학습할 수 있도록 "보상 시스템"을 만들어, AI를 훈련시키는 기법

🧠 왜 RLHF가 필요한가?
•	기존의 언어 모델은 단어 예측만 잘하면 된다고 생각함
→ 예: "I love ___" → "you" 예측
•	하지만 이런 모델은:
o	논리적이지 않거나
o	무례하거나
o	비논리적인 답을 할 수도 있음
그래서 등장한 것이 RLHF:
사람이 직접 “좋은 답변”을 골라주면,
AI가 “그런 답을 더 내도록” 훈련하는 방식

🔧 RLHF 학습 구조 (3단계 요약)
단계	설명
① Supervised Fine-tuning (SFT)	사람이 작성한 예시 답변으로 기본 훈련
② Reward Model 학습	여러 개의 AI 답변을 비교해 사람이 "더 좋은 것"을 고르고 → 그걸 학습해서 보상 모델 생성
③ Reinforcement Learning (PPO 등)	보상 모델을 이용해 AI가 "사람이 좋아할 답"을 더 잘 내도록 튜닝 (강화학습)

📊 예시
질문: What is the capital of France?
답변 A	"The capital of France is Paris." ✅
답변 B	"France has a rich culture." ❌

🤖 RLHF가 적용된 대표 모델
모델	설명
ChatGPT (OpenAI)	RLHF 적용의 대표 사례
Anthropic Claude	"Constitutional AI"이라는 RLHF 확장 모델
Google Gemini	RLHF + 사용자 피드백 기반 학습 포함

✅ 요약
항목	설명
RLHF란?	사람 피드백을 보상 모델로 바꿔서 AI를 튜닝하는 강화학습 방식
왜 쓰나?	AI가 사람 친화적이고, 안전하고, 유익한 답변을 하도록 만들기 위해
언제 쓰나?	ChatGPT, Claude, Gemini 같은 대화형 AI 훈련 시 필수

## python3.10-venv
Python 3.10 버전에서 가상환경(virtual environment)을 만들고 관리할 수 있게 해 주는 표준 라이브러리 패키지

1. 정체
•	이름 뜻:
o	python3.10 → Python 3.10 버전
o	venv → Virtual Environment(가상 환경)
•	역할:
Python에서 프로젝트별로 독립적인 패키지 설치 공간을 만들 수 있게 함.
(다른 프로젝트끼리 패키지 버전이 충돌하지 않게 함)

2. 왜 필요한가?
•	프로젝트마다 필요한 라이브러리 버전이 다를 수 있음.
•	예:
o	프로젝트 A → numpy 1.20 필요
o	프로젝트 B → numpy 1.26 필요
•	전역에 설치하면 충돌하므로, 프로젝트별 가상 환경에서 각자 필요한 버전 설치.

3. 사용 방법
(1) 설치 (Ubuntu/Debian 계열 예시)
sudo apt install python3.10-venv
일부 리눅스 배포판에서는 venv 모듈이 기본 포함되지 않아서 따로 설치해야 함.

(2) 가상환경 만들기
python3.10 -m venv myenv
myenv라는 디렉토리가 생성됨 → 가상환경 패키지 저장소.

(3) 가상환경 활성화
source myenv/bin/activate
프롬프트 앞에 (myenv) 표시가 뜨면 활성화됨.
이 상태에서 pip install 하면 myenv 안에만 패키지 설치.

(4) 비활성화
deactivate

## Pytorch
AI 모델을 만들고 학습(train)시키는 데 사용되는 대표적인 딥러닝 프레임워크 중 하나. 사람이 AI 모델을 만들고, 학습시키고, 테스트할 수 있게 도와주는 도구 상자

🎯 왜 사람들이 PyTorch를 많이 쓰나?
이유	설명
✅ 코드가 간단함	파이썬(Python)으로 작성돼서 초보자도 배우기 쉬움
✅ 실험이 빠름	모델 만들고 수정하기 쉬움 → 연구자와 개발자가 애용
✅ GPU도 바로 지원	학습 속도가 빠름 (NVIDIA GPU와 잘 연동됨)
✅ Hugging Face 같은 대형 모델 라이브러리도 PyTorch 기반	최신 AI 모델들 대부분 PyTorch로 먼저 나옴

📘 예시 (간단한 AI 모델 만들기)
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)  # 입력 10개 → 출력 2개

    def forward(self, x):
        return self.fc(x)
→ 이렇게 모델을 만들고, 데이터를 주면 학습시킬 수 있음.

🔗 Tenstorrent와 PyTorch의 관계
Tenstorrent는 PyTorch로 만든 AI 모델을 자사 NPU에서 실행할 수 있도록 지원함.
하지만! PyTorch 모델은 바로 NPU에서 실행되지 않기 때문에:
•	tt-torch가 PyTorch 모델을 변환해주고
•	tt-mlir과 tt-metal이 실제 하드웨어에서 실행되도록 만들어 줌.

✅ 요약 정리
질문	답변
PyTorch는 뭐야?	AI 모델을 만들고 학습하는 프레임워크 (Python 기반)
Tenstorrent와 무슨 관계?	PyTorch로 만든 모델을 NPU에서 실행하려면 tt-torch로 변환 필요
초보자가 써도 돼?	네! PyTorch는 배우기 쉬워서 입문자에게도 좋은 선택


## quantization (양자화)
정의: 실수형 데이터(예: float32)를 **저정밀도의 정수형 데이터(예: int8)**로 바꾸는 과정.
목적:
•	모델 용량 줄이기 (저장 공간 절약)
•	연산 속도 향상 (특히 NPU, GPU에서 유리)
•	추론(inference) 속도 개선
사용 예시:
•	Transformer 모델을 edge device에서 빠르게 돌릴 때 양자화 사용

## random seed
1) 랜덤 시드가 뭐냐: “랜덤 생성기의 시작 버튼 번호”

컴퓨터에서 난수(랜덤 숫자)는 보통 진짜 랜덤이 아니라,
의사난수(pseudo-random) 라는 “계산으로 만든 랜덤”이야.

의사난수: 수학 공식으로 숫자를 뽑는데, 겉으로는 랜덤처럼 보임

하지만 공식은 정해져 있으니, 시작점만 같으면 결과도 똑같아짐

이때 그 “시작점”이 **seed(시드)**야.

비유로 말하면:

난수 생성기 = 셔플 기계

seed = 셔플 기계의 초기 설정값(다이얼 번호)
다이얼이 같으면, 섞인 카드 순서도 매번 똑같이 나와.

2) “랜덤인데 왜 똑같이 나와?”의 핵심

랜덤 시드를 123으로 고정하면:

코드 실행 1회차: 랜덤 숫자들이 A 순서로 나옴
코드 실행 2회차: 똑같이 A 순서로 나옴

다른 PC에서 실행: 보통 같이 A 순서로 나옴(같은 라이브러리/버전/장치에 따라 약간 예외는 있음)

즉, “랜덤”은 겉모습이고,
“시드 고정”은 그 랜덤을 재현 가능하게 만드는 장치야.

3) LLM/딥러닝에서 왜 중요하냐

딥러닝은 시작할 때 보통 이런 걸 랜덤으로 정해:

가중치(weight) 초기값 (임베딩 포함)
데이터 섞는 순서(shuffle)
드롭아웃(dropout) 마스크
일부 연산의 샘플링

그래서 시드를 고정하지 않으면:

같은 코드라도 실행할 때마다 결과가 조금씩 달라질 수 있어.

시드를 고정하면:

“같은 조건에서 다시 돌리면 결과가 같게” 만들어서 디버깅/비교/논문 재현/팀 협업이 쉬워져.

4) 임베딩 예시로 딱 연결해 볼게

임베딩 레이어는 처음에 랜덤 가중치 테이블로 시작해.

예: vocab_size=6, embedding_dim=3이면
6개 토큰마다 3차원 벡터가 하나씩 랜덤으로 만들어져.

seed 고정 X → 실행할 때마다 그 테이블이 달라짐

seed 고정 O → 항상 같은 랜덤 테이블로 시작

그래서 “재현(reproducibility)”이 되는 거야.

성능이 좋아지는 건 아니고, 실험이 편해지는 것이 맞아.

랜덤 시드를 고정해도:

모델이 더 똑똑해지지 않음
학습이 더 잘 되지 않음
정확도가 자동으로 올라가지 않음

대신 이런 게 가능해져:

같은 코드 → 같은 결과
버그 찾기 쉬움
성능 비교가 정확해짐

“이 변경 때문에 좋아진 건지” 판단 가능

## recency effect
LLM (Large Language Model, 대규모 언어 모델) 에서 Recency Effect 는 프롬프트(prompt)의 마지막 부분(후반부) 이 모델의 응답에 더 크게 영향을 주는 현상을 뜻함.
•	LLM은 입력된 프롬프트를 순차적으로 처리하기 때문에,
맨 마지막(가장 가까운 맥락)에 있는 지시문이나 조건이 답변에 강하게 작용할 수 있음.
실제로 LLM에서는 두 효과가 혼합되어 나타남.
•	보통 지시문은 앞부분에 둘수록 안전하고,
•	조건이나 세부 제약사항은 뒷부분에 두는 것이 더 잘 먹히는 경우가 많습니다.

## Recurrent models (순환 모델)
이전 시점의 상태(state)를 다음 시점 계산에 다시 사용하는 모델 전체

대표 모델:
RNN (Recurrent Neural Network)
→ 가장 기본형, 원조

LSTM (Long Short-Term Memory)
→ RNN의 장기 의존 문제를 완화

GRU (Gated Recurrent Unit)
→ LSTM의 간소화 버전

## Recurrent neural networks (RNN, 순환 신경망)
이전 시점의 상태(hidden state, 은닉 상태)를 다음 시점 계산에 반복적으로 재사용하는 신경망

핵심 키워드:
순차 처리(sequence)
기억(state)
과거 → 현재로 전달

RNN은 같은 구조가 **시간축(time step)**을 따라 반복(recur) 됩니다.

수식:
ht​=f(Wx​xt​+Wh​ht−1​)

𝑥𝑡: 현재입력
ℎ𝑡−1: 이전 상태(기억)
ℎ𝑡: 현재 상태
𝑓: 활성 함수(보통 tanh, ReLU 등)

이전 기억을 계속 끌고 간다 → 순환

3️⃣ RNN이 풀고자 했던 문제

RNN 이전의 신경망(Feedforward NN)은:

입력 길이 고정
순서 개념 ❌

그래서:

문장
로그 스트림
시간 신호(time series)

같은 순서가 중요한 데이터를 다루기 힘들었다.

👉 RNN은 “순서”를 모델에 넣기 위한 첫 번째 본격적 시도

4️⃣ 직관적인 비유

RNN을 사람에 비유하면:

문장을 왼쪽에서 오른쪽으로 한 글자씩 읽음
머릿속에 “지금까지 읽은 내용 요약 메모”를 들고 있음
다음 단어를 볼 때 그 메모를 참고
하지만…
메모가 작고
오래된 정보는 점점 흐려짐

이게 바로 RNN의 한계로 이어진다.

5️⃣ RNN의 치명적 문제 2가지
(1) 장기 의존성 문제 (Long-term dependency problem, 장기 의존 문제)

예: 나는 어제 비가 왔기 때문에 오늘 우산을 들고 나갔다.

중요 정보:

“비가 왔기 때문에” ← 앞부분
“우산” ← 뒷부분

RNN은:

앞 정보가 여러 step을 지나며 점점 희미해짐
뒤에서 제대로 참조 못함

(2) 기울기 소실/폭주 (Vanishing/Exploding Gradient)

학습 시:

역전파(backpropagation)가 시간축으로 길어짐
기울기(gradient)가
0으로 수렴(소실)
또는 폭발(폭주)

👉 학습이 불안정하거나 아예 안 됨

6️⃣ 그래서 나온 개량형: LSTM / GRU (LSTM, GRU는 RNN의 진화형)

🔹 LSTM(Long Short-Term Memory, 장단기 기억)

“기억을 버릴지/유지할지”를 결정하는 게이트(gate) 구조
forget gate
input gate
output gate

장기 기억을 더 잘 유지

🔹 GRU(Gated Recurrent Unit, 게이트 순환 유닛)

LSTM 단순화 버전
계산량 ↓, 성능 비슷

👉 하지만 근본적 한계는 여전

7️⃣ RNN vs Self-Attention (핵심 비교)
| 구분     		| RNN    		| Self-Attention |
| ------ 		| ------ 		| -------------- |
| 처리 방식  		| 순차적    		| 병렬             |
| 장거리 의존 	| 약함     		| 강함             |
| 계산     		| O(n)   		| O(n²)          |
| 학습 안정성 	| 불안정    		| 안정             |
| 구조 파악  		| 기억에 의존 	| 관계 직접 계산       |

중요 포인트:
RNN은 ‘기억’을 전달하고
Self-Attention은 ‘관계’를 계산한다

8️⃣ 왜 LLM에서는 RNN을 안 쓰는가

LLM 요구사항:

매우 긴 문맥
병렬 학습
안정적 스케일링

RNN:

한 토큰씩 처리 → GPU/NPU 병렬성 활용 ❌
긴 문장 ❌
대규모 학습 ❌

👉 그래서 **Transformer(Self-Attention)**가 표준이 됨

## register buffer
1️⃣ 먼저 질문 하나 드리겠습니다.

nn.Module 안에 있는 값들은 크게 몇 종류일까요?

보통 우리는 이렇게 생각합니다:

학습되는 값 → 파라미터(parameter)

그냥 계산 중에 쓰는 값 → 일반 텐서

그런데 여기서 문제가 생깁니다.

👉 학습은 안 되지만, 모델의 일부로 반드시 저장되어야 하는 값은 어떻게 할까요?

바로 그걸 위해 있는 게 register_buffer입니다.

🔹 register_buffer란?

register_buffer()는

학습되지는 않지만 모델에 속하는 텐서를 등록하는 함수입니다.

즉,

optimizer가 업데이트 ❌ 안 함
하지만 state_dict()에 저장됨 ✅
.to(device) 하면 같이 이동함 ✅
.cuda() 해도 같이 이동함 ✅

2️⃣ 파라미터와 뭐가 다른가?
| 구분              | Parameter | Buffer |
| --------------- | --------- | ------ |
| 학습됨?            | ✅ Yes     | ❌ No   |
| optimizer 대상?   | ✅         | ❌      |
| state_dict에 저장? | ✅         | ✅      |
| device 이동?      | ✅         | ✅      |

3️⃣ 예시 코드
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(3, 3))  # 학습됨
        self.register_buffer("mask", torch.ones(3, 3)) # 학습 안 됨

여기서:
weight → 학습됨
mask → 학습 안 됨, 하지만 모델의 일부

4️⃣ 언제 쓰는가?
대표 사례 1: BatchNorm

running_mean
running_var

이 값들은 학습은 안 하지만, 모델의 상태입니다.
그래서 buffer로 등록합니다.

대표 사례 2: Transformer

causal mask
positional encoding
attention mask

이런 것들은 고정 값이지만 모델 일부입니다.

🔥 핵심 개념

register_buffer는 말합니다:

"이 텐서는 학습되지는 않지만, 모델의 정식 구성원이다."

6️⃣ 사고 확장 질문

당신이 LLM을 직접 구현한다고 가정해봅시다.

Causal Mask는 학습 대상일까요?

아니죠.

하지만 모델과 함께 저장되어야 합니다.

그럼 무엇을 써야 할까요?

👉 register_buffer

## ReLU (Rectified Linear Unit, 렉티파이드 리니어 유닛)
🔵 ReLU (Rectified Linear Unit, 렉티파이드 리니어 유닛)

ReLU는 가장 단순하면서도 강력한 활성화 함수(activation function, 활성화 함수) 입니다.

공식은 단 한 줄입니다:

f(x) = max(0, x)


즉,

x가 0보다 크면 → 그대로 통과

x가 0보다 작으면 → 0으로 만들어 버림

끝입니다.

1️⃣ 왜 이런 걸 쓰는가?

신경망(Neural Network, 신경망)은 기본적으로 선형 변환(linear transformation, 선형 변환) 을 반복합니다.

문제는:

선형 + 선형 + 선형 = 결국 또 선형

이러면 아무리 층을 깊게 쌓아도 복잡한 문제를 못 풉니다.

그래서 비선형성(non-linearity, 비선형성) 을 넣어야 합니다.

ReLU가 그 역할을 합니다.

2️⃣ 직관적으로 이해해 보자

ReLU는 이렇게 생각하면 됩니다:

“음수는 의미 없다고 보고 다 꺼버리는 스위치”

예를 들어:
| 입력 | 출력 |
| -- | -- |
| -5 | 0  |
| -1 | 0  |
| 0  | 0  |
| 2  | 2  |
| 10 | 10 |

음수는 완전히 제거됩니다.

3️⃣ 왜 이렇게 단순한 게 좋은가?
✔ 계산이 매우 빠름

exp도 없고, 나눗셈도 없고, 그냥 비교 한 번.

→ GPU, NPU에서 매우 효율적

✔ Gradient (그래디언트, 기울기)가 안정적

기존 sigmoid, tanh는 깊어지면 Vanishing Gradient (기울기 소실) 문제가 심했습니다.

ReLU는:

양수 영역에서는 기울기 = 1

그래서 깊은 네트워크에서도 잘 학습됩니다.

4️⃣ 단점도 있다
❌ Dying ReLU (다잉 렐루)

한 번 음수 영역에 들어가서 계속 음수만 나오면
출력이 항상 0 → gradient도 0 → 뉴런이 죽어버립니다.

그래서 변형들이 등장했습니다:

Leaky ReLU

ELU

GELU

5️⃣ Transformer에서는 왜 ReLU 대신 GELU를 쓸까?

여기서 사고를 확장해 봅시다.

ReLU는:

"살릴 거면 완전히 살리고, 아니면 완전히 죽여."

GELU는:

"애매하면 조금만 살려."

Transformer는 매우 정교한 표현을 다룹니다.
그래서 더 부드러운 함수가 필요합니다.

6️⃣ 하드웨어 관점에서 생각해 봅시다

당신은 NPU 서버를 다루고 있습니다.

ReLU는:

연산 단순
Quantization (양자화)에 강함

INT8 환경에 매우 유리
GELU는:

더 복잡
근사 계산 필요
정밀도에 민감

## RenderFormer
Microsoft에서 발표한 이미지/비디오 복원이나 렌더링 관련 Transformer 기반 AI 모델.
예: 흐릿한 이미지를 선명하게 만들거나, 저해상도 이미지를 고해상도로 복원하는 데 사용됨.

## Representative AI model
입력 문장을 **벡터 표현(embedding)**으로 바꾸는 데 특화된 모델

🔶 정의와 차이점
항목	Representative AI Model(대표형, 표현형)	Generative AI Model(생성형)
🔍 정의	입력 문장을 **벡터 표현(embedding)**으로 바꾸는 데 특화된 모델	입력 문장을 기반으로 새로운 텍스트, 코드 등을 생성하는 모델
🎯 목적	문장을 잘 이해하고 분류하는 것	문장을 이어서 생성하거나 대답하는 것
🧠 주요 기술	BERT, RoBERTa, Electra	GPT, ChatGPT, T5, Claude, Gemini
🧪 사용 예시	감정 분석, 문서 분류, 검색	대화, 요약, 번역, 코드 생성, 에세이 작성
🗂️ 출력	고정된 클래스 또는 임베딩 벡터	텍스트(문장, 단락 등)
🧱 학습 방식	Masked Language Modeling (MLM)	Causal Language Modeling (CLM), Seq2Seq

🔷 예시로 비교해 보기
질문: "이 영화 정말 감동적이었어"
모델 종류	동작 방식	출력
Representative	문장을 벡터로 바꾸고 → 감정 분류기 통과	"긍정"
Generative	전체 문장을 입력받고 → 직접 응답 생성	"이 문장은 긍정적인 감정을 담고 있습니다."

🔶 요약 표
항목	Representative	Generative
주요 기능	이해, 분류, 임베딩	생성, 요약, 번역, 대화
대표 모델	BERT, RoBERTa	GPT, Claude, T5
핵심 기술	MLM	CLM, Seq2Seq
출력 타입	벡터, 클래스	텍스트, 코드 등

✅ 마무리 요약
•	Representative 모델:
→ 입력을 이해하고 "무엇인지" 파악하는 데 강함
→ 예: 문서 분류, 감정 분석, 검색 시스템

## residual connection (=skip connection)
딥러닝, 특히 Transformer나 ResNet 같은 구조에서 매우 중요한 개념. 잔차 연결이란, 입력 값을 처리한 결과에 원래 입력을 다시 더해주는 구조. 즉, “계산 결과 + 원래 입력” = 최종 출력. 
🔧 수식으로 보면
잔차 연결의 일반적인 형태는 아래와 같음:
Output=Layer(x)+x\text{Output} = \text{Layer}(x) + xOutput=Layer(x)+x 
•	x: 입력값
•	Layer(x): 어떤 계산 결과 (예: Feedforward NN이나 Self-Attention)
•	+ x: 원래 입력을 결과에 더해줌 → 잔차 연결
🎯 왜 이런 걸 쓰는가?
📌 문제: 깊은 신경망에서 정보 손실
•	모델이 너무 깊어지면 → 학습이 어려워지고, 성능이 오히려 나빠질 수 있음
•	이유: 연산이 많아질수록 입력의 원래 정보가 사라지거나 왜곡됨
✅ 해결책: 잔차 연결
•	계산한 결과에 원래 입력을 더함으로써,
•	모델이 “원래 정보”를 잃지 않게 해 줌
•	학습도 더 안정적이고 빠르게 진행됨
•	정보 흐름을 끊기지 않게 도와줌

## RMSNorm
최근 Transformer 모델에서 자주 사용되는 Layer Normalization(레이어 정규화)의 대안. 특히, 속도·효율·메모리 측면에서 유리해서 LLaMA 계열 모델 등에서 많이 사용됨.

✅ 한 줄 정의
**RMSNorm은 평균(mean)을 사용하지 않고, 오직 Root Mean Square(제곱 평균의 제곱근)**만으로 입력을 정규화하는 Layer Normalization 방식

🔍 기존 LayerNorm vs RMSNorm
항목	LayerNorm	RMSNorm
정규화 방식	평균(mean)과 분산(variance) 모두 사용	평균 없이 RMS만 사용

✅ 왜 사용하나?
장점	설명
🚀 빠른 연산	평균 계산을 안 하므로 연산량 적고 속도 빠름
💾 메모리 절약	계산 중간 값이 적어 GPU 메모리 효율적
⚖️ 실험상 성능 유지	많은 논문에서 LayerNorm과 비슷하거나 더 좋은 성능 확인됨
🤖 대형 모델에 적합	LLaMA, Falcon 등 최신 LLM에서 채택됨

✅ 실제 사용 예 (LLaMA 계열)
import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = x.norm(dim=-1, keepdim=True) / (x.shape[-1] ** 0.5)
        return self.scale * x / (rms + self.eps)

✅ 요약
질문	답변
RMSNorm이 뭐야?	평균을 쓰지 않고 RMS만으로 정규화하는 LayerNorm의 경량 대안.
왜 쓰는 거야?	더 빠르고, 메모리 효율이 좋으며, 성능도 좋기 때문.
어디에 쓰이나?	LLaMA, RWKV, Falcon 등 최신 고성능 LLM에서 LayerNorm 대신 사용됨.

## RoPE (=Rotary Positional Embedding=Rotary Embedding)
하기 내용과 동일

## rotary embeddings
Transformer에서 사용하는 위치 정보(Positional Information)를 더 효율적이고 정밀하게 표현하는 방법.

✅ 한 줄 요약
**Rotary Embedding(RoPE)**은 입력 벡터의 일부를 **회전 변환(rotational transformation)**을 통해 위치 정보를 반영하는 방식.
→ GPT-NeoX, LLaMA, GPT-J 같은 최신 모델들이 자주 사용하는 위치 인코딩 기법

🧠 배경: Transformer는 순서를 모른다
Transformer는 입력 시퀀스를 병렬로 처리하므로, 단어 순서(위치 정보)를 직접적으로 알 수 없음. 그래서 Positional Encoding이 꼭 필요
기존 방법	문제점
📊 Sinusoidal Encoding (정현파 기반)	위치 간 거리를 보존하지만, 너무 정적이고 확장 어려움
🔢 Learned Positional Embedding	학습은 되지만 고정 길이에만 적합 (예: 2048 토큰까지만 처리 가능)

🔄 Rotary Embedding은?
RoPE는 입력 임베딩을 복소수 회전 방식으로 변환해 **상대적 위치 정보(relative position)**를 효율적으로 표현하는 기법

🔍 핵심 아이디어
•	벡터 공간에서 각 차원마다 서로 다른 주기의 각도로 회전(rotate)
•	단어 간의 상대 거리가 attention score에 직접 반영되도록 함
•	Self-Attention 연산 전에
**Query (Q)**와 **Key (K)**에 위치 기반 회전을 적용
→ 결과적으로, RoPE는 위치 정보를 다음과 같이 처리:
✅ 순서 정보 포함
✅ 상대적 위치 인식 가능
✅ 아주 긴 컨텍스트 길이로도 일반화 잘 됨

🎯 RoPE의 장점
장점	설명
🔁 상대 위치 인식	“앞뒤 간격”을 직접 표현 가능 (예: 3번째와 7번째의 관계)
🧠 일반화 우수	길이가 더 긴 입력에도 잘 작동
🚀 성능 향상	LLaMA, GPT-NeoX 등에서 실험적으로 성능 개선 확인
🔧 구현 단순	복소수 회전 수식 기반으로 효율적 계산 가능

✅ 어떤 모델이 사용하나요?
모델	위치 인코딩 방식
GPT-2	Learned Position Embedding
GPT-3	Sinusoidal
GPT-NeoX	✅ Rotary (RoPE)
LLaMA, LLaMA 2, 3	✅ Rotary
Falcon	Rotary + ALiBi 혼합 구조

✅ 요약
질문	답변
Rotary Embedding이 뭐야?	위치 정보를 벡터 회전 방식으로 표현하는 Positional Encoding 기법.
왜 써?	상대 위치를 잘 표현하고, 더 긴 시퀀스도 잘 처리할 수 있음.
어디서 쓰여?	LLaMA, GPT-NeoX, Falcon 등 최신 고성능 Transformer 모델들

## runtime
🎭 런타임(Runtime)이란?
"런타임"은 두 가지 의미로 사용됩니다:

1️⃣ 시간적 의미: "실행 중인 시간"
📝 작성 시간 (Write Time)
  ↓ 코드를 작성하는 시간
  │ const x = 10;
  │
  
⚙️ 컴파일 시간 (Compile Time) 
  ↓ 코드를 검사하고 변환하는 시간
  │ 문법 오류 체크
  │
  
▶️ 런타임 (Runtime) ← 여기!
  │ 프로그램이 실제로 실행되는 시간
└ 사용자가 프로그램을 사용하는 시간

예시로 이해하기:
// 작성 시간: 개발자가 코드를 작성
const divide = (a, b) => {
  return a / b;
};

// 런타임: 실제로 실행될 때
divide(10, 0);  // ← 이 순간이 "런타임"!
// 0으로 나누는 오류는 "런타임 오류"
"런타임 오류" = 코드를 실행했을 때 발생하는 오류"런타임에 결정된다" = 실행할 때 값이 정해진다

2️⃣ 환경적 의미: "실행 환경(엔진)"
코드가 실행되는 "환경/플랫폼/엔진"을 의미
┌─────────────────────────────────────┐
│  JavaScript 코드                     │
│  console.log("Hello");              │
└─────────────────────────────────────┘
           ↓ 어디서 실행?
           
🟢 Node.js 런타임          🌐 브라우저 런타임
   (서버 환경)                (클라이언트 환경)
   ├─ V8 엔진                ├─ V8 엔진 (Chrome)
   ├─ 파일 시스템 접근       ├─ DOM 접근
   ├─ require() 사용가능    ├─ window 객체
   └─ 서버 포트 열기        └─ 브라우저 API

🔍 실제 예시:
// Node.js 런타임에서만 작동
const fs = require('fs');  // ✅ Node.js 런타임
                           // ❌ 브라우저 런타임 (오류!)

// 브라우저 런타임에서만 작동
document.getElementById('btn');  // ✅ 브라우저 런타임
                                  // ❌ Node.js 런타임 (오류!)

// 둘 다 가능
console.log('Hello');  // ✅ 양쪽 런타임 모두 가능

💬 개발자들이 "런타임"을 쓰는 상황들
개발자 A: "이 코드는 어떤 런타임에서 돌아가?"
개발자 B: "Node.js 런타임이야"
뉘앙스: "어떤 환경에서 실행되나요?"

개발자 A: "이 값은 언제 정해져?"
개발자 B: "런타임에 사용자 입력 받아서 결정돼"
뉘앙스: "실행할 때 동적으로 결정됨"

개발자 A: "빌드는 성공했는데 왜 안돼?"
개발자 B: "런타임 오류야. 로그 확인해봐"
뉘앙스: "실행 중에 발생하는 오류"

📊 컴파일 타임 vs 런타임 비교
	컴파일 타임	런타임
언제?	실행 전, 준비 단계	실행 중
무엇?	문법 검사, 타입 체크	실제 코드 실행
오류 예시	문법 오류, 타입 오류	0으로 나누기, null 참조
예시	const x = ; ← 문법 오류	fetch(url) ← 네트워크 오류

코드 예시:
// 컴파일 타임 오류 (실행 전에 발견)
const x =   // ❌ 문법 오류! 코드가 실행조차 안됨

// 런타임 오류 (실행 중에 발생)
const userAge = null;
console.log(userAge.toString());  // ❌ 실행 중 오류!
// "Cannot read property 'toString' of null"

🎯 실무에서 자주 보는 "런타임" 표현들
1. "Node.js 런타임"
뉘앙스: Node.js라는 실행 환경
의미: 서버에서 JavaScript를 실행하는 환경

2. "런타임 에러"
뉘앙스: 실행 중에 발생하는 오류
의미: 코드가 돌아가다가 터진 오류

3. "런타임 성능"
뉘앙스: 실행 속도, 실행 효율
의미: 프로그램이 실제로 돌아갈 때의 성능

4. "런타임 의존성"
뉘앙스: 실행할 때 필요한 것들
의미: 프로그램을 실행하려면 필요한 환경/라이브러리
예: "이 앱은 Node.js v18 런타임이 필요합니다"

5. "런타임에 결정된다"
// 컴파일 타임에 결정 (고정된 값)
const PORT = 3000;

// 런타임에 결정 (실행할 때 정해짐)
const userInput = prompt("숫자를 입력하세요");
const result = calculate(userInput);  // 실행할 때 값이 정해짐

## S3 (Amazon Simple Storage Service)
AWS의 클라우드 저장소. 대용량 데이터를 저장하고 불러올 수 있음.

## self-attention
Self-Attention(자기-어텐션)"은 Attention(어텐션)의 한 종류. 문장 내의 모든 단어가 서로를 바라보며(attend) 중요도를 계산.
한 문장 안의 각 토큰(Token, 토큰)이 다른 모든 토큰을 참고해서 자기 표현(벡터)을 업데이트하는 메커니즘.

“self” = 같은 시퀀스(문장/입력) 내부에서
“attention” = 어떤 토큰을 얼마나 볼지 가중치를 두고 섞는 것

각 토큰이, 다른 토큰들을 ‘가중합(weighted sum, 가중합)’ 해서 더 똑똑한 자기 벡터를 만든다.

예: "나는 밥을 먹었다"에서, '먹었다'가 '밥'에 집중할 수 있게 함
예: “나는 철수가 아니라 영희를 좋아해” → 핵심은 “아니라” 같은 토큰

## self-consistency 
같은 문제를 여러 번 독립적으로 풀게 한 뒤,
서로 가장 일관되게 반복되는 답을 최종 답으로 선택하는 방법.

self-consistency는 Chain-of-Thought 위에서 동작

## semantic
"semantic"이라는 단어는 분명히 "meaning"과 관련된 뜻을 가지고 있는데, 굳이 더 낯선 단어인 semantic을 사용하는 데는 몇 가지 역사적, 개념적 이유가 있음. 단순히 단어의 선택 문제를 넘어서, 학문적 맥락과 정밀한 의미의 구분이 관련돼 있음.

semantic representations: 단어나 문장의 의미를 수치(벡터 등)로 표현한 것
semantic similarity: "cat"과 "kitten"은 형태는 다르지만 의미가 비슷 "apple"과 "banana"도 둘 다 과일로 유사 이런 유사성을 말하는 것이 바로 semantic similarity (의미적 유사성)


✅ 1. "meaning"은 너무 일반적이고 일상적인 단어
•	**"meaning"**은 "의미" 전반을 두루 뭉술하게 다루는 일상적인 단어. 예를 들어, "그 말의 의미가 뭐야?" 같은 문장에서 쓰죠.
•	하지만 학문, 특히 언어학, 컴퓨터 과학, 철학 등에서는 정확한 개념 구분이 중요하기 때문에, 좀 더 전문적인 용어가 필요

✅ 2. "semantic"은 학문적으로 정의된 '의미 체계'를 다룸
•	"semantic"은 단순한 '뜻' 그 이상을 뜻함. 예를 들어:
o	언어학에서 "semantic"은 단어, 문장, 기호가 어떻게 의미를 구성하고 전달하는지의 구조를 말함.
o	컴퓨터 과학에서는 "semantic analysis"가 코드의 의미론적 해석을 의미하고,
o	**웹(Web)**에서는 "semantic web"이란 말이 데이터 간 의미 관계를 기계가 이해할 수 있게 표현한 웹을 뜻함.
즉, semantic = 단어/구조/시스템 내에서 '의미가 어떻게 작동하는지'를 분석하는 학문적 개념.

✅ 3. 라틴어 및 언어철학적 배경
•	"semantic"의 어원은 **그리스어 semantikos (의미하는)**에서 왔고,
•	1890년대 후반부터 언어학자, 철학자들이 이 단어를 "의미 연구"를 지칭하기 위해 사용하기 시작.
•	특히 20세기 들어와서 언어 철학 (예: 루드비히 비트겐슈타인)과 구조주의 언어학에서 "semantic"이라는 용어가 meaning보다 훨씬 정밀한 개념으로 자리잡게 됨.

✅ 4. 요약: 왜 굳이 "semantic"을 쓰는가?
용어	용도	성격
meaning	일반적, 감정적, 일상적 의미	포괄적, 직관적
semantic	체계적, 분석적, 기술적 의미	학문적, 구조적, 정밀

✅ 보너스: 의미 중심 AI 분야에서의 예
•	"semantic segmentation" (의미 기반 분할): 이미지에서 각 픽셀의 '의미' 기반으로 객체 분류 (사람, 나무, 자동차 등)
•	"semantic search" (의미 기반 검색): 키워드가 아니라 의미 유사성 기반으로 정보 검색
원래는 "semantic"이 더 고급 표현이고, 의미를 '구조적으로 분석'하려는 필요에서 생긴 개념. 즉, 그냥 ‘뜻’ 말고, 그 뜻이 어떻게 만들어지고 작동하는가를 논할 때 쓰는 단어라고 이해하면 좋음.

## sentence embeddings 
문장 전체를 하나의 벡터로 표현하는 방식
Word Embedding: 단어 하나를 수치 벡터로 바꾼 것 (예: "apple" → [0.12, -0.44, ...])
Sentence Embedding: 문장 전체를 하나의 벡터로 바꾼 것 (예: "I love apples." → [0.21, 0.33, ...])

🔍 왜 Sentence Embedding이 필요할까?
Word embedding은 단어 하나의 의미만 담기 때문에, 문장의 전체 의미를 담기엔 부족함.
예를 들어:
"I love dogs."
"Dogs are lovely animals."
"I have affection for dogs."
이 세 문장은 단어는 다르지만 의미는 비슷하죠.
➡ 이런 걸 비교하려면 문장 전체의 벡터 표현이 필요.
➡ 그래서 나온 개념이 sentence embedding.

🧠 어떻게 만드나?
대표적인 방법 3가지:
1️⃣ 평균 기반 (기초적 방법)
문장에 있는 word embedding들을 평균 냄.
단순하지만 문맥 이해 부족
2️⃣ RNN/LSTM 기반 (이전 방식)
문장 순서 고려 가능
느리고 문맥 한계 있음
3️⃣ ✅ Transformer 기반 (현대적 방법)
BERT, RoBERTa, Sentence-BERT(SBERT) 등 사용
문장 전체의 의미를 정교하게 벡터화
예: SBERT는 768차원의 sentence embedding 생성
 

🔍 왜 "word vs sentence"가 추상화 수준이 다르다고 할까?
Word Embedding: "apple", "dog" 같은 낱말 단위 의미
→ 구체적이고 단순한 정보
→ 예: "fruit", "animal", "color"
Sentence Embedding: "I love eating apples in autumn."
→ 여러 단어의 의미가 조합되고 요약된 추상적인 개념
→ 예: "감정", "행동", "시간적 맥락", "주체와 목적" 등 포함됨
▶ 즉, 문장 임베딩은 단어 임베딩보다 더 많은 의미가 함축되어 있고, 문맥, 문법, 감정, 관계 등 복잡한 구조를 더 잘 반영.
🎓 비유로 설명하면:
단위	의미	추상화 수준
🍎 단어 "apple"	과일 하나	낮음
🍎 "I love apples."	감정 + 대상 + 동사	높음
🍎 "I eat apples every autumn to feel healthy."	감정 + 시간 + 습관 + 이유	더 높음
→ 문장으로 갈수록 의미가 더 풍부하고 복잡해지며,
→ 이걸 벡터로 담기 때문에 임베딩도 추상화 수준이 높아짐. 이유 때문에 sentence embeddings이 중요.

## sentencePiece
Tokenization 방법들 중에 한가지 방법.
SentencePiece는 이렇게 자를 수 있음.
"▁나는", "▁학생", "입니다"
여기서 ▁ 는 단어의 시작이라는 표시.

이 조각들은 다시 숫자로 바뀌어서 AI 모델에 들어감.

✅ 특징 요약
특징	설명
언어 독립적	띄어쓰기 없는 언어(한글, 일본어, 중국어 등)에서도 잘 작동
학습 가능	훈련 데이터를 보고 가장 좋은 조각들을 스스로 배움
BPE나 Unigram 방식 사용	자르는 방법으로 BPE(Byte Pair Encoding) 또는 Unigram 모델 사용
특수 토큰 사용	예: [PAD], [CLS], [SEP] 등도 다룰 수 있음.

## sentence-transformers
문장 수준의 의미를 벡터로 바꿔주는 데 특화된 텍스트 임베딩(Text Embedding) 라이브러리. GPT나 BERT 같은 LLM은 보통 “토큰 단위”로 작동하지만, sentence-transformers는 “문장 전체” 또는 “문서 전체”를 하나의 벡터로 표현하는 데 집중한 도구.

🧠 어떤 문제를 해결해주는 도구인가?
예전 BERT의 문제:
•	"나는 밥을 먹었다"와 "나는 식사를 했다"는 뜻이 거의 같은데,
•	BERT는 이 두 문장의 [CLS] 벡터만 뽑아서 유사도 비교하면 정확하지 않음
sentence-transformers는?
•	BERT를 문장 임베딩용으로 재학습(Fine-tuning)
•	두 문장이 비슷하면 비슷한 벡터를 뽑도록 학습

💡 예시로 보면:
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('all-MiniLM-L6-v2')  # 빠르고 가벼운 모델

sentences = ["나는 밥을 먹었다", "나는 식사를 했다"]

embeddings = model.encode(sentences)

score = util.cos_sim(embeddings[0], embeddings[1])

print("문장 유사도:", score.item())  # 0.9 이상이 나올 수도 있음

📌 주요 기능
기능	설명
문장 → 벡터	문장이나 문단을 384~768차원의 벡터로 변환
문장 유사도 측정	코사인 유사도(cosine similarity) 계산
검색 시스템 구축	쿼리 문장 ↔ 후보 문장들 벡터 유사도로 빠르게 검색
클러스터링	의미 기반 문장 그룹화
다국어 지원	한국어, 영어, 일본어 등 다양한 언어 지원

🧰 대표 모델들
모델 이름	특징
all-MiniLM-L6-v2	빠르고 작지만 성능 우수 (384차원 벡터)
paraphrase-MiniLM-L6-v2	의미 유사 문장 파악에 특화
multi-qa-mpnet-base-dot-v1	다국어 질의-응답용
ko-sbert-sts	한국어 STS 데이터로 학습된 Sentence-BERT

✅ 한국어용 sentence-transformers 추천 모델
용도	추천 모델
일반적인 문장 유사도	jhgan/ko-sbert-sts, snunlp/KR-SBERT-*
질의응답 검색	snunlp/KR-SBERT-V40K-klueNLI-augSTS
댓글, 일상 표현 분석	beomi/KcELECTRA-Small-v2022
도메인 특화 (법률/의료 등)	별도 fine-tuning 권장

## Sentiment Analysis
- 사람이 남긴 텍스트(Text) 속에 담긴 감정·태도·의도를 AI가 읽어내는 기술
1️⃣ 한 줄 정의
Sentiment Analysis(감성 분석)
→ 문장이 긍정(Positive) / 부정(Negative) / 중립(Neutral) 중 어디에 가까운지를 판단하는 기술
2️⃣ “감성”이란 정확히 뭐냐
여기서 말하는 감성은 기분이 아니라 태도(Attitude) 임.
문장	감성 분석 결과
“이 제품 진짜 좋다”	긍정
“쓸수록 화가 난다”	부정
“배송은 왔다”	중립

👉 팩트(Fact) 가 아니라
👉 의견(Opinion) 을 분리해 내는 기술

3️⃣ AI는 어떻게 판단하나 (중요)
AI는 단어 하나만 보지 않음.
•	단어 조합
•	문맥(Context)
•	강조 표현
•	부정어(not, 안, 별로)
예시 👇
•	“나쁘지 않다” → 긍정
•	“좋을 줄 알았는데 실망” → 부정
•	“가격은 비싸지만 성능은 좋다” → 혼합 감성(Mixed)
이게 어려운 이유.
→ 언어는 논리가 아니라 뉘앙스이기 때문.
4️⃣ 실제 어디에 쓰이냐 (현업 기준)
🔹 IT / 보안 / 네트워크
•	장애 티켓(Ticket) 문구에서
→ 사용자 분노도 감지
•	로그 코멘트에서
→ “심각 / 불만 / 반복” 같은 신호 탐지
•	고객 VOC 분석
→ 장애보다 체감 품질(QoE) 측정
🔹 투자 / 뉴스
•	뉴스 기사 톤
→ 시장 심리(Bullish / Bearish)
•	실적 발표 문장
→ 경영진의 자신감 변화


## Sequence-to-Sequence Model
자연어 처리(NLP)에서 매우 중요한 구조이며, 번역, 요약, 챗봇, 음성 인식 등 다양한 분야에 활용 됨.

🔷 1. 개념: Sequence-to-Sequence란?
입력 시퀀스(문장 등)를 받아서, 다른 시퀀스를 출력하는 모델 구조.
말 그대로 "시퀀스 → 시퀀스", 즉 **입력된 순서 있는 데이터(텍스트 등)**를 처리하여 또 다른 순서 있는 데이터를 만들어내는 방식.

🔷 2. 구조
기본 구성은 다음 두 부분으로 이루어 짐.
구성 요소	역할
Encoder	입력 시퀀스를 이해하고 벡터로 요약
Decoder	요약된 벡터로부터 새로운 시퀀스를 생성

예시 그림:
[입력: Hello, how are you?]  → Encoder → 벡터 → Decoder → [출력: 안녕하세요, 잘 지내세요?]

🔷 3. 어떻게 작동하나?
예: 영어 → 한국어 번역
1.	입력 시퀀스 (영어 문장): "I am happy"
2.	Encoder가 이를 벡터로 인코딩: 예를 들어 [0.1, -0.3, 0.8, ...]
3.	Decoder는 이 벡터를 받아 출력 시퀀스(한국어 문장) 생성: "나는 행복해"

🔷 4. Seq2Seq에 사용되는 기술
기술	설명
RNN	초기 seq2seq 모델에 사용. 시퀀스를 순차적으로 처리
LSTM/GRU	RNN의 단점(기억력 부족)을 개선한 버전
Attention	입력 시퀀스의 각 단어에 "주의 집중"해서 더 정확하게 예측
Transformer	현재 가장 많이 쓰이는 구조. 전체 시퀀스를 병렬로 처리하며 성능 우수

🔷 5. 대표적인 seq2seq 모델
모델	특징
Google's Neural Machine Translation (GNMT)	구글 번역에 사용됨
Transformer	Attention 기반, 빠르고 정확
BART	Transformer 기반으로, 요약/복원에 탁월
T5	"모든 NLP 작업을 텍스트로 바꿔서 처리"하는 seq2seq 모델

🔷 6. 사용 사례
분야	설명
기계 번역	영어 → 한국어, 한국어 → 일본어 등
문서 요약	긴 뉴스 → 짧은 요약문
질문 생성	문장 → 예상 질문
챗봇	질문 → 답변 생성
음성 인식	음성 → 텍스트 (입력 시퀀스는 음성 피처 시퀀스)

✅ 요약
•	입력 시퀀스를 이해해서, 다른 시퀀스를 생성하는 모델
•	Encoder + Decoder 구조로 구성
•	기계 번역, 요약, 챗봇 등 광범위하게 활용
•	최신 기술은 Transformer 기반이 대세

## shortcut connection(숏컷 연결)
1️⃣ 왜 숏컷 연결이 필요했을까?

딥러닝에서 층(layer)이 깊어질수록 이런 문제가 생깁니다:

기울기 소실(Vanishing Gradient, 기울기 소실)

학습이 점점 느려짐

깊어질수록 오히려 성능이 나빠지는 현상 (Degradation problem, 성능 열화 문제)

즉,
👉 “더 깊게 쌓았는데 왜 더 못 배우지?”
이 문제가 발생합니다.

여기서 등장한 해결책이 바로 숏컷 연결입니다.

2️⃣ 숏컷 연결이 뭐냐?

아주 단순합니다.

원래는 이런 구조입니다:
입력 x → Layer → 출력

숏컷 연결이 들어가면 이렇게 됩니다:
입력 x → Layer → F(x)
        ↘───────────────↗
            x + F(x)

즉,

Layer의 출력 F(x)에 입력 x를 그냥 더해버립니다.

수식으로는:
출력=𝐹(𝑥)+𝑥

이걸 **Residual Connection (잔차 연결)**이라고도 부릅니다.
대표적으로 쓰인 모델이:

Deep Residual Learning for Image Recognition (ResNet 논문)

ResNet

그리고 우리가 공부 중인 GPT, Transformer에도 반드시 들어갑니다.

3️⃣ 왜 이렇게 하면 좋아질까?

여기서 핵심 질문입니다.

🎯 생각해봅시다.

만약 Layer가 잘 못 배웠다면?

원래는:

출력 = F(x)


→ 망하면 끝입니다.

하지만 숏컷이 있으면:

출력 = x + F(x)


만약 F(x)가 0에 가까워지면?

출력 ≈ x


👉 최소한 원래 정보는 유지됩니다.

즉,

“못 배우면 그냥 원래 값이라도 유지하자.”

이게 엄청난 안정성을 만듭니다.


4️⃣ 더 깊은 원리 (엘리트 모드)

숏컷 연결은 실제로 이런 의미를 가집니다:
Layer가 배우는 것은 전체 변환이 아니라 ‘변화량(Residual, 잔차)’ 입니다.

𝐹(𝑥)=𝐻(𝑥)−𝑥

즉,
"완전히 새로 만들지 말고, 기존 것에서 얼마나 바꿀지만 학습해라."

이게 학습을 훨씬 쉽게 만듭니다.

5️⃣ Transformer에서의 숏컷 연결

Transformer Block 안에서는 이런 구조입니다:

x → Multi-Head Attention → + x
  → Feed Forward → + 이전 출력


즉, 블록 하나에 숏컷이 2번 들어갑니다.

그래서 깊게 쌓아도 학습이 안정적입니다.

## skip connection(스킵 연결)
1️⃣ 한 문장 정의

중간 층을 “건너뛰어서” 이전 층의 출력을 뒤쪽 층으로 직접 전달하는 구조

입니다.

2️⃣ 왜 이런 걸 만들었을까?

깊은 신경망에서는 이런 문제가 생깁니다:

기울기 소실(Vanishing Gradient, 기울기 소실)

깊어질수록 학습이 느려짐

정보가 점점 희석됨

그래서 연구자들은 생각했습니다:

“그냥 중간을 거치지 말고, 직접 연결해버리면 안 될까?”

3️⃣ 구조를 그림으로 보면

기본 구조:

x → Layer1 → Layer2 → Layer3 → 출력


스킵 연결이 있으면:

x → Layer1 → Layer2 → Layer3
 \___________________________↗


즉,
초기 입력 x가 뒤쪽 Layer3로 직접 전달됩니다.

4️⃣ 그럼 Shortcut이랑 뭐가 달라?

좋은 질문입니다. 여기서 사고가 깊어집니다.

📌 관계 정리

Shortcut Connection (숏컷 연결) ⟶ 스킵 연결의 한 형태

Skip Connection (스킵 연결) ⟶ 더 넓은 개념

즉,

모든 숏컷은 스킵이지만
모든 스킵이 숏컷은 아닙니다.

대표적인 예:

ResNet → 입력을 더해주는 방식 (Residual, 잔차 연결)

DenseNet → 이전 모든 출력을 이어 붙이는(concatenate) 방식

Transformer → Attention 뒤와 FeedForward 뒤에 residual + layer norm

5️⃣ 왜 효과가 좋을까?
🎯 핵심 이유 3가지

정보 보존
→ 원래 입력이 사라지지 않음

그래디언트 흐름 개선
→ 역전파(backpropagation) 시 기울기가 직접 전달됨

학습 난이도 감소
→ 모델은 전체 변환이 아니라 "수정량"만 배우면 됨

6️⃣ 직관적 비유

스킵 연결이 없는 네트워크는:

전달게임 (끝으로 갈수록 왜곡됨)

스킵 연결이 있는 네트워크는:

원본 파일을 항상 같이 들고 다니는 구조

## skip-gram
✅훈련 구조
중심 단어를 하나 정하고, 그 주변에 있는 이웃 단어들을 하나씩 묶어서 매번 새로운 학습 예제를 만듦. 그리고 모델이 학습 후에는 “이 두 단어는 문맥상 이웃인가요?” → 1 또는 0으로 판단할 수 있게 됨.
🧠 쉽게 풀어보기
📘 예시 문장:
"The cat sits on the mat"
이 문장에서 중심 단어가 "sits"라고 하면,
양 옆 단어들: "cat", "on" (← 이게 "neighbors")

🧩 훈련 데이터 만들기 (window size = 1 기준)
중심 단어 (input1)	이웃 단어 (input2)	정답 출력
sits	cat	1 (이웃이다)
sits	on	1 (이웃이다)
sits	elephant	0 (무작위 단어, 이웃 아님) ← negative sample

🔁 이 과정을 반복하며 모델은 학습:
•	입력: 두 단어 (예: sits, cat)
•	출력: 1 (이웃 관계)
•	이렇게 쌍을 여러 개 만들어서 신경망에 학습시킴.
→ 이 학습 결과, 각 단어는 의미 기반으로 벡터 공간에 잘 정리 됨.

## sliding window (다음 토큰 예측에서의)
목표: 아주 긴 토큰 시퀀스를 모델의 최대 길이(context length)에 맞게 “겹치게” 잘라서 여러 학습 샘플을 만든다.

윈도우는 “연속 구간(chunk)을 어느 길이로 자를지”와 “얼마나 겹칠지(stride)”를 의미한다.

중심 단어 개념이 없고, “구간 단위”로 움직인다.

예시:
윈도우 크기 = 3
이동 간격 = 1

샘플 1: [1][2][3]
샘플 2:    [2][3][4]
샘플 3:       [3][4][5]

이렇게 하면:
각 샘플은 길이 제한을 지키고
이전 문맥이 겹쳐서 유지된다

## sliding window (word2vec)
word2vec에서 사용하는 기법으로써, 중심 단어를 기준으로 양쪽에 일정한 수의 이웃 단어(문맥)를 함께 보는 것. 윈도우 크기(window size)가 2면, 중심 단어 + 왼쪽 2개, 오른쪽 2개 단어를 사용.
📘 예제 문장:
"Thou shalt not make a machine in the likeness of a human mind"
단어를 나열하면:
[Thou, shalt, not, make, a, machine, in, the, likeness, of, a, human, mind]
🔍 예시: 중심 단어 = "make" (위 문장에서 4번째 단어)
•	윈도우 크기 2이면:
o	왼쪽 이웃 단어: "not", "shalt"
o	오른쪽 이웃 단어: "a", "machine"
→ 그러면 Word2Vec은 아래와 같은 훈련쌍을 생성함:
중심 단어	주변 단어 (context word)
make	not
make	shalt
make	a
make	machine

## softmax
1️⃣ 한 문장 정의

Softmax는 여러 개의 점수를 확률처럼 보이게 만들어 주는 함수입니다.

즉,

입력: 아무 숫자들이나
출력: 모두 0~1 사이
그리고 전체 합 = 1

2️⃣ 왜 필요한가?

Attention score는 그냥 이런 값들입니다:

[2.3, 1.1, -0.4, 3.0]


이 상태로는 누가 얼마나 중요한지 직관적이지 않고 음수도 있고 합도 일정하지 않음 그래서 softmax를 씁니다.

3️⃣ 어떻게 계산되나?

공식:

softmax(x_i) = e^(x_i) / Σ e^(x_j)


즉,

모든 값을 exp(지수 함수)로 바꾸고 전체 합으로 나눕니다.

4️⃣ 왜 exp를 쓰는가?

exp의 특징:

값이 조금만 커져도 급격히 커짐
작은 차이를 크게 강조

예:

2.0 → e^2 = 7.39
3.0 → e^3 = 20.08


1 차이였는데 3배 가까이 벌어짐. 즉, 중요도 차이를 더 극적으로 만듦

5️⃣ Attention에서 무슨 역할?

attention score → softmax → attention weight

##
이때 나온 weight는:

모두 양수
합이 1
확률처럼 해석 가능

그래서

context vector = Σ (attention weight × Value)

이게 가능해집니다.

7️⃣ 여기서 중요한 통찰

만약 softmax를 안 쓰면?

음수 weight 가능
합이 1이 아님
값이 폭주 가능
학습 불안정

## Scaled Dot-Product Attention
Scaled Dot-Product Attention은
Transformer에서 사용하는 어텐션(attention) 계산 방식입니다.

이름을 그대로 해부해 보면:

1) Dot-Product → 쿼리(Query)와 키(Key)의 내적
2) Scaled → 그 내적 값을 √d 로 나눠서 스케일 조정
3) Attention → 그 결과를 이용해 **가중합(Weighted sum)**을 계산

즉,

Query와 Key의 유사도를 계산하고
그 값을 정규화해서
Value를 가중합해 Context vector를 만드는 방식

4️⃣ 왜 중요한가?

RNN은 순차 처리
CNN은 지역 정보만 봄

하지만 Scaled Dot-Product Attention은

✔ 전체 문장을 한 번에 본다
✔ 병렬 연산 가능
✔ 장거리 의존성(Long-range dependency) 해결

이게 Transformer 혁명의 핵심입니다.



## SoTA
✅ SoTA란?
SoTA = State of the Art
→ “최고 수준의 성능” 또는 “당시 기준에서 가장 앞선 기술”을 의미

✅ 쉽게 말하면?
•	어떤 작업(task)에서
👉 지금까지 나온 모든 모델 중 가장 높은 성능을 달성했다!
👉 그래서 “SoTA 모델이다” 라고 부름

✅ 예시
예시 문장	해석
"Our model achieves SoTA on MMLU"	“우리 모델이 MMLU 벤치마크에서 최고 성능을 기록했다”
"This is a SoTA result for 3B models"	“3B 모델 중에서 성능이 가장 좋다는 뜻”

✅ SoTA는 누가 정해?
SoTA는 **공식 벤치마크 평가 기준(예: MMLU, HellaSwag, GSM8K 등)**에서
점수가 가장 높을 때 인정받는 것이고, 논문이나 오픈 리더보드(Hugging Face, Papers With Code 등)를 통해 공개 됨.

✅ 요약
항목	설명
SoTA 뜻	State of the Art (최고 성능, 최신 최고 기술)
어디서 사용?	AI 성능 비교, 논문 제목, 모델 홍보
의미	현재 기준에서 가장 성능이 좋은 모델이라는 뜻

## sparse attention
기존의 full attention보다 계산량을 줄이기 위해 일부 토큰들만 선택적으로 주의(attend) 하도록 설계된 방식.
✅ 1. 기본 배경: Full Attention 문제점
기존 Transformer의 self-attention은 다음과 같음:
•	모든 토큰이 모든 다른 토큰들과 attention을 수행함
•	시간 복잡도: O(n²) (토큰 수가 n개일 때)
→ 예: 문장 길이가 길어질수록 계산량 폭증, 메모리 사용량도 큼

✅ 2. Sparse Attention이란?
모든 토큰끼리 연결하지 않고,
일부 토큰들끼리만 선택적으로 연결해서 attention을 수행하는 방법
예:
•	현재 토큰이 자기 앞뒤 3개만 참조
•	또는 특정 간격으로 떨어진 토큰들만 참조
📌 결과:
•	계산량 줄어듦 (복잡도 O(n log n), O(n√n) 등으로 줄이기도 함)
•	길이가 긴 입력에도 더 효율적
•	성능은 비슷하거나 약간 손해 보더라도, 속도와 메모리 효율성이 향상됨

✅ 3. 대표적인 Sparse Attention 기법들
기법 이름	설명	사용 사례
Local Attention	인접한 몇 개 토큰만 봄	Longformer
Strided Attention	일정 간격으로 띄엄띄엄 봄	BigBird
Global Tokens	중요한 일부 토큰만 전체 참조	Longformer, BigBird
Random Attention	무작위로 일부 토큰만 봄	BigBird
Axial Attention	2D 입력에 행/열 방향으로만 attention	Image Transformer 등

✅ 4. 사용 이유 요약
•	성능 vs 효율성 트레이드오프
•	긴 문장, DNA 서열, 문서 요약, 멀티모달 입력 등에서 특히 유용
•	대표 모델: Longformer, BigBird, Reformer, Sparse Transformer

## SST-2
✅ 먼저, SST-2란?
SST-2는 Stanford Sentiment Treebank v2의 줄임말로,
영화 리뷰 문장을 이진 감정(긍정 vs 부정)으로 분류하는 태스크용 데이터셋
🎬 데이터 출처: Rotten Tomatoes 영화 리뷰
📦 포함 내용: 문장(text), 감정 라벨(0=부정, 1=긍정)
🧪 GLUE 벤치마크의 구성 태스크 중 하나로 널리 사용됨

🔍 그런데 왜 Hugging Face에서 검색하면 여러 개가 나올까?
이유	설명
🧪 다양한 포맷	원본 SST-2를 GLUE 포맷, CSV, TFRecord 등 다양한 형식으로 제공
🏢 다양한 출처	Hugging Face 사용자나 기관이 자체적으로 재가공한 버전 업로드
🧪 평가 목적 차이	어떤 건 dev set 포함, 어떤 건 test label 없음 (공식 평가용)
🤖 전처리 방식 차이	소문자화(uncased), 특수문자 제거 등 사전처리 방식이 다를 수 있음


## stop words
자연어 처리(NLP)에서 의미 분석에 거의 도움이 되지 않는 단어들을 말함. 한국어로는 "불용어(不要語)"라고 번역되며, 주로 너무 자주 등장하지만 중요한 의미는 없는 단어들을 제거할 때 사용

📌 예시
언어	Stop words 예시
영어	the, is, a, of, in, and, to, it, that
한국어	이, 그, 저, 은, 는, 이, 가, 를, 을, 도, 의

🧠 왜 제거하나요?
•	예를 들어 "The dog is cute"에서 the, is는 거의 모든 문장에 나오는 단어라, 문장의 주제를 구별하는 데 도움이 안됨.
•	그래서 텍스트 클러스터링, 토픽 모델링, 검색엔진 등에서는 stop words를 미리 제거해서 중요한 단어(명사, 동사 등) 위주로 분석.

🧩 BERTopic 같은 모델에서는?
•	stop_words='english' 또는 stop_words=None 옵션으로 조절 가능
•	보통 텍스트 전처리 과정에서 stop words를 제거한 후 토픽 모델링을 하면 더 좋은 품질이 나옴.

왜 불용어를 제거 했는지에 대한 이유 (역사적 맥락 중요)
과거 NLP (Bow, TF-IDF 시절): 자주 나오는 단어는 중요하지 않음.
예)
‘나는 밥을 먹었다‘
‘나는 학교에 갔다‘
‘나는‘ 너무 흔함 분류에 도움이 안됨
그래서 불용어 제거, 차원 축소, 성능 및 속도를 개선시킴. 이 시기에는 Stop-word 제거가 필수 전략 이었음

지금 LLM 시대는?
불용어 전략이 필요 없음. Transformer는 문맥을 보기 때문에 ‘은/는/이/가＇도 주재, 강조, 대조 같은 역할을 하기 때문에 중요함.
조사, 전치사는 이미 Sub-word로 쪼개져서 Attention에서 자동으로 가중치를 조절하므로 사람이 ‘쓸모없다＇고 미리 판단할 필요가 줄어듦.

따라서 Stop-word (불용어)는 의미없는 단어가 아니라 ‘과거의 특정 알고리즘에서 의미 구분에 덜 도움 됐던 단어＇라고 봐야 함
하지만 Stop-word가 여전히 유효한 상황도 있음. ‘검색/색인‘, ‘로그/구조적 데이터‘, ‘VectorDB 비용 최적화(임베딩전에 노이즈 제거, 토큰 수 절감)’. 즉 ‘언어 이해＇가 아니라 ‘저장 / 검색 / 비용‘ 문제일 때 여전히 유효함.

## Structured Outputs


## suna.so
Kortix에서 개발한 오픈소스 AI 에이전트 플랫폼으로, 자연어 대화를 통해 실제 작업을 자동화할 수 있는 범용 AI 어시스턴트. 
•	Suna.so는 오픈소스로, 커스터마이징 가능한 범용 AI 에이전트 플랫폼.
•	자연어로 업무 자동화 요구를 적어주면, 웹 탐색 → 도구 사용 → 결과 작성까지 처리.
•	기업 내부 도입 시 EXCEL 자동화, 리서치, 보고서 작성, 이메일 발송 등 다양한 사용 사례에 활용도 높음.
•	Apache‑2.0 라이선스로 상업적 활용에도 적합.
GPU나 NPU 서버가 있어야 하나?
✅ 1. 간단한 테스트나 경량 모델 사용 시 → CPU만으로도 가능
•	suna는 Hugging Face나 OpenAI API 등 외부 LLM도 연동 가능하므로, LLM을 외부에서 제공받는 경우라면 CPU 서버에서도 충분히 실행 됨.
•	예: OpenAI API를 써서 GPT-4를 연결 → 연산은 OpenAI 서버에서 처리됨.

✅ 2. 자체 LLM 실행 (예: Phi-3, LLaMA, Mistral 등) → GPU 필수
•	로컬에서 대규모 언어 모델을 직접 실행하고자 한다면, **최소 1개의 GPU(VRAM 16GB 이상)**가 필요.
•	예: Phi-3-mini, Mistral-7B, LLaMA-3 등을 AutoModelForCausalLM.from_pretrained()으로 직접 로딩할 경우 → GPU 없이 느리거나 실행 불가.

✅ 3. NPU 서버 사용 가능?
•	NPU(Neural Processing Unit)도 PyTorch/XLA, PyBUDA 등의 프레임워크로 LLM 추론을 실행할 수 있다면 이론적으로는 가능.
•	하지만 Suna가 기본적으로 지원하는 건 일반 GPU(CUDA 기반).
Tenstorrent, 퓨리오사AI 같은 NPU를 쓰려면 추가 통합 작업이 필요할 수 있음.

**suna.so**는 아래 구조에서 웹서버 역할에 해당
✅ 구조로 정리하면:
[사용자]
   │
   ▼
[🌐 웹서버 역할: suna.so]
   │
   ▼
[⚙️ 디스패처 / 라우터 (옵션)]
   │
   ▼
[💻 NPU 서버 + Pre-Trained AI Model]

🔍 각 구성요소 설명
구성 요소	역할
사용자	웹브라우저나 앱으로 질문, 요청, 지시 등을 입력
🌐 웹서버 (예: suna.so)

•  사용자 요청을 받음
•  토크나이즈, LLM 프롬프트 구성
•  결과 포맷팅
•  UI와 연동하여 결과 반환
•  백엔드 LLM 서버 호출
→ 중간 관리자 & 프런트 + 비즈니스 로직 담당 |
| 디스패처/라우터 (선택) |
•  요청을 다양한 LLM 서버로 분산
•  GPU/NPU 서버 로드밸런싱
•  템플릿별 모델 선택 등 |
| 💻 NPU 서버 + AI Model |
•  Phi-3, LLaMA, Qwen 등 Pre-trained 모델 실행
•  텐스토렌트/퓨리오사 등의 NPU에서 추론 처리
→ 실제 연산 처리 담당 |

✅ 추가 정리: suna.so는 어떤 일을 하나?
•	사용자의 자연어 입력을 파싱하고
•	상황에 따라 적절한 프롬프트를 구성하며
•	GPU 혹은 NPU 서버에게 모델 추론 요청을 생성하고 결과를 받아서
•	답변/문서/요약 등 가공된 형태로 반환.
즉, 웹서버 + 에이전트 시스템 역할을 함께 수행

✅ 결론
•	suna.so는 자체적으로 추론을 하지 않음.
•	대신, 외부/로컬의 Pre-trained AI 모델에 요청을 보내고 결과를 받아 사용자에게 전달.
•	따라서 GPU/NPU 서버는 백엔드, suna.so는 웹 인터페이스 + 비즈니스 로직 계층

✅ 한국어 사용 가능 여부
항목	지원 여부
한국어 UI	✅ 가능
한국어 프롬프트 가능	✅ 사용 가능
공식 문서 한국어화	✅ README에 다국어 구성 포함

SPECInt
항목	설명
정식 명칭	SPECint = Standard Performance Evaluation Corporation + Integer
만든 곳	SPEC (국제 표준 성능 평가 협회)
측정 대상	CPU의 정수형 연산 처리 능력 (integer performance)
결과 단위	점수 (높을수록 성능 좋음)
사용 목적	CPU 설계자, 칩 회사, 서버 벤더 등이 성능 비교에 사용

🔹 예를 들어 이런 상황에서 사용
“우리 CPU가 경쟁사보다 SPECInt 10% 높습니다”
“이 CPU는 SPECInt2006 기준으로 300점 나옵니다”
즉, 정수 기반 컴퓨팅 처리 능력을 보여주는 성능 수치

🧠 실생활에서 중요한 이유
CPU 성능은 여러 가지 방식으로 측정할 수 있는데,
•	SPECInt: 정수 연산에 특화된 벤치마크
•	SPECfp: 실수 연산 (floating point)
•	Geekbench, Cinebench, PassMark 등 다양한 벤치마크 존재
하지만 SPECInt는 서버, HPC, RISC-V, ARM CPU 분야에서 특히 많이 쓰임.

📌 요약
항목	설명
SPECInt란?	CPU의 정수 연산 처리 성능을 측정하는 국제 표준 벤치마크
누가 씀?	칩 제조사, 서버 업체, 성능 비교 보고서
이 문장에서 의미	"그래프에 나온 성능 점수는 SPECInt 기준이고, 실제 응용 프로그램에서의 성능은 이와 다를 수 있다"는 뜻

## Supervised Finetuning (SFT)
대규모 언어 모델(LLM) 훈련의 핵심 단계 중 하나로, 모델에게 **“이런 식으로 대답하는 게 좋아”**라고 구체적으로 알려주는 교과서 기반 훈련 단계

🔷 Supervised Fine-Tuning (SFT) 한 줄 정의
정답이 있는 데이터(질문-답변 쌍 등)를 이용해, 이미 사전학습(pretraining)된 모델을 미세하게 조정하는 과정

🔶 SFT가 왜 중요한가요?
사전학습된 모델은 위키, 책, 웹문서 등을 바탕으로 지식을 배웠지만… 말을 어떻게 예쁘게 하고, 사용자 질문에 어떻게 응답해야 할지는 학습이 안 되어 있음.
➡ 그래서 SFT로 **"사람처럼 응답하는 방법"**을 학습시킴.

🔷 예시로 이해하기
입력 (Instruction)	출력 (Expected Answer)
“파리를 여행하려면 어떤 준비를 해야 해?”	“파리를 여행할 때는 비자, 항공권, 숙소를 미리 준비하셔야 합니다…”
“다항식을 미분해줘: 3x² + 2x + 1”	“정답은 6x + 2입니다.”

🔷 SFT vs Pretraining vs RLHF 비교
단계	설명	사용 데이터	목적
Pretraining	인터넷 데이터로 지식 학습	위키, 책, 웹문서 등	언어/지식 이해
SFT	정답이 있는 데이터로 대화 능력 훈련	질문-답변 쌍	말 잘하게 만들기
RLHF	사람 피드백을 기반으로 세련된 조정	사람 선호도 순위	인간 친화성 강화

🔷 SFT의 특징 요약
항목	설명
🧠 학습 방식	지도 학습(Supervised learning)
📚 데이터 형식	(입력, 정답 출력) 쌍
📦 데이터 예시	자연어 질문 → 모범 답변, 수학 문제 → 정답
🎯 목적	모델이 질문에 “사람처럼 자연스럽게” 대답하게 만듦
🔧 사용 시기	Pretraining 후, RLHF 전에 수행

🔸 어디서 쓰였나?
•	OpenAI GPT-4: Pretraining 후 SFT로 인간 응답 학습
•	Claude 3, Gemini, Mistral 등 거의 모든 LLM이 이 단계를 거침
•	HuggingFace, OpenAssistant, DeepSeek 등에서도 공개 SFT 데이터셋 다수 존재

✅ 요약
항목	내용
이름	Supervised Fine-Tuning (SFT)
의미	정답이 있는 데이터를 사용한 지도학습 기반 미세 조정
목적	모델이 사용자 질문에 정확하고 자연스럽게 대답하게 만듦
위치	Pretraining → SFT → RLHF
대표 예시	Alpaca, OpenAssistant, Dolly, DeepSeek-Coder-Instruct 등

## sub-word
언어를 어디까지 쪼개서 이해할 것인가라는 설계 철학

왜 Sub-word가 필요한가?
Word 기준이면 ‘공부한다‘ ‘공부했다’ ‘공부할까‘ ‘공부했었다‘ 가 전부 다 다른 단어. Vocabulary 폭발. OOV(Out-Of-Vocabulary, 미등록 단어) 지옥
한국어는 특히 조사, 어미, 활용적 특성 때문에 단어 수가 이론적으로 무한
Character 단위는 너무 원시적
공 / 부 / 하 / ㄴ / 다 => 의미가 거의 없음. 시퀀스 길이 폭증. 학습 비효율

그래서 등장한 것이 Sub-word
의미가 어느 정도 유지되는 최소 단위
공부 + 하 + ㄴ + 다 => ‘공부‘가 의미의 핵. ‘하‘ 동사화. ‘ㄴ다’는 시제종결  모델이 패턴을 재사용할 수 있게 됨.

한국어에 특히 잘 맞는 이유
한국어의 구조적 특징 때문 (특징; 교착어, 의미 덩어리 + 기능 덩어리의 결함)
예) AI를 = AI + 를(조사)

Sub-word는:
의미소에 가깝게 분해
형태소 분석기 없이도 통계적으로 분해 가능 (언어학 지식 없이도 작동 => 핵심포인트)

Sub-word가 모델에 주는 이득
OOV 문제 거의 제거
Vocabulary 크기 감소
희귀 단어의 일반화 가능
다국어 모델에 유리

Sub-word 기반 모델
BERT, GPT, T5, LLaMA, Qwen

## T5 architecture
항목	설명
정식 명칭	Text-to-Text Transfer Transformer
발표	Google, 2020년 논문 "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer"

핵심 개념	모든 NLP 작업을 '텍스트 → 텍스트' 문제로 통일해서 처리하자
구조	Transformer 기반의 Encoder-Decoder 아키텍처
활용	번역, 요약, 질문 생성, 감정 분석 등 거의 모든 NLP 작업

🧠 왜 Text-to-Text인가?
•	예전에는:
o	분류는: 텍스트 → 숫자 (0 or 1)
o	번역은: 텍스트 → 텍스트
o	요약은: 텍스트 → 텍스트
→ 형식이 각각 달랐음
•	그런데 T5는:
"다 텍스트 → 텍스트로 바꿔버리자!"
예:
•	분류: "문장이 긍정이야 부정이야?" → "positive" 또는 "negative" 생성
•	번역: "Translate English to French: I love you" → "Je t’aime"
•	요약: "Summarize: The article says..." → "Summary sentence"

✅ 요약 정리
항목	설명
구조	Encoder + Decoder (Transformer 기반)
입력	텍스트 시퀀스 (예: 질문, 문장 등)
출력	텍스트 시퀀스 (예: 정답, 요약 등)
목적	다양한 NLP 작업을 하나의 통일된 형식으로 처리
대표 특징	모든 작업을 텍스트 → 텍스트 형태로 다룬다

## Task-specific model
특정 과제(Task)에 맞게 훈련된 모델
예: 감정 분류, 요약, 번역, 질의응답 등
예시:
작업(Task)	모델 입력	모델 출력
감정 분석	영화 리뷰 텍스트	긍정 / 부정
요약	긴 기사	요약된 문장
개체명 인식 (NER)	문장	[서울] → 장소, [이재용] → 인물 등
➡ 이 모델은 **“문장에 대해 정답이 정해져 있는 태스크”**에 맞게 fine-tuning 됨.

⚖️ 비교 핵심:
항목	Task-specific model	Generative model
목적	분류 (classification)	텍스트 생성 (generation)
입력	텍스트 시퀀스	텍스트 시퀀스
출력	숫자 하나 (예: 1)	텍스트 시퀀스
모델 유형	Sequence-to-value	Sequence-to-sequence
유연성	낮음 (작업 하나에 특화됨)	높음 (다양한 작업 수행 가능)

## tensix core
Tenstorrent 칩 안에서 AI 연산을 실제로 해 주는 "작은 두뇌". Wormhole의 경우 카드 당 2개의 칩이 장착되어 있고, 각 칩 안에 무수히 많은 Tensix Cores가 위치해 있음.
 
그림 1. 위 이미지는 Wormhole 카드의 칩 구조를 보여주고 있음 (D = DRAM, T = Tensix, E = Ethernet, A = ARC/management, P = PCIe).

각 Tensix Core에는 5개의 RISC-V CPUs와 2 NoC 인터페이스, SFPU 벡터 유닛, FPU 매트릭스 유닛 그리고 paker와 unpakcer로 구성되어 있음. 또한 1.5MB의 SRAM도 있는데, 이 SRAM은 hold transient data(임시 데이터 저장), 다른 콤퍼넌트 간 데이터 교환, 전송에 사용 됨. 하기 이미지는 Tensix Core의 구조도를 보여 줌(파란색 화살표: 명령어 디스패치(해석; 제어 스케줄링 + 분배의 의미), 갈색 화살표: 데이터 전송/교환을 의미)

 
그림 2. 파란색 상자는 5개의 RISC-V CPU를 뜻함. 파란색 화살표는 명령어 디스패치(해석; 스케줄링 + 분배의 의미), 갈색 화살표는 데이터 전송/교환을 의미. 결국 RISC-V CPU는 NoC/Matrix/Vector/(Un)pakcer를 제어, 분배 역할을 하여 각 유닛들이 실제 연산을 할 수 있게 함.
일반적인 데이터의 흐름은 아래와 같음.
•	NoC interface 0은 DRAM 혹은 다른 Tensix Core로부터 데이터를 읽어 들여옴
•	Unpacker는 Matrix/Vector 유닛이 프로세스할 수 있는 포맷으로 데이터를 unpack(변형)함.
•	Matrix/Vector 유닛에서 연산 실시
•	Packer는 Unpacker가 변형한 포맷의 데이터를 원래 원형(일반 바이너리 데이터)으로 되돌림.
•	NoC interface 1은 결과 데이터를 DRAM이나 다른 Tesix Core로 보냄

 
그림 3. Tensix Core내에서의 전형적인 데이터 흐름도. 그림 상으로는 적색 화살표인 내부 네트워크(NoC)는 원형 단방향 구조로 표현되어 있지만, 두 개의 NoC가 서로 반대 방향으로 데이터를 흘려주기 때문에, 칩 전체적으로는 양방향 통신에 가까운 quasi-full-duplex 방식으로 동작함. 이로써 두 개의 NoC가 동시에 데이터를 보내고 받을 수 있음. 결과적으로 데이터 전송 속도가 빨라지고, 병목 현상이 줄어듦. 이러한 단방향 구조는 소모 전력(power)을 줄이고, 칩 면적(silicon area)도 절약함.
Noc: 칩 안에서 Tensix Core끼리 데이터를 주고받게 해주는 통신망

 
그림 4. 5개의 RISC-V 코어가 있으니 개발자는 그에 맞는 5개의 프로그램이 필요하다고 생각할 수 있지만, 실제로는 개발자는 Reader Kernel, Compute Kernel, Writer Kernel 세 부분으로 프로그램을 작성하면 되기 때문에 개발자의 프로그램 작성 부담을 줄임.
그림 4와 그림 5는 세 개의 Kernel 영역이 버퍼 용도로 SRAM을 사용하여 데이터를 주고 받는다는 것을 설명함.
 
그림 5. Reader, Compute, Writer Kernel 통신 형상


## tensor
✅ 1. Scalar (스칼라)
📌 뜻:
숫자 1개, 즉 크기만 있는 값.
📦 예시:
•	온도: 25도
•	몸무게: 70kg
•	시간: 3초
→ 방향이 없고, 단순한 수치 하나.
💻 프로그래밍에서는?
x = 5.0   # 스칼라 (Scalar)

✅ 2. Vector (벡터)
📌 뜻:
숫자 여러 개가 한 줄로 나열된 것
= 1차원 배열 (1D array)
📦 예시:
•	바람: [속도 5m/s, 북쪽 방향]
•	사람 키 정보: [150, 160, 170]
•	RGB 색상: [255, 0, 0] → 빨강
💻 프로그래밍에서는?
x = [1.0, 2.0, 3.0]  # 벡터 (Vector)

✅ 3. Tensor (텐서)
📌 뜻:
스칼라, 벡터, 행렬 등 모든 것을 포함하는 더 큰 개념
= 다차원 배열 (Multi-dimensional array)
차원	예	용도
0D	스칼라	숫자 하나
1D	벡터	리스트
2D	행렬(Matrix)	이미지, 표
3D 이상	텐서(Tensor)	딥러닝 모델 입력 등

📦 예시 (3D 텐서):
•	3장의 흑백 이미지 (각 28x28 픽셀)
→ [3, 28, 28] 텐서
•	128장의 이미지 배치: [128, 3, 224, 224]
→ (배치 크기, 채널, 높이, 너비)
💻 파이토치 예시:
import torch

scalar = torch.tensor(7)               # 0D
vector = torch.tensor([1, 2, 3])       # 1D
matrix = torch.tensor([[1, 2], [3, 4]]) # 2D
tensor = torch.rand(4, 3, 2)           # 3D 텐서

🎯 한 줄 요약
구분	의미	예시
Scalar	숫자 1개	3.14
Vector	숫자 줄	[1, 2, 3]
Tensor	숫자 뭉치	[[[...]]] (다차원 배열)

## Tensor Parallelism (텐서 병렬 처리, TP=2)
하나의 모델을 여러 GPU에 걸쳐서 나누어 계산. 예: 하나의 큰 레이어의 연산을 2개의 GPU가 분담.

## top_k
top-k Sampling (탑-k 샘플링) 은 확률이 가장 높은 상위 k개 후보 토큰만 남기고, 그 안에서 무작위로 샘플링하는 방식. 즉, **“다음 단어를 고를 때, 확률이 가장 높은 k개만 고려한다”**는 원리.
🔎 예시
모델이 다음 단어 확률을 이렇게 예측했다고 가정하면:
A(0.40), B(0.30), C(0.15), D(0.10), E(0.05)
•	top_k=1 → 가장 확률이 높은 A만 선택됨 → 항상 A. (= Greedy Search, 탐욕적 탐색)
•	top_k=2 → A, B만 후보로 두고 랜덤 샘플링
•	top_k=3 → A, B, C 후보 중 랜덤 샘플링
•	top_k=5 → 모든 후보 사용 (사실상 제한 없음)

📊 top_p vs top_k 차이
•	top_k = “후보 개수를 정해놓고 제한” (예: 무조건 상위 3개만 보겠다)
•	top_p = “확률 누적합 기준으로 제한” (예: 확률 합이 0.9가 될 때까지만 보겠다)

✅ 정리
•	top_k는 “상위 몇 개까지만 남길지”
•	top_p는 “확률 누적이 몇 %까지 남길지”
•	둘 다 무작위성을 조절해서, 너무 기계적인 답변 대신 다양성을 주는 역할을 함

## top_p
Nucleus Sampling (누클리어스 샘플링) 이라고도 불리는 샘플링 기법을 제어하는 하이퍼파라미터. LLM(대형 언어 모델)이 어떤 다음 토큰(token, 단어 조각)을 생성할 때, 모든 후보 토큰의 확률 분포를 계산.
동작 원리
1.	모델이 "다음에 올 수 있는 토큰 후보들"의 확률을 계산.
예: A(0.4), B(0.3), C(0.15), D(0.1), E(0.05)
2.	이 확률들을 큰 것부터 정렬한 뒤,
확률 누적합(cumulative probability)이 top_p 값 이상이 될 때까지 후보를 모음.
o	만약 top_p=0.9라면 → 상위에서 확률을 더해 0.9 이상이 될 때까지만 후보로 인정함.
o	위 예시라면 A(0.4) + B(0.3) + C(0.15) = 0.85 (아직 0.9 안 됨)
여기에 D(0.1)를 포함하면 0.95 → 즉, A, B, C, D만 후보가 됨.
3.	최종적으로 남은 후보들 안에서 무작위 샘플링을 함.
직관적인 의미
•	top_p 값이 작을수록 → 상위 확률이 높은 단어들만 남기므로, 보수적이고 예측 가능한 문장이 나옴.
•	top_p 값이 클수록 → 후보 풀(pool)이 넓어져서, 다양하고 창의적인 문장이 나옴.
자주 쓰는 값
•	top_p=1.0 → 모든 후보를 고려 (즉, nucleus sampling 비활성화 = 전체 확률 분포 사용)
•	top_p=0.9 → 많이 쓰이는 기본값, 다양성과 일관성의 균형
•	top_p=0.7 → 더 보수적인 생성
👉 쉽게 말하면:
•	temperature(온도) 는 “확률 분포를 날카롭게/퍼지게” 만드는 역할,
•	top_p 는 “어느 정도 누적 확률까지 후보를 자를지” 정하는 역할.

## tvm
AI 모델을 다양한 하드웨어에서 빠르고 효율적으로 실행할 수 있게 해 주는 컴파일러 프레임워크.

✅ 한 줄 정의
Apache TVM은 AI 모델을 GPU, CPU, NPU 등 다양한 하드웨어에 맞게 최적화해서 실행시켜주는 오픈소스 컴파일러

🔍 왜 필요한가요?
AI 모델은 대부분 PyTorch나 TensorFlow로 만들지만,
→ 이걸 그대로는 GPU, NPU, ARM 칩 등에 바로 실행할 수 없음.
그럼 어떻게 해야 할까?
🔧 모델을 하드웨어에 맞게 바꿔주고 최적화해주는 컴파일러가 필요한 거예요.
이걸 해주는 게 바로 Apache TVM

🧠 쉽게 예를 들면
•	모델: PyTorch로 만든 BERT
•	목표: 이걸 Nvidia GPU / ARM CPU / Tenstorrent NPU에서 실행하고 싶다!
•	해결책: TVM이 중간 IR(중간 코드)로 변환해서 → 하드웨어별 코드로 바꿔줌
→ 그래서 하나의 모델을 다양한 플랫폼에 최적화해서 돌릴 수 있음

💡 주요 특징
기능	설명
📦 프론트엔드 지원	PyTorch, TensorFlow, ONNX 등 모델 포맷 불러오기 가능
⚙ 백엔드 타겟 다양	x86, CUDA(GPU), OpenCL, ARM, RISC-V, NPU 등
🚀 자동 최적화	TVM이 알아서 가장 빠른 연산 방식으로 튜닝해줌
🧪 AutoTVM / Meta-Scheduler	머신러닝 기반으로 최적의 커널 조합 자동 탐색
🔬 커스텀 커널 가능	직접 low-level 연산을 정의해서 넣을 수도 있음

🎯 요약
항목	내용
이름	Apache TVM (Tensor Virtual Machine)
역할	AI 모델을 다양한 하드웨어에 맞게 컴파일하고 최적화
지원 하드웨어	CPU, GPU, NPU, ARM, RISC-V 등
핵심 기능	자동 튜닝, 커널 최적화, 하드웨어별 코드 생성
관련 분야	PyTorch → NPU 실행, 자동 커널 튜닝, 저전력 AI 배포 등

## temperature parameter
생성형 AI가 다음 단어를 고를 때의 “랜덤성(무작위성) 정도”를 조절하는 값. 쉽게 말해, 모델이 얼마나 창의적/보수적으로 말할지를 정하는 다이얼
✅ 한 줄 요약
temperature 값이 낮으면 → 더 정답처럼 말하고,
temperature 값이 높으면 → 더 다양하고 창의적으로 말함
✅ 개념 설명 (자연어로)
temperature 값	의미	예시 상황
0.0 ~ 0.3 (낮음)	매우 보수적 — 거의 항상 확률이 가장 높은 단어만 선택	기술 문서, 공식 문장, 예측 가능한 출력
1.0 (기본값)	보통의 랜덤성 — 일반적인 대화	대부분의 ChatGPT 대화 스타일
1.5 이상 (높음)	매우 창의적/랜덤 — 뜻밖의 단어나 문장이 나올 수 있음	시, 이야기 생성, 창의적 글쓰기

✅ 숫자로 설명 (수학적 뉘앙스)
모델은 다음 단어 후보에 대해 이런 **확률 분포(Probability Distribution)**를 가짐:
토큰	원래 확률	temperature 적용 후
"Hello"	60%	temperature ↓
"Hi"	25%	temperature ↑
"Hey"	10%	temperature ↑
"Yo"	5%	temperature ↑
•  temperature ↓ 하면 확률 차이가 더 극단적으로 벌어져서 "Hello"만 자주 뽑힘
•  temperature ↑ 하면 확률이 더 평평해져서 다양한 단어가 선택될 수 있음

✅ 예시
🎯 Temperature = 0.2
Prompt: What’s the best way to learn Python?
결과:
The best way to learn Python is to follow the official documentation and practice exercises regularly.
👉 모범답안 스타일, 반복해도 거의 같은 결과

🎨 Temperature = 1.2
Prompt: What’s the best way to learn Python?
결과:
Dive into messy code, join weird online hackathons, build a robot that brews coffee — and break stuff often.
👉 창의적이고 다양하지만, 다소 튀는 표현도 등장

✅ 코드 적용 예시 (transformers)
generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    temperature=0.8,   # 🔹 이 부분이 temperature 조절
    max_new_tokens=100
)

✅ 정리 요약
항목	설명
정의	모델이 다음 단어를 고를 때 “무작위성”을 얼마나 적용할지 조절하는 값
범위	보통 0.1 ~ 2.0 (0은 극단적인 정답 지향, 2는 매우 창의적)
적용 예	뉴스 요약 = 0.2 / 창작 이야기 = 1.2

## ternary LLMs
**"Ternary LLMs (3진수 대형 언어 모델)"**은 지금까지의 AI 모델과는 숫자 표현 방식부터 다른 접근법을 사용. 

✅ 먼저, 숫자 표현부터 이해하자: Float → Int → Ternary
방식	값 표현	설명
Float32 (부동소수점)	-0.00001 ~ +1000.25	정확도는 높지만 계산량이 크고 느림
INT8 (정수 8비트)	-128 ~ 127	딥러닝 경량화에서 자주 사용
Ternary (3진수)	{-1, 0, +1}	오직 세 가지 값만 사용 (매우 간단하고 계산 빠름)

🧠 Ternary LLM 이란?
모델의 파라미터(가중치)와 연산을 모두 { -1, 0, +1 }만 사용해서 구성하는 대형 언어 모델

예시로 보면:
기존 모델 weight:   [0.4, -0.6, 1.2, 0.0, -0.1]
Ternary 모델 weight: [ +1,  -1,  +1,  0,   -1 ]

📌 왜 이런 방식이 중요할까?
•	계산이 매우 단순함 (곱셈이 필요 없을 수도 있음!)
•	NPU, FPGA, ASIC 같은 하드웨어에서 전력 소모와 연산 속도를 크게 줄일 수 있음
•	모델 크기가 압축됨 (메모리↓)

🧩 어떤 기술들이 ternary 방식으로 LLM을 만들 수 있을까?
1. Ternary Weight Quantization (3진 양자화)
모델의 가중치(weight)를 훈련 후에 3진수로 바꾸는 방법.
2. Ternary Training (3진 기반 훈련)
훈련 중부터 { -1, 0, +1 }만을 사용해서 gradient를 제한하며 학습.
기술적으로 어려움 (non-differentiable이기 때문)
해결책: Straight Through Estimator (STE) 등 사용
3. Sparse Ternary
0이 많다는 것을 활용해 sparsity까지 겸함
→ { -1, 0, +1 } 중 대부분이 0이라면 계산량은 훨씬 더 줄어듬

🔋 장점
항목	설명
💡 연산 단순화	곱셈 없이 비트 수준 덧셈/뺄셈으로 처리 가능
⚡ 전력 효율	float 연산보다 훨씬 전기 적게 씀
📦 모델 크기 감소	32bit → 2bit면 무려 16배 압축 가능
💨 추론 속도 증가	메모리 이동량이 적고, 연산이 간단해서 빠름

⚠️ 단점
항목	설명
🧠 정확도 손실	너무 단순한 표현이라 모델 성능 하락 가능성 큼
🔧 훈련 난이도 ↑	gradient 계산이 어려워서 학습이 불안정할 수 있음
❌ 실전 적용 부족	아직 대형 LLM 수준에서 완전 ternary로 상용화된 사례는 거의 없음
🧩 HW 특화 필요	GPU 같은 범용 하드웨어에서 느림 → 전용 NPU나 커스텀 칩이 필요함

🧪 대표 논문 / 연구 예시
제목	설명
"TernaryBERT" (2020)	BERT 모델을 ternary로 줄이는 실험. 성능은 다소 떨어졌지만, 크기·속도 향상 확인
"TTQ: Ternary Weight Networks" (2016)	고전적인 3진 양자화 방식 연구. 모델 훈련 방법 제시
BitNet (2023)	1~2bit로 LLM을 만든다는 개념 연구로, Ternary 연구와 연결 가능

🔧 LLM에 ternary를 쓰려면 하드웨어도 바뀌어야 한다?
그렇습니다. 특히:
조건	설명
GPU	곱셈 기반이라 ternary 최적화가 어려움
NPU (Tenstorrent 등)	tile 단위, custom op 허용, low-bit 연산 가능 → 최적화 여지 있음
ASIC	ternary 전용 회로 구현 가능 → 이론상 최적

🧠 정리
질문	답변
Ternary LLM이란?	모델 파라미터를 { -1, 0, +1 }로 제한한 초경량 대형 언어 모델
왜 중요해?	추론 속도, 전력 소비, 모델 크기 크게 줄일 수 있음
문제는?	성능 저하, 훈련 난이도, 실전 적용 어려움
Tenstorrent 같은 NPU랑 관련 있어?	✅ 가능성 있음. 특히 tile-based 구조에서 효율적으로 적용될 수 있음


## text classification
**Text Classification (텍스트 분류)**는 자연어처리(NLP)의 대표적인 응용 분야 중 하나로, 주어진 문장이나 문서가 어떤 카테고리에 속하는지를 자동으로 분류하는 작업.
예시.
“너무 맛있어요! 또 시킬게요” / 긍정 리뷰 (positive)
“서비스가 너무 느리고 불친절했어요” / 부정 리뷰 (negative)
“이번 주 삼성전자 주가 분석” / 경제 뉴스
“해리포터 시리즈가 너무 좋아요” / 영화 리뷰
쓰이는 곳.
감정 분석 (Sentiment Analysis) / 긍정,부정,중립이메일 분류 / 스팸, 일반
고객 문의 자동 태깅 / 결제, 배송, 교환, 반품
뉴스 분류 / 정치, 경제, 사회, 스포츠
리뷰 자동 분류 / 별점, 제품 카테고리 등 

## text clustering
비지도 학습(unsupervised learning) 방법의 하나로, 대량의 문서나 텍스트 자료를 내용이 비슷한 것끼리 자동으로 묶어주는 기법.

1.	용어 설명
o	텍스트(Text): 분석 대상이 되는 텍스트(문장·문서)
o	클러스터링(Clustering): 데이터를 유사도(similarity)가 높은 그룹(클러스터)으로 묶는 작업
2.	주요 단계
o	텍스트 전처리
	토큰화(tokenization), 불용어 제거(stop-word removal), 어간 추출(stemming)·표제어 추출(lemmatization)
o	벡터화(Vectorization)
	TF-IDF 벡터, 워드 임베딩(word embeddings) (Word2Vec, GloVe), 문장 임베딩(sentence embeddings) (Sentence-BERT) 등을 이용하여 각 문서를 수치 벡터로 변환
o	유사도 계산(Similarity Measurement)
	코사인 유사도(cosine similarity), 유클리드 거리(Euclidean distance) 등으로 문서 간 거리를 측정
o	클러스터링 알고리즘 적용
	K-평균 클러스터링(K-Means Clustering): 미리 정한 K개의 클러스터 중심을 반복 갱신
	계층적 클러스터링(Hierarchical Clustering): 상향식(agglomerative) 또는 하향식(divisive)으로 군집 트리 생성
	밀도 기반 클러스터링(DBSCAN): 밀도 밀집 지역을 탐지하여 클러스터 형성
o	결과 해석 및 활용
	클러스터별 대표 키워드 추출, 토픽 모델링(topic modeling)과 결합
	문서 분류, 추천 시스템, 고객 의견 분석, 뉴스·리뷰 그룹화 등에 활용
3.	활용 예시
o	뉴스 기사 분류: 유사 주제의 기사를 자동으로 묶어 주제별 페이지 구성
o	고객 리뷰 분석: 제품에 대한 긍정·부정 피드백을 주제별로 분류하여 대응 전략 수립
o	문서 검색·추천: 사용자가 본 문서와 유사한 문서를 추천
4.	장단점
o	장점: 사전 레이블 없이도 대규모 텍스트를 빠르게 그룹화 가능
o	단점: 클러스터 개수(K)나 밀도 매개변수(ε) 등을 사전에 정해야 하며, 품질 평가가 어려울 수 있음

요약하자면, 텍스트 클러스터링은 레이블(정답) 없이 문서들을 내용 유사도 기준으로 자동 분류하는 기술이며, NLP 분야에서 토픽 탐색이나 대용량 문서 관리에 널리 쓰임.

사전 레이블링(labeling) 작업 없이 자동으로 문서를 그룹화하기 때문에, **수작업 개입(manual intervention)**을 크게 줄여주고 이에 따라 **비용(cost)**과 **시간(time)**을 절감하는 데 주로 활용.

장점	단점
사전 레이블링이 불필요해 레이블링(Labeling) 비용 절감	군집 개수(K)나 밀도 매개변수(ε) 등의 하이퍼파라미터 튜닝 필요
비지도 학습(Unsupervised Learning) 기반으로 대량 텍스트 자동 분류 가능	클러스터 품질 평가(Evaluation)가 어려워 최적화가 복잡
토픽 탐색(Topic Discovery)을 자동화해 초기 데이터 분석 효율화	노이즈(noise)나 이상치(Outliers)에 민감
실시간 처리를 통한 비용 및 시간 절감	주제 간 중첩(overlap)된 문서 분류에 한계

## text embedding
**“문장 전체”나 “문서 전체”를 하나의 벡터로 표현하는 기술. 
📌 용어 병기:
영어 용어	한글 의미	설명
Token Embedding	토큰 임베딩	단어(또는 조각)를 숫자 벡터로 변환
Text Embedding	텍스트 임베딩	문장 전체나 문서를 하나의 벡터로 변환
Embedding Vector	임베딩 벡터	의미를 숫자로 표현한 고차원 벡터
LLM (Large Language Model)	대형 언어 모델	GPT, BERT, Claude 같은 AI 언어 모델
🔍 예시로 쉽게 이해하기
항목	예시	설명
Token Embedding	"AI" → [0.12, -0.33, ..., 1.02]	단어 하나를 벡터로
Text Embedding	"AI는 세상을 바꾸고 있다" → [0.98, -1.12, ..., 0.07]	문장 전체를 벡터로

이렇게 만든 텍스트 임베딩은:
•	문장 유사도 측정
•	검색(Query ↔ 문서 매칭)
•	클러스터링/분류 등에 사용

## TF-IDF
•	TF-IDF는 어떤 단어가 문서에서 얼마나 중요한지 숫자로 표현하는 방법.
•	TF (Term Frequency, 단어 빈도): 단어가 얼마나 자주 나왔는지
•	IDF (Inverse Document Frequency, 역문서 빈도): 드물게 나오는 단어일수록 점수를 높게 줌
👉 예를 들어 ‘컴퓨터’라는 단어가 한 문서에는 자주 나오고, 다른 문서에는 잘 안 나올 때, 이 문서에서는 '컴퓨터'가 중요한 단어라고 판단

🎯 핵심 요약
구분	일반 TF-IDF	c-TF-IDF
기준	문서(document)	클래스(class, 그룹)
쓰임	단어 중요도	클래스별 대표 단어 찾기
대표 사용처	검색엔진, 문서요약	토픽 모델링, 설명 가능한 AI

## token embedding
텍스트(데이터)를 단순 숫자화 한 것은 token ID 임. 단순한 인덱스라 생각하면 됨.
예:
"bank" → 5021

하지만 token embedding
Embedding(5021) → [0.21, -1.33, 0.77, ...]

이 벡터는 이미 학습된 의미 공간(semantic space) 안에 위치.
즉,
Token embedding은 "연산 가능하게 만든 것"을 넘어서
이미 의미를 학습한 벡터 표현입니다.

| 구분    | Token Embedding | Context Vector |
| ----- | --------------- | -------------- |
| 의미    | 기본 의미           | 문맥 반영 의미       |
| 고정/변화 | 고정              | 위치마다 달라짐       |
| 계산    | lookup          | attention 연산   |
| 상호작용  | 없음              | 모든 토큰과 상호작용    |
| 정보량   | 단어 자체           | 문장 전체 정보 반영    |

## tokenization
**Tokenization(토크나이징)**은 문장을 단어, 형태소, 서브워드 같은 '작은 조각들(Token)'로 나누는 과정.
🧠 왜 필요한가?
컴퓨터는 문장 전체를 그대로 이해할 수 없음. 그래서 먼저 문장을 잘게 쪼개서 **의미 단위의 조각(token)**으로 나눈 다음, 이 조각들을 숫자 벡터로 바꿔야 AI 모델이 처리할 수 있음.
🔍 예시로 이해해보기
✏️ 원문:
"나는 밥을 먹었다"
🔹 ① 띄어쓰기 기준 Tokenization (기본적인 방식):
["나는", "밥을", "먹었다"]
🔹 ② WordPiece 방식 (서브워드 기반, BERT 사용):
["나", "##는", "밥", "##을", "먹", "##었다"]
여기서 ##는 앞 단어와 붙어있던 조각이라는 뜻.
🔹 ③ 영어 예시:
문장: "playing"
→ Token: "play", "##ing" (WordPiece 방식)
📌 Tokenization의 목적
•	기계가 문장을 처리할 수 있도록 준비	•	
•	단어를 숫자로 바꾸기 위한 전처리 단계	•	
•	희귀 단어를 조각내서 처리 가능하게 함 (서브워드)	•	
•	더 일반화된 패턴을 학습하게 도와줌

## tokens
문장을 컴퓨터가 이해할 수 있게 잘게 나눈 단위. 사람은 "나는 학교에 간다."라는 문장을 쉽게 이해하지만, AI는 이 문장을 숫자 데이터로 바꿔야 이해할 수 있음

예시:
문장: 나는 학교에 간다.
→ 토큰화: ['나', '##는', '학교', '##에', '간', '##다', '.']

•	보통은 단어 또는 부분 단어(subword) 수준으로 쪼갬.
•	이걸 수행하는 도구가 바로 Tokenizer (토크나이저).

## tokenization method (2가지)
-	Byte Pair Encoding (BPE) (GPT 모델)
-	WordPiece (BERT 모델).

## tokenization scheme (4가지)
토크나이저(tokenizer)를 어떻게 구성할지는 여러 tokenization scheme(토크나이징 방식) 에 따라 달라지며, 보통 다음과 같은 4가지 대표적인 방식(Word vs Subword vs Charactoer vs Byte Tokens)으로 나뉨.

✅ 1. Word-level Tokenization (단어 단위 토크나이징)
•	예시 문장: "I like apples."
•	결과: ["I", "like", "apples", "."]
•	각 단어를 그대로 하나의 토큰으로 봄.
장점
•	직관적이고 해석이 쉬움
•	단어의 의미 단위가 잘 유지됨
단점
•	사전에 없는 단어(OOV, Out-of-Vocabulary)에 약함
•	복잡한 언어(예: 한국어, 독일어)엔 부적절

✅ 2. Subword-level Tokenization (서브워드 단위)
🌟 GPT, BERT 등이 사용하는 방식! (예: BPE, WordPiece, Unigram)
•	예시: "unhappiness" → ["un", "happi", "ness"]
•	단어를 더 작은 조각으로 나누되, 자주 등장하는 조각을 사전에 등록
장점
•	희귀 단어도 부분적으로 이해 가능
•	단어 조각들을 조합해 새로운 단어 처리 가능
•	말뭉치 압축 효율이 좋음
단점
•	해석이 복잡함 (예: ##ness, Ġthe 등)

✅ 3. Character-level Tokenization (문자 단위)
•	예시: "apple" → ["a", "p", "p", "l", "e"]
장점
•	모든 단어를 처리 가능 (OOV 없음)
•	단순한 구조
단점
•	문장이 길어짐 → 모델이 처리할 토큰 수 증가
•	의미 정보가 너무 적음 → 학습 효율 낮음

✅ 4. Byte-level Tokenization (바이트 단위)
🌟 GPT-2 이후 OpenAI 계열 모델들이 사용하는 Byte-level BPE
•	예시: "I ♥ U" → 실제 유니코드 바이트 단위로 쪼갬
•	유니코드로 된 문자열을 바이트로 쪼갬 (총 256개 토큰으로 표현 가능)
장점
•	어떤 언어나 이모지도 무조건 처리 가능
•	매우 일반적인 처리 방식
단점
•	사람이 보기엔 이해 어려움
•	토큰 수가 많아질 수 있음

📌 요약 표
방식	예시	장점	단점
Word	["I", "like", "you"]	쉬움, 의미 단위 보존	희귀 단어에 약함
Subword	["un", "believ", "able"]	희귀 단어 처리 가능	복잡함
Character	["a", "p", "p", "l", "e"]	완전 일반화	의미 적음
Byte	[b'I', b'\xe2\x99\xa5', b'U']	완전 범용 (이모지 포함)	사람이 보기 어려움

💬 그래서 요즘은?
✅ Subword (BPE, WordPiece, Unigram) 또는
✅ Byte-level BPE 방식이 대세.
(왜냐하면, 이 방식들이 희귀 단어 문제를 해결하고, 다양한 언어에 대응 가능하기 때문!)

## tokenizer
텍스트를 특정 모델이 이해할 수 있는 토큰 ID 시퀀스로 변환하는 입력 변환기(Input Converter)

2️⃣ tokenizer가 “하는 일” 정확히 나열

Tokenizer는 의미를 이해하지 않는다.
아래 작업만 한다.

정규화(Normalization, 정규화)

전처리(Pre-tokenization, 프리 토크나이징)

토큰 분해(Token segmentation, 토큰 분해)

어휘 사전 매핑(Vocabulary lookup, 보캐뷸러리 매핑)

특수 토큰 추가(Special tokens, 특수 토큰)

결과물:
[토큰1, 토큰2, 토큰3] → [101, 2357, 9082]

3️⃣ tokenizer는 “모델의 일부”인가?

이 질문에서 많은 사람이 꼬인다.

❌ 신경망 관점

tokenizer = 모델 ❌

학습되지 않음(대부분)

gradient 없음

⭕ 시스템 관점

tokenizer = 모델의 인터페이스

모델과 1:1로 묶여 있음

토크나이저가 다르면 같은 문장도 다른 토큰

👉 그래서 “모델을 바꾸면 토크나이저도 같이 바꾼다”

4️⃣ tokenizer ≠ embedding

이건 정말 많이 헷갈린다. 딱 잘라 정리한다.

구분			tokenizer			embedding layer
역할			텍스트 → ID			ID → 벡터
학습			❌					⭕
의미 이해		❌					⭕(학습 결과)
위치			모델 바깥				모델 내부

Tokenizer는 숫자로 바꾸는 도구,
Embedding은 숫자에 의미를 입히는 층이다.

5️⃣ tokenizer가 “모델마다 다른” 이유

Tokenizer는 이렇게 설계된다:

모델의 어휘 사전 크기

최대 토큰 길이

특수 토큰 규칙

예:

GPT 계열: Byte/BPE 계열

BERT 계열: WordPiece

Sentence Embedding 모델: SentencePiece / Unigram

그래서:

❗ 다른 tokenizer로 만든 토큰 ID는
❗ 절대 다른 모델에 넣으면 안 된다

6️⃣ tokenizer가 “구조를 만든다”? ❌

중요한 오해 제거.

tokenizer는 순서만 유지

관계, 문법, 구조 ❌

의미 추론 ❌

구조는 전부:

Transformer(Self-Attention) 가 만든다

7️⃣ 아주 직관적인 비유 (이건 꼭 잡아라)

tokenizer = 키보드 스캔코드

embedding = 글자의 의미

transformer = 문장을 이해하는 뇌

키보드는 생각 안 한다.
하지만 뇌가 입력을 해석할 수 있게 만들어 준다.

## tokenizer’s decode
✅ 개념 요약:
▶️ tokenizer의 decode란?
**디코드(decode)**는 **"토큰(token)을 사람이 읽을 수 있는 텍스트로 다시 바꾸는 작업"**.
즉,
•	**"토크나이저(tokenizer)"**는 문장을 쪼개서 → 숫자나 토큰 조각으로 바꾸고,
•	**"디코드(decode)"**는 그 숫자나 조각들을 → 다시 원래 문장으로 합치는 기능

🔡 예시:
1. 원래 문장:
"Hello, world!"
2. 토크나이저로 인코딩(encoding):
tokens = tokenizer.encode("Hello, world!")
# 예: [15496, 11, 995]

3. 토크나이저로 디코딩(decoding):
text = tokenizer.decode([15496, 11, 995])
# 결과: "Hello, world!"
즉, decode 함수는 토큰 리스트를 받아서 사람이 읽을 수 있는 문장으로 "복원"

## tokenizer’s encode
✅ 개념 요약:
▶️ tokenizer의 encode란?
**인코드(encode)**는 **"사람이 쓴 문장(텍스트)을 토큰(token)이라는 기계가 이해할 수 있는 숫자들의 목록으로 바꾸는 작업"**

🔡 예시:
1. 원래 문장:
"Hello, world!"

2. 인코딩 실행:
tokens = tokenizer.encode("Hello, world!")

3. 결과 (예시):
[15496, 11, 995]
이 숫자들은 각 단어 또는 부분 단어에 대응하는 고유한 숫자(=토큰 ID).
📌 encode의 종류 (옵션에 따라):
옵션	설명
tokenizer.encode(text)	토큰 ID 목록만 리턴함
tokenizer.encode_plus(text)	토큰 ID, 마스크, 타입 등 여러 정보를 딕셔너리로 반환
tokenizer(text)	Hugging Face에서 추천하는 기본 방식 (encode_plus처럼 작동)

token dimensions
•  각 토큰이 표현되는 벡터의 차원 수, 즉 하나의 토큰을 몇 개 숫자로 표현할지 결정.
•  예: 토큰 "밥" → [0.12, -0.45, ..., 0.07] (3072개 float 값으로 구성된 벡터)

## Topic Modeling
토픽 모델링은 대량의 문서나 텍스트 데이터에서 **숨겨진 주제(Topic)**들을 자동으로 찾아내는 비지도 학습(Unsupervised Learning) 기법. **토픽 모델링(topic modeling)**에서 가장 대표적인 알고리즘 중 하나가 Latent Dirichlet Allocation (LDA)로 문서 집합(corpus)에서 **숨겨진 주제(latent topics)**를 자동으로 발견해내는 확률 기반의 비지도 학습 알고리즘.

📊 예시
여러 문서에서 추출된 단어들이 아래와 같이 **3개의 군집(Cluster)**으로 나눌 수 있음:
1.	Topic 1 – Pets (반려동물)
o	단어 예: cat, dog, animal shelter
o	같은 주제를 가진 단어들이 보라색 영역으로 묶여 있음.
2.	Topic 2 – Food (음식)
o	단어 예: pasta, pizza, rice
o	녹색 영역에 클러스터링됨.
3.	Topic 3 – Sports (스포츠)
o	단어 예: soccer, basketball, game
o	파란색 영역으로 묶임.

🧠 작동 방식 요약
1.	수많은 문서를 입력으로 넣음
2.	단어들의 출현 패턴을 분석
3.	자주 함께 나타나는 단어들을 바탕으로 **잠재적인 주제(Topic)**들을 자동 분류
이때, 어떤 주제(Topic)는 **"단어들의 분포"**로 표현됨.
예:
•	Topic A: {pizza: 0.3, pasta: 0.2, rice: 0.1...} ← 음식 관련 단어들

🔧 대표 알고리즘
알고리즘	설명
LDA (Latent Dirichlet Allocation)	가장 널리 쓰이는 토픽 모델링 알고리즘
NMF (Non-negative Matrix Factorization)	행렬 분해 기반으로 동작
BERTopic	최근에는 BERT 기반 문장 임베딩을 활용하는 방법도 인기

📌 정리하면
항목	내용
목적	문서 집합에서 숨겨진 주제들을 자동으로 발견
입력	수많은 문서나 텍스트
출력	각 문서에 포함된 주제와 단어들의 확률 분포
활용 분야	뉴스 자동 분류, 여론 분석, 챗봇 문맥 파악, 검색 최적화 등

## training (학습)
학습은 모델이 **정답(target)**을 알고 있는 상태에서 예측이 얼마나 틀렸는지를 계산하고 가중치(weight)를 업데이트하는 과정.
•	손실 함수(loss function) 계산
•	역전파(backpropagation)
•	파라미터 업데이트

## transformer block
Transformer는 여러 층의 Transformer Block으로 구성되어 있음.
그 한 블록 내부에 꼭 들어있는 두 가지 핵심 구조가 바로:
1.	Self-Attention (자기 어텐션)
2.	Feedforward Neural Network (피드포워드 신경망)

🔹 구조 요약: 하나의 Transformer Block 구성
[입력 텐서]
    ↓
LayerNorm
    ↓
🔵 Multi-Head Self-Attention
    ↓
Residual Connection + LayerNorm
    ↓
🟢 Feedforward Neural Network (2-layer MLP)
    ↓
Residual Connection + LayerNorm
    ↓
[출력 텐서]

✅ 1. Self-Attention: 단어들 간 관계 파악
•	각 토큰이 문장 내 다른 토큰들과 어떤 관계가 있는지 계산
•	예: "The cat sat on the mat" → “sat”이 “cat”과 연결됨
•	Multi-Head Attention으로 확장되어 다양한 시각으로 관계를 동시에 평가함

✅ 2. Feedforward Neural Network (FFN): 의미 확장 및 추상화
•	각 토큰에 대해 개별적으로 처리되는 작은 MLP (2개의 Linear Layer + ReLU)
•	예:
x → Linear → ReLU → Linear → 출력

•	Self-Attention으로 상호작용을 한 뒤, FFN으로 의미를 더 정제하고 확장

🧠 시각적으로 보면:
🔁 여러 개의 Transformer Block (N개 쌓음)
 └── 블록 안에 매번 다음이 있음:
     ├─ 🟦 Self-Attention
     └─ 🟩 Feedforward NN

✅ 정리표
구성 요소	설명	Transformer Block 내 위치
Self-Attention	단어들 간의 관계 계산	블록의 앞부분
FFN (Feedforward NN)	의미 확장, 정제	블록의 뒷부분
LayerNorm & Residual	학습 안정성 & 정보 유지	각각 앞뒤에 붙음


## transformer decoder layers
모델 로딩 후 print(model)하면, (layers): ModuleList() 코드라인이 보이며, 이 안에 포함된 라인들 중 모델 decoder layer와 관련된 라인이 있음. 예를 들어, (0-31): 32 x Phi3DecoderLayer 이렇게 나왔다고 하면, 이 구조는 Transformer Decoder Layer를 32개 쌓은 것이란 소리. 입력이 Transformer 내부에서 같은 구조의 층을 32번 통과하면서 점점 더 깊이 있는 표현으로 변환된다는 것.
🧠 쉽게 비유하면
📚 Transformer 32층 건물
단어 벡터가 이 건물의 1층부터 32층까지 올라가면서 점점 문맥을 깊이 이해하게 됨
⦁	1층: "이건 '밥'이라는 단어야."
⦁	8층: "이 단어는 '먹다'와 같이 자주 나와."
⦁	20층: "이 문장은 사과의 의미가 아닌 음식 이야기 같아."
⦁	32층: "앞뒤 문맥과 합쳐보니 ‘밥’은 ‘식사’ 의미이고, 다음 단어는 ‘먹었다’가 될 확률이 높아."
🔁 왜 여러 층을 쌓는가?
Transformer가 단어 간의 문맥 관계를 깊이 이해하려면:
1~3층: 근접한 단어 관계 정도만 파악 (ex: “밥” ↔ “먹다”) 
10층 이상긴 거리 문맥, 문장 구조, 의미 흐름 등도 파악 가능 
32층전체 문단 수준의 관계까지 포착 가능 → LLM처럼 정교한 텍스트 생성 가능
🔧 내부 구성 요약 (한 블록 기준)
[입력] →
├─ Self-Attention: (qkv_proj, o_proj)
│    └ 문맥 간 유사도 계산
├─ MLP (feedforward): (gate_up_proj → down_proj)
│    └ 비선형 변환으로 표현력 강화
├─ LayerNorm + Residual
│    └ 안정적 학습 + 정보 손실 방지
↓
[출력] → 다음 블록으로 전달

이 과정을 32번 반복하겠다는 소리.
그리고 이 decoder layer 안에는 attention layer, MLP(=feedforward neural network= multilevel perceptron), layernorm가 있음.

underlying architecture
“속에 숨어 있는 핵심 구조”, 즉 **“기본 뼈대”**를 말함

🔷 한 줄 정의
Underlying architecture란
👉 겉으로는 보이지 않지만, 모델이나 시스템이 동작할 수 있게 해 주는 핵심 구조/설계

🔶 분야별 의미 예시
분야	의미
💻 인공지능 모델	BERT, GPT, Transformer 같은 기본 모델 구조
📱 소프트웨어 시스템	어떤 프레임워크, 언어, 설계 패턴이 기반인지
🖥️ 하드웨어	CPU vs GPU vs NPU 같은 계산 구조
🏗️ 웹 서비스	백엔드 구조 (예: REST API, Microservice 등)

🔍 AI 모델에서의 예시
모델 이름	Underlying Architecture
ChatGPT	GPT (Transformer 기반)
Claude	Transformer 기반 (세부 구조 비공개)
BERT-based classifier	BERT (Transformer encoder)
Stable Diffusion	U-Net + VAE + CLIP 등 복합 구조
🟡 즉, 모델의 겉모습은 다르지만, **속에 깔린 구조(설계 원리)**는 공통될 수 있음.

🧠 비유로 쉽게
예시	Underlying architecture은?
아파트	철근 구조물 + 수도/전기 배관 구조
스마트폰 앱	iOS/Android SDK, 백엔드 API 설계
AI 챗봇	Transformer 구조, 어텐션 메커니즘 등

✅ 요약
항목	설명
용어	Underlying Architecture
의미	시스템 또는 모델의 내부 핵심 구조, 설계 방식
사용 예	BERT 기반인지, GPT 구조인지, CNN인지 등
유사 표현	내부 구조, 백본(backbone), 기반 설계, 핵심 프레임워크

## tree-of-thought
하나의 추론을 직선(linear)으로 밀지 않고, 여러 사고 분기(thought branch)를 트리(tree)처럼 펼쳐 탐색·평가·선택하면서 정답에 접근하는 방식

핵심 키워드:
⦁	분기(branching)
⦁	탐색(search)
⦁	평가(evaluation)
⦁	선택(pruning)

## tt-forge
PyTorch, ONNX 모델을 Tensix NPU에서 돌리려면, 그 모델을 MLIR 기반 중간 단계 코드로 바꿔서 최적화해야 하고, 그걸 담당하는 게 Tenstorrent의 **MLIR 기반 컴파일러(TT-Forge)**. Tenstorrent의 전체 컴파일러 시스템을 아우르는 상위 개념. 그 안에 하위 구성 요소로 tt-forge-fe, tt-forge-be, 그리고 tt-torch 등이 나뉘어 있음.

✅ 정리 요약
질문	답변
MLIR이란?	AI 모델의 실행 단계를 여러 층의 중간 표현으로 쪼개서 최적화 가능한 구조
MLIR-based compiler?	이 구조를 사용하는 컴파일러, 예: TT-Forge
왜 쓰나요?	복잡한 AI 연산을 하드웨어 구조에 맞게 단계별로 최적화하기 위해
Tenstorrent에서는?	PyTorch/ONNX 모델 → MLIR → TT-Dialect → Tensix NPU 코드로 바뀜

이름	뜻	역할
tt-forge-fe	Front-End	PyTorch/ONNX 등 모델을 받아서 처리하는 부분 (앞단)
tt-forge-be	Back-End	Tile 배치, DMA, L1 메모리 등 하드웨어에 최적화된 실행 코드 생성 (뒷단)
tt-forge	전체 컴파일러	위 두 개를 포함한 전체 시스템 또는 패키지 이름

🧱 구성 요소 설명 (TT-Forge 하위 구성)
구성 요소	설명	비유
tt-torch	PyTorch 인터페이스 부분. PyTorch 모델을 FX Graph로 변환하고 처리하는 전처리 단계	입구 (고객이 레시피를 들고 오는 곳)
tt-forge-fe (Front-End)	FX Graph에서 Torch-MLIR로 변환하고, 중간 표현(IR)을 StableHLO → TTIR로 넘기는 부분	주방 앞쪽 (레시피 보고 요리 순서 짜는 사람)
tt-forge-be (Back-End)	TTIR을 TTNN 및 flatbuffer로 변환해서 실행 가능한 코드로 만드는 엔진	주방 뒤쪽 (요리사들이 실제로 음식 만드는 단계)
tt-mlir	전체를 연결하는 MLIR 기반 변환 도구. 위의 모든 구성요소를 이어주는 백본	주방 도구 모음 (칼, 냄비, 조리대 등 도구 통합 플랫폼)


🔁 비유
단계	설명	해당 Forge 컴포넌트
1단계	고객이 모델을 가져옴 (PyTorch 등)	tt-forge-fe
2단계	모델을 내부 표현으로 변환 (MLIR IR)	tt-forge-fe
3단계	어떤 연산을 어떤 tile에서 할지 계산	tt-forge-be
4단계	L1 메모리, DMA, tile instruction 생성	tt-forge-be
전체 조립 라인	위 모든 걸 포함한 전체 빌드 시스템	tt-forge

✅ 정리하면
tt-forge는 tt-forge-fe + tt-forge-be 등 모든 구성요소를 포함한 전체 이름

## tt-forge-fe
다양한 딥러닝 프레임워크(PyTorch 등)로 만든 모델을 받아들여서, 하드웨어가 이해할 수 있는 “계산 그래프 형태”로 바꿔주는 컴파일러. 이때 TVM 기술을 바탕으로 만들어졌기 때문에, 고성능 최적화도 가능

🔄 전체 흐름도
단계	내용	예시
1. PyTorch 모델 준비	사람이 만든 모델	torch.nn.Transformer(...)
2. tt-forge-fe로 입력	모델을 읽어서 내부 IR(graph)로 바꿈	연산자들만 남긴 계산 그래프 형태
3. 최적화 & 변형	불필요한 연산 제거, 연산 순서 정렬 등	연산 합치기(fusion), 타일 배치 쉽게
4. tt-forge-be 전달	하드웨어 최적화 시작	L1 메모리, tile 배치 등

💡 용어 정리
용어	설명
TVM	다양한 하드웨어를 위한 AI 컴파일러 프레임워크
tt-tvm	Tenstorrent가 TVM을 내부용으로 변형해서 쓴 것
Computational Graph	AI 모델의 연산 흐름 (예: A → B → C)
IR (Intermediate Representation)	중간 코드 표현. 모델을 하드웨어에 최적화하기 전에 변환하는 단계
Ingestion	외부에서 모델을 가져와서 내부에서 쓸 수 있게 변환하는 과정

🎯 왜 중요하냐?
•	당신이 PyTorch나 Hugging Face에서 모델을 받았을 때
👉 그걸 바로 NPU에서 실행할 수는 없음.
•	그래서 tt-forge-fe가 중간에서 모델을 이해하고, 변환하고, 최적화해줘야
👉 뒷단(tt-forge-be, tt-metal)이 NPU 실행 코드를 만들 수 있음.

✅ 요약
질문	답변
tt-forge-fe가 뭐 하는 건가요?	다양한 프레임워크 모델을 받아서, Tenstorrent가 처리할 수 있게 그래프 형태로 변환하고 최적화하는 컴파일러입니다.
왜 TVM 기반인가요?	TVM은 이미 다양한 하드웨어 타겟을 지원하고 최적화 기술이 풍부하니까, 그걸 응용해서 만든 거예요.
어떤 프레임워크를 지원하나요?	PyTorch, ONNX, TensorFlow, PaddlePaddle 등

✅ tt-forge-fe와 tt-torch의 차이점
항목	tt-forge-fe	tt-torch
목적	다양한 프레임워크 모델을 지원하는 범용 front-end	PyTorch 2.x 전용 front-end
기반 기술	Apache TVM 기반	PyTorch 2.x + torch-mlir 기반
지원 프레임워크	PyTorch, ONNX, TensorFlow, Paddle 등	PyTorch 2.x 전용
출력 포맷	Tenstorrent의 내부 IR 또는 SHLO	StableHLO (SHLO)
통합도	tt-tvm과 tightly coupled	torch.compile() 기반 native PyTorch 통합
사용 예	레거시 모델, 다양한 포맷, 연구용 코드 지원	최신 PyTorch 2.x 환경에서 빠르게 사용

🔁 비유로 설명하면
•	tt-forge-fe는 멀티포맷 어댑터.
→ 다양한 모델 형식을 변환할 수 있는 범용 전처리기.
•	tt-torch는 PyTorch 전용 초고속 전용선.
→ PyTorch 2.x 모델만을 위한 최신 컴파일러 프론트.

🧠 조금 더 기술적으로 보면…
비교 항목	설명
tt-forge-fe	- tt-tvm을 기반으로 다양한 프레임워크 모델을 변환 가능- PyTorch, TensorFlow, ONNX 등 지원- 모델을 Tenstorrent 내부 IR로 바꿔 tt-forge-be로 넘김
tt-torch	- PyTorch 2.x의 torch.compile() 기능을 적극 활용- torch-mlir 기반으로 SHLO 생성- 곧바로 tt-mlir에 넘겨서 실행 가능

🔍 예를 들어 볼게요
•	PyTorch 1.x로 만든 ONNX 모델 → tt-forge-fe로 ingestion
•	PyTorch 2.x로 만든 최신 모델 → tt-torch로 ingestion (빠르고 효율적)

🎯 어떤 걸 쓰는 게 좋을까?
상황	추천 툴
다양한 프레임워크 모델 (ONNX, TF 등)을 NPU에서 테스트하고 싶다	✅ tt-forge-fe
최신 PyTorch 2.x 기반 모델을 고성능으로 빠르게 돌리고 싶다	✅ tt-torch
모델이 torch.compile()을 지원하지 않는다	→ tt-forge-fe
내부에서 MLIR 기반 직접 튜닝을 원한다	→ 둘 다 가능 (SHLO → tt-mlir)

✅ 정리
질문	답변
tt-forge-fe와 tt-torch는 같은 역할인가요?	역할은 비슷하지만, 기술적 기반과 최적화 방식이 다릅니다.
어떤 걸 쓰는 게 좋을까요?	최신 PyTorch 2.x → tt-torch / 다양한 모델 포맷 → tt-forge-fe

## tt-installer
Tenstorrent 기본 software stack을 하나의 명령어(tt-installer)로 편하게 설치할 수 있게 해 주는 커맨드

/bin/bash -c "$(curl -fsSL https://github.com/tenstorrent/tt-installer/releases/latest/download/install.sh)" # 

구성	설명
/bin/bash -c	bash 쉘로 뒤에 오는 문자열(command)을 실행해줘
$(...)	괄호 안 명령어를 먼저 실행해서 그 결과를 가져와
curl -fsSL ...	인터넷 주소에서 파일 내용을 가져와 (sh 스크립트 다운로드)
https://github.com/.../install.sh	Tenstorrent가 공식 배포한 설치 스크립트 위치
* curl로 설치 스크립트를 가져와서 → bash로 실행

1. 스크립트의 주요 기능 요약 (2025년 7월)
Tenstorrent의 설치 스크립트는 한 번의 명령으로 다음 작업들을 자동으로 수행:
•	시스템 의존성 패키지 설치 (e.g. DKMS, podman, 컴파일 도구)
•	Python 환경 구성 및 필수 Python 패키지 설치
•	커널 모듈 (TT-KMD) 설치 및 시스템 등록
•	펌웨어 관리 도구 (tt-flash) 설치 및 장치 펌웨어 업데이트
•	시스템 관리 인터페이스 (tt-smi) 설치
•	HugePages 구성 (고성능 메모리 접근을 위해 필수)
•	Podman 기반 컨테이너로 tt-metalium 실행 환경 구성 (tt-metalium 실행 커맨드 설정)

2. 각 단계별 동작 흐름 🧩
✅ 시스템 패키지 설치
OS에 맞는 패키지 매니저를 통해 dkms, podman, 컴파일 도구, hugepages 설정 도구 등을 설치.
✅ Python 도구 설치
•	tt-flash (펌웨어 업데이트 유틸리티)
•	tt-smi (시스템 관리 인터페이스)
위 두 가지는 Python 패키지로 설치.
✅ 커널 모듈 설치 (TT KMD)
DKMS를 이용해 tt-kmd를 자동으로 추가하고 설치.
✅ Firmware 업데이트 실행
tt-flash를 사용해 장치의 적합한 펌웨어 버전을 설치할 수 있도록 구성.
✅ HugePages 설정
1GB 단위로 메모리를 구성하여 NPU가 빠르게 메모리에 접근할 수 있도록 함.
✅ tt-metalium 설치 및 설정 (컨테이너 기반)
•	tt-metalium은 Podman 컨테이너로 설치되어 실행.
•	tt-metalium 명령어 하나로 컨테이너가 실행되며, 기본적으로 사용자의 홈 디렉토리를 마운트해 줌.
•	또한 tt-metalium "<command>" 형식으로 내부 명령을 실행할 수 있게 설정.

3. 고급 사용자 옵션 및 커스터마이징
고급 사용자라면 스크립트 실행 시 다양한 옵션을 지정할 수 있음:
•	--mode-non-interactive: 사용자 입력 없이 자동 실행
•	--no-install-kmd 또는 --no-install-hugepages: 특정 구성 요소 설치 건너뛰기
•	--kmd-version=... / --fw-version=...: 설치할 드라이버나 펌웨어 버전 지정 가능

./install.sh –help  # 자세한 옵션 보기

## tt-metal
✅ tt-metal이 뭐야?
💡 **Tenstorrent NPU 하드웨어를 제일 낮은 수준에서 직접 제어할 수 있게 해 주는 소프트웨어 도구(라이브러리)**. 2025년 7월 현재 0.61.0 버전 출시되어 있음.
즉, NPU의 뇌를 직접 만지는 도구

🎯 예를 들어볼게요
보통 PyTorch 같은 도구는 우리가 “뭘 할지”만 코드로 쓰면,
내부에서 알아서 GPU나 NPU에서 실행되도록 처리해 줌.
하지만 tt-metal은:
•	어떤 데이터를
•	어떤 순서로
•	어떤 tile (계산 코어)에
•	어떻게 복사해서
•	어떤 연산을 하고
•	다시 어디로 보낼지
까지 전부! 직접 지정

📦 쉽게 비유하면?
PyTorch / tt-torch	tt-metal
자동차 내비게이션	직접 운전 + 길 만들기까지
“서울에서 부산 가자”	“좌회전, 우회전, 기름 넣고, 엔진 돌리는 법도 내가 설계”
고수준 API	하드웨어 제어용 저수준 API

🔧 언제 tt-metal을 써야 할까?
상황	tt-metal 필요 여부	이유
✅ 새로운 연산(커널)을 직접 구현	필요함	기존에 없는 연산은 직접 만들어야 함
✅ tile-to-tile 연산을 수동 최적화	필요함	DMA 경로, L1 메모리 재사용 등
✅ NPU 성능을 끝까지 뽑고 싶다	필요함	자동 최적화로 안 되는 부분 직접 다룸
❌ 단순히 PyTorch 모델을 올리고 추론만	필요 없음	tt-torch, tt-inference-server로 충분

📘 예시로 이해하기
예1) PyTorch 모델 올리기만 할 때
# 사용 도구: tt-torch
model = torch.compile(my_model)
# 알아서 SHLO → tt-mlir → NPU 실행

예2) Tile 간 연산이 병목이라 직접 DMA 경로 조정하고 싶음
// 사용 도구: tt-metal
tt::CreateCircularBuffer(...);
tt::ConfigureDMAPipeline(...);
tt::LaunchKernel("my_custom_relu");

🧠 tt-metal로 할 수 있는 일 요약
기능	설명
Tile 연산 배치	어떤 코어에서 어떤 연산을 할지 직접 정함
DMA 경로 설정	Tile 간 메모리 복사 경로 수동 설정
L1 메모리 관리	텐서를 어디에 저장할지 명시
Custom Kernel 실행	직접 만든 연산 커널 실행 (ex. VecAdd, MatMul 등)
Profiling & Debug	성능 분석, 메모리 사용량 추적 등 가능

✅ 요약 정리
질문	답변
tt-metal이 뭐야?	Tenstorrent NPU를 하드웨어 수준에서 직접 제어하는 도구
언제 써야 해?	성능 최적화, 새로운 연산 만들기, 타일 경로 수동 조절이 필요할 때
추론 서비스만 할 때 써야 해?	❌ 전혀 필요 없음. tt-torch로 충분.
누가 써야 해?	NPU 커널 개발자, 성능 최적화 전문가, 딥러닝 시스템 엔지니어 등

## tt-metalium
✅ tt-metalium이란?
Tenstorrent NPU용 개발 환경을 미리 만들어 놓은 "컨테이너 실행 도구".
즉, tt-metal이라는 복잡한 프레임워크를 직접 설치하지 않아도
컨테이너로 포장된 환경을 바로 실행해서 쓸 수 있도록 만든 도구

🍱 예시 비유: 도시락과 주방
비유	의미
tt-metal	AI 요리를 만드는 레시피와 재료들 (프로그래머가 직접 요리함)
tt-metalium	레시피와 재료가 다 들어있는 전자레인지용 도시락 (바로 실행 가능)

즉, tt-metalium은 tt-metal을 포함한 실행환경.
두 개는 같은 것은 아니지만, tt-metal을 쉽게 쓰기 위한 포장된 형태라고 보면 됨.

✅ 1. 설치 먼저 하기 (한 번만)
/bin/bash -c "$(curl -fsSL https://github.com/tenstorrent/tt-installer/releases/latest/download/install.sh)"

✅ 2. 실행하기
tt-metalium

이 명령어 하나로 Podman 컨테이너가 실행되며,
그 안에 tt-metal, tt-buda, tt-mlir 등 모든 환경이 준비돼 있음.

✅ 3. 파일 복사 없이 내 코드 사용 가능
tt-metalium은 자동으로 당신의 홈 디렉토리를 컨테이너 안에 마운트해 줌.
그래서 /home/netsplus에 있는 코드나 모델 파일을 컨테이너 안에서도 바로 쓸 수 있음.

✅ 4. 컨테이너 안에서 명령어 실행
tt-metalium "python my_model.py"

위처럼 tt-metalium 뒤에 명령어를 쓰면,
컨테이너 안에서 Python 실행되며 NPU로 추론 수행이 가능

✅ tt-metalium과 tt-metal 관계 정리
항목	tt-metal	tt-metalium
정체	Tenstorrent의 AI NPU용 개발 프레임워크	tt-metal 포함한 실행 컨테이너
설치 방식	직접 GitHub에서 clone → build	tt-installer로 자동 설치
사용 난이도	개발자 수준 (코드 수정, 빌드 등 필요)	사용 편의형 (명령어만 있으면 실행됨)
실행 대상	직접 설치한 Ubuntu 환경	컨테이너 기반의 격리된 개발환경
둘의 관계	핵심 프레임워크	프레임워크를 싸놓은 도시락(컨테이너)

✅ 결론
tt-metalium은 tt-metal을 더 쉽게 쓰기 위해 만들어진 컨테이너 실행 환경 도구.
같은 것은 아니지만, tt-metal을 내부에 포함하고 있기 때문에 함께 움직이는 관계.


## tt-mlir
tt-mlir는 tt-forge의 핵심 구성 요소 중 하나

✅ 요약 먼저
용어	역할	관계
tt-forge	Tenstorrent 전체 컴파일러 스택	전체 시스템
tt-mlir	MLIR 기반 중간 컴파일러 계층	tt-forge 안의 핵심 컴포넌트 중 하나

🔁 구조적으로 보면 이렇게 이해하면 됨.
[PyTorch 모델]  
   ↓
tt-torch (tt-forge의 프론트엔드)
   ↓
tt-mlir (중간 IR 최적화 및 하드웨어 매핑)
   ↓
tt-metal (로우레벨 코드, NPU에 실제로 실행되는 코드)
   ↓
[NPU에서 실행]

🧠 각 컴포넌트 역할 다시 정리
구성 요소	설명
tt-torch	PyTorch 모델을 받아서 stableHLO로 변환
tt-mlir	stableHLO를 받아서 MLIR 기반으로 tile 배치, DMA, 메모리 최적화 수행
tt-metal	tt-mlir이 만든 실행 정보를 바탕으로 실제 NPU에 실행될 코드 생성
tt-forge	위 모든 과정을 아우르는 전체 컴파일러 이름

🔧 왜 tt-mlir이 중요할까?
•	MLIR은 다양한 하드웨어와 연산에 맞게 중간 표현(IR)을 구성할 수 있는 프레임워크.
•	Tenstorrent는 이걸 활용해 tile 단위 연산, L1 메모리, DMA 경로 등을 자동화 또는 수동 제어할 수 있게 만들었음.
•	즉, tt-mlir이 바로 “AI 모델 → 타일 실행 코드”로 바꿔주는 뇌 역할

🎯 비유하자면
비유 대상	의미
tt-forge	자동차 전체
tt-mlir	엔진
tt-torch	연료 주입기 (PyTorch 모델 받아오는 부분)
tt-metal	바퀴와 엔진을 땅에 연결하는 기계 부품 (실행기)

✅ 결론
tt-mlir는 tt-forge의 일부이며,
모델을 Tenstorrent NPU에서 실행할 수 있게 “타일 수준으로 컴파일”하는 가장 핵심적인 컴포넌트

## tt-nn
tt-torch와 같은 역할.
✅ 결론 요약
항목	추천 여부	이유
tt-nn	❌ 장기 사용 비추천	내부 개발용 / 유지 최소화 중
tt-torch	✅ 강력 추천	공식 최신 경로 (PyTorch 2.x, torch.compile 기반)
tt-xla	✅ (JAX 사용자 한정)	JAX → NPU 실행용 최신 경로

📌 왜 tt-nn이 deprecated된다고 하나?
이유	설명
범용성 부족	PyTorch와 다르게 생긴 고유 API라 학습 곡선이 있음
생태계 연동 약함	Hugging Face 등과 직접 연결하기 어려움
유지 비용 ↑	NPU 하드웨어와 직접 맞물리기 때문에 내부 로직이 복잡
대체 기술 존재	tt-torch, tt-xla, tt-forge-fe가 더 범용적이면서 강력함

🧠 지금은 어떤 걸 써야 하나요?
1. ✅ PyTorch 모델을 쓰고 있다면
→ tt-torch 사용이 가장 최신이고 강력한 방법

⚠️ tt-nn은 언제 쓰나?
•	내부 개발자들이 디버깅하거나
•	legacy 테스트 환경에서는 아직 사용합니다.
•	그러나 새로운 모델은 더 이상 tt-nn으로 작성하지 않음.

🧭 선택 가이드 요약
사용 환경	추천 툴	이유
PyTorch 2.x 기반 모델	✅ tt-torch	최신 공식 방식, 생태계 연동 우수
JAX 모델	✅ tt-xla	jit 기반 실행 지원
예전 테스트 코드 유지보수	⚠️ tt-nn	레거시 코드 지원용
PyBUDA	별도 경로	중간 API 레이어 (특정 연구/커스터마이징에 적합)

🙋🏻 요약
•	tt-nn은 더 이상 추천되지 않는 경량 프레임워크
•	tt-torch는 공식 추천 루트
•	당신이 PyTorch 모델을 쓰고 있다면 → 무조건 tt-torch 쓰는 게 정답


## tt-torch
최신 PyTorch 2.x 모델을 받아서, Tenstorrent 하드웨어가 이해할 수 있는 방식(SHLO)으로 변환해주는 도구. 이걸 통해서 NPU에서 실행될 준비를 마친 상태로 넘겨 줌. Python의 가상환경(venv) 안에서 작동하도록 만들어졌고, 필요한 모든 프로그램/라이브러리를 그 안에서 관리함.

🔁 전체 흐름 구조
[PyTorch 2.x 모델]        # 레고 설계도(PyTorch 모델)
    ↓
torch.compile()로 FX Graph 생성
    ↓
FX Graph 최적화 (상수 계산, 불필요한 코드 제거)
    ↓
torch-mlir → stableHLO 변환 (중간 표현)
    ↓
tt-mlir로 TTIR → TTNN → flatbuffer 변환    # 공장 기계(NPU)가 알아듣게 다시 그려서
    ↓
Executor 생성 (flatbuffer 기반)   # 레고 조립 로봇이 자동으로 조립하게
    ↓
사용자가 입력 주면:    #  필요할 때마다 레고 부품을 주면 바로 조립해주는 구조
  → NPU로 전송
  → flatbuffer 실행

📘 용어 하나씩 쉽게 설명
용어	뜻	쉬운 비유
MLIR-native	MLIR로 작성된 도구	중간 코드 변환에 특화된 설계
PyTorch 2.x	PyTorch 최신 버전 (컴파일 기능 포함)	예: torch.compile() 기능 사용 가능
torch-mlir	PyTorch → MLIR로 바꿔주는 오픈소스	다리 역할
stableHLO (SHLO)	안정적인 MLIR 기반 연산 표현 방식	Tenstorrent가 주로 받는 포맷
tt-mlir	Tenstorrent의 MLIR 시스템	SHLO 받아서 tile 배치, DMA까지 처리

🔍 예를 들어보면
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(16, 16)

    def forward(self, x):
        return torch.relu(self.linear(x))
→ 이 모델을 그냥은 NPU에서 못 돌려요!
1.	tt-torch가 이걸 PyTorch 2.x 방식으로 컴파일함 (torch.compile)
2.	내부에서 SHLO라는 포맷으로 바꿈 (이건 tt-mlir이 좋아하는 방식)
3.	tt-mlir이 이걸 받아서 타일 배치, 메모리 전송 등 처리
4.	최종적으로 NPU에서 실행 가능

🎯 요약 정리
질문	답변
tt-torch는 뭐하는 거야?	PyTorch 2.x 모델을 Tenstorrent 하드웨어용으로 바꿔주는 도구
SHLO는 뭐야?	MLIR의 한 형태. 모델을 연산 단위로 안정적으로 표현하는 방식
왜 중요해?	PyTorch 모델을 빠르고 쉽게 NPU에 올릴 수 있는 가장 현대적인 방법이기 때문
tt-torch가 없으면?	직접 ONNX 변환하거나 MLIR 수준에서 복잡하게 손으로 다뤄야 함



🔍 지원하는 모델 (2025.07)
Model Name	Variant	Pytest Command
Albert	Masked LM Base	tests/models/albert/test_albert_masked_lm.py::test_albert_masked_lm[single_device-full-albert/albert-base-v2-eval]
	Masked LM Large	tests/models/albert/test_albert_masked_lm.py::test_albert_masked_lm[single_device-full-albert/albert-large-v2-eval]
	Masked LM XLarge	tests/models/albert/test_albert_masked_lm.py::test_albert_masked_lm[single_device-full-albert/albert-xlarge-v2-eval]
	Masked LM XXLarge	tests/models/albert/test_albert_masked_lm.py::test_albert_masked_lm[single_device-full-albert/albert-xxlarge-v2-eval]
	Sequence Classification Base	tests/models/albert/test_albert_sequence_classification.py::test_albert_sequence_classification[single_device-full-textattack/albert-base-v2-imdb-eval]
	Token Classification Base	tests/models/albert/test_albert_token_classification.py::test_albert_token_classification[single_device-full-albert/albert-base-v2-eval]
Autoencoder	(linear)	tests/models/autoencoder_linear/test_autoencoder_linear.py::test_autoencoder_linear[full-eval]
DistilBert	base uncased	tests/models/distilbert/test_distilbert.py::test_distilbert[full-distilbert-base-uncased-eval]
Llama	3B	tests/models/llama/test_llama_3b.py::test_llama_3b[full-meta-llama/Llama-3.2-3B-eval]
MLPMixer		tests/models/mlpmixer/test_mlpmixer.py::test_mlpmixer[full-eval]
MNist		pytest -svv tests/models/mnist/test_mnist.py::test_mnist_train[single_device-full-eval]
MobileNet V2		tests/models/MobileNetV2/test_MobileNetV2.py::test_MobileNetV2[full-eval]
	TorchVision	tests/models/torchvision/test_torchvision_image_classification.py::test_torchvision_image_classification[single_device-full-mobilenet_v2]
MobileNet V3	Small TorchVision	tests/models/torchvision/test_torchvision_image_classification.py::test_torchvision_image_classification[single_device-full-mobilenet_v3_small]
	Large TorchVision	tests/models/torchvision/test_torchvision_image_classification.py::test_torchvision_image_classification[single_device-full-mobilenet_v3_large]
OpenPose		tests/models/openpose/test_openpose_v2.py::test_openpose_v2[full-eval]
Preciever_IO		tests/models/perceiver_io/test_perceiver_io.py::test_perceiver_io[full-eval]
ResNet	18	tests/models/resnet/test_resnet.py::test_resnet[single_device-full-eval]
	18 TorchVision	tests/models/torchvision/test_torchvision_image_classification.py::test_torchvision_image_classification[single_device-full-resnet18]
	34 TorchVision	tests/models/torchvision/test_torchvision_image_classification.py::test_torchvision_image_classification[single_device-full-resnet34]
	50	tests/models/resnet50/test_resnet50.py::test_resnet[single_device-full-eval]
	50 TorchVision	tests/models/torchvision/test_torchvision_image_classification.py::test_torchvision_image_classification[single_device-full-resnet50]
	101 TorchVision	tests/models/torchvision/test_torchvision_image_classification.py::test_torchvision_image_classification[single_device-full-resnet101]
	152 TorchVision	tests/models/torchvision/test_torchvision_image_classification.py::test_torchvision_image_classification[single_device-full-resnet152]
Wide ResNet	50	tests/models/torchvision/test_torchvision_image_classification.py::test_torchvision_image_classification[single_device-full-wide_resnet50_2]
	101	tests/models/torchvision/test_torchvision_image_classification.py::test_torchvision_image_classification[single_device-full-wide_resnet101_2]
ResNext	50	tests/models/torchvision/test_torchvision_image_classification.py::test_torchvision_image_classification[single_device-full-resnext50_32x4d]
	101_32x8d	tests/models/torchvision/test_torchvision_image_classification.py::test_torchvision_image_classification[single_device-full-resnext101_32x8d]
	101_64x4d	tests/models/torchvision/test_torchvision_image_classification.py::test_torchvision_image_classification[single_device-full-resnext101_64x4d]
Regnet	y 400	tests/models/torchvision/test_torchvision_image_classification.py::test_torchvision_image_classification[single_device-full-regnet_y_400mf]
	y 800	tests/models/torchvision/test_torchvision_image_classification.py::test_torchvision_image_classification[single_device-full-regnet_y_800mf]
	y 1 6	tests/models/torchvision/test_torchvision_image_classification.py::test_torchvision_image_classification[single_device-full-regnet_y_1_6gf]
	y 3 2	tests/models/torchvision/test_torchvision_image_classification.py::test_torchvision_image_classification[single_device-full-regnet_y_3_2gf]
	y 8	tests/models/torchvision/test_torchvision_image_classification.py::test_torchvision_image_classification[single_device-full-regnet_y_8gf]
	y 16	tests/models/torchvision/test_torchvision_image_classification.py::test_torchvision_image_classification[single_device-full-regnet_y_16gf]
	y 32	tests/models/torchvision/test_torchvision_image_classification.py::test_torchvision_image_classification[single_device-full-regnet_y_32gf]
	x 400	tests/models/torchvision/test_torchvision_image_classification.py::test_torchvision_image_classification[single_device-full-regnet_x_400mf]
	x 800	tests/models/torchvision/test_torchvision_image_classification.py::test_torchvision_image_classification[single_device-full-regnet_x_800mf]
	x 1 6	tests/models/torchvision/test_torchvision_image_classification.py::test_torchvision_image_classification[single_device-full-regnet_x_1_6gf]
	x 3 2	tests/models/torchvision/test_torchvision_image_classification.py::test_torchvision_image_classification[single_device-full-regnet_x_3_2gf]
	x 8	tests/models/torchvision/test_torchvision_image_classification.py::test_torchvision_image_classification[single_device-full-regnet_x_8gf]
	x 16	tests/models/torchvision/test_torchvision_image_classification.py::test_torchvision_image_classification[single_device-full-regnet_x_16gf]
	x 32	tests/models/torchvision/test_torchvision_image_classification.py::test_torchvision_image_classification[single_device-full-regnet_x_32gf]
Yolo	V3	tests/models/yolov3/test_yolov3.py::test_yolov3[full-eval]
-> 위 모델은 https://github.com/tenstorrent/tt-torch/의 tests 디렉토레어서에서 확인 가능

## tt-xla
✅ 한 줄로 요약하면?
💡 JAX로 만든 AI 모델을 Tenstorrent NPU에서 실행할 수 있게 해주는 다리 역할 도구.  tt-torch가 PyTorch AI 프레임워크 대상이라고 하다면, tt-xla는 JAX AI 프레임워크 대상.

🧠 용어부터 쉽게 정리
용어	쉽게 설명
JAX	구글이 만든 수학 계산/AI 모델 프레임워크 (딥러닝용 파이썬 도구)
PJRT	JAX가 다른 하드웨어와 연결할 수 있게 해주는 표준 인터페이스 (API)
jit compile	"Just-in-time compile"의 약자, 실행 직전에 모델을 빠르게 컴파일하는 기능
StableHLO (SHLO)	MLIR에서 쓰이는 중간 코드(IR). AI 연산을 표현하는 언어
tt-mlir	Tenstorrent가 만든 MLIR 기반 컴파일러. NPU에서 실행되게 바꿔줌
Tenstorrent hardware	실제 AI 연산을 수행하는 Tenstorrent의 NPU 하드웨어

🔁 전체 흐름을 쉽게 그리면:
[JAX 모델]
   ↓  (jit compile)
[PJRT 인터페이스 사용]
   ↓
tt-xla
   ↓
StableHLO (SHLO) 생성
   ↓
tt-mlir
   ↓
타일 배치, DMA 설정 등 최적화
   ↓
Tenstorrent NPU에서 실행

🎯 요약 정리
질문	답변
tt-xla는 뭐야?	JAX 모델을 Tenstorrent NPU에서 실행할 수 있게 해주는 도구예요.
어떻게 작동해?	PJRT라는 인터페이스를 통해 JAX 모델 → SHLO(IR)로 변환하고 → tt-mlir에 전달해서 하드웨어에서 실행되게 만듦
왜 필요해?	PyTorch 외에도 JAX 같은 프레임워크도 지원하려면 꼭 필요해요
지금은 JAX만 지원해?	네. 현재는 JAX 위주지만, 앞으로 다른 프레임워크도 지원할 수 있게 만들고 있어요

📌 비유로 말하면
구성 요소	비유
JAX	영어로 된 요리법 (AI 모델)
PJRT	번역기
tt-xla	번역한 요리법을 Tenstorrent가 이해할 수 있는 문서로 바꿔주는 사람
tt-mlir	그 문서를 보고 실제로 요리(연산)하는 사람
NPU	요리를 실제로 수행하는 로봇 요리사
 
## ULFM
ULFM = User-Level Fault Mitigation
•	MPI의 확장 버전.
•	노드(서버) 하나가 죽더라도 전체 작업이 멈추지 않도록 설계된 내결함성(fault-tolerant) 기능을 추가한 MPI.
•	Tenstorrent에서는 안정적인 실행을 위해 이 ULFM 버전의 MPI를 사용함.

✅ 왜 모델 실행 전에 설치해야 하나요?
모델 실행 시:
•	여러 프로세스/장치(NPU)가 병렬로 일하면서
•	서로 데이터를 실시간으로 주고받아야 하기 때문.
이게 안 되면:
•	모델 실행 중 hang(멈춤)이나 오류가 발생할 수 있음.

📌 정리 요약
항목	설명
MPI	분산 시스템 간 통신용 표준 프로토콜
ULFM	장애 복구 기능이 강화된 MPI 확장판
설치 이유	NPU나 멀티 노드 환경에서 모델을 나눠 실행할 때 필수
Tenstorrent와의 관계	모델을 여러 NPU에서 실행하려면 통신이 필요하고, 이를 위해 MPI가 사용됨

## vanishing gradients
1️⃣ Vanishing Gradients(기울기 소실)란 무엇인가?

Vanishing Gradient(기울기 소실) 은
딥러닝에서 역전파(Backpropagation, 역방향 전파) 를 할 때
앞쪽 레이어로 갈수록 Gradient(기울기, 미분값) 가 거의 0에 가까워져
학습이 멈춰버리는 현상입니다.

핵심 한 줄 요약:

“깊은 신경망에서 앞단 레이어가 거의 학습하지 못하는 문제”

2️⃣ 왜 이런 일이 생길까?

신경망 학습은 이런 구조죠:

입력 → Layer1 → Layer2 → Layer3 → ... → 출력


그리고 학습할 때는 거꾸로 갑니다:

출력 → Layer3 → Layer2 → Layer1


이때 각 레이어를 거칠 때마다 기울기를 계속 곱합니다.

문제의 핵심

기울기가 예를 들어 0.5라고 해봅시다.

Layer가 10개라면:

0.5 × 0.5 × 0.5 × ... (10번)
= 0.000976...


👉 거의 0이 됩니다.

이러면 앞쪽 레이어는:

weight가 거의 업데이트되지 않음

사실상 학습이 멈춤

3️⃣ 왜 특히 옛날 네트워크에서 심했을까?

예전에는 Sigmoid 함수 를 많이 썼습니다.

Sigmoid의 미분값은 최대가 0.25입니다.

즉, 역전파 때마다:

0.25 × 0.25 × 0.25 × ...


👉 급격히 작아짐

그래서 깊은 네트워크가 잘 안 배웠습니다.

4️⃣ 결과적으로 무슨 일이 벌어질까?

깊은 네트워크가 의미를 못 배움

앞단 레이어는 랜덤 초기값에 가까움

학습이 느리거나 멈춤

성능이 안 나옴 (=정밀도가 떨어짐)

5️⃣ 그래서 어떻게 해결했을까?

딥러닝이 발전한 이유가 여기 있습니다.

✅ 1. ReLU (Rectified Linear Unit, 렐루 활성화 함수)

ReLU는 미분값이 1 또는 0입니다.

→ 곱해도 작아지지 않음
→ Gradient 유지

✅ 2. Residual Connection (잔차 연결)

Deep Residual Learning for Image Recognition

ResNet에서 도입.

출력에 입력을 더해버립니다:

output = F(x) + x


→ 기울기가 직접 전달됨
→ Vanishing 거의 해결

✅ 3. LSTM (Long Short-Term Memory, 장단기 기억 네트워크)

Long Short-Term Memory

게이트 구조로 gradient를 유지합니다.
RNN에서 vanishing 문제를 극복한 구조입니다.

✅ 4. Layer Normalization (층 정규화)

Transformer에서 사용.

→ 값의 분포를 안정화
→ 기울기 흐름 안정화

6️⃣ 직관적 비유

당신이 100명에게 소문을 전달한다고 합시다.

각 사람이 전달할 때마다 내용을 10%만 남기고 전달한다면?

100명 뒤에는 거의 아무 정보도 안 남습니다.

그게 vanishing gradient입니다.

## vocabulary size
🧠 개념 설명
🔤 Vocabulary Size (어휘 집합 크기)란?
•	토크나이저는 **문장을 토큰(token)**이라는 조각으로 나눔.
•	이때, 어떤 조각들을 토큰으로 쓸지 정해 놓은 **사전(=Vocabulary, 어휘집)**이 필요.
•	vocabulary size는 이 어휘 사전에 들어 있는 토큰의 총 개수를 의미.
📦 예시
예: vocabulary size = 30,000
→ 토크나이저는 30,000개의 토큰 조각들만 기억하고, 그 외에는 쪼개거나 [UNK]로 처리.
•	hello, world, ##ing, ##tion, ##세요 같은 조각들이 포함될 수 있음.

📈 숫자 크기에 따른 차이
Vocabulary Size	특징	장점	단점
30K~50K (보통)	적당히 조각이 크고 적음	메모리, 속도 효율적	희귀 단어는 잘게 쪼개짐
100K 이상 (최근 증가 추세)	더 다양한 조각 기억	희귀어, 외래어 표현력 증가	모델 크기 커짐, 훈련 시간 증가

🧩 왜 중요할까?
•	Vocabulary size는 모델이 얼마나 다양한 단어를 세밀하게 다룰 수 있는지를 결정.
•	너무 작으면: "초전도체" → [초] [전] [도] [체] → 의미 단위 손실
•	너무 크면: 모델이 커지고 학습이 느려질 수 있음
📌 실제로 사용하는 값들
모델 이름	Vocabulary Size (어휘 크기)
BERT (base)	약 30,000
GPT-2	약 50,000
GPT-4 (추정)	약 100,000 이상
KoBERT	약 8,000~32,000 (설정에 따라 다름)

✅ 요약 정리
항목	설명
Vocabulary size	토크나이저가 기억하는 토큰 개수
보통 값	30K, 50K, 최근에는 100K까지 사용
작으면	속도는 빠르나 희귀어 표현력 약함
크면	표현력 좋지만 모델이 무거워짐

## weight
**딥러닝(Deep Learning)**에서 말하는 **“weight(가중치)”**는 모델이 입력 데이터를 얼마나 중요하게 여길지를 결정하는 숫자 값.
✅ 한 줄 정의
**Weight(가중치)**란
입력 값에 곱해져서 출력에 얼마나 영향을 줄지를 조절하는 숫자
🧠 쉽게 비유하면
Weight는 “입력 값에 주는 중요도 점수”.
어떤 정보는 더 중요하게 보고, 어떤 정보는 덜 중요하게 보게 하는 숫자
예를 들어:
•	“강아지 사진”을 보고 AI가 "강아지"라고 판단할 때
→ “귀”는 중요하니까 높은 weight
→ “배경 풀잎”은 덜 중요하니까 낮은 weight
🔢 수식 예시 (아주 단순한 뉴런 1개)
출력 = 입력 × 가중치 + 편향  
    y = x × w + b
구성 요소	설명
x	입력값
w	가중치 (weight)
b	편향 (bias)
y	결과 출력
이 w가 바로 weight
🎯 딥러닝에서의 역할
역할	설명
입력 특징에 “중요도” 부여	중요한 입력일수록 큰 weight 부여
학습 대상	모델이 학습하면서 weight를 스스로 조정
연결선의 강도 표현	뉴런 간 연결에서 얼마나 강하게 영향을 주는지
📌 용어 병기
영어	한글	의미
Weight	가중치	입력값에 곱해지는 중요도 조절 숫자
Trainable Weight	학습 가능한 가중치	학습 중 계속 업데이트되는 weight 값
Weight Matrix	가중치 행렬	여러 개의 weight가 행렬 형태로 연결된 것
✅ 정리
**Weight(가중치)**는 입력 값이 출력에 얼마나 큰 영향력을 미칠지 결정하는 숫자이며,
딥러닝에서 모델이 학습하는 가장 중요한 대상 중 하나.

## weight decay
신경망 학습 시 과적합(overfitting)을 막기 위한 정규화 기법.

✅ 한 줄 정의
Weight Decay란,
모델의 가중치(weight) 값이 너무 커지는 것을 방지하기 위해,
매 학습 단계마다 조금씩 줄여주는(감쇠시키는) 정규화 방식

✅ 왜 쓰는가?
문제 상황	해결 방식
학습이 잘 될수록 → 가중치가 커지고	→ 과적합 가능성 증가
너무 복잡한 함수(큰 weight)가 학습될 수 있음	→ Weight Decay로 가중치를 눌러서 일반화 유도

✅ 작동 방식 요약
학습할 때마다
W ← W - learning_rate × gradient + λ × W
→ 즉, 가중치에 작은 “마이너스” 페널티를 줌

✅ 실제 예
옵티마이저	Weight Decay 지원 여부
SGD	✅ 기본적으로 L2 정규화와 동일
AdamW	✅ Adam + Weight Decay 전용 구조 (Adam에선 L2랑 다름)
Adam	❌ L2 Regularization은 있지만, Weight Decay와 방식이 다름

✅ 왜 중요한가?
장점	설명
🧠 일반화 성능 향상	훈련 데이터에 과하게 맞추는 것 방지 (overfitting 방지)
⚖️ 모델 안정화	가중치가 급격히 커지는 현상 완화
🔧 간단한 튜닝 가능	하나의 하이퍼파라미터(λ)만 조절

✅ 요약
항목	내용
무엇인가?	학습 중 가중치를 조금씩 줄이는(감쇠시키는) 정규화 방법
왜 쓰나?	과적합 방지, 일반화 성능 향상
어떻게 작동하나?	손실 함수에 가중치 크기에 대한 패널티 추가하거나, 옵티마이저에서 직접 적용
주요 옵티마이저	AdamW, SGD + L2, 등

## W&B (Weights & Biases)
모델 학습 과정을 시각화하고 모니터링할 수 있게 도와주는 툴. (모델 정확도, 손실 값 등을 실시간으로 기록하고 그래프로 보여줌). 학습(training)과 평가(evaluation)의 **지표(metric, 성능 수치들)**는 W&B에 기록

## Weight tying (웨이트 타이잉)
1️⃣ 한 문장 정의

Weight Tying(가중치 공유) 는
👉 두 개 이상의 신경망 레이어(layer)가 같은 파라미터(parameter, 가중치)를 함께 사용하는 것 입니다.

즉,
레이어는 두 개지만 가중치는 하나입니다.

2️⃣ GPT에서 왜 나오는 개념인가?

LLM, 특히 GPT 구조에서 가장 유명한 가중치 공유는:

입력 임베딩 레이어 (Input Embedding Layer)
출력 선형 레이어 (Output Linear Layer, LM Head)

이 둘이 같은 가중치를 공유하는 경우입니다.

3️⃣ 구조를 그림으로 이해해 봅시다
(1) 일반적인 구조 – 가중치 공유 없음
입력 토큰 → Embedding Matrix A
마지막 히든 벡터 → Output Matrix B
A와 B는 서로 다른 가중치

👉 파라미터가 2배로 필요

(2) Weight Tying 적용 구조
입력 토큰 → Embedding Matrix W
출력 계산 → 같은 Matrix Wᵀ (전치, Transpose) 사용

👉 A와 B가 같은 가중치
👉 파라미터 절약
👉 의미 구조 일관성 증가

4️⃣ 왜 이게 논리적으로 말이 될까?

입력할 때

"apple" → 벡터로 변환

출력할 때

다음 단어 예측 → 다시 단어 공간으로 매핑

즉,

입력 공간 ↔ 출력 공간이 같은 "단어 의미 공간"

그렇다면
굳이 다른 가중치를 써야 할까요?

5️⃣ 수학적으로 보면

임베딩 행렬:

𝑊 ∈ 𝑅𝑉×𝑑


V = vocabulary size
d = embedding dimension

입력:

𝑥 → 𝑊[𝑥]

출력:

𝑙𝑜𝑔𝑖𝑡𝑠 = ℎ ⋅𝑊𝑇

여기서 같은 W를 씁니다.

6️⃣ 장점
1️⃣ 파라미터 감소

모델 크기 줄어듦
(대형 LLM에서 엄청난 차이)

2️⃣ 일반화 성능 향상

입력과 출력이 같은 의미 공간 공유 → 더 자연스러운 학습

3️⃣ 과적합 감소

불필요한 자유도 감소

7️⃣ 단점은?

표현력이 약간 제한될 수 있음

특정 구조에서는 성능이 떨어질 수도 있음

하지만 GPT 계열 모델에서는 거의 표준입니다.

특정 구조에서는 성능이 떨어질 수도 있음의 의미
모델 구조(architecture)와 목적(objective function) 에 따라 표현력이 제한될 수 있다는 뜻.

1️⃣ 입력과 출력의 역할이 대칭이 아닐 때

Weight Tying(가중치 공유)은 기본적으로 이런 가정을 합니다:

입력 토큰의 의미 공간과 출력 토큰의 예측 공간이 동일하다.

하지만 항상 그런가요?

대표적인 예:

번역 모델

Encoder-Decoder Transformer

예: Google 의 초기 번역 모델

OpenAI 의 일부 seq2seq 모델

왜 문제가 될 수 있을까?

입력:

영어 문장

출력:

프랑스어 문장

👉 입력 vocabulary ≠ 출력 vocabulary
👉 의미 공간이 완전히 다름

이 경우 weight tying을 쓰면:

영어 임베딩과 프랑스어 출력 공간을 강제로 동일하게 만듦

표현력 제한 발생

2️⃣ 입력과 출력의 통계적 역할이 다를 때

GPT는 “다음 단어 예측”을 합니다.

입력:

과거 문맥

출력:

다음 토큰 확률 분포

이 둘은 같은 단어 집합을 쓰지만, 역할은 다릅니다.

입력 임베딩은

문맥 표현을 위한 벡터

출력 레이어는

확률 분포를 만들기 위한 분류기(Classifier)

이 둘이 반드시 동일한 구조를 가져야 할까요?

만약 출력 레이어가:

더 복잡한 비선형 변환을 필요로 한다면?

다른 차원의 투영(projection)이 필요하다면?

그때 weight tying은 제약이 됩니다.

3️⃣ Adaptive Softmax 같은 구조
대형 vocabulary에서:

Adaptive Softmax

Hierarchical Softmax

이런 구조는 출력 레이어를 계층적으로 나눕니다.

이 경우:

출력 가중치 구조가 입력 임베딩과 다르게 설계됨

weight tying 불가능 또는 성능 저하 가능

4️⃣ Multi-Modal 구조

예를 들어:

텍스트 + 이미지 모델

텍스트 + 음성 모델

입력 임베딩:

이미지 벡터

음성 벡터

출력:

텍스트 토큰

이때 입력 공간과 출력 공간은 완전히 다른 modality입니다.

당연히 weight tying 불가.

5️⃣ 표현력 제한 문제

수학적으로 보면,

출력 logits:

logits = hWT

W가 고정되어 있으면
출력 분포는 입력 임베딩 공간의 구조에 종속됩니다.

즉,

출력 분포가 임베딩 공간에 의해 제약됨

만약 출력에 더 자유로운 결정 경계(decision boundary)가 필요하면
별도 weight가 더 유리할 수 있습니다.

6️⃣ 실제 사례

논문들에서는:

작은 모델에서는 weight tying이 성능 향상

매우 큰 모델에서는 차이가 거의 없음

특정 task에서는 독립 출력 레이어가 더 좋음

즉, “항상 좋다”는 건 아닙니다.

## Wheel
파이썬 프로그램(라이브러리나 도구)을 설치할 수 있게 만든 “파이썬용 설치 파일”
✅ 예:
•	tt-metal-1.2.3-cp310-cp310-linux_x86_64.whl
→ 이 파일 하나면 Tenstorrent의 Python API를 설치할 수 있음.

📦 Wheel은 어떤 느낌이냐면…
•	마치 “레고 조립 설명서가 포함된 부품 상자” 같은 느낌입니다.
•	내가 이미 Python 환경을 갖추고 있다면:
pip install tt-metal-1.2.3.whl
이렇게 하면 됨.

🔍 wheel vs docker
항목	Wheel (.whl)	Docker
용도	Python 라이브러리 배포	전체 실행 환경 배포
설치 방법	pip install xxx.whl	docker pull + docker run
Python 환경 필요?	✅ 있어야 함	❌ 없어도 됨 (Python 포함 가능)
용량	작음 (몇 MB~)	큼 (수백 MB~GB)
실행 환경 통제	사용자가 환경 세팅해야 함	모든 환경 포함됨 (OS, 패키지, 버전 등)
실행 대상	Python 사용자	누구나 (개발자/운영자 등)

🎯 예를 들면:
상황	어떤 걸 써야 하나요?
Python 개발자에게 API 배포	✅ Wheel (.whl)
팀원 모두가 똑같은 환경으로 AI 모델 실행하게 하고 싶음	✅ Docker
내 노트북에서 모델만 간단히 써보고 싶음	✅ Wheel
클라우드 서버에서 즉시 실행하려고 함	✅ Docker


## word2vec
자연어 처리(NLP)에서 단어를 숫자 벡터로 표현하는 기술 중 가장 유명한 모델 중 하나.
✅ 한 줄 정의
Word2Vec은
비슷한 의미를 가진 단어들을 벡터 공간에서 가깝게 표현하는 방법.
즉, 단어를 숫자로 바꾸되, 단순한 숫자가 아니라 의미를 담은 벡터로 바꾸는 기술
🔍 왜 필요한가?
컴퓨터는 단어 "apple"이나 "banana"를 그냥 문자열로는 이해 못 함.
그래서 단어를 숫자로 바꾸되,
비슷한 단어는 비슷한 숫자 벡터로 바꿔줘야 의미 처리가 가능.
📌 핵심 아이디어
“비슷한 문맥에서 자주 등장하는 단어는 의미도 비슷하다”
예:
•	"dog"와 "puppy"는 비슷한 문장에서 많이 나옴
•	→ 벡터 공간에서도 가까운 위치에 있게 됨
🔧 학습 방식 두 가지
방식	설명
CBOW (Continuous Bag of Words)	주변 단어들을 보고 중심 단어를 예측함
Skip-Gram	중심 단어를 보고 주변 단어들을 예측함 (더 많이 쓰임)
예시 문장: "The cat sits on the mat"
•	중심 단어: "sits"
•	CBOW: "cat, on" → 예측 → "sits"
•	Skip-Gram: "sits" → 예측 → "cat", "on"
🧠 결과: 벡터로 표현된 단어
•	"apple" → [0.31, -0.22, ..., 0.87]
•	"banana" → [0.35, -0.20, ..., 0.90]
→ 비슷한 의미 → 벡터도 비슷
💡 재미있는 특징
Word2Vec은 단어 간 연산도 가능
예:
vec("king") - vec("man") + vec("woman") = vec("queen")
📌 용어 병기
영어	한글	설명
Word2Vec	워드투벡	단어를 의미 벡터로 바꾸는 모델
CBOW	연속 단어 집합	주변 단어 → 중심 단어 예측
Skip-Gram	스킵그램	중심 단어 → 주변 단어 예측
Embedding Vector	임베딩 벡터	단어를 표현한 숫자 벡터

✅ 정리
Word2Vec은 단어를 벡터로 표현하면서 의미를 반영하게 만든 기술로, 오늘날의 AI 언어 모델(BERT, GPT 등)의 기초가 된 개념.

## word embeddings
AI에서 텍스트(단어)를 숫자로 표현하는 핵심 기술. 단어(Word)를 컴퓨터가 이해할 수 있게 숫자 벡터(Vector)로 바꾼 것. 
🔡 예를 들어
단어	임베딩 벡터 (예시)
"king"	[0.21, -0.43, 1.12, ..., 0.88]
"queen"	[0.19, -0.41, 1.08, ..., 0.85]
→ 이 숫자 벡터들은 단어의 의미와 문맥적 관계를 반영

🧠 왜 필요한가?
기계는 "apple"이라는 단어 자체는 이해 못 함.
하지만 [0.54, -0.12, 1.03, ...] 같은 벡터로 바꾸면:
•	단어 간 의미적 유사성 (예: "apple" ≈ "banana")
•	문맥 정보 (예: "drink" ≈ "eat")
를 계산할 수 있음.

📌 주요 Word Embedding 기법
방법	설명
Word2Vec	주변 단어 예측 또는 주어진 단어 예측 (CBOW/Skip-Gram)
GloVe (Global Vectors)	전체 말뭉치에서 단어들의 동시 등장 빈도 기반
FastText	**부분 단어(subword)**까지 고려 → 희귀 단어에도 강함
BERT, GPT (Contextual Embedding)	문맥에 따라 벡터가 달라짐 (다음에 더 자세히 설명 가능)

✅ 1. 추천 시스템(Recommender Engines)에서 Word Embeddings이 유용한 이유
📌 핵심 개념:
사용자와 아이템을 벡터로 표현해서,
벡터 간 유사도를 계산해 추천하는 방식을 사용함
🔍 어떻게 연결되나?
•	Word Embeddings처럼,
→ 사용자, 상품, 검색 쿼리도 벡터로 임베딩해서 표현
•	예:
o	사용자 A: [0.2, -0.1, 0.9, ...]
o	영화 B: [0.25, -0.05, 0.88, ...]
o	→ 두 벡터가 가까우면 = 추천
💡 실제 적용 예시:
항목	설명
쇼핑몰	상품명, 검색어, 리뷰를 임베딩 → 유사한 상품 추천
영화 추천	장르, 줄거리, 배우 정보를 Word Embedding으로 표현
음악 앱	가사 + 사용자 청취 기록 → 임베딩 기반 음악 추천

✅ 2. 로봇공학(Robotics)에서 Word Embeddings이 유용한 이유
📌 핵심 개념:
로봇이 명령어 또는 환경 인식 대상(객체, 동작 등)을 벡터로 표현해야 행동 가능
🔍 어떻게 연결되나?
•	인간의 음성 명령이나 텍스트 지시:
“Pick up the red cup and place it on the table.”
•	이걸 Word Embedding으로 처리하면:
o	"cup", "table", "red" 같은 단어들을 의미 벡터로 변환
o	→ 로봇이 "이건 물건이고", "이건 장소"라는 걸 벡터 간 관계로 파악 가능
💡 실제 적용 예시:
응용	설명
자연어 명령 인식	단어를 벡터로 바꿔서 로봇이 의미 이해
시각 정보 결합	카메라로 인식한 사물(Label) + Word Embedding → 환경 판단
강화학습 + 언어	"go to kitchen" 같은 지시 → 임베딩으로 바꿔 행동 계획 수립

## wordPiece (BERT 모델)
WordPiece는 자연어를 기계가 잘 이해할 수 있도록 단어를 잘게 쪼개는 토크나이징(tokenizing) 방식 중 하나. 특히 BERT나 KoBERT, DistilBERT 같은 모델들이 사용하는 토크나이저가 바로 이 WordPiece 방식.

▶️ 정의:
WordPiece는 단어를 서브워드(subword, 단어보다 작은 의미 단위, 즉, 부분단어) 단위로 나누는 서브워드 토크나이저 알고리즘. 이 방식은 완전히 새로운 단어가 들어와도 부분적으로 쪼개서 이해할 수 있도록 설계됨.
▶️ 예시:
-	문장: 
"unbelievable"
-	WordPiece 토크나이징 결과:
["un", "##bel", "##ievable"]
👉 설명:
•	"##"는 이 토큰이 단어의 앞부분이 아님을 나타내는 표시.
•	즉 "un"은 단어의 시작,
"##bel", "##ievable"은 앞 단어에서 이어짐을 뜻함.
📌 왜 WordPiece를 쓸까?
이유	설명
✅ 희귀 단어 처리	완전 처음 보는 단어도 부분단어로 쪼개면 해석 가능
✅ 어휘 크기 최적화	모든 단어를 외워 둘 필요 없이, 자주 쓰이는 조각만 기억하면 됨
✅ 다국어 대응	영어, 한국어, 독일어 등 다양한 언어에서 효과적

📦 예시: 한국어에서 WordPiece
입력 문장:
"서울대학교"
WordPiece 처리 결과 (예시):
["서울", "##대", "##학", "##교"]

## YARN
분산 컴퓨팅 환경에서 리소스 관리와 작업 스케줄링을 담당하는 핵심 기술입니다. 빅데이터나 AI 클러스터에서 자주 등장. 디스패처는 추론 서비스용으로 사용되는 반면, YARN은 대규모 학습용으로 사용됨.

🔷 YARN의 정식 이름
YARN = Yet Another Resource Negotiator
→ *“또 하나의 자원 협상자”*라는 뜻의 약자로, 이름은 유쾌하지만 기능은 아주 중요

🔷 YARN이 뭔지 한 줄 요약
YARN은 대규모 클러스터(서버 묶음)에서 CPU, GPU, NPU, 메모리 같은 자원을 관리하고, 어떤 작업(job)을 언제 어디서 실행할지 조정하는 시스템

🔸 어디에 쓰이냐?
•	Apache Hadoop의 핵심 컴포넌트 중 하나.
•	AI/ML 학습 파이프라인, Spark 클러스터, 데이터 파이프라인 등에서도 자주 사용

🔸 왜 필요하냐?
예를 들어, 수백 대의 서버(GPU 포함)를 묶은 클러스터가 있을 때:
•	어떤 노드에 어떤 작업을 넣을지?
•	리소스가 남아 있는지?
•	작업이 충돌 없이 잘 실행되게 하려면 어떻게 조율할지?
이런 걸 자동으로 조정해주는 게 YARN

🔷 구성요소 요약
구성 요소	역할 요약
ResourceManager	전체 클러스터의 자원을 통제하는 "총 관리자"
NodeManager	개별 서버의 자원 상태를 보고하고 작업 실행 담당
ApplicationMaster	사용자 작업(Job)의 생명 주기를 관리
Container	실제로 작업이 실행되는 공간 (CPU+RAM 등 리소스 포함)

🔶 예시 상황
예를 들어, 여러분이 AI 모델 학습 작업을 spark-submit으로 실행하면:
1.	YARN이 리소스 체크 후 가능한 노드에 Container를 할당
2.	ApplicationMaster가 학습 작업을 스케줄링
3.	NodeManager가 작업 실행
4.	작업이 끝나면 리소스 반환

🔷 한눈에 요약
항목	설명
정식 명칭	Yet Another Resource Negotiator
역할	클러스터 리소스 관리 및 작업 스케줄링
속한 프레임워크	Apache Hadoop (2.x 이상)
사용 환경	빅데이터 분석, Spark, AI 학습 클러스터 등
유사한 기술	Kubernetes, Mesos, Slurm 등

🔷 디스패처와 YARN의 관계 정리
기능	디스패처	YARN
작업 위치 지정	✅	✅
자원 상태 감시	❌ 또는 부분적	✅
작업 실행 지시	✅	✅
작업 생명 주기 관리	❌	✅ (ApplicationMaster)
자원 충돌 방지	❌	✅

## zero-shot classification
✅ 한 줄 정의
**“모델이 한 번도 훈련(training)받지 않은 라벨(label)을 사용해서 분류할 수 있는 능력”**
즉,
❌ 기존 분류 모델처럼 ‘정답 데이터’(labeled data)로 미리 학습하지 않고도
✅ 텍스트를 바로 분류할 수 있는 것

🔍 쉽게 설명해볼게요
기존 방식 (supervised classification)은:
훈련 데이터:  
- "이 영화 최고였어!" → 긍정  
- "진짜 별로였다..." → 부정

→ 그런 다음, 새 문장을 예측

Zero-shot 방식은:
훈련 없이 바로 질문:

"이 영화 최고였어!"  
→ 이 문장은 '긍정', '부정' 중 뭐에 더 가까워?

➡ 모델이 직접 판단

🔧 작동 방식 (실제 예)
from transformers import pipeline

classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

sequence = "This movie was absolutely fantastic!"
labels = ["positive", "negative", "neutral"]

result = classifier(sequence, candidate_labels=labels)
print(result)

➡ 출력 예:
{
  'labels': ['positive', 'neutral', 'negative'],
  'scores': [0.92, 0.06, 0.02]
}
✅ 즉, 모델이 학습 없이도 positive라는 레이블을 이해하고 예측

🧠 왜 가능한가요?
•	최근 LLM이나 대형 사전학습 모델 (예: BART, RoBERTa, GPT 등)은
문장의 의미와 레이블 단어(예: "positive")의 관계를 자연어 추론(NLI) 방식으로 판단할 수 있기 때문입니다.
•	그래서 **"이 문장이 '긍정'이라는 가설을 지지하나요?"**처럼 이해하고 분류할 수 있는 것.

📌 어디에 쓰이나요?
분야	활용 예
고객 리뷰 분석	“긍정/부정/중립” → Zero-shot으로 바로 분류
뉴스 분류	“정치/스포츠/경제/연예” 등 다양한 라벨
비정형 텍스트 라벨링	의료, 법률, 기업 내 이메일 등

✅ 요약
항목	설명
이름	Zero-shot classification
학습 방식	사전 fine-tuning 없이 분류
장점	라벨 없이도 다양한 분류 가능
기반 기술	대형 사전학습 모델 + 자연어 추론 (NLI)
대표 모델	bart-large-mnli, roberta-large-mnli, GPT, Claude, Gemini 등

