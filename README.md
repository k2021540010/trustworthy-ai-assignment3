# Trustworthy AI Assignment 3 - Marabou Neural Network Verification

## 개요
이 프로젝트는 SMT(Satisfiability Modulo Theories) 기반 신경망 검증 도구인 **Marabou**를 사용하여
Iris 데이터셋으로 학습한 FC(Fully Connected) 네트워크의 robustness를 형식적으로 검증합니다.

검증 목표: 특정 입력 샘플 x에 대해 ε=0.01 범위(ℓ∞-ball) 내의 모든 입력이 동일한 클래스로 분류되는지 확인합니다.

---

## 환경
- Python 3.9.21
- OS: Linux (Ubuntu)

---

## 설치 방법

### 1. 레포지토리 클론
```bash
git clone https://github.com/[유저네임]/trustworthy-ai-assignment3.git
cd trustworthy-ai-assignment3
```

### 2. 가상환경 생성 (선택사항)
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. 패키지 설치
```bash
pip install -r requirements.txt
```

### 4. Marabou 설치
```bash
pip install maraboupy
```

---

## 파일 구조
trustworthy-ai-assignment3/
├── Marabou/            # Marabou 서브모듈
├── train_model.py      # Iris 데이터셋으로 FC 네트워크 학습 및 ONNX 변환
├── verify.py           # Marabou 검증 쿼리 실행 및 결과 분석
├── test.py             # Marabou 검증 데모 (제출용)
├── iris_model.onnx     # 학습된 ONNX 모델
├── X_test.npy          # 테스트 입력 데이터
├── y_test.npy          # 테스트 레이블
├── requirements.txt    # 의존성 패키지 목록
└── README.md           # 프로젝트 설명

---

## 실행 방법

### 1. 모델 학습 및 ONNX 변환
```bash
python3 train_model.py
```
- Iris 데이터셋(150개 샘플, 4개 특성, 3개 클래스)으로 FC 네트워크 학습
- 학습된 모델을 `iris_model.onnx`로 저장
- 테스트 데이터를 `X_test.npy`, `y_test.npy`로 저장

### 2. 검증 쿼리 실행
```bash
python3 verify.py
```
- 테스트 샘플 첫 번째를 기준으로 ε=0.01 범위에서 검증
- SAT: 반례(적대적 예시) 발견
- UNSAT: 검증 성공 (robust함이 증명됨)

### 3. 테스트 데모 실행
```bash
python3 test.py
```

---

## 모델 구조
입력층 (4) → Linear(4→16) → ReLU → Linear(16→3) → 출력층 (3)
- 입력: Iris 4개 특성 (꽃받침 길이/너비, 꽃잎 길이/너비)
- 출력: 3개 클래스 (Setosa, Versicolor, Virginica)
- 테스트 정확도: 100%

---

## 검증 쿼리 설명
- **입력 제약**: 샘플 x 기준 ε=0.01 범위의 ℓ∞-ball
  - `x[i] - ε ≤ input[i] ≤ x[i] + ε`
- **출력 제약**: true_label이 아닌 다른 클래스가 더 큰 경우 탐색
  - `output[j] > output[true_label]` (j ≠ true_label)
- **결과**: SAT → 반례 존재, UNSAT → robust함 증명

---

## 검증 결과
- 검증 결과: **SAT** (반례 발견)
- 검증 시간: **약 24ms**
- 해석: ε=0.01 범위 내에서 클래스 분류가 바뀌는 적대적 입력이 존재함