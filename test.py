import numpy as np
from maraboupy import Marabou

print("=" * 50)
print("Marabou 검증 테스트 - Iris FC 네트워크")
print("=" * 50)

print("\n[1] ONNX 모델 로딩 중")
network = Marabou.read_onnx("iris_model.onnx")
print("모델 로드 완료!")

inputVars = network.inputVars[0].flatten()
outputVars = network.outputVars[0].flatten()
print(f"입력 변수 수: {len(inputVars)}")
print(f"출력 변수 수: {len(outputVars)}")

print("\n[2] 테스트 샘플 로딩 중")
X_test = np.load("X_test.npy")
y_test = np.load("y_test.npy")
sample = X_test[0]
true_label = int(y_test[0])
print(f"샘플: {sample}")
print(f"실제 클래스: {true_label}")

print("\n[3] 검증 쿼리 설정 중 (epsilon=0.01)")
epsilon = 0.01
for i, var in enumerate(inputVars):
    network.setLowerBound(var, float(sample[i] - epsilon))
    network.setUpperBound(var, float(sample[i] + epsilon))

for j in range(3):
    if j != true_label:
        network.addInequality(
            [outputVars[j], outputVars[true_label]],
            [1, -1],
            0
        )

print("\n[4] Marabou 실행 중")
result = network.solve()

print("\n" + "=" * 50)
print(f"검증 결과: {result[0].upper()}")
if result[0] == "unsat":
    print("검증 성공: epsilon=0.01 범위 내에서 모델이 robust함")
else:
    print("반례 발견: 적대적 예시가 존재함")
print("=" * 50)