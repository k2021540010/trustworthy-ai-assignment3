import numpy as np
from maraboupy import Marabou

network = Marabou.read_onnx("iris_model.onnx")

inputVars = network.inputVars[0].flatten()
outputVars = network.outputVars[0].flatten()

X_test = np.load("X_test.npy")
y_test = np.load("y_test.npy")

sample = X_test[0]
true_label = int(y_test[0])
print(f"검증 샘플: {sample}")
print(f"실제 클래스: {true_label}")

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

print(f"\nepsilon={epsilon} 범위에서 검증 시작")
result = network.solve()

print(f"\n결과: {result[0]}")
if result[0] == "unsat":
    print(f"검증 성공 -> {epsilon} 범위 내 모든 입력이 동일한 클래스로 분류됨")
else:
    print("반례 발견 -> 적대적 예시 존재")
    print(f"반례 입력값: {result[1]}")