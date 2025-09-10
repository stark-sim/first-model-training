# Feature 特征数据
X = [[10, 3], [20, 3], [25, 3], [28, 2.5], [30, 2], [35, 2.5], [40, 2.5]]
# Label 数据
Y = [60, 85, 100, 120, 140, 145, 163]

# 初始化参数，两个可调整参数带一个偏置值，也就是 3 个可调整参数
w = [0.0, 0.0, 0.0]
# 学习率
lr = 0.001
# 迭代次数
num_iterations = 10000000

# 梯度下降
for i in range(num_iterations):
    # 预测值
    y_pred = [w[0] + w[1] * x[0] + w[2] * x[1] for x in X]
    # 计算损失
    loss = sum([(y_pred[i] - Y[i]) ** 2 for i in range(len(Y))]) / len(Y)

    # 计算梯度
    grad_w0 = 2 * sum(y_pred[i] - Y[i] for i in range(len(Y))) / len(Y)
    grad_w1 = 2 * sum((y_pred[i] - Y[i]) * X[i][0] for i in range(len(Y))) / len(Y)
    grad_w2 = 2 * sum((y_pred[i] - Y[i]) * X[i][1] for i in range(len(Y))) / len(Y)
    # 更新参数，梯度负方向，步长是梯度分量乘学习率
    w[0] -= lr * grad_w0
    w[1] -= lr * grad_w1
    w[2] -= lr * grad_w2

    # 打印损失
    if i % 100 == 0: # 100 轮打一次
        print(f"Iteration {i}: Loss = {loss}")

# 输出最终参数
print(f"Final parameters: w0 = {w[0]}, w1 = {w[1]}, w2 = {w[2]}")
