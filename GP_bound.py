import numpy as np
import matplotlib.pyplot as plt

# ---------------------------
# 1. 定义 ReLU（arccosine）核函数
# ---------------------------
def relu_kernel(X1, X2, sigma_w=1.0, sigma_b=0.0):
    """
    计算 ReLU 激活对应的 arccosine 核矩阵.
    公式参考 Cho & Saul (2009)：
      k(x, x') = sigma_w^2 * ||x|| ||x'|| * (1/pi) * (sin(theta) + (pi - theta) * cos(theta)) + sigma_b^2,
    其中 theta = arccos( (x.T x')/(||x|| ||x'||) ).
    
    Parameters:
      X1: numpy 数组, shape = (N, d)
      X2: numpy 数组, shape = (M, d)
      sigma_w: 权重尺度 (默认1.0)
      sigma_b: 偏置尺度 (默认0.0)
      
    Returns:
      K: 核矩阵, shape = (N, M)
    """
    dot_prod = np.dot(X1, X2.T)
    norm1 = np.linalg.norm(X1, axis=1, keepdims=True)
    norm2 = np.linalg.norm(X2, axis=1, keepdims=True)
    norm_prod = np.dot(norm1, norm2.T)
    
    eps = 1e-6  # 避免除零
    cos_theta = np.clip(dot_prod / (norm_prod + eps), -1.0, 1.0)
    theta = np.arccos(cos_theta)
    
    K = sigma_w**2 * norm_prod * (1.0 / np.pi) * (np.sin(theta) + (np.pi - theta) * cos_theta) + sigma_b**2
    return K

# ---------------------------
# 2. 定义 GP 回归函数（单输出）
# ---------------------------
def gp_predict(X_train, y_train, X_test, kernel_func, sigma_n=0.1):
    """
    利用 GP 的闭式公式计算预测均值和标准差.
    
    Parameters:
      X_train: 训练输入, shape = (N, d)
      y_train: 训练输出, shape = (N,)
      X_test: 测试输入, shape = (M, d)
      kernel_func: 核函数，函数形式 kernel_func(X1, X2)
      sigma_n: 观测噪声标准差
      
    Returns:
      mu_star: 预测均值, shape = (M,)
      sigma_star: 预测标准差, shape = (M,)
    """
    K = kernel_func(X_train, X_train)
    N = X_train.shape[0]
    K += sigma_n**2 * np.eye(N)
    
    K_star = kernel_func(X_test, X_train)  # shape (M, N)
    K_star_star = kernel_func(X_test, X_test)
    diag_K_star_star = np.diag(K_star_star)
    
    alpha = np.linalg.solve(K, y_train)
    mu_star = np.dot(K_star, alpha)
    
    v = np.linalg.solve(K, K_star.T)
    var_star = diag_K_star_star - np.sum(K_star * v.T, axis=1)
    sigma_star = np.sqrt(np.maximum(var_star, 0))
    
    return mu_star, sigma_star

# ---------------------------
# 3. 定义 softplus 函数及其导数（Delta 方法）
# ---------------------------
def softplus(x):
    return np.log1p(np.exp(x))  # log(1+exp(x))

def softplus_derivative(x):
    return 1 / (1 + np.exp(-x))  # sigmoid(x)

# ---------------------------
# 4. 生成训练数据
# ---------------------------
np.random.seed(42)
N_train = 50
d = 2
X_train = np.random.uniform(-5, 5, size=(N_train, d))

# 模拟四个输出（输出维度为4）
def true_function(x, output_index):
    # 定义不同的函数，后面两个输出将经过 softplus 变换（注意：softplus 应用于神经网络原始输出）
    if output_index == 0:
        return np.sin(x[:, 0]) + 0.1 * x[:, 1]
    elif output_index == 1:
        return np.cos(x[:, 1]) - 0.1 * x[:, 0]
    elif output_index == 2:
        return 0.5 * x[:, 0] + 0.5 * x[:, 1]  # 需要 softplus 变换后
    elif output_index == 3:
        return np.sin(x[:, 0]*x[:, 1])  # 需要 softplus 变换后
    else:
        return np.zeros(x.shape[0])
    
noise_std = 0.1
Y_train = np.zeros((N_train, 4))
for j in range(4):
    Y_train[:, j] = true_function(X_train, j) + noise_std * np.random.randn(N_train)

# ---------------------------
# 5. 定义测试数据
# ---------------------------
N_test = 100
X_test = np.random.uniform(-5, 5, size=(N_test, d))

# ---------------------------
# 6. 对每个输出进行 GP 回归，部分输出经过 softplus 变换
# ---------------------------
# 假设输出维度 2 和 3（索引为2和3）需要经过 softplus 变换
mu_post = np.zeros((N_test, 4))
sigma_post = np.zeros((N_test, 4))

# 针对每个输出维度分别进行 GP 回归
for j in range(4):
    mu, sigma = gp_predict(X_train, Y_train[:, j], X_test, relu_kernel, sigma_n=noise_std)
    # 如果该输出需要 softplus 变换，则利用 Delta 方法传播不确定性
    if j in [2, 3]:
        # Delta 方法：y = softplus(z)
        # 近似均值： softplus(mu)
        # 近似标准差： softplus'(mu)*sigma
        mu_post[:, j] = softplus(mu)
        sigma_post[:, j] = softplus_derivative(mu) * sigma
    else:
        mu_post[:, j] = mu
        sigma_post[:, j] = sigma

# ---------------------------
# 7. 构造置信区间 (Bound)
# ---------------------------
beta = 2.0  # 例如，取2倍标准差近似95%置信区间
lower_bound = mu_post - beta * sigma_post
upper_bound = mu_post + beta * sigma_post

# 输出部分结果
for j in range(4):
    print(f"\nOutput dimension {j+1}:")
    print("Predicted mean (first 5 test points):", mu_post[:5, j])
    print("Predicted std (first 5 test points):", sigma_post[:5, j])
    print("95% Confidence interval (first 5 test points):")
    for i in range(5):
        print(f"  x = {X_test[i]}: [{lower_bound[i, j]:.3f}, {upper_bound[i, j]:.3f}]")

# ---------------------------
# 8. 绘制某个输出维度的预测分布（例如输出维度 3，经 softplus 变换）
# ---------------------------
j_plot = 2  # 选择第三个输出（索引2，已经过 softplus）
plt.figure(figsize=(10,5))
plt.errorbar(np.arange(N_test), mu_post[:, j_plot], yerr=beta * sigma_post[:, j_plot],
             fmt='o', capsize=3, label='Predicted mean with 95% CI')
plt.title(f"GP Prediction for Output Dimension {j_plot+1} (after softplus)")
plt.xlabel("Test sample index")
plt.ylabel("Prediction")
plt.legend()
plt.show()
