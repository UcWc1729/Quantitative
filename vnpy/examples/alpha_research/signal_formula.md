# Signal 的计算公式说明

## 一、Signal 是什么

**Signal = 模型对「未来收益」的预测值**。  
对测试集中每一个 (日期 \(t\), 股票 \(i\))，模型根据当天的因子特征算出一个数，即为该样本的 signal。

---

## 二、要预测的目标：Label（真实收益）

在 **Alpha158** 里，标签（被预测量）定义为：

```
label = ts_delay(close, -3) / ts_delay(close, -1) - 1
```

含义（按日频理解）：

- \( \mathrm{close}_{t+1} \)：\(t\) 日后第 1 天的收盘价  
- \( \mathrm{close}_{t+3} \)：\(t\) 日后第 3 天的收盘价  

所以：

\[
\boxed{
  \mathrm{label}_{t,i}
  = \frac{\mathrm{close}_{t+3,\,i}}{\mathrm{close}_{t+1,\,i}} - 1
  = \text{从 } t+1 \text{ 到 } t+3 \text{ 的持有收益（约 2 日收益）}
}
\]

即：在 \(t\) 日能看到的「未来一段收益」，用作训练和评估时的真实值。

（若用 **Alpha101**，label 的表达式不同，但同样是某种未来收益或收益相关量。）

---

## 三、输入：因子特征 \(X\)

对每个 (日期 \(t\), 股票 \(i\))，有一组**因子特征** \(x_{t,i}\)，来自 Alpha158 的 158 个表达式，例如：

- **K 线形态**：如  
  \( \mathrm{kmid} = \frac{\mathrm{close} - \mathrm{open}}{\mathrm{open}} \)，  
  \( \mathrm{klen} = \frac{\mathrm{high} - \mathrm{low}}{\mathrm{open}} \)
- **均线/动量**：如  
  \( \mathrm{ma\_5} = \mathrm{ts\_mean}(\mathrm{close}, 5) / \mathrm{close} \)，  
  \( \mathrm{roc\_5} = \mathrm{ts\_delay}(\mathrm{close}, 5) / \mathrm{close} \)
- **波动、成交量等**：如  
  \( \mathrm{std\_5}, \mathrm{vma\_5} \) 等  

所有这类特征在 \(t\) 日（及之前）都可计算，组成向量 \(x_{t,i}\)，作为模型输入。

---

## 四、模型如何得到 Signal（以 Lasso 为例）

### 4.1 训练阶段

用训练集 \((X_{\mathrm{train}}, y_{\mathrm{train}})\) 拟合 Lasso 回归（设 `fit_intercept=False`）：

\[
\min_{w} \;
\frac{1}{2n}\|y_{\mathrm{train}} - X_{\mathrm{train}} w\|_2^2
+ \alpha \|w\|_1
\]

得到系数向量 \(w \in \mathbb{R}^d\)（\(d\) 为因子个数）。

### 4.2 预测阶段（Signal 的公式）

对测试集中每个样本的特征向量 \(x_{t,i}\)（一行 \(X_{\mathrm{test}}\)）：

\[
\boxed{
  \mathrm{signal}_{t,i}
  = x_{t,i}^{\top} w
  = \sum_{k=1}^{d} x_{t,i}^{(k)} w_k
}
\]

即：**Signal = 测试集特征向量与训练得到的 Lasso 系数的内积**。

- 若用 **LightGBM / MLP**，公式不再是线性内积，但含义相同：  
  **signal = 模型(\(x_{t,i}\))**，即该模型对「未来收益」的预测值。

---

## 五、小结

| 符号 | 含义 | 公式/说明 |
|------|------|-----------|
| \(\mathrm{label}_{t,i}\) | 真实收益（目标） | \(\dfrac{\mathrm{close}_{t+3}}{\mathrm{close}_{t+1}} - 1\)（Alpha158） |
| \(x_{t,i}\) | 因子特征向量 | 158 维（Alpha158），由 K 线、均线、波动等表达式计算 |
| \(w\) | 模型参数 | Lasso 为 \(d\) 维系数；LGB/MLP 为内部参数 |
| \(\mathrm{signal}_{t,i}\) | 预测值（Signal） | Lasso：\(\mathrm{signal}_{t,i} = x_{t,i}^{\top} w\)；其他模型：\(\mathrm{signal}_{t,i} = \mathrm{model}(x_{t,i})\) |

因此：**Signal 的具体计算 = 用当前因子 \(x_{t,i}\) 通过已训练模型得到的目标收益预测值**；在 Lasso 下就是特征与系数的内积。
