# 面向含噪数据流的鲁棒在线学习算法

## 基分类器

### Logistic



### Linear SVM



### BernoulliNB



### Perceptron



### PassiveAggressiveClassifier





## 计算公式

### Ramp_loss

### Calculate_Weight 

###### Parameter:  

$$
\eta > 0
$$

###### Initialize:

$$
w_1 = (1/d,...,1/d)
$$

$x^4$

###### Update rule

$$

\forall i,w_{t+1}[i] = \frac{w_t[i]e^{-\eta z_t[i]}}{\sum_jw_t[j]e^{-\eta z_t[j]}} \quad

\\
z_t[i] = \left\{  
             \begin{array}{**lr**}  
             0 \quad h_i(x) = y&  \\  
             1 \quad h_i(x) \neq y &    
             \end{array}  
\right.
\\
h_i(x)为第i个base\_model的预测标签
\\
\eta 为初始指定参数

$$





