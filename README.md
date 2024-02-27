# MobiusE

## 公式推导

莫比乌斯环公式

$$
\begin{cases}
x(\theta,w)=(R+r\cos (\frac{\theta}{2}+w))\cos \theta\\
y(\theta,w)=(R+r\cos (\frac{\theta}{2}+w))\sin \theta\\
z(\theta,w)=r\sin (\frac{\theta}{2}+w)\\
\end{cases}
$$

定义三元组运算

$$
d(h,r,t)=||f(\theta_h,w_h)\oplus f(\theta_r,w_r)-f(\theta_t,w_t)||\\
f(\theta_h,w_h)\oplus f(\theta_r,w_r):\\
\begin{cases}
x(\theta_h+\theta_r,w_h+w_r)=(R+r\cos (\frac{\theta_h+\theta_r}{2}+w_h+w_r))\cos (\theta_h+\theta_r)\\
y(\theta_h+\theta_r,w_h+w_r)=(R+r\cos (\frac{\theta_h+\theta_r}{2}+w_h+w_r))\sin (\theta_h+\theta_r)\\
z(\theta_h+\theta_r,w_h+w_r)=r\sin(\frac{\theta_h+\theta_r}{2}+w_h+w_r)\\
\end{cases}\\
d(h,r,t)=\sqrt{||x_{h+r}-x_t||^2+||y_{h+r}-y_t||^2+||y_{h+r}-y_t||^2}
$$

## 与TorusE的参数对比

- TorusE直接以x, y, z为参数
- Mobius以$\theta,w$为参数

## 论文地址

[TorusE](https://arxiv.org/pdf/1711.05435.pdf)

[MobiusE](https://arxiv.org/pdf/2101.02352.pdf)
