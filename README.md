# VAE
- 다루기 어려운 사후분포를 가지는 연속 잠재변수를 모델로부터 효율적으로 추론하고 학습시키는 것
  
## Auto-Encoding Variational Bayes
- Variational lower bound의 Reparmeterization이 일반적인 Gradient 방법론들을 사용하여 직접적으로 최적화 될 수 있는 lower bound estimator를 만든다.
- 각 datapoint가 연속형 잠재 변수를 가지는 i.i.d 데이터셋에 대하여, 제안된 lower bound estimator를 사용하여 계산이 불가능한 사후분포를 적합시킴으로써 효율적인 사후추론이 가능해졌다.

- Lower Bound (ELBO)

$$\begin{aligned} 
\log p_{\theta}(x^{(i)})&=E_{z \sim q_{\phi}(z|x^{(i)})}[\log p_{\theta}(x^{(i)})] \\
&=E_{z \sim q_{\phi}(z|x^{(i)})}\bigg[\log\frac{p_{\theta}(x^{(i)}|z)p_{\theta}(z)}{p_{\theta}(z|x^{(i)})}\bigg] \\
&=E_{z \sim q_{\phi}(z|x^{(i)})}\bigg[\log\frac{p_{\theta}(x^{(i)}|z)p_{\theta}(z)}{p_{\theta}(z|x^{(i)})} \frac{q_{\phi}(z|x^{(i)})}{q_{\phi}(z|x^{(i)})}\bigg] \\
&=E_{z \sim q_{\phi}(z|x^{(i)})}[\log p_{\theta}(x^{(i)}|z)] - E_{z \sim q_{\phi}(z|x^{(i)})}\bigg[\log\cfrac{q_{\phi}(z|x^{(i)})}{p_{\theta}(z)}\bigg]+E_{z \sim q_{\phi}(z|x^{(i)})}\bigg[\log\frac{q_{\phi}(z|x^{(i)})}{p_{\theta}(z|x^{(i)})}\bigg] \\
&=E_{z \sim q_{\phi}(z|x^{(i)})}[\log p_{\theta}(x^{(i)}|z)]-\int_{z} q_{\phi}(z|x^{(i)})log \frac{q_{\phi}(z|x^{(i)})}{p_{\theta}(z)}dz + \int_{z} q_{\phi}(z|x^{(i)})\frac{q_{\phi}(z|x^{(i)})}{p_{\theta}(z|x^{(i)})}dz\\
&=E_{z \sim q_{\phi}(z|x^{(i)})}[\log p_{\theta}(x^{(i)}|z)]-D_{KL}(q_{\phi}(z|x^{(i)})||p_{\theta}(z))+D_{KL}(q_{\phi}(z|x^{(i)})||p_{\theta}(z|x^{(i)}))\\
&=L(\theta, \phi : x^{(i)}) + D_{KL}(q_{\phi}(z|x^{(i)})||p_{\theta}(z|x^{(i)}))
\end{aligned}$$


## Semi-supervised Learning with Deep Generative models
<img src = "https://github.com/ImJaeSung/VAE/assets/113405066/5d8e9792-6ba3-40a0-9744-7e3f0c29083c" width = "400" height = "400">
