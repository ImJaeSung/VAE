# VAE
- 다루기 어려운 사후분포를 가지는 연속 잠재변수를 모델로부터 효율적으로 추론하고 학습시키는 것
  
## Auto-Encoding Variational Bayes
- Variational lower bound의 Reparmeterization이 일반적인 Gradient 방법론들을 사용하여 직접적으로 최적화 될 수 있는 lower bound estimator를 만든다.
- 각 datapoint가 연속형 잠재 변수를 가지는 i.i.d 데이터셋에 대하여, 제안된 lower bound estimator를 사용하여 계산이 불가능한 사후분포를 적합시킴으로써 효율적인 사후추론이 가능해졌다.

- Lower Bound (ELBO)

$$\begin{aligned} 
\log p(\mathbf{x};\theta)&=&\mathbb{E}_{q(\mathbf{z}|\mathbf{x};\phi)}[\log p(\mathbf{x};\theta)] \nonumber \\
&=&\mathbb{E}_{q(\mathbf{z}|\mathbf{x};\phi)}\bigg[\log \cfrac{p(\mathbf{x}|\mathbf{z};\theta)p(\mathbf{z})}{p(\mathbf{z}|\mathbf{x};\theta)}\bigg] \nonumber \\
&=&\mathbb{E}_{q(\mathbf{z}|\mathbf{x};\phi)}\bigg[\log \cfrac{p(\mathbf{x}|\mathbf{z};\theta)p(\mathbf{z})}{p(\mathbf{z}|\mathbf{x};\theta)}\ \cfrac{q(\mathbf{z}|\mathbf{x};\phi)}{q(\mathbf{z}|\mathbf{x};\phi)}\bigg] \nonumber \\
&=&\mathbb{E}_{q(\mathbf{z}|\mathbf{x};\phi)}[\log p(\mathbf{x}|\mathbf{z};\theta)] - \mathbb{E}_{q(\mathbf{z}|\mathbf{x};\phi)}\bigg[\log \cfrac{q(\mathbf{z}|\mathbf{x};\phi)}{p(\mathbf{z})}\bigg]+\mathbb{E}_{q(\mathbf{z}|\mathbf{x};\phi)}\bigg[\log \cfrac{q(\mathbf{z}|\mathbf{x};\phi)}{p(\mathbf{z}|\mathbf{x};\theta)}\bigg] \nonumber \\
&=&\mathbb{E}_{q(\mathbf{z}|\mathbf{x};\phi)}[\log p(\mathbf{x}|\mathbf{z};\theta)]-\int_{\mathbf{z}} q(\mathbf{z}|\mathbf{x};\phi)\log \cfrac{q(\mathbf{z}|\mathbf{x};\phi)}{p(\mathbf{z})}dz + \int_{\mathbf{z}} q(\mathbf{z}|\mathbf{x};\phi)\cfrac{q(\mathbf{z}|\mathbf{x};\phi)}{p(\mathbf{z}|\mathbf{x};\theta)}dz \nonumber\\
&=& \mathbb{E}_{q(\mathbf{z}|\mathbf{x};\phi)}[\log p(\mathbf{x}|\mathbf{z};\theta)]-D_{\mathcal{KL}}(q(\mathbf{z}|\mathbf{x};\phi)||p(\mathbf{z}))+D_{\mathcal{KL}}(q(\mathbf{z}|\mathbf{x};\phi)||p(\mathbf{z}|\mathbf{x};\theta))\nonumber \\
&=& \mathcal{L}(\theta, \phi ; \mathbf{x}) + D_{\mathcal{KL}}(q(\mathbf{z}|\mathbf{x};\phi)||p(\mathbf{z}|\mathbf{x};\theta)) \nonumber
\end{aligned}$$

## Semi-supervised Learning with Deep Generative models
