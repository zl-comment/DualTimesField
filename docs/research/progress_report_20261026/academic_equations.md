$$
\widehat Q_{t,h}(\tau)=f_\theta(I_t,h,\tau),\quad h=0,\ldots,23,\quad \tau\in\{.05,.10,.50,.90,.95\}
$$

$$
\rho_\tau(u)=u\left[\tau-\mathbf{1}(u<0)\right]
$$

$$
\mathrm{MAE}=\operatorname{mean}|y-\widehat q_{.5}|,\qquad \mathrm{QS}=2\operatorname{mean}_{t,h,\tau}\rho_\tau(y-\widehat q_\tau)
$$

$$
\delta=150\tanh(d),\quad s_-=1+0.5\tanh(l),\quad s_+=1+0.5\tanh(u)
$$

$$
A_\tau=\begin{cases}B_{.5}+\delta+s_-(B_\tau-B_{.5}),&\tau<.5,\\ B_{.5}+\delta,&\tau=.5,\\ B_{.5}+\delta+s_+(B_\tau-B_{.5}),&\tau>.5.\end{cases}
$$

$$
\begin{aligned}T(h)&=a_0+a_1\cos(2\pi h/24)+a_2\sin(2\pi h/24),\\ E(h)&=\sum_k a_k\exp\left[-\frac{(h-\mu_k)^2}{2\sigma_k^2}\right].\end{aligned}
$$

$$
\begin{aligned}r_T&=T-\operatorname{LP}(B^{\mathrm{hist}}_{.5}),\\ r_E&=\min\left\{\max(E,0),\max[y^{\mathrm{hist}}-B^{\mathrm{hist}}_{.5}-\max(r_T,0),0]\right\}.\end{aligned}
$$

$$
\Delta_T=\sum_jw_{T,j}(r_{T,j}-\delta),\qquad\Delta_E=\sum_jw_{E,j}\operatorname{Move}(r_{E,j})
$$

$$
R=150\tanh\left(\frac{g_T\Delta_T+g_E\Delta_E}{150}\right),\qquad Q_\tau=A_\tau+R
$$

$$
\begin{aligned}\mathcal{L}={}&\frac{\operatorname{PB}(Q,y)+0.5\operatorname{PB}(A,y)}{300}\\ &+\frac{0.5[\operatorname{PB}_{.5}(A,y)+\operatorname{PB}_{.5}(Q,y)]}{300}\\ &+0.01\operatorname{mean}\left[\left(\frac{A-B}{300}\right)^2\right]+0.05\mathcal{L}_{\mathrm{rec}}+0.001\operatorname{mean}(g).
\end{aligned}
$$

$$
\operatorname{PB}_{.5}=0.5\mathrm{MAE},\qquad Q^{\mathrm{guarded}}=B+\alpha(Q-B)
$$
