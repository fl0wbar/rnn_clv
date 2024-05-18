## Compute Lyapunov Exponents and Covariant Lyapunov Vectors of an RNN trajectory


### Usage

```python
B = 2
D_inp = 16
D_hid = 8

num_lyap_exp = 16

n_pushforward_steps = 5
n_transient_steps = 20
n_simulation_steps = 50
n_reorth = 5

DEVICE = device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

rnn_cell_module = LSTM(input_dim = D_inp, hidden_dim = D_hid).to(DEVICE)

lyap_inputs = torch.randn([B, (n_transient_steps + n_simulation_steps), D_inp], dtype=torch.float32, device=DEVICE, requires_grad=True)

trs_inputs = lyap_inputs[:, :n_transient_steps, :]
sim_inputs = lyap_inputs[:, n_transient_steps:(n_transient_steps + n_simulation_steps), :]

print(f"{trs_inputs.shape = }, {sim_inputs.shape = }")

rnn_lyap_info = rnn_lyapunov(
    rnn_cell_step_module = rnn_cell_module.to(DEVICE),
    transience_inputs = trs_inputs.to(DEVICE),
    simulation_inputs = sim_inputs.to(DEVICE),
    num_of_subspaces = num_lyap_exp,  # number of Oseledets subspaces to compute (equal to no. of Lyapunov exponents and covariant Lyapunov vectors that will be computed)
    apply_transience = True,
    apply_pushforward = True,
    num_transient_steps = 200,
    num_pushforward_steps = 10,
    num_simulation_steps = 200,
    reorthonormalization_interval = n_reorth,
    use_id_clv_coeffs_init = True,
    teacher_forcing_alpha = 0.125,
    enable_grads = True,
    verbose = True,
)
```
`rnn_lyap_info` is an AttrDict containing

```python
pushforward_data = AttrDict(
    Q_pushforward = Q_pushforward,
)
lyap_pushforward_data = AttrDict(
    lyap_pushforward = lyap_pushforward,
    lyap_after_pushforward = lyap_after_pushforward,
)
transience_data = AttrDict(
    Q_transience = Q_transience,
    R_transience = R_transience,
    R_diag_transience = R_diag_transience,
)
lyap_transience_data = AttrDict(
    lyap_transience = lyap_transience,
    lyap_after_transience = lyap_after_transience,
)
simulation_data = AttrDict(
    Q_simulation = Q_simul,
    R_simulation = R_simul,
    R_diag_simulation = R_diag_simul,
)
lyap_simulation_data = AttrDict(
    lyap_simulation = lyap_simul,
    lyap_after_simulation = lyap_after_simul,
)
clv_data = AttrDict(
    CLV = CLV,
    Q0 = Q0,
    CLV_coeffs = CLV_coeffs,
    CLV_simulation = CLV_simul,
)

rnn_lyapunov_info = AttrDict(
    pushforward_data = pushforward_data,
    lyap_pushforward_data = lyap_pushforward_data,
    transience_data = transience_data,
    lyap_transience_data = lyap_transience_data,
    simulation_data = simulation_data,
    lyap_simulation_data = lyap_simulation_data,
    clv_data = clv_data,
    total_lyap_exponent = total_lyap_exponent,
)

```
#### Please read the codebase for more details and tricks involved for differentiablity and consistency


### Citations

```
[1] Gary Froyland, Thorsten Hüls, Gary P. Morriss, Thomas M. Watson,
    Computing covariant Lyapunov vectors, Oseledets vectors, and dichotomy projectors: A comparative numerical study,
    Physica D: Nonlinear Phenomena, Volume 247, Issue 1, 2013 (https://arxiv.org/abs/1204.0871)
```
```
[2] F. Ginelli, P. Poggi, A. Turchi, H. Chaté, R. Livi, and A. Politi.
    Characterizing dynamics with covariant Lyapunov vectors.
    Physical Review Letters, 99(13), September 28 2007.
```
```
[3] C. L. Wolfe and R. M. Samelson.
    An efficient method for recovering Lyapunov vectors from singular vectors.
    Tellus: Series A, 59(3):355–366, 2007.
```
```
[4] P. V. Kuptsov and U. Parlitz.
    Theory and computation of covariant Lyapunov vectors.
    http://arxiv.org/pdf/1105.5228v2.pdf, 2011.
```
```
[5] S. V. Ershov and A. B. Potapov.
    On the concept of stationary Lyapunov basis.
    Physica D, 118(3-4):167–198, 1998.
```
```
[6] V. I. Oseledec.
    A multiplicative ergodic theorem. Lyapunov characteristic numbers for dynamical systems.
    Trans. Moscow Math. Soc, 1968.

```
