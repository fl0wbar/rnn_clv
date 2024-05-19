import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def vprint(verbose, message):
    if verbose:
        print(message)


def lstm_cell_unit(x, hx, cx, w_ih, w_hh, b_ih, b_hh):
    # type: (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]
    gates = torch.matmul(x, w_ih.t()) + torch.matmul(hx, w_hh.t()) + b_ih + b_hh

    ingate, forgetgate, cellgate, outgate = gates.chunk(chunks=4, dim=-1)

    ingate = torch.sigmoid(ingate)
    forgetgate = torch.sigmoid(forgetgate)
    cellgate = torch.tanh(cellgate)
    outgate = torch.sigmoid(outgate)

    cy = (forgetgate * cx) + (ingate * cellgate)
    hy = outgate * torch.tanh(cy)

    return hy, cy


def generate_rnn_jacobian(rnn_module: torch.nn.Module, x: torch.Tensor, state: torch.Tensor):
    # create a function of rnn-cell's step/forward function that is compatible with jacrev's aux functionality
    # (
    #    jacrev uses the first output (output) for differentation and second output (aux) for function's computed output
    #    that is not differentiated by jacrev
    # )
    def jac_compat_cell_step(x, state):
        cell_output, next_state = rnn_module.forward(x, state)
        return next_state, (cell_output, next_state)

    cell_step_jacobian_func = torch.func.vmap(
        torch.func.jacrev(
            func=jac_compat_cell_step,  # compute jacobian of the cell's hidden states for only a single recurrent step
            argnums=1,  # with respect to state tuple (hx, cx) usualy passed at argument position [1]
            has_aux=True,  # cell's step function also generates the output tensor from the forward pass alongside the next recurrent state
            chunk_size=None,
            _preallocate_and_copy=False,
        ),
        in_dims=(0, (0,) * len(state)),  # (x, (hx, cx)) have batch dimension at axis=0
        out_dims=(
            (0,) * len(state),
            (0, (0,) * len(state)),
        ),  # ((J_hy, J_cy), (y, (hy, cy))) have batch dimension at axis=0
        randomness="different",  # use different seeds of randomness over the batch dimension.
        chunk_size=None,
    )
    cell_state_jacobian, (cell_output, cell_next_state) = cell_step_jacobian_func(x, state)
    return cell_state_jacobian, cell_output, cell_next_state


def build_full_jacobian_matrix(recurrent_state_jacobian):
    # build the full jacobian matrix containing interactions between all recurrent-state
    # returns here |J_hy_hx J_cy_hx|
    #              |J_hy_cx J_cy_cx|
    # returns a matrix of size (batch_size, num_recurrent_states*recurrent_feature_dim, num_recurrent_states*recurrent_feature_dim)
    J = []
    for j_col in recurrent_state_jacobian:
        j_col_i = torch.cat([j_row_j for j_row_j in j_col], dim=-1)
        J.append(j_col_i)
    return torch.cat(J, dim=-2)


def step_rnn_jacobian(rnn_module, x, state):
    cell_state_jacobian, cell_output, cell_next_state = generate_rnn_jacobian(rnn_module, x, state)
    jacobian_recurrent_state = build_full_jacobian_matrix(cell_state_jacobian)
    return jacobian_recurrent_state, cell_output, cell_next_state


class LSTM(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        W_ih = torch.zeros([4 * hidden_dim, input_dim], dtype=torch.float32).uniform_(
            -math.sqrt(1 / hidden_dim), math.sqrt(1 / hidden_dim)
        )
        W_hh = torch.zeros([4 * hidden_dim, hidden_dim], dtype=torch.float32).uniform_(
            -math.sqrt(1 / hidden_dim), math.sqrt(1 / hidden_dim)
        )

        b_ih = torch.zeros([4 * hidden_dim], dtype=torch.float32).uniform_(
            -math.sqrt(1 / hidden_dim), math.sqrt(1 / hidden_dim)
        )
        b_hh = torch.zeros([4 * hidden_dim], dtype=torch.float32).uniform_(
            -math.sqrt(1 / hidden_dim), math.sqrt(1 / hidden_dim)
        )

        W_oi = torch.zeros([input_dim, hidden_dim], dtype=torch.float32).uniform_(
            -math.sqrt(1 / input_dim), math.sqrt(1 / input_dim)
        )
        b_oi = torch.zeros([input_dim], dtype=torch.float32)

        self.W_ih = nn.Parameter(W_ih, requires_grad=True)
        self.W_hh = nn.Parameter(W_hh, requires_grad=True)

        self.b_ih = nn.Parameter(b_ih, requires_grad=True)
        self.b_hh = nn.Parameter(b_hh, requires_grad=True)

        self.W_oi = nn.Parameter(W_oi, requires_grad=True)
        self.b_oi = nn.Parameter(b_oi, requires_grad=True)

        self.hx0, self.cx0 = None, None

    def initialize_state(self, batch_size, device=DEVICE):
        hx0 = torch.randn([batch_size, self.hidden_dim], dtype=torch.float32, device=device)
        cx0 = torch.randn([batch_size, self.hidden_dim], dtype=torch.float32, device=device)
        return hx0, cx0

    def forward(self, x, state):
        return self.step(x, state)

    def step(self, x, state):
        # (input, recurrent_state)
        # unpack the inputs and the recurrent state matrices
        if state is None:
            self.hx0, self.cx0 = self.initialize_state(batch_size=x.shape[0], device=x.device)
            state = (self.hx0, self.cx0)
        hx, cx = state
        # compute the next recurrent output and states
        hy, cy = lstm_cell_unit(x=x, hx=hx, cx=cx, w_ih=self.W_ih, w_hh=self.W_hh, b_ih=self.b_ih, b_hh=self.b_hh)
        # set the recurrent state to the next computed state
        state = (hy, cy)
        output = F.linear(hy, self.W_oi, self.b_oi)
        # (next-recurrent_state, output)
        return output, state

    # returns J_hx, J_cx
    def jacobian_cell_step(self, x, state):
        # create a function of cell-step that is compatible with jacrev's aux functionality
        # (
        #    jacrev uses the first output (output) for differentation and second output (aux) for function's computed output
        #    that is not differentiated by jacrev
        # )
        def jac_compat_cell_step(x, state):
            cell_output, next_state = self.step(x, state)
            return next_state, (cell_output, next_state)

        cell_step_jacobian_func = torch.func.vmap(
            torch.func.jacrev(
                func=jac_compat_cell_step,  # compute jacobian of the cell's hidden states for only a single recurrent step
                argnums=1,  # with respect to state tuple (hx, cx) usualy passed at argument position [1]
                has_aux=True,  # cell's step function also generates the output tensor from the forward pass alongside the next recurrent state
                chunk_size=None,
                _preallocate_and_copy=False,
            ),
            in_dims=(0, (0,) * len(state)),  # (x, (hx, cx)) have batch dimension at axis=0
            out_dims=(
                (0,) * len(state),
                (0, (0,) * len(state)),
            ),  # ((J_hy, J_cy), (y, (hy, cy))) have batch dimension at axis=0
            randomness="different",  # use different seeds of randomness over the batch dimension.
            chunk_size=None,
        )
        cell_state_jacobian, (cell_output, cell_next_state) = cell_step_jacobian_func(x, state)
        return cell_state_jacobian, cell_output, cell_next_state

    def build_full_jacobian_matrix(self, recurrent_state_jacobian):
        # build the full jacobian matrix containing interactions between all recurrent-state
        # returns here |J_hy_hx J_cy_hx|
        #              |J_hy_cx J_cy_cx|
        # returns a matrix of size (batch_size, num_recurrent_states*recurrent_feature_dim, num_recurrent_states*recurrent_feature_dim)
        J = []
        for j_col in recurrent_state_jacobian:
            j_col_i = torch.cat([j_row_j for j_row_j in j_col], dim=-1)
            J.append(j_col_i)
        return torch.cat(J, dim=-2)

    def step_jacobian(self, x, state):
        cell_state_jacobian, cell_output, cell_next_state = self.jacobian_cell_step(x, state)
        J_recurrent_state = self.build_full_jacobian_matrix(cell_state_jacobian)
        return J_recurrent_state, cell_output, cell_next_state


def safe_log(x: torch.Tensor, **kwargs) -> torch.Tensor:
    return torch.log(x.clip(min=torch.finfo(x.dtype).tiny), **kwargs)


def safe_norm(
    x: torch.Tensor,
    min_norm: float,
    ord: Optional[Union[int, float, str]] = None,
    dim: Union[None, tuple[int, ...], int] = None,
    keepdim: bool = False,
) -> torch.Tensor:
    """Returns torch.maximum(torch.linalg.norm(x), min_norm) with correct gradients.

    The gradients of `torch.maximum(torch.linalg.norm(x), min_norm)` at 0.0 is `NaN`,
    because torch will evaluate both branches of the `torch.maximum`. This function
    will instead return the correct gradient of 0.0 also in such setting.

    Args:
      x: torch Tensor.
      min_norm: lower bound for the returned norm.
      ord: {non-zero int, inf, -inf, ‘fro’, ‘nuc’}, optional. Order of the norm.
        inf means numpy’s inf object. The default is None.
      dim: {None, int, 2-tuple of ints}, optional. If dim is an integer, it
        specifies the dim of x along which to compute the vector norms. If dim
        is a 2-tuple, it specifies the axes that hold 2-D matrices, and the matrix
        norms of these matrices are computed. If dim is None then either a vector
        norm (when x is 1-D) or a matrix norm (when x is 2-D) is returned. The
        default is None.
      keepdim: bool, optional. If this is set to True, the axes which are normed
        over are left in the result as dimensions with size one. With this option
        the result will broadcast correctly against the original x.

    Returns:
      The safe norm of the input vector, accounting for correct gradient.
    """
    norm = torch.linalg.norm(x, ord=ord, dim=dim, keepdim=True)
    x = torch.where(norm <= min_norm, torch.ones_like(x), x)
    norm = torch.squeeze(norm, dim=dim) if not keepdim else norm
    masked_norm = torch.linalg.norm(x, ord=ord, dim=dim, keepdim=keepdim)
    return torch.where(norm <= min_norm, min_norm, masked_norm)


def safe_normalize(
    x: torch.Tensor,
    min_norm: float = 1e-12,
    ord: Optional[Union[int, float, str]] = None,
    dim: Union[None, tuple[int, ...], int] = None,
) -> torch.Tensor:
    return x / safe_norm(x=x, min_norm=min_norm, ord=ord, dim=dim, keepdim=True)


def svd_flip(u: torch.Tensor, v: torch.Tensor, u_based_decision: bool = True):
    """
    sign(real)/phase(cplx) correction to ensure deterministic output from SVD.
    Impose fixed phase convention on left- and right-singular vectors.

    Given a set of left- and right-singular vectors as the columns of u
    and rows of vh, respectively, imposes the phase(cplx)/sign(real) convention
    that for each left-singular vector, the element with largest absolute value
    is real and positive.

    Adjusts the columns of u and the rows of v such that the loadings in the
    columns in u that are largest in absolute value are always positive.

    If u_based_decision is False, then the same sign correction is applied to
    so that the rows in v that are largest in absolute value are always
    positive.

    Parameters
    ----------
    u : torch.Tensor, shape (M, K)
        Unitary array containing the left-singular vectors as columns.
        Parameters u and v are the output of `linalg.svd` with matching inner
        dimensions so one can compute `torch.dot(u * s, v)`.
        u can be None if `u_based_decision` is False.

    v : torch.Tensor, shape (K, N)
        Unitary array containing the right-singular vectors as rows.
        Parameters u and v are the output of `linalg.svd` with matching inner
        dimensions so one can compute `torch.dot(u * s, v)`.
        v can be None if `u_based_decision` is True.

    u_based_decision : bool, default=True
        If True, use the columns of u as the basis for sign flipping.
        Otherwise, use the rows of v. The choice of which variable to base the
        decision on is generally algorithm dependent.

    Returns
    -------
    u_adjusted : torch.Tensor, shape (M, K)
        Array u with adjusted columns and the same dimensions as u.
        Unitary array containing the left-singular vectors as columns,
        conforming to the chosen phase convention.

    v_adjusted : torch.Tensor, shape (K, N)
        Array v with adjusted rows and the same dimensions as v.
        Unitary array containing the right-singular vectors as rows,
        conforming to the chosen phase convention.

    References:
        0.  https://stats.stackexchange.com/questions/34396/im-getting-jumpy-loadings-in-rollapply-pca-in-r-can-i-fix-it
        1.  https://github.com/scikit-learn/scikit-learn/blob/5e5cc3477025794c5b2ee6056a223d91adbfe925/sklearn/utils/extmath.py#L851
        2.  https://gist.github.com/David-Willo/1825bf9e8c30e13147e332734bcaebd5 (SVD with corrected signs)
        3.  Bro, R., Acar, E., & Kolda, T. G. (2008). Resolving the sign ambiguity in the singular value decomposition.
            Journal of Chemometrics: A Journal of the Chemometrics Society, 22(2), 135-140.
            URL: https://prod-ng.sandia.gov/techlib-noauth/access-control.cgi/2007/076422.pdf

    """
    if u_based_decision:
        # columns of u, rows of v, or equivalently rows of u.T and v
        max_abs_u_rows = torch.argmax(torch.abs(u), dim=-2)
        u_cols = torch.arange(u.shape[1], device=u.device)
        # use torch.sgn for compatibility with both reals and complex tensors
        signs = torch.sgn(u[..., max_abs_u_rows, u_cols])
        u = u * signs.unsqueeze(-2)
        if v is not None:
            v = v * signs.unsqueeze(-1)
    else:
        # rows of v, columns of u
        max_abs_v_cols = torch.argmax(torch.abs(v), dim=-1)
        v_rows = torch.arange(v.shape[0], device=v.device)
        # use torch.sgn for compatibility with both reals and complex tensors
        signs = torch.sgn(v[..., v_rows, max_abs_v_cols])
        if u is not None:
            u = u * signs.unsqueeze(-2)
        v = v * signs.unsqueeze(-1)
    return u, v


def compute_truncated_svd(A: torch.Tensor, k: int):
    max_subspaces = min(A.shape[-2], A.shape[-1])
    U, S, Vh = torch.linalg.svd(A, full_matrices=False)
    num_singular_values = S.shape[-1]
    if k < max_subspaces:
        U = U[..., :, : min(k, num_singular_values)]
        if k < num_singular_values:
            S = S[..., :k]
        Vh = Vh[..., : min(k, num_singular_values), :]
    # Impose a fixed phase convention on the singular vectors
    # to avoid phase ambiguity.
    U, Vh = svd_flip(U, Vh, u_based_decision=True)
    return U, S, Vh


def flip_signs_QR(Q: torch.Tensor, R: torch.Tensor, dim1: int = -2, dim2: int = -1):
    # the QR factorization is not unique.
    # Most codes for computing the QR factorization don't worry about this and let the diagonal elements of R have arbitrary signs
    # So, Negate columns of Q where diagonal elements of R are negative
    # ref : https://math.stackexchange.com/questions/2237262/is-there-a-correct-qr-factorization-result
    # ref : https://stackoverflow.com/questions/36637322/qr-decomposition-in-r-forcing-a-positive-diagonal
    R_diag = R.diagonal(offset=0, dim1=dim1, dim2=dim2).diag_embed(offset=0, dim1=dim1, dim2=dim2)
    R_sign = torch.sign(R_diag)
    Q = torch.matmul(Q, R_sign)
    R = torch.matmul(R_sign, R)
    return Q, R


def qr_accurate(A: torch.Tensor, dim1: int = -2, dim2: int = -1):
    Q, R = torch.linalg.qr(A, mode="reduced")  # QR decomposition
    Q, R = flip_signs_QR(Q=Q, R=R, dim1=dim1, dim2=dim2)
    return Q, R


def rnn_lyapunov(
    rnn_cell_step_module: nn.Module,  # nn.Module of a rnn-cell implementing a step and a step_jacobian functions/methods
    transience_inputs: torch.Tensor,  # torch.Tensor inputs for the transcience stage iterations of the rnn
    simulation_inputs: torch.Tensor,  # torch.Tensor inputs for the simulation stage iterations of the rnn
    num_of_subspaces: int = 3,  # number of Oseledets subspaces to compute (equal to no. of Lyapunov exponents and covariant Lyapunov vectors that will be computed)
    apply_pushforward: bool = False,
    apply_transience: bool = True,
    num_pushforward_steps: int = 10,
    num_transient_steps: int = 200,
    num_simulation_steps: int = 200,
    reorthonormalization_interval: int = 2,
    use_id_clv_coeffs_init: bool = True,
    teacher_forcing_alpha: float = 0.125,
    rng_seed: int = 42,
    verbose: bool = False,
    **kwargs,
):
    """
    Compute Lyapunov exponents/characteristic exponents (LE/LCE) and Covariant-Lyapunov-Vectors (CLV) of a Recurrent-Neural-Network (RNN)

    = Covariant vectors, Lyapunov vectors, or Oseledets vectors are used for model analyses of systems like
        partial differential equations, nonautonomous differentiable dynamical systems, random dynamical systems etc.
    = These vectors identify spatially varying directions of specific asymptotic growth rates and obey equivariance principles.
    = This method uses an improved algorithm based on Ginelli Scheme (Algorithm 4.5 from Froyland2013).
    = The Ginelli Scheme was first presented by Ginelli et al. in [2] as a method for accurately computing the covariant Lyapunov vectors
        of an orbit of an invertible differentiable dynamical system where the A(x) = DT(x) are the Jacobian matrices of the flow or map.
        1.  Estimates of the Wj(x) (CLV's) are found by constructing equivariant subspaces Sj(x) = W1(x) ⊕ · · · ⊕ Wj(x) and
            filtering the invariant directions contained therein using a power method on the inverse system restricted to the subspaces Sj(x).
        2.  To construct the subspaces Sj(x) we utilise the notion of the stationary Lyapunov basis.
        3.  Choose j orthonormal vectors s1(T^−n x), s2(T^−n x), . . . , sj(T^−n x), n ≥ 1, such that si(T^−n x) ∈/ Vj+1(T^−n x) for 1 ≤ i ≤ j
            and construct s̃i(x) = A(T^−n x, n) @ si(T^−n x), i = 1, . . . , j.
        4.  Using the Gram-Schmidt procedure, construct the orthonormal basis {s1(n)(x), . . . , sj(n)(x)} from {s̃1(-n)(x), . . . , s̃j(-n)(x)},
        5.  Then as n → ∞ the basis {s1(∞)(x), . . . , sj(∞)(x)} converges to a set of orthonormal vectors
            {s1(n)(x), . . . , sj(n)(x)} which span the j fastest expanding directions of the cocycle A
        6.  the QR-decomposition is equivalent to the Gram-Schmidt orthonormalisation that defines the stationary Lyapunov bases.
            The columns of Q(T(n) x) form the stationary Lyapunov basis at T(n)x.
            We have chosen the above notation R(x, n) specifically since, defined in this way, R forms a cocycle which is the restriction of A to the invariant subspaces Sj.
        7.  c(x) is the vector of coefficients of the Lyapunov/Oseledets-vector/CLV of the cocycle A
        8.  We may approximate c(x) numerically using a simple power method on the inverse cocycle R^(−1) (which exists since λ1 > λ2 > ... > −∞).
        9.  Wj(x) = Qj(x)*cj(x)


    References
    ----------
    [1] Gary Froyland, Thorsten Hüls, Gary P. Morriss, Thomas M. Watson,
        Computing covariant Lyapunov vectors, Oseledets vectors, and dichotomy projectors: A comparative numerical study,
        Physica D: Nonlinear Phenomena, Volume 247, Issue 1, 2013 (https://arxiv.org/abs/1204.0871)
    [2] F. Ginelli, P. Poggi, A. Turchi, H. Chaté, R. Livi, and A. Politi.
        Characterizing dynamics with covariant Lyapunov vectors.
        Physical Review Letters, 99(13), September 28 2007.
    [3] C. L. Wolfe and R. M. Samelson.
        An efficient method for recovering Lyapunov vectors from singular vectors.
        Tellus: Series A, 59(3):355–366, 2007.
    [4] P. V. Kuptsov and U. Parlitz.
        Theory and computation of covariant Lyapunov vectors.
        http://arxiv.org/pdf/1105.5228v2.pdf, 2011.
    [5] S. V. Ershov and A. B. Potapov.
        On the concept of stationary Lyapunov basis.
        Physica D, 118(3-4):167–198, 1998.
    [6] V. I. Oseledec.
        A multiplicative ergodic theorem. Lyapunov characteristic numbers for dynamical systems.
        Trans. Moscow Math. Soc, 1968.

    Note:
        For maintaining stability and differentiablity, lyapunov exponent computation uses
        LCE = safe_log(torch.abs(R_diag) + (R_diag == 0).float())
        instead of LCE = safe_log(torch.abs(R_diag.nonzero(as_tuple=True)))
        which only generates lyapunov exponents of nonzero elements of R's diagonal
        instead of replacing them with ones which results in generating zero as the lyapunov exponent after taking the logarithm
    """
    transience_bs = transience_inputs.shape[0]  # batch_size of transcience inputs
    simulation_bs = simulation_inputs.shape[0]  # batch_size of simulation inputs
    assert (
        transience_bs == simulation_bs
    ), f"Provide the computation with same batch-sizes for transience and simulation so that hidden states can be transferred from transience to simulation"

    transience_input_seq_len = transience_inputs.shape[1]  # sequence-length of transcience inputs
    simulation_input_seq_len = simulation_inputs.shape[1]  # sequence-length of simulation inputs

    assert transience_inputs.shape[-1] == simulation_inputs.shape[-1]

    device = transience_inputs.device

    # initialize the states of RNN
    rnn_state0 = rnn_cell_step_module.initialize_state(batch_size=transience_bs, device=device)

    num_recurrent_states = len(rnn_state0)
    recurrent_state_feature_size = rnn_state0[0].shape[-1]
    # Jacobian would be computed by assuming all the recurrent-states have the same feature_dim
    # resulting in a square matrix, Jacobian shape = [batch_size, jacobian_size, jacobian_size]
    jacobian_size = num_recurrent_states * recurrent_state_feature_size

    num_simulation_steps = min(simulation_input_seq_len, num_simulation_steps)
    num_transient_steps = min(transience_input_seq_len, num_transient_steps)
    assert num_simulation_steps > 0
    # ensure num_pushforward_steps are few as it can become inaccurate for larger iterations
    # numerical issues with this approach remain. They stem
    # primarily from the long iterative multiplications involved in building the Jacobian
    # This results in Jacobian becoming too singular and poorly approximates sj(∞)(x) which is the purpose of doing this step
    num_pushforward_steps = min(20, num_transient_steps, num_pushforward_steps)

    assert num_transient_steps > 0
    assert num_simulation_steps > 0
    assert num_pushforward_steps > 0

    # set alpha used for generalized teacher forcing using during rnn-cell iterations
    tf_alpha = float(max(0.0, teacher_forcing_alpha))

    Q = None
    J = None
    total_lyap_exponent = torch.zeros([num_of_subspaces], dtype=torch.float32, device=device)

    nREORTH = max(1, int(reorthonormalization_interval))

    # LCE stands for Lyapunov Charateristic Exponents.
    # CLV stands for Covariant Lyapunov Vectors

    Q_pushforward, lyap_pushforward, lyap_after_pushforward = (None,) * 3
    if (num_pushforward_steps > 0) or apply_pushforward:
        vprint(
            verbose,
            f"| ::: RNN-Lyap | Applying push-forward to compute left-singular vectors from Jacobian, {num_pushforward_steps = }",
        )
        # for better approximation apply push-forward over the final input of transient stage
        # and compute a SVD of the resultant Jacobian
        # implements the improved step from Algorithm 4.5 from Froyland2013 paper
        # apply push-forward on the first input from the transcience stage
        rnn_state_pushforward = rnn_state0
        rnn_pushforward_input = transience_inputs[:, 0, ...]
        # num. of local lyapunov exponents computed = jacobian_size
        lyap_pushforward = torch.zeros([num_pushforward_steps, jacobian_size], dtype=torch.float32, device=device)
        for i in range(0, num_pushforward_steps, 1):
            J, pushforward_output, rnn_state_pushforward = rnn_cell_step_module.step_jacobian(
                rnn_pushforward_input, rnn_state_pushforward
            )
            rnn_pushforward_input = pushforward_output
            # The sum of the Lyapunov exponents must be equal to the sum of the real-part of eigenvalues of the jacobian (local-LCE)
            # this is a better approximation for systems with relatively simple dynamics
            # or when the Jacobian matrix does not vary significantly along the trajectory.
            LCE = torch.linalg.eigvals(J.mean(dim=0)).real
            # total_lyap_exponent = total_lyap_exponent + LCE[..., :num_of_subspaces]
            lyap_pushforward[i] = LCE

        # total_lyap_exponent = total_lyap_exponent / num_pushforward_steps
        lyap_pushforward = lyap_pushforward / num_pushforward_steps
        lyap_after_pushforward = lyap_pushforward[-1]

        # compute the left-singular vectors using SVD of the push-forwarded Jacobian
        # to approximate stationary Lyapunov basis at transcience of infinite-steps
        Q = compute_truncated_svd(J.mean(dim=0), k=num_of_subspaces)[0]  # return only left-singular vectors

    else:
        # initialize the initial orthonormal basis
        Q = torch.rand([jacobian_size, num_of_subspaces], dtype=torch.float32, device=device, requires_grad=False)
        Q = qr_accurate(Q)[0]
        Q = safe_normalize(Q, min_norm=1e-12, dim=0)

    Q_transience, R_transience, R_diag_transience, lyap_transience, lyap_after_transience = (None,) * 5
    rnn_state_transience = None
    if (num_transient_steps > 0) or apply_transience:
        vprint(
            verbose,
            f"| ::: RNN-Lyap | Transience approximation of a stationary Lyapunov basis, {num_transient_steps = }",
        )

        # run the rnn upto a state of transience for constructing the approximation of a stationary Lyapunov basis
        # if apply_transience:
        qr_state_len = int(num_transient_steps / nREORTH)
        qr_state_idx = 0
        Q_transience = np.nan * torch.empty(
            [qr_state_len, jacobian_size, num_of_subspaces], dtype=torch.float32, device=device
        )
        R_transience = np.nan * torch.empty(
            [qr_state_len, num_of_subspaces, num_of_subspaces], dtype=torch.float32, device=device
        )
        R_diag_transience = np.nan * torch.empty([qr_state_len, num_of_subspaces], dtype=torch.float32, device=device)
        lyap_transience = torch.zeros([qr_state_len, num_of_subspaces], dtype=torch.float32, device=device)

        rnn_state_transience = rnn_state0
        transient_output = 0.0
        for i in range(0, num_transient_steps, 1):
            # transient_input = torch.randn_like(sample_rnn_input)
            if i == 0:
                transient_input = transience_inputs[:, 0, ...]
            else:
                transient_input = (
                    tf_alpha * transience_inputs[:, i, ...] + (1 - tf_alpha) * transient_output
                )  # generalized teacher-forcing
            J, transient_output, rnn_state_transience = rnn_cell_step_module.step_jacobian(
                transient_input, rnn_state_transience
            )
            # if transient_output.shape == transience_inputs[:, i+1, ...]:
            if int((i + 1) % nREORTH) == 0:
                # vprint(verbose, f"{i = }, {qr_state_idx = }")
                # evolve the initially orthonormal system Q in the tangent space along the trajectory using the Jacobian
                Q = torch.matmul(J.mean(dim=0), Q)  # [J, J] @ [J, k] -> [J, k]
                Q, R = qr_accurate(Q)  # Q - [J, k], R - [k, k]
                # compute lyapunov exponent during the transience
                R_diag = torch.diagonal(R, offset=0)  # R_diag - [k,]
                LCE = safe_log(torch.abs(R_diag) + (R_diag == 0).float())  # LCE - [k,]
                total_lyap_exponent = total_lyap_exponent + LCE  # LCE - [k,]
                lyap_transience[qr_state_idx] = LCE  # LCE - [k,]

                Q_transience[qr_state_idx] = Q
                R_transience[qr_state_idx] = R
                R_diag_transience[qr_state_idx] = R_diag

                qr_state_idx += 1

        total_lyap_exponent = total_lyap_exponent / num_transient_steps
        lyap_transience = lyap_transience / num_transient_steps
        lyap_after_transience = lyap_transience[-1]

    # after transience, Q converges to a set of orthonormal vectors
    # which span the k fastest expanding directions of the cocycle
    Q0 = Q

    # now construct the cocyle R over the forward simulation steps in R_simul as [R(x) | R(T x) | · · · | R(T N x)].
    # if apply_simulation
    qr_state_len = num_forward_steps = int(num_simulation_steps / nREORTH)
    qr_state_idx = 0

    vprint(
        verbose,
        f"| ::: RNN-Lyap | construct the cocyle R over the forward simulation steps, {num_simulation_steps = }, {num_forward_steps = }, {reorthonormalization_interval = }",
    )

    Q_simul = np.nan * torch.empty([qr_state_len, jacobian_size, num_of_subspaces], dtype=torch.float32, device=device)
    R_simul = np.nan * torch.empty(
        [qr_state_len, num_of_subspaces, num_of_subspaces], dtype=torch.float32, device=device
    )
    R_diag_simul = np.nan * torch.empty([qr_state_len, num_of_subspaces], dtype=torch.float32, device=device)
    lyap_simul = torch.zeros([qr_state_len, num_of_subspaces], dtype=torch.float32, device=device)

    rnn_state_simul = rnn_state_transience if rnn_state_transience is not None else rnn_state0
    simul_output = 0.0
    for i in range(0, num_simulation_steps, 1):
        # simul_input = torch.randn_like(sample_rnn_input)
        if i == 0:
            simul_input = simulation_inputs[:, 0, ...]
        else:
            simul_input = (
                tf_alpha * simulation_inputs[:, i, ...] + (1 - tf_alpha) * simul_output
            )  # generalized teacher-forcing
        J, simul_output, rnn_state_simul = rnn_cell_step_module.step_jacobian(simul_input, rnn_state_simul)
        if int((i + 1) % nREORTH) == 0:
            # vprint(verbose, f"{i = }, {qr_state_idx = }")
            # evolve the orthonormal system Q in the tangent space along the trajectory using the Jacobian
            Q = torch.matmul(J.mean(dim=0), Q)  # [J, J] @ [J, k] -> [J, k]
            Q, R = qr_accurate(Q)  # Q - [J, k], R - [k, k]
            # compute lyapunov exponent during the transience
            R_diag = torch.diagonal(R, offset=0)  # R_diag - [k,]
            LCE = safe_log(torch.abs(R_diag) + (R_diag == 0).float())  # LCE - [k,]
            total_lyap_exponent = total_lyap_exponent + LCE
            lyap_simul[qr_state_idx] = LCE

            Q_simul[qr_state_idx] = Q
            R_simul[qr_state_idx] = R
            R_diag_simul[qr_state_idx] = R_diag

            qr_state_idx += 1

    total_lyap_exponent = total_lyap_exponent / num_simulation_steps
    lyap_simul = lyap_simul / num_simulation_steps
    lyap_after_simul = lyap_simul[-1]

    # use the inverse iteration method to approximate CLV-coefficients
    # compute a simple power method on inverse(Rj) to find the coefficient vector c, which represents the
    # approximation of CLV in the basis s1(x), . . . , sj(x) . Thus, the approximation of CLV's is given by Q(x)*c
    num_backward_steps = R_simul.shape[0]
    vprint(
        verbose,
        f"| ::: RNN-Lyap | initialize CLV-coefficients and use the backwards power method on inverse of the cocycle[R] to approximate CLV-coefficients, {num_backward_steps = }",
    )

    if use_id_clv_coeffs_init:
        CLV_coeffs = torch.eye(num_of_subspaces, dtype=torch.float32, device=device)  # CLV_coeffs - [k, k]
    else:
        CLV_coeffs = torch.tensor([0] * (num_of_subspaces - 1) + [1], dtype=torch.float32, device=device)  # - [k,]
        CLV_coeffs = torch.diag_embed(CLV_coeffs, offset=0)  # CLV_coeffs - [k, k]

    CLV_simul = np.nan * torch.empty(
        [num_backward_steps, jacobian_size, num_of_subspaces], dtype=torch.float32, device=device
    )  # CLV_simul - [N, J, k]

    for i in range((num_backward_steps - 1), -1, -1):
        R = R_simul[i]  # R - [k, k]
        # compute CLV_coeffs = R.inverse() @ CLV_coeffs
        CLV_coeffs = torch.linalg.solve_triangular(R, CLV_coeffs, upper=True, left=True)  # [k, k], [k, k] -> [k, k]
        # normalize the coefficients
        CLV_coeffs = safe_normalize(CLV_coeffs, min_norm=1e-12, dim=(0, 1))  # [k, k]
        # compute varying CLV during orthonormal system's simulation trajectory
        CLV_simul[i] = torch.matmul(Q_simul[i], CLV_coeffs)  # [J, k] @ [k, k] -> [J, k]

    CLV = torch.matmul(Q0, CLV_coeffs)

    pushforward_data = AttrDict(
        Q_pushforward=Q_pushforward,
    )
    lyap_pushforward_data = AttrDict(
        lyap_pushforward=lyap_pushforward,
        lyap_after_pushforward=lyap_after_pushforward,
    )
    transience_data = AttrDict(
        Q_transience=Q_transience,
        R_transience=R_transience,
        R_diag_transience=R_diag_transience,
    )
    lyap_transience_data = AttrDict(
        lyap_transience=lyap_transience,
        lyap_after_transience=lyap_after_transience,
    )
    simulation_data = AttrDict(
        Q_simulation=Q_simul,
        R_simulation=R_simul,
        R_diag_simulation=R_diag_simul,
    )
    lyap_simulation_data = AttrDict(
        lyap_simulation=lyap_simul,
        lyap_after_simulation=lyap_after_simul,
    )
    clv_data = AttrDict(
        CLV=CLV,
        Q0=Q0,
        CLV_coeffs=CLV_coeffs,
        CLV_simulation=CLV_simul,
    )

    rnn_lyapunov_info = AttrDict(
        pushforward_data=pushforward_data,
        lyap_pushforward_data=lyap_pushforward_data,
        transience_data=transience_data,
        lyap_transience_data=lyap_transience_data,
        simulation_data=simulation_data,
        lyap_simulation_data=lyap_simulation_data,
        clv_data=clv_data,
        total_lyap_exponent=total_lyap_exponent,
    )

    return rnn_lyapunov_info


# ======================================================================================================================


# from : https://gist.github.com/sashgorokhov/2eb43dbed07548253f6ddce7c459ecb0
def parse_d(d):
    defaults = dict()
    for key in list(d.keys()):
        if key.startswith("_"):
            continue
        value = d.get(key)
        if callable(value) or isinstance(value, property) or isinstance(value, classmethod):
            continue
        defaults[key] = d.pop(key)
    return defaults


class AttrDictMeta(type):
    def __new__(mcs, name, bases, d):
        defaults = dict()
        for base in bases:
            defaults.update(getattr(base, "__defaults__", dict()))
        defaults.update(parse_d(d))
        instance = super(AttrDictMeta, mcs).__new__(mcs, name, bases, d)
        instance.__defaults__ = defaults
        return instance


class AttrDict(dict, metaclass=AttrDictMeta):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        for k, v in getattr(self, "__defaults__", dict()).items():
            if k not in self:
                self[k] = v

    def __setattr__(self, key, value):
        self[key] = value

    def __getattr__(self, item):
        return self[item]

    def __getstate__(self):
        return dict(self)
