import torch

import torch

# --- helpers (放在训练 loop 外面即可) ---
def _clone_grads(params):
    """clone current .grad into list (None -> None)"""
    out = []
    for p in params:
        out.append(None if p.grad is None else p.grad.detach().clone())
    return out

def _dot(g1, g2):
    s = None
    for a, b in zip(g1, g2):
        if a is None or b is None:
            continue
        v = (a * b).sum()
        s = v if s is None else (s + v)
    return s

def _norm2(g):
    s = None
    for a in g:
        if a is None:
            continue
        v = (a * a).sum()
        s = v if s is None else (s + v)
    return s


def _dot_list(a_list, b_list):
    s = None
    for a, b in zip(a_list, b_list):
        if a is None or b is None:
            continue
        v = (a * b).sum()
        s = v if s is None else (s + v)
    return s

def _norm2_list(a_list):
    s = None
    for a in a_list:
        if a is None:
            continue
        v = (a * a).sum()
        s = v if s is None else (s + v)
    return s

def _grad_list(params):
    return [None if p.grad is None else p.grad.detach() for p in params]

def _clone_grad_list(params):
    return [None if p.grad is None else p.grad.detach().clone() for p in params]

@torch.no_grad()
def _get_d(params, theta0, d_ema, use_ema=True, beta=0.97):
    d = [(p.detach() - p0) for p, p0 in zip(params, theta0)]
    if not use_ema:
        return d, d_ema
    if d_ema is None:
        d_ema = [x.clone() for x in d]
    else:
        for i in range(len(d)):
            d_ema[i].mul_(beta).add_(d[i], alpha=(1 - beta))
    return d_ema, d_ema


# @torch.no_grad()
# def _orth_perturb_first_step(optimizer, params, g_u, g_h, rho, eps=1e-12, zero_grad=True, fallback="gh"):
#     """
#     用 g_h 在 g_u 上投影剪掉后的方向 g_perp 做 SAM 扰动：
#       g_perp = g_h - <g_h,g_u> / (||g_u||^2+eps) * g_u
#       w <- w + rho * g_perp / (||g_perp||+eps)

#     这里把 old_p 存进 optimizer.state[p]["old_p"]，后面可直接用 optimizer.restore().
#     fallback:
#       - "gh": 若 g_perp 近似 0，就退化用 g_h
#       - "none": 若 g_perp 近似 0，就不扰动
#     """
#     gu_n2 = _norm2(g_u)
#     if gu_n2 is None or gu_n2.detach().item() == 0.0:
#         g_perp = g_h
#     else:
#         gh_dot_gu = _dot(g_h, g_u)
#         coef = gh_dot_gu / (gu_n2 + eps)
#         g_perp = []
#         for gh_i, gu_i in zip(g_h, g_u):
#             if gh_i is None:
#                 g_perp.append(None)
#             elif gu_i is None:
#                 g_perp.append(gh_i)
#             else:
#                 g_perp.append(gh_i - coef * gu_i)

#     # if g_perp is ~0, fallback
#     gp_n2 = _norm2(g_perp)
#     gp_norm = torch.sqrt((gp_n2 if gp_n2 is not None else torch.tensor(0.0, device=params[0].device)) + eps)
#     if gp_norm.detach().item() == 0.0:
#         if fallback == "gh":
#             g_perp = g_h
#             gp_n2 = _norm2(g_perp)
#             gp_norm = torch.sqrt((gp_n2 if gp_n2 is not None else torch.tensor(0.0, device=params[0].device)) + eps)
#         elif fallback == "none":
#             if zero_grad:
#                 optimizer.zero_grad(set_to_none=True)
#             return

#     scale = rho / (gp_norm + eps)

#     # save and perturb
#     for p, d in zip(params, g_perp):
#         if d is None:
#             continue
#         optimizer.state[p]["old_p"] = p.detach().clone()
#         p.add_(d, alpha=scale)

#     if zero_grad:
#         optimizer.zero_grad(set_to_none=True)


import torch

@torch.no_grad()
def _orth_perturb_first_step(
    optimizer,
    params,
    g_u,
    g_h,
    rho,
    eps=1e-12,
    zero_grad=True,
    fallback="gh",
    proj_scale=1.0,     # ✅ 新增：剪投影比例（0=不剪，1=全剪）
    one_sided=True,    # ✅ 新增：只在 alpha>0 时剪（更保守/稳）
):
    """
    Soft orthogonal SAM first step:
      coef = <g_h,g_u> / (||g_u||^2+eps)
      g_perp = g_h - proj_scale * coef * g_u   (optionally one-sided)
      w <- w + rho * g_perp / (||g_perp||+eps)

    old_p 存进 optimizer.state[p]["old_p"]，后面可直接 optimizer.restore()。

    Returns:
      alpha:     coef 的 float（原始投影系数）
      alpha_eff: proj_scale * alpha（实际剪掉强度；one_sided 不触发时为 0）
    """
    # default return values
    alpha = 0.0
    alpha_eff = 0.0

    gu_n2 = _norm2(g_u)
    if gu_n2 is None or gu_n2.detach().item() == 0.0:
        # no projection basis
        g_perp = g_h
    else:
        gh_dot_gu = _dot(g_h, g_u)
        coef = gh_dot_gu / (gu_n2 + eps)     # tensor scalar
        alpha = float(coef.detach().item())

        # one-sided gate: only remove if it pushes "along" g_u direction
        do_proj = (not one_sided) or (alpha > 0.0)

        if do_proj:
            coef_eff = proj_scale * coef
            alpha_eff = float((proj_scale * coef).detach().item())
            g_perp = []
            for gh_i, gu_i in zip(g_h, g_u):
                if gh_i is None:
                    g_perp.append(None)
                elif gu_i is None:
                    g_perp.append(gh_i)
                else:
                    g_perp.append(gh_i - coef_eff * gu_i)
        else:
            # don't project
            g_perp = g_h
            alpha_eff = 0.0

    # if g_perp is ~0, fallback
    gp_n2 = _norm2(g_perp)
    gp_norm = torch.sqrt(
        (gp_n2 if gp_n2 is not None else torch.tensor(0.0, device=params[0].device)) + eps
    )

    if gp_norm.detach().item() == 0.0:
        if fallback == "gh":
            g_perp = g_h
            gp_n2 = _norm2(g_perp)
            gp_norm = torch.sqrt(
                (gp_n2 if gp_n2 is not None else torch.tensor(0.0, device=params[0].device)) + eps
            )
        elif fallback == "none":
            if zero_grad:
                optimizer.zero_grad(set_to_none=True)
            return alpha, alpha_eff

    scale = rho / (gp_norm + eps)

    # save and perturb
    for p, d in zip(params, g_perp):
        if d is None:
            continue
        optimizer.state[p]["old_p"] = p.detach().clone()
        p.add_(d, alpha=scale)

    if zero_grad:
        optimizer.zero_grad(set_to_none=True)

    return alpha, alpha_eff


class OrthSAM(torch.optim.Optimizer):
    """
    SAM wrapper
      - refusal: SAM step-1 (perturb) + step-2 (grad at perturbed weights)
      - retain: normal grad at original weights
      - final update: ONE base_optimizer.step() using (grad_refusal_step2 + lam_u * grad_retain)

    APIs:
      - first_step(zero_grad=True): perturb params using current grads
      - restore(): restore params to pre-perturb weights (does NOT touch grads)
      - step_base(zero_grad=True): base_optimizer.step()
    """
    def __init__(self, params, base_optimizer_cls, rho=0.05, adaptive=False, eps=1e-12, **base_kwargs):
        if rho < 0:
            raise ValueError("rho must be non-negative")
        self.rho = float(rho)
        self.adaptive = bool(adaptive)
        self.eps = float(eps)

        self.base_optimizer = base_optimizer_cls(params, **base_kwargs)
        defaults = dict(rho=self.rho, adaptive=self.adaptive, eps=self.eps, **base_kwargs)
        super().__init__(self.base_optimizer.param_groups, defaults)

    @torch.no_grad()
    def first_step(self, zero_grad: bool = True):
        grad_norm = self._grad_norm()
        if grad_norm.detach().item() == 0.0:
            if zero_grad:
                self.zero_grad(set_to_none=True)
            return

        scale = self.rho / (grad_norm + self.eps)

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                # save old weights
                self.state[p]["old_p"] = p.detach().clone()

                # g~ (adaptive or not)
                g = p.grad
                if self.adaptive:
                    g = torch.abs(p) * g  # g~ = |w| ⊙ g

                # w <- w + scale * g~
                p.add_(g, alpha=scale)

        if zero_grad:
            self.zero_grad(set_to_none=True)

    @torch.no_grad()
    def restore(self):
        # restore parameters to pre-perturb values, keep grads untouched
        for group in self.param_groups:
            for p in group["params"]:
                old_p = self.state.get(p, {}).get("old_p", None)
                if old_p is None:
                    continue
                p.data.copy_(old_p)

    @torch.no_grad()
    def step_base(self, zero_grad: bool = True):
        # final update step using accumulated grads
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad(set_to_none=True)

    def zero_grad(self, set_to_none=True):
        self.base_optimizer.zero_grad(set_to_none=set_to_none)

    def _grad_norm(self):
        device = None
        sq_sum = None

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                device = p.grad.device if device is None else device

                g = p.grad
                if self.adaptive:
                    g = torch.abs(p) * g

                gn = torch.norm(g, p=2)
                sq = gn * gn
                sq_sum = sq if sq_sum is None else (sq_sum + sq)

        if sq_sum is None:
            return torch.tensor(0.0, device=device or "cpu")
        return torch.sqrt(sq_sum)
