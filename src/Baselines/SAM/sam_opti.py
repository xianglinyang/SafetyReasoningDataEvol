import torch

class SAM(torch.optim.Optimizer):
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
