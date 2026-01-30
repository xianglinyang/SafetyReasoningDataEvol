import torch

def set_projected_grads(trainable_params, gh_list, gu_list, eps=1e-12):
    # alpha = <gh, gu> / (||gu||^2 + eps)
    dot = None
    norm2 = None
    for gh, gu in zip(gh_list, gu_list):
        if gh is None or gu is None:
            continue
        v = (gh * gu).sum()
        dot = v if dot is None else (dot + v)
        u2 = (gu * gu).sum()
        norm2 = u2 if norm2 is None else (norm2 + u2)

    if dot is None or norm2 is None:
        alpha = 0.0
    else:
        alpha = (dot / (norm2 + eps)).item()

    # write projected grads into .grad
    for p, gh, gu in zip(trainable_params, gh_list, gu_list):
        if gh is None and gu is None:
            p.grad = None
        elif gu is None:
            p.grad = gh.to(p.dtype)
        elif gh is None:
            p.grad = (-alpha * gu).to(p.dtype)
        else:
            p.grad = (gh - alpha * gu).to(p.dtype)

    return alpha



class ORT(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer_cls, rho=0.05, adaptive=False, **base_kwargs):
        if rho < 0:
            raise ValueError("rho must be non-negative")
        self.rho = float(rho)
        self.adaptive = bool(adaptive)
        self.base_optimizer = base_optimizer_cls(params, **base_kwargs)
        defaults = dict(rho=rho, adaptive=adaptive, **base_kwargs)
        super().__init__(self.base_optimizer.param_groups, defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=True):
        grad_norm = self._grad_norm()
        if grad_norm == 0:
            return
        scale = self.rho / (grad_norm + 1e-12)

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                self.state[p]["old_p"] = p.data.clone()
                if self.adaptive:
                    e = torch.abs(p.data) * p.grad
                    p.add_(e, alpha=scale)
                else:
                    p.add_(p.grad, alpha=scale)

        if zero_grad:
            self.zero_grad(set_to_none=True)

    @torch.no_grad()
    def second_step(self, zero_grad=True):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.data = self.state[p]["old_p"]
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad(set_to_none=True)

    def zero_grad(self, set_to_none=True):
        self.base_optimizer.zero_grad(set_to_none=set_to_none)

    def _grad_norm(self):
        device = None
        norms = []
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                device = p.grad.device if device is None else device
                g = (torch.abs(p.data) * p.grad) if self.adaptive else p.grad
                norms.append(torch.norm(g, p=2))
        if not norms:
            return torch.tensor(0.0, device=device or "cpu")
        return torch.norm(torch.stack(norms), p=2)
