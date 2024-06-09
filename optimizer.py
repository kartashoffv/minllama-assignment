from typing import Callable, Iterable, Tuple

import torch
from torch.optim import Optimizer


class AdamW(Optimizer):
    def __init__(
        self,
        params: Iterable[torch.nn.parameter.Parameter],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-6,
        weight_decay: float = 0.0,
        correct_bias: bool = True,
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(
                "Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0])
            )
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(
                "Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1])
            )
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            correct_bias=correct_bias,
        )
        super().__init__(params, defaults)

    def step(self, closure: Callable = None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError(
                        "Adam does not support sparse gradients, please consider SparseAdam instead"
                    )
                # State should be stored in this dictionary
                state = self.state[p]

                # Access hyperparameters from the `group` dictionary
                alpha = group["lr"]
                # Update parameters. Update first and second moments of the gradients
                if "step" not in state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p.data)
                    state["exp_avg_sq"] = torch.zeros_like(p.data)
                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                state["step"] += 1

                exp_avg.mul_(group["betas"][0]).add_(grad, alpha=1 - group["betas"][0])
                exp_avg_sq.mul_(group["betas"][1]).addcmul_(
                    grad, grad, value=1 - group["betas"][1]
                )

                # Bias correction
                if group["correct_bias"]:
                    bias_correction1 = 1 - group["betas"][0] ** state["step"]
                    bias_correction2 = 1 - group["betas"][1] ** state["step"]
                    corrected_step_size = (
                        alpha * (bias_correction2**0.5) / bias_correction1
                    )
                else:
                    corrected_step_size = alpha

                # Update parameters
                denom = exp_avg_sq.sqrt().add_(group["eps"])
                p.data.addcdiv_(exp_avg, denom, value=-corrected_step_size)

                # Add weight decay after the main gradient-based updates.
                # Please note that the learning rate should be incorporated into this update.
                if group["weight_decay"] != 0:
                    p.data.add_(p.data, alpha=-group["lr"] * group["weight_decay"])

        return loss


from torch import nn


class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(10, 50)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(50, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


model = SimpleModel()
optimizer = AdamW(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# Dummy inputs and targets
inputs = torch.randn(10, 10)  # batch of 10 samples, 10 features each
targets = torch.randn(10, 1)  # batch of 10 targets

for epoch in range(100):  # run for 100 epochs
    optimizer.zero_grad()  # clear previous gradients
    outputs = model(inputs)  # forward pass
    loss = criterion(outputs, targets)  # compute loss
    loss.backward()  # backward pass to compute gradients
    optimizer.step()  # update model parameters

    print(f"Epoch {epoch+1}, Loss: {loss.item()}")

    #         raise NotImplementedError()

    #         # State should be stored in this dictionary
    #         state = self.state[p]

    #         # Access hyperparameters from the `group` dictionary
    #         alpha = group["lr"]

    #         # Update first and second moments of the gradients

    #         # Bias correction
    #         # Please note that we are using the "efficient version" given in
    #         # https://arxiv.org/abs/1412.6980

    #         # Update parameters

    #         # Add weight decay after the main gradient-based updates.
    #         # Please note that the learning rate should be incorporated into this update.

    # return loss
