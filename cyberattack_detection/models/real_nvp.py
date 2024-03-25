import torch.nn as nn
import torch

class NormalizingFlow(nn.Module):
    def __init__(self, layers, prior):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.prior = prior

    def log_prob(self, x):
        log_likelihood = None

        for layer in self.layers:
            x, change = layer.f(x)
            if log_likelihood is not None:
                log_likelihood = log_likelihood + change
            else:
                log_likelihood = change
        log_likelihood = log_likelihood + self.prior.log_prob(x)

        return log_likelihood.mean()

    def sample(self, num_samples):
        x = self.prior.sample((num_samples,))

        for layer in self.layers[::-1]:
            x = layer.g(x)

        return x
class InvertibleLayer(nn.Module):
    def __init__(self, var_size):
        super().__init__()
        self.var_size = var_size

    def f(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        """
        pass

    def g(self, x: torch.Tensor) -> torch.Tensor:
        """
        Inverse pass
        """
        pass

    def check_inverse(self) -> None:
        """
        Perform check of invertibility
        """
        x = torch.randn(10, self.var_size)
        assert torch.norm(x - self.g(self.f(x)[0])).item() < 0.001

    def check_log_det(self) -> None:
        """
        Perform check of log determinant
        """
        x = torch.randn(1, self.var_size).requires_grad_()
        _, log_det = self.f(x)

        jac = torch.autograd.functional.jacobian(lambda x: self.f(x)[0], x)
        assert torch.abs(log_det - torch.log(torch.det(jac[0, :, 0, :]))).item() < 0.001

class RealNVP(InvertibleLayer):
    def __init__(self, var_size, mask, hidden=16):
        super().__init__(var_size=var_size)

        self.mask = mask

        self.nn_t = nn.Sequential(
            nn.Linear(var_size, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, var_size),
        )
        self.nn_s = nn.Sequential(
            nn.Linear(var_size, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, var_size),
        )

    def f(self, x):
        t = self.nn_t(x * self.mask[None, :])
        s = torch.tanh(
            self.nn_s(x * self.mask[None, :])
        )

        new_x = (x * torch.exp(s) + t) * (1 - self.mask[None, :]) + x * self.mask[
            None, :
        ]
        log_det = (s * (1 - self.mask[None, :])).sum(dim=-1)
        return new_x, log_det

    def g(self, x):
        t = self.nn_t(x * self.mask[None, :])
        s = torch.tanh(self.nn_s(x * self.mask[None, :]))
        new_x = ((x - t) * torch.exp(-s)) * (1 - self.mask[None, :]) + x * self.mask[
                                                                           None, :
                                                                           ]
        return new_x

