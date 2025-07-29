import numpy as np
import torch

def GradientDescent(func, init_x, learning_rate=0.01, n_iter=1000, tol=1e-4, verbose=True):
    x = torch.tensor(init_x, dtype=torch.float64, requires_grad=True)
    prev_loss = float("inf")

    for i in range(n_iter):
        y = func(x)
        loss_val = y.item()

        if torch.isnan(y):
            print(f"NaN encountered in loss at iteration {i}.")
            break

        y.backward()

        if x.grad is None or torch.isnan(x.grad).any():
            print(f"NaN encountered in gradients at iteration {i}.")
            break

        with torch.no_grad():
            torch.nn.utils.clip_grad_norm_([x], max_norm=1.0)
            x = x - learning_rate * x.grad

        x.requires_grad_(True)

        if x.grad is not None:
            x.grad.zero_()

        if abs(prev_loss - loss_val) < tol and verbose:
            print(f"Converged at iteration {i}, Δloss = {abs(prev_loss - loss_val):.6f}")
            break

        prev_loss = loss_val

    return x.detach().cpu().numpy()

def LBFGS(fn, X, lr=0.001, n_iter=100, m=10, tol=1e-4, verbose=True):
    history = []  
    alphas = []
    x = torch.tensor(X, dtype=torch.float64)
    x = x.view(-1)  

    prev_loss = float('inf')

    for i in range(n_iter):
        x.requires_grad_(True)
        x_prev = x.clone().detach()

        y = fn(x)
        loss_val = y.item()
        y.backward()

        if abs(prev_loss - loss_val) < tol and verbose:
            print(f"Converged at iteration {i}, Δloss = {abs(prev_loss - loss_val):.6f}")
            break
        prev_loss = loss_val

        with torch.no_grad():
            x -= lr * x.grad

        sk = (x - x_prev).detach()
        gk = torch.autograd.grad(fn(x), x, create_graph=False)[0].detach()

        if i == 0:
            g_prev = gk
            x.grad.zero_()
            continue

        yk = (gk - g_prev).detach()
        rho = 1 / torch.dot(yk, sk)

        if len(history) == m:
            history.pop(0)
        history.append((sk, yk, rho))

        q = gk.clone()
        alphas = []

        for s_i, y_i, rho_i in reversed(history):
            alpha_i = rho_i * torch.dot(s_i, q)
            alphas.append(alpha_i)
            q = q - alpha_i * y_i

        gamma = torch.dot(sk, yk) / torch.dot(yk, yk)
        r = gamma * q

        for (s_i, y_i, rho_i), alpha_i in zip(history, reversed(alphas)):
            beta = rho_i * torch.dot(y_i, r)
            r = r + s_i * (alpha_i - beta)

        alpha = wolfe_line_search(fn, x, -r, gk, alpha_init=1.0)
        with torch.no_grad():
            x += alpha * (-r)

        g_prev = gk
        x.grad.zero_()

    return x.detach().cpu().numpy()


def wolfe_line_search(fn, x, p, g, alpha_init=1.0, c1=1e-4, c2=0.9, max_iter=20):
    alpha = alpha_init
    fx = fn(x)
    grad_phi0 = torch.dot(g, p)
    
    for _ in range(max_iter):
        x_new = x + alpha * p
        fx_new = fn(x_new)
        g_new = torch.autograd.grad(fn(x_new), x_new, create_graph=False)[0]
        grad_phi_new = torch.dot(g_new, p)
        
        if fx_new > fx + c1 * alpha * grad_phi0:
            alpha *= 0.5

        elif grad_phi_new < c2 * grad_phi0:
            alpha *= 1.1
        else:
            break
    return alpha






def AdaBeliefOptimizer(func, init_x, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                       n_iter=1000, tol=1e-4, weight_decay=0.0, verbose=True):
    x = torch.tensor(init_x, dtype=torch.float64, requires_grad=True)

    m = torch.zeros_like(x) 
    s = torch.zeros_like(x)  
    prev_loss = float('inf')

    for i in range(1, n_iter + 1):
        y = func(x)
        loss_val = y.item()

        if torch.isnan(y):
            print(f"NaN in loss at iteration {i}.")
            break

        y.backward()

        if x.grad is None or torch.isnan(x.grad).any():
            print(f"NaN in gradients at iteration {i}.")
            break

        g = x.grad.detach()

        if weight_decay > 0:
            g = g + weight_decay * x

        m = betas[0] * m + (1 - betas[0]) * g
        g_diff = g - m
        s = betas[1] * s + (1 - betas[1]) * (g_diff ** 2)

        m_hat = m / (1 - betas[0] ** i)
        s_hat = s / (1 - betas[1] ** i)

        update = lr * m_hat / (s_hat.sqrt() + eps)

        with torch.no_grad():
            x -= update

        x.requires_grad_(True)

        if x.grad is not None:
            x.grad.zero_()

        if abs(prev_loss - loss_val) < tol and verbose:
            print(f"Converged at iteration {i}, Δloss = {abs(prev_loss - loss_val):.6f}")
            break

        prev_loss = loss_val

    return x.detach().cpu().numpy()