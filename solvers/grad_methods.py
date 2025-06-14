import numpy as np
import torch

def GradientDescent(func, init_x, learning_rate=0.01, n_iter=1000):
    x = torch.tensor(init_x, requires_grad=True)
    for _ in range(n_iter):
        y = func(x)
        y.backward()
        with torch.no_grad():
            x -= learning_rate * x.grad
        x.grad.zero_()
    return x.detach().cpu().numpy()


def LBFGS(fn, X, lr=0.001, n_iter=100, m=10):
    history = []  
    alphas = []
    x = torch.tensor(X, dtype=torch.float64)
    x = x.view(-1)  

    for _ in range(n_iter):
        x.requires_grad_(True)
        x_prev = x.clone().detach()

        y = fn(x)
        y.backward()

        with torch.no_grad():
            x -= lr * x.grad

        sk = (x - x_prev).detach()

        gk = torch.autograd.grad(fn(x), x, create_graph=False)[0].detach()

        if _ == 0:
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