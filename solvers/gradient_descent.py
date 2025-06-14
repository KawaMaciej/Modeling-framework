import jax.numpy as jnp
from jax import jit, grad

def GradienDescent(X,Y, parameter, loss, n_iter, lr):
    loss_grad = jit(grad(loss))
    
    for _ in range(n_iter):
        grads = loss_grad(parameter, X, Y)
        parameter -= lr * grads
    if jnp.any(jnp.isnan(parameter)):
        raise ValueError("Gradient explosion, please change learning rate")