import jax
import jax.numpy as jnp
from jax.experimental.jet import jet
import numpy as np
import time

# ---------- MLP Definition ----------
def net(params, x):
    for W, b in params:
        x = jnp.tanh(x @ W + b)
    return x

# ---------- Jet Sampling ----------
def sample_jets(key, batch_size, dim):
    return jax.random.normal(key, (batch_size, dim))

# ---------- Potential Energy ----------
def potential_energy(x):
    # Harmonic oscillator potential: V(x) = 0.5 * ||x||^2
    return 0.5 * jnp.sum(x**2, axis=1)

# ---------- Laplacian Estimation ----------
def estimate_laplacian(params, x, jets):
    def f(z):
        return jnp.sum(net(params, z)**2)

    def estimator(x_i, v_i):
        _, jvp = jet(f, (x_i,), ((v_i,),))
        return jvp

    return jnp.array([estimator(x[i], jets[i]) for i in range(x.shape[0])])

# ---------- Distributed Energy Estimator ----------
def run_distributed_energy(batch_size, dim=128, layers=3, hidden_dim=128, save_results=False):
    num_devices = len(jax.devices())
    assert batch_size % num_devices == 0
    chunk_size = batch_size // num_devices

    print(f"Running Many-body Energy Estimator with {num_devices} GPUs, batch_size={batch_size}, dim={dim}")

    key = jax.random.PRNGKey(0)
    subkeys = jax.random.split(key, 4)

    # Random neural net params
    params = [(jax.random.normal(subkeys[0], (hidden_dim, hidden_dim)),
               jax.random.normal(subkeys[1], (hidden_dim,))) for _ in range(layers)]

    # Input and jets
    x = jax.random.normal(subkeys[2], (batch_size, dim))
    jets = sample_jets(subkeys[3], batch_size, dim)

    # Shard inputs
    x_chunks = [x[i * chunk_size:(i + 1) * chunk_size] for i in range(num_devices)]
    jet_chunks = [jets[i * chunk_size:(i + 1) * chunk_size] for i in range(num_devices)]
    x_sharded = jax.device_put_sharded(x_chunks, jax.devices())
    jet_sharded = jax.device_put_sharded(jet_chunks, jax.devices())

    # Distributed computation
    @jax.pmap
    def parallel_energy(x, jet):
        psi = net(params, x)
        laplacian = estimate_laplacian(params, x, jet)
        V = potential_energy(x)
        kinetic = -0.5 * laplacian
        energy_density = kinetic + V
        return energy_density

    # Timing the multi-GPU execution
    start = time.time()
    local_energies = parallel_energy(x_sharded, jet_sharded)
    end = time.time()

    total_time = end - start
    total_energy = jnp.mean(local_energies)

    print(f"Mean Energy: {total_energy:.6f}")
    print(f"Total computation time: {total_time:.4f} seconds")

    if save_results:
        from pathlib import Path
        Path("results").mkdir(exist_ok=True)
        np.save("results/energy_density.npy", np.array(local_energies))

    return float(total_energy), total_time

if __name__ == "__main__":
    run_distributed_energy(batch_size=128, dim=128, save_results=True)

# $ python3 scrodinger.py 
#Running Many-body Energy Estimator with 4 GPUs, batch_size=128, dim=128
#Mean Energy: 60.312801
#Total computation time: 9.8028 seconds
