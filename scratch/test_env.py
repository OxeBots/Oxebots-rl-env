import os
import sys
import jax
import jax.numpy as jnp

sys.path.append(os.path.abspath('training'))

from getup_env_mjx import GetUpFrontMjxEnv, GetUpBackMjxEnv

def test_envs():
    print("Testing GetUpFrontMjxEnv...")
    env_front = GetUpFrontMjxEnv()
    rng = jax.random.PRNGKey(0)
    state_front = env_front.reset(rng)
    print("Front reset obs shape:", state_front.obs.shape)
    print("Front reset metrics:", state_front.metrics)
    
    action = jnp.zeros(env_front.sys.nu)
    step_fn = jax.jit(env_front.step)
    state_front_next = step_fn(state_front, action)
    print("Front step 1 reward:", state_front_next.reward)
    print("Front step 1 metrics:", state_front_next.metrics)

    print("\nTesting GetUpBackMjxEnv...")
    env_back = GetUpBackMjxEnv()
    state_back = env_back.reset(rng)
    print("Back reset obs shape:", state_back.obs.shape)
    print("Back reset metrics:", state_back.metrics)
    
    state_back_next = step_fn(state_back, action)
    print("Back step 1 reward:", state_back_next.reward)
    print("Back step 1 metrics:", state_back_next.metrics)
    print("✅ Environments initialized and stepped successfully!")

if __name__ == "__main__":
    test_envs()
