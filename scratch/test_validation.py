import os
import sys
import jax
import jax.numpy as jnp

sys.path.append(os.path.abspath('training'))

from getup_env_mjx import GetUpFrontMjxEnv, GetUpBackMjxEnv

def test_hybrid():
    for name, env_cls in [("FRONT (HYBRID)", GetUpFrontMjxEnv), ("BACK (HYBRID)", GetUpBackMjxEnv)]:
        env = env_cls()
        rng = jax.random.PRNGKey(42)
        state = env.reset(rng)
        
        print(f"\n--- Testing {name} Environment ---")
        print(f"Keyframe targets loaded: {env.n_phases} phases")
        print(f"Observation vector shape: {state.obs.shape}")
        print(f"Reset initial torso height: {float(state.pipeline_state.x.pos[env.torso_id][2]):.3f}")

        step_fn = jax.jit(env.step)
        
        # Test step with zero action
        state_next = step_fn(state, jnp.zeros(env.sys.nu))
        print(f"\nStep 1 (Zero Action): reward={state_next.reward:.3f}, height={state_next.metrics['torso_height']:.3f}")
        print("Step 1 metrics:", {k: round(float(v), 2) for k, v in state_next.metrics.items()})
        
        # Test 10 resets to check multi-pose keyframe curriculum
        heights = []
        for i in range(15):
            rng, subkey = jax.random.split(rng)
            st = env.reset(subkey)
            torso_h = float(st.pipeline_state.x.pos[env.torso_id][2])
            heights.append(round(torso_h, 2))
        
        print(f"Keyframe Curriculum Reset Heights (15 samples): {heights}")

if __name__ == "__main__":
    test_hybrid()
