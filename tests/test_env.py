import numpy as np
import pytest

from hgbp_sim import OBS_NAMES, EnvConfig, HGBPVecEnv


def test_vec_env_api():
    cfg = EnvConfig(episode_time=60.0, k_points=(1, 2), hold_time=(20.0, 40.0), start_mode="random")
    env = HGBPVecEnv(6, cfg, seed=1)
    obs = env.reset()
    assert obs.shape == (6, len(OBS_NAMES)) and np.isfinite(obs).all()
    assert env.act_dim == 4
    n_done = 0
    for _ in range(80):
        a = env.expert_action() + 0.1 * env.rng.standard_normal((6, env.act_dim))
        obs, r, term, trunc, info = env.step(a)
        assert obs.shape == (6, len(OBS_NAMES)) and np.isfinite(obs).all()
        assert r.shape == (6,) and np.isfinite(r).all()
        n_done += int((term | trunc).sum())
    assert n_done >= 6  # episodes truncated at 60 s and auto-reset


def test_absolute_actions():
    cfg = EnvConfig(episode_time=30.0, action_mode="absolute", start_mode="warm", noise=False)
    env = HGBPVecEnv(2, cfg, seed=2)
    env.reset()
    a = env.expert_action()
    obs, r, term, trunc, info = env.step(a)
    assert np.allclose(info["u_cmd"], 0.5 * (a + 1.0))


def test_warm_start_is_near_setpoint():
    cfg = EnvConfig(start_mode="warm", p_warm_at_setpoint=1.0, noise=False, randomize_params=False,
                    p_charge_extreme=0.0)
    env = HGBPVecEnv(4, cfg, seed=3)
    obs, r, term, trunc, info = env.step(np.zeros((4, 4)))
    e = info["error"]
    assert np.abs(e[:, 0]).max() < 0.02e5
    assert np.abs(e[:, 1]).max() < 0.05e5
    assert np.abs(e[:, 2]).max() < 0.5
    assert np.abs(e[:, 3]).max() < 0.1e5
    assert info["in_tol"].all()


def test_charge_is_randomized():
    cfg = EnvConfig(start_mode="cold", p_charge_extreme=1.0, charge_extreme_range=(0.3, 1.8))
    env = HGBPVecEnv(16, cfg, seed=4)
    obs, r, term, trunc, info = env.step(np.zeros((16, 4)))
    f = info["charge_factor"]
    assert f.min() < 0.7 and f.max() > 1.4
    assert np.allclose(info["true"]["charge"] / env.plant.nominal_charge(), f, rtol=1e-6)


def test_gymnasium_wrapper():
    gym = pytest.importorskip("gymnasium")
    from gymnasium.utils.env_checker import check_env
    from hgbp_sim import HGBPEnv
    env = HGBPEnv(EnvConfig(episode_time=20.0), seed=0)
    check_env(env, skip_render_check=True)
    obs, info = env.reset(seed=0)
    total = 0.0
    for _ in range(25):
        obs, r, term, trunc, info = env.step(env.expert_action())
        total += r
        if term or trunc:
            break
    assert np.isfinite(total)
