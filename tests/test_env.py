import numpy as np
import pytest

from hgbp_sim import OBS_GROUPS, OBS_NAMES, EnvConfig, HGBPVecEnv
from hgbp_sim.env import ST_OFF, ST_RUNNING, ST_STARTING


def test_vec_env_api():
    cfg = EnvConfig(episode_time=60.0, k_points=(1, 2), hold_time=(20.0, 40.0), start_mode="random")
    env = HGBPVecEnv(6, cfg, seed=1)
    obs = env.reset()
    assert obs.shape == (6, len(OBS_NAMES)) and np.isfinite(obs).all()
    assert env.act_dim == 4
    assert sum(len(g) for g in OBS_GROUPS.values()) == len(OBS_NAMES)
    n_done = 0
    for _ in range(80):
        a = env.expert_action() + 0.1 * env.rng.standard_normal((6, env.act_dim))
        obs, r, term, trunc, info = env.step(a)
        assert obs.shape == (6, len(OBS_NAMES)) and np.isfinite(obs).all()
        assert r.shape == (6,) and np.isfinite(r).all()
        n_done += int((term | trunc).sum())
    assert n_done >= 6  # episodes end (truncation at 60 s) and auto-reset


def test_absolute_actions():
    cfg = EnvConfig(episode_time=30.0, action_mode="absolute", start_mode="warm", noise=False)
    env = HGBPVecEnv(2, cfg, seed=2)
    env.reset()
    a = env.expert_action()
    obs, r, term, trunc, info = env.step(a)
    assert np.allclose(info["u_cmd"], 0.5 * (a + 1.0))


def test_warm_start_is_near_setpoint_and_context_in_obs():
    cfg = EnvConfig(start_mode="warm", p_warm_at_setpoint=1.0, noise=False, randomize_params=False,
                    p_charge_extreme=0.0)
    env = HGBPVecEnv(4, cfg, seed=3)
    obs, r, term, trunc, info = env.step(np.zeros((4, 4)))
    e = info["error"]                       # kelvin
    assert np.abs(e[:, 0]).max() < 0.3 and np.abs(e[:, 1]).max() < 0.2
    assert np.abs(e[:, 2]).max() < 0.5 and np.abs(e[:, 3]).max() < 1.0
    assert info["in_tol"].all() and (info["state"] == ST_RUNNING).all()
    i = OBS_NAMES.index
    assert np.allclose(obs[:, i("V_disp_rel")], 1.0)          # default 250 cm3/rev
    assert np.allclose(obs[:, i("rf_T_crit")], 374.2 / 400.0, atol=1e-3)
    assert np.all(obs[:, i("mdot_norm")] > 0.3) and np.all(obs[:, i("mdot_norm")] < 1.0)
    assert np.all(obs[:, i("st_running")] == 1.0)


def test_charge_is_randomized():
    cfg = EnvConfig(start_mode="cold", p_charge_extreme=1.0, charge_extreme_range=(0.3, 1.8))
    env = HGBPVecEnv(16, cfg, seed=4)
    obs, r, term, trunc, info = env.step(np.zeros((16, 4)))
    f = info["charge_factor"]
    assert f.min() < 0.7 and f.max() > 1.4
    assert np.allclose(info["true"]["charge"] / env.plant.nominal_charge(), f, rtol=1e-6)


def test_start_stop_interlock_and_dwell_completion():
    cfg = EnvConfig(start_mode="cold", start_stop_action=True, k_points=(1, 1), hold_time=(900.0, 900.0),
                    dwell_required=60.0, min_off_time=10.0, min_run_time=30.0, noise=False,
                    randomize_params=False, p_charge_extreme=0.0, cold_liquid_in_accumulator=(0.0, 0.0),
                    auto_reset=False)
    env = HGBPVecEnv(2, cfg, seed=5)
    assert env.act_dim == 5
    a = env.expert_action()
    a[:, 4] = -1.0                                   # no run request: stays OFF
    for _ in range(15):
        obs, r, term, trunc, info = env.step(a)
    assert (info["state"] == ST_OFF).all() and (info["r_idle"] < 0).all()
    # request a run with valve 1 closed -> blocked by the permissives
    env.u_cmd[:, 0] = 0.0
    env.plant.x[:, 11] = 0.0
    a[:, 4] = 1.0
    obs, r, term, trunc, info = env.step(a)
    assert (info["state"] == ST_OFF).all() and (info["r_blocked"] < 0).all()
    # expert requests the run with sensible valve positions -> STARTING -> RUNNING
    env.u_cmd[:, 0] = 0.5
    env.plant.x[:, 11] = 0.5
    seen_starting = False
    states = []
    done = np.zeros(2, bool)
    completed = np.zeros(2, int)
    for _ in range(1500):
        obs, r, term, trunc, info = env.step(env.expert_action())
        states.append(info["state"][0])
        seen_starting |= bool((info["state"] == ST_STARTING).any())
        newly = (term | trunc) & ~done
        if newly.any():
            completed[newly] = info["episode_points_completed"][newly]
        done |= term | trunc
        if done.all():
            break
    assert seen_starting and ST_RUNNING in states
    assert done.all() and info["schedule_complete"].all()   # point completed by dwell ...
    assert (completed == 1).all()
    assert (info["state"] == ST_OFF).all()                  # ... then shut down


def test_gymnasium_wrapper():
    gym = pytest.importorskip("gymnasium")
    from gymnasium.utils.env_checker import check_env
    from hgbp_sim import HGBPEnv
    env = HGBPEnv(EnvConfig(episode_time=20.0, start_stop_action=True), seed=0)
    check_env(env, skip_render_check=True)
    obs, info = env.reset(seed=0)
    assert env.action_space.shape == (5,)
    total = 0.0
    for _ in range(25):
        obs, r, term, trunc, info = env.step(env.expert_action())
        total += r
        if term or trunc:
            break
    assert np.isfinite(total)
