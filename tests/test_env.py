import os

import numpy as np
import pytest

from hgbp_sim import OBS_GROUPS, OBS_NAMES, EnvConfig, HGBPVecEnv
from hgbp_sim.env import ST_OFF, ST_RUNNING, ST_STARTING
from hgbp_sim.plant import HGBPPlant
from hgbp_sim.scenarios import inside_polygon


def test_vec_env_api():
    cfg = EnvConfig(episode_time=60.0, k_points=(1, 2), hold_time=(20.0, 40.0), start_mode="random")
    env = HGBPVecEnv(6, cfg, seed=1)
    obs = env.reset()
    assert obs.shape == (6, len(OBS_NAMES)) and np.isfinite(obs).all()
    assert env.act_dim == 4
    assert sum(len(g) for g in OBS_GROUPS.values()) == len(OBS_NAMES)
    n_done = 0
    for _ in range(int(80.0 / cfg.dt_ctrl)):      # 80 s, past the 60 s truncation
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
    assert np.allclose(obs[:, i("V_disp_rel")], 1.42)         # 355 cm3/rev over the 250 cm3 reference
    assert np.allclose(obs[:, i("rf_T_crit")], 344.5 / 400.0, atol=1e-3)   # R410A
    assert np.all(obs[:, i("mdot_norm")] > 0.3) and np.all(obs[:, i("mdot_norm")] < 1.0)
    assert np.all(obs[:, i("st_running")] == 1.0)


STAND = os.path.join(os.path.dirname(__file__), "..", "webui", "config", "stand_defaults.json")


def test_test_points_follow_the_procedure():
    """Return gas temperature setpoints, the geometric-mean liquid pressure clamped by the
    cooling water, speeds inside the VFD range and points clear of the trip limits."""
    env = HGBPVecEnv(32, EnvConfig(start_mode="warm", noise=False), seed=5)
    pr, p = env.props, env.params
    k = env.sched_k
    P_s, P_d, RGT, P_i, N = (np.concatenate([env.sched_points[i, :k[i], c] for i in range(32)]) for c in range(5))
    T_wi = np.concatenate([np.full(k[i], env.plant.T_wi[i]) for i in range(32)])
    clamp = pr.P_sat(T_wi + env.cfg.envelope.dT_int_above_water)
    assert np.allclose(P_i, np.maximum(np.sqrt(P_s * P_d), clamp), rtol=1e-9)
    assert (P_i > np.sqrt(P_s * P_d) * 1.001).any() and (P_i > clamp * 1.001).any()     # both cases occur
    SH = RGT - pr.T_sat(P_s)
    assert SH.min() >= 5.0 - 1e-6 and SH.max() <= 50.0 + 1e-6
    assert np.isclose(RGT, 18.33 + 273.15).any()                                       # 65 degF points
    assert inside_polygon(pr.T_sat(P_s) - 273.15, pr.T_sat(P_d) - 273.15, env.cfg.envelope.polygon).all()
    Hz = N / p.N_nom * 60.0
    assert Hz.min() >= 35.0 - 1e-6 and Hz.max() <= 75.0 + 1e-6
    assert np.all((env.plant.T_wi > 3.3 + 273.15 - 1e-9) & (env.plant.T_wi < 4.6 + 273.15 + 1e-9))
    # warm starts track the return gas temperature, and the stand starts clear of the trips
    obs, r, term, trunc, info = env.step(env.expert_action())
    assert np.all(info["true"]["T_d"] < env.plant.p.T_d_max - 5.0)
    assert np.all(info["true"]["P_s"] > env.plant.p.P_s_min)
    i = OBS_NAMES.index
    assert np.allclose(obs[:, i("RGT_sp")] * 50.0, env.sp[:, 2] - 273.15)


def test_stand_definition_sets_plant_and_loops():
    import json
    doc = json.load(open(STAND))
    env = HGBPVecEnv(2, EnvConfig(stand=STAND, start_mode="warm", randomize_params=False), seed=6)
    assert env.params.T_d_max == doc["plant"]["T_d_max"] and env.params.N_min == doc["plant"]["N_min"]
    pid = env.expert.pid_1                       # dpv: P 150 % of a 0..50 bar span, I 50 s, DIR
    t = doc["loops"]["dpv"]
    Kp = -100.0 / (t["P"] * (t["RH"] - t["RL"])) / 1e5
    assert np.isclose(pid.Kp, Kp) and np.isclose(pid.Ki, Kp / t["I"])
    assert env.expert.pv_filter["stv"] == t["FL"] and np.isclose(env.expert.pid_1.u_min, t["OL"] / 100.0)
    for _ in range(20):
        obs, r, term, trunc, info = env.step(env.expert_action())
    assert np.isfinite(obs).all()
    assert np.all(env.meas["Vdot"] > 0.0)
    rho = env.meas["mdot"] / env.meas["Vdot"]                     # suction vapor density
    assert np.all((rho > 5.0) & (rho < 80.0))


def test_collections_complete_a_point():
    """Stable for the dwell, then n collections; leaving the band loses the running one."""
    cfg = EnvConfig(start_mode="warm", p_warm_at_setpoint=1.0, noise=False, randomize_params=False,
                    p_charge_extreme=0.0, k_points=(2, 2), dwell_required=10.0, n_collections=2,
                    collection_time=20.0, hold_time=(600.0, 600.0))
    env = HGBPVecEnv(1, cfg, seed=7)
    done_at, lost = None, False
    for step in range(int(120.0 / cfg.dt_ctrl)):
        if step == int(20.0 / cfg.dt_ctrl):        # mid first collection: push the stand off the band
            env.sp[:, 1] += 2e5
        if step == int(21.0 / cfg.dt_ctrl):
            env.sp[:, 1] -= 2e5
        obs, r, term, trunc, info = env.step(env.expert_action())
        lost |= bool(info["collection_lost"][0])
        if info["point_done"][0]:
            done_at = info["t"][0]
            break
    assert lost
    # first collection lost, then dwell + 2 x 20 s once back in the band
    assert done_at is not None and done_at >= 21.0 + 10.0 + 40.0 - 1e-6
    assert env.k[0] == 1 and env.n_coll[0] == 0                  # next point, counters reset


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
                    randomize_params=False, p_charge_extreme=0.0, cold_liquid_in_suction=(0.0, 0.0),
                    auto_reset=False)
    env = HGBPVecEnv(2, cfg, seed=5)
    assert env.act_dim == 5
    a = env.expert_action()
    a[:, 4] = -1.0                                   # no run request: stays OFF
    for _ in range(int(15.0 / cfg.dt_ctrl)):
        obs, r, term, trunc, info = env.step(a)
    assert (info["state"] == ST_OFF).all() and (info["r_idle"] < 0).all()
    # request a run with valve 1 closed -> blocked by the permissives
    env.u_cmd[:, 0] = 0.0
    env.plant.x[:, HGBPPlant.U1] = 0.0
    a[:, 4] = 1.0
    obs, r, term, trunc, info = env.step(a)
    assert (info["state"] == ST_OFF).all() and (info["r_blocked"] < 0).all()
    # expert requests the run with sensible valve positions -> STARTING -> RUNNING
    env.u_cmd[:, 0] = 0.5
    env.plant.x[:, HGBPPlant.U1] = 0.5
    seen_starting = False
    states = []
    done = np.zeros(2, bool)
    completed = np.zeros(2, int)
    for _ in range(int(1500.0 / cfg.dt_ctrl)):    # 1500 s: 900 s hold + dwell + shutdown
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


def test_given_schedule():
    """A procedure's points replace the sampled ones; the stand warm-starts at the first."""
    cfg = EnvConfig(start_mode="warm", p_warm_at_setpoint=1.0, noise=False, randomize_params=False,
                    p_charge_extreme=0.0, max_points=6)
    env = HGBPVecEnv(2, cfg, seed=11)
    T_wi = env.plant.T_wi
    lift = [(-10.0, 45.0), (0.0, 40.0), (5.0, 35.0), (7.0, 45.0), (-5.0, 50.0)]   # (evap, cond) degC
    pts = np.zeros((2, len(lift), 5))
    for j, (te, tc) in enumerate(lift):
        p = cfg.envelope.point(env.props, te, tc, 18.33, env.params.N_nom, T_wi)
        pts[:, j] = np.stack([p[f] for f in ("P_s", "P_d", "RGT", "P_i", "N")], 1)
    env.reset(schedule=dict(points=pts, k=[5, 3], hold=600.0))
    assert np.allclose(env.sched_points[:, :5], pts) and list(env.sched_k) == [5, 3]
    assert np.allclose(env.sched_hold[:, :5], 600.0) and np.allclose(env.sp, pts[:, 0])
    # the liquid pressure: geometric mean, clamped 6 K above the water
    Ti = env.props.T_sat(pts[:, :, 3]) - 273.15
    gm = env.props.T_sat(np.sqrt(pts[:, :, 0] * pts[:, :, 1])) - 273.15
    assert np.all(Ti >= gm - 1e-6) and np.all(Ti >= T_wi[:, None] - 273.15 + 6.0 - 1e-6)
    obs, r, term, trunc, info = env.step(np.zeros((2, 4)))
    assert info["in_tol"].all() and (info["state"] == ST_RUNNING).all()
    with pytest.raises(ValueError):
        env.reset(schedule=dict(points=np.zeros((2, 7, 5))))     # longer than max_points
    with pytest.raises(ValueError):
        HGBPVecEnv(1, EnvConfig(k_points=(1, 5)))                 # sampled schedules too
