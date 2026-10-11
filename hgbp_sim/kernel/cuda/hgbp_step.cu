// CUDA port of the model kernel's step (hgbp_sim/kernel/model.py, props.py and the
// scalar components in hgbp_sim/components.py): one thread per stand advances it by
// n_sub steps of dt with the CPU kernel's algorithm (sub-steps from the fastest local
// rate, implausible sub-steps redone four times finer, at most twice, post() after
// every sub-step).  Benchmark prototype: the CPU kernel stays the reference.
//
// Every device function is a template on its compute type R.  The Python wrapper
// (hgbp_sim/kernel/cuda_step.py) wraps every floating-point literal between the
// DEVICE markers as R(literal) before compiling, so that a float32 build computes
// in float32 throughout (C++ would otherwise promote to double), and generates
// hgbp_layout.h (state, parameter and table indices) from hgbp_sim.kernel.layout.
//
// Precision modes (template arguments of the kernel):
//   RS  state, sub-step accumulation, projections (post), stiffness and step control
//   RF  right-hand side arithmetic (stage states and derivatives)
//   T   property tables
//   fp64 = (double, double, double), fp32 = (float, float, float),
//   mixed = (double, float, float)
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
#include <cuda_runtime.h>

#include "hgbp_layout.h"

// smallest positive divisor guard of props._invert_row (1e-300 underflows in float)
template <class R> __device__ __forceinline__ R tiny_den();
template <> __device__ __forceinline__ double tiny_den<double>() { return 1e-300; }
template <> __device__ __forceinline__ float tiny_den<float>() { return 1e-30f; }

//@@DEVICE_BEGIN
// ----------------------------------------------------------------- math
__device__ __forceinline__ double m_exp(double a) { return exp(a); }
__device__ __forceinline__ float m_exp(float a) { return expf(a); }
__device__ __forceinline__ double m_log(double a) { return log(a); }
__device__ __forceinline__ float m_log(float a) { return logf(a); }
__device__ __forceinline__ double m_log10(double a) { return log10(a); }
__device__ __forceinline__ float m_log10(float a) { return log10f(a); }
__device__ __forceinline__ double m_sqrt(double a) { return sqrt(a); }
__device__ __forceinline__ float m_sqrt(float a) { return sqrtf(a); }
__device__ __forceinline__ double m_pow(double a, double b) { return pow(a, b); }
__device__ __forceinline__ float m_pow(float a, float b) { return powf(a, b); }
__device__ __forceinline__ double m_cos(double a) { return cos(a); }
__device__ __forceinline__ float m_cos(float a) { return cosf(a); }
__device__ __forceinline__ double m_sin(double a) { return sin(a); }
__device__ __forceinline__ float m_sin(float a) { return sinf(a); }
__device__ __forceinline__ double m_tan(double a) { return tan(a); }
__device__ __forceinline__ float m_tan(float a) { return tanf(a); }
__device__ __forceinline__ double m_abs(double a) { return fabs(a); }
__device__ __forceinline__ float m_abs(float a) { return fabsf(a); }
__device__ __forceinline__ double m_copysign(double a, double b) { return copysign(a, b); }
__device__ __forceinline__ float m_copysign(float a, float b) { return copysignf(a, b); }
__device__ __forceinline__ bool m_isfinite(double a) { return isfinite(a); }
__device__ __forceinline__ bool m_isfinite(float a) { return isfinite(a); }

// Python's (and numba's) min / max of two numbers: the first unless the second compares
// below / above it (NaN handling as in Python, unlike fmin / fmax)
template <class R> __device__ __forceinline__ R pmin(R a, R b) { return (b < a) ? b : a; }
template <class R> __device__ __forceinline__ R pmax(R a, R b) { return (b > a) ? b : a; }
template <class R> __device__ __forceinline__ R clip(R v, R lo, R hi) { return pmin(pmax(v, lo), hi); }
__device__ __forceinline__ int imin(int a, int b) { return (b < a) ? b : a; }
__device__ __forceinline__ int imax(int a, int b) { return (b > a) ? b : a; }
template <class R> __device__ __forceinline__ R sq(R a) { return a * a; }
template <class R> __device__ __forceinline__ R pi_() { return R(3.141592653589793); }

// --------------------------------------------------------------- tables
template <class R, class T> struct Tab {
    const T* __restrict__ sat;   // (n_sat, n_p)
    const T* __restrict__ tr;    // (3, n_p)
    const T* __restrict__ vap;   // (6, n_p * n_z)
    const T* __restrict__ liq;
    R lp0, inv_dlp, p_min, p_max;
    int n_p, n_z;
    __device__ __forceinline__ R s(int k, int i) const { return R(__ldg(sat + k * n_p + i)); }
    __device__ __forceinline__ R t(int k, int i) const { return R(__ldg(tr + k * n_p + i)); }
};

template <class R> struct Sat { R T_l, T_v, h_l, h_v, rho_l, rho_v, cp_v, dhmax_v; };
template <class R> struct PhS { R T, rho, x, drho_dP, drho_dh, s, cp, T_sat, h_l, h_v, rho_l, rho_v, cp_l; };

template <class R, class T>
__device__ __forceinline__ void pidx(const Tab<R, T>& tab, R P, int& i, R& w) {
    R lp = m_log(pmin(pmax(P, tab.p_min), tab.p_max));
    R f = (lp - tab.lp0) * tab.inv_dlp;
    i = imin(imax((int)f, 0), tab.n_p - 2);
    w = pmin(pmax(f - (R)i, 0.0), 1.0);
}

template <class R, class T>
__device__ __forceinline__ R sat_row(const Tab<R, T>& tab, int k, int i, R w) {
    return tab.s(k, i) * (1.0 - w) + tab.s(k, i + 1) * w;
}

template <class R, class T>
__device__ Sat<R> sat(const Tab<R, T>& tab, R P) {
    int i; R w;
    pidx(tab, P, i, w);
    Sat<R> s;
    s.T_l = sat_row(tab, SAT_T_L, i, w); s.T_v = sat_row(tab, SAT_T_V, i, w);
    s.h_l = sat_row(tab, SAT_H_L, i, w); s.h_v = sat_row(tab, SAT_H_V, i, w);
    s.rho_l = sat_row(tab, SAT_RHO_L, i, w); s.rho_v = sat_row(tab, SAT_RHO_V, i, w);
    s.cp_v = sat_row(tab, SAT_CP_V, i, w); s.dhmax_v = sat_row(tab, SAT_DHMAX_V, i, w);
    return s;
}

template <class R, class T>
__device__ void transport(const Tab<R, T>& tab, R P, R& mu_l, R& mu_v, R& sigma) {
    int i; R w;
    pidx(tab, P, i, w);
    mu_l = tab.t(TR_MU_L, i) * (1.0 - w) + tab.t(TR_MU_L, i + 1) * w;
    mu_v = tab.t(TR_MU_V, i) * (1.0 - w) + tab.t(TR_MU_V, i + 1) * w;
    sigma = tab.t(TR_SIGMA, i) * (1.0 - w) + tab.t(TR_SIGMA, i + 1) * w;
}

template <class R, class T>
__device__ __forceinline__ R bilinear(const T* __restrict__ flat, int k, int n_p, int n_z, int i, R w, int j, R wz) {
    const T* f = flat + (size_t)k * n_p * n_z;
    int k0 = i * n_z + j;
    int k1 = k0 + n_z;
    return (R(__ldg(f + k0)) * (1.0 - wz) + R(__ldg(f + k0 + 1)) * wz) * (1.0 - w)
           + (R(__ldg(f + k1)) * (1.0 - wz) + R(__ldg(f + k1 + 1)) * wz) * w;
}

template <class R, class T>
__device__ __forceinline__ void zidx(const Tab<R, T>& tab, R zeta, int& j, R& wz) {
    R f = zeta * (R)(tab.n_z - 1);
    j = imin(imax((int)f, 0), tab.n_z - 2);
    wz = pmin(pmax(f - (R)j, 0.0), 1.0);
}

template <class R, class T>
__device__ PhS<R> state(const Tab<R, T>& tab, R P, R h) {
    int i; R w;
    pidx(tab, P, i, w);
    R h_l = sat_row(tab, SAT_H_L, i, w), h_v = sat_row(tab, SAT_H_V, i, w);
    R rho_l = sat_row(tab, SAT_RHO_L, i, w), rho_v = sat_row(tab, SAT_RHO_V, i, w);
    R T_l = sat_row(tab, SAT_T_L, i, w), T_v = sat_row(tab, SAT_T_V, i, w);
    R cp_l = sat_row(tab, SAT_CP_L, i, w);
    R hfg = h_v - h_l;
    R x = (h - h_l) / hfg;
    PhS<R> S;
    S.T_sat = T_l; S.h_l = h_l; S.h_v = h_v; S.rho_l = rho_l; S.rho_v = rho_v; S.cp_l = cp_l;
    S.x = x;
    if (x > 1.0 || x < 0.0) {
        int j; R wz;
        const T* f;
        if (x > 1.0) {
            zidx(tab, pmin(pmax((h - h_v) / sat_row(tab, SAT_DHMAX_V, i, w), 0.0), 1.0), j, wz);
            f = tab.vap;
        } else {
            zidx(tab, pmin(pmax((h_l - h) / sat_row(tab, SAT_DHMAX_L, i, w), 0.0), 1.0), j, wz);
            f = tab.liq;
        }
        int np = tab.n_p, nz = tab.n_z;
        S.T = bilinear<R, T>(f, REG_T, np, nz, i, w, j, wz);
        S.rho = bilinear<R, T>(f, REG_RHO, np, nz, i, w, j, wz);
        S.drho_dP = bilinear<R, T>(f, REG_DRHO_DP, np, nz, i, w, j, wz);
        S.drho_dh = bilinear<R, T>(f, REG_DRHO_DH, np, nz, i, w, j, wz);
        S.s = bilinear<R, T>(f, REG_S, np, nz, i, w, j, wz);
        S.cp = bilinear<R, T>(f, REG_CP, np, nz, i, w, j, wz);
        return S;
    }
    // two-phase: homogeneous mixture on the saturation line
    R v_l = 1.0 / rho_l, v_v = 1.0 / rho_v;
    R rho2 = 1.0 / (v_l + x * (v_v - v_l));
    R dv_dh = (v_v - v_l) / hfg;
    R dvl_dP = -v_l * v_l * sat_row(tab, SAT_DRHO_L_DP, i, w);
    R dvv_dP = -v_v * v_v * sat_row(tab, SAT_DRHO_V_DP, i, w);
    R dhl_dP = sat_row(tab, SAT_DH_L_DP, i, w);
    R dx_dP = -(dhl_dP + x * (sat_row(tab, SAT_DH_V_DP, i, w) - dhl_dP)) / hfg;
    R dv_dP = dvl_dP + x * (dvv_dP - dvl_dP) + (v_v - v_l) * dx_dP;
    R s_l = sat_row(tab, SAT_S_L, i, w);
    S.T = T_l + x * (T_v - T_l);
    S.rho = rho2;
    S.drho_dP = -rho2 * rho2 * dv_dP;
    S.drho_dh = -rho2 * rho2 * dv_dh;
    S.s = s_l + x * (sat_row(tab, SAT_S_V, i, w) - s_l);
    S.cp = (cp_l + sat_row(tab, SAT_CP_V, i, w)) * 0.5;
    return S;
}

template <class R, class T>
__device__ void vapor_props(const Tab<R, T>& tab, R P, R h, R& Tv, R& rho) {
    int i; R w;
    pidx(tab, P, i, w);
    R zv = pmin(pmax((h - sat_row(tab, SAT_H_V, i, w)) / sat_row(tab, SAT_DHMAX_V, i, w), 0.0), 1.0);
    R f = zv * (R)(tab.n_z - 1);
    int j = imin(imax((int)f, 0), tab.n_z - 2);
    R wz = f - (R)j;
    Tv = bilinear<R, T>(tab.vap, REG_T, tab.n_p, tab.n_z, i, w, j, wz);
    rho = bilinear<R, T>(tab.vap, REG_RHO, tab.n_p, tab.n_z, i, w, j, wz);
}

template <class R, class T>
__device__ __forceinline__ R T_vapor(const Tab<R, T>& tab, R P, R h) {
    R Tv, rho;
    vapor_props(tab, P, h, Tv, rho);
    return Tv;
}

template <class R, class T>
__device__ R invert_row(const T* __restrict__ flat0, int k, int n_p, int n_z, int i, R w, R target, bool decreasing) {
    const T* flat = flat0 + (size_t)k * n_p * n_z;
    int cnt = 0;
    int base0 = i * n_z, base1 = (i + 1) * n_z;
    for (int j = 0; j < n_z; j++) {
        R r = R(__ldg(flat + base0 + j)) * (1.0 - w) + R(__ldg(flat + base1 + j)) * w;
        if (decreasing ? (r > target) : (r < target)) cnt += 1;
    }
    int j = imin(imax(cnt - 1, 0), n_z - 2);
    R r0 = R(__ldg(flat + base0 + j)) * (1.0 - w) + R(__ldg(flat + base1 + j)) * w;
    R r1 = R(__ldg(flat + base0 + j + 1)) * (1.0 - w) + R(__ldg(flat + base1 + j + 1)) * w;
    R d = r1 - r0;
    R denom = (m_abs(d) > tiny_den<R>()) ? d : tiny_den<R>();
    R frac = pmin(pmax((target - r0) / denom, 0.0), 1.0);
    return pmin(pmax(((R)j + frac) / (R)(n_z - 1), 0.0), 1.0);
}

template <class R, class T>
__device__ R h_Ps_vapor(const Tab<R, T>& tab, R P, R s) {
    int i; R w;
    pidx(tab, P, i, w);
    return sat_row(tab, SAT_H_V, i, w)
           + invert_row<R, T>(tab.vap, REG_S, tab.n_p, tab.n_z, i, w, s, false) * sat_row(tab, SAT_DHMAX_V, i, w);
}

// ----------------------------------------------------------- parameters
template <class R> struct Par {
    const R* __restrict__ b;     // structure of arrays: field k of stand e at k * n + e
    int n, e;
    __device__ __forceinline__ R operator()(int k) const { return __ldg(b + (size_t)k * n + e); }
};
#define PV(name) p(PI_##name)

// ----------------------------------------------------------- components
template <class R> __device__ __forceinline__ R regroot(R z, R eps) {
    return z / m_pow(z * z + eps * eps, 0.25);
}
template <class R> __device__ __forceinline__ R smoothstep(R t) {
    t = pmin(pmax(t, 0.0), 1.0);
    return t * t * (3.0 - 2.0 * t);
}
template <class R> __device__ __forceinline__ R valve_fraction(R u, R code, R Rv) {
    u = pmin(pmax(u, 0.0), 1.0);
    if (code == 0.0) return u;
    if (code == 1.0) return (m_pow(Rv, u) - 1.0) / (Rv - 1.0);
    return m_sqrt(u);
}
template <class R> __device__ __forceinline__ R kv_to_C(R Kv) { return Kv * (1.0 / 36000.0); }
template <class R>
__device__ __forceinline__ R train_C(R u, R Kv, R code, R Rv, R Kv2, R split, R Kv_bp, R b) {
    const R KVC = 1.0 / 36000.0;
    u = pmin(pmax(u, 0.0), 1.0);
    R C;
    if (Kv2 > 0.0)
        C = Kv2 * KVC * valve_fraction(pmin(u / split, 1.0), code, Rv)
            + Kv * KVC * valve_fraction(pmax((u - split) / (1.0 - split), 0.0), code, Rv);
    else
        C = Kv * KVC * valve_fraction(u, code, Rv);
    if (Kv_bp > 0.0 && b > 0.0) C += Kv_bp * KVC * valve_fraction(b, (R)1.0, 50.0);
    return C;
}
template <class R> __device__ __forceinline__ R series_share(R C, R rho_up, R C_hx, R rho_hx) {
    if (C_hx <= 0.0) return 1.0;
    R r = C / pmax(C_hx, 1e-12);
    return 1.0 / (1.0 + r * r * rho_up / pmax(rho_hx, 1e-9));
}
template <class R>
__device__ R gas_valve_flow_s(R C, R P1, R P2, R rho1, R rho2, R kappa, R xT, R eps, R C_hx, R rho_hx) {
    R dP = P1 - P2;
    bool fwd = dP >= 0.0;
    R P_up = fwd ? P1 : P2;
    R rho_up = fwd ? rho1 : rho2;
    R Fk = kappa / 1.4;
    R x = m_abs(dP) / P_up;
    R x_ch = Fk * xT;
    R Y = 1.0 - pmin(x, x_ch) / (3.0 * x_ch);
    if (C_hx > 0.0) {
        for (int it = 0; it < 2; it++) {
            R share = series_share(C * Y, rho_up, C_hx, rho_hx);
            Y = 1.0 - pmin(x * share, x_ch) / (3.0 * x_ch);
        }
    }
    R share = series_share(C * Y, rho_up, C_hx, rho_hx);
    R x_eff = pmin(x, x_ch / share);
    R dPe = (dP != 0.0) ? m_copysign(x_eff * P_up, dP) : (R)0.0;
    return C * Y * m_sqrt(share * rho_up) * regroot(dPe, eps);
}
template <class R>
__device__ R liquid_valve_flow_s(R C, R P1, R P2, R rho1, R rho2, R f_choke, R eps, R C_hx, R rho_hx) {
    R dP = P1 - P2;
    bool fwd = dP >= 0.0;
    R P_up = fwd ? P1 : P2;
    R rho_up = fwd ? rho1 : rho2;
    R share = series_share(C, rho_up, C_hx, rho_hx);
    R dPe = (dP != 0.0) ? m_copysign(pmin(m_abs(dP), f_choke * P_up / share), dP) : (R)0.0;
    return C * m_sqrt(share * rho_up) * regroot(dPe, eps);
}
template <class R> __device__ R martin_xi(R Re, R beta_deg) {
    Re = pmax(Re, 1e-6);
    R xi0, xi1;
    if (Re < 2000.0) {
        xi0 = 64.0 / Re;
        xi1 = 597.0 / Re + 3.85;
    } else {
        xi0 = 1.0 / sq(1.8 * m_log10(Re) - 1.5);
        xi1 = 39.0 / m_pow(Re, 0.289);
    }
    R be = beta_deg * (pi_<R>() / 180.0);
    R c = m_cos(be);
    R inv = c / m_sqrt(0.18 * m_tan(be) + 0.36 * m_sin(be) + xi0 / c) + (1.0 - c) / m_sqrt(3.8 * xi1);
    return 1.0 / (inv * inv);
}
template <class R> __device__ R amalfi_ftp(R G, R d_h, R rho_m, R rho_l, R rho_v, R sigma, R beta_deg) {
    R We = pmax(G * G * d_h / (rho_m * sigma), 1e-12);
    R Bd = (rho_l - rho_v) * 9.81 * d_h * d_h / sigma;
    R C = 2.125 * m_pow(beta_deg / 70.0, 9.993) + 0.955;
    return C * 15.698 * m_pow(We, -0.475) * m_pow(Bd, 0.255) * m_pow(rho_l / rho_v, -0.571);
}
template <class R> __device__ __forceinline__ R series_C3(R C1, R C2, R C3) {
    return 1.0 / m_sqrt(1.0 / sq(pmax(C1, 1e-30)) + 1.0 / sq(pmax(C2, 1e-30)) + 1.0 / sq(pmax(C3, 1e-30)));
}
template <class R> __device__ __forceinline__ R hx_drop(R mdot, R C, R rho) {
    return mdot * m_abs(mdot) / (C * C * pmax(rho, 1e-9));
}
template <class R> __device__ __forceinline__ R water_valve_flow(R C, R dP, R rho) {
    return C * m_sqrt(rho * pmax(dP, 0.0));
}
template <class R> __device__ R churchill_f(R Re, R rel_rough) {
    Re = pmax(Re, 1.0);
    // an even power: of the magnitude (the same value; fast-math pow is NaN for a negative base)
    R A = m_pow(m_abs(2.457 * m_log(m_pow(7.0 / Re, 0.9) + 0.27 * rel_rough)), 16.0);
    R B = m_pow(37530.0 / Re, 16.0);
    return 8.0 * m_pow(m_pow(8.0 / Re, 12.0) + m_pow(A + B, -1.5), 1.0 / 12.0);
}
template <class R> __device__ R pipe_drop_s(R mdot, R d, R L, R K, R rho, R mu, R rough) {
    R A = 0.25 * pi_<R>() * d * d;
    R Re = m_abs(mdot) * d / (A * pmax(mu, 1e-7));
    R f = churchill_f(Re, rough / pmax(d, 1e-4));
    return mdot * m_abs(mdot) / (2.0 * pmax(rho, 1e-6) * A * A) * (f * L / d + K);
}
template <class R> __device__ __forceinline__ R mixture_viscosity(R x, R mu_l, R mu_v) {
    x = pmin(pmax(x, 0.0), 1.0);
    return 1.0 / (x / mu_v + (1.0 - x) / mu_l);
}
template <class R> __device__ __forceinline__ R actuator_rate(R u_pos, R u_cmd, R tau, R rate_max) {
    u_cmd = pmin(pmax(u_cmd, 0.0), 1.0);
    return pmin(pmax((u_cmd - u_pos) / tau, -rate_max), rate_max);
}

template <class R> struct Comp { R mdot, h2, h2_ad, T2_ad, eta_v, eta_s, W_shaft, W_el, Q_motor, Q_gs, Pr; };

template <class R, class T>
__device__ Comp<R> compressor_s(const Par<R>& p, const Tab<R, T>& tab, R P_s, R h_s, R s_s, R rho_s, R P_d, R N,
                                R T_shell) {
    Comp<R> c;
    R Nr = pmax(N, 0.0);
    R Pr = pmax(P_d / P_s, 1.0);
    R eta_v = pmin(pmax(PV(eta_v0) - PV(c_cl) * (m_pow(Pr, 1.0 / PV(kappa)) - 1.0), 0.0), 1.0);
    R mdot = eta_v * rho_s * PV(V_disp) * Nr / 60.0;
    R h2s = pmax(h_Ps_vapor(tab, P_d, s_s), h_s);
    R eta_s = PV(eta_s0) * (1.0 - PV(a_s) * sq(Pr - PV(Pr_opt))) * (1.0 - PV(b_N) * sq(Nr / PV(N_nom) - 1.0));
    eta_s = pmin(pmax(eta_s, 0.25), 0.95);
    R a = PV(f_motor_gas) * (1.0 / PV(eta_motor) - 1.0) / eta_s;
    R h_s1 = (h_s + a * h2s) / (1.0 + a);
    R dh_ad = pmax(h2s - h_s1, 0.0) / eta_s;
    R h2_ad = h_s1 + dh_ad;
    R W_shaft = mdot * dh_ad;
    R W_el = W_shaft / PV(eta_motor);
    R Q_motor = W_el - W_shaft;
    R T2_ad = T_vapor(tab, P_d, h2_ad);
    R C_g = mdot * PV(cp_gas);
    R eps = 1.0 - m_exp(-PV(UA_gs) / pmax(C_g, 1e-9));
    R Q_gs = eps * C_g * (T2_ad - T_shell);
    R h2 = h2_ad - ((mdot > 1e-9) ? Q_gs / pmax(mdot, 1e-9) : (R)0.0);
    c.mdot = mdot; c.h2 = h2; c.h2_ad = h2_ad; c.T2_ad = T2_ad; c.eta_v = eta_v; c.eta_s = eta_s;
    c.W_shaft = W_shaft; c.W_el = W_el; c.Q_motor = Q_motor; c.Q_gs = Q_gs; c.Pr = Pr;
    return c;
}

template <class R>
__device__ __forceinline__ void cv_balance(R V, R rho, R drho_dP, R drho_dh, R dm, R E, R& dP, R& dh) {
    R denom = V * (drho_dP + drho_dh / rho);
    dP = (dm - drho_dh * E / rho) / denom;
    dh = (E + V * dP) / (rho * V);
}

// ----------------------------------------------------------- model helpers
template <class R> __device__ __forceinline__ R softplus(R v, R w) {
    if (v >= w) return v;
    if (v <= -w) return 0.0;
    return (v + w) * (v + w) / (4.0 * w);
}
template <class R> __device__ __forceinline__ R nanmax(R a, R b) {
    if (b > a || b != b) return b;
    return a;
}
template <class R> __device__ __forceinline__ R pipe(R D, R t, R L, R K, R mdot, R rho, R mu, R share) {
    return pipe_drop_s(mdot, pmax(D - 2.0 * t, 1e-4), share * L, share * K, rho, mu, R(PIPE_ROUGHNESS_C));
}
template <class R> __device__ __forceinline__ R equiv_C(R m0, R dP0, R rho_ref) {
    dP0 = m_abs(dP0);
    return (dP0 > 1.0) ? m_abs(m0) / m_sqrt(rho_ref * pmax(dP0, 1.0)) : (R)1e3;
}
template <class R> __device__ __forceinline__ R valve_slope(R Cf, R rho_up, R dP, R eps) {
    R z2 = dP * dP;
    R e2 = eps * eps;
    return Cf * m_sqrt(pmax(rho_up, 1e-9)) * (0.5 * z2 + e2) / m_pow(z2 + e2, 1.25);
}
template <class R> __device__ void pool_split(const Par<R>& p, R V_pool, R& rec_level, R& ll_fill) {
    R seg0 = p(PI_seg_i + 0), seg1 = p(PI_seg_i + 1), seg2 = p(PI_seg_i + 2);
    R part0 = clip(V_pool, (R)0.0, seg0);
    R part2 = clip(V_pool - seg0 - seg1, (R)0.0, seg2);
    rec_level = (part0 + part2) / PV(rec_V);
    R w_ll = pmax(seg1, 2e-4);
    ll_fill = clip((V_pool - seg0) / w_ll, (R)0.0, (R)1.0);
}
template <class R> __device__ __forceinline__ R liquid_share(R x, R r23) {
    if (x >= 1.0) return 0.0;
    if (x <= 0.0) return 1.0;
    return 1.0 - 1.0 / (1.0 + (1.0 - x) / x * r23);
}
template <class R> __device__ R film_capacity(R r23) {
    const int n = MX;
    R f = 0.0;
    for (int j = 0; j < n; j++) f += liquid_share(1.0 - ((R)j + 0.5) / (R)n, r23) / (R)n;
    return f;
}

template <class R>
__device__ void quench_drops(const Par<R>& p, R mdot, R rho_in, const R* rho_c, const R* x_c, R rho_l, R rho_v,
                             R mu_l, R mu_v, R sigma, R* seg, R& dist, R& inlet, R& outlet) {
    const int n = MX;
    R m = m_abs(mdot);
    R sgn = (mdot == 0.0) ? (R)0.0 : m_copysign((R)1.0, mdot);
    R dh = 2.0 * PV(mx_b) / PV(mx_phi);
    R Gc = m / ((PV(V_mx_q) / PV(mx_V_ch)) * PV(mx_b) * PV(mx_W));
    R dz = PV(mx_H) / (R)n;
    R beta = PV(mx_beta);
    for (int j = 0; j < n; j++) {
        R f_tp = amalfi_ftp(Gc, dh, rho_c[j], rho_l, rho_v, sigma, beta);
        R dp_tp = 2.0 * f_tp * Gc * Gc * dz / (dh * rho_c[j]);
        R mu = (x_c[j] >= 0.5) ? mu_v : mu_l;
        R dp_sp = martin_xi(Gc * dh / mu, beta) * dz / dh * Gc * Gc / (2.0 * rho_c[j]);
        R w_sp = smoothstep((x_c[j] - 0.975) / 0.05) + smoothstep((0.025 - x_c[j]) / 0.05);
        R fric = sgn * ((1.0 - w_sp) * dp_tp + w_sp * dp_sp);
        R v_prev = (j == 0) ? 1.0 / rho_in : 1.0 / rho_c[j - 1];
        R acc = (mdot > 0.0) ? Gc * Gc * (1.0 / rho_c[j] - v_prev) : (R)0.0;
        seg[j] = fric + acc;
    }
    R A_p = 0.25 * pi_<R>() * sq(PV(mx_d_port)), A_c = 0.25 * pi_<R>() * sq(PV(mx_d_S34));
    R d_mo = pmax(PV(D_mo) - 2.0 * PV(t_mo), PV(mx_d_S34));
    R a_cp = sq(PV(mx_d_S34) / PV(mx_d_port));
    R K_in = sq(1.0 - a_cp);
    R K_out = 0.5 * (1.0 - a_cp) + sq(1.0 - sq(PV(mx_d_S34) / d_mo));
    R rho_out = rho_c[n - 1];
    dist = mdot * m / (sq(kv_to_C(PV(mx_Kv_dist))) * rho_in);
    inlet = dist + 0.75 * (mdot * m / (A_p * A_p * 2.0 * rho_in)) + K_in * (mdot * m / (A_c * A_c * 2.0 * rho_in));
    outlet = 0.75 * (mdot * m / (A_p * A_p * 2.0 * rho_out)) + K_out * (mdot * m / (A_c * A_c * 2.0 * rho_out));
}

template <class R>
__device__ __forceinline__ R quench_heat(R Hq, R Tq, R T_mw, R H_up, R dTs, R phi_f, R Q_2ph, R UA_e, R UA_v, R h_vs) {
    R w_wet = 1.0 - smoothstep((Hq - h_vs) / R(DRYOUT_BAND_C));
    Tq = Tq + w_wet * dTs;
    R phi_h = clip((h_vs - H_up) / pmax(Hq - H_up, 1.0), (R)0.0, (R)1.0);
    R phi = pmin(phi_f, (Hq > H_up) ? phi_h : (R)1.0);
    R Q_dry = phi * Q_2ph + (1.0 - phi) * UA_v * (T_mw - Tq);
    return w_wet * UA_e * (T_mw - Tq) + (1.0 - w_wet) * Q_dry;
}

template <class R, class T>
__device__ R cond_cell_heat(const Tab<R, T>& tab, R P, const Sat<R>& s, R h_up, R Tc, R Tw, R m, R A, R a2, R a1,
                            R w_flow, R T_iv) {
    if (A <= 0.0) return 0.0;
    R w_wet = smoothstep((Tc - Tw) / R(WET_BAND_C) + 0.5);
    R Qs_wet, Qs_dry;
    if (h_up > s.h_v && m > 0.0) {
        R dT = T_vapor(tab, P, h_up) - s.T_v;
        R cp = clip((dT > 0.05) ? (h_up - s.h_v) / dT : s.cp_v, (R)300.0, (R)1e5);
        R eps = 1.0 - m_exp(-a1 * A / (m * cp));
        Qs_wet = eps * m * (h_up - s.h_v);
        Qs_dry = eps * m * (h_up - s.h_v - cp * (Tw - s.T_v));
    } else {
        Qs_wet = 0.0;
        Qs_dry = a1 * A * (Tc - Tw);
    }
    R Q_dry = w_flow * Qs_dry + (1.0 - w_flow) * a1 * A * (T_iv - Tw);
    return w_wet * (a2 * A * (Tc - Tw) + Qs_wet) + (1.0 - w_wet) * Q_dry;
}

template <class R, class T>
__device__ R condenser(const Par<R>& p, const Tab<R, T>& tab, R P, const Sat<R>& s, const R* T_w, const R* f_fl,
                       R h_hot, R h_vap, R T_iv, R m_hot, R* Q, R* h_mid, R* h_out, R* Tc) {
    const int n = MX;
    R A = PV(cond_A) / (R)n;
    R a2 = PV(alpha_r_2ph), a1 = PV(alpha_r_1ph);
    R dh_lv = pmax(s.h_v - s.h_l, 1e3);
    R m = 0.0;
    for (int j = 0; j < n; j++) {
        Tc[j] = 0.5 * (s.T_l + s.T_v);
        if (T_w[j] < Tc[j]) m += a2 * A * (1.0 - f_fl[j]) * (Tc[j] - T_w[j]);
    }
    m /= dh_lv;
    R w = 0.0;
    for (int it = 0; it < 3; it++) {
        R s_hot = (m > 1e-12) ? pmin(m_hot / m, 1.0) : ((m_hot > 0.0) ? (R)1.0 : (R)0.0);
        R h_in = s_hot * h_hot + (1.0 - s_hot) * h_vap;
        w = smoothstep(m / R(CF_STAGNANT_C));
        R h = h_in;
        R Q_sum = 0.0;
        for (int j = 0; j < n; j++) {
            Q[j] = cond_cell_heat(tab, P, s, h, Tc[j], T_w[j], m, A * (1.0 - f_fl[j]), a2, a1, w, T_iv);
            Q_sum += Q[j];
            h = pmax(h - Q[j] / pmax(m, 1e-9), s.h_l);
        }
        m = pmax(Q_sum, 0.0) / pmax(h_in - s.h_l, 1e4);
        h = h_in;
        for (int j = 0; j < n; j++) {
            R h_o = h - Q[j] / pmax(m, 1e-9);
            h_mid[j] = clip(0.5 * (h + h_o), s.h_l, pmax(h_in, h_vap));
            h_out[j] = clip(h_o, s.h_l, pmax(h_in, h_vap));
            Tc[j] = s.T_l + clip((h_mid[j] - s.h_l) / dh_lv, (R)0.0, (R)1.0) * (s.T_v - s.T_l);
            h = h_o;
        }
    }
    for (int j = 0; j < n; j++) {
        h_mid[j] = w * h_mid[j] + (1.0 - w) * h_vap;
        h_out[j] = w * h_out[j] + (1.0 - w) * h_vap;
    }
    return m;
}

// ------------------------------------------------------------------ RHS
// model.rhs with hold_P_s = False and want_aux = False: writes dx and the bypass gas
// side enthalpies hg_out, returns the fastest local rate
template <class R, class T>
__device__ R rhs(const R* __restrict__ x, const R* u_cmd, R N_cmd, R T_amb, R T_wi, const Par<R>& p,
                 const Tab<R, T>& tab, R* __restrict__ dx, R* __restrict__ hg_out) {
    const int n = MX;
    R P_s = x[XI_P_s];
    const R* H = x + XI_h_q0;
    const R* T_mw = x + XI_T_mw0;
    R T_sw = x[XI_T_sw];
    R P_d = x[XI_P_d], h_d = x[XI_h_d], T_dw = x[XI_T_dw];
    R P_i = x[XI_P_i], h_i = x[XI_h_i], T_rw = x[XI_T_rw];
    const R* T_cw = x + XI_T_cw0;
    R T_sh = x[XI_T_sh], N = x[XI_N];
    R u1 = x[XI_u1], u2 = x[XI_u1 + 1], u3 = x[XI_u1 + 2], u4 = x[XI_u1 + 3];
    R h_l1 = H[n], h_l2 = H[n + 1];

    // ---- suction side cells (common pressure)
    R c_rho[NS], c_T[NS], c_x[NS], rho_P[NS], rho_h[NS], M_c[NS];
    R T_sat_s = 0.0, h_ls = 0.0, h_vs = 0.0;
    for (int j = 0; j < NS; j++) {
        PhS<R> S = state(tab, P_s, H[j]);
        c_rho[j] = S.rho; c_T[j] = S.T; c_x[j] = S.x; rho_P[j] = S.drho_dP; rho_h[j] = S.drho_dh;
        M_c[j] = S.rho * p(PI_V_sc + j);
        T_sat_s = S.T_sat; h_ls = S.h_l; h_vs = S.h_v;
    }
    Sat<R> sat_s = sat(tab, P_s);
    PhS<R> D = state(tab, P_d, h_d);
    PhS<R> I = state(tab, P_i, h_i);

    // ---- compressor
    R h_cin = pmax(h_l2, h_ls + PV(comp_x_min) * (h_vs - h_ls));
    PhS<R> Cin = state(tab, P_s, h_cin);
    Comp<R> cp = compressor_s(p, tab, P_s, h_cin, Cin.s, Cin.rho, P_d, N, T_sh);
    R mdot_c = cp.mdot, h2 = cp.h2, W_el = cp.W_el, Q_motor = cp.Q_motor, Q_gs = cp.Q_gs;

    // ---- intermediate section
    Sat<R> sat_i = sat(tab, P_i);
    R hfg_i = pmax(sat_i.h_v - sat_i.h_l, 1e3);
    R M_L = pmax(x[XI_M_L], 0.0);
    R h_L = x[XI_h_L];
    PhS<R> Lq = state(tab, P_i, pmin(h_L, sat_i.h_l));
    R T_L = Lq.T + pmax(h_L - sat_i.h_l, 0.0) / Lq.cp_l;
    R V_pool = M_L / Lq.rho;
    R V_pmax = PV(rec_V) + PV(V_line_liq);
    R V_C = pmax(PV(V_i) - V_pool, 1e-3 * PV(V_i));
    R s_full = smoothstep((V_pool / V_pmax - 1.0) / R(POOL_FULL_C) + 1.0);
    R rec_level, ll_fill;
    pool_split(p, V_pool, rec_level, ll_fill);
    R dry = 1.0 - smoothstep(ll_fill);
    R M_C = I.rho * V_C;
    R M_lC = (1.0 - clip(I.x, (R)0.0, (R)1.0)) * M_C;
    R r23 = m_pow(sat_i.rho_v / sat_i.rho_l, 2.0 / 3.0);
    R V_ex = M_lC / sat_i.rho_l - PV(V_cond) * film_capacity(r23) + pmax(V_pool - V_pmax, 0.0) - PV(V_line_drn);
    R cond_flood = pmin(softplus(V_ex, R(FLOOD_BAND_C) * PV(V_cond)) / PV(V_cond), 1.0);
    R h_vap_i = pmax(h_i, I.h_v);
    R h_iv = h_vap_i;
    R h_cv_out = h_L * (1.0 - dry) + h_vap_i * dry;
    PhS<R> S_co = state(tab, P_i, h_cv_out);
    R rho_co = S_co.rho;

    // ---- condenser cells flooding, condensing flow
    R f_fl[MX];
    R m_cond = 0.0;
    R T_c2 = 0.5 * (sat_i.T_l + sat_i.T_v);
    for (int j = 0; j < n; j++) {
        f_fl[j] = clip(cond_flood * (R)n - (R)(n - 1 - j), (R)0.0, (R)1.0);
        m_cond += PV(alpha_r_2ph) * (PV(cond_A) / (R)n) * (1.0 - f_fl[j]) * pmax(T_c2 - T_cw[j], 0.0);
    }
    R mdot_cr = m_cond / hfg_i;
    R w2 = smoothstep(mdot_cr / R(CF_STAGNANT_C)) * (1.0 - smoothstep(-I.x / 0.05));
    R a_c = w2 * (1.0 - cond_flood);

    // ---- condenser pressure drop
    R rho_v = (I.x >= 1.0) ? I.rho : sat_i.rho_v;
    R rho_l = sat_i.rho_l;
    R rho_fr = w2 * 2.0 / (1.0 / rho_v + 1.0 / rho_l) + (1.0 - w2) * rho_v;
    R r_lv = pmax(rho_l / rho_v, 1.001);
    R rho_hm = rho_v * m_log(r_lv) / (1.0 - 1.0 / r_lv);
    R rho_col = cond_flood * rho_l + a_c * rho_hm + (1.0 - cond_flood - a_c) * rho_v;
    R dP_cr = (1.0 - cond_flood) * hx_drop(mdot_cr, kv_to_C(PV(cond_Kv_r)), rho_fr) - R(G_C) * PV(cond_H) * rho_col;

    // ---- pressure chain of the intermediate side
    R mu_l_i, mu_v_i, sig_i, mu_l_d, mu_v_d, sig_d, mu_l_s, mu_v_s, sigma_s;
    transport(tab, P_i, mu_l_i, mu_v_i, sig_i);
    transport(tab, P_d, mu_l_d, mu_v_d, sig_d);
    transport(tab, P_s, mu_l_s, mu_v_s, sigma_s);
    R mu_co = mixture_viscosity(clip(S_co.x, (R)0.0, (R)1.0), mu_l_i, mu_v_i);
    R dP_drn = pipe(PV(D_drn), PV(t_drn), PV(L_drn), PV(K_drn), mdot_cr, rho_l, mu_l_i, (R)1.0);
    R tmpT, rho_hg;
    vapor_props(tab, P_i + dP_drn + dP_cr, h_d, tmpT, rho_hg);
    R dP_hdr2 = pipe(PV(D_hdr), PV(t_hdr), PV(L_hdr), PV(K_hdr), mdot_cr, rho_hg, mu_v_i, (R)0.5);
    R P_h = pmax(P_i + dP_drn + dP_cr + dP_hdr2, tab.p_min * 1.001);

    // ---- suction side line
    R mu_l1 = mixture_viscosity(c_x[n], mu_l_s, mu_v_s);
    R dP_suc = pipe(PV(D_suc), PV(t_suc), PV(L_suc), PV(K_suc), mdot_c, c_rho[n], mu_l1, (R)1.0);
    R P_tee = P_s + dP_suc;

    // ---- valve 1
    R T_g, rho_g;
    vapor_props(tab, P_h, h_d, T_g, rho_g);
    R C1 = train_C(u1, PV(Kv_dpv), PV(dpv_code), PV(dpv_R), PV(Kv_dpv2), PV(dpv_split), PV(Kv_dpv_bp), PV(dpv_bp));
    R m0 = gas_valve_flow_s(C1, P_d, P_h, D.rho, rho_g, PV(kappa), PV(xT), PV(eps_valve), (R)0.0, (R)1.0);
    R dP0 = (pipe(PV(D_dis), PV(t_dis), PV(L_dis), PV(K_dis), m0, D.rho, mu_v_d, (R)1.0)
             + pipe(PV(D_hdr), PV(t_hdr), PV(L_hdr), PV(K_hdr), m0, rho_g, mu_v_i, (R)0.5));
    R mdot_1 = gas_valve_flow_s(C1, P_d, P_h, D.rho, rho_g, PV(kappa), PV(xT), PV(eps_valve),
                                equiv_C(m0, dP0, rho_g), rho_g);
    R h_1f = (mdot_1 >= 0.0) ? h_d : h_iv;

    // ---- valve 2
    R C2 = train_C(u2, PV(Kv_spv), PV(spv_code), PV(spv_R), PV(Kv_spv2), PV(spv_split), PV(Kv_spv_bp), PV(spv_bp));
    R rho_gin;
    vapor_props(tab, P_s, h_d, tmpT, rho_gin);
    R rho_gm = 2.0 / (1.0 / rho_gin + 1.0 / sat_s.rho_v);
    R head_g = R(G_C) * PV(mx_H) * rho_gm;
    R C_mg = kv_to_C(PV(mx_Kv_g));
    m0 = gas_valve_flow_s(C2, P_h, P_tee + head_g, rho_g, c_rho[n], PV(kappa), PV(xT), PV(eps_valve), (R)0.0, (R)1.0);
    dP0 = (pipe(PV(D_bp), PV(t_bp), PV(L_bp), PV(K_bp), m0, rho_gin, mu_v_s, (R)1.0) + hx_drop(m0, C_mg, rho_gm)
           + pipe(PV(D_mo), PV(t_mo), PV(L_mo), PV(K_mo), m0, sat_s.rho_v, mu_v_s, (R)0.5));
    R mdot_2 = gas_valve_flow_s(C2, P_h, P_tee + head_g, rho_g, c_rho[n], PV(kappa), PV(xT), PV(eps_valve),
                                equiv_C(m0, dP0, rho_gm), rho_gm);
    R m_from_inlet = pmax(pmin(mdot_2, pmax(mdot_1, 0.0)), 0.0);
    bool fwd2 = mdot_2 > 0.0;
    R h_2f;
    if (fwd2)
        h_2f = (m_from_inlet * h_d + (pmax(mdot_2, 0.0) - m_from_inlet) * h_iv) / pmax(mdot_2, 1e-12);
    else
        h_2f = h_l1;

    // ---- valve 3
    R f3 = valve_fraction(u3, PV(stv_code), PV(stv_R));
    const R* rho_c = c_rho;
    const R* x_c = c_x;
    R head_q[MX];
    R head_sum = 0.0;
    for (int j = 0; j < n; j++) {
        head_q[j] = R(G_C) * (PV(mx_H) / (R)n) * rho_c[j];
        head_sum += head_q[j];
    }
    PhS<R> S3 = state(tab, P_s, h_cv_out);
    R rho_3 = S3.rho;
    R mu_3 = mixture_viscosity(S3.x, mu_l_s, mu_v_s);
    R mu_qo = mixture_viscosity(x_c[n - 1], mu_l_s, mu_v_s);
    R C3 = kv_to_C(PV(Kv_stv)) * f3;
    R P_3dn = P_tee - head_sum;
    R seg[MX];
    m0 = liquid_valve_flow_s(C3, P_i, P_3dn, rho_co, rho_c[0], PV(f_choke_liq), PV(eps_valve), (R)0.0, (R)1.0);
    R q_dist, q_in, q_out;
    quench_drops(p, m0, rho_3, rho_c, x_c, sat_s.rho_l, sat_s.rho_v, mu_l_s, mu_v_s, sigma_s, seg, q_dist, q_in, q_out);
    R seg_sum = 0.0;
    for (int j = 0; j < n; j++) seg_sum += seg[j];
    dP0 = (pipe(PV(D_liq), PV(t_liq), PV(L_liq), PV(K_liq), m0, rho_co, mu_co, (R)1.0)
           + pipe(PV(D_q), PV(t_q), PV(L_q), PV(K_q), m0, rho_3, mu_3, (R)1.0) + q_in + seg_sum + q_out
           + pipe(PV(D_mo), PV(t_mo), PV(L_mo), PV(K_mo), m0, rho_c[n - 1], mu_qo, (R)0.5));
    R mdot_3 = liquid_valve_flow_s(C3, P_i, P_3dn, rho_co, rho_c[0], PV(f_choke_liq), PV(eps_valve),
                                   equiv_C(m0, dP0, rho_3), rho_3);
    bool fwd3 = mdot_3 >= 0.0;
    quench_drops(p, mdot_3, rho_3, rho_c, x_c, sat_s.rho_l, sat_s.rho_v, mu_l_s, mu_v_s, sigma_s, seg, q_dist, q_in,
                 q_out);
    R dP_moq = pipe(PV(D_mo), PV(t_mo), PV(L_mo), PV(K_mo), mdot_3, rho_c[n - 1], mu_qo, (R)0.5);
    R seg_q[MX];
    for (int j = 0; j < n; j++) seg_q[j] = seg[j] - head_q[j];
    R dTs_dP = sat_s.T_v * (1.0 / sat_s.rho_v - 1.0 / sat_s.rho_l) / pmax(sat_s.h_v - sat_s.h_l, 1e3);
    R dTs[MX];
    R below = 0.0;
    R base = dP_suc + dP_moq + q_out;
    for (int j = n - 1; j >= 0; j--) {
        below += seg_q[j];
        R dP_qc = base + below - 0.5 * seg_q[j];
        dTs[j] = dTs_dP * dP_qc;
    }

    // ---- walls of suction and discharge lines, receiver shell
    R Q_sg = PV(UA_sg) * (T_sw - c_T[n]);
    R dT_sw = (PV(UA_sa) * (T_amb - T_sw) - Q_sg) / PV(C_sw);
    R Q_dg = PV(UA_dg) * (T_dw - D.T);
    R dT_dw = (PV(UA_da) * (T_amb - T_dw) - Q_dg) / PV(C_dw);
    R w_wet = clip(rec_level, (R)0.0, (R)1.0);
    R Q_rwL = PV(rec_UA_r) * w_wet * (T_rw - T_L);
    R Q_rwV = PV(rec_UA_rv) * (1.0 - w_wet) * (T_rw - I.T);
    R Q_rw = Q_rwL + Q_rwV;
    R dT_rw = (PV(rec_UA_a) * (T_amb - T_rw) - Q_rw) / PV(C_rw);

    // ---- condenser, water side
    R f_w = valve_fraction(u4, PV(w_code), PV(w_R));
    R C_cw = kv_to_C(PV(cond_Kv_w));
    R mdot_w = water_valve_flow(series_C3(kv_to_C(PV(Kv_w)) * f_w, C_cw, kv_to_C(PV(Kv_wpipe))),
                                PV(P_w_sup) - PV(P_w_ret), PV(rho_w));
    R Cw = mdot_w * PV(cp_w);
    R UA_w = PV(alpha_w0) * PV(cond_A) * m_pow(pmax(mdot_w / PV(mdot_w_ref), 1e-6), 0.8);
    R T_co = S_co.T;
    R h_3i = fwd3 ? h_cv_out : H[0];
    R h_3f = h_3i;

    // ---- condenser, refrigerant -> walls
    R T_iv = (I.x >= 1.0) ? I.T : sat_i.T_v;
    R Q_c[MX], h_cm[MX], h_co_c[MX], Tc_s[MX];
    condenser(p, tab, P_i, sat_i, T_cw, f_fl, h_d, h_vap_i, T_iv, pmax(mdot_1 - pmax(mdot_2, 0.0), 0.0), Q_c, h_cm,
              h_co_c, Tc_s);
    R Q_r = 0.0;
    R M_film = 0.0;
    for (int j = 0; j < n; j++) {
        Q_r += Q_c[j];
        M_film += (PV(V_cond) / (R)n) * (1.0 - f_fl[j]) * sat_i.rho_l * liquid_share((h_cm[j] - sat_i.h_l) / hfg_i, r23);
    }

    // ---- condensate drain
    R m3L = (1.0 - dry) * mdot_3;
    R m_drn0 = pmax(M_lC - M_film, 0.0) / PV(cond_tau_drain);
    R m_drn = (1.0 - s_full) * m_drn0 + s_full * pmin(m_drn0, pmax(m3L, 0.0));
    R h_lC = (I.x < 0.0) ? h_i : sat_i.h_l;
    R T_lC = (I.x < 0.0) ? I.T : sat_i.T_l;
    R a_sc = pmax(cond_flood, PV(cond_sc_film));
    R o_sc[MX];
    R o_sum = 0.0, T_wsc = 0.0;
    for (int j = 0; j < n; j++) {
        o_sc[j] = clip(a_sc * (R)n - (R)(n - 1 - j), (R)0.0, (R)1.0);
        o_sum += o_sc[j];
        T_wsc += o_sc[j] * T_cw[j];
    }
    T_wsc /= pmax(o_sum, 1e-12);
    R C_l = m_drn * Lq.cp_l;
    R eps_sc = 1.0 - m_exp(-PV(alpha_sc) * PV(cond_A) * a_sc / pmax(C_l, 1e-9));
    R Q_sc = eps_sc * C_l * pmax(T_lC - T_wsc, 0.0);
    R h_drn = h_lC - Q_sc / pmax(m_drn, 1e-9);

    // ---- receiver pool
    R s_L = h_L - sat_i.h_l;
    R m_lv, h_lv;
    if (s_L > 0.0) {
        m_lv = -M_L * s_L / (hfg_i * PV(rec_tau_flash));
        h_lv = sat_i.h_v;
    } else {
        m_lv = (1.0 - s_full) * PV(rec_UA_lv) * (-s_L / Lq.cp_l) / hfg_i;
        h_lv = h_vap_i;
    }

    // ---- condenser, walls -> water
    R T_wat = T_wi;
    R dT_cw[MX];
    R eps_w = 1.0 - m_exp(-UA_w / (R)n / pmax(Cw, 1e-9));
    for (int j = n - 1; j >= 0; j--) {
        R dTw = eps_w * (T_cw[j] - T_wat);
        R Q_wj = Cw * dTw;
        T_wat += dTw;
        R Q_scj = Q_sc * o_sc[j] / pmax(o_sum, 1e-12);
        dT_cw[j] = (Q_c[j] + Q_scj - Q_wj + (PV(UA_ca) / (R)n) * (T_amb - T_cw[j])) / (PV(C_cw) / (R)n);
    }

    // ---- mixing exchanger, gas side
    R m2p = pmax(mdot_2, 0.0);
    R UA_g = PV(mx_alpha_g0) * PV(A_mx_cell) * m_pow(pmax(m2p, 1e-3 * PV(mx_mdot_g_ref)) / PV(mx_mdot_g_ref), 0.8);
    R hg = h_2f;
    R Q_g[MX], rho_gc[MX];
    R Q_g_sum = 0.0;
    R T_up = T_vapor(tab, P_s, hg);
    for (int j = n - 1; j >= 0; j--) {
        R Tw_eff = pmax(T_mw[j], T_sat_s);
        R h_eq = h_vs + sat_s.cp_v * pmax(Tw_eff - sat_s.T_v, 0.0);
        R dT = T_up - Tw_eff;
        R dh = hg - h_eq;
        R cp_s = clip((m_abs(dT) > 0.05) ? dh / dT : (R)1100.0, (R)300.0, (R)1e5);
        R Q = (1.0 - m_exp(-UA_g / pmax(m2p * cp_s, 1e-9))) * m2p * dh;
        hg = hg - Q / pmax(m2p, 1e-12);
        Q_g[j] = Q;
        hg_out[j] = hg;
        vapor_props(tab, P_s, hg, T_up, rho_gc[j]);
    }
    for (int j = 0; j < n; j++) Q_g_sum += Q_g[j];
    R h_go = hg;
    R Fg_b = 0.0;
    for (int j = 0; j < n; j++) Fg_b += p(PI_V_gc + j) * (1.2 * rho_gc[j] / P_s);
    Fg_b = -Fg_b;

    // ---- mixing exchanger, quench side
    R m3p = pmax(mdot_3, 0.0);
    R fq = pmax(m3p / PV(mx_mdot_q_ref), 0.04);
    R UA_e = PV(mx_alpha_e) * PV(A_mx_cell) * m_sqrt(fq);
    R UA_v = PV(mx_alpha_v0) * PV(A_mx_cell) * m_pow(fq, 0.8);
    R H_up[MX], phi_f[MX], Q_2ph[MX], Q_q[MX], dT_mw[MX];
    R Q_q_sum = 0.0;
    for (int j = 0; j < n; j++) {
        if (fwd3)
            H_up[j] = (j == 0) ? h_3f : H[j - 1];
        else
            H_up[j] = (j < n - 1) ? H[j + 1] : h_l1;
        R dT_sat = T_mw[j] - (T_sat_s + dTs[j]);
        Q_2ph[j] = UA_e * dT_sat;
        R need = m_abs(mdot_3) * pmax(h_vs - H_up[j], 0.0);
        phi_f[j] = clip(need / pmax(UA_e * pmax(dT_sat, 0.0), 1e-9), (R)0.0, (R)1.0);
        Q_q[j] = quench_heat(H[j], c_T[j], T_mw[j], H_up[j], dTs[j], phi_f[j], Q_2ph[j], UA_e, UA_v, h_vs);
        Q_q_sum += Q_q[j];
        dT_mw[j] = (Q_g[j] - Q_q[j] + (PV(mx_UA_a) / (R)n) * (T_amb - T_mw[j])) / PV(C_mw_cell);
    }

    // ---- lumped-pressure balances of the suction side
    R A_face[MX + 1], B_face[MX + 1];
    R a_f = mdot_3, b_f = 0.0;
    A_face[0] = mdot_3; B_face[0] = 0.0;
    for (int j = 0; j < n; j++) {
        R h_up, adv_a;
        if (fwd3) {
            h_up = (j == 0) ? h_3f : H[j - 1];
            adv_a = a_f;
        } else {
            h_up = (j < n - 1) ? H[j + 1] : h_l1;
            adv_a = -mdot_3;
        }
        R Vj = p(PI_V_sc + j);
        R c = (adv_a * (h_up - H[j]) + Q_q[j]) / M_c[j];
        R d = Vj / M_c[j];
        a_f = a_f - Vj * rho_h[j] * c;
        b_f = b_f - Vj * (rho_P[j] + rho_h[j] * d);
        A_face[j + 1] = a_f; B_face[j + 1] = b_f;
    }
    R Fq_a = a_f, Fq_b = b_f;
    R Fg_a = mdot_2;
    R wq = fwd3 ? (R)1.0 : (R)0.0;
    R wg = fwd2 ? (R)1.0 : (R)0.0;
    R dhq = H[n - 1] - h_l1, dhg = h_go - h_l1;
    R Vn = p(PI_V_sc + n), Vn1 = p(PI_V_sc + n + 1);
    R c = (wg * Fg_a * dhg + wq * Fq_a * dhq + Q_sg) / M_c[n];
    R d = Vn / M_c[n];
    R F12_a = Fg_a + Fq_a - Vn * rho_h[n] * c;
    R F12_b = Fg_b + Fq_b - Vn * (rho_P[n] + rho_h[n] * d);
    R w12 = (F12_a > 0.0) ? (R)1.0 : (R)0.0;
    c = (w12 * F12_a * (h_l1 - h_l2) - mdot_c * (h_cin - h_l2)) / M_c[n + 1];
    d = Vn1 / M_c[n + 1];
    R Fo_a = F12_a - Vn1 * rho_h[n + 1] * c;
    R Fo_b = F12_b - Vn1 * (rho_P[n + 1] + rho_h[n + 1] * d);
    R dP_s = (mdot_c - Fo_a) / pmin(Fo_b, -1e-15);
    R Pr = dP_s;
    R F[MX + 1];
    for (int j = 0; j < n + 1; j++) F[j] = A_face[j] + B_face[j] * Pr;
    R Fg_now = Fg_a + Fg_b * Pr;
    R F12 = F12_a + F12_b * Pr;
    for (int j = 0; j < n; j++) {
        R h_above = (j == 0) ? h_3f : H[j - 1];
        R h_below = (j < n - 1) ? H[j + 1] : h_l1;
        dx[XI_h_q0 + j] = (pmax(F[j], 0.0) * (h_above - H[j]) + pmax(-F[j + 1], 0.0) * (h_below - H[j]) + Q_q[j]
                           + p(PI_V_sc + j) * Pr) / M_c[j];
    }
    dx[XI_h_q0 + n] = (pmax(Fg_now, 0.0) * dhg + pmax(F[n], 0.0) * dhq + pmax(-F12, 0.0) * (h_l2 - h_l1) + Q_sg
                       + Vn * Pr) / M_c[n];
    dx[XI_h_q0 + n + 1] = (pmax(F12, 0.0) * (h_l1 - h_l2) - mdot_c * (h_cin - h_l2) + Vn1 * Pr) / M_c[n + 1];

    // ---- fastest local rates
    R k1 = valve_slope(C1, pmax(D.rho, rho_g), P_d - P_h, PV(eps_valve));
    R k2 = valve_slope(C2, rho_g, P_h - P_s, PV(eps_valve));
    R k3 = valve_slope(kv_to_C(PV(Kv_stv)) * f3, rho_co, P_i - P_s, PV(eps_valve));
    R C_d = PV(V_d) * pmax(D.drho_dP + D.drho_dh / D.rho, 1e-12);
    R C_i = V_C * pmax(I.drho_dP + I.drho_dh / I.rho, 1e-12);
    R C_s = pmax(-Fo_b, 1e-12);
    R dh_fd = 200.0;
    R lam = nanmax(nanmax(k1 / C_d, (k1 + k2 + k3) / C_i), (k2 + k3 + mdot_c / P_s) / C_s);
    R lam_q = 0.0;
    for (int j = 0; j < n; j++) {
        R dT_fd = (c_x[j] >= 1.0) ? dh_fd / 1100.0 : (R)0.0;
        R dQ = quench_heat(H[j] + dh_fd, c_T[j] + dT_fd, T_mw[j], H_up[j], dTs[j], phi_f[j], Q_2ph[j], UA_e, UA_v, h_vs)
               - Q_q[j];
        R F_cell = pmax(m_abs(F[j]), m_abs(F[j + 1]));
        R r = (F_cell + m_abs(dQ) / dh_fd) / M_c[j];
        lam_q = (j == 0) ? r : nanmax(lam_q, r);
    }
    lam = nanmax(lam, lam_q);
    R lam_L = 1.0 / PV(cond_tau_drain);
    if (s_L > 0.0) {
        R dTs_dP_i = sat_i.T_v * (1.0 / sat_i.rho_v - 1.0 / sat_i.rho_l) / hfg_i;
        lam_L = pmax(lam_L, (1.0 + M_L * Lq.cp_l * dTs_dP_i / (hfg_i * C_i)) / PV(rec_tau_flash));
    }
    lam = nanmax(lam, lam_L);

    // ---- tee and suction line probe
    R Fq = pmax(Fq_a + Fq_b * dP_s, 0.0) * wq;
    R Fg = pmax(Fg_a + Fg_b * dP_s, 0.0) * wg;
    R F_in = Fq + Fg;
    R x_qo = x_c[n - 1];
    R y_in = clip(1.0 - x_qo, (R)0.0, (R)1.0) * Fq / pmax(F_in, 1e-9);
    R t_res = M_c[n] / pmax(F_in, 1e-6);
    R y_eq = clip(1.0 - c_x[n], (R)0.0, (R)1.0);
    R y_liq = pmax(y_in * m_exp(-t_res / PV(tee_tau_evap)), y_eq);
    R h_vap = (y_liq > 1e-12) ? (h_l1 - y_liq * h_ls) / pmax(1.0 - y_liq, 1e-3) : h_l1;
    R T_port = (y_liq < 0.999) ? T_vapor(tab, P_s, h_vap) : c_T[n];

    // ---- compressor shell
    R dT_sh = (Q_gs + (1.0 - PV(f_motor_gas)) * Q_motor - PV(UA_sha) * (T_sh - T_amb)) / PV(C_shell);

    // ---- lumped volumes with their own pressure
    R dm_d = mdot_c - mdot_1;
    R E_d = mdot_c * (h2 - h_d) - mdot_1 * (h_1f - h_d) + Q_dg;
    R dP_d, dh_d;
    cv_balance(PV(V_d), D.rho, D.drho_dP, D.drho_dh, dm_d, E_d, dP_d, dh_d);
    R h_3v = fwd3 ? h_vap_i : H[0];
    R dM_L = m_drn + m_lv - m3L;
    R E_L = m_drn * (h_drn - h_L) + m_lv * (h_lv - h_L) + Q_rwL + (fwd3 ? (R)0.0 : -m3L * (H[0] - h_L));
    R dm_C = mdot_1 - mdot_2 - dry * mdot_3 - m_drn - m_lv;
    R E_C = (mdot_1 * (h_1f - h_i) - mdot_2 * (h_2f - h_i) - dry * mdot_3 * (h_3v - h_i) - m_drn * (h_lC - h_i)
             - m_lv * (h_lv - h_i) - Q_r + Q_rwV);
    R dP_i, dh_i;
    cv_balance(V_C, I.rho, I.drho_dP, I.drho_dh, dm_C + I.rho * dM_L / Lq.rho, E_C, dP_i, dh_i);
    R dh_L = (E_L + V_pool * dP_i) / pmax(M_L, 1e-3);
    R dm_i = mdot_1 - mdot_2 - mdot_3;
    R dU_i = mdot_1 * h_1f - mdot_2 * h_2f - mdot_3 * h_3i - Q_r - Q_sc + Q_rw;

    // ---- speed, actuators, sensors
    R Nc = (N_cmd > 0.0) ? clip(N_cmd, PV(N_min), PV(N_max)) : (R)0.0;
    R dN = clip((Nc - N) / PV(tau_N), -PV(ramp_N), PV(ramp_N));

    dx[XI_P_s] = dP_s;
    for (int j = 0; j < n; j++) {
        dx[XI_T_mw0 + j] = dT_mw[j];
        dx[XI_h_g0 + j] = 0.0;
    }
    dx[XI_T_sw] = dT_sw;
    dx[XI_P_d] = dP_d; dx[XI_h_d] = dh_d; dx[XI_T_dw] = dT_dw;
    dx[XI_P_i] = dP_i; dx[XI_h_i] = dh_i; dx[XI_h_L] = dh_L; dx[XI_T_rw] = dT_rw;
    for (int j = 0; j < n; j++) dx[XI_T_cw0 + j] = dT_cw[j];
    dx[XI_T_sh] = dT_sh; dx[XI_N] = dN;
    dx[XI_u1] = actuator_rate(u1, u_cmd[0], PV(tau_dpv), PV(rate_dpv));
    dx[XI_u1 + 1] = actuator_rate(u2, u_cmd[1], PV(tau_spv), PV(rate_spv));
    dx[XI_u1 + 2] = actuator_rate(u3, u_cmd[2], PV(tau_stv), PV(rate_stv));
    dx[XI_u1 + 3] = actuator_rate(u4, u_cmd[3], PV(tau_w), PV(rate_w));
    dx[XI_Tm_s] = (T_port - x[XI_Tm_s]) / PV(tau_T);
    dx[XI_Tm_d] = (D.T - x[XI_Tm_d]) / PV(tau_T);
    dx[XI_Tm_co] = (T_co - x[XI_Tm_co]) / PV(tau_T);
    dx[XI_mm] = (mdot_c - x[XI_mm]) / PV(tau_m);
    dx[XI_Wm] = (W_el - x[XI_Wm]) / PV(tau_W);
    dx[XI_M_s] = mdot_2 + mdot_3 - mdot_c;
    dx[XI_U_s] = mdot_2 * h_2f + mdot_3 * h_3f - mdot_c * h_cin + Q_sg + Q_q_sum - Q_g_sum;
    dx[XI_M_d] = dm_d;
    dx[XI_U_d] = E_d + h_d * dm_d;
    dx[XI_M_i] = dm_i;
    dx[XI_U_i] = dU_i;
    dx[XI_M_L] = dM_L;
    return lam;
}

// -------------------------------------------------------- conserved states
template <class R, class T>
__device__ void suction_inventory(const R* x, const Par<R>& p, const Tab<R, T>& tab, R delta, R& M, R& U, R& MP,
                                  R& Md, R& UP, R& Ud) {
    R P = x[XI_P_s];
    R Mc = 0.0, Uc = 0.0, MPc = 0.0, Mdc = 0.0, UPc = 0.0, Udc = 0.0;
    for (int j = 0; j < NS; j++) {
        R H = x[XI_h_q0 + j] + delta;
        PhS<R> S = state(tab, P, H);
        R V = p(PI_V_sc + j);
        Mc += S.rho * V;
        Uc += S.rho * V * H;
        MPc += S.drho_dP * V;
        Mdc += S.drho_dh * V;
        UPc += S.drho_dP * V * H;
        Udc += (S.rho + S.drho_dh * H) * V;
    }
    R Mg = 0.0, Ug = 0.0, MPg = 0.0, UPg = 0.0;
    for (int j = 0; j < MX; j++) {
        R Hg = x[XI_h_g0 + j];
        R Tv, rho_g;
        vapor_props(tab, P, Hg, Tv, rho_g);
        R rP_g = 1.2 * rho_g / P;
        R Vg = p(PI_V_gc + j);
        Mg += rho_g * Vg;
        Ug += rho_g * Vg * Hg;
        MPg += rP_g * Vg;
        UPg += rP_g * Vg * Hg;
    }
    M = Mc + Mg;
    U = Uc + Ug - P * PV(V_s);
    MP = MPc + MPg;
    Md = Mdc;
    UP = UPc + UPg - PV(V_s);
    Ud = Udc;
}

template <class R, class T>
__device__ void project_suction(R* x, const Par<R>& p, const Tab<R, T>& tab, int* iters) {
    R Mt = x[XI_M_s], Ut = x[XI_U_s];
    R delta = 0.0;
    int it;
    for (it = 0; it < 4; it++) {
        R M, U, MP, Md, UP, Ud;
        suction_inventory(x, p, tab, delta, M, U, MP, Md, UP, Ud);
        R rM = M - Mt, rU = U - Ut;
        if (m_abs(rM) <= 1e-9 * Mt && m_abs(rU) <= 1e-9 * m_abs(Ut) + 1e-3) break;
        R det = MP * Ud - Md * UP;
        R den = (det != 0.0) ? det : (R)1.0;
        R dP = (rM * Ud - rU * Md) / den;
        R dd = (MP * rU - UP * rM) / den;
        R P = x[XI_P_s];
        if (!(det > 0.0 && m_abs(dP) < 0.05 * P && m_abs(dd) < 2e4)) {
            dP = rM / MP;
            dd = 0.0;
        }
        x[XI_P_s] = clip(P - dP, tab.p_min * 1.001, tab.p_max * 0.999);
        delta = delta - dd;
    }
    if (it == 4) iters[0] += 1;          // not converged within the iterations
    for (int j = 0; j < NS; j++) x[XI_h_q0 + j] += delta;
}

template <class R, class T>
__device__ void flash_rho_u(const Tab<R, T>& tab, R rho_t, R u_t, R P0, R& Pout, R& hout, int* iters) {
    R lo = tab.p_min * 1.001;
    R hi = tab.p_max * 0.999;
    R P = clip(P0, lo, hi);
    int it;
    for (it = 0; it < 8; it++) {
        PhS<R> S = state(tab, P, u_t + P / rho_t);
        R r = S.rho - rho_t;
        if (m_abs(r) <= 1e-5 * rho_t) break;
        if (r > 0.0)
            hi = pmin(hi, P);
        else if (r < 0.0)
            lo = pmax(lo, P);
        R slope = pmax(S.drho_dP + S.drho_dh / rho_t, 1e-12);
        R Pn = P - r / slope;
        if (!m_isfinite(Pn) || Pn <= lo || Pn >= hi) Pn = 0.5 * (lo + hi);
        P = Pn;
    }
    if (it == 8) iters[0] += 1;
    Pout = P;
    hout = u_t + P / rho_t;
}

template <class R, class T>
__device__ __forceinline__ R pool_density(const Tab<R, T>& tab, R P, R h_L) {
    return state(tab, P, pmin(h_L, sat(tab, P).h_l)).rho;
}

template <class R, class T>
__device__ void project_intermediate(R* x, const Par<R>& p, const Tab<R, T>& tab, int* iters) {
    R M = x[XI_M_i];
    R M_L = pmin(pmax(x[XI_M_L], 0.0), 0.999 * M);
    x[XI_M_L] = M_L;
    R M_C = M - M_L;
    R P = x[XI_P_i];
    R h = x[XI_h_i];
    for (int it = 0; it < 2; it++) {
        R rho_L = pool_density(tab, P, x[XI_h_L]);
        R V_C = pmax(PV(V_i) - M_L / rho_L, 1e-3 * PV(V_i));
        R U_C = x[XI_U_i] - M_L * (x[XI_h_L] - P / rho_L);
        flash_rho_u(tab, M_C / V_C, U_C / M_C, P, P, h, iters);
    }
    x[XI_P_i] = P;
    x[XI_h_i] = h;
}

// counters of projections that ran out of iterations (summed per stand)
struct Counters { int suction, flash; };

template <class R, class T>
__device__ void post(R* x, const Par<R>& p, const Tab<R, T>& tab, Counters& cnt) {
    x[XI_N] = pmax(x[XI_N], 0.0);
    for (int k = 0; k < 4; k++) x[XI_u1 + k] = clip(x[XI_u1 + k], (R)0.0, (R)1.0);
    R M = x[XI_M_d];
    R P, h;
    flash_rho_u(tab, M / PV(V_d), x[XI_U_d] / M, x[XI_P_d], P, h, &cnt.flash);
    x[XI_P_d] = P;
    x[XI_h_d] = h;
    project_intermediate(x, p, tab, &cnt.flash);
    project_suction(x, p, tab, &cnt.suction);
}

// ------------------------------------------------------------ integration
template <class R, class T>
__device__ bool stiff(const R* x, const Par<R>& p, const Tab<R, T>& tab) {
    for (int v = 0; v < 2; v++) {
        int iP = v == 0 ? XI_P_d : XI_P_i;
        int ih = v == 0 ? XI_h_d : XI_h_i;
        Sat<R> s = sat(tab, x[iP]);
        R xq = (x[ih] - s.h_l) / (s.h_v - s.h_l);
        R fill = (1.0 - xq) / (1.0 + xq * (s.rho_l / s.rho_v - 1.0));
        if (xq < 0.0 || fill > R(STIFF_FILL_C)) return true;
    }
    Sat<R> s = sat(tab, x[XI_P_s]);
    R liq = 0.0;
    for (int j = 0; j < NS; j++) {
        R xq = (x[XI_h_q0 + j] - s.h_l) / (s.h_v - s.h_l);
        R fill = (1.0 - xq) / (1.0 + xq * (s.rho_l / s.rho_v - 1.0));
        liq += clip(fill, (R)0.0, (R)1.0) * p(PI_V_sc + j);
    }
    return liq / PV(V_s) > R(STIFF_FILL_C);
}

template <class R, class T>
__device__ int subdivisions(const R* x, R lam, const Par<R>& p, const Tab<R, T>& tab, R dt) {
    if (!m_isfinite(lam)) lam = (R)INFINITY;
    R kk = pmin(lam * dt / R(LAM_DT_MAX_C), (R)MAX_SUBDIV_C);
    int k = (int)ceil(kk);
    if (stiff(x, p, tab)) k = imax(k, STIFF_SUBDIV_C);
    k = imin(imax(k, 1), MAX_SUBDIV_C);
    int e = (int)ceil(log2((double)k));
    return imin(1 << e, MAX_SUBDIV_C);
}

template <class R>
__device__ bool plausible(const R* x0, const R* x1) {
    for (int k = 0; k < NX; k++)
        if (!m_isfinite(x1[k])) return false;
    for (int j = 0; j < NS; j++)
        if (m_abs(x1[XI_h_q0 + j] - x0[XI_h_q0 + j]) >= R(MAX_DH_STEP_C)) return false;
    if (m_abs(x1[XI_P_s] / x0[XI_P_s] - 1.0) >= R(MAX_DP_STEP_C)) return false;
    if (m_abs(x1[XI_P_d] / x0[XI_P_d] - 1.0) >= R(MAX_DP_STEP_C)) return false;
    if (m_abs(x1[XI_P_i] / x0[XI_P_i] - 1.0) >= R(MAX_DP_STEP_C)) return false;
    return true;
}

// Per-thread work arrays of a sub-step
template <class R, class RF> struct Work {
    RF k[NX];       // stage derivative
    R ks[NX];       // RK weighted sum of the stage derivatives
    RF xs[NX];      // stage state
    RF hg[MX];
};

// One sub-step of length h from x to xn; returns the rate estimate of its last stage.
template <class R, class RF, class T>
__device__ RF step_once(const R* x, R* xn, const RF* u, RF N_cmd, RF T_amb, RF T_wi, const Par<RF>& pf,
                        const Tab<RF, T>& tf, const Par<R>& p, const Tab<R, T>& tab, R h, int integrator,
                        Work<R, RF>& w, Counters& cnt, long long* nrhs) {
    for (int i = 0; i < NX; i++) w.xs[i] = (RF)x[i];
    RF lam = rhs(w.xs, u, N_cmd, T_amb, T_wi, pf, tf, w.k, w.hg);
    if (integrator == 0) {                     // RK4
        for (int i = 0; i < NX; i++) {
            w.ks[i] = (R)w.k[i];
            w.xs[i] = (RF)(x[i] + 0.5 * h * (R)w.k[i]);
        }
        rhs(w.xs, u, N_cmd, T_amb, T_wi, pf, tf, w.k, w.hg);
        for (int i = 0; i < NX; i++) {
            w.ks[i] = w.ks[i] + 2.0 * (R)w.k[i];
            w.xs[i] = (RF)(x[i] + 0.5 * h * (R)w.k[i]);
        }
        rhs(w.xs, u, N_cmd, T_amb, T_wi, pf, tf, w.k, w.hg);
        for (int i = 0; i < NX; i++) {
            w.ks[i] = w.ks[i] + 2.0 * (R)w.k[i];
            w.xs[i] = (RF)(x[i] + h * (R)w.k[i]);
        }
        lam = rhs(w.xs, u, N_cmd, T_amb, T_wi, pf, tf, w.k, w.hg);
        for (int i = 0; i < NX; i++) xn[i] = x[i] + (h / 6.0) * (w.ks[i] + (R)w.k[i]);
        nrhs[0] += 4;
    } else if (integrator == 1) {              // Heun
        for (int i = 0; i < NX; i++) {
            w.ks[i] = (R)w.k[i];
            w.xs[i] = (RF)(x[i] + h * (R)w.k[i]);
        }
        lam = rhs(w.xs, u, N_cmd, T_amb, T_wi, pf, tf, w.k, w.hg);
        for (int i = 0; i < NX; i++) xn[i] = x[i] + 0.5 * h * (w.ks[i] + (R)w.k[i]);
        nrhs[0] += 2;
    } else {                                   // explicit Euler
        for (int i = 0; i < NX; i++) xn[i] = x[i] + h * (R)w.k[i];
        nrhs[0] += 1;
    }
    for (int j = 0; j < MX; j++) xn[XI_h_g0 + j] = (R)w.hg[j];
    post(xn, p, tab, cnt);
    return lam;
}

template <class R> __device__ __forceinline__ void copy_state(R* dst, const R* src) {
    for (int i = 0; i < NX; i++) dst[i] = src[i];
}

template <class R, class RF, class T> struct Scratch {
    R xn[NX], y[NX], yn[NX];
    Work<R, RF> w;
};

// model.advance: one step of dt in k sub-steps, implausible sub-steps redone four times
// finer (twice at most).  x is updated in place; returns the last stage's rate estimate.
template <class R, class RF, class T>
__device__ RF advance(R* x, const RF* u, RF N_cmd, RF T_amb, RF T_wi, const Par<RF>& pf, const Tab<RF, T>& tf,
                      const Par<R>& p, const Tab<R, T>& tab, R dt, int k, int integrator, Scratch<R, RF, T>& s,
                      Counters& cnt, int* retries, long long* nrhs) {
    R h = dt / (R)k;
    RF lam = (RF)NAN;
    for (int sub = 0; sub < k; sub++) {
        lam = step_once(x, s.xn, u, N_cmd, T_amb, T_wi, pf, tf, p, tab, h, integrator, s.w, cnt, nrhs);
        if (!plausible(x, s.xn)) {
            retries[0] += 1;
            copy_state(s.y, x);                        // level 1: four steps of h / 4 from x
            R h1 = 0.25 * h;
            for (int a = 0; a < 4; a++) {
                lam = step_once(s.y, s.yn, u, N_cmd, T_amb, T_wi, pf, tf, p, tab, h1, integrator, s.w, cnt, nrhs);
                if (!plausible(s.y, s.yn)) {
                    retries[1] += 1;                   // level 2: four steps of h / 16 from y, unchecked
                    R h2 = 0.25 * h1;
                    for (int b = 0; b < 4; b++) {
                        lam = step_once(s.y, s.yn, u, N_cmd, T_amb, T_wi, pf, tf, p, tab, h2, integrator, s.w, cnt,
                                        nrhs);
                        copy_state(s.y, s.yn);
                    }
                } else {
                    copy_state(s.y, s.yn);
                }
            }
            copy_state(x, s.y);
        } else {
            copy_state(x, s.xn);
        }
    }
    return lam;
}

// ------------------------------------------------------------------ kernel
struct TabRaw {
    const void* sat; const void* tr; const void* vap; const void* liq;
    double lp0, inv_dlp, p_min, p_max;
    int n_p, n_z;
};

template <class R, class T> __device__ __forceinline__ Tab<R, T> make_tab(const TabRaw& t) {
    Tab<R, T> o;
    o.sat = (const T*)t.sat; o.tr = (const T*)t.tr; o.vap = (const T*)t.vap; o.liq = (const T*)t.liq;
    o.lp0 = (R)t.lp0; o.inv_dlp = (R)t.inv_dlp; o.p_min = (R)t.p_min; o.p_max = (R)t.p_max;
    o.n_p = t.n_p; o.n_z = t.n_z;
    return o;
}

// X, lam: (NX, n) / (n) structure of arrays in the state type; params: (NPARAM, n) in the
// state type (P) and the right-hand side's type (PF); U: (4, n), N_cmd, T_amb, T_wi: (n) double.
// order: optional permutation (thread -> stand).  stats: (6, n) int: sub-steps, level-1
// and level-2 retries, unconverged suction projections and flashes; nrhs: right-hand
// side evaluations (sum).
template <class R, class RF, class T>
__global__ void LAUNCH_BOUNDS step_kernel(R* __restrict__ X, R* __restrict__ LAM, const R* __restrict__ P,
                                          const RF* __restrict__ PF, TabRaw traw, const double* __restrict__ U,
                                          const double* __restrict__ Ncmd, const double* __restrict__ Tamb,
                                          const double* __restrict__ Twi, const int* __restrict__ order, int n,
                                          double dt, int n_sub, int integrator, int* __restrict__ stats,
                                          unsigned long long* __restrict__ nrhs_total) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) return;
    int e = order ? order[t] : t;
    Par<R> p; p.b = P; p.n = n; p.e = e;
    Par<RF> pf; pf.b = PF; pf.n = n; pf.e = e;
    Tab<R, T> tab = make_tab<R, T>(traw);
    Tab<RF, T> tf = make_tab<RF, T>(traw);
    R x[NX];
    for (int i = 0; i < NX; i++) x[i] = X[(size_t)i * n + e];
    RF u[4];
    for (int k = 0; k < 4; k++) u[k] = (RF)U[(size_t)k * n + e];
    RF N_cmd = (RF)Ncmd[e], T_amb = (RF)Tamb[e], T_wi = (RF)Twi[e];
    R l = LAM[e];
    Scratch<R, RF, T> s;
    Counters cnt; cnt.suction = 0; cnt.flash = 0;
    int retries[2] = {0, 0};
    long long nrhs = 0;
    int subs = 0;
    for (int i = 0; i < n_sub; i++) {
        int k = subdivisions(x, l, p, tab, (R)dt);
        l = (R)advance(x, u, N_cmd, T_amb, T_wi, pf, tf, p, tab, (R)dt, k, integrator, s, cnt, retries, &nrhs);
        subs += k;
    }
    for (int i = 0; i < NX; i++) X[(size_t)i * n + e] = x[i];
    LAM[e] = l;
    stats[0 * n + e] = subs;
    stats[1 * n + e] = retries[0];
    stats[2 * n + e] = retries[1];
    stats[3 * n + e] = cnt.suction;
    stats[4 * n + e] = cnt.flash;
    atomicAdd(nrhs_total, (unsigned long long)nrhs);
}
//@@DEVICE_END

// -------------------------------------------------------------------- host
namespace {

struct Ctx {
    int n, mode, device;
    size_t ssz;                // bytes of the state type
    void *X, *LAM, *P, *PF;
    void *sat, *tr, *vap, *liq;
    double *U, *Ncmd, *Tamb, *Twi;
    int* order;
    int* stats;
    unsigned long long* nrhs;
    TabRaw traw;
    cudaEvent_t e0, e1;
    char err[256];
};

thread_local char g_err[512];

#define CK(call)                                                                       \
    do {                                                                               \
        cudaError_t _e = (call);                                                       \
        if (_e != cudaSuccess) {                                                       \
            snprintf(g_err, sizeof g_err, "%s:%d %s: %s", __FILE__, __LINE__, #call,   \
                     cudaGetErrorString(_e));                                          \
            return -1;                                                                 \
        }                                                                              \
    } while (0)

template <class T> int upload(void** dst, const double* src, size_t count) {
    std::vector<T> tmp(count);
    for (size_t i = 0; i < count; i++) tmp[i] = (T)src[i];
    CK(cudaMalloc(dst, count * sizeof(T)));
    CK(cudaMemcpy(*dst, tmp.data(), count * sizeof(T), cudaMemcpyHostToDevice));
    return 0;
}

// host (n, w) row-major double -> device (w, n) structure of arrays of type T
template <class T> int upload_soa(void* dst, const double* src, int n, int w) {
    std::vector<T> tmp((size_t)n * w);
    for (int e = 0; e < n; e++)
        for (int k = 0; k < w; k++) tmp[(size_t)k * n + e] = (T)src[(size_t)e * w + k];
    CK(cudaMemcpy(dst, tmp.data(), tmp.size() * sizeof(T), cudaMemcpyHostToDevice));
    return 0;
}

template <class T> int download_soa(double* dst, const void* src, int n, int w) {
    std::vector<T> tmp((size_t)n * w);
    CK(cudaMemcpy(tmp.data(), src, tmp.size() * sizeof(T), cudaMemcpyDeviceToHost));
    for (int e = 0; e < n; e++)
        for (int k = 0; k < w; k++) dst[(size_t)e * w + k] = (double)tmp[(size_t)k * n + e];
    return 0;
}

}  // namespace

extern "C" {

const char* hg_error() { return g_err; }

// mode 0 fp64, 1 fp32, 2 mixed (fp64 state and projections, fp32 tables and RHS)
int hg_create(void** out, int device, int n, int mode, const double* params, int nparam, const double* sat,
              int n_sat, const double* tr, const double* vap, const double* liq, int n_reg, int n_p, int n_z,
              double lp0, double inv_dlp, double p_min, double p_max) {
    CK(cudaSetDevice(device));
    Ctx* c = new Ctx();
    memset(c, 0, sizeof(Ctx));
    c->n = n; c->mode = mode; c->device = device;
    bool sd = mode == 0 || mode == 2;     // state in double
    bool td = mode == 0;                  // tables in double
    bool fd = mode == 0;                  // RHS in double
    c->ssz = sd ? 8 : 4;
    // parameters: (n, nparam) -> (nparam, n)
    std::vector<double> soa((size_t)n * nparam);
    for (int e = 0; e < n; e++)
        for (int k = 0; k < nparam; k++) soa[(size_t)k * n + e] = params[(size_t)e * nparam + k];
    if (sd) { if (upload<double>(&c->P, soa.data(), soa.size())) return -1; }
    else    { if (upload<float>(&c->P, soa.data(), soa.size())) return -1; }
    if (fd) { if (upload<double>(&c->PF, soa.data(), soa.size())) return -1; }
    else    { if (upload<float>(&c->PF, soa.data(), soa.size())) return -1; }
    size_t ns = (size_t)n_sat * n_p, nt = (size_t)3 * n_p, nr = (size_t)n_reg * n_p * n_z;
    if (td) {
        if (upload<double>(&c->sat, sat, ns) || upload<double>(&c->tr, tr, nt) || upload<double>(&c->vap, vap, nr)
            || upload<double>(&c->liq, liq, nr)) return -1;
    } else {
        if (upload<float>(&c->sat, sat, ns) || upload<float>(&c->tr, tr, nt) || upload<float>(&c->vap, vap, nr)
            || upload<float>(&c->liq, liq, nr)) return -1;
    }
    c->traw.sat = c->sat; c->traw.tr = c->tr; c->traw.vap = c->vap; c->traw.liq = c->liq;
    c->traw.lp0 = lp0; c->traw.inv_dlp = inv_dlp; c->traw.p_min = p_min; c->traw.p_max = p_max;
    c->traw.n_p = n_p; c->traw.n_z = n_z;
    CK(cudaMalloc(&c->X, (size_t)NX * n * c->ssz));
    CK(cudaMalloc(&c->LAM, (size_t)n * c->ssz));
    CK(cudaMalloc((void**)&c->U, (size_t)4 * n * 8));
    CK(cudaMalloc((void**)&c->Ncmd, (size_t)n * 8));
    CK(cudaMalloc((void**)&c->Tamb, (size_t)n * 8));
    CK(cudaMalloc((void**)&c->Twi, (size_t)n * 8));
    CK(cudaMalloc((void**)&c->stats, (size_t)6 * n * sizeof(int)));
    CK(cudaMalloc((void**)&c->nrhs, sizeof(unsigned long long)));
    CK(cudaMemset(c->nrhs, 0, sizeof(unsigned long long)));
    CK(cudaEventCreate(&c->e0));
    CK(cudaEventCreate(&c->e1));
    *out = c;
    return 0;
}

int hg_set_state(void* h, const double* X, const double* lam) {
    Ctx* c = (Ctx*)h;
    CK(cudaSetDevice(c->device));
    if (c->ssz == 8) {
        if (upload_soa<double>(c->X, X, c->n, NX) || upload_soa<double>(c->LAM, lam, c->n, 1)) return -1;
    } else {
        if (upload_soa<float>(c->X, X, c->n, NX) || upload_soa<float>(c->LAM, lam, c->n, 1)) return -1;
    }
    return 0;
}

int hg_get_state(void* h, double* X, double* lam, int* stats) {
    Ctx* c = (Ctx*)h;
    CK(cudaSetDevice(c->device));
    if (c->ssz == 8) {
        if (download_soa<double>(X, c->X, c->n, NX) || download_soa<double>(lam, c->LAM, c->n, 1)) return -1;
    } else {
        if (download_soa<float>(X, c->X, c->n, NX) || download_soa<float>(lam, c->LAM, c->n, 1)) return -1;
    }
    if (stats) CK(cudaMemcpy(stats, c->stats, (size_t)6 * c->n * sizeof(int), cudaMemcpyDeviceToHost));
    return 0;
}

int hg_get_stats(void* h, int* stats) {
    Ctx* c = (Ctx*)h;
    CK(cudaSetDevice(c->device));
    CK(cudaMemcpy(stats, c->stats, (size_t)6 * c->n * sizeof(int), cudaMemcpyDeviceToHost));
    return 0;
}

// U: (n, 4) row-major; the others (n)
int hg_set_inputs(void* h, const double* U, const double* Ncmd, const double* Tamb, const double* Twi) {
    Ctx* c = (Ctx*)h;
    CK(cudaSetDevice(c->device));
    std::vector<double> u((size_t)4 * c->n);
    for (int e = 0; e < c->n; e++)
        for (int k = 0; k < 4; k++) u[(size_t)k * c->n + e] = U[(size_t)e * 4 + k];
    CK(cudaMemcpy(c->U, u.data(), u.size() * 8, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(c->Ncmd, Ncmd, (size_t)c->n * 8, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(c->Tamb, Tamb, (size_t)c->n * 8, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(c->Twi, Twi, (size_t)c->n * 8, cudaMemcpyHostToDevice));
    return 0;
}

int hg_set_order(void* h, const int* order) {
    Ctx* c = (Ctx*)h;
    CK(cudaSetDevice(c->device));
    if (!order) {
        if (c->order) CK(cudaFree(c->order));
        c->order = nullptr;
        return 0;
    }
    if (!c->order) CK(cudaMalloc((void**)&c->order, (size_t)c->n * sizeof(int)));
    CK(cudaMemcpy(c->order, order, (size_t)c->n * sizeof(int), cudaMemcpyHostToDevice));
    return 0;
}

// n_ctrl launches, each advancing every stand by n_sub steps of dt; *ms receives the
// GPU time (events around the launches)
int hg_step(void* h, double dt, int n_sub, int integrator, int n_ctrl, int block, float* ms) {
    Ctx* c = (Ctx*)h;
    CK(cudaSetDevice(c->device));
    int grid = (c->n + block - 1) / block;
    CK(cudaEventRecord(c->e0));
    for (int s = 0; s < n_ctrl; s++) {
        if (c->mode == 0)
            step_kernel<double, double, double><<<grid, block>>>(
                (double*)c->X, (double*)c->LAM, (const double*)c->P, (const double*)c->PF, c->traw, c->U, c->Ncmd,
                c->Tamb, c->Twi, c->order, c->n, dt, n_sub, integrator, c->stats, c->nrhs);
        else if (c->mode == 1)
            step_kernel<float, float, float><<<grid, block>>>(
                (float*)c->X, (float*)c->LAM, (const float*)c->P, (const float*)c->PF, c->traw, c->U, c->Ncmd,
                c->Tamb, c->Twi, c->order, c->n, dt, n_sub, integrator, c->stats, c->nrhs);
        else
            step_kernel<double, float, float><<<grid, block>>>(
                (double*)c->X, (double*)c->LAM, (const double*)c->P, (const float*)c->PF, c->traw, c->U, c->Ncmd,
                c->Tamb, c->Twi, c->order, c->n, dt, n_sub, integrator, c->stats, c->nrhs);
        CK(cudaGetLastError());
    }
    CK(cudaEventRecord(c->e1));
    CK(cudaEventSynchronize(c->e1));
    CK(cudaEventElapsedTime(ms, c->e0, c->e1));
    return 0;
}

int hg_nrhs(void* h, unsigned long long* out) {
    Ctx* c = (Ctx*)h;
    CK(cudaSetDevice(c->device));
    CK(cudaMemcpy(out, c->nrhs, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    return 0;
}

// registers / local memory per thread of each mode's kernel
int hg_kernel_info(int mode, int* regs, int* local_bytes, int* max_threads) {
    cudaFuncAttributes a;
    if (mode == 0) CK(cudaFuncGetAttributes(&a, step_kernel<double, double, double>));
    else if (mode == 1) CK(cudaFuncGetAttributes(&a, step_kernel<float, float, float>));
    else CK(cudaFuncGetAttributes(&a, step_kernel<double, float, float>));
    *regs = a.numRegs;
    *local_bytes = (int)a.localSizeBytes;
    *max_threads = a.maxThreadsPerBlock;
    return 0;
}

int hg_destroy(void* h) {
    Ctx* c = (Ctx*)h;
    cudaSetDevice(c->device);
    void* ptrs[] = {c->X, c->LAM, c->P, c->PF, c->sat, c->tr, c->vap, c->liq, c->U, c->Ncmd, c->Tamb,
                    c->Twi, c->order, c->stats, c->nrhs};
    for (void* p : ptrs)
        if (p) cudaFree(p);
    cudaEventDestroy(c->e0);
    cudaEventDestroy(c->e1);
    delete c;
    return 0;
}

}  // extern "C"
