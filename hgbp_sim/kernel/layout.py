"""Array layouts shared by the model kernel and its Python wrapper: the state
vector, the packed parameter record and the auxiliary output vector.

Index constants are module globals so that the compiled kernel sees them as
compile-time constants (``X_P_S``, ``A_P_s``, ...).
"""
from __future__ import annotations

import numpy as np

from ..geometry import LINES
from ..params import PlantParams

MX = 5          # finite-volume cells per side of the mixing exchanger and of the condenser

# ------------------------------------------------------------------ state
STATE_NAMES = tuple(["P_s"] + [f"h_q{j}" for j in range(MX)] + ["h_l1", "h_l2"]
                    + [f"T_mw{j}" for j in range(MX)]
                    + ["T_sw", "P_d", "h_d", "T_dw", "P_i", "h_i", "h_L"] + [f"T_cw{j}" for j in range(MX)]
                    + ["T_rw", "T_sh", "N",
                       "u1", "u2", "u3", "u4", "Tm_s", "Tm_d", "mm", "Wm", "Tm_co",
                       "M_s", "M_d", "M_i", "U_s", "U_d", "U_i", "M_L"]
                    + [f"h_g{j}" for j in range(MX)])
IX = {k: i for i, k in enumerate(STATE_NAMES)}
NX = len(STATE_NAMES)
X_P_S = IX["P_s"]
X_HQ = IX["h_q0"]                   # quench cells h_q0.., then h_l1, h_l2 (the dynamic suction cells)
X_H_L1, X_H_L2 = IX["h_l1"], IX["h_l2"]
X_TMW = IX["T_mw0"]
X_T_SW = IX["T_sw"]
X_P_D, X_H_D, X_T_DW = IX["P_d"], IX["h_d"], IX["T_dw"]
X_P_I, X_H_I, X_T_RW = IX["P_i"], IX["h_i"], IX["T_rw"]     # condensing zone of the intermediate section
X_H_L, X_M_L = IX["h_L"], IX["M_L"]                          # receiver liquid pool
X_TCW = IX["T_cw0"]                 # condenser plate walls T_cw0.. (top -> bottom)
X_T_SH, X_N = IX["T_sh"], IX["N"]
X_U1 = IX["u1"]                     # u1..u4
X_TM_S, X_TM_D, X_MM, X_WM, X_TM_CO = IX["Tm_s"], IX["Tm_d"], IX["mm"], IX["Wm"], IX["Tm_co"]
X_M_S, X_M_D, X_M_I, X_U_S, X_U_D, X_U_I = IX["M_s"], IX["M_d"], IX["M_i"], IX["U_s"], IX["U_d"], IX["U_i"]
X_HG = IX["h_g0"]

# -------------------------------------------------------------- parameters
# Valve flow characteristics as the kernel takes them
CHAR_CODES = {"linear": 0, "eqpct": 1, "quick": 2}
CHAR_FIELDS = {"dpv_char": "dpv_code", "spv_char": "spv_code", "stv_char": "stv_code", "w_char": "w_code"}
# volumes and heat capacities added by geometry.derive (scalars per environment)
DERIVED = ("V_cond", "V_cond_w", "V_mx_g", "V_mx_q", "V_d", "V_i", "V_L1", "V_L2", "V_s",
           "A_mx_cell", "C_mw_cell", "C_cw", "C_sw", "C_dw", "C_rw")
DERIVED_VEC = (("V_sc", MX + 2), ("V_gc", MX), ("seg_i", 6))


def _param_dtype():
    f = [(k, "f8") for k in PlantParams.numeric_fields()] + [("charge", "f8")]
    f += [(k, "f8") for k in DERIVED] + [(k, "f8", (w,)) for k, w in DERIVED_VEC]
    f += [(f"V_line_{k}", "f8") for k in LINES] + [(c, "f8") for c in CHAR_FIELDS.values()]
    return np.dtype(f)


PARAM_DTYPE = _param_dtype()


def pack(p) -> np.ndarray:
    """Structured array (one record per environment, ``PARAM_DTYPE``) with the
    values of parameter namespace ``p`` (after geometry.derive)."""
    n = len(np.atleast_1d(p.V_d))
    rec = np.zeros(n, PARAM_DTYPE)
    for name in PARAM_DTYPE.names:
        if name.startswith("V_line_"):
            rec[name] = p.V_lines[name[7:]]
        elif name in CHAR_FIELDS.values():
            continue
        else:
            rec[name] = getattr(p, name)
    for attr, code in CHAR_FIELDS.items():
        rec[code] = CHAR_CODES[getattr(p, attr)]
    return rec


# ------------------------------------------------------------------ outputs
# auxiliary outputs of the right-hand side: (name, width)
AUX_FIELDS = (
    ("P_s", 1), ("P_d", 1), ("P_i", 1), ("T_s", 1), ("T_d", 1), ("T_i", 1), ("T_co", 1),
    ("T_sat_s", 1), ("T_sat_d", 1), ("T_sat_i", 1), ("SH", 1), ("SC", 1), ("x_out", 1), ("y_liq", 1),
    ("x_l1", 1), ("x_i", 1), ("x_qo", 1), ("T_qo", 1), ("T_go", 1), ("h_go", 1), ("T_l1", 1),
    ("fill_i", 1), ("rec_level", 1), ("ll_fill", 1), ("cond_flood", 1), ("M_q_liq", 1),
    ("T_L", 1), ("h_L", 1), ("SC_L", 1), ("M_L", 1), ("M_cl", 1), ("M_film", 1), ("mdot_drn", 1),
    ("mdot_lv", 1),
    ("h_q", MX), ("x_q", MX), ("T_q", MX), ("T_g", MX), ("h_g", MX), ("T_mw", MX),
    ("h_c", MX), ("x_c", MX), ("T_c", MX), ("T_wc", MX), ("T_cwc", MX),
    ("Q_mx", 1), ("Q_q", 1), ("rho_s", 1), ("h_l1", 1), ("h_l2", 1), ("h_cin", 1), ("x_l2", 1),
    ("h_d", 1), ("h_i", 1), ("h_co", 1), ("h2", 1), ("h_2f", 1), ("h_3f", 1), ("T2_ad", 1),
    ("P_tee", 1), ("dP_suc", 1), ("dP_dis", 1), ("dP_hdr", 1), ("dP_bp", 1), ("dP_mog", 1),
    ("dP_moq", 1), ("dP_q", 1), ("dP_liq", 1), ("dP_drn", 1), ("P_h", 1), ("dP_cr", 1), ("dP_cw", 1),
    ("dP_mg", 1), ("dP_mq", 1), ("dP_mq_ch", 1), ("dP_qc", MX), ("dP_q_dist", 1), ("dP_q_in", 1),
    ("dP_q_out", 1), ("mdot_cr", 1),
    ("mdot_c", 1), ("mdot_1", 1), ("mdot_2", 1), ("mdot_3", 1), ("mdot_w", 1),
    ("W_el", 1), ("W_shaft", 1), ("Pr", 1), ("eta_v", 1), ("eta_s", 1),
    ("Q_r", 1), ("Q_w", 1), ("Q_sc", 1), ("Q_sg", 1), ("Q_dg", 1), ("Q_rw", 1), ("T_wo", 1),
    ("T_sw", 1), ("T_dw", 1), ("T_cw", 1), ("T_rw", 1), ("T_sh", 1), ("N", 1),
    ("u1", 1), ("u2", 1), ("u3", 1), ("u4", 1),
    ("M_s", 1), ("M_g", 1), ("M_d", 1), ("M_i", 1), ("M_tot", 1),
    ("Tm_s", 1), ("Tm_d", 1), ("Tm_co", 1), ("mm", 1), ("Wm", 1),
    ("dx", NX),
)
AUX_SLICES = {}
_o = 0
for _name, _w in AUX_FIELDS:
    AUX_SLICES[_name] = (_o, _w)
    globals()[f"A_{_name}"] = _o          # A_P_s, A_h_q, ... (offset of the field)
    _o += _w
NA = _o
del _o, _name, _w
