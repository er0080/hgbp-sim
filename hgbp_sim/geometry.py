"""Refrigerant volumes and wall heat capacities derived from the stand's
component and piping specification.

The plant parameters describe the hardware the way it is specified (tube
outside diameter, wall and length; plate count, channel volume, heating
surface and weight of the brazed-plate heat exchangers; receiver volume).
:func:`derive` turns them into the lumped volumes and heat capacities the
model integrates, and stores them on the parameter namespace next to the
input parameters (so batched / randomized parameters work unchanged).

Port and channel convention of the Alfa Laval ACH-70X brazed plate heat
exchangers (drawing 3287083725): ``n_plates`` plates form ``n_plates - 1``
channels; the S1-S2 side has the larger half (39 of 77 for 78 plates), the
S3-S4 side the smaller half (38).

* condenser: refrigerant on S3-S4, cooling water on S1-S2
* mixing exchanger: quench (liquid from valve 3) on S3-S4, entering at S3 on
  top and leaving at S4 at the bottom; hot bypass gas on S1-S2, entering at S1
  at the bottom and leaving at S2 on top (counterflow)
* suction exchanger (``mixer`` = 1, in place of the mixing exchanger): the stream from
  the tee rises through S1-S2 (pass 1), a jumper takes it to S3, and it falls through
  S3-S4 (pass 2), counterflow to pass 1.  Pass 1 takes the mixing exchanger's quench cells
  (the bypass and quench lines and the tee join its inlet cell), pass 2 its gas cells (the
  jumper joins its inlet cell), and the outlet piping is split between the tee leg and
  the S4 leg.
"""
from __future__ import annotations

import numpy as np

RHO_CU, CP_CU = 8940.0, 385.0        # copper tube [kg/m^3], [J/kg/K]
CP_SS = 500.0                        # stainless steel plates [J/kg/K]
CP_STEEL = 480.0                     # carbon steel receiver shell [J/kg/K]

# (outside diameter, wall, length) parameter names of every refrigerant line
LINES = {
    "dis": "discharge line, compressor -> valve 1",
    "hdr": "hot gas header, valve 1 -> condenser inlet and valve 2",
    "bp": "bypass line, valve 2 -> mixing exchanger S1",
    "q": "quench line, valve 3 -> mixing exchanger S3",
    "mo": "mixing exchanger outlets S2 / S4 -> tee",
    "suc": "suction line, tee -> compressor suction port",
    "drn": "condensate drain, condenser -> receiver",
    "liq": "liquid line, receiver dip tube -> valve 3",
}


def tube_volume(D, t, L):
    """Inner volume of a tube with outside diameter D, wall t, length L [m^3]."""
    d = np.maximum(D - 2.0 * t, 0.0)
    return 0.25 * np.pi * d * d * L


def tube_mass(D, t, L, rho=RHO_CU):
    d = np.maximum(D - 2.0 * t, 0.0)
    return 0.25 * np.pi * (D * D - d * d) * L * rho


def phe_channels(n_plates):
    """Channels on the S1-S2 side and on the S3-S4 side."""
    n_ch = np.maximum(np.asarray(n_plates, float) - 1.0, 2.0)
    return np.ceil(n_ch / 2.0), np.floor(n_ch / 2.0)


def _line(p, key):
    return getattr(p, "D_" + key), getattr(p, "t_" + key), getattr(p, "L_" + key)


def derive(p, n_cells: int) -> None:
    """Add the derived volumes / heat capacities to parameter namespace ``p``
    (arrays of shape (n,); ``V_sc`` (n, n_cells + 2) and ``V_gc`` (n, n_cells))."""
    vol = {k: tube_volume(*_line(p, k)) for k in LINES}
    p.V_lines = vol

    # --- brazed plate heat exchangers
    cA, cB = phe_channels(p.cond_n_plates)
    p.V_cond = cB * p.cond_V_ch                  # refrigerant side (S3-S4)
    p.V_cond_w = cA * p.cond_V_ch                # water side (S1-S2)
    mA, mB = phe_channels(p.mx_n_plates)
    p.V_mx_g = mA * p.mx_V_ch                    # bypass gas side (S1-S2)
    p.V_mx_q = mB * p.mx_V_ch                    # quench side (S3-S4)

    # --- lumped control volumes
    p.V_d = p.V_comp_dis + vol["dis"]
    p.V_i = vol["hdr"] + p.V_cond + vol["drn"] + p.rec_V + vol["liq"]
    p.V_L1 = vol["mo"] + vol["suc"]              # tee -> suction port (the sensor)
    p.V_L2 = np.asarray(p.V_comp_suc, float) * np.ones_like(p.V_d)   # inside the compressor
    n = n_cells
    one = np.ones((len(np.atleast_1d(p.V_d)), 1))
    Vq = (p.V_mx_q / n)[:, None] * np.ones((1, n))
    Vq[:, 0] += vol["q"]                         # quench line joins the first (top) cell
    Vg = (p.V_mx_g / n)[:, None] * np.ones((1, n))
    Vg[:, n - 1] += vol["bp"]                    # bypass line joins the bottom (inlet) cell

    # --- suction exchanger (mixer = 1): replaces the mixing exchanger's cells
    bA, bB = phe_channels(p.bx_n_plates)
    p.bx_nch1, p.bx_nch2 = bA, bB                # pass 1 (S1-S2), pass 2 (S3-S4) channels
    V_ch = p.bx_b * p.bx_W * p.bx_L
    A_ch = 2.0 * p.bx_phi * p.bx_L * p.bx_W      # wall area of one channel (both faces)
    p.A_b1_cell, p.A_b2_cell = bA * A_ch / n, bB * A_ch / n
    p.bx_A = np.maximum(p.bx_n_plates - 2.0, 1.0) * p.bx_phi * p.bx_L * p.bx_W   # heating surface
    V_jmp = 0.25 * np.pi * p.bx_d_conn ** 2 * p.bx_L_jmp
    Vb1 = (bA * V_ch / n)[:, None] * np.ones((1, n))
    Vb1[:, 0] += vol["q"] + vol["bp"] + 0.5 * vol["mo"]     # tee, its lines and the inlet leg
    Vb2 = (bB * V_ch / n)[:, None] * np.ones((1, n))
    Vb2[:, n - 1] += V_jmp                       # the jumper joins pass 2's inlet cell
    bx = (np.asarray(p.mixer, float) > 0.5) * np.ones_like(p.V_d)
    w = bx[:, None]
    Vq = w * Vb1 + (1.0 - w) * Vq
    Vg = w * Vb2 + (1.0 - w) * Vg
    p.V_L1 = p.V_L1 - bx * 0.5 * vol["mo"]      # the outlet leg S4 -> suction line
    p.V_sc = np.concatenate([Vq, p.V_L1[:, None] * one, p.V_L2[:, None] * one], axis=1)
    p.V_gc = Vg
    p.V_s = p.V_sc.sum(axis=1) + p.V_gc.sum(axis=1)

    # --- intermediate section, in the order liquid collects: receiver below the dip
    # tube inlet, liquid line, rest of the receiver, drain line, condenser, header
    p.seg_i = np.stack([p.rec_dip * p.rec_V, vol["liq"], (1.0 - p.rec_dip) * p.rec_V,
                        vol["drn"], p.V_cond, vol["hdr"]], axis=1)

    # --- heat capacities
    p.A_mx_cell = p.mx_A / n
    p.C_mw_cell = np.where(bx > 0.5, p.bx_mass, p.mx_mass) * CP_SS / n
    p.C_cw = p.cond_mass * CP_SS + p.V_cond_w * p.rho_w * p.cp_w
    p.C_sw = CP_CU * (tube_mass(*_line(p, "mo")) + tube_mass(*_line(p, "suc"))) + CP_STEEL * p.suc_metal_mass
    p.C_dw = CP_CU * tube_mass(*_line(p, "dis"))
    p.C_rw = p.rec_mass * CP_STEEL


def volume_table(p) -> list[dict]:
    """Breakdown of the refrigerant-side volumes of the first environment [L],
    for documentation and the UI."""
    f = lambda v: float(np.atleast_1d(v)[0]) * 1e3
    rows = [dict(section="discharge", item="compressor discharge (internal)", L=f(p.V_comp_dis))]
    rows.append(dict(section="discharge", item=LINES["dis"], L=f(p.V_lines["dis"])))
    for k in ("hdr",):
        rows.append(dict(section="intermediate", item=LINES[k], L=f(p.V_lines[k])))
    rows += [dict(section="intermediate", item="condenser, refrigerant side (S3-S4)", L=f(p.V_cond)),
             dict(section="intermediate", item=LINES["drn"], L=f(p.V_lines["drn"])),
             dict(section="intermediate", item="receiver", L=f(p.rec_V)),
             dict(section="intermediate", item=LINES["liq"], L=f(p.V_lines["liq"]))]
    rows += [dict(section="suction", item=LINES["bp"], L=f(p.V_lines["bp"])),
             dict(section="suction", item="mixing exchanger, gas side (S1-S2)", L=f(p.V_mx_g)),
             dict(section="suction", item=LINES["q"], L=f(p.V_lines["q"])),
             dict(section="suction", item="mixing exchanger, quench side (S3-S4)", L=f(p.V_mx_q)),
             dict(section="suction", item=LINES["mo"], L=f(p.V_lines["mo"])),
             dict(section="suction", item=LINES["suc"], L=f(p.V_lines["suc"])),
             dict(section="suction", item="compressor suction (internal, after the sensor)", L=f(p.V_comp_suc))]
    return rows
