from typing import Tuple
import numpy as np

from atmosphere import atmosisa

def perform_constraint_analysis(
    TOGW_lb: float,
    V_stall_kts: float,
    BFL_ft: float,
    LFL_ft: float,
    CLmax_clean: float,
    CLmax_TO: float,
    CLmax_land: float,
    CD0: float,
    K1: float,
    cruise_speed_kts: float,
    cruise_alt_ft: float,
    prop_efficiency: float,
    config,
    blown_lift_augmentation: float = 1.0,
    use_blown_lift_sizing: bool = False,
) -> Tuple[float, float]:
    """
    Standalone constraint analysis with optional blown lift wing sizing.

    Args:
        blown_lift_augmentation: Lift augmentation factor from blown lift (1.0 = no augmentation)
        use_blown_lift_sizing: If True, apply blown lift augmentation to CLmax for wing sizing
                               This enables smaller wing area optimized for cruise

    Returns:
        WS_design (lb/ft²), P_shaft_kW (total for all engines).
    """
    # Apply blown lift augmentation to CLmax values if enabled
    if use_blown_lift_sizing:
        CLmax_clean_eff = CLmax_clean * blown_lift_augmentation
        CLmax_TO_eff = CLmax_TO * blown_lift_augmentation
        CLmax_land_eff = CLmax_land * blown_lift_augmentation
    else:
        CLmax_clean_eff = CLmax_clean
        CLmax_TO_eff = CLmax_TO
        CLmax_land_eff = CLmax_land
    rho_SL = 0.002377  # slug/ft³ at sea level
    V_stall_fps = V_stall_kts * 1.688
    N_engines = config.get('propulsion', 'number_of_engines')

    # 1. Stall speed constraint
    WS_stall_max = 0.5 * rho_SL * V_stall_fps**2 * CLmax_clean_eff

    # 2. Landing constraint
    W_land_ratio = 0.95
    sigma_SL = 1.0
    WS_landing_max = (LFL_ft - 600) * sigma_SL * CLmax_land_eff / 80
    WS_landing_max_TO = WS_landing_max / W_land_ratio

    # 3. Takeoff constraint
    TOP = BFL_ft / 37.5

    # 4. Climb constraints
    ks_climb = 1.2
    CL_climb = CLmax_clean_eff / ks_climb**2
    CD_climb = CD0 + K1 * CL_climb**2

    gamma_climb_OEI = 0.024
    T_W_climb_OEI_req = (ks_climb**2 / CLmax_clean_eff) * CD_climb + gamma_climb_OEI
    OEI_factor = N_engines / (N_engines - 1)
    TW_climb_OEI = OEI_factor * T_W_climb_OEI_req

    gamma_climb_AEO = 0.05
    T_W_climb_AEO_req = (ks_climb**2 / CLmax_clean_eff) * CD_climb + gamma_climb_AEO
    TW_climb_AEO = T_W_climb_AEO_req

    # 5. Service ceiling constraint
    ROC_ceiling_fpm = config.get('performance_requirements', 'rate_of_climb_ceiling_fpm')
    ROC_ceiling_fps = ROC_ceiling_fpm / 60

    h_ceiling_m = config.get('mission_requirements', 'service_ceiling_ft') * 0.3048
    _, _, rho_ceiling_kg_m3, _ = atmosisa(h_ceiling_m)
    rho_ceiling_slug = rho_ceiling_kg_m3 / 515.379

    alpha_ceiling = 0.50
    V_climb_ceiling = np.sqrt(2 * 60 / (rho_ceiling_slug * CLmax_clean_eff))
    T_W_ceiling_req = ROC_ceiling_fps / V_climb_ceiling + 2 * np.sqrt(CD0 * K1)
    TW_ceiling = (1.0 / alpha_ceiling) * T_W_ceiling_req

    # 6. Cruise constraint
    h_cruise_m = cruise_alt_ft * 0.3048
    _, _, rho_cruise_kg_m3, _ = atmosisa(h_cruise_m)
    rho_cruise_slug = rho_cruise_kg_m3 / 515.379
    V_cruise_fps = cruise_speed_kts * 1.688

    alpha_cruise = 0.75
    beta_cruise = 1.0
    q_cruise = 0.5 * rho_cruise_slug * V_cruise_fps**2

    # Design WS from stall and landing
    WS_design = min(WS_stall_max, WS_landing_max_TO)

    TW_takeoff = WS_design / (sigma_SL * CLmax_TO_eff * TOP)

    CL_cruise = TOGW_lb / (q_cruise * (TOGW_lb / WS_design))
    CD_cruise = CD0 + K1 * CL_cruise**2
    L_D_cruise = CL_cruise / CD_cruise
    TW_cruise = (beta_cruise / alpha_cruise) * (1.0 / L_D_cruise)

    TW_required = max(TW_takeoff, TW_climb_OEI, TW_climb_AEO, TW_ceiling, TW_cruise)

    # Identify sizing constraint (for printing only)
    sizing_constraint = "Unknown"
    if TW_required == TW_takeoff:
        sizing_constraint = "Takeoff"
    elif TW_required == TW_climb_OEI:
        sizing_constraint = "OEI Climb"
    elif TW_required == TW_climb_AEO:
        sizing_constraint = "AEO Climb"
    elif TW_required == TW_ceiling:
        sizing_constraint = "Service Ceiling"
    elif TW_required == TW_cruise:
        sizing_constraint = "Cruise"

    # Convert T/W to power
    V_climb_fps = ks_climb * V_stall_fps
    T_total_lb = TW_required * TOGW_lb

    P_takeoff_HP = (TW_takeoff * TOGW_lb) * 1.15 * V_stall_fps / (550 * prop_efficiency)
    P_climb_OEI_HP = (TW_climb_OEI * TOGW_lb) * V_climb_fps / (550 * prop_efficiency)
    P_climb_AEO_HP = (TW_climb_AEO * TOGW_lb) * V_climb_fps / (550 * prop_efficiency)
    P_ceiling_HP = (TW_ceiling * TOGW_lb) * V_climb_ceiling / (550 * prop_efficiency)
    P_cruise_HP = (TW_cruise * TOGW_lb) * V_cruise_fps / (550 * prop_efficiency)

    P_shaft_HP = max(P_takeoff_HP, P_climb_OEI_HP, P_climb_AEO_HP, P_ceiling_HP, P_cruise_HP)
    P_shaft_kW = P_shaft_HP * 0.7457

    # Console output kept here so behavior matches original
    print(f"\n{'='*70}")
    print(f"CONSTRAINT ANALYSIS RESULTS")
    if use_blown_lift_sizing:
        print(f"  (Using Blown Lift Wing Sizing - Augmentation: {blown_lift_augmentation:.3f}x)")
    print(f"{'='*70}")
    if use_blown_lift_sizing:
        print(f"Effective CLmax (with blown lift):")
        print(f"  CLmax_clean:   {CLmax_clean:.2f} → {CLmax_clean_eff:.2f}")
        print(f"  CLmax_takeoff: {CLmax_TO:.2f} → {CLmax_TO_eff:.2f}")
        print(f"  CLmax_landing: {CLmax_land:.2f} → {CLmax_land_eff:.2f}")
        print(f"")
    print(f"Wing Loading Limits:")
    print(f"  Stall speed (V_stall = {V_stall_kts} kts):  {WS_stall_max:6.1f} lb/ft²")
    print(f"  Landing (LFL = {LFL_ft} ft):       {WS_landing_max_TO:6.1f} lb/ft²")
    print(f"  Design WS:                           {WS_design:6.1f} lb/ft² ← SELECTED")
    print(f"\nThrust-to-Weight Requirements:")
    print(f"  Takeoff (BFL = {BFL_ft} ft):         {TW_takeoff:6.3f}")
    print(f"  OEI Climb (2.4% gradient):           {TW_climb_OEI:6.3f}")
    print(f"  AEO Climb (5.0% gradient):           {TW_climb_AEO:6.3f}")
    print(f"  Service Ceiling ({ROC_ceiling_fpm} fpm):       {TW_ceiling:6.3f}")
    print(f"  Cruise (L/D = {L_D_cruise:.1f}):                {TW_cruise:6.3f}")
    print(f"  Required T/W:                        {TW_required:6.3f} ← {sizing_constraint}")
    print(f"\nPower Requirements:")
    print(f"  Total Thrust:    {T_total_lb:7.0f} lb")
    print(f"  Total Power:     {P_shaft_kW:7.0f} kW ({P_shaft_HP:7.0f} HP)")
    print(f"  Power per Engine:{P_shaft_kW/N_engines:7.0f} kW ({P_shaft_HP/N_engines:7.0f} HP)")
    print(f"  Specific Power:  {P_shaft_HP/TOGW_lb:7.3f} HP/lb")
    print(f"{'='*70}\n")

    power_constraints = {
        'Takeoff': P_takeoff_HP,
        'OEI Climb': P_climb_OEI_HP,
        'AEO Climb': P_climb_AEO_HP,
        'Service Ceiling': P_ceiling_HP,
        'Cruise': P_cruise_HP
    }
    sizing_constraint_power = max(power_constraints, key=power_constraints.get)
    print(f"Sizing Constraint for Power: {sizing_constraint_power}\n")

    return WS_design, P_shaft_kW
