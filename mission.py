from dataclasses import dataclass
from atmosphere import atmosisa
from typing import List, Dict, Tuple
import numpy as np
from config_loader import load_config

# Load configuration
config = load_config()

# need to add rest of mission functions from aircraft.py to here

@dataclass
class MissionSegment:
    """Mission segment definition"""
    name: str
    altitude_start_ft: float
    altitude_end_ft: float
    Hp: float = 0.0  # Hybridization ratio for this segment
    blown_lift_active: bool = False  # Whether blown lift from DEP high-lift motors is active

    # Results
    time_sec: float = 0.0
    fuel_lb: float = 0.0
    battery_Wh: float = 0.0
    distance_nm: float = 0.0


def get_highlift_motor_power(self, blown_lift_active: bool) -> float:
    """
    Calculate high-lift motor power consumption for DEP system.

    Args:
        blown_lift_active: Whether the DEP high-lift motors are active

    Returns:
        Power draw in kW (0 if motors not active or DEP disabled)
    """
    if not blown_lift_active or not self.dep_enabled:
        return 0.0

    # Total high-lift motor power = num_motors × power_per_motor
    # From config: 12 motors × 10.5 kW = 126 kW
    P_highlift_kW = self.dep_num_motors * config.get('dep_system', 'motor_power_kW')

    return P_highlift_kW


def create_mission(self, hybridization_profile: Dict[str, float],
                   blown_lift_profile: Dict[str, bool] = None) -> List[MissionSegment]:
    """
    Create mission profile with segment-specific hybridization and blown lift control

    Args:
        hybridization_profile: Dict mapping segment names to hybridization ratios (0.0-1.0)
        blown_lift_profile: Dict mapping segment names to blown lift active status (True/False)
                           If None, defaults to True for takeoff/landing, False for cruise
    """
    if blown_lift_profile is None:
        # Default: blown lift active for low-speed segments, off for cruise
        blown_lift_profile = {
            'takeoff': True,
            'climb': True,
            'cruise': False,
            'descent': False,
            'loiter': False,
            'landing': True,
        }

    segments = [
        MissionSegment('takeoff', 0, 35,
                      hybridization_profile.get('takeoff', 0.0),
                      blown_lift_profile.get('takeoff', False)),
        MissionSegment('climb', 35, self.cruise_alt_ft,
                      hybridization_profile.get('climb', 0.0),
                      blown_lift_profile.get('climb', False)),
        MissionSegment('cruise', self.cruise_alt_ft, self.cruise_alt_ft,
                      hybridization_profile.get('cruise', 0.0),
                      blown_lift_profile.get('cruise', False)),
        MissionSegment('descent', self.cruise_alt_ft, 450,
                      hybridization_profile.get('descent', 0.0),
                      blown_lift_profile.get('descent', False)),
        MissionSegment('loiter', 450, 450,
                      hybridization_profile.get('loiter', 0.0),
                      blown_lift_profile.get('loiter', False)),
        MissionSegment('landing', 450, 0,
                      hybridization_profile.get('landing', 0.0),
                      blown_lift_profile.get('landing', False)),
    ]

    return segments

# ======================= Simulation helpers ======================= #
def simulate_cruise_segment(self, segment: MissionSegment, W_lb: float, S_ft2: float) -> Tuple:
    """Cruise segment simulation"""
    # Atmospheric conditions
    h_m = segment.altitude_start_ft * 0.3048
    _, _, rho_kg_m3, _ = atmosisa(h_m)
    rho_slug_ft3 = rho_kg_m3 / 515.379

    # Cruise speed
    V_fps = self.cruise_speed_kts * 1.688

    # Apply blown lift augmentation if active
    lift_aug_factor = self.get_lift_augmentation_factor(segment.blown_lift_active)

    # L/D calculation
    CL_base = W_lb / (0.5 * rho_slug_ft3 * V_fps**2 * S_ft2)
    CL = CL_base * lift_aug_factor
    # Note: CD calculation uses base CL since induced drag is based on actual circulation
    CD = self.CD0 + self.K1 * CL_base**2
    D_lb = 0.5 * rho_slug_ft3 * V_fps**2 * S_ft2 * CD

    # Power required
    P_shaft_HP = D_lb * V_fps / (550 * self.tech.prop_efficiency)
    P_shaft_kW = P_shaft_HP * 0.7457

    # Power split
    power_split = self.powertrain.get_power_split(P_shaft_kW, segment.Hp)

    # Cruise time
    distance_ft = self.range_nm * 6076.12
    time_sec = distance_ft / V_fps

    # Add high-lift motor power if active (draws from battery)
    P_highlift_kW = get_highlift_motor_power(self, segment.blown_lift_active)
    P_highlift_W = P_highlift_kW * 1000.0

    # Consumption (add high-lift motor energy to battery draw)
    fuel_lb = power_split['fuel_rate_kg_s'] * time_sec * 2.20462
    battery_Wh = (power_split['battery_power_W'] * time_sec) / 3600 + (P_highlift_W * time_sec) / 3600

    segment.distance_nm = self.range_nm

    return time_sec, fuel_lb, battery_Wh

def simulate_climb_segment(self, segment: MissionSegment, W_lb: float, S_ft2: float) -> Tuple:
    """Climb segment simulation"""
    # Average weight during climb (assume 2% fuel burn)
    W_avg = W_lb * 0.99

    # Apply blown lift augmentation if active
    lift_aug_factor = self.get_lift_augmentation_factor(segment.blown_lift_active)
    CLmax_clean_effective = self.CLmax_clean * lift_aug_factor

    # Climb at 1.3 * V_stall for best angle
    rho_sl = 0.002377  # slug/ft³ at sea level
    V_stall_fps = np.sqrt(2 * W_avg / (rho_sl * S_ft2 * CLmax_clean_effective))
    V_climb_fps = 1.3 * V_stall_fps

    # Climb gradient (5%)
    gamma = 0.05
    ROC_fps = V_climb_fps * gamma

    # L/D in climb (apply augmentation to operating CL as well)
    CL_climb_base = W_avg / (0.5 * rho_sl * V_climb_fps**2 * S_ft2)
    CL_climb = CL_climb_base * lift_aug_factor
    # Note: CD calculation uses base CL since induced drag is based on actual circulation
    CD_climb = self.CD0 + self.K1 * CL_climb_base**2

    # Thrust required
    T_lb = W_avg * (1/(CL_climb/CD_climb) + gamma)
    P_shaft_HP = T_lb * V_climb_fps / (550 * self.tech.prop_efficiency)
    P_shaft_kW = P_shaft_HP * 0.7457

    # Power split
    power_split = self.powertrain.get_power_split(P_shaft_kW, segment.Hp)

    # Climb time
    alt_change_ft = segment.altitude_end_ft - segment.altitude_start_ft
    time_sec = alt_change_ft / ROC_fps if ROC_fps > 0 else 600

    # Add high-lift motor power if active (draws from battery)
    P_highlift_kW = get_highlift_motor_power(self, segment.blown_lift_active)
    P_highlift_W = P_highlift_kW * 1000.0

    # Consumption (add high-lift motor energy to battery draw)
    fuel_lb = power_split['fuel_rate_kg_s'] * time_sec * 2.20462
    battery_Wh = (power_split['battery_power_W'] * time_sec) / 3600 + (P_highlift_W * time_sec) / 3600

    return time_sec, fuel_lb, battery_Wh

def simulate_descent_segment(self, segment: MissionSegment, W_lb: float, S_ft2: float) -> Tuple:
    """Descent segment - physics-based calculation with idle power"""
    # Atmospheric conditions at average altitude
    h_avg_ft = (segment.altitude_start_ft + segment.altitude_end_ft) / 2
    h_avg_m = h_avg_ft * 0.3048
    _, _, rho_kg_m3, _ = atmosisa(h_avg_m)
    rho_slug_ft3 = rho_kg_m3 / 515.379

    # Descent speed (slightly below cruise speed for passenger comfort)
    V_descent_kts = self.cruise_speed_kts * 0.9
    V_fps = V_descent_kts * 1.688

    # Descent rate (typical: 500-800 fpm for comfort)
    descent_rate_fpm = 600
    descent_rate_fps = descent_rate_fpm / 60

    # Flight path angle
    gamma = np.arctan(descent_rate_fps / V_fps)

    # Apply blown lift augmentation if active
    lift_aug_factor = self.get_lift_augmentation_factor(segment.blown_lift_active)

    # Lift coefficient (slightly less than weight, due to descent)
    W_effective = W_lb * np.cos(gamma)
    CL_base = W_effective / (0.5 * rho_slug_ft3 * V_fps**2 * S_ft2)
    CL = CL_base * lift_aug_factor
    # Note: CD calculation uses base CL since induced drag is based on actual circulation
    CD = self.CD0 + self.K1 * CL_base**2

    # Drag force
    D_lb = 0.5 * rho_slug_ft3 * V_fps**2 * S_ft2 * CD

    # Thrust required (reduced due to descent - gravity assists)
    T_required_lb = D_lb - W_lb * np.sin(gamma)

    # If negative thrust (descending too fast), use flight idle power
    if T_required_lb < 0:
        # Flight idle: ~7% of rated power
        P_shaft_kW = self.powertrain.P_GT_kW * 0.07
    else:
        # Power required
        P_shaft_HP = T_required_lb * V_fps / (550 * self.tech.prop_efficiency)
        P_shaft_kW = P_shaft_HP * 0.7457
        # Minimum flight idle power
        P_shaft_kW = max(P_shaft_kW, self.powertrain.P_GT_kW * 0.07)

    # Power split
    power_split = self.powertrain.get_power_split(P_shaft_kW, segment.Hp)

    # Descent time
    alt_change_ft = segment.altitude_start_ft - segment.altitude_end_ft
    time_sec = (alt_change_ft / descent_rate_fps) if descent_rate_fps > 0 else 480

    # Add high-lift motor power if active (draws from battery)
    P_highlift_kW = get_highlift_motor_power(self, segment.blown_lift_active)
    P_highlift_W = P_highlift_kW * 1000.0

    # Consumption (add high-lift motor energy to battery draw)
    fuel_lb = power_split['fuel_rate_kg_s'] * time_sec * 2.20462
    battery_Wh = (power_split['battery_power_W'] * time_sec) / 3600 + (P_highlift_W * time_sec) / 3600

    return time_sec, fuel_lb, battery_Wh

def simulate_takeoff_segment(self, segment: MissionSegment, W_lb: float, S_ft2: float) -> Tuple:
    """Takeoff segment - ground roll + rotation + climb to 35 ft"""
    # Sea level conditions
    rho_sl = 0.002377  # slug/ft³

    # Apply blown lift augmentation if active
    lift_aug_factor = self.get_lift_augmentation_factor(segment.blown_lift_active)
    CLmax_TO_effective = self.CLmax_TO * lift_aug_factor

    # Takeoff speed (1.1 * stall speed with flaps)
    V_stall_TO_fps = np.sqrt(2 * W_lb / (rho_sl * S_ft2 * CLmax_TO_effective))
    V_liftoff_fps = 1.1 * V_stall_TO_fps

    # Ground roll time (assume constant acceleration to simplify)
    # Average thrust during ground roll ~ 80% of static thrust
    # Using simplified kinematic equation: t = 2*distance / V_avg
    t_ground_sec = 30  # Typical: 20-40 seconds for regional aircraft

    # Climb to 35 ft at V_y (best rate of climb speed ~ 1.3 * V_stall)
    V_climb_fps = 1.3 * V_stall_TO_fps
    climb_gradient = 0.08  # 8% gradient typical for takeoff
    ROC_fps = V_climb_fps * climb_gradient
    t_climb_sec = 35 / ROC_fps if ROC_fps > 0 else 20

    # Total time
    time_sec = t_ground_sec + t_climb_sec

    # Takeoff power (max continuous power, typically 95-100% rated)
    P_takeoff_kW = self.powertrain.P_GT_kW * 0.95
    if hasattr(self.powertrain, 'P_EM_kW') and self.powertrain.P_EM_kW > 0:
        # Hybrid: can add electric power
        P_takeoff_kW = self.powertrain.P_GT_kW + self.powertrain.P_EM_kW

    # Power split
    power_split = self.powertrain.get_power_split(P_takeoff_kW, segment.Hp)

    # Add high-lift motor power if active (draws from battery)
    P_highlift_kW = get_highlift_motor_power(self, segment.blown_lift_active)
    P_highlift_W = P_highlift_kW * 1000.0

    # Consumption (add high-lift motor energy to battery draw)
    fuel_lb = power_split['fuel_rate_kg_s'] * time_sec * 2.20462
    battery_Wh = (power_split['battery_power_W'] * time_sec) / 3600 + (P_highlift_W * time_sec) / 3600

    return time_sec, fuel_lb, battery_Wh

def simulate_loiter_segment(self, segment: MissionSegment, W_lb: float, S_ft2: float) -> Tuple:
    """Loiter segment - level flight at best endurance speed"""
    # Loiter altitude (pattern altitude, typically 450-1000 ft)
    h_m = segment.altitude_start_ft * 0.3048
    _, _, rho_kg_m3, _ = atmosisa(h_m)
    rho_slug_ft3 = rho_kg_m3 / 515.379

    # Apply blown lift augmentation if active
    lift_aug_factor = self.get_lift_augmentation_factor(segment.blown_lift_active)
    CLmax_clean_effective = self.CLmax_clean * lift_aug_factor

    # Best endurance speed: minimum fuel flow = minimum (P_required)
    # For propeller aircraft: V_endurance ≈ V_stall * sqrt(3) * (1/sqrt(CD0/K1))
    # Simplified: fly at ~1.3 * V_stall for good L/D
    V_stall_fps = np.sqrt(2 * W_lb / (rho_slug_ft3 * S_ft2 * CLmax_clean_effective))
    V_loiter_fps = 1.3 * V_stall_fps

    # Level flight power required
    CL_base = W_lb / (0.5 * rho_slug_ft3 * V_loiter_fps**2 * S_ft2)
    CL = CL_base * lift_aug_factor
    # Note: CD calculation uses base CL since induced drag is based on actual circulation
    CD = self.CD0 + self.K1 * CL_base**2
    D_lb = 0.5 * rho_slug_ft3 * V_loiter_fps**2 * S_ft2 * CD

    P_shaft_HP = D_lb * V_loiter_fps / (550 * self.tech.prop_efficiency)
    P_shaft_kW = P_shaft_HP * 0.7457

    # Power split
    power_split = self.powertrain.get_power_split(P_shaft_kW, segment.Hp)

    # Loiter time (typically 30 minutes for reserves - FAA requirement)
    time_sec = 30 * 60  # 30 minutes

    # Add high-lift motor power if active (draws from battery)
    P_highlift_kW = get_highlift_motor_power(self, segment.blown_lift_active)
    P_highlift_W = P_highlift_kW * 1000.0

    # Consumption (add high-lift motor energy to battery draw)
    fuel_lb = power_split['fuel_rate_kg_s'] * time_sec * 2.20462
    battery_Wh = (power_split['battery_power_W'] * time_sec) / 3600 + (P_highlift_W * time_sec) / 3600

    return time_sec, fuel_lb, battery_Wh

def simulate_landing_segment(self, segment: MissionSegment, W_lb: float, S_ft2: float) -> Tuple:
    """Landing segment - approach + flare + ground roll"""
    # Pattern altitude conditions
    h_m = segment.altitude_start_ft * 0.3048
    _, _, rho_kg_m3, _ = atmosisa(h_m)
    rho_slug_ft3 = rho_kg_m3 / 515.379

    # Apply blown lift augmentation if active
    lift_aug_factor = self.get_lift_augmentation_factor(segment.blown_lift_active)
    CLmax_land_effective = self.CLmax_land * lift_aug_factor

    # Approach speed (1.3 * stall speed with landing flaps)
    V_stall_land_fps = np.sqrt(2 * W_lb / (rho_slug_ft3 * S_ft2 * CLmax_land_effective))
    V_approach_fps = 1.3 * V_stall_land_fps

    # Approach phase (450 ft descent at 3° glideslope)
    gamma_approach = 3.0 * np.pi / 180  # 3° approach
    descent_distance_ft = segment.altitude_start_ft / np.tan(gamma_approach)
    t_approach_sec = descent_distance_ft / V_approach_fps

    # Power required for 3° approach (reduced thrust)
    CL_base = W_lb / (0.5 * rho_slug_ft3 * V_approach_fps**2 * S_ft2)
    CL = CL_base * lift_aug_factor
    # Note: CD calculation uses base CL since induced drag is based on actual circulation
    CD = self.CD0 + self.K1 * CL_base**2 + 0.02  # Extra drag from landing gear/flaps
    D_lb = 0.5 * rho_slug_ft3 * V_approach_fps**2 * S_ft2 * CD
    T_required_lb = D_lb - W_lb * np.sin(gamma_approach)

    P_shaft_HP = max(0, T_required_lb * V_approach_fps / (550 * self.tech.prop_efficiency))
    P_shaft_kW = P_shaft_HP * 0.7457

    # Flare and ground roll (minimal power, reverse thrust not modeled)
    t_flare_rollout_sec = 30  # Typical: 20-40 seconds

    # Total time
    time_sec = t_approach_sec + t_flare_rollout_sec

    # Power split (average power during approach)
    power_split = self.powertrain.get_power_split(P_shaft_kW, segment.Hp)

    # Add high-lift motor power if active (draws from battery)
    P_highlift_kW = get_highlift_motor_power(self, segment.blown_lift_active)
    P_highlift_W = P_highlift_kW * 1000.0

    # Consumption (add high-lift motor energy to battery draw)
    fuel_lb = power_split['fuel_rate_kg_s'] * time_sec * 2.20462
    battery_Wh = (power_split['battery_power_W'] * time_sec) / 3600 + (P_highlift_W * time_sec) / 3600

    return time_sec, fuel_lb, battery_Wh

# ======================= Public API ======================= #
def simulate_mission(self, segments: List[MissionSegment], TOGW_lb: float,
                        S_wing_ft2: float) -> Dict:
    """Time-stepping mission simulation (Method A approach)"""
    W_current_lb = TOGW_lb
    total_fuel_lb = 0.0
    total_battery_Wh = 0.0
    total_time_sec = 0.0

    for segment in segments:
        # Simulate based on segment type
        if segment.name == 'cruise':
            t, f, b = simulate_cruise_segment(self, segment, W_current_lb, S_wing_ft2)
        elif segment.name == 'climb':
            t, f, b = simulate_climb_segment(self, segment, W_current_lb, S_wing_ft2)
        elif segment.name == 'descent':
            t, f, b = simulate_descent_segment(self, segment, W_current_lb, S_wing_ft2)
        elif segment.name == 'takeoff':
            t, f, b = simulate_takeoff_segment(self, segment, W_current_lb, S_wing_ft2)
        elif segment.name == 'loiter':
            t, f, b = simulate_loiter_segment(self, segment, W_current_lb, S_wing_ft2)
        elif segment.name == 'landing':
            t, f, b = simulate_landing_segment(self, segment, W_current_lb, S_wing_ft2)
        else:
            # Fallback for unknown segment types
            raise ValueError(f"Unknown segment type: {segment.name}")

        # Update weight
        W_current_lb -= f

        # Store results
        segment.time_sec = t
        segment.fuel_lb = f
        segment.battery_Wh = b

        # Accumulate
        total_fuel_lb += f
        total_battery_Wh += b
        total_time_sec += t

    return {
        'total_fuel_lb': total_fuel_lb,
        'total_battery_Wh': total_battery_Wh,
        'total_time_sec': total_time_sec,
        'segments': segments
    }
