"""
Dual Motor DEP Powertrain for X-57 Maxwell style distributed electric propulsion.

This module implements a powertrain with:
  - 12 high-lift motors (inboard wing) for blown lift
  - 2 cruise motors (wingtip/outboard) for primary propulsion
  - Power management for switching between motor sets
"""

from dataclasses import dataclass
from typing import Dict, List
import numpy as np

@dataclass
class MotorSet:
    """Represents a set of identical motors"""
    name: str
    num_motors: int
    power_per_motor_kW: float
    weight_per_motor_lb: float
    efficiency: float
    active_phases: List[str]  # Which flight phases this set is active
    can_fold: bool = False
    folded_drag_cd: float = 0.001

    @property
    def total_power_kW(self) -> float:
        return self.num_motors * self.power_per_motor_kW

    @property
    def total_weight_lb(self) -> float:
        return self.num_motors * self.weight_per_motor_lb

    def is_active(self, flight_phase: str) -> bool:
        """Check if this motor set is active during given flight phase"""
        return 'all' in self.active_phases or flight_phase in self.active_phases


class DualMotorDEPPowertrain:
    """
    Dual motor distributed electric propulsion powertrain.

    Implements NASA X-57 Maxwell style architecture with separate high-lift
    and cruise motor sets.
    """

    def __init__(self, tech_spec, config: Dict):
        """
        Initialize dual motor DEP powertrain

        Args:
            tech_spec: TechnologySpec with battery/motor parameters
            config: Configuration dict with DEP system parameters
        """
        self.tech = tech_spec
        self.config = config

        # High-lift motors (inboard, for blown lift)
        hl_config = config['dep_system']['highlift_motors']
        self.highlift_motors = MotorSet(
            name="High-Lift Motors",
            num_motors=hl_config['number_of_motors'],
            power_per_motor_kW=hl_config['power_per_motor_kW'],
            weight_per_motor_lb=hl_config['weight_per_motor_lb'],
            efficiency=hl_config['efficiency'],
            active_phases=hl_config['active_phases'],
            can_fold=hl_config['folding']['can_fold'],
            folded_drag_cd=hl_config['folding']['folded_drag_coefficient_increment']
        )

        # Cruise motors (will be sized based on mission requirements)
        self.cruise_motors = None  # Sized during aircraft sizing
        self.num_cruise_motors = config['dep_system']['cruise_motors']['number_of_motors']

        # Determine cruise motor architecture
        cruise_config = config['dep_system']['cruise_motors']
        self.cruise_architecture = cruise_config.get('architecture', 'pure_electric')

        # Set name based on architecture
        if self.cruise_architecture == 'parallel_hybrid':
            self.name = "Dual Motor DEP (Parallel Hybrid)"
        else:
            self.name = "Dual Motor DEP (Pure Electric)"

        # Power source
        self.power_architecture = config['dep_system']['power_architecture']['type']
        self.battery_specific_energy_Wh_kg = tech_spec.battery_specific_energy_Wh_kg
        self.battery_DOD = tech_spec.battery_DOD

        # Component weights (to be calculated)
        self.m_battery_lb = 0.0
        self.m_highlift_lb = self.highlift_motors.total_weight_lb
        self.m_cruise_motors_lb = 0.0
        self.m_wiring_lb = 0.0
        self.m_controllers_lb = 0.0

        # Compatibility properties for existing sizing loop
        self.P_GT_kW = 0.0  # Gas turbine power (for parallel hybrid cruise motors)
        self.P_EM_kW = 0.0  # Electric motor power (cruise motors)
        self.P_GEN_kW = 0.0  # No generator in this architecture
        self.m_GT_lb = 0.0
        self.m_EM_lb = 0.0
        self.m_GEN_lb = 0.0

    def size_cruise_motors(self, P_cruise_kW: float, Hp: float = 0.0):
        """
        Size the cruise motors based on cruise power requirement

        Args:
            P_cruise_kW: Required cruise power (kW)
            Hp: Hybridization ratio for sizing (electric fraction during peak power)
        """
        cr_config = self.config['dep_system']['cruise_motors']
        num_motors = cr_config['number_of_motors']

        # Power per motor (total shaft power / number of motors)
        power_per_motor = P_cruise_kW / num_motors

        if self.cruise_architecture == 'parallel_hybrid':
            # Parallel hybrid: GT + EM on same shaft
            # Size components for peak power (using Hp as electric fraction)

            # Electric motor power per cruise motor (for boost)
            P_EM_per_motor = power_per_motor * Hp
            P_EM_total = P_EM_per_motor * num_motors

            # Gas turbine power per cruise motor (for sustained cruise)
            P_GT_per_motor = power_per_motor * (1 - Hp)
            P_GT_total = P_GT_per_motor * num_motors

            # Size electric motors (cruise)
            EM_specific_power = cr_config['specific_power_kW_kg']
            m_EM_per_motor_kg = P_EM_per_motor / EM_specific_power
            m_EM_per_motor_lb = m_EM_per_motor_kg * 2.20462

            # Size gas turbines (one per cruise motor)
            GT_specific_power = self.tech.GT_specific_power_kW_kg
            m_GT_per_motor_kg = P_GT_per_motor / GT_specific_power
            m_GT_per_motor_lb = m_GT_per_motor_kg * 2.20462

            # Total cruise motor weight (GT + EM per motor)
            weight_per_motor_lb = m_GT_per_motor_lb + m_EM_per_motor_lb

            # Create cruise motor set
            self.cruise_motors = MotorSet(
                name="Cruise Motors (Parallel Hybrid)",
                num_motors=num_motors,
                power_per_motor_kW=power_per_motor,
                weight_per_motor_lb=weight_per_motor_lb,
                efficiency=cr_config['efficiency'],
                active_phases=['all'],
                can_fold=False
            )

            # Update compatibility properties
            self.P_GT_kW = P_GT_total
            self.P_EM_kW = P_EM_total
            self.m_GT_lb = m_GT_per_motor_lb * num_motors
            self.m_EM_lb = m_EM_per_motor_lb * num_motors
            self.m_cruise_motors_lb = self.cruise_motors.total_weight_lb

        else:
            # Pure electric: Battery → EM only

            # Weight per motor (using specific power)
            specific_power_kW_kg = cr_config['specific_power_kW_kg']
            weight_per_motor_kg = power_per_motor / specific_power_kW_kg
            weight_per_motor_lb = weight_per_motor_kg * 2.20462

            # Create cruise motor set
            self.cruise_motors = MotorSet(
                name="Cruise Motors (Pure Electric)",
                num_motors=num_motors,
                power_per_motor_kW=power_per_motor,
                weight_per_motor_lb=weight_per_motor_lb,
                efficiency=cr_config['efficiency'],
                active_phases=['all'],
                can_fold=False
            )

            self.m_cruise_motors_lb = self.cruise_motors.total_weight_lb

            # Update compatibility properties (no GT in pure electric)
            self.P_GT_kW = 0.0
            self.P_EM_kW = self.cruise_motors.total_power_kW
            self.m_GT_lb = 0.0
            self.m_EM_lb = self.m_cruise_motors_lb

        # Size wiring and controllers
        total_power_kW = P_cruise_kW + self.highlift_motors.total_power_kW
        self.m_wiring_lb = total_power_kW * self.config['dep_system']['wiring_and_controls']['wiring_weight_factor']
        self.m_controllers_lb = total_power_kW * self.config['dep_system']['wiring_and_controls']['controller_weight_per_kW']

    def size_components(self, P_shaft_kW: float, Hp: float):
        """
        Compatibility method for existing sizing loop interface.

        Args:
            P_shaft_kW: Cruise power requirement from constraint analysis
            Hp: Hybridization ratio (max across all segments)
        """
        # Size cruise motors for cruise power
        self.size_cruise_motors(P_shaft_kW, Hp)

    def get_power_split(self, P_required_kW: float, Hp: float) -> Dict:
        """
        Calculate power distribution for dual motor DEP system.

        NOTE: This method returns power split for CRUISE MOTORS ONLY.
        High-lift motor power is added separately in mission.py based on
        blown_lift_active flag.

        Args:
            P_required_kW: Shaft power required (from propulsion simulation)
            Hp: Hybridization ratio for this mission segment (electric fraction)

        Returns:
            Dict with power breakdown for cruise motors and battery
        """
        if self.cruise_motors is None:
            # Motors not sized yet
            return {
                'P_GT_kW': 0.0,
                'P_EM_kW': 0.0,
                'fuel_rate_kg_s': 0.0,
                'battery_power_W': 0.0,
            }

        # Cruise motors provide all shaft power
        # (High-lift motors handled separately in mission.py)
        P_cruise_shaft_kW = P_required_kW

        if self.cruise_architecture == 'parallel_hybrid':
            # Parallel hybrid: GT + EM on same shaft
            # Power split based on Hp (electric fraction)
            P_EM_shaft_kW = P_cruise_shaft_kW * Hp
            P_GT_shaft_kW = P_cruise_shaft_kW * (1 - Hp)

            # Electrical power required from battery (accounting for motor efficiency)
            cr_efficiency = self.cruise_motors.efficiency
            P_elec_cruise_kW = P_EM_shaft_kW / cr_efficiency

            # Fuel consumption from gas turbines
            # BSFC is already kg/kWh of output power, so no efficiency division needed
            BSFC = self.tech.GT_BSFC_kg_kWh  # kg/kWh
            fuel_rate_kg_s = (P_GT_shaft_kW * BSFC) / 3600.0

            # Battery power draw (in Watts)
            battery_power_W = P_elec_cruise_kW * 1000.0

            return {
                'P_GT_kW': P_GT_shaft_kW,
                'P_EM_kW': P_EM_shaft_kW,
                'fuel_rate_kg_s': fuel_rate_kg_s,
                'battery_power_W': battery_power_W,
            }

        else:
            # Pure electric: All power from battery

            # Electrical power required (accounting for motor efficiency)
            cr_efficiency = self.cruise_motors.efficiency
            P_elec_cruise_kW = P_cruise_shaft_kW / cr_efficiency

            # Battery power draw (in Watts for consistency with mission sim)
            battery_power_W = P_elec_cruise_kW * 1000.0

            return {
                'P_GT_kW': 0.0,  # No gas turbine in pure electric
                'P_EM_kW': P_cruise_shaft_kW,  # Cruise motor power
                'fuel_rate_kg_s': 0.0,  # Fully electric, no fuel
                'battery_power_W': battery_power_W,  # Battery draw for cruise motors
            }

    def size_battery(self, mission_energy_Wh: float):
        """
        Size battery based on total mission energy requirement

        Args:
            mission_energy_Wh: Total energy required for mission
        """
        # Add reserve margin
        energy_with_reserve = mission_energy_Wh / self.battery_DOD

        # Battery weight
        energy_kg = energy_with_reserve / self.battery_specific_energy_Wh_kg
        self.m_battery_lb = energy_kg * 2.20462

    def get_total_propulsion_weight(self) -> float:
        """Get total weight of propulsion system"""
        return (self.m_battery_lb +
                self.m_highlift_lb +
                self.m_cruise_motors_lb +
                self.m_wiring_lb +
                self.m_controllers_lb)

    def get_weight_breakdown(self) -> Dict[str, float]:
        """Get detailed weight breakdown"""
        return {
            'battery': self.m_battery_lb,
            'highlift_motors': self.m_highlift_lb,
            'cruise_motors': self.m_cruise_motors_lb,
            'wiring': self.m_wiring_lb,
            'controllers': self.m_controllers_lb,
            'total': self.get_total_propulsion_weight()
        }

    def get_drag_increment(self, flight_phase: str, S_wing_ft2: float) -> float:
        """
        Calculate drag increment from folded high-lift motors

        Args:
            flight_phase: Current flight phase
            S_wing_ft2: Wing reference area

        Returns:
            Additional drag force (lb) from folded motors
        """
        # If high-lift motors are folded, add drag increment
        if not self.highlift_motors.is_active(flight_phase) and self.highlift_motors.can_fold:
            # Additional CD from folded nacelles
            delta_CD = self.highlift_motors.folded_drag_cd
            return delta_CD * S_wing_ft2  # Simplified, should include dynamic pressure
        return 0.0


def create_dual_motor_dep_example():
    """Example of how to use the DualMotorDEPPowertrain class"""

    # Mock technology spec
    from powertrain import TechnologySpec
    tech = TechnologySpec(
        GT_specific_power_kW_kg=0.0,  # Not used for fully electric
        GT_efficiency=0.0,
        GT_BSFC_kg_kWh=0.0,
        EM_specific_power_kW_kg=5.0,
        EM_efficiency=0.95,
        GEN_specific_power_kW_kg=0.0,
        GEN_efficiency=0.0,
        battery_specific_energy_Wh_kg=250,
        battery_specific_power_kW_kg=1.0,
        battery_SOC_margin=0.2,
        battery_DOD=0.8,
        prop_efficiency=0.85
    )

    # Load dual motor config
    import json
    with open('config_dual_motor_dep.json', 'r') as f:
        config = json.load(f)

    # Create powertrain
    powertrain = DualMotorDEPPowertrain(tech, config)

    # Size cruise motors for 500 kW cruise power
    powertrain.size_cruise_motors(P_cruise_kW=500)

    # Get power split for different phases
    print("="*70)
    print("DUAL MOTOR DEP POWER MANAGEMENT")
    print("="*70)

    for phase, P_req in [('takeoff', 800), ('cruise', 500), ('landing', 300)]:
        power_split = powertrain.get_power_split(P_req, phase)
        print(f"\n{phase.upper()}:")
        print(f"  High-lift motors: {'ACTIVE' if power_split['highlift_active'] else 'FOLDED'}")
        print(f"  Cruise motors: {'ACTIVE' if power_split['cruise_active'] else 'OFF'}")
        print(f"  High-lift power: {power_split['P_highlift_shaft_kW']:.1f} kW")
        print(f"  Cruise power: {power_split['P_cruise_shaft_kW']:.1f} kW")
        print(f"  Total shaft power: {power_split['P_total_shaft_kW']:.1f} kW")
        print(f"  Battery draw: {power_split['P_battery_kW']:.1f} kW")

    # Show weight breakdown
    print("\n" + "="*70)
    print("WEIGHT BREAKDOWN")
    print("="*70)
    weights = powertrain.get_weight_breakdown()
    for component, weight in weights.items():
        print(f"  {component:20s}: {weight:8.1f} lb")

    print("="*70)


if __name__ == "__main__":
    create_dual_motor_dep_example()
