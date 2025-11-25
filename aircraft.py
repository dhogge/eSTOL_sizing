from typing import Dict, List, Tuple, Optional
import numpy as np

from config_loader import load_config, ConfigLoader
from powertrain import (
    TechnologySpec,
    PowertrainBase,
    ConventionalPowertrain,
    ParallelHybridPowertrain,
    SerialHybridPowertrain,
    FullyElectricPowertrain,
    MultiEnginePowertrain,
)
from dual_motor_powertrain import DualMotorDEPPowertrain
from mission import MissionSegment, simulate_mission
from constraints import perform_constraint_analysis

# Global default config (for backward compatibility)
_default_config = load_config()

class HybridElectricAircraft:
    """
    Complete hybrid-electric aircraft sizing with validated methodology.
    """

    def __init__(self, name: str = "eSTOL-19", config_path: str = None):
        self.name = name

        # Load config from file or use default
        if config_path:
            config = ConfigLoader(config_path)
        else:
            config = _default_config

        # Store config for later use
        self.config = config

        # ===== REQUIREMENTS =====
        self.W_payload_lb = config.get('mission_requirements', 'payload_weight_lb')
        self.range_nm = config.get('mission_requirements', 'design_range_nm')
        self.cruise_speed_kts = config.get('mission_requirements', 'cruise_speed_kts')
        self.cruise_alt_ft = config.get('mission_requirements', 'cruise_altitude_ft')
        # ===== AERODYNAMICS =====
        self.AR = config.get('aerodynamics', 'aspect_ratio')
        self.e = config.get('aerodynamics', 'oswald_efficiency')
        self.CD0 = config.get('aerodynamics', 'zero_lift_drag_coefficient')
        self.K1 = 1 / (np.pi * self.AR * self.e)
        self.CLmax_clean = config.get('aerodynamics', 'CLmax_clean')
        self.CLmax_TO = config.get('aerodynamics', 'CLmax_takeoff')
        self.CLmax_land = config.get('aerodynamics', 'CLmax_landing')
        # ===== DISTRIBUTED ELECTRIC PROPULSION =====
        self.dep_enabled = config.get('dep_system', 'enabled')
        self.dep_lift_aug_max = config.get('dep_system', 'lift_augmentation_factor_max')
        self.dep_blown_span_fraction = config.get('dep_system', 'blown_span_fraction')
        self.dep_num_motors = config.get('dep_system', 'number_of_highlift_motors')
        self.dep_use_for_wing_sizing = config.get('dep_system', 'use_for_wing_sizing')
        # ===== PERFORMANCE =====
        self.V_stall_kts = config.get('performance_requirements', 'stall_speed_requirement_kts')
        self.BFL_ft = config.get('performance_requirements', 'balanced_field_length_ft')
        self.LFL_ft = config.get('performance_requirements', 'landing_field_length_ft')
        self.ROC_fpm = config.get('performance_requirements', 'rate_of_climb_sea_level_fpm')
        # ===== TECHNOLOGY =====
        self.tech = TechnologySpec.from_config(config)
        # ===== DESIGN VARIABLES =====
        self.TOGW_lb = config.get('weight_iteration', 'initial_TOGW_guess_lb')
        self.OEW_lb = 0.0
        self.W_fuel_lb = 0.0
        self.W_battery_lb = 0.0
        self.S_wing_ft2 = 0.0
        self.WS_psf = 0.0
        self.PW_hp_lb = 0.0
        self.weight_breakdown: Dict[str, float] = {}
        # Powertrain
        self.powertrain: Optional[PowertrainBase] = None
        self.Hp_design = 0.0
        self.hybridization_profile: Optional[Dict[str, float]] = None

    def set_powertrain(self, architecture: str, Hp_design: float = 0.0,
                      hybridization_profile: Optional[Dict[str, float]] = None):
        """
        Set powertrain architecture and hybridization strategy.

        Args:
            architecture: Powertrain type ('conventional', 'parallel', 'serial', 'electric',
                                          'dual_motor_dep', 'multi_engine')
            Hp_design: Single hybridization ratio for all high-power segments (backward compatible)
            hybridization_profile: Per-segment hybridization ratios (overrides Hp_design if provided)
                                  Example: {'takeoff': 0.7, 'climb': 0.4, 'cruise': 0.0, 'landing': 0.5}

        Note: For component sizing, the maximum Hp across all segments is used.
        """
        arch_map = {
            'conventional': ConventionalPowertrain,
            'parallel': ParallelHybridPowertrain,
            'serial': SerialHybridPowertrain,
            'electric': FullyElectricPowertrain,
            'dual_motor_dep': DualMotorDEPPowertrain,
            'multi_engine': MultiEnginePowertrain,
        }
        if architecture.lower() not in arch_map:
            raise ValueError(f"Unknown architecture: {architecture}")

        # Special handling for architectures requiring extra config parameters
        if architecture.lower() == 'dual_motor_dep':
            self.powertrain = DualMotorDEPPowertrain(self.tech, self.config.config)
        elif architecture.lower() == 'multi_engine':
            num_engines = self.config.get('propulsion', 'number_of_engines')
            self.powertrain = MultiEnginePowertrain(self.tech, num_engines)
        else:
            self.powertrain = arch_map[architecture.lower()](self.tech)

        # Store hybridization profile if provided, otherwise use simple Hp_design
        if hybridization_profile is not None:
            self.hybridization_profile = hybridization_profile
            # For component sizing, use maximum Hp across all segments
            max_Hp = max(hybridization_profile.values())
            self.Hp_design = max_Hp
            print(f"✓ Powertrain set: {self.powertrain.name}")
            print(f"  Hybridization profile:")
            for segment, hp in sorted(hybridization_profile.items()):
                print(f"    {segment:10s}: Hp = {hp:.2f}")
            print(f"  Component sizing based on max Hp = {max_Hp:.2f}")
        else:
            self.Hp_design = Hp_design
            self.hybridization_profile = None
            print(f"✓ Powertrain set: {self.powertrain.name} (Hp = {Hp_design:.2f})")

    def get_lift_augmentation_factor(self, blown_lift_active: bool) -> float:
        """
        Calculate effective lift augmentation factor from distributed electric propulsion.

        Based on NASA X-57 research showing ~1.7x lift augmentation from 12 high-lift propellers.
        The augmentation is scaled by the blown span fraction (portion of wing affected).

        Args:
            blown_lift_active: Whether the DEP high-lift motors are active (binary on/off)

        Returns:
            Lift augmentation multiplier to apply to CL and CLmax

        Physics:
            CL_effective = CL_base × augmentation_factor
            augmentation_factor = 1 + (lift_aug_max - 1) × blown_span_fraction

        Example:
            With lift_aug_max = 1.80 (80% increase) and blown_span_fraction = 0.65:
            augmentation_factor = 1 + (1.80 - 1.0) × 0.65 = 1.52 (52% increase)
        """
        if not blown_lift_active or not self.dep_enabled:
            return 1.0  # No augmentation

        # Calculate effective augmentation scaled by blown span fraction
        augmentation_factor = 1.0 + (self.dep_lift_aug_max - 1.0) * self.dep_blown_span_fraction

        return augmentation_factor

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

    def simulate_mission(self, segments: List[MissionSegment], TOGW_lb: float,
                         S_wing_ft2: float) -> Dict:
        return simulate_mission(self, segments, TOGW_lb, S_wing_ft2)

    def constraint_analysis(self, TOGW_lb: float) -> Tuple[float, float]:
        # Calculate blown lift augmentation factor if wing sizing uses it
        blown_lift_aug = 1.0
        use_blown_sizing = False
        if self.dep_enabled and self.dep_use_for_wing_sizing:
            blown_lift_aug = self.get_lift_augmentation_factor(blown_lift_active=True)
            use_blown_sizing = True

        return perform_constraint_analysis(
            TOGW_lb=TOGW_lb,
            V_stall_kts=self.V_stall_kts,
            BFL_ft=self.BFL_ft,
            LFL_ft=self.LFL_ft,
            CLmax_clean=self.CLmax_clean,
            CLmax_TO=self.CLmax_TO,
            CLmax_land=self.CLmax_land,
            CD0=self.CD0,
            K1=self.K1,
            cruise_speed_kts=self.cruise_speed_kts,
            cruise_alt_ft=self.cruise_alt_ft,
            prop_efficiency=self.tech.prop_efficiency,
            config=self.config,
            blown_lift_augmentation=blown_lift_aug,
            use_blown_lift_sizing=use_blown_sizing,
        )

    def calculate_OEW(self, TOGW_lb: float, S_wing_ft2: float) -> float:
        N_ult = 3.75
        W_wing_lb = 0.04674 * (TOGW_lb**0.397) * (S_wing_ft2**0.360) * (N_ult**0.397) * (self.AR**1.712)
        W_fuselage_lb = 0.23 * TOGW_lb**0.5 * 100
        W_empennage_lb = 0.04 * TOGW_lb
        W_gear_lb = 0.02 * TOGW_lb
        W_propulsion_lb = self.powertrain.get_total_propulsion_weight()
        W_systems_lb = 0.2 * TOGW_lb

        breakdown = {
            'wing': W_wing_lb,
            'fuselage': W_fuselage_lb,
            'empennage': W_empennage_lb,
            'gear': W_gear_lb,
            'propulsion': W_propulsion_lb,
            'systems': W_systems_lb,
        }
        self.weight_breakdown = breakdown
        OEW_lb = sum(breakdown.values())
        return OEW_lb

    def size_aircraft(self, max_iterations: int = 100, tolerance: float = 0.01,
                      hybridization_profile: Optional[Dict] = None) -> Dict:
        if self.powertrain is None:
            raise ValueError("Must set powertrain before sizing")

        # Use profile from set_powertrain if available, otherwise use parameter or create default
        if hybridization_profile is None:
            if self.hybridization_profile is not None:
                # Use profile set in set_powertrain()
                hybridization_profile = self.hybridization_profile
            else:
                # Default hybridization profile optimized for power-assist hybrid
                # Cruise Hp = 0.0 for optimal efficiency (all fuel, no battery draw)
                # High-power phases use Hp_design for battery assist
                hybridization_profile = {
                    'takeoff': self.Hp_design,
                    'climb': self.Hp_design,
                    'cruise': 0.0,  # CRITICAL: Cruise on fuel only for optimal efficiency
                    'descent': 0.0,
                    'loiter': 0.0,
                    'landing': self.Hp_design,
                }

        print(f"\n{'='*70}")
        print(f"SIZING: {self.name} - {self.powertrain.name}")
        print(f"{'='*70}")
        print(f"Payload:    {self.W_payload_lb:.0f} lb")
        print(f"Range:      {self.range_nm:.0f} nm")
        print(f"Cruise:     {self.cruise_speed_kts:.0f} kts at {self.cruise_alt_ft:.0f} ft")

        TOGW_guess_lb = 15000

        for iteration in range(max_iterations):
            WS_psf, P_shaft_kW = self.constraint_analysis(TOGW_guess_lb)
            S_wing_ft2 = TOGW_guess_lb / WS_psf

            self.powertrain.size_components(P_shaft_kW, self.Hp_design)

            OEW_lb = self.calculate_OEW(TOGW_guess_lb, S_wing_ft2)
            segments = self.create_mission(hybridization_profile)
            mission_results = self.simulate_mission(segments, TOGW_guess_lb, S_wing_ft2)

            W_fuel_lb = mission_results['total_fuel_lb'] * 1.06

            # ===== BATTERY SIZING WITH C-RATE CONSTRAINT =====
            # Size battery for both ENERGY and POWER requirements (take max)
            # Skip if no battery system (battery_specific_energy == 0)

            if self.tech.battery_specific_energy_Wh_kg > 0 and self.tech.battery_DOD > 0:
                # Energy requirement (existing calculation)
                W_battery_Wh = mission_results['total_battery_Wh'] / self.tech.battery_DOD
                W_battery_kWh = W_battery_Wh / 1000.0
                m_battery_energy_kg = W_battery_kWh / (self.tech.battery_specific_energy_Wh_kg / 1000.0)

                # Power requirement (NEW: c-rate constraint)
                # Calculate peak battery power across all segments
                max_battery_power_W = 0.0
                max_power_segment = ""
                for seg in mission_results['segments']:
                    if seg.time_sec > 0:
                        # Average power during segment
                        seg_power_W = (seg.battery_Wh / (seg.time_sec / 3600))
                        if seg_power_W > max_battery_power_W:
                            max_battery_power_W = seg_power_W
                            max_power_segment = seg.name

                max_battery_power_kW = max_battery_power_W / 1000.0

                if self.tech.battery_specific_power_kW_kg > 0:
                    m_battery_power_kg = max_battery_power_kW / self.tech.battery_specific_power_kW_kg
                else:
                    m_battery_power_kg = 0.0

                # Take the larger of energy-limited or power-limited sizing
                m_battery_kg = max(m_battery_energy_kg, m_battery_power_kg)
                W_battery_lb = m_battery_kg * 2.20462

                # Calculate actual c-rate for reporting
                c_rate_actual = max_battery_power_kW / W_battery_kWh if W_battery_kWh > 0 else 0.0
                c_rate_max = self.tech.battery_specific_power_kW_kg / (self.tech.battery_specific_energy_Wh_kg / 1000.0) if self.tech.battery_specific_energy_Wh_kg > 0 else 0.0

                # Determine which constraint is active
                sizing_constraint = "POWER" if m_battery_power_kg > m_battery_energy_kg else "energy"
            else:
                # No battery system
                W_battery_lb = 0.0
                W_battery_Wh = 0.0
                W_battery_kWh = 0.0
                max_battery_power_kW = 0.0
                max_power_segment = "N/A"
                c_rate_actual = 0.0
                c_rate_max = 0.0
                sizing_constraint = "N/A"

            TOGW_new_lb = OEW_lb + self.W_payload_lb + W_fuel_lb + W_battery_lb
            error = abs(TOGW_new_lb - TOGW_guess_lb) / TOGW_guess_lb

            print(f"  Iter {iteration+1:2d}: TOGW = {TOGW_new_lb:7.0f} lb, "
                  f"OEW = {OEW_lb:7.0f} lb, Fuel = {W_fuel_lb:6.0f} lb, "
                  f"Battery = {W_battery_lb:6.0f} lb ({sizing_constraint}), "
                  f"C-rate = {c_rate_actual:.2f}C, Error = {error:.4f}")

            if error < tolerance:
                print(f"  ✓ Converged in {iteration+1} iterations!")
                break

            TOGW_guess_lb = 0.7 * TOGW_guess_lb + 0.3 * TOGW_new_lb
        else:
            print(f"  ⚠ Did not converge in {max_iterations} iterations")

        self.TOGW_lb = TOGW_new_lb
        self.OEW_lb = OEW_lb
        self.W_fuel_lb = W_fuel_lb
        self.W_battery_lb = W_battery_lb
        self.S_wing_ft2 = S_wing_ft2
        self.WS_psf = TOGW_new_lb / S_wing_ft2

        fuel_fraction = W_fuel_lb / TOGW_new_lb
        battery_fraction = W_battery_lb / TOGW_new_lb
        payload_fraction = self.W_payload_lb / TOGW_new_lb

        E_fuel_Wh = W_fuel_lb * 0.453592 * 43000 / 3.6
        E_battery_Wh = W_battery_Wh
        E_total_Wh = E_fuel_Wh + E_battery_Wh
        PREE = (self.W_payload_lb * 4.44822) * (self.range_nm * 1852) / E_total_Wh

        results = {
            'TOGW_lb': TOGW_new_lb,
            'OEW_lb': OEW_lb,
            'W_fuel_lb': W_fuel_lb,
            'W_battery_lb': W_battery_lb,
            'W_payload_lb': self.W_payload_lb,
            'S_wing_ft2': S_wing_ft2,
            'WS_psf': self.WS_psf,
            'fuel_fraction': fuel_fraction,
            'battery_fraction': battery_fraction,
            'payload_fraction': payload_fraction,
            'PREE': PREE,
            'mission_time_min': mission_results['total_time_sec'] / 60,
            'converged': error < tolerance,
            'battery_c_rate': c_rate_actual,
            'battery_c_rate_max': c_rate_max,
            'battery_sizing_constraint': sizing_constraint,
            'battery_peak_power_kW': max_battery_power_kW,
            'battery_peak_segment': max_power_segment,
        }

        print(f"\n{'='*70}")
        print(f"FINAL SIZING RESULTS")
        print(f"{'='*70}")
        print(f"TOGW:           {TOGW_new_lb:8.0f} lb")
        print(f"OEW:            {OEW_lb:8.0f} lb ({OEW_lb/TOGW_new_lb*100:.1f}%)")
        print(f"Payload:        {self.W_payload_lb:8.0f} lb ({payload_fraction*100:.1f}%)")
        print(f"Fuel:           {W_fuel_lb:8.0f} lb ({fuel_fraction*100:.1f}%)")
        print(f"Battery:        {W_battery_lb:8.0f} lb ({battery_fraction*100:.1f}%)")
        print(f"")
        print(f"Wing Area:      {S_wing_ft2:8.1f} ft²")
        print(f"Wing Loading:   {self.WS_psf:8.1f} lb/ft²")
        print(f"")
        print(f"GT Power:       {self.powertrain.P_GT_kW:8.1f} kW ({self.powertrain.m_GT_lb:.0f} lb)")
        print(f"EM Power:       {self.powertrain.P_EM_kW:8.1f} kW ({self.powertrain.m_EM_lb:.0f} lb)")
        if self.powertrain.P_GEN_kW > 0:
            print(f"GEN Power:      {self.powertrain.P_GEN_kW:8.1f} kW ({self.powertrain.m_GEN_lb:.0f} lb)")
        print(f"")
        print(f"Battery Energy: {W_battery_kWh:8.1f} kWh")
        print(f"Battery Power:  {max_battery_power_kW:8.1f} kW (peak in {max_power_segment})")
        print(f"Battery C-rate: {c_rate_actual:8.2f}C (max: {c_rate_max:.2f}C)")
        print(f"Sizing Driver:  {sizing_constraint.upper()}")
        print(f"")
        print(f"PREE:           {PREE:8.3f}")
        print(f"Mission Time:   {results['mission_time_min']:8.1f} min")
        print(f"{'='*70}\n")

        return results
