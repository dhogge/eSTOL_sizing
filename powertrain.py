from dataclasses import dataclass
from typing import Dict, Optional

# NOTE: config_loader is not imported here; TechnologySpec is created via from_config in main file.

@dataclass
class TechnologySpec:
    """
    Component technology specifications.
    Values are provided via a config-like object with .get(section, key, subkey).
    """
    GT_specific_power_kW_kg: float = None
    GT_efficiency: float = None
    GT_BSFC_kg_kWh: float = None
    EM_specific_power_kW_kg: float = None
    EM_efficiency: float = None
    GEN_specific_power_kW_kg: float = None
    GEN_efficiency: float = None
    battery_specific_energy_Wh_kg: float = None
    battery_specific_power_kW_kg: float = None
    battery_SOC_margin: float = None
    battery_DOD: float = None
    prop_efficiency: float = None

    @classmethod
    def from_config(cls, cfg):
        return cls(
            GT_specific_power_kW_kg=cfg.get('hybrid_system', 'gas_turbine', 'specific_power_kW_kg'),
            GT_efficiency=cfg.get('hybrid_system', 'gas_turbine', 'efficiency'),
            GT_BSFC_kg_kWh=cfg.get('hybrid_system', 'gas_turbine', 'BSFC_kg_kWh'),
            EM_specific_power_kW_kg=cfg.get('hybrid_system', 'electric_motor', 'specific_power_kW_kg'),
            EM_efficiency=cfg.get('hybrid_system', 'electric_motor', 'efficiency'),
            GEN_specific_power_kW_kg=cfg.get('hybrid_system', 'generator', 'specific_power_kW_kg'),
            GEN_efficiency=cfg.get('hybrid_system', 'generator', 'efficiency'),
            battery_specific_energy_Wh_kg=cfg.get('hybrid_system', 'battery', 'specific_energy_Wh_kg'),
            battery_specific_power_kW_kg=cfg.get('hybrid_system', 'battery', 'specific_power_kW_kg'),
            battery_SOC_margin=cfg.get('hybrid_system', 'battery', 'SOC_margin_percent') / 100,
            battery_DOD=cfg.get('hybrid_system', 'battery', 'depth_of_discharge_percent') / 100,
            prop_efficiency=cfg.get('propulsion', 'propeller_efficiency'),
        )

class PowertrainBase:
    """Base class for all powertrain architectures"""
    def __init__(self, name: str, tech: TechnologySpec):
        self.name = name
        self.tech = tech
        # Component masses (lb)
        self.m_GT_lb = 0.0
        self.m_EM_lb = 0.0
        self.m_GEN_lb = 0.0
        self.m_battery_lb = 0.0
        self.m_fuel_lb = 0.0
        # Component powers (kW)
        self.P_GT_kW = 0.0
        self.P_EM_kW = 0.0
        self.P_GEN_kW = 0.0

    def size_components(self, P_shaft_kW: float, Hp: float):
        raise NotImplementedError

    def get_power_split(self, P_required_kW: float, Hp: float) -> Dict:
        raise NotImplementedError

    def get_total_propulsion_weight(self) -> float:
        """Total propulsion system weight (excluding fuel/battery)."""
        return self.m_GT_lb + self.m_EM_lb + self.m_GEN_lb

class ConventionalPowertrain(PowertrainBase):
    """Conventional fuel-only powertrain"""
    def __init__(self, tech: TechnologySpec):
        super().__init__("Conventional", tech)

    def size_components(self, P_shaft_kW: float, Hp: float = 0.0):
        self.P_GT_kW = P_shaft_kW
        self.m_GT_lb = (P_shaft_kW / self.tech.GT_specific_power_kW_kg) * 2.20462
        self.m_EM_lb = 0.0
        self.m_GEN_lb = 0.0

    def get_power_split(self, P_required_kW: float, Hp: float = 0.0) -> Dict:
        fuel_rate_kg_s = (P_required_kW * self.tech.GT_BSFC_kg_kWh) / 3600
        return {
            'P_GT_kW': P_required_kW,
            'P_EM_kW': 0.0,
            'fuel_rate_kg_s': fuel_rate_kg_s,
            'battery_power_W': 0.0,
        }

class ParallelHybridPowertrain(PowertrainBase):
    """
    Parallel Hybrid Architecture: GT and EM both connect to propeller shaft.
    Hp = P_EM / (P_EM + P_GT)
    """
    def __init__(self, tech: TechnologySpec):
        super().__init__("Parallel Hybrid", tech)

    def size_components(self, P_shaft_kW: float, Hp: float):
        self.P_EM_kW = P_shaft_kW * Hp
        self.P_GT_kW = P_shaft_kW * (1 - Hp)
        self.m_EM_lb = (self.P_EM_kW / self.tech.EM_specific_power_kW_kg) * 2.20462
        self.m_GT_lb = (self.P_GT_kW / self.tech.GT_specific_power_kW_kg) * 2.20462
        self.m_GEN_lb = 0.0

    def get_power_split(self, P_required_kW: float, Hp: float) -> Dict:
        P_GT_kW = P_required_kW * (1 - Hp)
        P_EM_kW = P_required_kW * Hp
        fuel_rate_kg_s = (P_GT_kW * self.tech.GT_BSFC_kg_kWh) / 3600
        battery_power_W = (P_EM_kW * 1000) / self.tech.EM_efficiency
        return {
            'P_GT_kW': P_GT_kW,
            'P_EM_kW': P_EM_kW,
            'fuel_rate_kg_s': fuel_rate_kg_s,
            'battery_power_W': battery_power_W,
        }

class SerialHybridPowertrain(PowertrainBase):
    """
    Serial Hybrid: GT drives generator, EM drives propeller.
    Battery supplements generator power during peak loads.

    CRITICAL: Generator is sized for 100% of CRUISE power (continuous operation).
    Battery adds extra power during takeoff/climb based on Hp parameter.

    Example with P_cruise=500 kW, Hp=0.5:
      - Generator: 500 kW (sized for 100% of cruise)
      - Motor: 500 kW (cruise) or 750 kW (takeoff with 50% battery boost)
      - During cruise (Hp=0.0): Gen provides 500 kW, Battery provides 0 kW
      - During takeoff (Hp=0.5): Gen provides 500 kW, Battery adds 250 kW
    """
    def __init__(self, tech: TechnologySpec):
        super().__init__("Serial Hybrid", tech)

    def size_components(self, P_shaft_kW: float, Hp: float):
        """
        Size components for serial hybrid.

        Generator is sized for 100% of cruise power (continuous operation).
        Motor is sized for cruise + peak battery assist.
        Hp parameter controls battery boost during high-power phases.
        """
        # Motor sized for cruise + battery boost during peaks
        self.P_EM_kW = P_shaft_kW * (1 + Hp)  # Can handle cruise + battery boost

        # Electric power required for cruise (through motor efficiency)
        P_electric_cruise_kW = P_shaft_kW / self.tech.EM_efficiency

        # Generator sized for 100% of cruise requirement (NOT reduced by Hp!)
        self.P_GEN_kW = P_electric_cruise_kW

        # GT sized to drive generator
        self.P_GT_kW = self.P_GEN_kW / self.tech.GEN_efficiency

        # Component weights
        self.m_EM_lb = (self.P_EM_kW / self.tech.EM_specific_power_kW_kg) * 2.20462
        self.m_GEN_lb = (self.P_GEN_kW / self.tech.GEN_specific_power_kW_kg) * 2.20462
        self.m_GT_lb = (self.P_GT_kW / self.tech.GT_specific_power_kW_kg) * 2.20462

    def get_power_split(self, P_required_kW: float, Hp: float) -> Dict:
        """
        Calculate power distribution during flight.

        Generator provides what it can (up to its rating).
        Battery supplements during high-power phases when demand exceeds generator capacity.
        """
        P_EM_kW = P_required_kW
        P_electric_kW = P_EM_kW / self.tech.EM_efficiency

        # Generator provides power up to its rating (throttles down if less power needed)
        P_generator_kW = min(P_electric_kW, self.P_GEN_kW)

        # Battery supplements when demand exceeds generator capacity
        P_battery_kW = max(0, P_electric_kW - P_generator_kW)

        # GT powers the generator at the required level
        P_GT_kW = P_generator_kW / self.tech.GEN_efficiency

        fuel_rate_kg_s = (P_GT_kW * self.tech.GT_BSFC_kg_kWh) / 3600
        battery_power_W = P_battery_kW * 1000

        return {
            'P_GT_kW': P_GT_kW,
            'P_EM_kW': P_EM_kW,
            'fuel_rate_kg_s': fuel_rate_kg_s,
            'battery_power_W': battery_power_W,
        }

class FullyElectricPowertrain(PowertrainBase):
    """Fully electric - battery only."""
    def __init__(self, tech: TechnologySpec):
        super().__init__("Fully Electric", tech)

    def size_components(self, P_shaft_kW: float, Hp: float = 1.0):
        self.P_EM_kW = P_shaft_kW
        self.m_EM_lb = (P_shaft_kW / self.tech.EM_specific_power_kW_kg) * 2.20462
        self.m_GT_lb = 0.0
        self.m_GEN_lb = 0.0

    def get_power_split(self, P_required_kW: float, Hp: float = 1.0) -> Dict:
        battery_power_W = (P_required_kW * 1000) / self.tech.EM_efficiency
        return {
            'P_GT_kW': 0.0,
            'P_EM_kW': P_required_kW,
            'fuel_rate_kg_s': 0.0,
            'battery_power_W': battery_power_W,
        }

class MultiEnginePowertrain(PowertrainBase):
    """
    Multi-Engine Parallel Hybrid Architecture.

    Supports N independent engines (typically 2 turboprops), each with:
    - Gas turbine (GT) on propeller shaft
    - Optional electric motor (EM) on same shaft for hybrid assist

    Key Features:
    - Independent power control per engine
    - Symmetric power distribution (equal power per engine)
    - OEI (One Engine Inoperative) capability
    - Per-engine fuel flow tracking
    - Parallel hybrid per engine (GT + EM share shaft)

    Example with 2 engines, P_total=1000 kW, Hp=0.3:
      - Each engine: 500 kW total
      - Per engine: GT=350 kW (70%), EM=150 kW (30%)
      - Total GT: 700 kW, Total EM: 300 kW

    OEI Scenario (one engine out):
      - Operating engine must provide full required power
      - Battery can boost to compensate for lost engine
    """
    def __init__(self, tech: TechnologySpec, num_engines: int = 2):
        super().__init__(f"Multi-Engine ({num_engines} engines)", tech)
        self.num_engines = num_engines
        # Per-engine powers (each engine has GT + EM)
        self.P_GT_per_engine_kW = 0.0
        self.P_EM_per_engine_kW = 0.0
        # Individual engine component masses
        self.m_GT_per_engine_lb = 0.0
        self.m_EM_per_engine_lb = 0.0

    def size_components(self, P_shaft_kW: float, Hp: float):
        """
        Size multi-engine powertrain components.

        Args:
            P_shaft_kW: Total shaft power required (sum across all engines)
            Hp: Hybridization ratio (electric fraction per engine)

        Sizing Philosophy:
        - All-Engines-Operating (AEO): Power distributed equally across engines
        - Each engine sized for: P_per_engine = P_total / num_engines
        - Each engine has: GT (1-Hp fraction) + EM (Hp fraction)
        - OEI capability: Single engine can provide full power with battery boost
        """
        # Power per engine (symmetric distribution)
        P_per_engine_kW = P_shaft_kW / self.num_engines

        # Each engine: parallel hybrid architecture (GT + EM on same shaft)
        self.P_GT_per_engine_kW = P_per_engine_kW * (1 - Hp)
        self.P_EM_per_engine_kW = P_per_engine_kW * Hp

        # Total powers (sum across all engines)
        self.P_GT_kW = self.P_GT_per_engine_kW * self.num_engines
        self.P_EM_kW = self.P_EM_per_engine_kW * self.num_engines

        # Component masses per engine
        self.m_GT_per_engine_lb = (self.P_GT_per_engine_kW / self.tech.GT_specific_power_kW_kg) * 2.20462
        self.m_EM_per_engine_lb = (self.P_EM_per_engine_kW / self.tech.EM_specific_power_kW_kg) * 2.20462

        # Total masses (all engines)
        self.m_GT_lb = self.m_GT_per_engine_lb * self.num_engines
        self.m_EM_lb = self.m_EM_per_engine_lb * self.num_engines
        self.m_GEN_lb = 0.0  # No generator in parallel architecture

    def get_power_split(self, P_required_kW: float, Hp: float, oei_mode: bool = False) -> Dict:
        """
        Calculate power distribution during flight.

        Args:
            P_required_kW: Total required shaft power
            Hp: Hybridization ratio for this flight segment
            oei_mode: One Engine Inoperative mode (single engine operation)

        Returns:
            Dict with power breakdown and consumption rates

        Normal Operation (AEO):
            - Power split equally across all engines
            - Each engine: GT (1-Hp) + EM (Hp)

        OEI Operation:
            - Only one engine operating
            - May require battery boost if P_required > P_single_engine
        """
        if oei_mode:
            # Single engine operation
            num_operating_engines = 1
            P_per_engine_kW = P_required_kW  # One engine provides all power
        else:
            # All engines operating
            num_operating_engines = self.num_engines
            P_per_engine_kW = P_required_kW / self.num_engines

        # Power split per engine (parallel hybrid)
        P_GT_per_engine_kW = P_per_engine_kW * (1 - Hp)
        P_EM_per_engine_kW = P_per_engine_kW * Hp

        # Total powers
        P_GT_total_kW = P_GT_per_engine_kW * num_operating_engines
        P_EM_total_kW = P_EM_per_engine_kW * num_operating_engines

        # Fuel consumption (all operating GTs)
        fuel_rate_kg_s = (P_GT_total_kW * self.tech.GT_BSFC_kg_kWh) / 3600

        # Battery power (all operating EMs, accounting for motor efficiency)
        battery_power_W = (P_EM_total_kW * 1000) / self.tech.EM_efficiency

        return {
            'P_GT_kW': P_GT_total_kW,
            'P_EM_kW': P_EM_total_kW,
            'fuel_rate_kg_s': fuel_rate_kg_s,
            'battery_power_W': battery_power_W,
            'num_operating_engines': num_operating_engines,
            'P_per_engine_kW': P_per_engine_kW,
            'P_GT_per_engine_kW': P_GT_per_engine_kW,
            'P_EM_per_engine_kW': P_EM_per_engine_kW,
        }
