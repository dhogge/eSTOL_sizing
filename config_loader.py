"""
Configuration Loader for eSTOL Hybrid Aircraft Sizing

This module provides utilities to load and access configuration parameters
from the config.json file, providing a clean interface for all Python scripts.
"""

import json
import os
from typing import Any, Dict
from dataclasses import dataclass


class ConfigLoader:
    """Load and access configuration parameters from config.json"""

    def __init__(self, config_path: str = None):
        """
        Initialize config loader

        Parameters:
        -----------
        config_path : str, optional
            Path to config.json. If None, looks in script directory.
        """
        if config_path is None:
            # Default to config.json in same directory as this script
            script_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(script_dir, 'config.json')

        self.config_path = config_path
        self.config = self._load_config()

    def _load_config(self) -> Dict:
        """Load configuration from JSON file"""
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Configuration file not found: {self.config_path}\n"
                "Please ensure config.json exists in the same directory."
            )
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in configuration file: {e}")

    def get(self, *keys, default=None) -> Any:
        """
        Get configuration value using dot notation

        Parameters:
        -----------
        *keys : str
            Nested keys to access (e.g., 'aerodynamics', 'aspect_ratio')
        default : Any
            Default value if key not found

        Returns:
        --------
        Any : Configuration value

        Example:
        --------
        >>> config = ConfigLoader()
        >>> AR = config.get('aerodynamics', 'aspect_ratio')
        >>> battery_energy = config.get('hybrid_system', 'battery', 'specific_energy_Wh_kg')
        """
        value = self.config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value

    def get_section(self, section: str) -> Dict:
        """
        Get entire configuration section

        Parameters:
        -----------
        section : str
            Top-level section name

        Returns:
        --------
        Dict : Configuration section
        """
        return self.config.get(section, {})

    def reload(self):
        """Reload configuration from file"""
        self.config = self._load_config()

    def save(self, config_path: str = None):
        """
        Save current configuration to file

        Parameters:
        -----------
        config_path : str, optional
            Path to save config. If None, uses original path.
        """
        save_path = config_path or self.config_path
        with open(save_path, 'w') as f:
            json.dump(self.config, f, indent=2)

    def update(self, *keys, value):
        """
        Update configuration value

        Parameters:
        -----------
        *keys : str
            Nested keys to access
        value : Any
            New value to set

        Example:
        --------
        >>> config.update('mission_requirements', 'design_range_nm', value=500)
        """
        d = self.config
        for key in keys[:-1]:
            d = d.setdefault(key, {})
        d[keys[-1]] = value


@dataclass
class TechnologySpec:
    """
    Component technology specifications

    Compatible with estol_hybrid_sizing.py structure
    """
    # Gas Turbine
    GT_specific_power_kW_kg: float
    GT_efficiency: float
    GT_BSFC_kg_kWh: float

    # Electric Motor
    EM_specific_power_kW_kg: float
    EM_efficiency: float

    # Generator
    GEN_specific_power_kW_kg: float
    GEN_efficiency: float

    # Battery System
    battery_specific_energy_Wh_kg: float
    battery_specific_power_kW_kg: float
    battery_SOC_margin: float
    battery_DOD: float

    # Propeller
    prop_efficiency: float

    @classmethod
    def from_config(cls, config: ConfigLoader):
        """Create TechnologySpec from ConfigLoader"""
        return cls(
            GT_specific_power_kW_kg=config.get('hybrid_system', 'gas_turbine', 'specific_power_kW_kg'),
            GT_efficiency=config.get('hybrid_system', 'gas_turbine', 'efficiency'),
            GT_BSFC_kg_kWh=config.get('hybrid_system', 'gas_turbine', 'BSFC_kg_kWh'),
            EM_specific_power_kW_kg=config.get('hybrid_system', 'electric_motor', 'specific_power_kW_kg'),
            EM_efficiency=config.get('hybrid_system', 'electric_motor', 'efficiency'),
            GEN_specific_power_kW_kg=config.get('hybrid_system', 'generator', 'specific_power_kW_kg'),
            GEN_efficiency=config.get('hybrid_system', 'generator', 'efficiency'),
            battery_specific_energy_Wh_kg=config.get('hybrid_system', 'battery', 'specific_energy_Wh_kg'),
            battery_specific_power_kW_kg=config.get('hybrid_system', 'battery', 'specific_power_kW_kg'),
            battery_SOC_margin=config.get('hybrid_system', 'battery', 'SOC_margin_percent') / 100,
            battery_DOD=config.get('hybrid_system', 'battery', 'depth_of_discharge_percent') / 100,
            prop_efficiency=config.get('propulsion', 'propeller_efficiency')
        )


# Convenience function for quick access
def load_config(config_path: str = None) -> ConfigLoader:
    """
    Load configuration file

    Parameters:
    -----------
    config_path : str, optional
        Path to config.json

    Returns:
    --------
    ConfigLoader : Configuration loader instance
    """
    return ConfigLoader(config_path)


if __name__ == '__main__':
    # Test configuration loader
    config = load_config()

    print("=== Configuration Loaded Successfully ===\n")

    print("Aerodynamics:")
    print(f"  Aspect Ratio: {config.get('aerodynamics', 'aspect_ratio')}")
    print(f"  Oswald Efficiency: {config.get('aerodynamics', 'oswald_efficiency')}")
    print(f"  CD0: {config.get('aerodynamics', 'zero_lift_drag_coefficient')}")

    print("\nMission Requirements:")
    print(f"  Range: {config.get('mission_requirements', 'design_range_nm')} nm")
    print(f"  Cruise Speed: {config.get('mission_requirements', 'cruise_speed_kts')} kts")
    print(f"  Payload: {config.get('mission_requirements', 'payload_weight_lb')} lb")

    print("\nHybridization Profile:")
    profile = config.get_section('hybridization_profile')
    for segment, value in profile.items():
        if segment != 'description':
            print(f"  {segment}: {value*100:.1f}% electric")

    print("\nDEP System:")
    dep = config.get_section('dep_system')
    print(f"  Enabled: {dep['enabled']}")
    print(f"  High-lift motors: {dep['number_of_highlift_motors']}")
    print(f"  Power per motor: {dep['motor_power_kW']} kW")

    print("\n✓ All configuration sections accessible")
