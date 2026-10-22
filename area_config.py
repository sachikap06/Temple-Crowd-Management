"""
area_config.py
==============
Area-based configurable threshold management for temple zones.
Smart Temple Crowd Management Project

Features:
  - Multiple temple areas with independent thresholds
  - Count-based AND density-based classification
  - CrowdClassifier with hysteresis to prevent rapid status flicker
  - JSON persistence for area configurations
  - Validation to ensure thresholds are logically ordered
  - Illustrative defaults until actual capacities are provided
"""

import os
import json
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

CONFIG_FILE = os.path.join("output", "area_config.json")


# ═══════════════════════════════════════════════════════════════════════════════
# AREA CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class AreaConfig:
    """
    Configuration for a single monitored temple area.

    Threshold semantics:
      count <= count_threshold_moderate   → NORMAL
      moderate < count <= congested       → MODERATE
      count > count_threshold_congested   → CONGESTED

    Density thresholds (optional, used when area_sqm > 0):
      density = count / area_sqm
      If density exceeds density_threshold_congested → CONGESTED (overrides count)
      If density exceeds density_threshold_moderate  → MODERATE  (overrides count)
      Set to 0.0 to disable density-based classification.
    """
    name: str
    count_threshold_moderate: int = 15
    count_threshold_congested: int = 30
    area_sqm: float = 0.0                     # 0 = unknown area
    density_threshold_moderate: float = 0.0    # people/sqm; 0 = disabled
    density_threshold_congested: float = 0.0   # people/sqm; 0 = disabled
    max_capacity: int = 50

    def validate(self) -> List[str]:
        """Validate thresholds are logically ordered. Returns error messages."""
        errors = []
        if not self.name or not self.name.strip():
            errors.append("Area name must not be empty")
        if self.count_threshold_moderate < 1:
            errors.append(
                f"Moderate threshold must be >= 1 (got {self.count_threshold_moderate})"
            )
        if self.count_threshold_congested <= self.count_threshold_moderate:
            errors.append(
                f"Congested threshold ({self.count_threshold_congested}) "
                f"must be > Moderate threshold ({self.count_threshold_moderate})"
            )
        if self.max_capacity < self.count_threshold_congested:
            errors.append(
                f"Max capacity ({self.max_capacity}) should be >= "
                f"Congested threshold ({self.count_threshold_congested})"
            )
        if self.area_sqm < 0:
            errors.append("Area (sqm) must be non-negative")
        if self.density_threshold_moderate < 0:
            errors.append("Density moderate threshold must be non-negative")
        if self.density_threshold_congested < 0:
            errors.append("Density congested threshold must be non-negative")
        if (
            self.density_threshold_congested > 0
            and self.density_threshold_moderate > 0
            and self.density_threshold_congested <= self.density_threshold_moderate
        ):
            errors.append(
                f"Density congested ({self.density_threshold_congested}) "
                f"must be > density moderate ({self.density_threshold_moderate})"
            )
        return errors

    def classify_crowd(self, count: int) -> str:
        """Classify crowd status based on configured thresholds (no hysteresis)."""
        # Density-based override (when area is known and thresholds are set)
        if self.area_sqm > 0 and count > 0:
            density = count / self.area_sqm
            if self.density_threshold_congested > 0 and density > self.density_threshold_congested:
                return "CONGESTED"
            if self.density_threshold_moderate > 0 and density > self.density_threshold_moderate:
                return "MODERATE"

        # Count-based classification
        if count > self.count_threshold_congested:
            return "CONGESTED"
        elif count > self.count_threshold_moderate:
            return "MODERATE"
        return "NORMAL"

    def get_density(self, count: int) -> Optional[float]:
        """Calculate crowd density (people/sqm). Returns None if area is unknown."""
        if self.area_sqm > 0:
            return round(count / self.area_sqm, 3)
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# CROWD CLASSIFIER WITH HYSTERESIS
# ═══════════════════════════════════════════════════════════════════════════════

class CrowdClassifier:
    """
    Stabilised crowd status classifier with configurable hysteresis.

    Prevents rapid status oscillation caused by minor count fluctuations
    while still allowing genuine severe-crowd warnings through quickly.

    Parameters:
      escalation_frames   – consecutive readings needed to move UP
      deescalation_frames – consecutive readings needed to move DOWN
      (direct jump to CONGESTED from NORMAL uses half the escalation count)
    """

    STATUS_SEVERITY = {"NORMAL": 0, "MODERATE": 1, "CONGESTED": 2}

    def __init__(self, escalation_frames: int = 3, deescalation_frames: int = 5):
        self.escalation_frames = max(1, escalation_frames)
        self.deescalation_frames = max(1, deescalation_frames)
        self._confirmed_status: str = "NORMAL"
        self._consecutive_status: str = "NORMAL"
        self._consecutive_count: int = 0

    def update(self, count: int, area_config: AreaConfig) -> str:
        """
        Feed a new count reading. Returns the stable (hysteresis-filtered) status.
        """
        raw_status = area_config.classify_crowd(count)

        # Same as confirmed → reset streak, return immediately
        if raw_status == self._confirmed_status:
            self._consecutive_count = 0
            self._consecutive_status = raw_status
            return self._confirmed_status

        # Accumulate consecutive readings of the new status
        if raw_status == self._consecutive_status:
            self._consecutive_count += 1
        else:
            self._consecutive_status = raw_status
            self._consecutive_count = 1

        # Determine required consecutive count
        current_sev = self.STATUS_SEVERITY.get(raw_status, 0)
        confirmed_sev = self.STATUS_SEVERITY.get(self._confirmed_status, 0)
        is_escalation = current_sev > confirmed_sev

        if is_escalation:
            required = self.escalation_frames
            # Fast-track CONGESTED from NORMAL (half frames) for urgency
            if raw_status == "CONGESTED" and self._confirmed_status == "NORMAL":
                required = max(1, self.escalation_frames // 2)
        else:
            required = self.deescalation_frames

        # Transition when threshold is met
        if self._consecutive_count >= required:
            self._confirmed_status = raw_status
            self._consecutive_count = 0

        return self._confirmed_status

    def reset(self):
        """Reset classifier state to NORMAL."""
        self._confirmed_status = "NORMAL"
        self._consecutive_count = 0
        self._consecutive_status = "NORMAL"

    @property
    def current_status(self) -> str:
        return self._confirmed_status


# ═══════════════════════════════════════════════════════════════════════════════
# DEFAULT TEMPLE AREAS (Illustrative)
# ═══════════════════════════════════════════════════════════════════════════════

DEFAULT_AREAS: Dict[str, AreaConfig] = {
    "Main Sanctum": AreaConfig(
        name="Main Sanctum",
        count_threshold_moderate=10,
        count_threshold_congested=20,
        area_sqm=50.0,
        density_threshold_moderate=0.3,
        density_threshold_congested=0.5,
        max_capacity=30,
    ),
    "Entrance Gate": AreaConfig(
        name="Entrance Gate",
        count_threshold_moderate=20,
        count_threshold_congested=40,
        area_sqm=100.0,
        density_threshold_moderate=0.25,
        density_threshold_congested=0.45,
        max_capacity=60,
    ),
    "Corridor": AreaConfig(
        name="Corridor",
        count_threshold_moderate=8,
        count_threshold_congested=15,
        area_sqm=30.0,
        density_threshold_moderate=0.35,
        density_threshold_congested=0.55,
        max_capacity=20,
    ),
    "Prasadam Hall": AreaConfig(
        name="Prasadam Hall",
        count_threshold_moderate=25,
        count_threshold_congested=50,
        area_sqm=120.0,
        density_threshold_moderate=0.25,
        density_threshold_congested=0.45,
        max_capacity=70,
    ),
    "Parking Area": AreaConfig(
        name="Parking Area",
        count_threshold_moderate=30,
        count_threshold_congested=60,
        area_sqm=200.0,
        density_threshold_moderate=0.2,
        density_threshold_congested=0.35,
        max_capacity=80,
    ),
}


# ═══════════════════════════════════════════════════════════════════════════════
# AREA MANAGER (Persistence)
# ═══════════════════════════════════════════════════════════════════════════════

class AreaManager:
    """Manages multiple temple area configurations with JSON persistence."""

    def __init__(self, config_file: str = CONFIG_FILE):
        self.config_file = config_file
        self.areas: Dict[str, AreaConfig] = {}
        self.load()

    def load(self):
        """Load area configs from JSON, or initialise with defaults."""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, "r") as f:
                    data = json.load(f)
                self.areas = {name: AreaConfig(**cfg) for name, cfg in data.items()}
                return
            except Exception:
                pass
        # First run → seed with illustrative defaults
        self.areas = {k: AreaConfig(**asdict(v)) for k, v in DEFAULT_AREAS.items()}
        self.save()

    def save(self):
        """Persist area configs to JSON."""
        out_dir = os.path.dirname(self.config_file)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(self.config_file, "w") as f:
            json.dump({n: asdict(c) for n, c in self.areas.items()}, f, indent=2)

    def get_area(self, name: str) -> Optional[AreaConfig]:
        """Get area config by name."""
        return self.areas.get(name)

    def add_area(self, config: AreaConfig) -> List[str]:
        """Add or update an area. Returns validation errors (empty = success)."""
        errors = config.validate()
        if not errors:
            self.areas[config.name] = config
            self.save()
        return errors

    def remove_area(self, name: str) -> bool:
        """Remove an area config. Returns True if removed."""
        if name in self.areas:
            del self.areas[name]
            self.save()
            return True
        return False

    def list_areas(self) -> List[str]:
        """List all configured area names."""
        return list(self.areas.keys())

    def get_default_area(self) -> AreaConfig:
        """Get the first area, or a generic fallback."""
        if self.areas:
            return next(iter(self.areas.values()))
        return AreaConfig(name="Default Area")
