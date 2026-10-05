"""
Configuration management module.
Loads YAML configurations and provides typed access to system paths and settings.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional
import os
import yaml


# Base directory is the project root (parent of src)
BASE_DIR = Path(__file__).resolve().parent.parent


@dataclass
class AppConfig:
    name: str = "Adaptive Food Demand Forecasting & Waste Risk Intelligence"
    version: str = "1.0.0"
    institution: str = "Manipal Institute of Technology, Bengaluru"
    department: str = "School of Computer Engineering"
    course: str = "Machine Learning (CSE_3125) - V Semester Mini Project"
    authors: list = field(default_factory=lambda: [
        {"name": "K Sri Praneetha", "reg_no": "245805006"},
        {"name": "K Ruthika Reddy", "reg_no": "245805344"}
    ])
    supervisor: str = "Shreya Banerjee"
    academic_year: str = "September 2026"


@dataclass
class PathsConfig:
    raw_data_dir: Path = BASE_DIR / "data" / "raw"
    demo_data_dir: Path = BASE_DIR / "data" / "demo"
    processed_data_dir: Path = BASE_DIR / "data" / "processed"
    models_dir: Path = BASE_DIR / "models"
    database_path: Path = BASE_DIR / "data" / "feedback_history.db"

    def ensure_directories(self) -> None:
        """Ensure all required project directories exist."""
        for path in [self.raw_data_dir, self.demo_data_dir, self.processed_data_dir, self.models_dir]:
            path.mkdir(parents=True, exist_ok=True)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)


@dataclass
class WasteRiskConfig:
    default_buffer_percentage: float = 5.0
    min_buffer_percentage: float = 0.0
    max_buffer_percentage: float = 25.0
    low_waste_pct: float = 5.0
    medium_waste_pct: float = 15.0
    shortage_critical_pct: float = 10.0


@dataclass
class RetrainingConfig:
    rolling_window_days: int = 7
    error_threshold_multiplier: float = 1.25
    consecutive_days_trigger: int = 3


@dataclass
class DataConfig:
    train_test_split_ratio: float = 0.80
    demo_samples_per_center: int = 300
    num_centers: int = 5
    num_meals: int = 8
    random_seed: int = 42


@dataclass
class SystemConfig:
    app: AppConfig = field(default_factory=AppConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    data: DataConfig = field(default_factory=DataConfig)
    waste_risk: WasteRiskConfig = field(default_factory=WasteRiskConfig)
    retraining: RetrainingConfig = field(default_factory=RetrainingConfig)
    raw_config: Dict[str, Any] = field(default_factory=dict)


def load_config(config_path: Optional[Path] = None) -> SystemConfig:
    """Load configuration from config.yaml or return defaults."""
    if config_path is None:
        config_path = BASE_DIR / "config.yaml"

    config_data: Dict[str, Any] = {}
    if config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config_data = yaml.safe_load(f) or {}
        except Exception as e:
            print(f"Warning: Failed to load config from {config_path}: {e}")

    paths_cfg = PathsConfig(
        raw_data_dir=BASE_DIR / config_data.get("paths", {}).get("raw_data_dir", "data/raw"),
        demo_data_dir=BASE_DIR / config_data.get("paths", {}).get("demo_data_dir", "data/demo"),
        processed_data_dir=BASE_DIR / config_data.get("paths", {}).get("processed_data_dir", "data/processed"),
        models_dir=BASE_DIR / config_data.get("paths", {}).get("models_dir", "models"),
        database_path=BASE_DIR / config_data.get("paths", {}).get("database_path", "data/feedback_history.db"),
    )
    paths_cfg.ensure_directories()

    wr_data = config_data.get("waste_risk", {})
    thresh = wr_data.get("thresholds", {})
    wr_cfg = WasteRiskConfig(
        default_buffer_percentage=float(wr_data.get("default_buffer_percentage", 5.0)),
        min_buffer_percentage=float(wr_data.get("min_buffer_percentage", 0.0)),
        max_buffer_percentage=float(wr_data.get("max_buffer_percentage", 25.0)),
        low_waste_pct=float(thresh.get("low_waste_pct", 5.0)),
        medium_waste_pct=float(thresh.get("medium_waste_pct", 15.0)),
        shortage_critical_pct=float(thresh.get("shortage_critical_pct", 10.0)),
    )

    rt_data = config_data.get("retraining", {})
    rt_cfg = RetrainingConfig(
        rolling_window_days=int(rt_data.get("rolling_window_days", 7)),
        error_threshold_multiplier=float(rt_data.get("error_threshold_multiplier", 1.25)),
        consecutive_days_trigger=int(rt_data.get("consecutive_days_trigger", 3)),
    )

    d_data = config_data.get("data", {})
    d_cfg = DataConfig(
        train_test_split_ratio=float(d_data.get("train_test_split_ratio", 0.80)),
        demo_samples_per_center=int(d_data.get("demo_samples_per_center", 300)),
        num_centers=int(d_data.get("num_centers", 5)),
        num_meals=int(d_data.get("num_meals", 8)),
        random_seed=int(d_data.get("random_seed", 42)),
    )

    return SystemConfig(
        app=AppConfig(),
        paths=paths_cfg,
        data=d_cfg,
        waste_risk=wr_cfg,
        retraining=rt_cfg,
        raw_config=config_data,
    )


# Global instance
CONFIG = load_config()
