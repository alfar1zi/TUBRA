from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ScreenPayload:
    """Validated payload for POST /api/v1/screen."""
    age: int
    sex: str  # "male" | "female"
    cough_duration: str  # "2_4" | "4_8" | "8_plus"
    hemoptysis: Optional[bool]  # None = "don't know"
    fever: Optional[bool]
    night_sweats: Optional[bool]
    weight_loss: Optional[bool]
    prior_tb: Optional[bool]
    smoking: Optional[str]  # "never" | "former" | "current" | None

    @staticmethod
    def from_json(data: dict) -> "ScreenPayload":
        """Parse and validate JSON payload."""
        # Age
        age = data.get("age")
        if not isinstance(age, int) or age < 18 or age > 100:
            raise ValueError("age must be integer 18-100")

        # Sex
        sex = data.get("sex")
        if sex not in ("male", "female"):
            raise ValueError("sex must be 'male' or 'female'")

        # Cough duration
        cough_duration = data.get("cough_duration")
        if cough_duration not in ("2_4", "4_8", "8_plus"):
            raise ValueError("cough_duration must be '2_4', '4_8', or '8_plus'")

        # Optional boolean symptoms
        def parse_bool(key: str) -> Optional[bool]:
            v = data.get(key)
            if v is None:
                return None
            if isinstance(v, bool):
                return v
            if isinstance(v, str) and v.lower() in ("true", "false"):
                return v.lower() == "true"
            raise ValueError(f"{key} must be boolean or null")

        hemoptysis = parse_bool("hemoptysis")
        fever = parse_bool("fever")
        night_sweats = parse_bool("night_sweats")
        weight_loss = parse_bool("weight_loss")
        prior_tb = parse_bool("prior_tb")

        # Smoking
        smoking = data.get("smoking")
        if smoking is not None and smoking not in ("never", "former", "current"):
            raise ValueError("smoking must be 'never', 'former', 'current', or null")

        return ScreenPayload(
            age=age,
            sex=sex,
            cough_duration=cough_duration,
            hemoptysis=hemoptysis,
            fever=fever,
            night_sweats=night_sweats,
            weight_loss=weight_loss,
            prior_tb=prior_tb,
            smoking=smoking,
        )

    def to_clinical_record(self) -> dict:
        """Map the 9 UI fields onto the CODA TB clinical feature record.

        The record is consumed by `tbcare_api.inference.ensemble.ServingPreprocessor`,
        which mirrors `ml.src.features.clinical.ClinicalPreprocessor._extract_features`.
        Fields TBCare does not collect (height, weight, vitals, HIV, country) are left
        absent so the preprocessor imputes them with training-set medians / modes.
        Unknown booleans stay None so "don't know" is imputed, never guessed as "no".
        """
        smoking_yes: Optional[bool]
        if self.smoking in ("former", "current"):
            smoking_yes = True
        elif self.smoking == "never":
            smoking_yes = False
        else:
            smoking_yes = None

        return {
            "sex": self.sex,
            "age": self.age,
            "reported_cough_dur": self.to_cough_duration_value(),
            "hemoptysis": self.hemoptysis,
            "fever": self.fever,
            "night_sweats": self.night_sweats,
            "weight_loss": self.weight_loss,
            "tb_prior": self.prior_tb,
            "smoke_lweek": smoking_yes,
        }

    def to_cough_duration_value(self) -> float:
        """Map cough_duration to numeric value for model (midpoint in weeks)."""
        mapping = {"2_4": 3.0, "4_8": 6.0, "8_plus": 10.0}
        return mapping[self.cough_duration]

    def sex_value(self) -> float:
        """0.0 for female, 1.0 for male."""
        return 1.0 if self.sex == "male" else 0.0