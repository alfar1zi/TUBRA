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

    def to_symptom_vector(self) -> tuple[list[float], list[int]]:
        """
        Convert to model input format.
        Returns: (symptom_values, availability_mask)
        Order: hemoptysis, fever, night_sweats, weight_loss, prior_tb,
               smoking_never, smoking_former, smoking_current, age_normalized
        """
        # Boolean symptoms: 1.0 = yes, 0.0 = no, 0.5 = unknown (availability=0)
        symptoms = []
        availability = []

        bool_fields = [
            self.hemoptysis,
            self.fever,
            self.night_sweats,
            self.weight_loss,
            self.prior_tb,
        ]
        for v in bool_fields:
            if v is None:
                symptoms.append(0.5)
                availability.append(0)
            else:
                symptoms.append(1.0 if v else 0.0)
                availability.append(1)

        # Smoking: one-hot (3 dims)
        smoking_map = {"never": 0, "former": 1, "current": 2}
        if self.smoking is None:
            symptoms.extend([1/3, 1/3, 1/3])
            availability.extend([0, 0, 0])
        else:
            onehot = [0.0, 0.0, 0.0]
            onehot[smoking_map[self.smoking]] = 1.0
            symptoms.extend(onehot)
            availability.extend([1, 1, 1])

        # Age normalized: (age - 18) / (100 - 18) -> [0, 1]
        age_norm = (self.age - 18) / 82.0
        symptoms.append(age_norm)
        availability.append(1)

        return symptoms, availability

    def to_cough_duration_value(self) -> float:
        """Map cough_duration to numeric value for model (midpoint in weeks)."""
        mapping = {"2_4": 3.0, "4_8": 6.0, "8_plus": 10.0}
        return mapping[self.cough_duration]

    def sex_value(self) -> float:
        """0.0 for female, 1.0 for male."""
        return 1.0 if self.sex == "male" else 0.0