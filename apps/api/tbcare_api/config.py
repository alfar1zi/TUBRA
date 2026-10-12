import os


class Config:
    MAX_BODY_MB = int(os.getenv("MAX_BODY_MB", "5"))
    ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:5173,http://localhost:3000").split(",")
    DEMO_ENABLED = os.getenv("DEMO_ENABLED", "false").lower() == "true"
    TBCARE_ARTIFACTS = os.getenv("TBCARE_ARTIFACTS", "/app/artifacts")


config = Config()