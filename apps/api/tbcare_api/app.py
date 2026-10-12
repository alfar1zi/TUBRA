from flask import Flask, jsonify
from flask_cors import CORS

from .config import config
from .routes.health import health_bp
from .routes.echo import echo_bp
from .routes.screen import screen_bp


def create_app():
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = config.MAX_BODY_MB * 1024 * 1024

    CORS(app, origins=config.ALLOWED_ORIGINS, supports_credentials=False)

    app.register_blueprint(health_bp)
    app.register_blueprint(echo_bp)
    app.register_blueprint(screen_bp)

    @app.errorhandler(413)
    def too_large(e):
        return jsonify({"error": {"code": "too_large", "retryable": False}}), 413

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=8080, debug=True)