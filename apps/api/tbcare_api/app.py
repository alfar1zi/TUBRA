import os

from flask import Flask, jsonify
from flask_cors import CORS

from .config import config
from .routes.health import health_bp
from .routes.echo import echo_bp
from .routes.screen import screen_bp
from .schemas.response import create_error_response


def create_app():
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = config.MAX_BODY_MB * 1024 * 1024

    CORS(app, origins=config.ALLOWED_ORIGINS, supports_credentials=False)

    app.register_blueprint(health_bp)
    app.register_blueprint(echo_bp)
    app.register_blueprint(screen_bp)

    @app.errorhandler(413)
    def too_large(e):
        return jsonify(create_error_response("too_large").to_json()), 413

    return app


if __name__ == "__main__":
    app = create_app()
    # Dev-only server entrypoint. Never enable the Werkzeug debugger outside
    # local development: its interactive console allows arbitrary code
    # execution for anyone who can reach the port.
    debug = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    host = os.getenv("FLASK_HOST", "127.0.0.1")
    app.run(host=host, port=8080, debug=debug)