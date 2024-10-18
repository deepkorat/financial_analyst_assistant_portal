# app/__init__.py
from flask import Flask

# Initialize Flask app
def create_app():
    app = Flask(__name__, static_folder='../static')

    # Register routes from routes.py
    from .routes import main as main_blueprint
    app.register_blueprint(main_blueprint)

    return app
