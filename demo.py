from app.model import load_model
from app.ui import build_ui

demo = build_ui(load_model())

if __name__ == "__main__":
    demo.launch()