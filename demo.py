from app.model import load_model
from app.ui import build_ui

try:
    import spaces
    gpu = spaces.GPU
except ImportError:
    def gpu(fn):
        return fn

demo = build_ui(load_model(), wrap=gpu)

if __name__ == "__main__":
    demo.launch()