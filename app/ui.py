import gradio as gr
from app.model import predict

def build_ui(model):
    def classify(img):
        if img is None:
            return {}
        return predict(model, img)

    return gr.Interface(
        fn=classify,
        inputs=gr.Image(type="pil", label="Brain MRI slice"),
        outputs=gr.Label(num_top_classes=4, label="Prediction"),
        title="Brain MRI tumour classifier",
        description=(
            "Research demo only. NOT a medical device; do not use for diagnosis. "
            "Trained on one public Kaggle dataset. It can be confidently wrong on "
            "images unlike its training data, such as CT scans."
        ),
        flagging_mode="never",
    )