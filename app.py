import gradio as gr
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
from transformers import ViTImageProcessor, ViTForImageClassification

# Constants from training
model_str = "google/vit-base-patch16-224-in21k"
id2label = {
    0: 'edna_krabappel', 1: 'bart_simpson', 2: 'snake_jailbird', 3: 'nelson_muntz',
    4: 'lionel_hutz', 5: 'principal_skinner', 6: 'apu_nahasapeemapetilon', 7: 'agnes_skinner',
    8: 'lenny_leonard', 9: 'ned_flanders', 10: 'chief_wiggum', 11: 'maggie_simpson',
    12: 'martin_prince', 13: 'patty_bouvier', 14: 'mayor_quimby', 15: 'sideshow_mel',
    16: 'abraham_grampa_simpson', 17: 'selma_bouvier', 18: 'carl_carlson', 19: 'ralph_wiggum',
    20: 'barney_gumble', 21: 'moe_szyslak', 22: 'homer_simpson', 23: 'sideshow_bob',
    24: 'lisa_simpson', 25: 'disco_stu', 26: 'fat_tony', 27: 'otto_mann',
    28: 'troy_mcclure', 29: 'charles_montgomery_burns', 30: 'cletus_spuckler', 31: 'marge_simpson',
    32: 'milhouse_van_houten', 33: 'professor_john_frink', 34: 'rainier_wolfcastle',
    35: 'comic_book_guy', 36: 'krusty_the_clown', 37: 'kent_brockman',
    38: 'miss_hoover', 39: 'groundskeeper_willie', 40: 'gil', 41: 'waylon_smithers'
}

# Load model processor info
processor = ViTImageProcessor.from_pretrained(model_str)
image_mean = processor.image_mean
image_std = processor.image_std
size = processor.size["height"]

# Define transform (same as validation)
transform = transforms.Compose([
    transforms.Resize((size, size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=image_mean, std=image_std)
])

# Load model
model = ViTForImageClassification.from_pretrained(model_str, num_labels=len(id2label))
model.load_state_dict(torch.load("simpsons_model.pt", map_location="cpu"))
model.eval()

# Classification function
def classify(image):
    if image is None:
        print("No image provided.")
        return {}, gr.update(value="", visible=False)
    
    image = image.convert("RGB")
    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(input_tensor)

    probs = F.softmax(outputs.logits, dim=1)[0]
    top3 = torch.topk(probs, k=3)

    confidences = {}
    for i in range(3):
        label = id2label[top3.indices[i].item()]
        score = float(top3.values[i].item())
        confidences[label] = score

    top_label = list(confidences.keys())[0]
    message = f"Most matching with **{top_label.replace('_', ' ').title()}**!"

    print(f'Returning: {confidences}')        
    return confidences, gr.update(value=message, visible=True)

# Gradio css styling
custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Freckle+Face&display=swap');

body, .gradio-container {
    background-color: black !important;
    color: #fada00 !important;
}

h1, .gr-header, .gr-title {
    font-family: 'Freckle Face', cursive !important;
    color: #fada00 !important;
}

.gr-image .upload-label {
    font-family: 'Freckle Face', cursive !important;
    color: #fada00 !important;
}

label, .output-class, .gr-label {
    color: #fada00 !important;
}

.gr-button {
    background-color: #fada00 !important;
    color: black !important;
    font-weight: bold;
}

.gr-button:hover {
    background-color: #ffe347 !important;
}

/* Change the 'share link' section text */
a.share-btn__copy-btn + span {
    color: #fada00 !important;
    font-family: 'Freckle Face', cursive !important;
    font-size: 1.1em !important;
}

a.share-btn__copy-btn + span::before {
    content: "Copy this link to share your results! ";
}
"""

# Gradio app
with gr.Blocks(css=custom_css) as demo:
    gr.Markdown("# Which Simpsons Character Are You?")
    gr.Markdown("Tip: If using webcam, be sure to **click the camera icon** to take a picture before submitting.")

    image_input = gr.Image(type="pil", sources=["upload", "webcam"], label="Upload or Take a Picture")
    output = gr.Label(num_top_classes=3)
    share_message = gr.Markdown(
        "<h2 style='color: #fada00; font-family: Freckle Face, cursive;'>Click the button below to copy a link to share your results!</h2>",
        visible=False
    )
    submit_btn = gr.Button("Submit")

    submit_btn.click(fn=classify, inputs=image_input, outputs=[output, share_message])

demo.launch(share=True, server_name="0.0.0.0", server_port=7860)
