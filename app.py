import gradio as gr
from gradio_litmodel3d import LitModel3D
import os
from src.SketchToImage import sketch_to_image
from src.BackgroundRemove import background_remove
#from src.ImageToModel import image_to_model

def generate_model(input_img):
    return output_3d

def input_image(input_img):
    input_img.save("src/tmp/sketch.png")
    return

with gr.Blocks() as demo:
    gr.Markdown("""
        # SketchModeling: From Sketch to 3D Model

        **SketchModeling** is a method for 3D mesh reconstruction from a sketch.

        It has three steps:
        1. It generates image from sketch using stable diffusion and controlnet.
        2. It removes the background of the image using RMBG.
        3. It reconsturcted the 3D model of the image using LGM.

        On below, you can either upload a sketch image or draw the sketch yourself. Then press Run and wait for the model to be generated.
        """)
    with gr.Row(variant="panel"):
        with gr.Column():
            with gr.Row():
                input_img = gr.Image(
                    type="pil", label="Input Image", sources="upload", image_mode="RGBA"
                )
                generated_img = gr.Image(
                    type="pil", label="Gnerated Image", image_mode="RGBA", interactive=False
                )
                processed_img = gr.Image(
                    type="pil", label="Processed Image", image_mode="RGBA", interactive=False
                )
            with gr.Row():
                prompt = gr.Textbox(label="Pompt", interactive=True)
                controlnet_conditioning_scale = gr.Slider(
                    label="Controlnet Conditioning Scale",
                    minimum=0.5,
                    maximum=1.5,
                    value=0.85,
                    step=0.05,
                    interactive=True
                )
            with gr.Accordion('Advanced options', open=False):
                with gr.Row():
                    negative_prompt = gr.Textbox(label="Negative Prompt", value="low quality, black and white image", interactive=True)
                    add_prompt = gr.Textbox(label="Styles", value=", 3d rendered, shadeless, white background, intact and single object", interactive=True)
                    num_inference_steps = gr.Number(label="Inference Steps", value=50, interactive=True)
            run_btn = gr.Button("Run", variant="primary")

        with gr.Column():
            output_3d = LitModel3D(
                label="3D Model",
                visible=True,
                clear_color=[0.0, 0.0, 0.0, 0.0],
                tonemapping="aces",
                contrast=1.0,
                scale=1.0,
            )
            with gr.Column(visible=True, scale=1.0) as hdr_row:
                gr.Markdown("""## HDR Environment Map
                Select an HDR environment map to light the 3D model. You can also upload your own HDR environment maps.
                """)

                with gr.Row():
                    hdr_illumination_file = gr.File(
                        label="HDR Env Map", file_types=[".hdr"], file_count="single"
                    )
                    example_hdris = [
                        os.path.join("src/hdri", f)
                        for f in os.listdir("src/hdri")
                    ]
                    hdr_illumination_example = gr.Examples(
                        examples=example_hdris,
                        inputs=hdr_illumination_file,
                    )
                    hdr_illumination_file.change(
                        lambda x: gr.update(env_map=x.name if x is not None else None),
                        inputs=hdr_illumination_file,
                        outputs=[output_3d],
                    )

    run_btn.click(fn=input_image, inputs=[input_img]).success(
        fn=sketch_to_image,
        inputs=[input_img, prompt, negative_prompt, add_prompt, controlnet_conditioning_scale, num_inference_steps],
        outputs=[generated_img]
    ).success(
        fn=background_remove,
        inputs=[generated_img],
        outputs=[processed_img]
    ).success(
        fn=generate_model,
        inputs=[processed_img],
        outputs=[output_3d]
    )

demo.launch()