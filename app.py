import gradio as gr
from gradio_litmodel3d import LitModel3D
import os

def run_button(input_img):
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
            run_btn = gr.Button("Run", variant="primary", visible=True)

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

    run_btn.click(
        run_button,
        inputs=input_img,
        outputs=output_3d,
    )

demo.launch()