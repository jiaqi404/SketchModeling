import gradio as gr

def greet(name, intensity):
    return "Hello, " + name + "!" * int(intensity)

demo = gr.Interface(
    title="Sketch to 3D",
    description="Import or draw your sketch then run our model",
    fn=greet,
    inputs=["text", "slider"],
    outputs=["text"],
)

demo.launch()