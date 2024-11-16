from transformers import pipeline

def background_remove(input_img):
    pipe = pipeline("image-segmentation", model="briaai/RMBG-1.4", trust_remote_code=True, device=0)
    output = pipe(input_img)

    output.save("src/tmp/image_nobg.png")
    
    return output
