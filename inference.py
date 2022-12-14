
from diffusers import DiffusionPipeline
generator = DiffusionPipeline.from_pretrained("unconditional_data")

generator.to("cuda")

image = generator().images[0]

image.save("generated_image.png")
