from diffusers import DiffusionPipeline

generator = DiffusionPipeline.from_pretrained("CompVis/ldm-text2im-large-256")

generator.to("cuda")

image = generator("An image of a dude in Picasso style").images[0]

image.save("image_of_squirrel_painting.png")
