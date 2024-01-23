import torch
import matplotlib.pyplot as plt
from datasets import load_dataset
from torchvision.transforms import Compose
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from pathlib import Path
from torch.optim import Adam

from unet import Unet
from tools import Diffusion

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

# define image transformations (e.g. using torchvision)
transform = Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Lambda(lambda t: (t * 2) - 1)
])

# define function
def transforms(examples):
   examples["pixel_values"] = [transform(image.convert("L")) for image in examples["image"]]
   del examples["image"]

   return examples

# load dataset from the hub
dataset = load_dataset("fashion_mnist")
transformed_dataset = dataset.with_transform(transforms).remove_columns("label")

# create dataloader
dataloader = DataLoader(transformed_dataset["train"], batch_size=128, shuffle=True)

# init model
image_size = 28
channels = 1
device = "cuda" if torch.cuda.is_available() else "cpu"


model = Unet(
    dim=image_size,
    channels=channels,
    dim_mults=(1, 2, 4,)
)
model.to(device)

optimizer = Adam(model.parameters(), lr=1e-3)




results_folder = Path("./results")
results_folder.mkdir(exist_ok = True)

save_and_sample_every = 1000
epochs = 6
timesteps = 300

diffusion = Diffusion(noise_steps=timesteps)

# start train
for epoch in range(epochs):
    for step, batch in enumerate(dataloader):
      optimizer.zero_grad()

      batch_size = batch["pixel_values"].shape[0]
      batch = batch["pixel_values"].to(device)

      # Algorithm 1 line 3: sample t uniformally for every example in the batch
      t = torch.randint(0, timesteps, (batch_size,), device=device).long()

      loss = diffusion.p_losses(model, batch, t, loss_type="huber")

      if step % 100 == 0:
        print("Loss:", loss.item())

      loss.backward()
      optimizer.step()

      # save generated images
      if step != 0 and step % save_and_sample_every == 0:
        milestone = step // save_and_sample_every
        batches = num_to_groups(4, batch_size)
        all_images_list = list(map(lambda n: diffusion.sample(model, batch_size=n, channels=channels), batches))
        all_images = torch.cat(all_images_list, dim=0)
        all_images = (all_images + 1) * 0.5
        save_image(all_images, str(results_folder / f'sample-{milestone}.png'), nrow = 6)

torch.save(model.state_dict(), 'model_fashion.pth')
# 进行推理
# sample 64 images
samples = diffusion.sample(model, image_size=image_size, batch_size=64, channels=channels)
random_index = 6
plt.imshow(samples[-1][random_index].reshape(image_size, image_size, channels), cmap="gray")
plt.savefig('output_image.png')

