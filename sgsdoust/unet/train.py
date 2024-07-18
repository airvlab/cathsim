import torch
import torch.nn as nn

import numpy as np

from model_mobile_unet import MobileUNet 
from torch.utils.data import DataLoader

from pathlib import Path
from guide3d.dataset.segment import Guide3D

from tqdm import tqdm


ROOT_PATH = Path("/users/sgsdoust/sharedscratch/cathsim/sgsdoust/guide3d/data/annotations/guide3d")
model = MobileUNet().to("cuda")

ds = Guide3D(ROOT_PATH)
dl = DataLoader(ds, batch_size = 4, shuffle = True)

# Training hyperparameters.
TOTAL_EPOCHS = 50
LEARNING_RATE = 1e-4
SCHEDULER_TMAX = 15

def Train():
    loss_f = nn.CrossEntropyLoss().to("cuda")
    optimiser = torch.optim.SGD(model.parameters(), lr = LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser,T_max = SCHEDULER_TMAX)

    losses = np.array([])

    with torch.set_grad_enabled(True):
        torch.cuda.empty_cache()

        for curr_epoch in tqdm(range(1, TOTAL_EPOCHS + 1)):
            curr_epoch_losses = np.array([])
            optimiser.zero_grad()

            for imgs_batch in dl:
                imgs = np.stack((imgs_batch[0],) * 3, axis = -1).reshape(4, 3, 1024, 1024)
                imgs = torch.tensor(imgs, dtype = torch.float).to("cuda")
                segms = torch.tensor(imgs_batch[1], dtype = torch.long).to("cuda")

                Y_preds = model(imgs)
                loss = loss_f(Y_preds, segms)

                loss.backward()
                optimiser.step()
                scheduler.step()

                curr_epoch_losses = np.append(curr_epoch_losses, loss.cpu().detach())

            losses = np.append(losses, np.mean(curr_epoch_losses))

            # After each epoch run, save the model to the filesystem.
            torch.save(model, f"/users/sgsdoust/sharedscratch/trained_mobileunet_{curr_epoch}_epochs.pth")
            torch.save(model.state_dict(), f"/users/sgsdoust/sharedscratch/trained_mobileunet_sd_{curr_epoch}_epochs.pth")

    np.save("/users/sgsdoust/sharedscratch/losses.npy", losses)


if __name__ == "__main__":
    Train()
