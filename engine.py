import os
import numpy as np
import torch
import torch.optim as optim
import torchvision
from time import time
from torch.nn.functional import softmax
import matplotlib.pyplot as plt
from IPython.display import clear_output
from matplotlib import rcParams
rcParams['figure.figsize'] = (15,4)




"""
LOSSES
"""


def bce_loss(inputs: torch.Tensor, targets: torch.Tensor, reduction="mean"):

    loss = inputs - targets*inputs + torch.log(1+torch.exp(-inputs))
    
    if reduction=="mean":
        return loss.mean()
    if reduction=="sum":
        return loss.sum()

    return loss


def focal_loss(inputs: torch.Tensor, targets: torch.Tensor, eps = 1e-8, gamma = 2, reduction="mean"):

    p = torch.sigmoid(inputs)
    ce_loss = bce_loss(inputs, targets, reduction="none")
    p_t = p*targets + (1-p)*(1-targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if reduction=="mean":
        return loss.mean()
    if reduction=="sum":
        return loss.sum()

    return loss


"""
TRAINING
"""


def train(model, opt, loss_fn, epochs, data_tr, data_val, device, scheduler=None):
    train_loss = []
    val_loss = []

    for epoch in range(epochs):
        tic = time()
        print('* Epoch %d/%d' % (epoch+1, epochs))

        avg_loss = 0
        model.train()  # train mode
        for X_batch, Y_batch in data_tr:

            X_batch_t = torch.Tensor(X_batch).to(device=device)
            Y_batch_t = torch.Tensor(Y_batch).to(device=device)

            opt.zero_grad()
            Y_pred_logits = model(X_batch_t)
            loss = loss_fn(Y_pred_logits, Y_batch_t)
            loss.backward()    # backward-pass
            opt.step()    # update weights

            # calculate loss to show the user
            avg_loss += loss / len(data_tr)
            torch.cuda.empty_cache()

        if scheduler is not None:
            scheduler.step()   
            
        toc = time()
        print('loss: %f' % avg_loss)
        train_loss.append(avg_loss.to('cpu').detach())

        # show intermediate results
        model.eval()    # testing mode
        X_val, Y_val = next(iter(data_val))
        
        with torch.set_grad_enabled(False):

            X_val_t = torch.Tensor(X_val).to(device=device)
            Y_val_t = torch.Tensor(Y_val).to(device=device)

            Y_hat_logits = model(X_val_t)
            Y_hat_probas = 1/(1+torch.exp(-Y_hat_logits))

            loss = loss_fn(Y_hat_logits, Y_val_t)
            val_loss.append(loss.to('cpu').detach())

            Y_hat = (Y_hat_probas>0.5).to(dtype=torch.long).to('cpu')    # the predicted mask itself
            # Visualize tools
            clear_output(wait=True)
            n_img = min(data_val.batch_size, 6)
            for k in range(n_img):
                plt.subplot(2, n_img, k+1)
                plt.imshow(X_val[k].numpy()[0], cmap='gray')
                plt.title('Real')
                plt.axis('off')

                plt.subplot(2, n_img, k+1+n_img)
                plt.imshow(Y_hat[k].numpy()[0], cmap='gray')
                plt.title('Output')
                plt.axis('off')
            plt.suptitle('%d / %d - loss: %f' % (epoch+1, epochs, avg_loss))
            plt.show()
    
    torch.cuda.empty_cache()
    
    return train_loss, val_loss


"""
INFERENCE
"""

def inference_single_image(model, img, device, threshold):
    image_t = torch.from_numpy(img).to(dtype=torch.float32)/255
    image_t = torch.moveaxis(image_t, 2, 0)

    padding = torchvision.transforms.Pad((0, 20), 0)
    image_t = padding(image_t)

    with torch.set_grad_enabled(False):
        logits = model(image_t.unsqueeze(0).to(device=device))
        probas = 1/(1+torch.exp(-logits))
        pipe_mask_pred_rgb = (probas>threshold).to(dtype=torch.long).to('cpu').numpy()    # the predicted mask itself
        pipe_mask_pred_rgb *= 255
        pipe_mask_pred_rgb = np.uint8(pipe_mask_pred_rgb[0][:,20:1100,:])

    return pipe_mask_pred_rgb, probas[0][:,20:1100,:]