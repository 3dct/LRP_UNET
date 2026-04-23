# Copyright 2020 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import sys
import tempfile
from glob import glob

import torch
from PIL import Image


import monai
from monai.data import ArrayDataset, create_test_image_2d, decollate_batch
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.transforms import (
    Activations,
    AsDiscrete,
    Compose,
    EnsureType,
)

from monai.transforms import (
    Activations,
    EnsureChannelFirstd,
    AsDiscrete,
    Compose,
    LoadImaged,
    Rotate90d,
    ScaleIntensityd,
    RandSpatialCropSamplesd,
    Resized,
    Flipd
)

from monai.data import (
    DataLoader,
    decollate_batch,
    list_data_collate
)

from cityscapesscripts.preparation  import createTrainIdLabelImgs

from tqdm import tqdm

import models
from models import modelFactory



saveAll = False
salt = "60"
numEpochs = 60

def train(tempdir_train,tempdir_label, model, Network,norm, out_dir):


    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    #print(os.path.join(tempdir_train, "*.png"))
    images = sorted(glob(tempdir_train + "*.png",recursive=True))
    segs = sorted(glob(tempdir_label+ "*.png",recursive=True))

    valNum = int(len(images)*0.2)

    train_files = [{"img": img, "seg": seg} for img, seg in zip(images[:-valNum], segs[:-valNum])]
    val_files = [{"img": img, "seg": seg} for img, seg in zip(images[-valNum:], segs[-valNum:])]

    # define transforms for image and segmentation
    train_transforms = Compose(
        [
            LoadImaged(keys=["img", "seg"], reader="PILReader"),
            EnsureChannelFirstd(keys=["img", "seg"]),
            Resized(keys=["img", "seg"],spatial_size=[512,623],),
            RandSpatialCropSamplesd(
                keys=["img", "seg"],  roi_size=[512, 512], random_size=False, num_samples=2
            ),
            ScaleIntensityd(keys=["img", "seg"]),
            

            Rotate90d(keys=["img", "seg"],k=-1),
            Flipd(keys=["img", "seg"], spatial_axis=1)
        ]
    )
    val_transforms = Compose(
        [
            LoadImaged(keys=["img", "seg"], reader="PILReader"),
            EnsureChannelFirstd(keys=["img", "seg"]),
            Resized(keys=["img", "seg"],spatial_size=[512,623]),
            ScaleIntensityd(keys=["img", "seg"]),
            Rotate90d(keys=["img", "seg"],k=-1),
            Flipd(keys=["img", "seg"], spatial_axis=1)

        ]
    )


    # define array dataset, data loader
    check_ds = monai.data.Dataset(data=train_files, transform=train_transforms)
    check_loader = DataLoader(check_ds, batch_size=4, num_workers=0, collate_fn=list_data_collate)
    check_data = monai.utils.misc.first(check_loader)
    print(check_data["img"].shape, check_data["seg"].shape)

    # create a training data loader
    train_ds = monai.data.CacheDataset(data=train_files, transform=train_transforms, num_workers=10, cache_rate=1.0)
    train_loader = DataLoader(train_ds,prefetch_factor=2, batch_size=2, shuffle=True, num_workers=2, pin_memory=torch.cuda.is_available(),persistent_workers=True)
    # create a validation data loader
    val_ds = monai.data.PersistentDataset(data=val_files, transform=val_transforms, cache_dir="./val")
    val_loader = DataLoader(val_ds,prefetch_factor=4, batch_size=1, num_workers=2, pin_memory=torch.cuda.is_available(),persistent_workers=True)
    dice_metric = DiceMetric(reduction="mean_batch", get_not_nans=False, include_background=False)
    post_trans = Compose([EnsureType(), Activations(softmax=True), AsDiscrete(threshold=0.5)])

    post_pred = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
    post_label = Compose([AsDiscrete( threshold=0.5)])


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



    loss_function = monai.losses.DiceLoss(sigmoid=True, include_background=False)
    optimizer = torch.optim.Adam(model.parameters(), 8e-4)
    scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lambda epoch: 1.0)
    scaler = torch.cuda.amp.GradScaler()

    val_interval = 3
    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = list()
    metric_values = list()

    model = model.to(device)

    for epoch in tqdm(range(numEpochs)):
        print("-" * 10)
        print(f"epoch {epoch }/{numEpochs}")
        model.train()
        epoch_loss = 0
        step = 0
        for batch_data in tqdm(train_loader):
            step += 1
            inputs, labels =  batch_data["img"].to(device), batch_data["seg"].to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = loss_function(outputs.float(), labels.float())
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        scheduler.step()
        learn_rate = scheduler.get_last_lr()
        print(f"epoch {epoch + 1} learn rate: {learn_rate[0]:.7f}")
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.7f}")

        if (epoch ) % val_interval == 0:

            model.eval()
            params = []
           
            with torch.no_grad():
                val_images = None
                val_labels = None
                val_outputs = None

                for val_data in tqdm(val_loader):
                    val_images, val_labels = val_data["img"].to(device), val_data["seg"].to(device)
                    roi_size = (512, 512)
                    sw_batch_size = 2

                    with torch.cuda.amp.autocast():
                        val_outputs = sliding_window_inference(val_images, roi_size, sw_batch_size, model)
                    val_outputs = [post_pred(i) for i in decollate_batch(val_outputs.float())]
                    val_labels = [post_label(i) for i in decollate_batch(val_labels.float())]
                    # compute metric for current iteration
                    max = val_labels[0].max()
                    dice_metric(y_pred=val_outputs, y=val_labels)
                # aggregate the final mean dice result
                metric = dice_metric.aggregate()

                print(metric)
                metric = metric.mean().item()
                # reset the status for next validation round
                dice_metric.reset()
                metric_values.append(metric)
                if metric > best_metric:
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(), out_dir+"/best_metric_model_segmentation2d_array_"+Network+"_"+norm+salt+".pth")
                    print("saved new best metric model")
                print(
                    "current epoch: {} current mean dice: {:.4f} best mean dice: {:.4f} at epoch {}".format(
                        epoch + 1, metric, best_metric, best_metric_epoch
                    )
                )
                torch.save(model.state_dict(), out_dir+"/best_metric_model_segmentation2d_array_"+Network+"_"+norm+salt+"_last.pth")

                if saveAll:
                    torch.save(model.state_dict(), out_dir+"/best_metric_model_segmentation2d_array"+"{:03d}".format(epoch)+".pth")

    print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")

def trainModel(tempdir_train,tempdir_label, Network_name,norm, out_dir):

    factory = modelFactory()
    factory.norm = norm

    model = factory.getModel(Network_name)
    train(tempdir_train,tempdir_label,model,Network_name,norm, out_dir)




if __name__ == "__main__":
    

    tempdir_train = r"C:\Users\P41877\Downloads\Montgomery\MontgomerySet\CXR_png\\"
    tempdir_label = r"C:\Users\P41877\Downloads\Montgomery\MontgomerySet\ManualMask\rightMask\\" 

    trainModel(tempdir_train,tempdir_label, "BasicUnet","instance",out_dir="cxr")
    trainModel(tempdir_train,tempdir_label, "ResiduelUnet","instance",out_dir="cxr")
    trainModel(tempdir_train,tempdir_label,  "UnetPlusPlus","instance",out_dir="cxr")
    trainModel(tempdir_train,tempdir_label,  "BasicUnet","batch",out_dir="cxr")
    trainModel(tempdir_train,tempdir_label,  "ResiduelUnet","batch",out_dir="cxr")
    trainModel(tempdir_train,tempdir_label, "UnetPlusPlus","batch",out_dir="cxr")


    # tempdir_train = r"C:\Repos\LRP\out\img"
    # tempdir_label = r"C:\Repos\LRP\out\seg" 

    # # trainModel(tempdir_train,tempdir_label,  "BasicUnet","instance",out_dir="ndt")
    # trainModel(tempdir_train,tempdir_label,  "ResiduelUnet","instance",out_dir="ndt")
    # # trainModel(tempdir_train,tempdir_label,  "UnetPlusPlus","instance",out_dir="ndt")
    # # trainModel(tempdir_train,tempdir_label,  "BasicUnet","batch",out_dir="ndt")
    # trainModel(tempdir_train,tempdir_label,  "ResiduelUnet","batch",out_dir="ndt")
    # # trainModel(tempdir_train,tempdir_label,  "UnetPlusPlus","batch",out_dir="ndt")




