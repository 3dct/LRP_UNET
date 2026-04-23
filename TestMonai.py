from math import nan

from captum.attr import LRP
from captum.attr._utils.lrp_rules import EpsilonRule,IdentityRule, GammaRule, Alpha1_Beta0_Rule
from monai.networks.nets import UNet, BasicUNet, UNETR, AttentionUnet, BasicUNetPlusPlus, SwinUNETR
from monai.losses.dice import DiceLoss
import torch
import torch.nn as nn
import numpy
from monai.data import ArrayDataset, create_test_image_2d, decollate_batch
from matplotlib import pyplot as plt
import matplotlib 
matplotlib.use('Agg')
import SimpleITK as sitk
from tqdm import tqdm

import einops
import PIL
from scipy.ndimage import label

from glob import glob
import os

from models import modelFactory

normMode = "batch"
salt = "60_last"
channels = 1
size = 512

Outlier = 5
ThresholdValue = 1e1
sign= "positive"

height = size
width = size
offsetHeight = 0
offsetWidth = 40
outChannel = 0


def wrapper(inp):
    return net(inp).mean(dim=(2,3))

class Sum(nn.Module):
    def __init__(self, index=None, onlyMask=False):
        super().__init__()
        #self.seg =  torch.tensor(numpy.float32(segImage)).reshape(1,1,size,size)
        #self.loss = DiceLoss(sigmoid=True,reduction="none")
        self.index = index
        self.onlyMask = onlyMask
    #rule = EpsilonRule(0.0)

    def forward(self, x):
        bin = x.cpu().detach().numpy()[0][outChannel]> 0.5
        device = torch.device('cpu')
        

        component_array, num_components = label(bin, structure=numpy.ones((3,3)))
        

        test = numpy.unique(component_array,return_counts=True)[1]
        test = test.argsort(axis=0)
        BiggestValueIndex = test[-2] #-2
        mask = component_array == BiggestValueIndex
        ii = numpy.where(mask == 1)
        if self.index != None:
            mask = numpy.zeros(mask.shape)
            mask[ii[0][self.index],ii[1][self.index]] = 1
            device = torch.device('cuda')
        mask = torch.tensor(mask).to(device)
        mask2 = torch.ones([1,1,size,size])
        mask2[0][outChannel] = mask
        sumClasses = torch.sum(x.to(device)*mask2.to(device),dim=[2,3])
        if not self.onlyMask:
            return sumClasses
        else:
            return x.to(device)*mask2.to(device)
        self.seg.requires_grad_()
        dice = self.loss(x,self.seg)
        res = dice.mean(dim=[2,3])
        return res




def outputResults(input,nameFolder, net):
    input.requires_grad_()

    sig = nn.Sigmoid()

    net3Sum = Sum(onlyMask=True)

    result = net.forward(input)

    sigResult = sig(result)
    mask = net3Sum(sigResult)
    (sigResult*mask).sum().backward()



    plt.figure()
    plt.imshow(input.detach().numpy()[0].transpose((1,2,0)), interpolation='nearest', cmap="Greys_r")
    plt.colorbar()
    plt.title('Input')
    plt.savefig(nameFolder+"\\Input.png")
    #plt.show(block=False)
    plt.close()

    plt.figure()
    plt.imshow((input.grad).clip(min=0.01).numpy()[0][0], interpolation='nearest')
    plt.colorbar()
    plt.title('Input grad')
    plt.savefig(nameFolder+"\\Input_grad.png")
    #plt.show(block=False)
    plt.close()

    plt.figure()
    plt.imshow((input.grad).clip(min=0.01).log().numpy()[0][0], interpolation='nearest')
    plt.colorbar()
    plt.title('Input grad log')
    plt.savefig(nameFolder+"\\Input_grad_log.png")
    #plt.show(block=False)
    plt.close()

    plt.figure()
    plt.imshow(net3Sum(sig(result)).detach().numpy()[0][outChannel], interpolation='nearest')
    plt.colorbar()
    plt.title('Result')
    plt.savefig(nameFolder+"\\Result.png")
    #plt.show(block=False)
    plt.close()

    plt.figure()
    plt.imshow(sig(result).detach().numpy()[0][outChannel], interpolation='nearest')
    plt.colorbar()
    plt.title('Result sigmoid')
    plt.savefig(nameFolder+"\\Result_sigmoid.png")
    #plt.show(block=False)
    plt.close()


def loadImage(dir):
    img = PIL.Image.open(dir)
    im = numpy.array(img)[offsetHeight:height+offsetHeight,offsetWidth:width+offsetWidth]
    return im

def doLRP(input,nameFolder, net):
    sumRule = Sum()
    sumRule.rule = EpsilonRule()

    net2 = nn.Sequential(
    net,
    nn.Sigmoid(),
    sumRule
    )


    custom_rules = {
        nn.modules.Conv2d:EpsilonRule,
        nn.InstanceNorm2d:EpsilonRule,
        nn.LayerNorm:EpsilonRule,
        nn.modules.conv.ConvTranspose2d:EpsilonRule,
        nn.Identity:EpsilonRule,
        einops.layers.torch.Rearrange:EpsilonRule,
        torch.nn.modules.container.ModuleList:EpsilonRule
    }

    custom_nonLinear = [nn.PReLU, nn.Sigmoid, nn.LeakyReLU, nn.Softmax, nn.GELU]


    #im, seg = create_test_image_2d(size, size, num_seg_classes=1, noise_max=0.0,rad_min=22, rad_max=23, num_objs=24)
    #pred_label_idx = torch.zeros(1,1,224,224)
    #input = torch.tensor(numpy.float32(im)).permute(2, 0, 1).reshape(1,3,height,width)/255.0

    lrp = LRP(net2)

    lrp.CUSTOM_LAYERS_WITH_RULES = custom_rules
    lrp.CUSTOM_NON_LINEAR_LAYERS = custom_nonLinear


    attributions_lrp = lrp.attribute(input, 
                                    target=outChannel, verbose=False,)
    

    attributions_lrp_raw = attributions_lrp
    attributions_lrp_neg = -1*attributions_lrp.clip(max=-1/ThresholdValue).mul(-ThresholdValue).log().nan_to_num(0) 
    attributions_lrp = attributions_lrp.clip(min=1/ThresholdValue).mul(ThresholdValue).log().nan_to_num(0) 


    plt.figure()
    plt.imshow((attributions_lrp_neg)[0].detach().numpy()[0], interpolation='nearest')
    plt.colorbar()
    plt.title('attributions_lrp_neg')
    plt.savefig(nameFolder+"\\attributions_lrp_neg.png")
    #plt.show(block=False)
    plt.close()


    plt.figure()
    plt.imshow((attributions_lrp)[0].detach().numpy()[0], interpolation='nearest')
    plt.colorbar()
    plt.title('attributions_lrp_pos')
    plt.savefig(nameFolder+"\\attributions_lrp_pos.png")
    #plt.show(block=False)
    plt.close()

    plt.figure()
    plt.imshow((attributions_lrp_raw)[0].detach().numpy()[0], interpolation='nearest')
    plt.colorbar()
    plt.title('attributions_lrp_raw')
    plt.savefig(nameFolder+"\\attributions_lrp_raw.png")
    #plt.show(block=False)
    plt.close()
    #(attributions_lrp_neg+attributions_lrp)
    from captum.attr import visualization as viz
    fig,ax =viz.visualize_image_attr((attributions_lrp + attributions_lrp_neg)[0].permute(1,2,0).detach().numpy(),im.reshape(height,width,channels),outlier_perc=Outlier,sign=sign, method="blended_heat_map",alpha_overlay=0.55,  cmap="viridis")
    plt.close(fig)
    fig.frameon=False
    ax.axis('off')
    fig.savefig(nameFolder+"\\overlapping.png")
    plt.close(fig)
    from captum.attr import visualization as viz
    fig,ax =viz.visualize_image_attr((attributions_lrp_raw)[0].permute(1,2,0).detach().numpy(),im.reshape(height,width,channels),outlier_perc=Outlier,sign=sign, method="blended_heat_map",alpha_overlay=0.55,  cmap="viridis")
    plt.close(fig)
    fig.frameon=False
    ax.axis('off')
    fig.savefig(nameFolder+"\\overlapping_raw.png")
    plt.close(fig)
    #number = int(name.replace("best_metric_model_segmentation2d_array","0"))
    #fig.savefig(f".\\out_result3\\overlapping\\{Network}.png")
    #plt.close(fig)

def evaluate(im, dirModels, outDir):

    for Network in [ "ResiduelUnet","BasicUnet","UnetPlusPlus"]: 


        shape = im.shape
        if len(shape) == 2:
            input = torch.tensor(numpy.float32(im)).reshape(1,1,height,width)/im.max()
            channels=1
        else:
            input = torch.tensor(numpy.float32(im)).permute(2, 1, 0).unsqueeze(0)/im.max()
            channels=3


        models = sorted(glob(dirModels+r"\*.pth"))
        #models = sorted(glob(r".\steps_basicMul2\*.pth"))
        
        print(models)
        for model in tqdm(models):

            if "_"+Network+"_"+normMode+salt not in model: 
                continue

            factory = modelFactory()
            factory.norm = normMode
            net = factory.getModel(Network)
            net.eval()
            loadedModel = torch.load(model)
            net.load_state_dict(loadedModel)

            name = os.path.basename(model).replace(".pth","")
            nameFolder = ".\\"+outDir+"\\"+ name
            if not os.path.exists(nameFolder):
                os.makedirs(nameFolder)

            outputResults(input,nameFolder,net)
            doLRP(input,nameFolder,net)




if __name__ == "__main__":


    im = loadImage(r"C:\Users\P41877\Downloads\Montgomery\MontgomerySet\MCUCXR_0002_0_resized2.png")
    evaluate(im,"./cxr","cxr_lrp")

    # sumDings = None


    # def calcLRP(input, pixel):
    #     sumRule = Sum(pixel)
    #     sumRule.rule = EpsilonRule()

    #     net2 = nn.Sequential(
    #     net,
    #     nn.Sigmoid(),
    #     sumRule
    #     )

    #     cuda = torch.device('cuda')
    #     net2.to("cuda")
    #     input2 = input.to(device=cuda)

    #     lrp = LRP(net2)


    #     lrp.CUSTOM_LAYERS_WITH_RULES = custom_rules
    #     lrp.CUSTOM_NON_LINEAR_LAYERS = custom_nonLinear
    #     attributions_lrp = lrp.attribute(input2, 
    #                                     target=outChannel, verbose=False,)
        
    #     return attributions_lrp.detach()

    # for pixel in tqdm(range(1,2,1)):

    #     if sumDings == None:
    #         sumDings = calcLRP(input,pixel)
    #     else:
    #         sumDings = sumDings + calcLRP(input,pixel)
    

    # plt.figure()
    # plt.imshow((sumDings.cpu())[0].detach().numpy()[0], interpolation='nearest')
    # plt.colorbar()
    # plt.title('attributions_lrp_raw')
    # plt.savefig(nameFolder+"\\attributions_lrp_raw2.png")
    # #plt.show(block=False)
    # plt.close()




    # attributions_lrp_neg = -1*sumDings.cpu().clip(max=-1/ThresholdValue).mul(-ThresholdValue).log().nan_to_num(0) 
    # attributions_lrp = sumDings.cpu().clip(min=1/ThresholdValue).mul(ThresholdValue).log().nan_to_num(0) 

    # plt.figure()
    # plt.imshow((attributions_lrp_neg)[0].detach().numpy()[0], interpolation='nearest')
    # plt.colorbar()
    # plt.title('attributions_lrp_neg')
    # plt.savefig(nameFolder+"\\attributions_lrp_neg2.png")
    # #plt.show(block=False)
    # plt.close()


    # plt.figure()
    # plt.imshow((attributions_lrp)[0].detach().numpy()[0], interpolation='nearest')
    # plt.colorbar()
    # plt.title('attributions_lrp_pos')
    # plt.savefig(nameFolder+"\\attributions_lrp_pos2.png")
    # #plt.show(block=False)
    # plt.close()

    # fig,ax =viz.visualize_image_attr((attributions_lrp_neg+attributions_lrp)[0].permute(2,1,0).detach().numpy(),im.reshape(height,width,channels),outlier_perc=Outlier,sign=sign, method="blended_heat_map",alpha_overlay=0.55,  show_colorbar=True , title="overlapping", cmap="viridis")
    # plt.close(fig)
    # fig.savefig(nameFolder+"\\overlapping2.png")
    # plt.close(fig)
    # fig,ax =viz.visualize_image_attr((sumDings.cpu())[0].permute(2,1,0).detach().numpy(),im.reshape(height,width,channels),outlier_perc=Outlier,sign=sign, method="blended_heat_map",alpha_overlay=0.55,  show_colorbar=True , title="overlapping", cmap="viridis")
    # plt.close(fig)
    # fig.savefig(nameFolder+"\\overlapping2_raw.png")
    # plt.close(fig)
    # # #number = int(name.replace("best_metric_model_segmentation2d_array","0"))
    # # fig.savefig(f".\\out_result3\\overlapping\\{Network}.png")
    # # plt.close(fig)