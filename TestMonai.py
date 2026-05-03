from math import nan
import os
from monai.networks.nets import UNet, BasicUNet, UNETR, AttentionUnet, BasicUNetPlusPlus, SwinUNETR
from monai.losses.dice import DiceLoss
import torch
import torch.nn as nn
import numpy
import matplotlib.pyplot as plt
import matplotlib 
matplotlib.use('Agg')
import SimpleITK as sitk
from tqdm import tqdm
import einops
import PIL
from scipy.ndimage import label
from glob import glob

# Import required libraries and modules
from captum.attr import LRP, visualization as viz
from captum.attr._utils.lrp_rules import EpsilonRule
from monai.data import ArrayDataset, create_test_image_2d, decollate_batch
from models import modelFactory


# Constants configuration
normMode = "batch"
salt = "60_last"
channels = 1
size = 512

Outlier = 5
ThresholdValue = 1e1
sign = "positive"

height = size
width = size
offsetHeight = 0
offsetWidth = 40
outChannel = 0


class Sum(nn.Module):
    """Custom sum module for processing network outputs"""
    
    def __init__(self, index=None, onlyMask=False):
        super().__init__()
        self.index = index
        self.onlyMask = onlyMask
        self.device = torch.device('cpu')

    def forward(self, x):
        """Process input tensor and generate mask-based sums"""
        # Get binary segmentation from output
        bin = x.cpu().detach().numpy()[0][outChannel] > 0.5
        
        # Find connected components
        component_array, num_components = label(bin, structure=numpy.ones((3,3)))
        
        # Identify largest components and create mask
        test = numpy.unique(component_array, return_counts=True)[1]
        test_argsort = test.argsort(axis=0)
        biggest_value_index = test_argsort[-2]  # Use second largest component
        
        mask = component_array == biggest_value_index
        
        if self.index is not None:
            ii = numpy.where(mask == 1)
            mask = numpy.zeros(mask.shape)
            mask[ii[0][self.index], ii[1][self.index]] = 1
            
        mask = torch.tensor(mask).to(self.device)
        mask2 = torch.ones([1, 1, size, size])
        mask2[0][outChannel] = mask
        
        # Calculate sum based on configuration
        if not self.onlyMask:
            return torch.sum(x.to(self.device)*mask2.to(self.device), dim=[2,3])
        else:
            return x.to(self.device)*mask2.to(self.device)


def outputResults(input, nameFolder, net):
    """Generate and save visualization results for a given input and network"""
    # Ensure input is ready for backpropagation
    input.requires_grad_()
    
    # Initialize wrapper networks and modules
    sig = nn.Sigmoid()
    net3Sum = Sum(onlyMask=True)
    
    # Forward pass through network
    result = net.forward(input)
    sigResult = sig(result)
    
    # Calculate gradients using sum rule
    mask = net3Sum(sigResult)
    (sigResult * mask).sum().backward()
    
    # Create and save visualization plots
    def save_input_plot():
        plt.figure()
        plt.imshow(input.detach().numpy()[0].transpose((1,2,0)), 
                   interpolation='nearest', cmap="Greys_r")
        plt.colorbar()
        plt.title('Input')
        plt.savefig(nameFolder+"\\Input.png")
        plt.close()

    def save_grad_plot(grad_title):
        plt.figure()
        plt.imshow((input.grad).clip(min=0.01).numpy()[0][0], 
                   interpolation='nearest')
        plt.colorbar()
        plt.title(f'Input grad {grad_title}')
        plt.savefig(nameFolder+f"\\Input_grad_{grad_title}.png")
        plt.close()

    def save_result_plot(title, result):
        plt.figure()
        plt.imshow(result.detach().numpy()[0][outChannel], 
                   interpolation='nearest')
        plt.colorbar()
        plt.title(title)
        plt.savefig(nameFolder+f"\\{title}.png")
        plt.close()

    # Generate and save plots
    save_input_plot()
    save_grad_plot('')
    save_grad_plot('log')
    save_result_plot('Result', net3Sum(sig(result)))
    save_result_plot('Result sigmoid', sig(result))
def loadImage(dir):
    img = PIL.Image.open(dir)
    im = numpy.array(img)[offsetHeight:height+offsetHeight,offsetWidth:width+offsetWidth]
    return im

def doLRP(input, nameFolder, net):
    """Perform LRP analysis and generate visualization plots"""
    # Initialize custom sum rule for LRP
    sumRule = Sum()
    sumRule.rule = EpsilonRule()

    # Create network with sigmoid and sum module
    net2 = nn.Sequential(net, nn.Sigmoid(), sumRule)
    
    # Define custom rules for different layers
    custom_rules = {
        nn.modules.Conv2d: EpsilonRule,
        nn.InstanceNorm2d: EpsilonRule,
        nn.LayerNorm: EpsilonRule,
        nn.modules.conv.ConvTranspose2d: EpsilonRule,
        nn.Identity: EpsilonRule,
        einops.layers.torch.Rearrange: EpsilonRule,
        torch.nn.modules.container.ModuleList: EpsilonRule
    }
    
    custom_nonLinear = [nn.PReLU, nn.Sigmoid, nn.LeakyReLU, 
                        nn.Softmax, nn.GELU]
    
    # Perform LRP attribution
    lrp = LRP(net2)
    lrp.CUSTOM_LAYERS_WITH_RULES = custom_rules
    lrp.CUSTOM_NON_LINEAR_LAYERS = custom_nonLinear
    
    attributions_lrp = lrp.attribute(input, 
                                     target=outChannel, verbose=False)
    
    # Process and visualize attributions
    def process_and_save_attributions(attributions, suffix):
        plt.figure()
        plt.imshow((attributions[0].detach().numpy()[0]), 
                   interpolation='nearest')
        plt.colorbar()
        plt.title(f'Attributions {suffix}')
        plt.savefig(nameFolder+f"\\attributions_lrp_{suffix}.png")
        plt.close()

    process_and_save_attributions(attributions_lrp, 'raw')
    
    # Generate blended heatmaps
    fig, ax = viz.visualize_image_attr(
        (attributions_lrp + -1*attributions_lrp.clip(max=-1/ThresholdValue)
         .mul(-ThresholdValue).log().nan_to_num(0))[0].permute(1,2,0),
        input.reshape(height,width,channels), outlier_perc=Outlier,
        sign=sign, method="blended_heat_map", alpha_overlay=0.55,
        cmap="viridis"
    )
    fig.frameon = False
    ax.axis('off')
    fig.savefig(nameFolder+"\\overlapping.png")
    plt.close(fig)


def evaluate(im, dirModels, outDir):
    """Evaluate multiple models and generate results"""
    for Network in ["ResiduelUnet", "BasicUnet", "UnetPlusPlus"]:
        # Prepare input based on image dimensions
        if len(im.shape) == 2:
            input = torch.tensor(numpy.float32(im)).reshape(1, 1, height, width)/im.max()
            channels=1
        else:
            input = torch.tensor(numpy.float32(im)).permute(2, 1, 0).unsqueeze(0)/im.max()
            channels=3

        # Load and evaluate each model
        models = sorted(glob(dirModels + r"\*.pth"))
        
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

            outputResults(input, nameFolder, net)
            doLRP(input, nameFolder, net)


if __name__ == "__main__":
    # Load and evaluate images
    im = loadImage(r".\Montgomery\MontgomerySet\MCUCXR_0002_0_resized2.png")
    evaluate(im, "./cxr", "cxr_lrp")