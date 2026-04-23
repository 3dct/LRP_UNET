from monai.networks.nets import UNet, BasicUNet, SwinUNETR, UNETR, BasicUNetPlusPlus, AttentionUnet

class modelFactory:

    def __init__(self) -> None:

        self.unetFilter = (  32, 64, 128, 256, 512,  32)
        self.unetFilterRes = (  32, 64, 128, 256, 512)
        self.norm = "instance"
        self.act = "ReLu"
        self.stride = 2

    def getModel(self, Network:str):


        if Network == "BasicUnet":
                model = BasicUNet(
                    spatial_dims=2,
                    in_channels=1,
                    out_channels=1,
                    features= self.unetFilter,
                    act=self.act,
                    norm=self.norm
                )

        elif Network == "ResiduelUnet":
            model = UNet(spatial_dims=2,
                in_channels=1,
                out_channels=1,
                channels= self.unetFilterRes,
                strides=(self.stride, self.stride, self.stride, self.stride),
                num_res_units=2,
                kernel_size=3,
                up_kernel_size=3,
                norm=self.norm,
                act=self.act
                )
        elif Network == "UnetPlusPlus": 
            model = BasicUNetPlusPlus(spatial_dims=2,
            act = self.act,
            in_channels=1,
            out_channels=1,
            norm=self.norm)

        
        elif Network == "AttentionUnet":
            model = AttentionUnet(spatial_dims=2,
                in_channels=1,
                out_channels=1,
                channels= self.unetFilterRes,
                strides=(self.stride, self.stride, self.stride, self.stride),
                norm=self.norm,
            )
        elif Network == "Unetr":
            model =  UNETR(in_channels=1, out_channels=1, img_size=256, feature_size=32, norm_name='batch', spatial_dims=2)
        elif Network == "SwinUnetr":
            model = SwinUNETR(img_size=(256,256), in_channels=1, out_channels=1, spatial_dims=2)
        
        return model