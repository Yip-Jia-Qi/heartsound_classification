from torch import nn
import torch


class WaveNetModel(nn.Module):
    def __init__(self,input_channels=1,classes=5,layers=10,blocks=5, dilation_channels=15,residual_channels=32,skip_channels=512,kernel_size=2,dtype=torch.FloatTensor,bias=False,fast=False):

        super(WaveNetModel, self).__init__()

        self.input_channels = input_channels
        self.classes = classes
        self.layers = layers
        self.blocks = blocks
        self.dilation_channels = dilation_channels
        self.residual_channels = residual_channels
        self.skip_channels = skip_channels
        self.kernel_size = kernel_size
        self.dtype = dtype
        self.fast = fast

        # build model
        receptive_field = 1
        init_dilation = 1

        self.dilations = []

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()

        # 1x1 convolution to create channels
        self.start_conv = nn.Conv1d(in_channels=self.input_channels,
                                    out_channels=residual_channels,
                                    kernel_size=1,
                                    bias=bias)

        for b in range(blocks):
            additional_scope = kernel_size - 1
            new_dilation = 1
            for i in range(layers):
                # dilations of this layer
                self.dilations.append((new_dilation, init_dilation))

                # dilated convolutions
                self.filter_convs.append(nn.Conv1d(in_channels=residual_channels,
                                                   out_channels=dilation_channels,
                                                   kernel_size=kernel_size,
                                                   bias=bias,
                                                   dilation=new_dilation))

                self.gate_convs.append(nn.Conv1d(in_channels=residual_channels,
                                                 out_channels=dilation_channels,
                                                 kernel_size=kernel_size,
                                                 bias=bias,
                                                 dilation=new_dilation))

                # 1x1 convolution for residual connection
                self.residual_convs.append(nn.Conv1d(in_channels=dilation_channels,
                                                     out_channels=residual_channels,
                                                     kernel_size=1,
                                                     bias=bias))

                # 1x1 convolution for skip connection
                self.skip_convs.append(nn.Conv1d(in_channels=dilation_channels,
                                                 out_channels=skip_channels,
                                                 kernel_size=1,
                                                 bias=bias))

                receptive_field += additional_scope
                additional_scope *= 2
                init_dilation = new_dilation
                new_dilation *= 2

        self.end_conv_1 = nn.Conv1d(in_channels=skip_channels,
                                  out_channels=skip_channels,
                                  kernel_size=1,
                                  bias=True)

        self.end_conv_2 = nn.Conv1d(in_channels=skip_channels,
                                    out_channels=classes,
                                    kernel_size=1,
                                    bias=True)

        # self.output_length = 2 ** (layers - 1)
        # self.output_size = output_length
        self.receptive_field = receptive_field
        # self.input_size = receptive_field + output_length - 1

    def forward(self, input, mode="normal"):
        if mode == "save":
            self.inputs = [None]* (self.blocks * self.layers)
            
        x = self.start_conv(input)
        skip = torch.zeros((x.shape[0],self.skip_channels,x.shape[2])).to(x.device)
        
        # WaveNet layers
        for i in range(self.blocks * self.layers):

            (dilation, init_dilation) = self.dilations[i]

            if mode == "save":
                self.inputs[i] = x[:,:,-(dilation*(self.kernel_size-1) + 1):]
            elif mode == "step":
                self.inputs[i] = torch.cat([self.inputs[i][:,:,1:], x], dim=2)
                x = self.inputs[i]

            # dilated convolution
            residual = x

            filter = self.filter_convs[i](x)
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](x)
            gate = torch.sigmoid(gate)
            x = filter * gate

            # parametrized skip connection
            s = self.skip_convs[i](x)
            # print(s.shape)
            # if skip is not 0:
            skip = skip[:, :, -s.size(2):]
            # print(skip.shape)
            skip = s + skip

            x = self.residual_convs[i](x)
            x = x + residual[:, :, dilation * (self.kernel_size - 1):]
        
        # print("last layers")
        # print(skip.shape)
        x = torch.relu(skip)
        # print(x.shape)
        x = torch.mean(x, dim=2).unsqueeze(-1)
        # print(x.shape)
        x = torch.relu(self.end_conv_1(x))
        # print(x.shape)

        x = self.end_conv_2(x).squeeze(-1)
        x = torch.nn.functional.log_softmax(x, dim=-1)
        return x

if __name__ == '__main__':
    model = WaveNetModel()
    print(model)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    n = count_parameters(model)

    print(n)