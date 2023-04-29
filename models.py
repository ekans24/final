import torch
import os
import torch.nn.functional as F

def save_model(model, name: str = 'image_agent.pt'):
    # Get the path to the directory where the current script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Create the full path to the file where the model will be saved
    save_path = os.path.join(script_dir, name)

    # Save the model's state_dict using the torch.save function
    torch.save(model.state_dict(), save_path)

def load_model(name: str = 'image_agent.pt'):
    # Get the path to the directory where the current script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Create the full path to the file where the model is saved
    load_path = os.path.join(script_dir, name)

    # Instantiate the Detector class
    r = Detector()

    # Load the model's state_dict using the torch.load function with map_location set to 'cpu'
    r.load_state_dict(torch.load(load_path, map_location='cpu'))

    return r

class Detector(torch.nn.Module):

    class BlockUpConv(torch.nn.Module):
        
        ONE_VAL = 1
        TRUE_VAL = True
        FALSE_VAL = False
        NONE_VAL = None
        
        def __init__(self, input_channels, output_channels, stride=ONE_VAL, residual=True):
            super().__init__()
    
            self.residual = residual
            self.upsample = NONE_VAL
            if stride != ONE_VAL or input_channels != output_channels:
                self.upsample = torch.nn.Sequential(
                    torch.nn.ConvTranspose2d(input_channels, output_channels, kernel_size=1, stride=stride, output_padding=1,
                                             bias=FALSE_VAL),
                    torch.nn.BatchNorm2d(output_channels)
                )
    
            # Define the net containing the main operations for the BlockUpConv
            self.net = torch.nn.Sequential(
                torch.nn.ConvTranspose2d(input_channels, output_channels, kernel_size=3, padding=1, stride=1, bias=False),
                torch.nn.BatchNorm2d(output_channels),
                torch.nn.ReLU(),
                torch.nn.ConvTranspose2d(output_channels, output_channels, kernel_size=3, padding=1, stride=stride, output_padding=1,
                                         bias=False),
                torch.nn.BatchNorm2d(output_channels),
                torch.nn.ReLU(),
                torch.nn.ConvTranspose2d(output_channels, output_channels, kernel_size=3, padding=1, stride=1, bias=False),
                torch.nn.BatchNorm2d(output_channels),
                torch.nn.ReLU()
            )
    
        def forward(self, x):
            # Calculate the identity tensor if the residual flag is set and upsample is not None
            identity = x if not self.residual or self.upsample is None else self.upsample(x)
        
            # Pass the input tensor through the net, and add the identity tensor if the residual flag is set
            output = self.net(x) + identity if self.residual else self.net(x)
        
            return output
    
    
    class BlockConv(torch.nn.Module):
        
        ONE_VAL = 1
        TRUE_VAL = True
        FALSE_VAL = False
        NONE_VAL = None        
        
        def __init__(self, input_channels, output_channels, kernel_size=3, stride=1, residual=True):
            super().__init__()
    
            self.residual = residual
            self.downsample = None
            if stride != ONE_VAL or input_channels != output_channels:
                self.downsample = torch.nn.Sequential(
                    torch.nn.Conv2d(input_channels, output_channels, kernel_size=1, stride=stride, bias=False),
                    torch.nn.BatchNorm2d(output_channels)
                )
    
            # Define the net containing the main operations for the BlockConv
            TWO_VAL = 2
            self.net = torch.nn.Sequential(
                torch.nn.Conv2d(input_channels, output_channels, kernel_size=kernel_size, padding=(kernel_size // TWO_VAL), stride=1,
                                bias=False),
                torch.nn.BatchNorm2d(output_channels),
                torch.nn.ReLU(),
                torch.nn.Conv2d(output_channels, output_channels, kernel_size=kernel_size, padding=(kernel_size // TWO_VAL), stride=stride,
                                bias=False),
                torch.nn.BatchNorm2d(output_channels),
                torch.nn.ReLU(),
                torch.nn.Conv2d(output_channels, output_channels, kernel_size=kernel_size, padding=(kernel_size // TWO_VAL), stride=1,
                                bias=False),
                torch.nn.BatchNorm2d(output_channels),
                torch.nn.ReLU()
            )

        def forward(self, x):
            # Calculate the identity tensor if the residual flag is set and downsample is not None
            identity = x if not self.residual or self.downsample is None else self.downsample(x)
            
            # Pass the input tensor through the net, and add the identity tensor if the residual flag is set
            output = self.net(x) + identity if self.residual else self.net(x)
        
            return output


    def __init__(self, dim_layers=[32, 64, 128], input_channels=3, output_channels=2, input_normalization=True,
                 skip_connections=True, residual=False):
        
        ZERO_VAL = 0
        TWO_VAL = 2
        
        super().__init__()
    
        self.skip_connections = skip_connections
        initial_channels = dim_layers[ZERO_VAL]
    
        self.net_conv = torch.nn.ModuleList([torch.nn.Sequential(
            torch.nn.Conv2d(input_channels, initial_channels, kernel_size=7, padding=3, stride=2, bias=False),
            torch.nn.BatchNorm2d(initial_channels),
            torch.nn.ReLU()
        )])
    
        self.net_upconv = torch.nn.ModuleList([
            torch.nn.ConvTranspose2d(initial_channels * TWO_VAL if skip_connections else initial_channels, output_channels, kernel_size=7,
                                     padding=3, stride=2, output_padding=1)
        ])
    
        for k, current_layer in enumerate(dim_layers):
            # Add BlockConv layers
            self.net_conv.append(self.BlockConv(initial_channels, current_layer, stride=2, residual=residual))
    
            # Adjust the number of channels for skip connections and the last layer
            dim_layers = len(dim_layers)
            current_layer = current_layer * TWO_VAL if skip_connections and k != dim_layers - 1 else current_layer
    
            # Add BlockUpConv layers
            self.net_upconv.insert(0, self.BlockUpConv(current_layer, initial_channels, stride=2, residual=residual))
    
            initial_channels = current_layer
    
        # Add input normalization if required
        self.norm = torch.nn.BatchNorm2d(input_channels) if input_normalization else None


    def forward(self, x):
        # Intialize some things
        three = 3
        two = 2
        one = 1
        false_Val = False
        zero = 0
        
        # Normalize the input tensor if the norm attribute is not None
        x = self.norm(x) if self.norm is not None else x
    
        height, width = x.shape[two], x.shape[three]
    
        skip_connection = []
        
        for l in self.net_conv:
            x = l(x)
            skip_connection.append(x)
    
        skip_connection.pop(-one)
        skip = false_Val
    
        for l in self.net_upconv:
            # If skip is True and there are skip connections left, concatenate the tensors
            if skip and len(skip_connection) > zero:
                x = torch.cat([x, skip_connection.pop(-one)], one)
            x = l(x)
            
            #skip
            skip = self.skip_connections
    
        pred, boxes = x[:, zero, :height, :width], x[:, one, :height, :width]
    
        return pred, boxes


    def detect(self, image, max_pool_ks=7, min_score=0.2, max_det=15):
        # Pass the input image through the model to obtain heatmap and boxes
        heatmap, boxes = self(image[None])
    
        # Apply the sigmoid function to the heatmap to normalize the values between 0 and 1
        heatmap = torch.sigmoid(heatmap.squeeze(0).squeeze(0))
    
        # Remove the first dimension of the boxes tensor
        sizes = boxes.squeeze(0)
    
        # Extract the peaks from the heatmap using the provided function
        peaks = extract_peak(heatmap, max_pool_ks, min_score, max_det)
    
        # Create a list of tuples containing the peak information and size of the box
        # Each tuple has the format: (y, x, class_index, size)
        return [(peak[0], peak[1], peak[2], sizes[peak[2], peak[1]].item()) for peak in peaks]


# Define the extract_peak function, this is just taken from assignment solutions
def extract_peak(heatmap, max_pool_ks=7, min_score=-5, max_det=100):
    max_cls = F.max_pool2d(heatmap[None, None], kernel_size=max_pool_ks, padding=max_pool_ks // 2, stride=1)[0, 0]
    possible_det = heatmap - (max_cls > heatmap).float() * 1e5
    if max_det > possible_det.numel():
        max_det = possible_det.numel()
    score, loc = torch.topk(possible_det.view(-1), max_det)
    return [(float(s), int(l) % heatmap.size(1), int(l) // heatmap.size(1))
            for s, l in zip(score.cpu(), loc.cpu()) if s > min_score]            