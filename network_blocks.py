import torch

class ResidualBlock(torch.nn.Module):

    """Residual Block, conv->norm->relu->conv->norm->add """

    def __init__(self, d_input, d_output):
        super(ResidualBlock, self).__init__()
        self.sequential = torch.nn.Sequential(
            torch.nn.Conv2d(d_input, d_output, kernel_size=3, stride=1, padding=1, bias=False),
            torch.nn.InstanceNorm2d(d_output, affine=True),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(d_output, d_output, kernel_size=3, stride=1, padding=1, bias=False),
            torch.nn.InstanceNorm2d(d_output, affine=True)
        ) # using instance norm (normalization per channel & per image) instead of batchnorm.

    def forward(self, x):
        return x + self.sequential(x)

class Generator(torch.nn.Module):

    """
    Generator block consisted of downsampling, concatenation, residual blocks, and upsampling.
    """

    def __init__(self, repeat_num, conv_dim, n_channel):
        super(Generator, self).__init__()
        self.Branch_0 = self._branch_input(n_channel, conv_dim)
        self.Branch_1 = self._branch_input(n_channel, conv_dim)
        
        self.bottleneck = self._bottleneck_block(conv_dim * 2, repeat_num)
        
        self.branch_1 = self._output_branch(conv_dim, n_channel)
        self.branch_2 = self._output_branch(conv_dim, n_channel)

    def _branch_input(self, n_channel, conv_dim):
        """
        Two branch for reference and source image, respectively.
        conv->norm->relu->conv->norm->relu
        image size: downsample by 2
        channel: from input channel (default to 3) to conv_dim * 2
        """
        return torch.nn.Sequential(
            torch.nn.Conv2d(n_channel, conv_dim, kernel_size=7, stride=1, padding=3, bias=False),
            torch.nn.InstanceNorm2d(conv_dim, affine=True),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(conv_dim, conv_dim*2, kernel_size=4, stride=2, padding=1, bias=False),
            torch.nn.InstanceNorm2d(conv_dim*2, affine=True),
            torch.nn.ReLU(inplace=True),
        )

    def _bottleneck_block(self, hidden_dim, repeat_num):
        """
        Encoder-Decoder bottleneck and up-sampling.
        (downsampling)->(repeated residual blocks)->(up-sampling)
        Args:
            hidden_dim:
                hidden dimension of the one branch after _branch_input
                because two branches are concatenated, the input dimension of the main block is hidden_dim * 2
            repeat_num:
                number of residual blocks in the block
        Returns:
            Activation with number of channels = hidden_dim // 2
            and image size = 2 * input image size
        """
        hidden_dim *= 2
        layers = [
            torch.nn.Conv2d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1, bias=False),
            torch.nn.InstanceNorm2d(hidden_dim, affine=True),
            torch.nn.ReLU(inplace=True)
        ]

        layers += [ResidualBlock(hidden_dim, hidden_dim) for _ in range(repeat_num)]

        for _ in range(2):
            layers += [
                torch.nn.ConvTranspose2d(hidden_dim, hidden_dim // 2, kernel_size=4, stride=2, padding=1, bias=False),
                torch.nn.InstanceNorm2d(hidden_dim // 2, affine=True),
                torch.nn.ReLU(inplace=True)
            ]
            hidden_dim //= 2
        return torch.nn.Sequential(*layers)

    def _output_branch(self, curr_dim, n_channel):
        """
        Build output branch layers.
        conv->norm->relu->conv->norm->relu->conv->tanh
        tanh activation used for the very last layer, adopted from the literature.
        """
        return torch.nn.Sequential(
            torch.nn.Conv2d(curr_dim, curr_dim, kernel_size=3, stride=1, padding=1, bias=False),
            torch.nn.InstanceNorm2d(curr_dim, affine=True),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(curr_dim, curr_dim, kernel_size=3, stride=1, padding=1, bias=False),
            torch.nn.InstanceNorm2d(curr_dim, affine=True),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(curr_dim, n_channel, kernel_size=7, stride=1, padding=3, bias=False),
            torch.nn.Tanh()
        )

    def forward(self, src, ref):
        src = self.Branch_0(src)
        ref = self.Branch_1(ref)
        concat = torch.cat((src, ref), dim=1) # concat. along channel
        out = self.bottleneck(concat)
        out_src = self.branch_1(out)
        out_ref = self.branch_2(out)
        return out_src, out_ref


class Discriminator(torch.nn.Module):

    """
    Discriminator in PatchGAN style (1 for makeup, 1 for non-makeup)
    Initial convolution layer: n_channel -> conv_dim, downsample by 2
    Repeated blocks: channel doubled, downsample by 2
    Final layer: 1 channel output
    Final output size: 256 -> 128 -> 64 -> 32 -> 31 -> 30
    """

    def __init__(self, repeat_num, conv_dim, n_channel):
        super(Discriminator, self).__init__()
        layers = [
            torch.nn.utils.spectral_norm(
                torch.nn.Conv2d(
                    n_channel, conv_dim, kernel_size=4, stride=2, padding=1
                )
            ),
            torch.nn.LeakyReLU(0.01, inplace=True)
        ]
        for repeat_idx in range(repeat_num):
            layers += [
                torch.nn.utils.spectral_norm(
                    torch.nn.Conv2d(
                        conv_dim, conv_dim*2, kernel_size=4,
                        stride=(2 if repeat_idx < repeat_num - 1 else 1),
                        padding=1
                    ) # downsample by 2, channel doubled repeatedly (except for last)
                ),
                torch.nn.LeakyReLU(0.01, inplace=True)
            ]
            conv_dim *= 2
        layers += [
            torch.nn.utils.spectral_norm(
                torch.nn.Conv2d(conv_dim, 1, kernel_size=4, stride=1, padding=1)
            ) # final layer: output 1 channel
        ]
        self.block = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x).squeeze()