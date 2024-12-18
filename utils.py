import torch

def num_param(model):
    """ Returns number of trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def bbox(mask):
    """
    Get bounding box of a binary mask.
    Args:
        mask: binary mask of shape (batch_size, 3, H, W)
    Returns:
        tensor of the same size as mask, with bounding boxes filled with 1.0 for each sample.
    """
    batch_size, channels, H, W = mask.shape
    bbox_tensor = torch.zeros_like(mask)
    
    for i in range(batch_size):
        # Combine the 3 channels to calculate a common bounding box
        combined_mask = mask[i].sum(dim=0) > 0
        
        # Find the non-zero indices (bounding box coordinates)
        nonzero_indices = torch.nonzero(combined_mask, as_tuple=True)
        if nonzero_indices[0].numel() > 0:  # If there are non-zero pixels
            ymin, ymax = nonzero_indices[0].min().item(), nonzero_indices[0].max().item()
            xmin, xmax = nonzero_indices[1].min().item(), nonzero_indices[1].max().item()
            # Fill the bounding box area with 1.0 for all 3 channels
            bbox_tensor[i, :, ymin:ymax+1, xmin:xmax+1] = 1.0
    
    return bbox_tensor


def histogram_matching(image_1, image_2, image_1_mask, image_2_mask):
    """
    Perform histogram matching on two images.
    Args:
        image_1: source image (after generation)
        image_2: reference image
        image_1_mask: mask for source image
        image_2_mask: mask for reference image
        - each is torch tensor of shape (batch_size, 3, H, W)
    Returns: histogram matching loss
    """
    # loss function is L1 loss
    loss_fn = torch.nn.L1Loss()
    # first, convert to the range [0, 255]
    image_1 = (image_1 + 1.0) / 2.0 * 255.0
    image_2 = (image_2 + 1.0) / 2.0 * 255.0

    loss_list = []
    # for each sample in batch, perform histogram matching
    for batch_idx in range(image_1.shape[0]):
        roi_1 = image_1_mask[batch_idx].nonzero()
        roi_2 = image_2_mask[batch_idx].nonzero()
        if roi_1.shape[0] == 0 or roi_2.shape[0] == 0:
            # no loss if no mask (region not exist)
            loss_list.append(torch.tensor(0.0, device=image_1.device))
        for channel_idx in range(3):
            # calculate CDF for each channel
            hist_1 = torch.histc(
                image_1[batch_idx, channel_idx, roi_1[:, 0], roi_1[:, 1]],
                bins=256, min=0, max=256
            )
            hist_2 = torch.histc(
                image_2[batch_idx, channel_idx, roi_2[:, 0], roi_2[:, 1]],
                bins=256, min=0, max=256
            )
            cdf_1 = torch.cumsum(hist_1, dim=0)
            cdf_2 = torch.cumsum(hist_2, dim=0)
            cdf_1 = cdf_1 / cdf_1[-1]
            cdf_2 = cdf_2 / cdf_2[-1]
            # histogram matching
            matched_values = torch.zeros_like(cdf_1)
            for i in range(256):
                diff = torch.abs(cdf_2 - cdf_1[i])
                matched_values[i] = torch.argmin(diff)
            # apply matched values to the source ROI
            converted_pixels = matched_values[
                image_1[batch_idx, channel_idx, roi_1[:, 0], roi_1[:, 1]].long()
            ]
            original_pixels = image_1[batch_idx, channel_idx, roi_1[:, 0], roi_1[:, 1]]
            loss_list.append(
                loss_fn(converted_pixels, original_pixels)
                * roi_1.shape[0] / (image_1.shape[2] * image_1.shape[3])
            )
    return torch.mean(torch.stack(loss_list))