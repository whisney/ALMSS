import SimpleITK as sitk
import numpy as np


def NiiDataRead(path, as_type=np.float32):
    nii = sitk.ReadImage(path)
    spacing = nii.GetSpacing()  # [x,y,z]
    volumn = sitk.GetArrayFromImage(nii)  # [z,y,x]
    origin = nii.GetOrigin()
    direction = nii.GetDirection()

    spacing_x = spacing[0]
    spacing_y = spacing[1]
    spacing_z = spacing[2]

    spacing_ = np.array([spacing_z, spacing_y, spacing_x])
    return volumn.astype(as_type), spacing_.astype(np.float32), origin, direction


def NiiDataWrite(save_path, volumn, spacing, origin, direction, as_type=np.float32):
    spacing = spacing.astype(np.float64)
    raw = sitk.GetImageFromArray(volumn[:, :, :].astype(as_type))
    spacing_ = (spacing[2], spacing[1], spacing[0])
    raw.SetSpacing(spacing_)
    raw.SetOrigin(origin)
    raw.SetDirection(direction)
    sitk.WriteImage(raw, save_path)
