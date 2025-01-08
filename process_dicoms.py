import boto3
import os
import pydicom
import pandas as pd
from tqdm import tqdm  # Import tqdm for progress bar
import os
import pydicom
from PIL import Image
import numpy as np

import pandas as pd
def get_image_path(dicom_path, out_folder):
    return os.path.join(out_folder, f"{os.path.basename(dicom_path)}_middle_slice_rgb.jpg")


# read all dicom files
# for those with opt modality which are image data, save middle slice in specified directory (if does not exist)
# also prepare meta_df
class DicomProcessor:
    def __init__(self, out_folder, valid_opt=None):
        self.s3 = boto3.client(
            "s3",
            aws_secret_access_key="SzaxXKo59U7zLolqaSZ9ft0802JBXXZR/1hx1dyo",
            aws_access_key_id="GXMPQ5T9K9BJ80XKJ5OY",
            endpoint_url="https://s3-medver.med.uni-muenchen.de",
        )
        self.bucket_name = "scivias-eyeimages"
        self.output_folder = out_folder
        self.paginator = self.s3.get_paginator("list_objects_v2")
        self.page_iterator = self.paginator.paginate(Bucket=self.bucket_name)
        # self.valid_opt = pd.read_csv(valid_opt_path)
        self.use_valid_opt = False
        self.folder_path = '/data/core-kind1/Zeiss'
        self.valid_opt = valid_opt
        
    def check_valid(self, filename):
        return filename in self.valid_opt.Filename.values
        
    def extract_meta(self, dicom, dicom_path):
        
        try:
            ndim = dicom.pixel_array.ndim
        except:
            ndim = -1
        return {
            "Filename": os.path.basename(dicom_path),
            "Status": "Saved",
            "Modality": dicom.Modality,
            "SOPClassUID": dicom.SOPClassUID,
            "Laterality": dicom.get("Laterality", ""),
            "PatientName": dicom.get("PatientName", ""),
            "has_pixel_array": hasattr(dicom, "pixel_array"),
            "ndim": ndim
        }

    def process_single_dicom_old(self, dicom_path):
        """
        Process a single DICOM file to extract and save the middle slice as an RGB JPEG image.
        Returns metadata about the DICOM file.

        Parameters:
            dicom_path (str): Path to the DICOM file.
            output_folder (str): Path to the folder where the JPEG image will be saved.

        Returns:
            dict: Metadata about the DICOM file, including save status and file information.
        """
        output_file_path = get_image_path(dicom_path, self.output_folder)
        if os.path.isfile(output_file_path):
            return None, False
        # Read the DICOM file
        dicom = pydicom.dcmread(dicom_path)
        meta = self.extract_meta(dicom, dicom_path)
        
        # if os.path.isfile(output_file_path):
        #     return meta

        # Check modality
        if dicom.Modality != "OPT":
            return meta, False

        # Check for pixel array
        if not hasattr(dicom, "pixel_array") or dicom.pixel_array.ndim != 3:
            return meta, False
        # Extract the middle slice
        pixel_array = dicom.pixel_array
        middle_index = pixel_array.shape[0] // 2
        middle_slice = pixel_array[middle_index]

        # Normalize pixel values to 0-255
        middle_slice = (
            (middle_slice - np.min(middle_slice))
            / (np.max(middle_slice) - np.min(middle_slice))
            * 255
        ).astype(np.uint8)

        # Convert to RGB
        rgb_slice = Image.fromarray(middle_slice).convert("RGB")

        # Create output folder if it doesn't exist
        os.makedirs(self.output_folder, exist_ok=True)

        # Save as JPEG

        rgb_slice.save(output_file_path, format="JPEG")

        # Return metadata with success status
        return meta, True

    def process_single_dicom(self, dicom_path):
        """
        Process a single DICOM file to extract and save an RGB JPEG image based on its modality.
        Returns metadata about the DICOM file.

        Parameters:
            dicom_path (str): Path to the DICOM file.

        Returns:
            dict: Metadata about the DICOM file, including save status and file information.
        """
        output_file_path = get_image_path(dicom_path, self.output_folder)
        if os.path.isfile(output_file_path):
            return None, False

        # Read the DICOM file
        dicom = pydicom.dcmread(dicom_path)
        meta = self.extract_meta(dicom, dicom_path)

        # Check modality
        if dicom.Modality not in ["OPT", "OP"]:
            return meta, False

        # Check for pixel array
        if not hasattr(dicom, "pixel_array"):
            return meta, False

        pixel_array = dicom.pixel_array

        # Handle modality-specific logic
        if dicom.Modality == "OPT":
            if pixel_array.ndim != 3:
                return meta, False
            middle_index = pixel_array.shape[0] // 2
            slice_to_process = pixel_array[middle_index]
        elif dicom.Modality == "OP":
            if pixel_array.ndim != 2:
                return meta, False
            slice_to_process = pixel_array

        # Normalize pixel values to 0-255
        slice_to_process = (
            (slice_to_process - np.min(slice_to_process))
            / (np.max(slice_to_process) - np.min(slice_to_process))
            * 255
        ).astype(np.uint8)

        # Convert to RGB
        rgb_slice = Image.fromarray(slice_to_process).convert("RGB")

        # Create output folder if it doesn't exist
        os.makedirs(self.output_folder, exist_ok=True)

        # Save as JPEG
        rgb_slice.save(output_file_path, format="JPEG")

        # Return metadata with success status
        return meta, True


    def download_file(self, key):
        file_path = f"/tmp/{key.split('/')[-1]}"  # Save to temporary directory
        self.s3.download_file(self.bucket_name, key, file_path)
        return file_path

    def process_all_old(self):
        print('process with df')
        meta_list = {'valid': [], 'invalid': [], 'all': []}
        
        for i, page in enumerate(self.page_iterator):
            # meta_list['invalid'] = []
            if "Contents" in page:
                for obj in tqdm(page["Contents"], desc=f"Processing DICOM files of page {i}"):
                    key = obj["Key"]
                    if key.lower().endswith(".dcm"):
                        if (not self.check_valid(os.path.basename(key))) and self.use_valid_opt:
                            continue
                        file_path = self.download_file(key)
                        meta, saved = self.process_single_dicom(file_path)
                        meta['page'] = i
                        meta_list['all'].append(meta)
                        if saved:
                            meta_list['valid'].append(meta)
                        else:
                            meta_list['invalid'].append(meta)
                    else:
                        print(f'not dcm key: {key}')
            print(len(meta_list['valid']), len(meta_list['invalid']))
            pd.DataFrame(meta_list['valid']).to_csv(os.path.join(self.output_folder, 'meta.csv'))
            pd.DataFrame(meta_list['invalid']).to_csv(os.path.join(self.output_folder, 'others.csv'))
            pd.DataFrame(meta_list['all']).to_csv(os.path.join(self.output_folder, 'all.csv'))
            
    
    def process_all(self):
        print('process zeiss, valid opt')
        meta_list = {'valid': [], 'invalid': [], 'all': []}
        # os.listdir(self.folder_path)
        for filename in tqdm(self.valid_opt.Filename):
        
            if filename.lower().endswith(".dcm"):
                
                meta, saved = self.process_single_dicom(os.path.join(self.folder_path, filename))
                
                meta_list['all'].append(meta)
                if saved:
                    meta_list['valid'].append(meta)
                else:
                    meta_list['invalid'].append(meta)
            else:
                print(f'not dcm key: {key}')
            # print(len(meta_list['valid']), len(meta_list['invalid']))
            pd.DataFrame(meta_list['valid']).to_csv(os.path.join(self.output_folder, 'meta.csv'))
            pd.DataFrame(meta_list['invalid']).to_csv(os.path.join(self.output_folder, 'others.csv'))
            pd.DataFrame(meta_list['all']).to_csv(os.path.join(self.output_folder, 'all.csv'))