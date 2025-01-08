import boto3
import os
import pydicom
import pandas as pd
from tqdm import tqdm  # Import tqdm for progress bar
import os
import pydicom
from PIL import Image
import numpy as np

s3 = boto3.client(
    "s3",
    aws_secret_access_key="SzaxXKo59U7zLolqaSZ9ft0802JBXXZR/1hx1dyo",
    aws_access_key_id="GXMPQ5T9K9BJ80XKJ5OY",
    endpoint_url="https://s3-medver.med.uni-muenchen.de",
)


def is_dicom_file(key):
    return key.lower().endswith(".dcm")


def extract_patient_data(metadata):
    patient_info = "".join(metadata["PatientName"])
    patient_info = patient_info.split("^")
    metadata["patient_id"] = patient_info[0]
    metadata["doctor_id"] = patient_info[1] if len(patient_info) > 1 else ""
    metadata["record_name"] = (
        f"{metadata['patient_id']}-{metadata.get('Laterality', '')}"
    )
    return metadata


def extract_metadata_from_dicom(file):
    dicom = pydicom.dcmread(file)
    metadata = {
        "Modality": dicom.Modality,
        "SOPClassUID": dicom.SOPClassUID,
        "Laterality": dicom.get("Laterality", ""),
        "PatientName": dicom.get("PatientName", ""),
    }
    # Extract patient data
    metadata = extract_patient_data(metadata)
    return metadata


def download_file(bucket, key):
    file_path = f"/tmp/{key.split('/')[-1]}"  # Save to temporary directory
    s3.download_file(bucket, key, file_path)
    return file_path


def analyze_dicom_files(bucket):
    all_objects = []
    modality_count = {}
    sop_uid_count = {}
    eye_samples = {"Left": 0, "Right": 0}
    patient_ids = set()

    paginator = s3.get_paginator("list_objects_v2")
    page_iterator = paginator.paginate(Bucket=bucket)

    total_objects = 0
    for page in page_iterator:
        if "Contents" in page:
            total_objects += len(page["Contents"])

    print(f"Total number of objects: {total_objects}")

    # Iterate through the objects with a progress bar
    for page in page_iterator:
        if "Contents" in page:
            for obj in tqdm(
                page["Contents"], total=total_objects, desc="Processing DICOM files"
            ):
                key = obj["Key"]
                if is_dicom_file(key):
                    all_objects.append(key)
                    file_path = download_file(bucket, key)
                    metadata = extract_metadata_from_dicom(file_path)

                    # Update counts
                    patient_ids.add(metadata["patient_id"])
                    if "L" in metadata["Laterality"]:
                        eye_samples["Left"] += 1
                    elif "R" in metadata["Laterality"]:
                        eye_samples["Right"] += 1

                    # Update modality counts
                    modality = metadata["Modality"]
                    sop_uid = metadata["SOPClassUID"]

                    modality_count[modality] = modality_count.get(modality, 0) + 1
                    if modality not in sop_uid_count:
                        sop_uid_count[modality] = {}
                    sop_uid_count[modality][sop_uid] = (
                        sop_uid_count[modality].get(sop_uid, 0) + 1
                    )

    return {
        "Total Samples": len(all_objects),
        "Eye Samples": eye_samples,
        "Modality Count": modality_count,
        "SOPClassUID Count": sop_uid_count,
        "Total Patients": len(patient_ids),
    }


def main():
    # Example usage
    bucket_name = "scivias-eyeimages"
    results = analyze_dicom_files(bucket_name)

    # Print results
    print(f"Total Samples: {results['Total Samples']}")
    print(f"Left Eye Samples: {results['Eye Samples']['Left']}")
    print(f"Right Eye Samples: {results['Eye Samples']['Right']}")
    print("Modality Count:")
    for modality, count in results["Modality Count"].items():
        print(f"  {modality}: {count}")
    print("SOPClassUID Count:")
    for modality, sop_counts in results["SOPClassUID Count"].items():
        print(f"  {modality}:")
        for sop_uid, count in sop_counts.items():
            print(f"    {sop_uid}: {count}")
    print(f"Total Patients: {results['Total Patients']}")


def save_middle_slice_as_rgb_jpeg(dicom_path, output_folder):
    """
    Process a DICOM file to extract the middle slice if the modality is OPT and it contains pixel data,
    then save it as an RGB JPEG image in the specified folder.

    Parameters:
        dicom_path (str): Path to the DICOM file.
        output_folder (str): Path to the folder where the JPEG image will be saved.

    Returns:
        str: Path to the saved JPEG image or None if conditions are not met.
    """
    try:
        # Read the DICOM file
        dicom_file = pydicom.dcmread(dicom_path)

        # Check modality
        if dicom_file.Modality != "OPT":
            # print(f"File {dicom_path} is not of OPT modality.")
            return None

        # Check for pixel array
        if not hasattr(dicom_file, "pixel_array"):
            # print(f"File {dicom_path} does not contain pixel data.")
            return None

        # Extract pixel array
        pixel_array = dicom_file.pixel_array

        # Check if pixel data is 3D
        if pixel_array.ndim != 3:
            # print(f"File {dicom_path} does not contain 3D pixel data.")
            return None

        # Get the middle slice
        middle_index = pixel_array.shape[0] // 2
        middle_slice = pixel_array[middle_index]

        # Normalize the pixel values to 0-255 for saving as JPEG
        middle_slice = (
            (middle_slice - np.min(middle_slice))
            / (np.max(middle_slice) - np.min(middle_slice))
            * 255
        ).astype(np.uint8)

        # Convert grayscale to RGB
        rgb_slice = Image.fromarray(middle_slice).convert("RGB")

        # Create output folder if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)

        # Save the image as JPEG
        output_file_path = os.path.join(
            output_folder, f"{os.path.basename(dicom_path)}_middle_slice_rgb.jpg"
        )
        rgb_slice.save(output_file_path, format="JPEG")

        print(f"Middle slice saved as RGB to {output_file_path}")
        return output_file_path

    except Exception as e:
        print(f"An error occurred: {e}")
        return None


import os
import pydicom
from PIL import Image
import numpy as np


def process_single_dicom(dicom_path, output_folder):
    """
    Process a single DICOM file to extract and save the middle slice as an RGB JPEG image.
    Returns metadata about the DICOM file.

    Parameters:
        dicom_path (str): Path to the DICOM file.
        output_folder (str): Path to the folder where the JPEG image will be saved.

    Returns:
        dict: Metadata about the DICOM file, including save status and file information.
    """
    try:
        # Read the DICOM file
        dicom = pydicom.dcmread(dicom_path)

        # Check modality
        if dicom.Modality != "OPT":
            return None

        # Check for pixel array
        if not hasattr(dicom, "pixel_array") or dicom.pixel_array.ndim != 3:
            return None
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
        os.makedirs(output_folder, exist_ok=True)

        # Save as JPEG
        output_file_path = os.path.join(
            output_folder, f"{os.path.basename(dicom_path)}_middle_slice_rgb.jpg"
        )
        rgb_slice.save(output_file_path, format="JPEG")

        # Return metadata with success status
        return {
            "Filename": os.path.basename(dicom_path),
            "Status": "Saved",
            "OutputPath": output_file_path,
            "Modality": dicom.Modality,
            "SOPClassUID": dicom.SOPClassUID,
            "Laterality": dicom.get("Laterality", ""),
            "PatientName": dicom.get("PatientName", ""),
        }

    except Exception:
        # Return metadata with error status
        return None
