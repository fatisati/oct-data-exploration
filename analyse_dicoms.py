import os
import pydicom
import pandas as pd
from tqdm import tqdm

class DicomAnalyzer:
    def __init__(self, folder_path, cnt=None):
        self.folder_path = folder_path
        if cnt is None:
            cnt = len(list(os.listdir(folder_path)))
        self.cnt = cnt

    def extract_metadata(self, dicom):
        metadata = {
            "Filename": os.path.basename(dicom.filename),
            **{elem: getattr(dicom, elem, "Unknown") for elem in dicom.dir()}
        }
        return metadata

    def extract_patient_data(self, metadata):
        patient_info = "".join(metadata['PatientName'])
        patient_info = patient_info.split("^")
        metadata['patient_id'] = patient_info[0]
        metadata['doctor_id'] = patient_info[1] if len(patient_info) > 1 else ""
        metadata['record_name'] = f"{metadata['patient_id']}-{metadata.get('Laterality', '')}"
        return metadata

    def analyze_dicoms(self):
        all_data = []
        for filename in tqdm(os.listdir(self.folder_path)[:self.cnt]):
            if filename.endswith(".dcm"):
                dicom_path = os.path.join(self.folder_path, filename)
                dicom = pydicom.dcmread(dicom_path)
                
                # Extract metadata and patient data
                metadata = self.extract_metadata(dicom)
                patient_data = self.extract_patient_data(metadata)
                
                # Collect the necessary fields
                data = {
                    "Filename": patient_data["Filename"],
                    "PatientID": patient_data["patient_id"],
                    "SOPClassUID": patient_data.get("SOPClassUID", "Unknown"),
                    "Modality": patient_data.get("Modality", "Unknown"),
                    "Laterality": patient_data.get("Laterality", "Unknown")
                }
                all_data.append(data)
        
        # Convert to DataFrame
        df = pd.DataFrame(all_data)
        
        # Split based on SOPClassUID
        df_op = df[df.Modality == 'OP']
        opt = df[df.Modality == 'OPT']
        sop_uid = "1.2.840.10008.5.1.4.1.1.77.1.5.4"
        df_opt = opt[opt["SOPClassUID"] == sop_uid]
        df_other = opt[opt["SOPClassUID"] != sop_uid]
        
        # Save to CSV
        df.to_csv("all_dicoms.csv", index=False)
        df_opt.to_csv("opt_modality_dicoms.csv", index=False)
        df_other.to_csv("other_dicoms.csv", index=False)

        return df, df_op, df_opt, df_other

# Example usage
folder_path = "/data/core-kind1/Zeiss"
analyzer = DicomAnalyzer(folder_path)
df_all, df_op, df_opt, df_other = analyzer.analyze_dicoms()
print(len(df_op), len(df_opt), len(df_other))
print(df_other.groupby(['Modality', 'SOPClassUID']).size().reset_index(name='counts'))
print(df_all.Laterality.value_counts())
print(df_all.PatientID.nunique())
df_op.to_csv('op.csv')
df_opt.to_csv('opt.csv')