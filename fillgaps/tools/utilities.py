import numpy as np
import matplotlib as plt
import math, sys, os, re
import nibabel as nib
from tqdm import tqdm


from fillgaps.tools.messages import MessagePrinter


msgprint=MessagePrinter()

def find_root_path():
    current_path = os.path.abspath(__file__)
    while os.path.basename(current_path) != "Connectonome":
        current_path = os.path.dirname(current_path)
        if current_path == os.path.dirname(current_path):
            return None

    return current_path

class Utilities(object):
    def __init__(self):
        self.ROOTPATH = find_root_path()
        self.DATAPATH = os.path.join(self.ROOTPATH,"Data")
        self.DEVPATH  = os.path.join(self.ROOTPATH,"Code")

    def list_unique_ids(self,file_types):
        unique_ids_info = {}
        
        dirpath = os.path.join(self.DATAPATH, "MRSI_reconstructed")

        for root, dirs, files in os.walk(dirpath):
            for file in files:
                if file.endswith('.nii'):
                    for file_type in file_types:
                        if file.startswith(file_type):
                            match = re.search(rf'{file_type}(\d+).nii', file)
                            if match:
                                id = int(match.group(1))
                                if id not in unique_ids_info:
                                    unique_ids_info[id] = {ft: False for ft in file_types}
                                unique_ids_info[id][file_type] = True
                                break

        # Sort the keys and return the information
        return {id: unique_ids_info[id] for id in sorted(unique_ids_info)}


    def list_nii(self):
        nii_files = []
        dirpath = os.path.join(self.DATAPATH,"MRSI_reconstructed")
        for root, dirs, files in os.walk(dirpath):
            for file in files:
                if file.endswith('.nii'):
                    nii_files.append(os.path.join(root, file))
        self.nii_files = nii_files
        return nii_files


    def load_nii(self,file_type="Conc",id=1):
        self.list_nii()
            # Construct the filename pattern
        filename_pattern = f"{file_type}{id:04d}.nii"
        # msgprint.info("Looking for " + filename_pattern)
        # Search for the file in the list
        for file in self.nii_files:
            if filename_pattern in file:
                # msgprint.success("Extracting data from "+file)
                data = nib.load(file)
                break
        return data.get_fdata(),data.header
    
    def load_nii_all(self,file_type="Conc"):
        msgprint.info("Extracting all" + file_type +" data from ")
        pattern = re.compile(f"{file_type}(\\d+).nii")
        self.list_nii()

        ids = []
        for filename in self.nii_files:
            match = pattern.search(filename)
            if match:
                id_str = match.group(1)
                ids.append(int(id_str))  # Convert ID to integer
        ids.sort()
        __template,_ = self.load_nii(file_type,0)
        data, headers = np.zeros((len(ids),)+__template.shape),list()
        for i in tqdm(ids):
            __data,__header = self.load_nii(file_type,i)
            data[i]=__data
            headers.append(__header)
        msgprint.success("Done")
        return data, headers

    def save_nii_file(self,file_type,fileid,tensor3D):
        nifti_img = nib.Nifti1Image(tensor3D.astype(np.float64), np.eye(4))
        dirpath   = os.path.join(self.DATAPATH,"MRSI_reconstructed",file_type)
        os.makedirs(dirpath,exist_ok=True)
        outpath   = os.path.join(dirpath,f"{file_type}{fileid:04d}.nii") 
        nifti_img.to_filename(outpath)
        msgprint.success("Saved to " + outpath)


    

if __name__=='__main__':
    u = Utilities()
    u.load_nii_all("Conc")
    u.load_nii_all("Basic")
    u.load_nii_all("Qmask")





