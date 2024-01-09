import numpy as np
import matplotlib as plt
import math, sys, os, re
import nibabel as nib
from tqdm import tqdm
from natsort import natsorted


from tools.debug import Debug


debug=Debug()

def find_root_path():
    current_path = os.path.abspath(__file__)
    while os.path.basename(current_path) != "Connectome":
        current_path = os.path.dirname(current_path)
        if current_path == os.path.dirname(current_path):
            return None

    return current_path

class DataUtils(object):
    def __init__(self):
        self.ROOTPATH      = find_root_path()
        self.DATAPATH      = os.path.join(self.ROOTPATH,"Data")
        self.DEVPATH       = os.path.join(self.ROOTPATH,"Dev")
        self.DATARECONPATH = os.path.join(self.DATAPATH, "MRSI_reconstructed")
        debug.info("ROOTPATH set to",self.ROOTPATH)
        debug.info("DATAPATH set to",self.DATAPATH)
        debug.info("DEVPATH set to",self.DEVPATH)
        debug.info("DATARECONPATH set to",self.DATARECONPATH)
        self.list_nii()

    def list_unique_ids(self, file_types):
        unique_ids_info = {}
        for root, dirs, files in os.walk(self.DATARECONPATH):
            for file in files:
                if file.endswith('.nii'):
                    for file_type in file_types:
                        if file_type == "OrigRes":
                            match = re.search(r'OrigRes_[\w+]+_conc([A-Za-z0-9]{4})\.nii', file)
                            if match:
                                id = match.group(1)
                                if id not in unique_ids_info:
                                    unique_ids_info[id] = {ft: False for ft in file_types}
                                unique_ids_info[id][file_type] = True
                        else:
                            if file.startswith(file_type):
                                match = re.search(rf'{file_type}(\d+).nii', file)
                                if match:
                                    id = int(match.group(1))
                                    if id not in unique_ids_info:
                                        unique_ids_info[id] = {ft: False for ft in file_types}
                                    unique_ids_info[id][file_type] = True
                                    break
        # Sort the keys and return the information
        return {id: unique_ids_info[id] for id in natsorted(unique_ids_info)}


    def __list_unique_ids_orig(data):
        pattern = re.compile(r'OrigRes_[\w+]+_conc([A-Za-z0-9]{4})\.nii')
        unique_ids = set()

        for line in data.split('\n'):
            match = pattern.search(line)
            if match:
                unique_ids.add(match.group(1))

        return sorted(unique_ids)

    def list_nii(self):
        nii_files = []
        for root, dirs, files in os.walk(self.DATARECONPATH):
            for file in files:
                if file.endswith('.nii'):
                    nii_files.append(os.path.join(root, file))
        self.nii_files = list(set(nii_files))
        return self.nii_files


    def __load_orig(self,metabolic_str, fileid):
        # script_dir = os.path.dirname(os.path.realpath(__file__))
        script_dir = os.path.join(self.DATARECONPATH,"OrigRes")
        target_pattern = f"OrigRes_{metabolic_str}_conc{fileid}.nii"
        debug.info("Looking for",target_pattern,"in",script_dir)
        for root, dirs, files in os.walk(script_dir):
            for file in files:
                if file == target_pattern:
                    file_path = os.path.join(root, file)
                    return nib.load(file_path)
        return None

    def load_nii(self,file_type="Conc",fileid=1,metabolic_str=None,normalization=False,rawnii=False):
        if file_type=="OrigRes":
            if metabolic_str is None:
                debug.error("load nii: empty metabolic_str")
                return None
            data = self.__load_orig(metabolic_str, fileid)
            if data is None: 
                debug.error("load nii: pattern not found")
                return None
        else:
            filename_pattern = f"/{file_type}{fileid:04d}.nii"
            # Search for the file in the list
            # sys.exit()
            for file in self.nii_files:
                if filename_pattern in file:
                    try:
                        data = nib.load(file)
                        break
                    except Exception as e:pass
        imgdata,header = data.get_fdata(),data.header
        if normalization:
            imgdata=imgdata/imgdata.max()
            imgdata[imgdata<0.0001]=0
        if rawnii:
            return data
        else:
            return imgdata,header
    
    def load_nii_all(self,file_type="Conc",normalization=False):
        pattern = re.compile(f"/{file_type}(\\d+).nii")
        ids = []
        for filename in self.nii_files:
            match = pattern.search(filename)
            if match:
                id_str = match.group(1)
                if id_str not in ids:
                    ids.append(int(id_str))  # Convert ID to integer
        ids.sort()
        id=list(set(ids))
        __template,_ = self.load_nii(file_type,0)
        data, headers = np.zeros((len(ids),)+__template.shape),list()
        for i in tqdm(ids):
            __data,__header = self.load_nii(file_type,i,metabolic_str=None,normalization=normalization)
            data[i]=__data
            headers.append(__header)
        debug.success("Done")

        return data, headers

    def save_nii_file(self,file_type,fileid,tensor3D):
        nifti_img = nib.Nifti1Image(tensor3D.astype(np.float64), np.eye(4))
        dirpath   = os.path.join(self.DATARECONPATH,file_type)
        os.makedirs(dirpath,exist_ok=True)
        outpath   = os.path.join(dirpath,f"{file_type}{fileid:04d}.nii") 
        nifti_img.to_filename(outpath)
        debug.success("Saved NII to " + outpath)


    

if __name__=='__main__':
    u = DataUtils()
    u.load_nii_all("Conc")
    u.load_nii_all("Basic")
    u.load_nii_all("Qmask")





