import torch
import torch.nn
import numpy as np
import os
import os.path
import numpy as np
from scipy import ndimage

class BRATSDataset(torch.utils.data.Dataset):
    def __init__(self, directory, test_flag=False):
        '''
        directory is expected to contain some folder structure:
                  if some subfolder contains only files, all of these
                  files are assumed to have a name like
                  brats_train_001_XXX_123_w.nii.gz
                  where XXX is one of t1, t1ce, t2, flair, seg
                  we assume these five files belong to the same image
                  seg is supposed to contain the segmentation
        '''
        super().__init__()
        self.directory = os.path.expanduser(directory) # full absolute path
        # cd /mntcephfs/lab_data/wangcm/panyongjia/dataset/test/number
        self.test_flag=test_flag
        if test_flag:
            self.seqtypes = ['t1', 't1ce', 't2', 'flair']
        else:
            self.seqtypes = ['t1', 't1ce', 't2', 'flair', 'seg']

        self.seqtypes_set = set(self.seqtypes)
        self.database = []
        for root, dirs, files in os.walk(self.directory):
            # if there are no subdirs, we have data
            if not dirs:
                files.sort()
                datapoint = dict()
                # extract all files as channels
                for f in files:
                    seqtype = f.split('_')[3]
                    datapoint[seqtype] = os.path.join(root, f)
                # assert set(datapoint.keys()) == self.seqtypes_set, \
                #     f'datapoint is incomplete, keys are {datapoint.keys()}'
                self.database.append(datapoint)

    def __getitem__(self, x):
        out = []
        filedict = self.database[x]
        number=filedict['t1'].split('/')[6] 
        for seqtype in self.seqtypes:
            data = np.load(filedict[seqtype])
            data_max = np.max(data)
            data_min = np.min(data)
            data_leap = data_max - data_min
            if data_leap > 1e-6:
                data = (data - data_min) / (data_leap)
            else:
                data = np.zeros_like(data)
            
            if data.shape[-1] == 1:
                data = np.squeeze(data, axis=-1)
            out.append(torch.tensor(data))

        out = torch.stack(out)
        out_dict = {}
        if self.test_flag:
            path2 = filedict['t1'].replace('t1', 'seg')
            seg=np.load(path2)
            seg_max = np.max(seg)
            seg_min = np.min(seg)
            seg_leap = seg_max - seg_min
            if seg_leap > 1e-6:
                seg = (seg - seg_min) / (seg_leap)
            else:
                seg = np.zeros_like(seg)

            if seg.shape[-1] == 1:
                seg = np.squeeze(seg, axis=-1)
            image = torch.zeros(4, 256, 256)
            image[:, 8:-8, 8:-8] = out
            label = seg[None, ...]
            if seg.max() > 0:
                weak_label = 1
            else:
                weak_label = 0
            out_dict["y"]=weak_label
            out_dict["seg_path"]=path2
        else:
            image = torch.zeros(4,256,256)
            image[:,8:-8,8:-8]=out[:-1,...]		#pad to a size of (256,256)
            label = out[-1, ...][None, ...]
            if label.max()>0:
                weak_label=1
            else:
                weak_label=0
            out_dict["y"] = weak_label

        return (image, out_dict, weak_label, label, number)

    def __len__(self):
        return len(self.database)



