import os
import ipdb

import h5py
import numpy as np

import tarfile
from lxml import etree

from fuel.datasets.hdf5 import H5PYDataset

data_path = os.environ['FUEL_DATA_PATH']
data_path = os.path.join(data_path,'handwriting/')

input_file = os.path.join(data_path,'lineStrokes-all.tar.gz')
file_name = "handwriting.hdf5"
hdf5_path = os.path.join(data_path, file_name)

h5file = h5py.File(hdf5_path, mode='w')

raw_data = tarfile.open(input_file)
num_files = sum([x.isreg() for x in raw_data.getmembers()])

features = h5file.create_dataset('features', (num_files,),
           dtype = h5py.special_dtype(vlen = np.dtype('int16')))

features_shapes = h5file.create_dataset('features_shapes', (num_files,2),
            dtype='int32')

features.dims.create_scale(features_shapes, 'shapes')
features.dims[0].attach_scale(features_shapes)

features_shape_labels = h5file.create_dataset(
    'features_shape_labels', (2,), dtype='S7')
features_shape_labels[...] = [
    'time_step'.encode('utf8'),
    'feature_type'.encode('utf8')]
features.dims.create_scale(
    features_shape_labels, 'shape_labels')
features.dims[0].attach_scale(features_shape_labels)


transcript_files = []
idx = 0
for member in raw_data.getmembers():
    if member.isreg():
        transcript_files.append(member.name)
        content = raw_data.extractfile(member)
        tree = etree.parse(content)
        root = tree.getroot()
        content.close()

        points = []
        for StrokeSet in root:
            for i, Stroke in enumerate(StrokeSet):
                for Point in Stroke:
                    points.append([i,
                        int(Point.attrib['x']),
                        int(Point.attrib['y'])])
        points = np.array(points)
        points[:, 2] = -points[:, 2]
        change_stroke = points[:-1,0] != points[1:,0]
        pen_up = points[:,0]*0
        pen_up[change_stroke]=1
        pen_up[-1] = 1
        points[:,0] = pen_up
        features[idx] = points.flatten()
        features_shapes[idx] = np.array(points.shape)
        idx += 1

features.dims[0].label = 'batch'

#print len(all_results)


#train_set = H5PYDataset(hdf5_path, which_sets=('all',), sources=('features',))

transcript_files = [x.split("/")[-1] for x in transcript_files]

import re
transcript_files = [re.sub('-[0-9][0-9].xml','.txt',x) for x in transcript_files]

import collections
counter=collections.Counter(transcript_files)

#######################
# TRANSCRIPTS
#######################

input_file = os.path.join(data_path,'ascii-all.tar.gz')

file_name = "handwriting.hdf5"
#hdf5_path = os.path.join(data_path, file_name)

#h5file = h5py.File(hdf5_path, mode='w')

raw_data = tarfile.open(input_file)
num_files = sum([x.isreg() for x in raw_data.getmembers()])

member = raw_data.getmembers()[10]

all_transcripts = []
for member in raw_data.getmembers():
    if member.isreg() and member.name.split("/")[-1] in transcript_files:
        fp = raw_data.extractfile(member)

        cleaned = [t.strip() for t in fp.readlines()
                   if t != '\r\n'
                   and t != '\n'
                   and t != '\r\n'
                   and t.strip() != '']

        # Try using CSR
        idx = [n for n, li in enumerate(cleaned) if li == "CSR:"][0]
        cleaned_sub = cleaned[idx + 1:]
        corrected_sub = []
        for li in cleaned_sub:
            # Handle edge case with %%%%% meaning new line?
            if "%" in li:
                #ipdb.set_trace()
                li2 = re.sub('\%\%+', '%', li).split("%")
                li2 = [l.strip() for l in li2]
                corrected_sub.extend(li2)
            else:
                corrected_sub.append(li)

        if counter[member.name.split("/")[-1]] != len(corrected_sub):
            #ipdb.set_trace()
            pass

        all_transcripts.extend(corrected_sub)

#Last file transcripts are almost garbage
all_transcripts[-1]= 'A move to stop'
all_transcripts.append('garbage')
all_transcripts.append('A move to stop')
all_transcripts.append('garbage')
all_transcripts.append('A move to stop')
all_transcripts.append('A move to stop')
all_transcripts.append('Marcus Luvki')
all_transcripts.append('Hallo Well')

all_chars = ([chr(ord('a') + i) for i in range(26)] +
             [chr(ord('A') + i) for i in range(26)] +
             [chr(ord('0') + i) for i in range(10)] +
             [',', '.', ' ', '"', '<UNK>', "'"])

code2char = dict(enumerate(all_chars))
char2code = {v: k for k, v in code2char.items()}
unk_char = '<UNK>'

transcripts = h5file.create_dataset('transcripts', (len(all_transcripts),),
           dtype = h5py.special_dtype(vlen = np.dtype('int16')))

ipdb.set_trace()

for i,transcript_text in enumerate(all_transcripts):
    transcripts[i] = np.array([char2code.get(x, char2code[unk_char]) for x in transcript_text])

split_dict = {
    'all': {'features': (0, len(all_transcripts)),
            'transcripts': (0, len(all_transcripts))}
    }

h5file.attrs['split'] = H5PYDataset.create_split_array(split_dict)

h5file.flush()
h5file.close()