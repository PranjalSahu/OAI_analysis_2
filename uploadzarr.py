from pathlib import Path
import os
import subprocess
import itk
import pandas as pd
import zarr
from numcodecs import Blosc
import json
import numpy as np
import multiscale_spatial_image as msi
from multiscale_spatial_image import to_multiscale
import s3fs

#oai_data_root = '/media/pranjal.sahu/moredata/OAI-DATASET/Package_1200013/results/18m/2.D.2'
oai_data_root = '/data/OAIFULLDATA'
dess_file = '/home/pranjal.sahu/OAI/OAI_analysis/data/SEG_3D_DESS_all.csv'

os.chdir(oai_data_root)

# months
time_points = [0, 12, 18, 24, 30, 36, 48, 72, 96]
time_point_folders = ['OAIBaselineImages',] + [f'OAI{m}MonthImages' for m in time_points[1:]]
time_point_patients = { tp: set() for tp in time_points }

print('time_point_folders are ')
print(time_point_folders)

dess_df = pd.read_csv(dess_file)

# We can use all images in dess_df in the future
#target_patients = [9993846, 9992358, 9986207]
target_patients = [9010060]

#output_prefix = Path('/data/OAI_analysis_2/ZARR')
output_prefix = 's3://oaisample1/ZARRDATA'


image_types = set()
data_index = {}
data_dtypes = np.dtype([('Months', np.uint16),
                    ('PatientID', np.unicode_, 64),
                    ('Laterality', np.unicode_, 8),
                    ('Path', np.unicode_, 1024),
                    ('CID', np.unicode_, 64),
                    ('Type', np.unicode_, 128),
                    ('Name', np.unicode_, 256)])

s3_path = 's3://oaisample1/zarr_example'
s3 = s3fs.S3FileSystem()
#store = s3fs.S3Map(root=s3_path, s3=s3, check=False)

data_rows = []
for patient in target_patients[:3]:
    for time_point_index, time_point in enumerate([0, 12, 18, 24, 30, 36, 48, 72, 96]):
        folder = Path(time_point_folders[time_point_index]) / 'results'
        #print('folder is ', folder)
        for study in folder.iterdir():
            #print('Study is ', study)
            if study.is_dir():
                for patient_dir in study.iterdir():
                    if patient_dir.match(str(patient)):
                        print('Match found ', str(patient), time_point)

                        acquisition_id = patient_dir.relative_to(folder)
                        acquisition_dess = dess_df['Folder'].str.contains(str(acquisition_id))
                        acquisition_df = dess_df.loc[acquisition_dess, :]
                        dess_count = 0

                        for _, descr in acquisition_df.iterrows():
                            is_left = descr['SeriesDescription'].find('LEFT') > -1
                            vol_folder = folder / descr['Folder']
                            image = itk.imread(str(vol_folder))
                            md = itk.MetaDataDictionary()
                            if is_left:
                                md['laterality'] = 'left'
                            else:
                                md['laterality'] = 'right'
                            image.SetMetaDataDictionary(md)
                            #patient_cid = subprocess.check_output(['ipfs', 'add', '--cid-version', '1', '--raw-leaves', '-Q'], input=str(patient).encode()).decode().strip()
                            patient_cid = str(patient)
                            month_id = f'Month-{time_point}'
                            patient_id = f'PatientID-{patient_cid}'
                            output_dir = output_prefix +'/' + patient_id  +'/' + month_id
                            
                            #if not output_dir.exists():
                            #    output_dir.mkdir(parents=True)
                            if not patient_id in data_index:
                                data_index[patient_id] = {}
                            if not month_id in data_index[patient_id]:
                                data_index[patient_id][month_id] = {}
                            
                            image_name = f'SAG_3D_DESS_{dess_count}.zarr'
                            image_suffix = patient_id  +'/' + month_id +'/'+ 'Images' +'/'+ image_name
                            output_image_dir = output_prefix +'/' + image_suffix

                            print('Pranjal path is ')
                            print(output_image_dir)
                            store = s3fs.S3Map(root=output_image_dir, s3=s3, check=False)
                            image_da = itk.xarray_from_image(image)
                            image_da.attrs['laterality'] = md['laterality']
                            
                            #name = output_image_dir.stem
                            name = 'image'
                            image_ds = image_da.to_dataset(name='image', promote_attrs=True)
                            
                            #print('image_ds is ')
                            #print(image_ds)

                            multiscale_image = msi.to_multiscale(image_ds.image, [2, 4], msi.Methods.ITK_GAUSSIAN)
                            # use attrs if the same image name needs to be used
                            
                            #print('multiscale_image is ')
                            #print(multiscale_image)
                            #store = zarr.DirectoryStore(output_image_dir)
                            chunk_size = 64
                            compressor = Blosc(cname='zstd', clevel=5, shuffle=Blosc.SHUFFLE)
                            multiscale_image.to_zarr(store, mode='w', compute=True,
                                            encoding={name: {'chunks': [chunk_size]*image.GetImageDimension(), 'compressor': compressor}})
                            #cid = subprocess.check_output(['ipfs', 'add', '-r', '--hidden', '-s', 'size-1000000',
                            #                                '--raw-leaves', '--cid-version', '1', '-Q',
                            #                                str(output_image_dir)]).decode().strip()
                            cid = 'cid'
                            data_index[patient_id][month_id][str(image_name)] = 'cid'
                            data_row = np.array([(time_point,
                                                patient_cid,
                                                str(md['laterality']),
                                                str(image_suffix),
                                                cid,
                                                'Image',
                                                f'SEG_3D_DESS_{dess_count}')],
                                                dtype=data_dtypes)
                            
                            print(data_row.shape)
                            print(data_row)
                            data_frame = pd.DataFrame(data_row, index=np.arange(1))
                            print('data_frame is')
                            print(data_frame)
                            
                            data_rows.append(data_frame)
                            dess_count += 1



# Test Code
# import zarr
# import xarray as xr
# import numpy as np

# Read as xarray
# p1 = xr.open_zarr('s3://oaisample1/ZARRDATA/PatientID-9010060/Month-12/Images/SAG_3D_DESS_1.zarr')
# print(p1)

# # Read a zarr file and check the laterality
# z2 = zarr.open('s3://oaisample1/ZARRDATA/PatientID-9010060/Month-12/Images/SAG_3D_DESS_1.zarr', mode='r')
# scale0 = z2['scale0']
# print(scale0.attrs['laterality'])

# # Get the image from zarr file
# image = z2['scale0/image']
# image = np.array(image)