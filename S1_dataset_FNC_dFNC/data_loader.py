import os
import random
from random import shuffle
import numpy as np
import torch
import nibabel as nib
from scipy.io import loadmat, savemat
from torch.utils import data

import pandas as pd
from scipy.signal import hilbert
from scipy.signal import butter, filtfilt
from infant_dataset import longitudinal_dataset, session_infomation_whole, dataset_info
from infant_dataset import match_session, fisher_z_transform


class General_dataset(data.Dataset):
	def __init__(self, mode, dataset_name, GSR=True):
		""" Initializes image paths and preprocessing module."""
		self.mode = mode
		self.dataset_name = dataset_name
		self.component_priors = 'None'
		self.gsr_signal = 'GSR' if GSR else 'noGSR'
		self.workdir = './dynamic_volumne'
		self.dataset = f'{self.workdir}/{self.dataset_name}'

		self.sublist = [None ]*100
		self.label = np.array([0]*len(self.sublist))
		self.timeserieslist = np.array([ f'{self.dataset}/{sub}.npy' for sub in self.sublist])
		self.TR_list = [None]*len(self.sublist)
		self.length = len(self.sublist)
		self.sub_index, self.begin_length, self.end_length  = None, None, None
		# self.sub_index, self.begin_length, self.end_length = self.index_for_list()
		# print('total', len(self.sublist), len(self.timeserieslist ), len(self.label))

	def index_for_list(self):
		if self.mode == 'train':
			begin_length = 0
			end_length = int(0.8*self.length*0.9)

		elif self.mode =='val':
			begin_length = int(0.8*self.length*0.9)
			end_length = int(0.8*self.length)

		elif self.mode == 'test':
			begin_length = int(0.8*self.length)
			end_length = int(self.length)

		elif self.mode == 'all':
			begin_length = 0
			end_length = len(self.sublist)

		index_path = f'{self.dataset}/sub_index.npy'
		if os.path.exists(index_path):
			sub_index = np.load(index_path)
		else:
			sub_index = np.arange(self.length)
			np.random.shuffle(sub_index)
			np.save(index_path, sub_index)
		return sub_index, begin_length, end_length
		
	def __getitem__(self, index):
		timeseries_path = self.timeserieslist[index]
		if timeseries_path.endwith('.npy'):
			timeseries = np.load(timeseries_path)
		elif timeseries_path.endwith('.mat'):
			timeseries = loadmat(timeseries_path)['data']
		elif timeseries_path.endwith('.nii'):
			timeseries = nib.load(timeseries_path).get_fdata().T

		if not isinstance(self.component_priors, str):
			timeseries = timeseries[self.component_priors, :]

		# connected areas
		adj = self.get_functional_connectome(timeseries, bin_top_percent=10)

		# row of phase represent relative phase from connected areas to a center area
		# fluctuation for areal oscillation
		phase, fluctuation = self.get_phase_fluctuation(timeseries, adj)

		phase = torch.from_numpy(phase.astype('float32'))
		frequency = torch.from_numpy(frequency.astype('int64'))
		target =  torch.from_numpy(self.label[index].astype('long'))

		return adj, phase, fluctuation, target
		
	def __len__(self):
		"""Returns the total number of font files."""
		return len(self.sublist)

	def butter_bandpass_filter(self, timeseries, lowcut, highcut, fs, order=5):
		nyquist = 0.5 * fs
		low = lowcut / nyquist
		high = highcut / nyquist
		b, a = butter(order, [low, high], btype='band')
		timeseries_filtered = filtfilt(b, a, timeseries)
		return timeseries_filtered
	
	def get_functional_connectome(self, timeseries, bin_top_percent=10):
		"""each row in timeseries represent the timeserie of an area"""
		functional_corr = np.corrcoef(timeseries)
		functional_corr  = fisher_z_transform(functional_corr )
		threshold = np.percentile(functional_corr, int(100-bin_top_percent))
		bin_connectome = (functional_corr>=threshold).astype('int')
		return bin_connectome

	def get_brain_modularity(self, timeseries, uptriangle=True):
		corr = np.corrcoef(timeseries)
		# np.fill_diagonal(corr, 0)
		corr = fisher_z_transform(corr)
		if uptriangle:
			corr = corr[np.triu_indices(corr.shape[0], k=1)]
		return corr

	def dynamic_corr(self, timeseries_list, TR_list, slide_window=45):
		
		# print('Calculate dynamic FNC in ', len(timeseries_list), ' timeseries shape:', timeseries_list[0].shape)

		TR_list = np.array(TR_list).astype('float')
		whole_dFNC = []
		for timeseries, TR in zip(timeseries_list, TR_list):
			slice_range = int(np.ceil(slide_window/TR))
			dynamic_matrix = []
			for i in range(len(timeseries)-slice_range+1): 
				window_data = timeseries[i:i+slice_range,:]
				corr = np.corrcoef(window_data.T)
				dynamic_matrix.append(corr)
			dynamic_matrix = np.array(dynamic_matrix)
			whole_dFNC.append(dynamic_matrix)
		whole_dFNC = np.array(whole_dFNC)
		return whole_dFNC 
	


class get_infant_dataset(General_dataset):
	def __init__(self, mode, dataset_name='infant', GSR=False):
		super().__init__(mode, dataset_name, GSR)
		""" Initializes image paths and preprocessing module."""
		self.mode = mode
		self.workdir = './dynamic_volumne'
		self.dataset = f'./dynamic_volumne/{dataset_name}'
		if not os.path.exists(self.dataset):
			os.mkdir(self.dataset)
		workdir = './Autism_baby'
		base_dFNC = np.array([matrix[np.triu_indices(matrix.shape[0], k=1)] for matrix in base_dFNC])
		component_priors = pd.read_excel('./Functional/MatchTable_High_NetworkLabling_Finalized.xlsx')
		self.component_priors = (component_priors['\'GSP_IC_ID\''].values[:53]-1).astype('int') # python begin at 0
		datasetdir = './Autism_baby'
		path_list, TR_list, session_list, number_list = dataset_info()
		print('load whole infant dataset with', len(session_list) , 'sessions')
		self.timeserieslist = [f'{datasetdir}/cleaned_ASD_Baby_sub{number+1:03}_timecourses_ica_s1_.nii' for number in number_list]
		self.sublist = session_list
		np.savetxt(f'{self.dataset}/sublist.txt', self.sublist, fmt='%s')
		np.savetxt(f'{self.dataset}/timeserieslist.txt', self.timeserieslist, fmt='%s')

		# (timepoint, 100 components)
		# timeseries_select = np.array(timeseries[:, component_number])

		self.sublist, self.timeserieslist = self.check_timeseries()

		self.length = len(self.sublist)
		self.label = self.get_sub_info(self.sublist, 'risk')
		self.sub_index, self.begin_length, self.end_length = self.index_for_list()
		self.sublist = self.sublist[self.sub_index[self.begin_length: self.end_length]]
		self.timeserieslist = self.timeserieslist[self.sub_index[self.begin_length: self.end_length]]
		self.label = self.label[self.sub_index[self.begin_length: self.end_length]]
		
		print("{} image count in {}".format(self.mode, len(self.sublist)), self.begin_length, self.end_length)
		return None


	def get_sub_info(self, session_list, info):
		infolist = []
		df = pd.read_csv('./Demo_baby.csv', header=0)
		df_session_list = df['session_id'].values
		
		if info == 'risk':
			term = df['Risk_Status'] # month
			for session in session_list:
				if session in list(df['session_id'].values):
					vv = term[df['session_id'].values==session].values[0]
					if info == 'risk':
						vv = 0 if vv=='LR' else 1
						infolist.append(vv)
		infolist = np.array(infolist)

		return infolist

	def check_timeseries(self):
		sublist_new = []
		timeserieslist = []
		
		for sub, path in zip(self.sublist, self.timeserieslist):
			if os.path.exists(path) :
				timeserieslist.append(path)
				sublist_new.append(sub)
			else:
				print(path,'not exist')
		return np.array(sublist_new), np.array(timeserieslist)



class get_longitudinal_infant_dataset(General_dataset):
	def __init__(self, mode, state_number, dataset_name='longitudinal_infant', timepoint_number=200,  TD_only=True):
		""" Initializes image paths and preprocessing module."""
		self.mode = mode
		self.kind = 'TP_only' if TD_only else 'all'
		self.state_number = state_number
		self.workdir = './dynamic_volumne'
		self.dataset_config = f'./dynamic_volumne/{dataset_name}'
		if not os.path.exists(self.dataset_config):
			os.mkdir(self.dataset_config)
		self.datasetdir = './Autism_baby'
		self.timepoint_number = timepoint_number
		# neuromark priors
		component_priors = pd.read_excel('./MatchTable_High_NetworkLabling_Finalized.xlsx')
		self.component_priors = (component_priors['\'GSP_IC_ID\''].values[:53]-1).astype('int') # python begin at 0

		self.path_list, self.TR_list, session_list, self.number_list = dataset_info()
		print('load whole infant dataset with', len(session_list) , 'sessions')

		paired_longitudinal_session = longitudinal_dataset(FNC=False, first_date_as_baseline=False)
		self.paired_baseline_timeseries = []
		self.paired_afterline_timeseries = []
		self.baseline_state_timeseries = []
		self.afterline_state_timeseries = []
		self.paired_sub_id = []
		self.paired_demo = []
		self.all_paired_session = []
		for base_session, after_session, age_duration, date_days in paired_longitudinal_session:
			sub_id = base_session[:8]
			base_indice_list = match_session(session_list, [base_session], type='all')
			base_number_list = self.number_list[base_indice_list[0]]
			base_TR_list = self.TR_list[base_indice_list[0]]
			gender_base =  session_infomation_whole( [base_session], 'gender')[0]
			birth_base = session_infomation_whole( [base_session], 'gestational_birth')[0]
			age_base = session_infomation_whole( [base_session], 'corrected_age_week')[0]
			risk_base = session_infomation_whole( [base_session], 'risk')[0] # TP 100% belong to Low Risk
			label_base = session_infomation_whole( [base_session], 'label')[0]
			headmotion_base = session_infomation_whole( [base_session],  'head_motion', self.path_list[base_indice_list[0]])[0]
			if label_base != 0:
				continue
			after_indice_list = match_session(session_list, [after_session], type='all')
			after_number_list = self.number_list[after_indice_list[0]]
			print(base_session,after_session, 'counts:', len(base_number_list), len(after_number_list))
			after_TR_list = self.TR_list[after_indice_list[0]]
			age_after = session_infomation_whole( [after_session], 'corrected_age_week')[0]
			predict_duration = age_after - age_base
			for base_number, base_TR in zip(base_number_list, base_TR_list):
				base_timeseries = f'{self.datasetdir}/cleaned_ASD_Baby_sub{int(base_number)+1:03}_timecourses_ica_s1_.nii' # matlab number begin with 1
				data = nib.load(base_timeseries).get_fdata()
				base_state_dir = f'./dynamic_volumne/dFNC_dataset/{base_session}_{base_number}'
				base_state_timeseries = f'{base_state_dir}/constrained_dFNC_pos_timecourse_5.mat' # [ state, timeseries]

				if (len(data)-45/base_TR)<200: # filter timepoint less than 250 at the baseline
					continue
				for after_number, after_TR in zip(after_number_list, after_TR_list):
					after_timeseries = f'{self.datasetdir}/cleaned_ASD_Baby_sub{int(after_number)+1:03}_timecourses_ica_s1_.nii' # matlab number begin with 1
					after_state_dir = f'./dynamic_volumne/dFNC_dataset/{after_session}_{after_number}'
					after_state_timeseries = f'{after_state_dir}/constrained_dFNC_pos_timecourse_5.mat' # [ state, timeseries]
					self.baseline_state_timeseries.append(base_state_timeseries)
					self.afterline_state_timeseries.append(after_state_timeseries)
					self.paired_baseline_timeseries.append(base_timeseries)
					self.paired_afterline_timeseries.append(after_timeseries)
					self.paired_sub_id.append(sub_id)
					self.paired_demo.append([age_base, predict_duration, gender_base, birth_base, risk_base, label_base,  base_TR, after_TR, headmotion_base])
					self.all_paired_session.append([base_session, after_session])
		self.paired_demo = np.array(self.paired_demo).astype('float')
		self.length = len(self.paired_demo) 
		print('total longitudianl dataset includs', self.length, 'samples')

		self.sample_index= self.index_for_list()
		self.paired_baseline_timeseries = np.array(self.paired_baseline_timeseries)[self.sample_index]
		self.paired_afterline_timeseries = np.array(self.paired_afterline_timeseries)[self.sample_index]
		self.paired_sub_id = np.array(self.paired_sub_id)[self.sample_index]
		self.paired_demo = np.array(self.paired_demo)[self.sample_index]
		self.baseline_state_timeseries = np.array(self.baseline_state_timeseries)[self.sample_index]
		self.afterline_state_timeseries = np.array(self.afterline_state_timeseries)[self.sample_index]
		self.all_paired_session = np.array(self.all_paired_session)[self.sample_index]
		self.sublist = self.paired_baseline_timeseries 

		print("__________{} image count in {} {} {}______________".format(self.mode, self.sample_index.sum(), len(self.paired_baseline_timeseries), len(self.paired_demo)))
		
		return None
	
	def index_for_list(self):
		# split base on the sub_id, rather than scans
		# 80% for training and validation, 20% for testing
		unique_sub_id_list = list(set(self.paired_sub_id))
		unique_sub_number = len(unique_sub_id_list)

		print('unique longitudinal subject number', unique_sub_number )
		if self.mode == 'train':
			begin_length = 0
			end_length = int(0.8*unique_sub_number*0.8)

		elif self.mode =='val':
			begin_length = int(0.8*unique_sub_number*0.8)
			end_length = int(0.8*unique_sub_number)

		elif self.mode == 'test':
			begin_length = int(0.8*unique_sub_number)
			end_length = int(unique_sub_number)

		elif self.mode == 'all':
			begin_length = 0
			end_length = int(unique_sub_number)

		random_sub_id_list_path = f'{self.dataset_config}/random_sub_id_list_{self.kind}.npy'
		if os.path.exists(random_sub_id_list_path):
			random_sub_id_list = np.load(random_sub_id_list_path)
		else:
			
			np.random.shuffle(unique_sub_id_list)
			random_sub_id_list = unique_sub_id_list
			np.save(random_sub_id_list_path, unique_sub_id_list)

		mode_sub_list_path = f'{self.dataset_config}/{self.mode}_sub_id_list_{self.kind}.npy'
		if os.path.exists(mode_sub_list_path):
			self.sub_id_list = np.load(mode_sub_list_path)
		else:
			self.sub_id_list = random_sub_id_list[begin_length: end_length]
			np.save(mode_sub_list_path, self.sub_id_list)
		
		index_path =  f'{self.dataset_config}/{self.mode}_sub_id_index_{self.kind}.npy'
		if os.path.exists(index_path):
			index = np.load(index_path).astype(bool)
		else:
			index = []
			for sub_id in self.paired_sub_id:
				if sub_id in self.sub_id_list:
					index.append(True)
				else:
					index.append(False)
			index = np.array(index)
			np.save(index_path, index)
		return index

	
	def __getitem__(self, index):
		 
		base_timeseries = nib.load(self.paired_baseline_timeseries[index]).get_fdata()
		after_timeseries = nib.load(self.paired_afterline_timeseries[index]).get_fdata()
		
		age_base, predict_duration, gender_base, birth_base, risk_base, label_base, base_TR, after_TR, headmotion_base = self.paired_demo[index]
		demo = np.array([age_base, predict_duration, gender_base, birth_base, headmotion_base, base_TR, after_TR ])

		# [component, timepoint]
		base_timeseries = base_timeseries[:, self.component_priors]
		after_timeseries = after_timeseries[:, self.component_priors]

		# connected areas
		base_connectome = self.get_brain_modularity(base_timeseries.T, uptriangle=True) # [uptriangle]
		after_connectome = self.get_brain_modularity(after_timeseries.T, uptriangle=True) # [uptriangle]

		# get dFNC
		base_dFNC = self.dynamic_corr([base_timeseries], [base_TR], slide_window=45)[0]
		base_dFNC = np.array([matrix[np.triu_indices(matrix.shape[0], k=1)] for matrix in base_dFNC])
		# [timepoint, uptriangle]
		after_dFNC = self.dynamic_corr([after_timeseries], [after_TR], slide_window=45)[0]
		after_dFNC = np.array([matrix[np.triu_indices(matrix.shape[0], k=1)] for matrix in after_dFNC])

		# get state timeseries [timepoint, uptriangle]
		state_timeseries  = loadmat(self.baseline_state_timeseries[index])['timecourse'].T # [state, timepoint]
		after_state_timeseries = loadmat(self.afterline_state_timeseries[index])['timecourse'].T # [state, timepoint]

		if isinstance(self.timepoint_number, int):
			if state_timeseries.shape[1] < self.timepoint_number+1 :
				print('error select timepoint length is larger than whole dynamic timepoint', state_timeseries.shape[1], self.timepoint_number)
			elif state_timeseries.shape[1] == self.timepoint_number+1 :
				pass
			else:
				# print('timepoint number is cut from', state_timeseries.shape[1], 'to', self.timepoint_number)
				index = np.random.randint(0, state_timeseries.shape[1]-self.timepoint_number-1)
				state_timeseries = state_timeseries[:, index: index+self.timepoint_number+1]

		demo = torch.from_numpy(demo.astype('float32'))
		# change_connectome  = torch.from_numpy(change_connectome.astype('float32'))
		base_connectome = torch.from_numpy(base_connectome.astype('float32'))
		after_connectome = torch.from_numpy(after_connectome.astype('float32'))
		state_timeseries = torch.from_numpy(state_timeseries[:,1:].astype('float32'))
		after_state_timeseries = torch.from_numpy(after_state_timeseries[:,1:].astype('float32'))
		return self.baseline_state_timeseries[index], self.afterline_state_timeseries[index], demo, base_connectome, after_connectome, state_timeseries, after_state_timeseries


def get_data_loader(dataset, batch_size, num_workers):
	"""Builds and returns Dataloader."""
	
	# dataset = HCP_data( mode=mode, hemi='L')
	data_loader = data.DataLoader(dataset=dataset,
								  batch_size=batch_size,
								  shuffle=True)
	return data_loader


def get_evaluation_loader(dataset, batch_size, num_workers):
	"""Builds and returns Dataloader."""
	# dataset = HCP_data( mode=mode, hemi='L')
	data_loader = data.DataLoader(dataset = dataset,
								  batch_size = batch_size,
								  shuffle = False,
								  num_workers = num_workers,
								  drop_last = False)
	# batchnum = len(image_paths)//batch_size
	# image_paths = image_paths[:batchnum*batch_size].reshape(-1, batch_size)
	return data_loader



if __name__ == '__main__':
	dataset = get_longitudinal_infant_dataset('all', 5, dataset_name='longitudinal_infant', TD_only=True)
	dataloader = get_evaluation_loader(dataset, 1, 1)

