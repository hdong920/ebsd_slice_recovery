import h5py


class DREAM3D:
    def __init__(self, path: str, synthetic: bool):
        self.path = path 
        self.file = h5py.File(self.path, 'a')
        self.data_container_prefix = '/DataContainers/'
        self.data_container_name = 'IN625InitialConditions/'
        self.feature_am_name = 'FeatureData/'
        self.voxel_data_am_name = 'VoxelData/'
        self.geometry_info_attribute_matrix_name = '_SIMPL_GEOMETRY/'
        self.synthetic = synthetic
        
    def get_feature_data_array(self, array_name):
        """
        Parameters
        ----------
        array_name (str)
            the name of the feature array name to return
        Returns
        -------
        feature_data_array(arr)
            the feature data array
        """
        full_array_path = self.data_container_prefix + self.data_container_name  + self.feature_am_name + array_name
        feature_data_array = self.file[full_array_path][()][1:, :] #.value


        return feature_data_array

    def get_element_data_array(self, array_name): 
        if self.synthetic:
            feature_data_array = self.file['DataContainers']['SyntheticVolumeDataContainer']['CellData'][array_name][()]
        else:
            full_array_path = self.data_container_prefix + self.data_container_name  + self.voxel_data_am_name + array_name
            feature_data_array = self.file[full_array_path][()]
        return feature_data_array

    def get_box_dims(self): 
        dimension_array_path = self.data_container_prefix + self.data_container_name  + self.geometry_info_attribute_matrix_name + 'DIMENSIONS'
        spacing_array_path = self.data_container_prefix + self.data_container_name  + self.geometry_info_attribute_matrix_name + 'SPACING'

        dimension_array = self.file[dimension_array_path][()]
        spacing_array = self.file[spacing_array_path][()]

        box_dims = dimension_array * spacing_array
        volume = box_dims[0]*box_dims[1]*box_dims[2]

        return box_dims, volume, dimension_array, spacing_array 
    
    def close(self):
        self.file.close()
