import random
import numpy as np
from pyspark.sql import Row
from sklearn import neighbors
from pyspark.ml import Transformer
from pyspark.sql import DataFrame
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql.functions import udf
from pyspark import keyword_only


class SMOTEMultiClassBalancer(Transformer):

    @keyword_only
    def __init__(self, target, k=5, minority_class=1, majority_class=0, \
                 upsample_percentage=100, downsample_percentage=50):

        self.df = None
        self.target = target
        self.k = k
        self.minority_class = minority_class
        self.majority_class = majority_class
        self.upsample_percentage = upsample_percentage
        self.downsample_percentage = downsample_percentage
        self.dataframe_list = []
        self.nearest_neighbours = None
        self.feature_array = np.empty([0, 0])
        self.synthetic_array = []
        self.new_index = 0
        self.converted_feature_array = []
        self.converted_index = 0
        super(SMOTEMultiClassBalancer, self).__init__()

    def _transform(self, df: DataFrame) -> DataFrame:
        self.df = df
        return self.smote_sampler()

    def smote_sampler(self):
        if (self.downsample_percentage > 100 | self.downsample_percentage < 15):
            raise ValueError('Downsample percentage must be between 15 and 100 percent')
        if (self.upsample_percentage < 100):
            raise ValueError('Upsample percentage must be at least 100 percent')

        dataframe_list = self.core_smote_processor(self.k, self.minority_class, \
                                                   self.majority_class, \
                                                   self.upsample_percentage, self.downsample_percentage)
        upsampled_df = self.dataframe_unifier(dataframe_list)
        return upsampled_df

    def feature_array_generator(self, vectorised_df, minority_class, target):
        df_minority = vectorised_df[vectorised_df[target] == minority_class]
        feature_vector_rdd = df_minority.rdd
        feature_vector_pipelined = feature_vector_rdd.map(lambda x: x[0])
        feature_vector_list = feature_vector_pipelined.collect()
        feature_array = np.asarray(feature_vector_list)
        return feature_array

    def find_k_nearest_neighbours(self, count):
        nearest_neighbour_array = self.nearest_neighbours.kneighbors([self.feature_array[count]], return_distance=False)
        if (len(nearest_neighbour_array) == 1):
            return nearest_neighbour_array[0]
        else:
            return []

    def generate_synthetic_samples(self, num_times, count, nearest_neighbour_array):
        while (num_times > 0):
            random_number = random.randint(0, self.k - 1)
            self.synthetic_array.append([])
            for attr in range(len(self.feature_array[count])):
                # difference gives the difference between feature vector and its nearest neighbour
                diffs = self.feature_array[nearest_neighbour_array[random_number]][attr] - self.feature_array[count][
                    attr]
                # generating random number between 0 and 1
                difference_multiplier = random.random()
                self.synthetic_array[self.new_index].append(
                    self.feature_array[count][attr] + difference_multiplier * diffs)
            self.new_index += 1
            num_times -= 1

    def core_smote_processor(self, k, minority_class, majority_class, \
                             upsample_percentage, downsample_percentage):
        self.feature_array = self.feature_array_generator(self.df, self.minority_class, self.target)
        self.nearest_neighbours = neighbors.NearestNeighbors(n_neighbors=k, algorithm='auto').fit(self.feature_array)
        # synthetic count is used in determining the number of synthetic samples along with the upsample percentage
        synthetic_count = len(self.feature_array)
        # num_times give the number of records to be generated for each synthetic count
        num_times = int(upsample_percentage / 100)
        for i in range(synthetic_count):
            nearest_neighbour_array = self.find_k_nearest_neighbours(i)
            self.generate_synthetic_samples(num_times, i, nearest_neighbour_array)

        combined_array = np.concatenate((self.feature_array, self.synthetic_array))
        self.dtype_converter(combined_array)
        self.dataframe_creator()
        self.synthetic_array = []
        self.new_index = 0
        self.converted_feature_array = []
        self.converted_index = 0
        complete_dataframe_list = self.class_prompt(self.dataframe_list)
        return complete_dataframe_list

    def dtype_converter(self, numpy_float64_feature_array):
        for x in numpy_float64_feature_array:
            self.converted_feature_array.append([])
            for y in x:
                self.converted_feature_array[self.converted_index].append(float(y))
            self.converted_index += 1

    def dataframe_creator(self):
        synthetic_data_rdd = sc.parallelize(self.converted_feature_array)
        label = self.minority_class
        synthetic_data_rdd_pipelined = synthetic_data_rdd.map(lambda x: Row(features=x, label=label))
        synthetic_dataframe = synthetic_data_rdd_pipelined.toDF()

        list_to_vector_udf = udf(lambda x: Vectors.dense(x), VectorUDT())
        synthetic_dataframe_with_vectors = synthetic_dataframe.select(
            list_to_vector_udf(synthetic_dataframe['features']).alias('features'),
            synthetic_dataframe['label'])
        self.dataframe_list = self.dataframe_list + [synthetic_dataframe_with_vectors]

    def class_prompt(self, dataframe_list):
        txt = input('Is there any more minority classes? Type Yes or No. ')
        if (txt.lower() == 'yes'):
            self.minority_class = int(input('Enter minority class:'))
            self.upsample_percentage = int(input('Enter upsample percentage:'))

            # calling the smote sampler function for remaining other minority classes
            self.core_smote_processor(self.k, self.minority_class, self.majority_class, \
                                      self.upsample_percentage, self.downsample_percentage)



        elif (txt.lower() == 'no'):
            last_minority_class = self.minority_class
            # when the minority class equal to 3 is detected, we receive the merged data containing both minority and majority class
            # we do not merge with other minority classes as we'll merge later in a new function
            if (self.minority_class == last_minority_class):
                new_data_majority = self.df[self.df[self.target] == self.majority_class].sample(False, (
                            float(self.downsample_percentage) / float(100)))
                self.dataframe_list = self.dataframe_list + [new_data_majority]
        return self.dataframe_list

    def dataframe_unifier(self, complete_dataframe_list):
        # taking out the last item in the list for merging later
        temp_df = complete_dataframe_list.pop(len(complete_dataframe_list) - 1)
        # creating a merge upsampled dataframe
        for i in range(len(complete_dataframe_list)):
            if (len(complete_dataframe_list) > 0):
                temp_df = temp_df.unionAll(complete_dataframe_list.pop(len(complete_dataframe_list) - 1))
        return temp_df