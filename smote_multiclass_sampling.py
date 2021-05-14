import random
import numpy as np
from pyspark.sql import Row
from sklearn import neighbors
from pyspark.ml.feature import VectorAssembler


class SMOTEMultiClassBalancer:

    def __init__(self, dataframe_list=None):
        self.dataframe_list = []

    def core_smote_sampler(self, vectorised_df, Target, k=5, minority_class=1, majority_class=0,
                           last_minority_class=None, \
                           upsample_percentage=100, downsample_percentage=15):
        if (downsample_percentage > 100 | downsample_percentage < 0):
            raise ValueError('Downsample percentage must be between 15 and 100 percent')
        if (upsample_percentage < 100):
            raise ValueError('Upsample percentage must be at least 100 percent')
        df_minority = vectorised_df[vectorised_df[Target] == minority_class]
        df_majority = vectorised_df[vectorised_df[Target] == majority_class]
        feature = df_minority.select('features')
        feature = feature.rdd
        feature = feature.map(lambda x: x[0])
        feature = feature.collect()
        feature = np.asarray(feature)
        # feature = feature.reshape(-1,1)
        nearest_neighbours = neighbors.NearestNeighbors(n_neighbors=k, algorithm='auto').fit(feature)
        neighbours = nearest_neighbours.kneighbors(feature)
        gap = neighbours[0]
        neighbours = neighbours[1]
        minority_rdd = df_minority.drop(Target).rdd
        pos_rdd_array = minority_rdd.map(lambda x: list(x))
        pos_list_array = pos_rdd_array.collect()
        minority_array = list(pos_list_array)
        new_rows = []
        nt = len(minority_array)
        nexs = int(upsample_percentage / 100)
        for i in range(nt):
            for j in range(nexs):
                neigh = random.randint(1, k)
                difs = minority_array[neigh][0] - minority_array[i][0]
                new_rec = (minority_array[i][0] + random.random() * difs)
                new_rows.insert(0, (new_rec))
        new_data_rdd = sc.parallelize(new_rows)
        new_data_rdd_new = new_data_rdd.map(lambda x: Row(features=x, label=minority_class))
        new_data = new_data_rdd_new.toDF()
        new_data_minority = df_minority.unionAll(new_data)
        self.dataframe_list = self.dataframe_list + [new_data_minority]

        txt = input('Is there any more minority classes? Type Yes or No. ')
        if (txt.lower() == 'yes'):
            minority_class = int(input('Enter minority class:'))
            upsample_percentage = int(input('Enter upsample percentage:'))

            # calling the smote sampler function for remaining other minority classes
            self.core_smote_sampler(df, Target, k, minority_class=minority_class, \
                                    majority_class=majority_class, last_minority_class=None, \
                                    upsample_percentage=upsample_percentage,
                                    downsample_percentage=downsample_percentage)



        elif (txt.lower() == 'no'):
            last_minority_class = minority_class
            # when the minority class equal to 3 is detected, we move to merging the single majority class with other minority classes
            if (minority_class == last_minority_class):
                new_data_majority = df_majority.sample(False, (float(downsample_percentage) / float(100)))
                self.dataframe_list = self.dataframe_list + [new_data_majority]
                print('The length of the upsampled list is ', len(self.dataframe_list))
        return self.dataframe_list

    def smote_sampler(self, upsampled_dataframe_list):
        # taking out the last item in the list for merging later
        temp_df = upsampled_dataframe_list.pop(len(upsampled_dataframe_list) - 1)
        print('The records in the last item is', temp_df.count())
        # creating a merge upsampled dataframe
        for i in range(len(upsampled_dataframe_list)):
            if (len(upsampled_dataframe_list) > 0):
                temp_df = temp_df.unionAll(upsampled_dataframe_list.pop(len(upsampled_dataframe_list) - 1))
                print('The records after merging next item is', temp_df.count())
        return temp_df