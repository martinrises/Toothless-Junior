import src.data.label.origin_data_getter as data_getter
import src.data.label.data_labeler as data_labeler
import src.nn.config as config
from random import shuffle


class DataGetter:
    __data = None
    __test_data_list = None

    @property
    def data(self):
        if self.__data is None:
            origin_records = data_getter.get_future_records()
            labeled_records_list = data_labeler.get_future_feature_record(origin_records, config.DAYS, config.THRESHOLD)
            self.__data = [j for i in labeled_records_list for j in i]
            self.__data = list(filter(lambda data: float("nan") not in data.features, self.__data))
        return self.__data

    def get_test_data_list(self):
        if self.__test_data_list is None:
            origin_records = data_getter.get_future_records()
            self.__test_data_list = data_labeler.get_future_feature_record_with_names(origin_records, config.DAYS, config.THRESHOLD)[58:]
        return self.__test_data_list

    def get_test_data(self):
        return self.data[-config.TEST_DATA_SIZE:]

    def get_non_test_records(self):
        non_test_records = self.data[:-config.TEST_DATA_SIZE]
        return non_test_records

    def get_random_training_data(self):
        non_test_records = self.data[:-config.TEST_DATA_SIZE]
        shuffle(non_test_records)
        return non_test_records[:-config.TEST_DATA_SIZE]

    def get_random_cv_data(self):
        non_test_records = self.data[:-config.TEST_DATA_SIZE]
        shuffle(non_test_records)
        return non_test_records[-config.TEST_DATA_SIZE:]

    @staticmethod
    def get_features(labeled_records):
        features = []
        for record in labeled_records:
            features.append([record.features])
        return features

    @staticmethod
    def get_labels(labeled_records):
        labels = []
        for record in labeled_records:
            labels.append([record.label])
        return labels
