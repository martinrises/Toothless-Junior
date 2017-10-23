from src.data.entity.LabeledData import LabeledData


def get_future_feature_record(records, days, threshold):
    labeled_records_result = []
    for _, record_list in records.items():
        labeled_records = []
        for i in range(days, int(len(record_list) - days/2)):
            labeled_records.append(LabeledData(record_list, i, days, threshold))
        labeled_records_result.append(labeled_records)
    return labeled_records_result
