from src.data.entity.LabeledData import LabeledData


def get_future_feature_record_with_names(records, days, threshold):
    labeled_records_result = []
    items = records.items()
    items = sorted(items, key=lambda items: items[0])
    for order_id, record_list in items:
        labeled_records = []
        for i in range(days, int(len(record_list) - days / 2)):
            labeled_records.append(LabeledData(record_list, i, days, threshold))
        labeled_records_result.append(labeled_records)
    result = []
    for i in range(len(items)):
        result.append([items[i][0], labeled_records_result[i]])
    return result


def get_future_feature_record(records, days, threshold):
    labeled_records_result = []
    items = records.items()
    items = sorted(items, key=lambda items: items[0])
    for _, record_list in items:
        labeled_records = []
        for i in range(days, int(len(record_list) - days/2)):
            labeled_records.append(LabeledData(record_list, i, days, threshold))
        labeled_records_result.append(labeled_records)
    return labeled_records_result
