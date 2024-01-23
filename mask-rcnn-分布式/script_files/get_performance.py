import re
import os

def calculate_performance(best_model, work_dir):
    
    f = open(work_dir+'/middle_dir_mask_dis/data_info/classes.txt', 'r')
    class_list = f.readlines()
    class_list = [c.strip() for c in class_list]
    class_list = sorted(class_list)
    f.close()
    
    f = open(work_dir+'/middle_dir_mask_dis/evaluate_result/results_'+best_model+'/results.txt')
    contents = f.read()
    f.close()

    ap_value_dict = {}
    threshold_value_dict = {}
    precision_value_dict = {}
    f1_value_dict = {}
    recall_value_dict = {}
    precision_dict = {}
    recall_dict = {}
    f1_dict = {}
    score_dict = {}
    nan = 0 
    for i in class_list:
        each_calss_data = re.match(r'%s AP = (\d+.\d+).+score_threshold=(\d+.\d+) : F1=(.+) ; Recall=(\d+.\d+). ; Precision=(\d+.\d+).\sPrecision:(.+)\sRecall:(.+)\sF1:(.+)\sscore:(.+)\s\s' % (i), contents)
        ap_value_data = each_calss_data.group(1)
        ap_value_dict[i] = round(eval(ap_value_data) / 100, 4)
        threshold_value_data = each_calss_data.group(2)
        threshold_value_dict[i] = threshold_value_data
        f1_value_data = each_calss_data.group(3)
        f1_value_dict[i] = f1_value_data
        recall_value_data = each_calss_data.group(4)
        recall_value_dict[i] = round(eval(recall_value_data) / 100, 4)
        precision_value = each_calss_data.group(5)
        precision_value_dict[i] = round(eval(precision_value) / 100, 4)
        precision_data = each_calss_data.group(6)
        precision_dict[i] = eval(precision_data)
        recall_data = each_calss_data.group(7)
        recall_dict[i] = eval(recall_data)
        f1_data = each_calss_data.group(8)
        f1_dict[i] = eval(f1_data)
        score_data = each_calss_data.group(9)
        score_dict[i] = eval(score_data)
        contents = contents.replace(each_calss_data.group(), '')

    mAP_data = re.match(r'mAP = (\d+.\d+).\n\n.+\n(.+)\n\n.+\n(.+)\n\n.+\n(.+)', contents)
    mAP = round(eval(mAP_data.group(1)) / 100, 4)
    ground_truth_dict = eval(mAP_data.group(2))
    lamr_dict = eval(mAP_data.group(3))
    detect_results_dict = eval(mAP_data.group(4))

    performance = []
    AP_curve = {"type": "precision_recall_curve", "name": "precision_recall_curve"}
    Precision_curve = {"type": "precision_curve", "name": "precision_curve"}
    Recall_curve = {"type": "recall_curve", "name": "recall_curve"}
    F1_curve = {"type": "f1_curve", "name": "f1_curve"}
    metric_data = {"type": "metrics", "name": "metrics", "data": {"mAP": mAP}}
    mAP_bar = {"type": "mAP", "name": "mAP", "data": {"mAP": ap_value_dict}}
    ground_truth_bar = {"type": "ground_truth", "name": "ground_truth", "data": {"ground_truth": ground_truth_dict}}
    lamr_bar = {"type": "lamr", "name": "lamr", "data": {"lamr": lamr_dict}}
    detect_results_bar = {"type": "detect_results", "name": "detect_results", "data": {"detect_results": detect_results_dict}}

    ap_data_dict = {}
    precision_data_dict = {}
    recall_data_dict = {}
    f1_data_dict = {}
    for i in class_list:
        ap_chartData_list = []
        precision_chartData_list = []
        recall_chartData_list = []
        f1_chartData_list = []
        for index in range(len(precision_dict[i])):
            ap_chartData_list.append({"Recall": recall_dict[i][index], "Precision": precision_dict[i][index]})
            precision_chartData_list.append({"Threshold": score_dict[i][index], "Precision": precision_dict[i][index]})
            recall_chartData_list.append({"Threshold": score_dict[i][index], "Recall": recall_dict[i][index]})
            f1_chartData_list.append({"Threshold": score_dict[i][index], "F1": f1_dict[i][index]})
        ap_data_dict[i] = {"charData": ap_chartData_list, "precision_recall_curve": ap_value_dict[i]}
        precision_data_dict[i] = {"charData": precision_chartData_list, "precision_curve": precision_value_dict[i]}
        recall_data_dict[i] = {"charData": recall_chartData_list, "recall_curve": recall_value_dict[i]}
        f1_data_dict[i] = {"charData": f1_chartData_list, "f1_curve": f1_value_dict[i]}
    AP_curve['data'] = ap_data_dict
    Precision_curve['data'] = precision_data_dict
    Recall_curve['data'] = recall_data_dict
    F1_curve['data'] = f1_data_dict

    performance.append(AP_curve)
    performance.append(Precision_curve)
    performance.append(Recall_curve)
    performance.append(F1_curve)
    performance.append(metric_data)
    performance.append(mAP_bar)
    performance.append(ground_truth_bar)
    performance.append(lamr_bar)
    performance.append(detect_results_bar)

    return performance
    