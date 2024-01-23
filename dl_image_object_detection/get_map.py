import ast
import json
import math
import os
import shutil
import numpy as np
from pathlib import Path


def voc_ap(rec, prec):
    """
    --- Official matlab code VOC2012---
    mrec=[0 ; rec ; 1];
    mpre=[0 ; prec ; 0];
    for i=numel(mpre)-1:-1:1
            mpre(i)=max(mpre(i),mpre(i+1));
    end
    i=find(mrec(2:end)~=mrec(1:end-1))+1;
    ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
    """
    rec.insert(0, 0.0)  # insert 0.0 at begining of list
    rec.append(1.0)  # insert 1.0 at end of list
    mrec = rec[:]
    prec.insert(0, 0.0)  # insert 0.0 at begining of list
    prec.append(0.0)  # insert 0.0 at end of list
    mpre = prec[:]
    """
     This part makes the precision monotonically decreasing
        (goes from the end to the beginning)
        matlab: for i=numel(mpre)-1:-1:1
                    mpre(i)=max(mpre(i),mpre(i+1));
    """

    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    """
     This part creates a list of indexes where the recall changes
        matlab: i=find(mrec(2:end)~=mrec(1:end-1))+1;
    """
    i_list = []
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i - 1]:
            i_list.append(i)  # if it was matlab would be i + 1
    """
     The Average Precision (AP) is the area under the curve
        (numerical integration)
        matlab: ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
    """
    ap = 0.0
    for i in i_list:
        ap += ((mrec[i] - mrec[i - 1]) * mpre[i])
    return ap, mrec, mpre


def log_average_miss_rate(precision, fp_cumsum, num_images):
    """
        log-average miss rate:
            Calculated by averaging miss rates at 9 evenly spaced FPPI points
            between 10e-2 and 10e0, in log-space.

        output:
                lamr | log-average miss rate
                mr | miss rate
                fppi | false positives per image

        references:
            [1] Dollar, Piotr, et al. "Pedestrian Detection: An Evaluation of the
               State of the Art." Pattern Analysis and Machine Intelligence, IEEE
               Transactions on 34.4 (2012): 743 - 761.
    """

    # if there were no detections of that class
    if precision.size == 0:
        lamr = 0
        mr = 1
        fppi = 0
        return lamr, mr, fppi

    fppi = fp_cumsum / float(num_images)
    mr = (1 - precision)

    fppi_tmp = np.insert(fppi, 0, -1.0)
    mr_tmp = np.insert(mr, 0, 1.0)

    # Use 9 evenly spaced reference points in log-space
    ref = np.logspace(-2.0, 0.0, num=9)
    for i, ref_i in enumerate(ref):
        # np.where() will always find at least 1 index, since min(ref) = 0.01 and min(fppi_tmp) = -1.0
        j = np.where(fppi_tmp <= ref_i)[-1][-1]
        ref[i] = mr_tmp[j]

    # log(0) is undefined, so we use the np.maximum(1e-10, ref)
    lamr = math.exp(np.mean(np.log(np.maximum(1e-10, ref))))

    return lamr, mr, fppi


def calculate_map(predictions, work_dir):
    # 忽略浮点数运算错误
    np.seterr(divide='ignore', invalid='ignore')
    evaluate_result_path = work_dir + '/middle_dir/evaluate_result'
    MINOVERLAP = 0.5  # default value (defined in the PASCAL VOC2012 challenge)

    GT_PATH = predictions['target']
    DR_PATH = predictions['prediction']
    IMG_PATH = predictions['path']
    """
     Create a ".temp_files/" and "results/" directory
    """
    TEMP_FILES_PATH = work_dir + "/temp_files"
    if not os.path.exists(TEMP_FILES_PATH):  # if it doesn't exist already
        os.makedirs(TEMP_FILES_PATH)
    results_files_path = evaluate_result_path + "/results_"
    if os.path.exists(results_files_path):  # if it exist already
        # reset the results directory
        shutil.rmtree(results_files_path)
    os.makedirs(results_files_path)

    """
     ground-truth
         Load each of the ground-truth files into a temporary ".json" file.
         Create a list of all the class names present in the ground-truth (gt_classes).
    """
    # dictionary with counter per class
    gt_counter_per_class = {}
    counter_images_per_class = {}
    for img_path, sample in zip(IMG_PATH, GT_PATH):
        file_id = Path(img_path).stem
        try:
            lines_list = ast.literal_eval(sample)
        except:
            lines_list = sample
        # create ground-truth dictionary
        bounding_boxes = []
        is_difficult = False
        already_seen_classes = []
        for line in lines_list:
            left, top, right, bottom, class_name = line['left'], line['top'], line['right'], line['bottom'], line[
                'label']
            bbox = str(left) + " " + str(top) + " " + str(right) + " " + str(bottom)
            if is_difficult:
                bounding_boxes.append(
                    {"class_name": class_name, "bbox": bbox, "used": False, "difficult": True})
                is_difficult = False
            else:
                bounding_boxes.append({"class_name": class_name, "bbox": bbox, "used": False})
                # count that object
                if class_name in gt_counter_per_class:
                    gt_counter_per_class[class_name] += 1
                else:
                    # if class didn't exist yet
                    gt_counter_per_class[class_name] = 1

                if class_name not in already_seen_classes:
                    if class_name in counter_images_per_class:
                        counter_images_per_class[class_name] += 1
                    else:
                        # if class didn't exist yet
                        counter_images_per_class[class_name] = 1
                    already_seen_classes.append(class_name)

        # dump bounding_boxes into a ".json" file
        with open(TEMP_FILES_PATH + "/" + str(file_id) + "_ground_truth.json", 'w') as outfile:
            json.dump(bounding_boxes, outfile)

    gt_classes = list(gt_counter_per_class.keys())
    # let's sort the classes alphabetically
    gt_classes = sorted(gt_classes)
    n_classes = len(gt_classes)
    print("gt_classes:", gt_classes)

    """
     detection-results
         Load each of the detection-results files into a temporary ".json" file.
    """
    for class_index, class_name in enumerate(gt_classes):
        bounding_boxes = []
        for img_path, sample in zip(IMG_PATH, DR_PATH):
            file_id = Path(img_path).stem

            # 如果没有检测到目标的图片不参与mAP计算,走continue
            try:
                lines_list = ast.literal_eval(sample)
            except Exception as e:
                # print("读取预测结果发生错误：", e)
                continue

            # 如果单张图片检测结果超过100个，采样前100个用于评估
            if len(lines_list) > 100:
                lines_list = lines_list[:100]

            for line in lines_list:
                left, top, right, bottom, tmp_class_name, confidence = \
                    line['left'], line['top'], line['right'], line['bottom'], line['label'], line['confidence']
                if tmp_class_name == class_name:
                    # print("match")
                    bbox = str(left) + " " + str(top) + " " + str(right) + " " + str(bottom)
                    bounding_boxes.append({"confidence": confidence, "file_id": file_id, "bbox": bbox})

        # sort detection-results by decreasing confidence
        bounding_boxes.sort(key=lambda x: float(x['confidence']), reverse=True)
        with open(TEMP_FILES_PATH + "/" + class_name + "_dr.json", 'w') as outfile:
            json.dump(bounding_boxes, outfile)

    """
     Calculate the AP for each class
    """
    sum_AP = 0.0
    ap_dictionary = {}
    lamr_dictionary = {}
    # open file to store the results
    with open(results_files_path + "/results.txt", 'w') as results_file:
        # results_file.write("# AP and precision/recall per class\n")
        count_true_positives = {}

        for class_index, class_name in enumerate(gt_classes):
            count_true_positives[class_name] = 0
            """
             Load detection-results of that class
            """
            dr_file = TEMP_FILES_PATH + "/" + class_name + "_dr.json"
            dr_data = json.load(open(dr_file))

            """
             Assign detection-results to ground-truth objects
            """
            nd = len(dr_data)
            tp = [0] * nd  # creates an array of zeros of size nd
            fp = [0] * nd
            score = [0] * nd
            score05_idx = 0
            for idx, detection in enumerate(dr_data):
                file_id = detection["file_id"]
                score[idx] = float(detection["confidence"])
                if score[idx] > 0.5:
                    score05_idx = idx

                # assign detection-results to ground truth object if any
                # open ground-truth with that file_id
                gt_file = TEMP_FILES_PATH + "/" + str(file_id) + "_ground_truth.json"
                ground_truth_data = json.load(open(gt_file))
                ovmax = -1
                gt_match = -1
                # load detected object bounding-box
                bb = [float(x) for x in detection["bbox"].split()]
                for obj in ground_truth_data:
                    # look for a class_name match
                    if obj["class_name"] == class_name:
                        bbgt = [float(x) for x in obj["bbox"].split()]
                        bi = [max(bb[0], bbgt[0]), max(bb[1], bbgt[1]), min(bb[2], bbgt[2]),
                              min(bb[3], bbgt[3])]
                        iw = bi[2] - bi[0] + 1
                        ih = bi[3] - bi[1] + 1
                        if iw > 0 and ih > 0:
                            # compute overlap (IoU) = area of intersection / area of union
                            ua = (bb[2] - bb[0] + 1) * (bb[3] - bb[1] + 1) + (bbgt[2] - bbgt[0]
                                                                              + 1) * (
                                         bbgt[3] - bbgt[1] + 1) - iw * ih
                            ov = iw * ih / ua
                            if ov > ovmax:
                                ovmax = ov
                                gt_match = obj

                # set minimum overlap
                min_overlap = MINOVERLAP
                if ovmax >= min_overlap:
                    if "difficult" not in gt_match:
                        if not bool(gt_match["used"]):
                            # true positive
                            tp[idx] = 1
                            gt_match["used"] = True
                            count_true_positives[class_name] += 1
                            # update the ".json" file
                            with open(gt_file, 'w') as f:
                                f.write(json.dumps(ground_truth_data))
                        else:
                            # false positive (multiple detection)
                            fp[idx] = 1
                else:
                    # false positive
                    fp[idx] = 1

            # compute precision/recall
            cumsum = 0
            for idx, val in enumerate(fp):
                fp[idx] += cumsum
                cumsum += val
            cumsum = 0
            for idx, val in enumerate(tp):
                tp[idx] += cumsum
                cumsum += val
            # print(tp)
            rec = tp[:]

            for idx, val in enumerate(tp):
                rec[idx] = float(tp[idx]) / gt_counter_per_class[class_name]
            # print(rec)
            prec = tp[:]
            for idx, val in enumerate(tp):
                prec[idx] = float(tp[idx]) / (fp[idx] + tp[idx])
            # print(prec)
            ap, mrec, mprec = voc_ap(rec[:], prec[:])
            F1 = np.array(rec) * np.array(prec) / (np.array(prec) + np.array(rec)) * 2
            sum_AP += ap
            text = class_name + " AP = {0:.2f}%".format(ap * 100)

            try:
                results_file.write(text + " || score_threshold=0.5 : " + "F1=" + "{0:.2f}".format(
                    F1[score05_idx]) + " ; Recall=" + "{0:.2f}%".format(
                    rec[score05_idx] * 100) + " ; Precision=" + "{0:.2f}%".format(
                    prec[score05_idx] * 100) + "\n")
                results_file.write('Precision:' + str(prec) + '\nRecall:' + str(rec) + '\nF1:' + str(
                    list(F1)) + '\nscore:' + str(score) + "\n\n")
            except:
                results_file.write(text + " || score_threshold=0.5 : " + "F1=" + "{0:.2f}".format(
                    0) + " ; Recall=" + "{0:.2f}%".format(0 * 100) + " ; Precision=" + "{0:.2f}%".format(
                    0 * 100) + "\n")
                results_file.write('Precision:' + str(prec) + '\nRecall:' + str(rec) + '\nF1:' + str(
                    list(F1)) + '\nscore:' + str(score) + "\n\n")
            else:
                print(text + "\t||\tscore_threhold=0.5 : " + "F1=" + "{0:.2f}".format(
                    F1[score05_idx]) + " ; Recall=" + "{0:.2f}%".format(
                    rec[score05_idx] * 100) + " ; Precision=" + "{0:.2f}%".format(prec[score05_idx] * 100))
            ap_dictionary[class_name] = ap

            n_images = counter_images_per_class[class_name]
            lamr, mr, fppi = log_average_miss_rate(np.array(rec), np.array(fp), n_images)
            lamr_dictionary[class_name] = lamr

        # results_file.write("\n# mAP of all classes\n")
        mAP = sum_AP / n_classes
        text = "mAP = {0:.2f}%".format(mAP * 100)
        results_file.write(text + "\n")
        print(text)

    # remove the temp_files directory
    shutil.rmtree(TEMP_FILES_PATH)

    """
     Count total of detection-results
    """
    # iterate through all the files
    det_counter_per_class = {}
    for sample in DR_PATH:
        # get lines to list
        try:
            lines_list = ast.literal_eval(sample)
        except Exception as e:
            # print("读取预测结果发生错误：", e)
            continue
        for line in lines_list:
            class_name = line['label']

            # count that object
            if class_name in det_counter_per_class:
                det_counter_per_class[class_name] += 1
            else:
                # if class didn't exist yet
                det_counter_per_class[class_name] = 1
    dr_classes = list(det_counter_per_class.keys())

    """
     Write number of ground-truth objects per class to results.txt
    """
    ground_truth_dict = {}
    with open(results_files_path + "/results.txt", 'a') as results_file:
        results_file.write("\n# Number of ground-truth objects per class\n")
        for class_name in sorted(gt_counter_per_class):
            ground_truth_dict[class_name] = gt_counter_per_class[class_name]
        results_file.write(str(ground_truth_dict) + "\n")

    """
     Write number of lamr per class to results.txt
    """
    with open(results_files_path + "/results.txt", 'a') as results_file:
        results_file.write("\n# Number of lamr per class\n")
        results_file.write(str(lamr_dictionary) + "\n")

    """
     Finish counting true positives
    """
    for class_name in dr_classes:
        # if class exists in detection-result but not in ground-truth then there are no true positives in that class
        if class_name not in gt_classes:
            count_true_positives[class_name] = 0

    """
     Write number of detected objects per class to results.txt
    """
    detect_result_dict = {}
    with open(results_files_path + "/results.txt", 'a') as results_file:
        results_file.write("\n# Number of detected objects per class\n")
        for class_name in sorted(dr_classes):
            n_det = det_counter_per_class[class_name]
            class_tp_fp_dict = {'tp': str(count_true_positives[class_name]),
                                'fp': str(n_det - count_true_positives[class_name])}
            detect_result_dict[class_name] = class_tp_fp_dict
        results_file.write(str(detect_result_dict))

    return mAP, gt_classes
