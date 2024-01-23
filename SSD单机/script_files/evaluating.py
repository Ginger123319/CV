

def get_gt_dr_map(test_data, image_size, classes, work_dir):
    from script_files import get_gt_txt, get_dr_txt, get_map
    
    get_gt_txt.get_gt(test_data, classes, work_dir)
    get_dr_txt.get_dr(test_data, image_size, work_dir)
    best_model = get_map.calculate_map(work_dir)
    
    return best_model
    
    
def evaluate(best_model, work_dir):
    from script_files import get_performance
    performance = get_performance.calculate_performance(best_model, work_dir)
    
    for i in performance:
        if i['name'] in ["precision_recall_curve", "precision_curve", "recall_curve", "f1_curve"]:
            for j in i['data'].values():
                steps = int(len(j['charData'])/500)
                if steps == 0:
                    steps = 1
                j['charData'] = j['charData'][::steps]
    
    return performance