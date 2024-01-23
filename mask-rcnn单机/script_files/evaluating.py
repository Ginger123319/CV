

def get_gt_dr_map(test_data, image_size, classes, work_dir):
    from script_files import get_gt_txt, get_dr_txt, get_map
    
    get_gt_txt.get_gt(test_data, classes, work_dir)
    get_dr_txt.get_dr(test_data, image_size, work_dir)
    best_model = get_map.calculate_map(work_dir)
    
    return best_model
    
    
def evaluate(best_model, work_dir):
    from script_files import get_performance
    performance = get_performance.calculate_performance(best_model, work_dir)
    
    return performance