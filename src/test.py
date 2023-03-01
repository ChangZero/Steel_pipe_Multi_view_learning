import yaml
import os
import tensorflow as tf
import numpy as np
from util import load_df, load_df2, load_x_data, load_y_data, recall_m, precision_m, f1_m, seed_fix
from inference import export_test_result, export_eval_result, export_hit_eval_result, export_excel

def test(test_path, model_path):
    target_size = [400, 400, 1]
    
    test_result_path = model_path.split('/')[:-2]
    experiment_path = model_path.split('/')[:-3]
    model_name = test_result_path[-1]
    test_result_path = "/".join(test_result_path)
    experiment_path = "/".join(experiment_path)
    
    # test_df = load_df1(test_path)
    test_df = load_df(test_path)
    b1_x_test = load_x_data(test_path, "median_blur", target_size, test_df)
    b2_x_test = load_x_data(test_path, "sobel_masking_y", target_size, test_df)
    b3_x_test = load_x_data(test_path, "original_image", target_size, test_df)
    y_test = load_y_data(test_df)
    
    model = tf.keras.models.load_model(model_path, custom_objects={"recall_m": recall_m, "precision_m": precision_m, "f1_m": f1_m})

    # loss, accuracy, f1, precision, recall = model.evaluate(x=(b1_x_test, b2_x_test, b3_x_test), y=y_test)
    
    y_prob = model.predict(x=(b1_x_test, b2_x_test, b3_x_test))
    y_pred = (y_prob > 0.5).astype("int32")
    print(y_pred)
    
    test_df = export_test_result(test_df, y_pred, y_prob, test_result_path)
    #export_eval_result(y_test, y_pred, test_result_path)
    export_hit_eval_result(test_df, y_prob, test_result_path, model_name)
    
    
    result_dic = {}
    result_dic[model_name] = experiment_path
    model_list = [model_name]
    export_excel(model_list, result_dic)
        
    
def main():
    with open("./test-config.yaml", "r") as f:
        data = yaml.full_load(f)
        
    test_path = data["test_path"]
    model_path = data["model_path"]
    seed = data["seed"]
    
    seed_fix(seed)
    test(test_path, model_path)

    return 0

if __name__ == '__main__':
    # config = parse_opt()
    main()