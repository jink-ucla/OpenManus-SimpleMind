import sys
import argparse
import yaml
import ast

"""
import inspect
import your_module  # replace with the module you imported

# List all functions defined in your_module
functions_list = [name for name, obj in inspect.getmembers(your_module, inspect.isfunction)]

"""
def get_function_names_from_file(file_path):
    with open(file_path, 'r') as file:
        node = ast.parse(file.read(), filename=file_path)

    func_names = [n.name for n in ast.walk(node) if isinstance(n, ast.FunctionDef)]
    return func_names

def learning_rate_scheduler_parameters(file_path, non_tunable=('epoch', 'lr')):
    with open(file_path, "r") as file:
        tree = ast.parse(file.read(), filename=file_path)

    tunable_params_dict = {}

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            all_args = [arg.arg for arg in node.args.args]
            tunable_args = [arg for arg in all_args if arg not in non_tunable]
            tunable_params_dict[node.name] = tunable_args

    return tunable_params_dict

DEFAULT_REFERENCE_YAML = "/cvib2/apps/personal/wasil/lib/sm/sm_verifier/simplemind/agent/nn/tf2/engine/updated_example_settings.yaml"
DEFAULT_REFERENCE_LRS_PATH = "/cvib2/apps/personal/ssshin/lib/simplemind/simplemind/agent/nn/tf2/engine/learning_rate_schedulers.py"
DEFAULT_REFERENCE_METRICS_LOSS_PATH = "/cvib2/apps/personal/ssshin/lib/simplemind/simplemind/agent/nn/tf2/engine/metrics_and_loss.py"


def get_reference_lookup(ref_yaml=DEFAULT_REFERENCE_YAML, lrs_path=DEFAULT_REFERENCE_LRS_PATH, metrics_loss_path=DEFAULT_REFERENCE_METRICS_LOSS_PATH):

    arch_list = ['EfficientNetB0', 'EfficientNetB1', 'EfficientNetB2', 'EfficientNetB3', 'EfficientNetB4', 'EfficientNetB5', 'EfficientNetB6', 'EfficientNetB7', 'ResNet18', 'ResNet34', 'ResNet50', 'ResNet101', 'ResNet152', 'unet_2d', 'unet_3d']

    augmentation_list = ['random_rotation', 'random_brightness', 'random_contrast', 'random_zoom', 'random_deformation', 'no_augmentation', 'flip_x', 'flip_y']

    learning_rate_scheduler_dict = learning_rate_scheduler_parameters(lrs_path) # dictionary of learning rate schedulers and their tunable parameters as values
    metrics_loss_list = get_function_names_from_file(metrics_loss_path)

    with open(ref_yaml, 'r') as f:
        tf2_yaml = yaml.load(f, Loader=yaml.FullLoader)
        tf2_headers = list((tf2_yaml).keys())

    return dict(tf2_headers=tf2_headers, arch=arch_list, augmentation=augmentation_list, learning_rate_scheduler=learning_rate_scheduler_dict, metrics_loss=metrics_loss_list)
    # learning_rate_scheduler_list = get_function_names_from_file('/cvib2/apps/personal/ssshin/lib/simplemind/simplemind/agent/nn/tf2/engine/learning_rate_schedulers.py')



def check_yaml(llm_yaml, ref_yaml=DEFAULT_REFERENCE_YAML, lrs_path=DEFAULT_REFERENCE_LRS_PATH, metrics_loss_path=DEFAULT_REFERENCE_METRICS_LOSS_PATH):
    ref_lookup = get_reference_lookup(ref_yaml, lrs_path, metrics_loss_path)

    with open(llm_yaml, 'r') as f:
        llm_yaml = yaml.load(f, Loader=yaml.FullLoader)
        llm_yaml_headers = list(llm_yaml.keys())

    check_passed = True
    feedback = "All checks passed."

    if not llm_yaml_headers:
        check_passed = False
        feedback = "YAML file is empty."
        return check_passed, feedback

    for header in llm_yaml_headers:
        if header not in ref_lookup["tf2_headers"]:
            print(header)
            check_passed = False
            feedback = f"{header} setting does not exist."
            return check_passed, feedback
    


    if isinstance(llm_yaml['augmentations'], dict):
        for augmentator in list(llm_yaml['augmentations'].keys()):
            if augmentator not in ref_lookup["augmentation"]:
                check_passed = False
                feedback = f"{augmentator} augmentation does not exist."
                return check_passed, feedback

    else:
        check_passed = False
        feedback = "Augmentations is missing inputs."
        return check_passed, feedback       
    


    if llm_yaml['architecture']['arch_name'] not in ref_lookup["arch"]:
        check_passed = False
        feedback = f"{llm_yaml['architecture']['arch_name']} does not exist."
        return check_passed, feedback
    


    if isinstance(llm_yaml['metrics'], list):
        for metric in llm_yaml['metrics']:
            if metric not in ref_lookup["metrics_loss"]:
                check_passed = False
                feedback = f"{metric} metric does not exist."
                return check_passed, feedback

    else:
        check_passed = False
        feedback = "No metrics found."
        return check_passed, feedback
    


    if isinstance(llm_yaml['loss_function'], list):
        for loss in llm_yaml['loss_function']:
            if loss not in ref_lookup["metrics_loss"]:
                check_passed = False
                feedback = f"{loss} loss function does not exist."
                return check_passed, feedback

    else:
        check_passed = False
        feedback = "No loss functions found."
        return check_passed, feedback
    


    if isinstance(llm_yaml['learning_rate_scheduler'], dict):
        if llm_yaml['learning_rate_scheduler']['scheduler'] not in list(ref_lookup["learning_rate_scheduler"].keys()):
            check_passed = False
            feedback = f"{llm_yaml['learning_rate_scheduler']['scheduler']} learning rate scheduler does not exist."
            return check_passed, feedback



        if not list(llm_yaml['learning_rate_scheduler'])[1:] == list(ref_lookup["learning_rate_scheduler"][str(llm_yaml['learning_rate_scheduler']['scheduler'])]):
            check_passed = False
            feedback = "Incorrect parameters for learning rate scheduler."
            return check_passed, feedback

    else:
        check_passed = False
        feedback = "No learning rate scheduler found."
        return check_passed, feedback
    
    return check_passed, feedback