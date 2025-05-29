import yaml


"""
SmDecisionTree:
  name: centroid_offset_y
  reference: lung_init
  units: mm
  threshold: 20
  left:
    value: [1.0, 0] # Probabilities of each class (1st class is 'no', 2nd class is 'yes')
  right:
    name: circularity
    units: decimal
    threshold: 0.50
    right:
      value: [1.0, 0] # Probabilities of each class (1st class is 'no', 2nd class is 'yes')
    left:
      name: RightOfPlanarCentroid
      reference: lung_init
      units: mm
      threshold: 30
      left:
        value: [1.0, 0] # Probabilities of each class (1st class is 'no', 2nd class is 'yes')
      right:
        name: volume
        units: mm3
        threshold: 10000
        left:
          value: [1.0, 0] # Probabilities of each class (1st class is 'no', 2nd class is 'yes')
        right:
          value: [0.0, 1.0] # Probabilities of each class (1st class is 'no', 2nd class is 'yes')


"""

def check_dt_feature_existence(feature_name):
    check_passed = True
    feedback = "All checks passed."
    return check_passed, feedback

def check_value(value):
    # value = eval(value)
    check_passed = True
    feedback = "All checks passed."
    ### needs to be a list of two values
    if not(isinstance(value, list) or isinstance(value, tuple)):
        check_passed = False
        feedback = f"Value '{value}' is not a list. It is currently a {type(value)}. Value must be a list of two elements, each of a float between 0 and 1."
        return check_passed, feedback

    if not len(value)==2:
        check_passed = False
        feedback = f"Value '{value}' is a list/tuple greater than 2 elements. Value must be a list of two elements, each of a float between 0 and 1."
    for val in value:
        if not isinstance(val, float) and not isinstance(val, int):
            check_passed = False
            feedback = f"Value '{value}' is a list that contains non-float elements. It currently includes at least a {type(val)}. Value must be a list of two elements, each of a float between 0 and 1."
        if val > 1 or val < 0:
            check_passed = False
            feedback = f"Value '{value}' is a list that contains float values that are not between 0 to 1. Value must be a list of two elements, each of a float between 0 and 1."

    return check_passed, feedback

def check_threshold(value):
    ### probably needs to be a float
    check_passed = True
    feedback = "All checks passed."
    return check_passed, feedback


def check_current_branch_level(current_branch_level, i_level, ref_roi_added=False):
    check_passed = True
    feedback = "All checks passed."

    if current_branch_level.get("value") is not None:
        ### check it
        # if False:
        #     check_passed = True
        #     feedback = "All checks passed."
        check_passed, feedback = check_value(current_branch_level.get("value"))
        if not check_passed:
            feedback = f"Error with the 'value' field defined at level {i_level}: " + feedback
        return check_passed, feedback

    if current_branch_level.get("right") is None:
        check_passed = False
        feedback = f"'right' doesn't exist at branch level {i_level}."
        return check_passed, feedback
    if current_branch_level.get("left") is None:
        check_passed = False
        feedback = f"'left' doesn't exist at branch level {i_level}."
        return check_passed, feedback
    
    ### checking name
    if current_branch_level.get("name") is None:
        check_passed = False
        feedback = f"'name' doesn't exist at branch level {i_level}."
        return check_passed, feedback
    else:
        check_passed, feedback = check_dt_feature_existence(current_branch_level["name"])
        if not check_passed:
            return check_passed, feedback

    ### checking threshold
    if current_branch_level.get("threshold") is None:
        check_passed = False
        feedback = f"'threshold' doesn't exist at branch level {i_level}."
        return check_passed, feedback
    else:
        check_passed, feedback = check_threshold(current_branch_level["threshold"])
        if not check_passed:
            return check_passed, feedback
        
    ### NOTE: not ready

    # if current_branch_level.get("reference") is not None:
    #     if ref_roi is None:
    #         check_passed = False
    #         feedback = f"Reference ROI was defined in the decision tree in level {i_level} as '{current_branch_level.get("reference")}' but wasn't defined as an input."
    #         return check_passed, feedback
    #     else:
    #         if not current_branch_level.get("reference")==ref_roi:
    #             check_passed = False
    #             feedback = f"The reference ROI that was defined in the decision tree in level {i_level} as '{current_branch_level.get("reference")}' but doesn't match the input ROI {ref_roi}."
    #             return check_passed, feedback
    if current_branch_level.get("reference") is not None:
        if not ref_roi_added:
            check_passed = False
            feedback = f"Reference ROI was defined in the decision tree in level {i_level} as '{current_branch_level.get('reference')}' but wasn't defined as an input."
            return check_passed, feedback


    check_passed, feedback = check_current_branch_level(current_branch_level["left"], i_level+1, ref_roi_added)
    if not check_passed:
        return check_passed, feedback
    check_passed, feedback = check_current_branch_level(current_branch_level["right"], i_level+1, ref_roi_added)
    if not check_passed:
        return check_passed, feedback

        

    return check_passed, feedback

def check_dt_yaml(dt_yaml, ref_roi_added=False):
    with open(dt_yaml, 'r') as f:
        dt = yaml.load(f, Loader=yaml.FullLoader)

    check_passed = True
    feedback = "All checks passed."
    if dt.get("SmDecisionTree") is None:
        check_passed = False
        feedback = "'SmDecisionTree' must be defined on the base level of the Decision Tree dictionary."
        return check_passed, feedback


    check_passed, feedback = check_current_branch_level(dt["SmDecisionTree"], 1, ref_roi_added=ref_roi_added)






    return check_passed, feedback

if __name__ == "__main__":    

    dt_yaml = "/radraid/apps/personal/wasil/simplemind-examples/ct_chest/dt/right_lung_base-lung_init.yaml"
    print(check_dt_yaml(dt_yaml, True))