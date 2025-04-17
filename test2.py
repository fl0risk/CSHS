from methods import ParameterOptimization

try:
    obj = ParameterOptimization(X=[1,2,3], y=[1,2,3], categorical_indicator=[1,1,1,], suite_id=334, try_num_leaves=True, try_max_depth = True, joint_tuning_depth_leaves=False, seed=32644)
except ValueError as e:
    print(f"Error during initialization: {e}")