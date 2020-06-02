from feat_imp.generate_dataset import prepare_ML_sets
from feat_imp.conf_file_generation import GENERATION_CONF


(X_train, y_train, X_eval, y_eval, X_test, y_test) = prepare_ML_sets(
    GENERATION_CONF, 1000, test_size=0.25, seed=17
)
