from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC


def make_linear_svm_with_probs(
    *,  # enforce kwargs
    method="sigmoid",
    cv=5,
    n_jobs=5,
    ensemble=False,
    **svc_kwargs,
) -> CalibratedClassifierCV:
    # this utility function attempts to address several implementation limitations
    # of SVMs with sklearn and libsvm
    # 1. libsvm scales poorly with dataset size, prefer liblinear with LinearSVC
    # 2. LinearSVC does not build in probability estimates as in SVC/libsvm
    # 3. CalibratedClassifierCV can be used to reproduce the probability estimates
    # 4. CalibratedClassifierCV also enables parallelizing internal cv jobs
    # highly recommend reading some of the relevant sklearn documentation:
    # https://scikit-learn.org/stable/modules/generated/sklearn.calibration.CalibratedClassifierCV.html
    # https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
    return CalibratedClassifierCV(
        estimator=LinearSVC(**svc_kwargs),
        method=method,
        cv=cv,
        n_jobs=n_jobs,
        ensemble=ensemble,
    )
