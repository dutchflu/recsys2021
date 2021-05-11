"""
Evaluation Methods
"""
from abc import abstractmethod
from sklearn.metrics import average_precision_score, log_loss


class Evaluator:
    """Abstract base class"""
    @abstractmethod
    def score(self, ypred, ytrue):
        pass


def calculate_ctr(ytrue):
    positive = len([x for x in ytrue if x == 1])
    ctr = positive/float(len(ytrue))
    return ctr


def compute_rce(pred, ytrue):
    cross_entropy = log_loss(ytrue, pred)
    data_ctr = calculate_ctr(ytrue)
    strawman_cross_entropy = log_loss(ytrue, [data_ctr for _ in range(len(ytrue))])
    return (1.0 - cross_entropy/strawman_cross_entropy)*100.0


class RceEvaluator(Evaluator):
    """rce"""
    def score(self, ypred, ytrue):
        scores = compute_rce(ypred, ytrue)

        return scores


class AvgPrecisionEvaluator(Evaluator):
    """average_precision_score"""
    def score(self, ypred, ytrue):
        scores = average_precision_score(ytrue, ypred)

        return scores
