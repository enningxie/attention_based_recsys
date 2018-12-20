import math
import numpy as np
from multiprocessing import Pool

# global variables to share
_model = None
_testRatings = None
_testNegatives = None
_K = None
_train_dict = None


def evaluate_model(model, testRatings, testNegatives, K, num_thread):
    global _model
    global _testRatings
    global _testNegatives
    global _K
    global _train_dict
    _model = model
    _testRatings = testRatings
    _testNegatives = testNegatives
    _K = K

    hits, ndcgs = [], []
    if num_thread > 1:  # multi-thread
        with Pool(processes=num_thread) as p:
            res = p.map(eval_one_rating, range(len(_testRatings)))
        hits = [r[0] for r in res]
        ndcgs = [r[1] for r in res]
        return hits, ndcgs

    # single thread
    for testRating_id in range(len(_testRatings)):
        hr, ndcg = eval_one_rating(testRating_id)
        hits.append(hr)
        ndcgs.append(ndcg)
    return hits, ndcgs


def eval_one_rating(idx):
    rating = _testRatings[idx]
    items = _testNegatives[idx]
    user_id = rating[0]
    item_id = rating[1]
    items.append(item_id)
    # get prediction scores
    map_item_score = {}
    users = np.full(len(items), user_id, dtype=np.int32)
    predictions = _model.predict([users, np.array(items)], batch_size=100, verbose=0)
    for i in range(len(items)):
        item = items[i]
        map_item_score[item] = predictions[i]
    items.pop()
    ranklist = sorted(map_item_score, key=map_item_score.get, reverse=True)[:_K]
    hr = getHitRatio(ranklist, item_id)
    ndcg = getNDCG(ranklist, item_id)
    return hr, ndcg


def getHitRatio(ranklist, item_id):
    for item in ranklist:
        if item == item_id:
            return 1
    return 0


def getNDCG(ranklist, item_id):
    for i, item in enumerate(ranklist):
        if item == item_id:
            return math.log(2) / math.log(i+2)
    return 0
