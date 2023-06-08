TASK_TO_SENTENCE_KEY = {
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst-2": ("sentence", None),
}

TRAIN_SPLIT_NAME = "train"
TASK_TO_VAL_SPLIT_NAME = {
    "mrpc": "validation",
    "qnli": "validation",
    "qqp": "validation",
    "rte": "validation",
    "sst-2": "validation",
    "sts-b": "validation",
    "cola": "validation",
    "mnli": "validation"
}
