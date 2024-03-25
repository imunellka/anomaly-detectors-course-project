import time

from tqdm import tqdm
import numpy as np

from evaluate.evaluation import evaluate
from utils.backprop import backprop


def training(model, trainD, test_final, label, num_epochs, optimizer,scheduler):
    epoch = 0
    start = time()
    f1_scores = []
    accuracy_list =[]
    roc_scores =[]
    for epoch in tqdm(list(range(0, num_epochs))):
        print(epoch)
        lossT, lr = backprop(epoch, model, trainD, trainD, optimizer, scheduler)

        accuracy_list.append((lossT, lr))

        loss0, recons = backprop(0, model, test_final, test_final, optimizer, scheduler, training=False)
        loss_w = loss0.mean(axis=2)

        # wandb.log({
        #     'sum_loss_train': lossT,
        #     'epoch': epoch
        # }, step=epoch)

        loss0 = loss0.reshape(-1,len(label))
        lossFinal = np.mean(np.array(loss0), axis=0)
        labelsFinal = label
        result, _, _ = evaluate(lossFinal, labelsFinal)
        result_roc = result["ROC/AUC"]
        result_f1 = result["f1"]
        roc_scores.append(result_roc)
        f1_scores.append(result_f1)
        # wandb.log({'roc': result_roc, 'f1': result_f1})

