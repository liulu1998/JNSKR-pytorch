import os
import time
import torch
import matplotlib.pyplot as plt


def truncated_normal_(tensor, mean=0, std=1):
    """ initialize weight
    """
    with torch.no_grad():
        size = tensor.shape
        tmp = tensor.new_empty(size + (4,)).normal_()
        valid = (tmp < 2) & (tmp > -2)
        ind = valid.max(-1, keepdim=True)[1]
        tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
        tensor.data.mul_(std).add_(mean)
        return tensor


def early_stopping(recall_list, evaluate_every: int, stopping_steps):
    best_recall = max(recall_list)
    best_step = recall_list.index(best_recall)
    if evaluate_every * (len(recall_list) - best_step - 1) >= stopping_steps:
        should_stop = True
    else:
        should_stop = False
    best_epoch = evaluate_every * (best_step + 1)
    return best_recall, best_epoch, should_stop


def checkpoint(epoch, model, optimizer, **kwargs):
    """
    save checkpoint, including model weights and optimizer params
    """
    state = {
        'epoch': epoch,
        # only save parameters without structure
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    info = ''.join(f"-{k}_{v:.4f}" for k, v in kwargs.items())
    file_path = f"checkpoint_epoch-{epoch}{info}.pth"
    torch.save(state, file_path)


def visualize_result(result: dict, show: bool = False):
    time_stamp = str(time.strftime("%Y-%m-%d_%H-%M", time.localtime()))
    save_dir = os.path.join('./log', 'figures/')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    Ks = result['Ks']
    # train loss
    epochs = range(1, result["epochs"] + 1)

    plt.figure(1)
    plt.title('Train Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.plot(epochs, result["train_loss"], 'bo-', label='Train Loss')

    filename = f"train-loss_epoch-{result['epochs']}_" + time_stamp + ".png"
    plt.savefig(os.path.join(save_dir, filename), dpi=400)

    # validation Recall
    evaluate_every = result["evaluate_every"]
    epochs = range(evaluate_every, result["epochs"] + 1, evaluate_every)
    plt.figure(2, figsize=(8, 14))
    plt.title('Val Recall')
    plt.xlabel("Epoch")
    plt.ylabel("Recall")
    # for i, k in enumerate(result['Ks']):
    #     plt.plot(epochs, result['recall'][i], 'o-', label=f"Recall@{k}")
    #     plt.xticks(epochs)
    #     plt.yticks(torch.arange(0.1, 0.25, 0.002))

    # visualize Recall@20
    plt.plot(epochs, result['recall'][1], 'o-', label=f"Recall@{Ks[1]}")
    plt.xticks(epochs)
    plt.yticks(torch.arange(0.14, 0.17, 0.0005))
    plt.legend()
    filename = f"val-recall_epoch-{result['epochs']}_" + time_stamp + ".png"
    plt.savefig(os.path.join(save_dir, filename), dpi=400)

    # validation NDCG
    plt.figure(3, figsize=(8, 12))
    plt.title('Val NDCG')
    plt.xlabel("Epoch")
    plt.ylabel("NDCG")
    # for i, k in enumerate(result['Ks']):
    #     plt.plot(epochs, result['ndcg'][i], 'o-', label=f"NDCG@{k}")
    #     plt.xticks(epochs)
    #     plt.yticks(torch.arange(0.08, 0.15, 0.002))
    plt.plot(epochs, result['ndcg'][1], 'o-', label=f"NDCG@{Ks[1]}")
    plt.xticks(epochs)
    plt.yticks(torch.arange(0.1, 0.13, 0.0005))
    plt.legend()
    filename = f"val-ndcg_epoch-{result['epochs']}_" + time_stamp + ".png"
    plt.savefig(os.path.join(save_dir, filename), dpi=400)

    if show:
        plt.show()
