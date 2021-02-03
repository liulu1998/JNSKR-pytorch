import os
import random
import multiprocessing
from torch import optim
from torch.utils.data import DataLoader

from models.JNSKR import JNSKR
from models.utils.metrics import *
from models.utils.log_helper import *
from models.utils.parser_jnskr import parse_JNSKR_args
from models.utils.dataset_jnskr import DatasetJNSKR, TestDataset
from models.utils.test import ranklist_by_heapq, get_performance
from models.utils.helper import early_stopping, checkpoint, visualize_result

# visible GPU for pytorch
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

cores = multiprocessing.cpu_count() // 2

test_set = None


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def test_one_user(x):
    global test_set
    Ks = [10, 20, 40]
    # user u's ratings for user u
    rating = x[0]
    # uid
    u = x[1]
    try:
        # user u's items in the training set
        training_items = test_set.train_user_dict[u]
    except Exception:
        training_items = []

    # user u's items in the test set
    user_pos_test = test_set.test_user_dict[u]

    n_items = len(test_set.item_ids)
    all_items = set(range(n_items))

    test_items = list(all_items - set(training_items))

    r, auc = ranklist_by_heapq(user_pos_test, test_items, rating, Ks)

    return get_performance(user_pos_test, r, auc, Ks)


def evaluate(model, users_to_test, item_test, r_test, t_test, K=[10, 20, 40]):
    """
    :param model: JNSKR model
    :param users_to_test: users in test set
    :param item_test: all items in train set and test set
    :param r_test: relations
    :param t_test: tail entities
    :param K: recall@K, NDCG@K
    """
    global test_set
    res = {'recall': np.zeros(len(K)), 'ndcg': np.zeros(len(K))}
    u_batch_size = 128

    test_users = users_to_test
    n_test_users = len(test_users)
    n_user_batchs = n_test_users // u_batch_size + 1

    count = 0

    pool = multiprocessing.Pool(cores)

    for u_batch_id in range(n_user_batchs):
        start = u_batch_id * u_batch_size
        end = (u_batch_id + 1) * u_batch_size

        user_batch = test_users[start: end]

        with torch.no_grad():
            rate_batch = model(
                "predict",
                r_test=r_test,
                t_test=t_test,
                pos_items=item_test,
                users=torch.LongTensor(user_batch).cuda()
            ).squeeze_().cpu().numpy()

        user_batch_rating_uid = zip(rate_batch, user_batch)

        batch_result = pool.map(test_one_user, user_batch_rating_uid)

        count += len(batch_result)

        for re in batch_result:
            res['recall'] += re['recall'] / n_test_users
            res['ndcg'] += re['ndcg'] / n_test_users

    pool.close()
    assert count == n_test_users
    return res


def train(args):
    global test_set
    log_save_id = create_log_id(args.save_dir)
    logging_config(folder=args.save_dir, name='log{:d}'.format(log_save_id), no_console=False)
    logging.info(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # train data
    dataset = DatasetJNSKR(args)

    n_users, n_items, n_relations, n_entities, max_i_u, max_i_r, negative_c, negative_ck \
        = dataset.stat()

    negative_c = negative_c.to(device)
    negative_ck = negative_ck.to(device)

    train_loader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=cores,
        pin_memory=True
    )

    # test data
    item_ids = torch.arange(n_items, dtype=torch.long).to(device)
    test_set = TestDataset(args, item_ids)

    relation_test, tail_test = dataset.prepare_test()
    relation_test = relation_test.to(device)
    tail_test = tail_test.to(device)

    # model and optimizer
    model = JNSKR(
        n_users=n_users, n_items=n_items, n_relations=n_relations, n_entities=n_entities,
        max_i_u=max_i_u, max_i_r=max_i_r,
        negative_c=negative_c, negative_ck=negative_ck,
        args=args
    )
    model.to(device)

    optimizer = optim.Adagrad(
        model.parameters(),
        lr=args.lr,
        initial_accumulator_value=1e-8
    )

    # train
    Ks = eval(args.Ks)
    epochs = args.epochs
    evaluate_every = args.evaluate_every
    patience = args.stop_patience
    best_epoch = -1

    result = {
        "epochs": epochs, "evaluate_every": evaluate_every, "train_loss": [], "val_loss": [],
        'Ks': Ks, 'recall': [], 'ndcg': []
    }
    for i in range(len(Ks)):
        result['recall'].append([])
        result['ndcg'].append([])

    logging.info(f"start to train, {epochs} epochs in all")

    for epoch in range(epochs):
        # train
        model.train()
        batch_train_losses = []

        for d in train_loader:
            item_batch, user_batch, relation_batch, tail_batch = d

            item_batch = item_batch.to(device).long()
            user_batch = user_batch.to(device).long()
            relation_batch = relation_batch.to(device).long()
            tail_batch = tail_batch.to(device).long()

            optimizer.zero_grad()
            tot_loss = model("cal_loss",
                             input_i=item_batch,
                             input_iu=user_batch,
                             input_hr=relation_batch,
                             input_ht=tail_batch
                             )
            tot_loss.backward()
            optimizer.step()
            # save loss on current batch
            batch_train_losses.append(tot_loss.item())
        # <<< train_loader
        avg_train_loss = np.mean(batch_train_losses)
        result["train_loss"].append(avg_train_loss)

        logging.info(f"epoch {epoch + 1} train loss: {avg_train_loss: .6f}")

        # evaluate on test set
        if (epoch + 1) % evaluate_every == 0:
            logging.info(f"epoch {epoch + 1} evaluating ...")
            model.eval()

            val_result = evaluate(
                model=model,
                users_to_test=np.array(list(test_set.test_user_dict.keys())),
                item_test=test_set.item_ids,
                r_test=relation_test,
                t_test=tail_test,
                K=Ks
            )

            recall_info = f"[test] epoch {epoch+1} Recall"
            for i, (k, r) in enumerate(zip(Ks, val_result["recall"])):
                recall_info += f" @{k}: {r}"
                result["recall"][i].append(r)
            logging.info(recall_info)

            ndcg_info = f"[test] epoch {epoch+1}   NDCG"
            for i, (k, r) in enumerate(zip(Ks, val_result["ndcg"])):
                ndcg_info += f" @{k}: {r}"
                result['ndcg'][i].append(r)
            logging.info(ndcg_info)

            # monitor NDCG@10
            best_ndcg, tmp_best_epoch, should_stop = early_stopping(
                result['ndcg'][0],
                evaluate_every=evaluate_every,
                stopping_steps=patience
            )

            if tmp_best_epoch != best_epoch:
                best_epoch = tmp_best_epoch
                logging.info(f"save checkpoint at epoch {epoch+1}")
                checkpoint(
                    epoch+1, model, optimizer,
                    recall=val_result['recall'][-1],
                    ndcg=val_result['ndcg'][-1]
                )

            if should_stop:
                result['epochs'] = epoch + 1
                logging.info(f"early-stop at epoch {epoch+1}")
                break
        # <<< evaluate
    # <<< epochs

    for i, k in enumerate(Ks):
        cur_list = result['recall'][i]
        logging.info(f"max Recall@{k}: {max(cur_list)} at epoch {evaluate_every * cur_list.index(max(cur_list))}")

    for i, k in enumerate(Ks):
        cur_list = result['ndcg'][i]
        logging.info(f"max NDCG@{k}: {max(cur_list)} at epoch {evaluate_every * cur_list.index(max(cur_list))}")

    logging.info(f"best epoch: {best_epoch}")
    visualize_result(result, show=False)
    return model


if __name__ == '__main__':
    set_seed(2021)

    args = parse_JNSKR_args()
    if args.pretrain != 1:
        model = train(args)
    else:
        from models.utils.evaluate import evaluate_torch
        chk = torch.load("checkpoint_epoch-30-recall_0.2196-ndcg_0.1411.pth")
        model = chk["model"]

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # train data
        dataset = DatasetJNSKR(args)

        n_users, n_items, n_relations, n_entities, max_i_u, max_i_r, negative_c, negative_ck \
            = dataset.stat()

        model = JNSKR(
            n_users=n_users, n_items=n_items, n_relations=n_relations, n_entities=n_entities,
            max_i_u=max_i_u, max_i_r=max_i_r,
            negative_c=negative_c, negative_ck=negative_ck,
            args=args
        )
        model.load_state_dict(chk["model"])
        model.to(device)
        model.eval()

        # test data
        item_ids = torch.arange(n_items, dtype=torch.long).to(device)
        test_set = TestDataset(args, item_ids)

        relation_test, tail_test = dataset.prepare_test()
        relation_test = relation_test.to(device)
        tail_test = tail_test.to(device)

        for k in [10, 20, 40]:
            recall, ndcg = evaluate_torch(
                model=model,
                train_user_dict=test_set.train_user_dict,
                test_user_dict=test_set.test_user_dict,
                user_ids_batches=test_set.user_ids_batches,
                item_ids=test_set.item_ids,
                r_test=relation_test,
                t_test=tail_test,
                Ks=k
            )
            print(f"recall@{k}: {recall}  ndcg@{k}: {ndcg}")
