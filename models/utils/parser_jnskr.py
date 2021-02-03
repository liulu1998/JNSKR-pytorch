import argparse


def parse_JNSKR_args():
    parser = argparse.ArgumentParser(description="Run JNSKR")

    parser.add_argument('--data_name', nargs='?', default='amazon-book',
                        help='Choose a dataset from {amazon-book, yelp2018}')
    parser.add_argument('--pretrain', type=int, default=0, help="whether use pretrained model")
    parser.add_argument('--evaluate_every', type=int, default=5,
                        help='Interval of evaluation.')
    parser.add_argument('--batch_size', nargs='?', type=int, default=256,
                        help='batch_size')
    parser.add_argument('--epochs', type=int, default=105,
                        help='Number of epochs.')
    parser.add_argument('--embed_size', type=int, default=64,
                        help='Embedding size.')
    parser.add_argument('--lr', type=float, default=0.05,
                        help='Learning rate.')
    parser.add_argument('--coefficient', type=float, default=[1.0, 0.01],
                        help='weight of multi-task, CF and KG')
    parser.add_argument('--lambda_bilinear', type=float, default=[1, 1e-5],
                        help='weight of L2 regularization, CF and KG')
    # for amazon-book, dropout_kg = 0.2
    # for yelp-2018, dropout_kg = 0.1
    parser.add_argument('--dropout_kg', type=float, default=0.2,
                        help='Dropout ratio of KG Embedding')
    parser.add_argument('--dropout_cf', type=float, default=0.3,
                        help='Dropout ratio of CF')
    """
    c0 and c1 determine the overall weight of non-observed instances in implicit feedback data.
    Specifically, c0 is for the recommendation task and c1 is for the knowledge embedding task.
    """
    # for amazon-book, c0=300, c1=600
    # for yelp-2018, c0=1000, c1=7000
    parser.add_argument('--c0', type=float, default=300,
                        help='initial weight of non-observed data')
    parser.add_argument('--c1', type=float, default=600,
                        help='initial weight of non-observed knowledge data')

    parser.add_argument('--p', type=float, default=0.5,
                        help='significance level of weight')
    parser.add_argument('--sparsity', type=float, default=0,
                        help='sparsity test')

    parser.add_argument('--stop_patience', type=int, default=20,
                        help='patience of early-stopping')

    parser.add_argument('--test_batch_size', type=int, default=128,
                        help='batch size of evaluation')

    parser.add_argument('--Ks', nargs='?', default='[10, 20, 40]',
                        help='evaluation top K')

    args = parser.parse_args()

    # data_root
    args.data_root = "./data"

    # save training log
    save_dir = f"./log/JNSKR/{args.data_name}/emb_size{args.embed_size}_lr{args.lr}/"
    args.save_dir = save_dir
    return args
