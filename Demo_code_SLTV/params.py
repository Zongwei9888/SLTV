import argparse
def ParseArgs():
    parser = argparse.ArgumentParser(description='Model Params')
    parser.add_argument('--lr', default=0.0005, type=float, help='learning rate')  # tune source:1e-3
    parser.add_argument('--batch', default=4096, type=int, help='batch size')
    parser.add_argument('--tstBat', default=1024, type=int, help='number of users in a testing batch')
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--model_name', type=str, default="SSL_LTV")
    parser.add_argument('--data', default='G1', type=str, help='name of dataset')

    parser.add_argument('--gpu', default='6', type=str, help='indicates which gpu to use')
    parser.add_argument('--temp', default=0.2, type=float, help='indicates which gpu to use')
    parser.add_argument('--task_names', default='ltv3', type=str, help='indicates which task to use')
    parser.add_argument('--mode', default='tune', type=str, help='indicates which task to use')

    return parser.parse_args()


args = ParseArgs()



