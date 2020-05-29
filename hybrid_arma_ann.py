def train_and_pre_ARIMA(p,d,q):
    pass

def train_and_pre_ANN():
    pass

def total_predict():
    pass

if __name__  == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--mode',default='both',type=str,help='Run mode')

    args = parser.parse_args()

    if args.mode == 'both':
        pass
    elif args.mode == 'arima':
        pass
    else:
        raise RuntimeError('Mode must be train both or train only arima!')
