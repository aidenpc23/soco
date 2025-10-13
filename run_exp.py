def main():
    import pandas as pd
    from soco.base_env import SOCOEnvironment
    from soco.algorithms.robd import R_OBD_L2
    from soco.algorithms.lstm import LSTM
    from analysis import plot_all

    df = pd.read_csv("data/train.csv")
    train = df.iloc[:, 2:].values.astype(float)

    df = pd.read_csv("data/test.csv")
    test = df.iloc[:, 2:].values.astype(float)

    m = 1.0

    robd = R_OBD_L2(m=m)
    robd_env = SOCOEnvironment(robd, m=m)

    lstm = LSTM(m=m, hidden=12, num_lstm_layers=1,
                dropout=0.0, input_window=1, lr=1e-3)
    lstm.fit(train, epochs=40, unroll=12)

    lstm_env = SOCOEnvironment(lstm, m=m)

    robd_env.run(test)
    lstm_env.run(test)

    histories = {
        "L2 R-OBD": robd_env.cum_history,
        "LSTM": lstm_env.cum_history
    }
    plot_all(histories)


if __name__ == "__main__":
    main()
