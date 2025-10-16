def main():
    import pandas as pd
    from soco.base_env import SOCOTestEnv
    from soco.algorithms.robd import R_OBD_L2
    from soco.algorithms.lstm import LSTM
    from soco.algorithms.ec_l2o_step import EC_L2O_STEP
    from soco.algorithms.ec_l2o import EC_L2O
    from analysis import plot_all
    from soco.base_env import OracleEnv

    df = pd.read_csv("data/train.csv")
    train = df.iloc[:, 2:].values.astype(float)

    df = pd.read_csv("data/test.csv")
    test = df.iloc[:, 2:].values.astype(float)

    m = 1.0

    robd = R_OBD_L2(m=m)
    robd_env = SOCOTestEnv(robd, m=m)

    lstm = LSTM(m=m, hidden=16, num_lstm_layers=1,
                dropout=0.0, lr=1e-3)
    lstm.fit(train, epochs=8, unroll=12)

    lstm_env = SOCOTestEnv(lstm, m=m)

    ec_step = EC_L2O_STEP(m=m, hidden=64, num_lstm_layers=1,
                          dropout=0.0, lr=1e-3,
                          theta=0.5, u=0.5)
    ec_step.fit(train, epochs=30)

    ec_step_env = SOCOTestEnv(ec_step, m=m)

    ec = EC_L2O(m=m, hidden=64, num_lstm_layers=1,
                dropout=0.0, lr=1e-3,
                theta=0.5, u=0.5)
    ec.fit(train, epochs=30)

    ec_env = SOCOTestEnv(ec, m=m)

    robd_env.run(test)
    lstm_env.run(test)
    ec_step_env.run(test)
    ec_env.run(test)

    oracle_env = OracleEnv(m=m)
    oracle_env.run(test)

    histories = {
        "L2 R-OBD": robd_env.cum_history,
        "LSTM": lstm_env.cum_history,
        "EC-L2O_step": ec_step_env.cum_history,
        "EC-L2O": ec_env.cum_history,
        "Oracle": oracle_env.cum_history
    }
    plot_all(histories)


if __name__ == "__main__":
    main()
