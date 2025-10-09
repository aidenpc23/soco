def main():
    import pandas as pd
    from soco.base_env import SOCOEnvironment
    from soco.algorithms.robd import R_OBD_L2
    from analysis import plot_all

    df = pd.read_csv("data/train.csv")
    y = df.iloc[:, 2:].values.astype(float)

    m = 5.0
    algo = R_OBD_L2(m=m)
    robd_env = SOCOEnvironment(algo, m=m)

    robd_env.run(y)

    histories = {"L2 R-OBD": robd_env.cum_history}
    plot_all(histories)


if __name__ == "__main__":
    main()
