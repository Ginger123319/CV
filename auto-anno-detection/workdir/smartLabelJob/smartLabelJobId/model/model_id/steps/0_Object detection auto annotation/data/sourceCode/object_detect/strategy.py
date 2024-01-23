
def query(df_input):
    cnt = df_input.shape[0]
    df_input["isHard"] = 0
    df_input.loc[0:cnt:2, "isHard"] = 1
    return df_input
