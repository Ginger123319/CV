from sklearn.metrics import r2_score, explained_variance_score

if __name__ == '__main__':
    y_true = [[5, 3, 2, 7], [1, 2, 4, 5]]
    y_pred = [[4.9, 3.1, 2, 8], [1, 2, 5, 3]]
    y1_pred = [[5, 3, 2, 7], [1, 2, 4, 5]]

    print(r2_score(y_true, y1_pred))
    print(explained_variance_score(y_true, y1_pred))
