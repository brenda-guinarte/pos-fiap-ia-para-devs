"""Basic tests for data loading and modeling pipelines."""

from src.train import TARGET, build_models, load_data


def test_load_data_removes_identifiers_and_preserves_target() -> None:
    X, y = load_data()

    assert len(X) == len(y) == 541
    assert TARGET not in X.columns
    assert "Nº sequencial" not in X.columns
    assert "Nº do prontuário" not in X.columns
    assert "Idade (anos)" in X.columns
    assert set(y.unique()) == {0, 1}


def test_all_models_fit_and_predict() -> None:
    X, y = load_data()
    sample_X, sample_y = X.iloc[:100], y.iloc[:100]

    for model in build_models(X).values():
        model.fit(sample_X, sample_y)
        predictions = model.predict(sample_X.iloc[:5])
        assert len(predictions) == 5
