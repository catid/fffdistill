from __future__ import annotations


def test_package_imports() -> None:
    import cifar_mamba_fff

    assert cifar_mamba_fff.__version__


def test_optional_official_imports_are_not_faked() -> None:
    from cifar_mamba_fff.models.mamba3_cifar import find_official_mamba3

    try:
        cls = find_official_mamba3()
    except RuntimeError as exc:
        assert "Do not fall back" in str(exc)
    else:
        assert cls.__name__ == "Mamba3"
