import ninetoothed.naming as naming


def test_make_constexpr():
    assert naming.make_constexpr(_NAME) == f"ninetoothed_constexpr_prefix_{_NAME}"


def test_make_meta():
    assert naming.make_meta(_NAME) == f"ninetoothed_meta_prefix_{_NAME}"


def test_make_next_power_of_2():
    assert (
        naming.make_next_power_of_2(_NAME)
        == f"ninetoothed_next_power_of_2_prefix_{_NAME}"
    )
    assert (
        naming.make_next_power_of_2(naming.make_constexpr(_NAME))
        == f"ninetoothed_next_power_of_2_prefix_ninetoothed_constexpr_prefix_{_NAME}"
    )


def test_is_constexpr():
    assert naming.is_constexpr(naming.make_constexpr(_NAME))
    assert naming.is_constexpr(naming.make_meta(_NAME))


def test_is_meta():
    assert naming.is_meta(naming.make_meta(_NAME))


def test_is_next_power_of_2():
    assert naming.is_next_power_of_2(naming.make_next_power_of_2(_NAME))
    assert naming.is_next_power_of_2(
        naming.make_next_power_of_2(naming.make_constexpr(_NAME))
    )


def test_remove_prefixes():
    assert naming.remove_prefixes(naming.make_constexpr(_NAME)) == _NAME
    assert naming.remove_prefixes(naming.make_meta(_NAME)) == _NAME
    assert naming.remove_prefixes(naming.make_next_power_of_2(_NAME)) == _NAME
    assert (
        naming.remove_prefixes(
            naming.make_next_power_of_2(naming.make_constexpr(_NAME))
        )
        == _NAME
    )


_NAME = "ninetoothed_name"
