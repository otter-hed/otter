"""Library reproduction enforces the same native charge bound as its archive."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from benchmarks.runners.ion_structure_validation import (
    MAX_RAW_SCREENING_CHARGE_ERROR, validate_screening_charge,
)


@pytest.mark.parametrize("raw", [4.026627542984556, np.nan, np.inf])
def test_unacceptable_native_charge_is_not_hidden_by_dst_renormalization(raw):
    with pytest.raises(RuntimeError, match="Native screening-charge"):
        validate_screening_charge({"zbar_screening_integral_raw": raw,
            "zbar_partition": 4.016565208591169,
            "charge_fix": {"q_scr_used": 4.016565208591169}})


def test_consistent_background_charge_passes_unchanged_bound():
    assert MAX_RAW_SCREENING_CHARGE_ERROR == 7e-3
    validate_screening_charge({"zbar_screening_integral_raw": 4.0175222463438995,
                              "zbar_partition": 4.016565208591169})


def test_both_reproduction_paths_use_default_carbon_precision():
    root = Path(__file__).resolve().parents[1]
    state = dict(state_id="c_starrett_rho20_te50_ti50", element="C",
                 rho_g_cc=20., te_ev=50., ti_ev=50.)
    for relative, function, kwargs in (
        ("benchmarks/runners/regenerate_ion_structure_library.py", "_configuration", {}),
        ("benchmarks/examples/plot_ion_structure_library.py", "workflow_config", {"ion_temperature_ev": 50.}),
    ):
        spec = importlib.util.spec_from_file_location("library_settings_test", root / relative)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        config = getattr(module, function)(state, **kwargs)
        for setting in ("n_points", "scf_dn_tol", "scf_dv_tol",
                        "ext_scf_dn_tol", "ext_scf_dv_tol"):
            assert setting not in config.aa_overrides
        assert config.qoz_linear_n_points == 4096
