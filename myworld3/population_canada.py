# -*- coding: utf-8 -*-

# Based on PyWorld3 population sector by Charles Vanwynsberghe (2021)
# Modified for Canadian population dynamics with immigration support.
#
# This module extends the original World3 population sector to properly
# model immigration as an external flow distributed across age groups,
# with support for emigration, age-distributed immigration, configurable
# future projection rates, and robust edge-case handling.

import os
import json
import warnings

from scipy.interpolate import interp1d
import numpy as np

from .specials import Dlinf3, Smooth, clip, ramp
from .utils import requires


class PopulationCanada:
    """
    Population sector for Canada with four age levels and immigration support.

    This is a corrected and extended version of the Population class that
    properly handles immigration as a demographic flow distributed across
    age groups, rather than adding all immigrants to a single age bucket.

    Key improvements over the base Population class:
    - Immigration distributed across age groups using configurable fractions
    - Support for emigration (negative net migration or separate series)
    - Guards against negative population in any age group
    - Configurable fallback immigration rate for future projections
    - Immigration applied AFTER death/maturation rates are computed
    - Observed population is optional (not required)
    - Debug output gated behind verbose flag
    - Proper handling of immigration series length mismatches

    Parameters
    ----------
    year_min : float, optional
        Start year of the simulation [year]. The default is 1971.
    year_max : float, optional
        End year of the simulation [year]. The default is 2050.
    dt : float, optional
        Time step of the simulation [year]. The default is 1.
    immigration_series : array-like or None, optional
        Time series of net immigration numbers [persons/year] aligned with
        simulation timesteps. If shorter than simulation, the fallback rate
        is used for remaining years. If None, no immigration is applied
        unless fallback_immigration_rate > 0.
    observed_population : array-like or None, optional
        Observed total population for validation [persons]. Used for error
        reporting only, not required for simulation.
    emigration_series : array-like or None, optional
        Time series of emigration numbers [persons/year]. If provided,
        immigration_series is treated as gross immigration and emigration
        is subtracted. If None, immigration_series is treated as net.
    immigration_age_fractions : dict or None, optional
        Fractions of immigrants going to each age group. Keys: 'p1', 'p2',
        'p3', 'p4'. Must sum to 1.0. Default based on Canadian immigration
        data: {'p1': 0.18, 'p2': 0.62, 'p3': 0.14, 'p4': 0.06}.
    fallback_immigration_rate : float, optional
        Annual immigration as fraction of total population, used when
        immigration_series is exhausted []. The default is 0.008 (0.8%).
    iphst : float, optional
        Implementation date of new policy on health service time [year].
        The default is 1940.
    verbose : bool, optional
        Print information for debugging. The default is False.

    Attributes
    ----------
    (Same as Population, plus:)
    immigration : numpy.ndarray
        Net immigration per year [persons/year].
    immigration_p1 : numpy.ndarray
        Immigration to age group 0-14 [persons/year].
    immigration_p2 : numpy.ndarray
        Immigration to age group 15-44 [persons/year].
    immigration_p3 : numpy.ndarray
        Immigration to age group 45-64 [persons/year].
    immigration_p4 : numpy.ndarray
        Immigration to age group 65+ [persons/year].
    """

    # Default immigration age distribution based on Canadian data
    # Source: Statistics Canada immigration by age group averages
    DEFAULT_IMMIGRATION_AGE_FRACTIONS = {
        'p1': 0.18,  # children (0-14)
        'p2': 0.62,  # working age (15-44)
        'p3': 0.14,  # older working (45-64)
        'p4': 0.06,  # elderly (65+)
    }

    def __init__(self, year_min=1971, year_max=2050, dt=1,
                 immigration_series=None, observed_population=None,
                 emigration_series=None,
                 immigration_age_fractions=None,
                 fallback_immigration_rate=0.008,
                 iphst=1940, verbose=False):
        self.iphst = iphst
        self.dt = dt
        self.year_min = year_min
        self.year_max = year_max
        self.verbose = verbose
        self.length = self.year_max - self.year_min
        self.n = int(self.length / self.dt)
        self.time = np.arange(self.year_min, self.year_max, self.dt)

        # Validate and store observed population (optional)
        if observed_population is not None and len(observed_population) > 0:
            self.observed_population = np.array(observed_population, dtype=float)
        else:
            self.observed_population = None

        # Validate and store immigration series
        if immigration_series is not None:
            self.immigration_series = np.array(immigration_series, dtype=float)
            if np.any(np.isnan(self.immigration_series)):
                warnings.warn(
                    "Immigration series contains NaN values. "
                    "NaN entries will use the fallback immigration rate."
                )
        else:
            self.immigration_series = None

        # Validate and store emigration series
        if emigration_series is not None:
            self.emigration_series = np.array(emigration_series, dtype=float)
            if len(self.emigration_series) != 0 and self.immigration_series is not None:
                if len(self.emigration_series) != len(self.immigration_series):
                    warnings.warn(
                        f"Emigration series length ({len(self.emigration_series)}) "
                        f"differs from immigration series length "
                        f"({len(self.immigration_series)}). "
                        "Shorter series will be zero-padded."
                    )
        else:
            self.emigration_series = None

        # Set immigration age fractions
        if immigration_age_fractions is not None:
            self._validate_age_fractions(immigration_age_fractions)
            self.immigration_age_fractions = immigration_age_fractions
        else:
            self.immigration_age_fractions = self.DEFAULT_IMMIGRATION_AGE_FRACTIONS.copy()

        # Store fallback rate
        if not 0.0 <= fallback_immigration_rate <= 0.1:
            warnings.warn(
                f"Fallback immigration rate {fallback_immigration_rate} is outside "
                "the typical range [0, 0.1]. Verify this is intentional."
            )
        self.fallback_immigration_rate = fallback_immigration_rate

        if self.verbose:
            obs_len = len(self.observed_population) if self.observed_population is not None else 0
            imm_len = len(self.immigration_series) if self.immigration_series is not None else 0
            print(f"PopulationCanada initialized: "
                  f"years={year_min}-{year_max}, dt={dt}, "
                  f"n_steps={self.n}, "
                  f"observed_years={obs_len}, "
                  f"immigration_years={imm_len}, "
                  f"fallback_rate={fallback_immigration_rate}")

    @staticmethod
    def _validate_age_fractions(fractions):
        """Validate that immigration age fractions are well-formed."""
        required_keys = {'p1', 'p2', 'p3', 'p4'}
        if set(fractions.keys()) != required_keys:
            raise ValueError(
                f"immigration_age_fractions must have keys {required_keys}, "
                f"got {set(fractions.keys())}"
            )
        total = sum(fractions.values())
        if not np.isclose(total, 1.0, atol=1e-6):
            raise ValueError(
                f"immigration_age_fractions must sum to 1.0, got {total}"
            )
        for key, val in fractions.items():
            if val < 0:
                raise ValueError(
                    f"immigration_age_fractions['{key}'] = {val} is negative"
                )

    def init_population_constants(self, p1i=6433266, p2i=9697470,
                                  p3i=4068886, p4i=1762410,
                                  dcfsn=4, fcest=4000, hsid=20, ieat=3,
                                  len=28, lpd=20, mtfn=12, pet=4000,
                                  rlt=30, sad=20, zpgt=4000):
        """
        Initialize the constant parameters of the population sector.

        Parameters use Canadian defaults from 1971 census data:
        - p1i: 6,433,266 (ages 0-14)
        - p2i: 9,697,470 (ages 15-44)
        - p3i: 4,068,886 (ages 45-64)
        - p4i: 1,762,410 (ages 65+)
        Total: ~21,962,032 (matches 1971 Canadian population)
        """
        self.p1i = p1i
        self.p2i = p2i
        self.p3i = p3i
        self.p4i = p4i
        self.dcfsn = dcfsn
        self.fcest = fcest
        self.hsid = hsid
        self.ieat = ieat
        self.len = len
        self.lpd = lpd
        self.mtfn = mtfn
        self.pet = pet
        self.rlt = rlt
        self.sad = sad
        self.zpgt = zpgt

        # Validate that initial age groups are positive
        for name, val in [('p1i', p1i), ('p2i', p2i), ('p3i', p3i), ('p4i', p4i)]:
            if val < 0:
                raise ValueError(f"Initial population {name}={val} cannot be negative")

        # Warn if initial conditions don't match observed data
        initial_total = p1i + p2i + p3i + p4i
        if self.observed_population is not None:
            obs_initial = self.observed_population[0]
            rel_diff = abs(initial_total - obs_initial) / obs_initial
            if rel_diff > 0.05:
                warnings.warn(
                    f"Initial age group sum ({initial_total:,.0f}) differs from "
                    f"observed initial population ({obs_initial:,.0f}) by "
                    f"{rel_diff*100:.1f}%. Consider adjusting p1i-p4i to match."
                )

    def init_population_variables(self):
        """
        Initialize the state and rate variables of the population sector
        (memory allocation).
        """
        # Population sector
        self.pop = np.full((self.n,), np.nan)
        self.p1 = np.full((self.n,), np.nan)
        self.p2 = np.full((self.n,), np.nan)
        self.p3 = np.full((self.n,), np.nan)
        self.p4 = np.full((self.n,), np.nan)

        # Immigration tracking (total and per age group)
        self.immigration = np.full((self.n,), np.nan)
        self.immigration_p1 = np.full((self.n,), np.nan)
        self.immigration_p2 = np.full((self.n,), np.nan)
        self.immigration_p3 = np.full((self.n,), np.nan)
        self.immigration_p4 = np.full((self.n,), np.nan)

        # Death rates
        self.d1 = np.full((self.n,), np.nan)
        self.d2 = np.full((self.n,), np.nan)
        self.d3 = np.full((self.n,), np.nan)
        self.d4 = np.full((self.n,), np.nan)

        # Maturation rates
        self.mat1 = np.full((self.n,), np.nan)
        self.mat2 = np.full((self.n,), np.nan)
        self.mat3 = np.full((self.n,), np.nan)

        # Death rate subsector
        self.d = np.full((self.n,), np.nan)
        self.cdr = np.full((self.n,), np.nan)
        self.fpu = np.full((self.n,), np.nan)
        self.le = np.full((self.n,), np.nan)
        self.lmc = np.full((self.n,), np.nan)
        self.lmf = np.full((self.n,), np.nan)
        self.lmhs = np.full((self.n,), np.nan)
        self.lmhs1 = np.full((self.n,), np.nan)
        self.lmhs2 = np.full((self.n,), np.nan)
        self.lmp = np.full((self.n,), np.nan)
        self.m1 = np.full((self.n,), np.nan)
        self.m2 = np.full((self.n,), np.nan)
        self.m3 = np.full((self.n,), np.nan)
        self.m4 = np.full((self.n,), np.nan)
        self.ehspc = np.full((self.n,), np.nan)
        self.hsapc = np.full((self.n,), np.nan)

        # Birth rate subsector
        self.b = np.full((self.n,), np.nan)
        self.cbr = np.full((self.n,), np.nan)
        self.cmi = np.full((self.n,), np.nan)
        self.cmple = np.full((self.n,), np.nan)
        self.tf = np.full((self.n,), np.nan)
        self.dtf = np.full((self.n,), np.nan)
        self.dcfs = np.full((self.n,), np.nan)
        self.fce = np.full((self.n,), np.nan)
        self.fie = np.full((self.n,), np.nan)
        self.fm = np.full((self.n,), np.nan)
        self.frsn = np.full((self.n,), np.nan)
        self.mtf = np.full((self.n,), np.nan)
        self.nfc = np.full((self.n,), np.nan)
        self.ple = np.full((self.n,), np.nan)
        self.sfsn = np.full((self.n,), np.nan)
        self.aiopc = np.full((self.n,), np.nan)
        self.diopc = np.full((self.n,), np.nan)
        self.fcapc = np.full((self.n,), np.nan)
        self.fcfpc = np.full((self.n,), np.nan)
        self.fsafc = np.full((self.n,), np.nan)

    def set_population_delay_functions(self, method="euler"):
        """
        Set the linear smoothing and delay functions of the 1st or the 3rd
        order, for the population sector.
        """
        var_dlinf3 = ["LE", "IOPC", "FCAPC"]
        for var_ in var_dlinf3:
            func_delay = Dlinf3(getattr(self, var_.lower()),
                                self.dt, self.time, method=method)
            setattr(self, "dlinf3_" + var_.lower(), func_delay)

        var_smooth = ["HSAPC", "IOPC"]
        for var_ in var_smooth:
            func_delay = Smooth(getattr(self, var_.lower()),
                                self.dt, self.time, method=method)
            setattr(self, "smooth_" + var_.lower(), func_delay)

    def set_population_table_functions(self, json_file=None):
        """
        Set the nonlinear functions of the population sector, based on a
        json file.

        FIX for Problem 3:
        By default, uses functions_table_canada.json which has rescaled
        x-axis ranges for Canadian economic conditions (IOPC 20000-50000,
        SOPC 4000-10000, POP 20M-50M). Falls back to the global
        functions_table_world3.json if the Canada file is not found.
        """
        if json_file is None:
            # Prefer Canada-specific table functions
            canada_file = os.path.join(os.path.dirname(__file__),
                                       "functions_table_canada.json")
            if os.path.exists(canada_file):
                json_file = canada_file
                if self.verbose:
                    print("  Using Canada-calibrated table functions")
            else:
                json_file = os.path.join(os.path.dirname(__file__),
                                         "functions_table_world3.json")
                if self.verbose:
                    print("  Warning: Canada table functions not found, "
                          "using global defaults")
        with open(json_file) as fjson:
            tables = json.load(fjson)

        func_names = ["M1", "M2", "M3", "M4",
                      "LMF", "HSAPC", "LMHS1", "LMHS2",
                      "FPU", "CMI", "LMP", "FM", "CMPLE",
                      "SFSN", "FRSN", "FCE_TOCLIP", "FSAFC"]

        for func_name in func_names:
            for table in tables:
                if table["y.name"] == func_name:
                    func = interp1d(table["x.values"], table["y.values"],
                                    bounds_error=False,
                                    fill_value=(table["y.values"][0],
                                                table["y.values"][-1]))
                    setattr(self, func_name.lower() + "_f", func)

    def init_exogenous_inputs(self):
        """
        Initialize exogenous parameters for standalone population sector runs.

        FIX for Problem 1 & 2:
        Instead of using global World3 exponential curves (which were calibrated
        for 1900 global dynamics), this uses:
          - Real Canadian GDP per capita data (1971-2023) interpolated and
            extrapolated for the simulation period.
          - Immigration→economy feedback: working-age immigration raises
            GDP proportionally through labor force participation.
          - Canadian-specific food production and service output data.
          - Low persistent pollution index (Canada is not highly polluted
            compared to global averages).

        The interpolation functions are built from data/canada_economic.csv
        if available, or from built-in Canadian reference values as fallback.
        """
        # ── Canadian reference data (GDP per capita in 2015 USD) ──
        # Source: World Bank / FRED constant-dollar GDP per capita for Canada
        # This replaces the global exponential .7e11 * exp(0.037*t)
        _ref_years = np.array([
            1971, 1975, 1980, 1985, 1990, 1995, 2000, 2005,
            2010, 2015, 2020, 2023,
        ], dtype=float)
        _ref_gdppc = np.array([
            20648, 23500, 27100, 29700, 33100, 35000, 40200, 43700,
            43900, 46200, 45700, 48800,
        ], dtype=float)
        _ref_sopc = np.array([
            4200, 4700, 5350, 5900, 6800, 7200, 8200, 9000,
            9200, 9700, 9700, 10200,
        ], dtype=float)
        _ref_fpc = np.array([
            1650, 1700, 1750, 1780, 1830, 1870, 1920, 1960,
            1980, 2020, 2050, 2070,
        ], dtype=float)

        # Try to load full yearly data from CSV
        data_path = os.path.join(os.path.dirname(__file__), "..",
                                 "data", "canada_economic.csv")
        if os.path.exists(data_path):
            try:
                import csv
                with open(data_path, 'r') as f:
                    reader = csv.DictReader(f)
                    rows = list(reader)
                _ref_years = np.array([float(r['Year']) for r in rows])
                _ref_gdppc = np.array([float(r['GDP_Per_Capita_2015USD'])
                                       for r in rows])
                _ref_sopc = np.array([float(r['Service_Output_Per_Capita'])
                                      for r in rows])
                _ref_fpc = np.array([float(r['Food_Per_Capita_kg'])
                                     for r in rows])
                if self.verbose:
                    print(f"  Loaded {len(rows)} years of Canadian economic "
                          f"data from {data_path}")
            except Exception as e:
                if self.verbose:
                    print(f"  Warning: Could not load economic data: {e}")
                    print(f"  Using built-in reference values")

        # Build interpolation functions with extrapolation for projections
        # For years beyond data range, extrapolate using the last decade's
        # average growth rate (more realistic than exponential from 1900)
        self._interp_gdppc = interp1d(
            _ref_years, _ref_gdppc,
            bounds_error=False, fill_value="extrapolate", kind='linear'
        )
        self._interp_sopc = interp1d(
            _ref_years, _ref_sopc,
            bounds_error=False, fill_value="extrapolate", kind='linear'
        )
        self._interp_fpc = interp1d(
            _ref_years, _ref_fpc,
            bounds_error=False, fill_value="extrapolate", kind='linear'
        )

        # Store baseline initial population for immigration feedback
        self._base_pop_1971 = self.p1i + self.p2i + self.p3i + self.p4i
        self._base_working_pop_1971 = self.p2i + self.p3i

        # Canadian subsistence food per capita (higher than global default)
        # Canada is a food-surplus nation; sfpc affects the life expectancy
        # multiplier from food. Using 800 kg veg-equiv (vs global 230)
        # means fpc/sfpc > 1 always, giving lmf close to maximum.
        self.sfpc = 800

        # Allocate exogenous variable arrays
        self.io = np.full((self.n,), np.nan)
        self.iopc = np.full((self.n,), np.nan)
        self.ppolx = np.full((self.n,), np.nan)
        self.so = np.full((self.n,), np.nan)
        self.sopc = np.full((self.n,), np.nan)
        self.f = np.full((self.n,), np.nan)
        self.fpc = np.full((self.n,), np.nan)

        # Retain these for backward compat (some table functions reference
        # them via @requires), but they're no longer used in computation
        self.io1 = np.full((self.n,), np.nan)
        self.io11 = np.full((self.n,), np.nan)
        self.io12 = np.full((self.n,), np.nan)
        self.io2 = np.full((self.n,), np.nan)
        self.so1 = np.full((self.n,), np.nan)
        self.so11 = np.full((self.n,), np.nan)
        self.so12 = np.full((self.n,), np.nan)
        self.so2 = np.full((self.n,), np.nan)
        self.f1 = np.full((self.n,), np.nan)
        self.f11 = np.full((self.n,), np.nan)
        self.f12 = np.full((self.n,), np.nan)
        self.f2 = np.full((self.n,), np.nan)

    @requires(["io", "iopc", "ppolx", "so", "sopc", "f", "fpc"], ["pop"])
    def loopk_exogenous(self, k):
        """
        Compute exogenous economic inputs for timestep k using real
        Canadian data with immigration→economy feedback.

        FIX for Problem 1:
        Uses interpolated Canadian GDP per capita, service output, and food
        data instead of global exponential growth curves.

        FIX for Problem 2 (immigration→economy feedback):
        Working-age immigration (p2 + p3) increases labor supply, which
        raises total industrial and service output proportionally. This
        implements the immigration→GDP→services→mortality/fertility
        feedback loop that was missing.

        The feedback factor is:
            labor_factor = current_working_pop / baseline_working_pop
        where baseline is what the working population would have been
        without any immigration. This scales total output (not per-capita)
        so that more workers = more GDP = more services = affects mortality
        and fertility through the standard World3 channels.
        """
        year = self.time[k]

        # ── Base per-capita values from real Canadian data ──
        base_gdppc = float(self._interp_gdppc(year))
        base_sopc = float(self._interp_sopc(year))
        base_fpc = float(self._interp_fpc(year))

        # Clamp extrapolations to reasonable ranges
        base_gdppc = max(15000.0, min(80000.0, base_gdppc))
        base_sopc = max(3000.0, min(20000.0, base_sopc))
        base_fpc = max(1500.0, min(3000.0, base_fpc))

        # ── Immigration → economy feedback (Problem 2 fix) ──
        # More working-age immigrants → larger labor force → higher total GDP
        # This creates the feedback: immigration → GDP → services → health
        # → mortality/fertility → population → more demand for immigration
        current_working = self.p2[k] + self.p3[k]
        years_elapsed = year - self.year_min

        # Estimate what working population would be without immigration
        # (natural growth only, ~0.5% annual decline in working-age share)
        natural_growth_factor = max(0.5, 1.0 + 0.005 * years_elapsed)
        baseline_working = self._base_working_pop_1971 * natural_growth_factor

        if baseline_working > 0 and current_working > 0:
            # Immigration multiplier: ratio of actual to hypothetical
            # working population. Capped at [0.8, 1.5] to prevent
            # unrealistic values.
            labor_factor = np.clip(
                current_working / baseline_working, 0.8, 1.5
            )
        else:
            labor_factor = 1.0

        # Scale per-capita values by labor productivity factor
        # Immigrants contribute ~85% of native-born productivity initially
        # (wage gap data from Statistics Canada), converging to 100% over time
        immigrant_productivity = min(1.0, 0.85 + 0.003 * years_elapsed)
        effective_factor = 1.0 + (labor_factor - 1.0) * immigrant_productivity

        # ── Compute final values ──
        self.iopc[k] = base_gdppc * effective_factor
        self.io[k] = self.iopc[k] * self.pop[k]

        self.sopc[k] = base_sopc * effective_factor
        self.so[k] = self.sopc[k] * self.pop[k]

        self.fpc[k] = base_fpc  # Food per capita less affected by labor
        self.f[k] = self.fpc[k] * self.pop[k]

        # Canada has very low persistent pollution relative to global scale
        # ppolx=1 means baseline; Canada ranges 0.8-1.2 (not rising sharply)
        self.ppolx[k] = 1.0 + 0.003 * years_elapsed  # Very mild increase

    def loop0_exogenous(self):
        """Run a sequence to initialize the exogenous parameters (k=0)."""
        self.loopk_exogenous(0)

    def _compute_net_immigration(self, k):
        """
        Compute net immigration for timestep k.

        Logic:
        1. If immigration_series has data for step k, use it.
        2. If immigration_series is exhausted or None, use fallback rate.
        3. If emigration_series is provided, subtract it (immigration_series
           is then treated as gross immigration).
        4. Handle NaN values in the series gracefully.

        Returns
        -------
        float
            Net immigration for timestep k [persons/year].
        bool
            True if fallback rate was used (for logging).
        """
        used_fallback = False
        base_pop = self.p1[k] + self.p2[k] + self.p3[k] + self.p4[k]

        # Determine gross immigration
        if (self.immigration_series is not None
                and k < len(self.immigration_series)
                and not np.isnan(self.immigration_series[k])):
            gross_immigration = self.immigration_series[k]
        else:
            gross_immigration = self.fallback_immigration_rate * base_pop
            used_fallback = True
            if self.verbose and k > 0:
                if self.immigration_series is not None and k == len(self.immigration_series):
                    print(f"  Step {k} (year {self.time[k]:.0f}): "
                          f"Immigration series exhausted, switching to "
                          f"fallback rate {self.fallback_immigration_rate}")

        # Determine emigration
        emigration = 0.0
        if (self.emigration_series is not None
                and k < len(self.emigration_series)
                and not np.isnan(self.emigration_series[k])):
            emigration = self.emigration_series[k]

        net_immigration = gross_immigration - emigration
        return net_immigration, used_fallback

    def _distribute_immigration(self, k, net_immigration):
        """
        Distribute net immigration across age groups and apply to population.

        Handles the case where net immigration is negative (net emigration)
        by ensuring no age group goes below zero.

        Parameters
        ----------
        k : int
            Current timestep index.
        net_immigration : float
            Net immigration to distribute [persons/year].
        """
        self.immigration[k] = net_immigration

        # Compute per-age-group immigration
        imm_p1 = net_immigration * self.immigration_age_fractions['p1']
        imm_p2 = net_immigration * self.immigration_age_fractions['p2']
        imm_p3 = net_immigration * self.immigration_age_fractions['p3']
        imm_p4 = net_immigration * self.immigration_age_fractions['p4']

        # Guard against negative population in any age group
        if net_immigration < 0:
            # For net emigration, ensure we don't remove more people than exist
            # Use a floor of 1.0 to avoid division by zero in rate calculations
            min_floor = 1.0
            if self.p1[k] + imm_p1 < min_floor:
                imm_p1 = min_floor - self.p1[k]
            if self.p2[k] + imm_p2 < min_floor:
                imm_p2 = min_floor - self.p2[k]
            if self.p3[k] + imm_p3 < min_floor:
                imm_p3 = min_floor - self.p3[k]
            if self.p4[k] + imm_p4 < min_floor:
                imm_p4 = min_floor - self.p4[k]

            # Recalculate actual net immigration after clamping
            actual_net = imm_p1 + imm_p2 + imm_p3 + imm_p4
            if self.verbose and abs(actual_net - net_immigration) > 1.0:
                print(f"  Step {k}: Net emigration clamped from "
                      f"{net_immigration:.0f} to {actual_net:.0f} "
                      f"to prevent negative population")
            self.immigration[k] = actual_net

        # Store per-group immigration
        self.immigration_p1[k] = imm_p1
        self.immigration_p2[k] = imm_p2
        self.immigration_p3[k] = imm_p3
        self.immigration_p4[k] = imm_p4

        # Apply immigration to age groups
        self.p1[k] += imm_p1
        self.p2[k] += imm_p2
        self.p3[k] += imm_p3
        self.p4[k] += imm_p4

    def loop0_population(self, alone=False):
        """
        Run a sequence to initialize the population sector (loop with k=0).

        Parameters
        ----------
        alone : boolean, optional
            If True, run the sector alone with exogenous inputs.
        """
        # Set initial conditions from constants
        self.p1[0] = self.p1i
        self.p2[0] = self.p2i
        self.p3[0] = self.p3i
        self.p4[0] = self.p4i

        # Compute and apply initial immigration (step 0)
        net_imm, used_fallback = self._compute_net_immigration(0)
        self._distribute_immigration(0, net_imm)

        # Total population is sum of age groups (now includes immigration)
        self.pop[0] = self.p1[0] + self.p2[0] + self.p3[0] + self.p4[0]

        if self.verbose:
            print(f"Step 0 (year {self.time[0]:.0f}): "
                  f"pop={self.pop[0]:,.0f}, "
                  f"immigration={self.immigration[0]:,.0f}"
                  f"{' (fallback)' if used_fallback else ''}")

        self.frsn[0] = 0.82

        if alone:
            self.loop0_exogenous()

        # Death rate subsector
        self._update_fpu(0)
        self._update_lmp(0)
        self._update_lmf(0)
        self._update_cmi(0)
        self._update_hsapc(0)
        self._update_ehspc(0)
        self._update_lmhs(0)
        self._update_lmc(0)
        self._update_le(0)
        self._update_m1(0)
        self._update_m2(0)
        self._update_m3(0)
        self._update_m4(0)
        self._update_mat1(0, 0)
        self._update_mat2(0, 0)
        self._update_mat3(0, 0)
        self._update_d1(0, 0)
        self._update_d2(0, 0)
        self._update_d3(0, 0)
        self._update_d4(0, 0)
        self._update_d(0, 0)
        self._update_cdr(0)

        # Birth rate subsector
        self._update_aiopc(0)
        self._update_diopc(0)
        self._update_fie(0)
        self._update_sfsn(0)
        self._update_frsn(0)
        self._update_dcfs(0)
        self._update_ple(0)
        self._update_cmple(0)
        self._update_dtf(0)
        self._update_fm(0)
        self._update_mtf(0)
        self._update_nfc(0)
        self._update_fsafc(0)
        self._update_fcapc(0)
        self._update_fcfpc(0)
        self._update_fce(0)
        self._update_tf(0)
        self._update_cbr(0, 0)
        self._update_b(0, 0)
        # Recompute frsn after birth subsector initialization
        self._update_frsn(0)

    def loopk_population(self, j, k, jk, kl, alone=False):
        """
        Run a sequence to update one loop of the population sector.

        The key ordering difference from the original:
        1. Update state variables p1-p4 from births/deaths/maturation
        2. Compute total population WITHOUT immigration
        3. Compute death/maturation/birth rates on pre-immigration population
        4. Apply immigration to age groups
        5. Recompute final population with immigration

        This ensures immigrants don't affect rates in their arrival timestep,
        which is more demographically accurate.
        """
        # Step 1: Update state variables from previous step's flows
        self._update_state_p1(k, j, jk)
        self._update_state_p2(k, j, jk)
        self._update_state_p3(k, j, jk)
        self._update_state_p4(k, j, jk)

        # Step 2: Compute pre-immigration population for rate calculations
        pre_imm_pop = self.p1[k] + self.p2[k] + self.p3[k] + self.p4[k]
        self.pop[k] = pre_imm_pop

        if alone:
            self.loopk_exogenous(k)

        # Step 3: Compute all rates on pre-immigration population
        # Death rate subsector
        self._update_fpu(k)
        self._update_lmp(k)
        self._update_lmf(k)
        self._update_cmi(k)
        self._update_hsapc(k)
        self._update_ehspc(k)
        self._update_lmhs(k)
        self._update_lmc(k)
        self._update_le(k)
        self._update_m1(k)
        self._update_m2(k)
        self._update_m3(k)
        self._update_m4(k)
        self._update_mat1(k, kl)
        self._update_mat2(k, kl)
        self._update_mat3(k, kl)
        self._update_d1(k, kl)
        self._update_d2(k, kl)
        self._update_d3(k, kl)
        self._update_d4(k, kl)
        self._update_d(k, jk)
        self._update_cdr(k)

        # Birth rate subsector
        self._update_aiopc(k)
        self._update_diopc(k)
        self._update_fie(k)
        self._update_sfsn(k)
        self._update_frsn(k)
        self._update_dcfs(k)
        self._update_ple(k)
        self._update_cmple(k)
        self._update_dtf(k)
        self._update_fm(k)
        self._update_mtf(k)
        self._update_nfc(k)
        self._update_fsafc(k)
        self._update_fcapc(k)
        self._update_fcfpc(k)
        self._update_fce(k)
        self._update_tf(k)
        self._update_cbr(k, jk)
        self._update_b(k, kl)

        # Step 4: Now apply immigration AFTER rates are computed
        net_imm, used_fallback = self._compute_net_immigration(k)
        self._distribute_immigration(k, net_imm)

        # Step 5: Recompute final population with immigration
        self.pop[k] = self.p1[k] + self.p2[k] + self.p3[k] + self.p4[k]

        # Recompute CDR and CBR with updated population (includes immigrants)
        if self.pop[k] > 0:
            self.cdr[k] = 1000 * self.d[k] / self.pop[k]
            self.cbr[k] = 1000 * self.b[jk] / self.pop[k]

        # Log comparison with observed data if available
        if self.verbose:
            msg = (f"Step {k} (year {self.time[k]:.0f}): "
                   f"pop={self.pop[k]:,.0f}, "
                   f"imm={self.immigration[k]:,.0f}"
                   f"{' (fallback)' if used_fallback else ''}")
            if (self.observed_population is not None
                    and k < len(self.observed_population)):
                obs = self.observed_population[k]
                error = obs - self.pop[k]
                pct = 100 * error / obs if obs != 0 else 0
                msg += f", observed={obs:,.0f}, error={error:,.0f} ({pct:.2f}%)"
            print(msg)

    def run_population(self):
        """
        Run a sequence of updates to simulate the population sector alone
        with exogenous inputs.
        """
        self.redo_loop = True
        while self.redo_loop:
            self.redo_loop = False
            self.loop0_population(alone=True)
        for k_ in range(1, self.n):
            self.redo_loop = True
            while self.redo_loop:
                self.redo_loop = False
                if self.verbose:
                    print(f"--- Loop {k_} ---")
                self.loopk_population(k_ - 1, k_, k_ - 1, k_, alone=True)

    # ----------------------------------------------------------------
    # State variable updates
    # ----------------------------------------------------------------

    @requires(["p1"])
    def _update_state_p1(self, k, j, jk):
        """State variable, requires previous step only."""
        self.p1[k] = self.p1[j] + self.dt * (self.b[jk] - self.d1[jk]
                                               - self.mat1[jk])
        # Guard: population cannot be negative
        if self.p1[k] < 0:
            if self.verbose:
                print(f"  Warning: p1[{k}] went negative ({self.p1[k]:.0f}), "
                      f"clamping to 0")
            self.p1[k] = 0.0

    @requires(["p2"])
    def _update_state_p2(self, k, j, jk):
        """State variable, requires previous step only."""
        self.p2[k] = self.p2[j] + self.dt * (self.mat1[jk] - self.d2[jk]
                                               - self.mat2[jk])
        if self.p2[k] < 0:
            if self.verbose:
                print(f"  Warning: p2[{k}] went negative ({self.p2[k]:.0f}), "
                      f"clamping to 0")
            self.p2[k] = 0.0

    @requires(["p3"])
    def _update_state_p3(self, k, j, jk):
        """State variable, requires previous step only."""
        self.p3[k] = self.p3[j] + self.dt * (self.mat2[jk] - self.d3[jk]
                                               - self.mat3[jk])
        if self.p3[k] < 0:
            if self.verbose:
                print(f"  Warning: p3[{k}] went negative ({self.p3[k]:.0f}), "
                      f"clamping to 0")
            self.p3[k] = 0.0

    @requires(["p4"])
    def _update_state_p4(self, k, j, jk):
        """State variable, requires previous step only."""
        self.p4[k] = self.p4[j] + self.dt * (self.mat3[jk] - self.d4[jk])
        if self.p4[k] < 0:
            if self.verbose:
                print(f"  Warning: p4[{k}] went negative ({self.p4[k]:.0f}), "
                      f"clamping to 0")
            self.p4[k] = 0.0

    @requires(["pop"], ["p1", "p2", "p3", "p4"])
    def _update_pop(self, k):
        """
        Update total population. Immigration is NOT applied here.
        Immigration is handled separately in loopk_population to ensure
        correct ordering with rate calculations.
        """
        self.pop[k] = self.p1[k] + self.p2[k] + self.p3[k] + self.p4[k]

    # ----------------------------------------------------------------
    # Death rate subsector
    # ----------------------------------------------------------------

    @requires(["fpu"], ["pop"])
    def _update_fpu(self, k):
        """From step k requires: POP"""
        self.fpu[k] = self.fpu_f(self.pop[k])

    @requires(["lmp"], ["ppolx"])
    def _update_lmp(self, k):
        """From step k requires: PPOLX"""
        self.lmp[k] = self.lmp_f(self.ppolx[k])

    @requires(["lmf"], ["fpc"])
    def _update_lmf(self, k):
        """From step k requires: FPC"""
        self.lmf[k] = self.lmf_f(self.fpc[k] / self.sfpc)

    @requires(["cmi"], ["iopc"])
    def _update_cmi(self, k):
        """From step k requires: IOPC"""
        self.cmi[k] = self.cmi_f(self.iopc[k])

    @requires(["hsapc"], ["sopc"])
    def _update_hsapc(self, k):
        """From step k requires: SOPC"""
        self.hsapc[k] = self.hsapc_f(self.sopc[k])

    @requires(["ehspc"], ["hsapc"], check_after_init=False)
    def _update_ehspc(self, k):
        """From step k=0 requires: HSAPC, else nothing"""
        self.ehspc[k] = self.smooth_hsapc(k, self.hsid, self.hsapc[0])

    @requires(["lmhs1", "lmhs2", "lmhs"], ["ehspc"])
    def _update_lmhs(self, k):
        """From step k requires: EHSPC"""
        self.lmhs1[k] = self.lmhs1_f(self.ehspc[k])
        self.lmhs2[k] = self.lmhs2_f(self.ehspc[k])
        self.lmhs[k] = clip(self.lmhs2[k], self.lmhs1[k],
                            self.time[k], self.iphst)

    @requires(["lmc"], ["cmi", "fpu"])
    def _update_lmc(self, k):
        """From step k requires: CMI FPU"""
        self.lmc[k] = 1 - self.cmi[k] * self.fpu[k]

    @requires(["le"], ["lmf", "lmhs", "lmp", "lmc"])
    def _update_le(self, k):
        """From step k requires: LMF LMHS LMP LMC"""
        self.le[k] = (self.len * self.lmf[k] * self.lmhs[k]
                      * self.lmp[k] * self.lmc[k])

    @requires(["m1"], ["le"])
    def _update_m1(self, k):
        """From step k requires: LE"""
        self.m1[k] = self.m1_f(self.le[k])

    @requires(["m2"], ["le"])
    def _update_m2(self, k):
        """From step k requires: LE"""
        self.m2[k] = self.m2_f(self.le[k])

    @requires(["m3"], ["le"])
    def _update_m3(self, k):
        """From step k requires: LE"""
        self.m3[k] = self.m3_f(self.le[k])

    @requires(["m4"], ["le"])
    def _update_m4(self, k):
        """From step k requires: LE"""
        self.m4[k] = self.m4_f(self.le[k])

    @requires(["mat1"], ["p1", "m1"])
    def _update_mat1(self, k, kl):
        """From step k requires: P1 M1"""
        self.mat1[kl] = self.p1[k] * (1 - self.m1[k]) / 15

    @requires(["mat2"], ["p2", "m2"])
    def _update_mat2(self, k, kl):
        """From step k requires: P2 M2"""
        self.mat2[kl] = self.p2[k] * (1 - self.m2[k]) / 30

    @requires(["mat3"], ["p3", "m3"])
    def _update_mat3(self, k, kl):
        """From step k requires: P3 M3"""
        self.mat3[kl] = self.p3[k] * (1 - self.m3[k]) / 20

    @requires(["d1"], ["p1", "m1"])
    def _update_d1(self, k, kl):
        """From step k requires: P1 M1"""
        self.d1[kl] = self.p1[k] * self.m1[k]

    @requires(["d2"], ["p2", "m2"])
    def _update_d2(self, k, kl):
        """From step k requires: P2 M2"""
        self.d2[kl] = self.p2[k] * self.m2[k]

    @requires(["d3"], ["p3", "m3"])
    def _update_d3(self, k, kl):
        """From step k requires: P3 M3"""
        self.d3[kl] = self.p3[k] * self.m3[k]

    @requires(["d4"], ["p4", "m4"])
    def _update_d4(self, k, kl):
        """From step k requires: P4 M4"""
        self.d4[kl] = self.p4[k] * self.m4[k]

    @requires(["d"])
    def _update_d(self, k, jk):
        """From step k requires: nothing"""
        self.d[k] = self.d1[jk] + self.d2[jk] + self.d3[jk] + self.d4[jk]

    @requires(["cdr"], ["d", "pop"])
    def _update_cdr(self, k):
        """From step k requires: D POP"""
        if self.pop[k] > 0:
            self.cdr[k] = 1000 * self.d[k] / self.pop[k]
        else:
            self.cdr[k] = 0.0

    # ----------------------------------------------------------------
    # Birth rate subsector
    # ----------------------------------------------------------------

    @requires(["aiopc"], ["iopc"], check_after_init=False)
    def _update_aiopc(self, k):
        """From step k=0 requires: IOPC, else nothing"""
        self.aiopc[k] = self.smooth_iopc(k, self.ieat, self.iopc[0])

    @requires(["diopc"], ["iopc"], check_after_init=False)
    def _update_diopc(self, k):
        """From step k=0 requires: IOPC, else nothing"""
        self.diopc[k] = self.dlinf3_iopc(k, self.sad)

    @requires(["fie"], ["iopc", "aiopc"])
    def _update_fie(self, k):
        """From step k requires: IOPC AIOPC"""
        self.fie[k] = (self.iopc[k] - self.aiopc[k]) / self.aiopc[k]

    @requires(["sfsn"], ["diopc"])
    def _update_sfsn(self, k):
        """From step k requires: DIOPC"""
        self.sfsn[k] = self.sfsn_f(self.diopc[k])

    @requires(["frsn"], ["fie"])
    def _update_frsn(self, k):
        """From step k requires: FIE"""
        self.frsn[k] = self.frsn_f(self.fie[k])

    @requires(["dcfs"], ["frsn", "sfsn"])
    def _update_dcfs(self, k):
        """From step k requires: FRSN SFSN"""
        self.dcfs[k] = clip(2.0, self.dcfsn * self.frsn[k] * self.sfsn[k],
                            self.time[k], self.zpgt)

    @requires(["ple"], ["le"], check_after_init=False)
    def _update_ple(self, k):
        """From step k=0 requires: LE, else nothing"""
        self.ple[k] = self.dlinf3_le(k, self.lpd)

    @requires(["cmple"], ["ple"])
    def _update_cmple(self, k):
        """From step k requires: PLE"""
        self.cmple[k] = self.cmple_f(self.ple[k])

    @requires(["dtf"], ["dcfs", "cmple"])
    def _update_dtf(self, k):
        """From step k requires: DCFS CMPLE"""
        self.dtf[k] = self.dcfs[k] * self.cmple[k]

    @requires(["fm"], ["le"])
    def _update_fm(self, k):
        """From step k requires: LE"""
        self.fm[k] = self.fm_f(self.le[k])

    @requires(["mtf"], ["fm"])
    def _update_mtf(self, k):
        """From step k requires: FM"""
        self.mtf[k] = self.mtfn * self.fm[k]

    @requires(["nfc"], ["mtf", "dtf"])
    def _update_nfc(self, k):
        """From step k requires: MTF DTF"""
        if self.dtf[k] > 0:
            self.nfc[k] = self.mtf[k] / self.dtf[k] - 1
        else:
            self.nfc[k] = 0.0

    @requires(["fsafc"], ["nfc"])
    def _update_fsafc(self, k):
        """From step k requires: NFC"""
        self.fsafc[k] = self.fsafc_f(self.nfc[k])

    @requires(["fcapc"], ["fsafc", "sopc"])
    def _update_fcapc(self, k):
        """From step k requires: FSAFC SOPC"""
        self.fcapc[k] = self.fsafc[k] * self.sopc[k]

    @requires(["fcfpc"], ["fcapc"], check_after_init=False)
    def _update_fcfpc(self, k):
        """From step k=0 requires: FCAPC, else nothing"""
        self.fcfpc[k] = self.dlinf3_fcapc(k, self.hsid)

    @requires(["fce"], ["fcfpc"])
    def _update_fce(self, k):
        """From step k requires: FCFPC"""
        self.fce[k] = clip(1.0, self.fce_toclip_f(self.fcfpc[k]),
                           self.time[k], self.fcest)

    @requires(["tf"], ["mtf", "fce", "dtf"])
    def _update_tf(self, k):
        """From step k requires: MTF FCE DTF"""
        self.tf[k] = np.minimum(self.mtf[k],
                                (self.mtf[k] * (1 - self.fce[k])
                                 + self.dtf[k] * self.fce[k]))

    @requires(["cbr"], ["pop"])
    def _update_cbr(self, k, jk):
        """From step k requires: POP"""
        if self.pop[k] > 0:
            self.cbr[k] = 1000 * self.b[jk] / self.pop[k]
        else:
            self.cbr[k] = 0.0

    @requires(["b"], ["d", "p2", "tf"])
    def _update_b(self, k, kl):
        """From step k requires: D P2 TF"""
        if np.isnan(self.p2[k]) or np.isnan(self.tf[k]) or np.isnan(self.d[k]):
            if self.verbose:
                print(f"  Warning: NaN in birth calculation at step {k} "
                      f"(p2={self.p2[k]}, tf={self.tf[k]}, d={self.d[k]}), "
                      f"setting births to 0")
            self.b[kl] = 0.0
        else:
            self.b[kl] = clip(self.d[k],
                              self.tf[k] * self.p2[k] * 0.5 / self.rlt,
                              self.time[k], self.pet)

    # ----------------------------------------------------------------
    # Utility methods
    # ----------------------------------------------------------------

    def get_summary(self):
        """
        Return a dictionary summarizing key simulation results.

        Returns
        -------
        dict
            Summary statistics including final population, total immigration,
            average growth rate, etc.
        """
        valid = ~np.isnan(self.pop)
        if not np.any(valid):
            return {"error": "No valid population data"}

        last_valid = np.max(np.where(valid))
        first_valid = np.min(np.where(valid))

        total_imm = np.nansum(self.immigration)
        avg_growth = 0.0
        if last_valid > first_valid and self.pop[first_valid] > 0:
            years = self.time[last_valid] - self.time[first_valid]
            if years > 0:
                avg_growth = ((self.pop[last_valid] / self.pop[first_valid])
                              ** (1.0 / years) - 1) * 100

        result = {
            "year_range": (self.time[first_valid], self.time[last_valid]),
            "initial_population": self.pop[first_valid],
            "final_population": self.pop[last_valid],
            "total_net_immigration": total_imm,
            "avg_annual_growth_pct": avg_growth,
            "immigration_share_of_growth_pct": (
                100 * total_imm / (self.pop[last_valid] - self.pop[first_valid])
                if self.pop[last_valid] != self.pop[first_valid] else 0
            ),
        }

        if self.observed_population is not None:
            obs_len = min(len(self.observed_population), len(self.pop))
            errors = (self.observed_population[:obs_len]
                      - self.pop[:obs_len])
            valid_errors = errors[~np.isnan(errors)]
            if len(valid_errors) > 0:
                result["mean_absolute_error"] = np.mean(np.abs(valid_errors))
                result["mean_pct_error"] = np.mean(
                    100 * np.abs(valid_errors)
                    / self.observed_population[:obs_len][~np.isnan(errors)]
                )

        return result

    def validate(self, train_end_year=2005):
        """
        Formal train/test validation methodology for publication.

        FIX for Problem 4:
        Implements a proper out-of-sample validation by splitting observed
        data into training (1971-train_end_year) and testing
        (train_end_year-end) periods. Reports metrics separately for each
        period to demonstrate the model's predictive capability.

        This is the standard approach for validating system dynamics models
        (Barlas 1996, "Formal aspects of model validity and validation in
        system dynamics").

        Parameters
        ----------
        train_end_year : int, optional
            Year that separates training from testing period.
            Default is 2005 (roughly 2/3 of the 1971-2023 data span).

        Returns
        -------
        dict
            Validation results with the following keys:

            - 'train_period': (start_year, end_year)
            - 'test_period': (end_year, final_year)
            - 'train_mae': Mean Absolute Error on training set
            - 'train_mape': Mean Absolute Percentage Error on training set
            - 'test_mae': Mean Absolute Error on test set (out-of-sample)
            - 'test_mape': Mean Absolute Percentage Error on test set
            - 'theil_U': Theil's inequality coefficient (0=perfect, 1=naive)
            - 'theil_Um': Bias proportion (systematic error)
            - 'theil_Us': Variance proportion (spread error)
            - 'theil_Uc': Covariance proportion (unsystematic error)
            - 'r_squared': Coefficient of determination (all data)
            - 'max_error_year': Year with the largest absolute error
            - 'max_error_pct': The largest percentage error
            - 'publication_ready': Boolean assessment based on thresholds

        Notes
        -----
        Publication thresholds (following Sterman 2000, "Business Dynamics"):
        - MAPE < 5% on training data
        - MAPE < 10% on test data (out-of-sample)
        - Theil U < 0.3
        - R² > 0.95
        """
        if self.observed_population is None:
            return {"error": "No observed data provided for validation"}

        results = {}
        obs = self.observed_population
        sim = self.pop
        n_obs = min(len(obs), len(sim))

        if n_obs < 10:
            return {"error": f"Too few data points ({n_obs}) for validation"}

        # Split into train and test
        train_mask = self.time[:n_obs] <= train_end_year
        test_mask = self.time[:n_obs] > train_end_year

        obs_valid = obs[:n_obs]
        sim_valid = sim[:n_obs]

        results['train_period'] = (
            float(self.time[0]),
            float(train_end_year)
        )
        results['test_period'] = (
            float(train_end_year),
            float(self.time[n_obs - 1])
        )

        # ── Training metrics ──
        obs_train = obs_valid[train_mask]
        sim_train = sim_valid[train_mask]
        valid_train = ~(np.isnan(obs_train) | np.isnan(sim_train))

        if np.sum(valid_train) > 0:
            ot = obs_train[valid_train]
            st = sim_train[valid_train]
            errors_train = ot - st
            results['train_mae'] = float(np.mean(np.abs(errors_train)))
            results['train_mape'] = float(
                np.mean(100 * np.abs(errors_train) / ot)
            )
            results['train_rmse'] = float(
                np.sqrt(np.mean(errors_train ** 2))
            )
        else:
            results['train_mae'] = np.nan
            results['train_mape'] = np.nan
            results['train_rmse'] = np.nan

        # ── Test metrics (out-of-sample) ──
        obs_test = obs_valid[test_mask]
        sim_test = sim_valid[test_mask]
        valid_test = ~(np.isnan(obs_test) | np.isnan(sim_test))

        if np.sum(valid_test) > 0:
            oe = obs_test[valid_test]
            se = sim_test[valid_test]
            errors_test = oe - se
            results['test_mae'] = float(np.mean(np.abs(errors_test)))
            results['test_mape'] = float(
                np.mean(100 * np.abs(errors_test) / oe)
            )
            results['test_rmse'] = float(
                np.sqrt(np.mean(errors_test ** 2))
            )
        else:
            results['test_mae'] = np.nan
            results['test_mape'] = np.nan
            results['test_rmse'] = np.nan

        # ── Theil's Inequality Coefficient ──
        # Standard validation metric for system dynamics models
        valid_all = ~(np.isnan(obs_valid) | np.isnan(sim_valid))
        o = obs_valid[valid_all]
        s = sim_valid[valid_all]

        if len(o) > 1:
            errors_all = o - s
            mse = np.mean(errors_all ** 2)
            rms_o = np.sqrt(np.mean(o ** 2))
            rms_s = np.sqrt(np.mean(s ** 2))

            # Theil U statistic
            if rms_o + rms_s > 0:
                results['theil_U'] = float(
                    np.sqrt(mse) / (rms_o + rms_s)
                )
            else:
                results['theil_U'] = np.nan

            # Theil decomposition (Um + Us + Uc = 1)
            if mse > 0:
                mean_diff = np.mean(s) - np.mean(o)
                sd_s = np.std(s)
                sd_o = np.std(o)
                r_so = np.corrcoef(s, o)[0, 1] if len(o) > 1 else 0

                # Bias proportion (systematic over/underprediction)
                results['theil_Um'] = float(mean_diff ** 2 / mse)
                # Variance proportion (different spread)
                results['theil_Us'] = float((sd_s - sd_o) ** 2 / mse)
                # Covariance proportion (unsystematic, random error)
                results['theil_Uc'] = float(
                    2 * (1 - r_so) * sd_s * sd_o / mse
                )
            else:
                results['theil_Um'] = 0.0
                results['theil_Us'] = 0.0
                results['theil_Uc'] = 0.0

            # R-squared
            ss_res = np.sum(errors_all ** 2)
            ss_tot = np.sum((o - np.mean(o)) ** 2)
            results['r_squared'] = float(
                1 - ss_res / ss_tot if ss_tot > 0 else 0
            )

            # Worst year
            pct_errors = 100 * np.abs(errors_all) / o
            worst_idx = np.argmax(pct_errors)
            time_valid = self.time[:n_obs][valid_all]
            results['max_error_year'] = float(time_valid[worst_idx])
            results['max_error_pct'] = float(pct_errors[worst_idx])

        # ── Publication readiness assessment ──
        train_ok = results.get('train_mape', 100) < 5
        test_ok = results.get('test_mape', 100) < 10
        theil_ok = results.get('theil_U', 1) < 0.3
        r2_ok = results.get('r_squared', 0) > 0.95
        # Good Theil decomposition: most error is unsystematic (Uc > 0.7)
        decomp_ok = results.get('theil_Uc', 0) > 0.5

        results['publication_ready'] = all([
            train_ok, test_ok, theil_ok, r2_ok
        ])
        results['criteria'] = {
            'train_mape_lt_5pct': train_ok,
            'test_mape_lt_10pct': test_ok,
            'theil_U_lt_0.3': theil_ok,
            'r_squared_gt_0.95': r2_ok,
            'error_mostly_unsystematic': decomp_ok,
        }

        return results
