# PyWorld3-03: Canadian Population Dynamics Model

A system dynamics model built on the World3 framework, adapted for Canadian demographics with real economic data integration and immigration feedback loops.

## Features

- **4-Age-Cohort Population Model** - Age-structured demographics (0-14, 15-44, 45-64, 65+)
- **Real Canadian Data** - Calibrated with 53 years (1971-2023) of economic and demographic data
- **Immigration Feedback Loop** - Labor force effects on GDP, which affects health services and mortality
- **Interactive Dashboard** - Streamlit UI with 7 tabs of plots and analysis
- **Windows/Mac/Linux Compatible** - One-click setup scripts for all platforms

## Quick Start

### Prerequisites

- Python 3.8+ installed (https://python.org)
- 500 MB free disk space

### Windows Setup (Recommended)

1. Download the repository (green Code button → Download ZIP, or use git clone)
2. Unzip into a folder
3. **Double-click** `setup_env.bat`
   - Wait for "Setup complete!" message
   - Press any key when prompted
4. Open Command Prompt and navigate to the folder:
   ```cmd
   cd C:\path\to\PyWorld3-03-Canada-Population
   venv\Scripts\activate
   streamlit run app_population.py
   ```

### Mac/Linux Setup

```bash
git clone https://github.com/YOUR_USERNAME/PyWorld3-03-Canada-Population.git
cd PyWorld3-03-Canada-Population
chmod +x setup_env.sh
./setup_env.sh
source venv/bin/activate
streamlit run app_population.py
```

Your browser will automatically open the interactive dashboard at `http://localhost:8501`

## Project Structure

```
PyWorld3-03-Canada-Population/
├── myworld3/                      # System dynamics model package
│   ├── population_canada.py       # MAIN: Canadian population module
│   ├── functions_table_canada.json # Canada-calibrated lookup tables
│   ├── agriculture.py             # Agriculture sector (future work)
│   └── [other World3 sectors]
├── data/
│   ├── canada_population.csv      # 53 years of population & immigration data
│   └── canada_economic.csv        # GDP, life expectancy, fertility, etc.
├── app_population.py              # Interactive Streamlit dashboard
├── requirements.txt               # Python package dependencies (10 packages)
├── setup_env.bat / setup_env.sh   # Automated environment setup

```

## Dashboard Tabs

1. **Total Population** - Historical and projected population with growth rates
2. **Age Structure** - Population by age cohort with dependency ratios
3. **Vital Rates** - Births, deaths, life expectancy, total fertility rate
4. **Immigration Impact** - Immigration by age group and cumulative effects
5. **Economic Feedback** - Economic inputs and multiplier effects
6. **Data Table** - Raw results with CSV export

## Model Parameters (From Streamlit Sidebar)

### Immigration
- **Scenario:** Historical, Zero, High, +0.5%, +1%, +2%
- **Emigration:** Toggle on/off with configurable rate
- **Age Distribution:** Adjust what percentage of immigrants go to each age cohort

### Demographics
- **Life Expectancy (Normal):** 73-85 years
- **Total Fertility Rate:** 1.3-3.5 children per woman
- **Desired Family Size:** 1-4 children
- **Maturation Time:** 14-15 years (duration of age cohort 1)

## Data Sources

- **Population & Immigration:** Statistics Canada, Immigration, Refugees & Citizenship Canada
- **Economic Data:** World Bank, Statistics Canada, FAO
- **Historical Range:** 1971-2023 (53 years)


## Model Validation

The model is validated using:

| Metric | Target | Typical Result |
|--------|--------|---|
| **Train MAPE** | < 5% | ~2-3% |
| **Test MAPE** | < 10% | ~3-5% |
| **R²** | > 0.95 | ~0.98+ |
| **Theil U** | < 0.3 | ~0.08-0.12 |

## Troubleshooting

### "Python not found"
Install Python from python.org with "Add Python to PATH" checked.

### "No module named streamlit"
Make sure you activated the virtual environment: `venv\Scripts\activate` (Windows) or `source venv/bin/activate` (Mac/Linux)

### Port already in use
Run: `streamlit run app_population.py --server.port 8502`


## Citation

If you use this model in research, please cite:

```
[Your Name] (2025). "System Dynamics Modeling of Canadian Population Dynamics with
Immigration Feedback and Economic Integration." [Journal/Conference], [Year].
```

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## Contact & Support

For questions or issues:
- Open a GitHub Issue
- Check Population_Module_Manual.md for technical documentation
- Review CHANGES_SUMMARY.md for recent updates

## Acknowledgments

- Built on the World3 model (Limits to Growth, 1972)
- Canadian demographic data from Statistics Canada
- Economic data from World Bank and FAO