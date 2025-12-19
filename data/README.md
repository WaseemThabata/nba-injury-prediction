# NBA Injury Prediction - Data Directory

## Required Data Files

Place the following data files in the `data/raw/` directory before running the pipeline:

### 1. injuries_2010_2020.xlsx
**Description**: NBA injury records from 2010-2020  
**Expected Columns**:
- `name` - Player name
- `Date` - Injury date
- `injury` - Binary label (0=no injury, 1=injury)
- Other injury-related metadata

### 2. all_seasons.xlsx
**Description**: Player statistics aggregated by season  
**Expected Columns**:
- `name` - Player name
- `season` - Season (e.g., "2019-20")
- `gp` - Games played
- `age` - Player age
- `pts` - Points per game
- `reb` - Rebounds per game
- `ast` - Assists per game
- Other player statistics

### 3. NBA_Salaries.xlsx
**Description**: NBA player salary data  
**Expected Columns**:
- `name` - Player name
- `season` - Season year
- `salary` - Annual salary

## Directory Structure

```
data/
├── raw/                    # Place original data files here
│   ├── injuries_2010_2020.xlsx
│   ├── all_seasons.xlsx
│   └── NBA_Salaries.xlsx
└── processed/              # Generated during pipeline execution
    └── (intermediate data files)
```

## Data Sources

The original data was compiled from:
- NBA injury reports (various sources)
- Basketball-Reference.com player statistics
- Spotrac salary database

## Notes

- All data files should be in Excel format (.xlsx)
- Column names must match the expected schema
- The pipeline will automatically merge datasets based on player name and season
- Missing values will be handled by the preprocessing module
- Data files are not included in the repository due to size/licensing restrictions

## Getting Started

1. Obtain the required data files (contact repository owner)
2. Place files in `data/raw/` directory
3. Run the pipeline: `python scripts/run_pipeline.py`

If data files are not available, the pipeline will display an error message with instructions.
