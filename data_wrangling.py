import polars as pl
import numpy as np

def load_data(path: str) -> pl.DataFrame:
    """Read CSV into a Polars DataFrame."""
    return pl.read_csv(path)


def standardize_columns(df: pl.DataFrame) -> pl.DataFrame:
    """Rename columns to snake_case."""
    mapping = {c: c.strip().lower().replace(' ', '_') for c in df.columns}
    return df.rename(mapping)


def parse_dates(df: pl.DataFrame) -> pl.DataFrame:
    """Convert 'start_date' from string to Date."""
    return df.with_columns(
        pl.col('start_date').str.strptime(pl.Date, '%m/%d/%Y')
    )


def drop_duplicates(df: pl.DataFrame) -> pl.DataFrame:
    """Remove duplicate rows."""
    initial = df.height
    df_clean = df.unique()
    print(f"Dropped {initial - df_clean.height} duplicates.")
    return df_clean


def drop_message(df: pl.DataFrame) -> pl.DataFrame:
    """Drop 'message' column if over 50% values are null."""
    if 'message' in df.columns:
        pct_null = df['message'].null_count() / df.height
        if pct_null > 0.5:
            print("Dropping 'message' (>50% null).")
            return df.drop('message')
    return df


def impute_numeric(df: pl.DataFrame) -> pl.DataFrame:
    """Fill nulls in numeric columns using NumPy median."""
    for col, dtype in df.schema.items():
        if dtype in (pl.Int64, pl.Float64) and df[col].null_count() > 0:
            arr = df[col].to_numpy()
            median_val = np.nanmedian(arr)
            df = df.with_column(
                pl.col(col).fill_null(median_val)
            )
            print(f"Imputed '{col}' nulls with median {median_val}.")
    return df


def main():
    # Updated dataset path
    input_path = 'data/Air_Quality.csv'
    output_path = 'data/Air_Quality_clean.csv'

    # Load and inspect
    df = load_data(input_path)
    print("Columns:", df.columns)
    print("Initial shape:", df.shape)

    # Clean
    df = (
        df
        .pipe(standardize_columns)
        .pipe(parse_dates)
        .pipe(drop_duplicates)
        .pipe(drop_message)
        .pipe(impute_numeric)
    )

    # Save
    df.write_csv(output_path)
    print(f"Cleaned dataset saved to '{output_path}'. Final shape: {df.shape}")


if __name__ == '__main__':
    main()
