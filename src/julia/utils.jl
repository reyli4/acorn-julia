using CSV
using DataFrames
using Dates

function filter_by_year(df, target_year)
    date_cols = names(df)[2:end]  # Assuming first column is always bus_id
    year_cols = filter(col -> year(DateTime(col, "yyyy-mm-dd HH:MM:SS+00:00")) == target_year, date_cols)
    return select(df, [:bus_id; Symbol.(year_cols)])
end;