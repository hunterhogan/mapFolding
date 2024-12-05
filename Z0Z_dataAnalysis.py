import pathlib
import pandas as pd
def processBenchmarkResults(listBenchmarkResults):
    pathFilenameCSV = pathlib.Path("/apps/mapFolding/Z0Z_speedHistory.csv")
# Convert results to a DataFrame for analysis
    dfResults = pd.DataFrame(listBenchmarkResults)

# Pivot the DataFrame to have dimensions as columns and callables as rows
    pivot_table = dfResults.pivot(index='Callable', columns='Dimension', values='Mean Time (ms)')

# Calculate mean for each callable (excluding empty values)
    row_means = pivot_table.apply(lambda row: pd.to_numeric(row[row != '']).mean(), axis=1)
# Sort pivot table by mean values
    pivot_table = pivot_table.loc[row_means.sort_values().index]

# Replace NaN values with empty strings for better display
    pivot_table = pivot_table.fillna('')

# Calculate maximum width for the 'Callable' column
    callable_width = max(len(str(callable_name)) for callable_name in pivot_table.index) + 2  # Adding padding

# Calculate maximum width for each dimension column
    dimension_widths = []
    for dim in pivot_table.columns:
    # Get the width of the dimension name
        dim_width = len(str(dim)) + 2  # Adding padding
    # Get the maximum width of the data in this column
        data_width = max(len(f"{value:.2f}") if value != '' else 0 for value in pivot_table[dim])
    # Take the maximum of the two widths
        col_width = max(dim_width, data_width)
        dimension_widths.append(col_width)

# Create a format string for the header and rows
    header_format = f"{{:<{callable_width}}}" + "".join(f"{{:>{w}}}" for w in dimension_widths)
    row_format = f"{{:<{callable_width}}}" + "".join(f"{{:>{w}.2f}}" if w > 0 else "{{}}}" for w in dimension_widths)

# Print header with dimensions
    print(header_format.format("Callable", *[str(dim) for dim in pivot_table.columns]))

# Print each callable's times
    for callable_name, row in pivot_table.iterrows():
        listExecutionTimes = [row[dim] if row[dim] != '' else '' for dim in pivot_table.columns]
        formatted_times = [f"{time:.2f}" if time != '' else ''.rjust(w) for time, w in zip(listExecutionTimes, dimension_widths)]
        print(header_format.format(callable_name, *formatted_times))

# Handle CSV persistence
    try:
    # Read existing CSV if it exists
        existing_df = pd.read_csv(pathFilenameCSV)
    
    # Convert pivot_table back to long format for merging
        new_df = dfResults[['Callable', 'Dimension', 'Mean Time (ms)']]
    
    # Remove existing entries that we're about to update
        existing_df = existing_df[~((existing_df['Callable'].isin(new_df['Callable'])) & 
                               (existing_df['Dimension'].isin(new_df['Dimension'])))]
    
    # Combine existing and new results
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    
    except FileNotFoundError:
    # If no existing file, use current results
        combined_df = dfResults

# Save to CSV
    combined_df.to_csv(pathFilenameCSV, index=False)
