from mapFolding import MapFolding

def main():
    folding = MapFolding()
    series = 'n'  # '2', '3', '2 X 2', or 'n'
    X_n = 5  # A non-negative integer
    foldingsTotal = folding.computeSeries(series, X_n)
    print(f"Number of foldings for dimensions {series} X {X_n} = {foldingsTotal}")

if __name__ == "__main__":
    main()
