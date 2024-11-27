from mapFolding import MapFolding
import time
def main():
    folding = MapFolding()
    normalFoldings = True
    series = 'n'  # '2', '3', '2 X 2', or 'n'
    X_n = 5 # A non-negative integer
    timeStart = time.time()
    # foldingsTotal = folding.computeSeries(series, X_n)
    foldingsTotal = folding.computeSeriesConcurrently(series, X_n)
    print(f"Number of foldings for dimensions {series} X {X_n} = {foldingsTotal}. {time.time() - timeStart:.2f} seconds.")

if __name__ == "__main__":
    main()
