# n-choose-r terminology: a descriptive inventory; 2026-10-19

## Findings

- Terms, 105 spellings
- In 58 non-cache files.
- Related to
  - binomial,
  - Catalan and A000108
  - ‼️ Lobb: I was didn't know this is nCr
  - ‼️ Dyck: I was didn't know this is nCr
  - combinations (n-choose-r)
- Found in
  - identifiers,
  - import/path words,
  - prose,
  - quoted labels
  - table headers.

❓What is Narayana?

## Scope

- 269 Python files
- 8 notebooks
- 227 text files

Directories

- mapFolding,
- Z0Z_research,
- research,
- archive,
- easyRun,
- development,
- Z0Z_notes,
- textual literature notes,
- tests,
- generated Python,
- configuration,
- CSV labels,
- OEIS text caches.

Six .xlsx workbooks were searched for binomial/Catalan/Lobb/Pascal/ ballot/Narayana/nCr/nCk
terminology and COMBIN/CHOOSE formula calls. No matches were found. I assumed this would be true.

## Issues

### Most Important Distinctions Already Present

1. "Binomial coefficient" has two different referents. binomial(top, degree) and
   comb(...) produce n-choose-r values. binomialCoefficients in the Catalan CSV,
   _binomialCoefficients in the cyclic-form generator, and the notebook's
   "binomial coefficients (descending)" describe multipliers of those values.
   The report text "integer binomial coefficients" also means basis multipliers.

2. binomial is not one implementation everywhere. _beDry.py:15 defines an
   int-returning SymPy wrapper; counting/archCurses.py:6 imports math.comb AS
   binomial. archReducePrototype.py imports both math.comb and the SymPy wrapper.
   analyze.工basisBinomialGrowth explicitly implements the negative-top extension
   in its before-anchor branch. Shared spelling does not establish shared domain.

3. The Catalan family includes scalars, evaluated basis sequences, weighted sums,
   mixed-basis evaluations, correction rows, residual coefficient tables, and
   on-disk paths. catalanMultiplyThis is a callable, while CatalanMultiplyThis is
   a mapping type and catalanMultiplyThis.csv is a coefficient table. Their names
   resemble one another, but they denote different kinds of objects.

4. Coefficient levels are distinct: theCoefficients / X / c_{k,i}; whole-growth
   multipliers a_{s,less}; CSV residual multipliers; degree-law multipliers
   b_{less,q}; Catalan/binomial correction multipliers; and Lobb multipliers
   h_{k,r}. All can be called "coefficients", but they are not interchangeable.

5. Coordinate identity changes between contexts. 次k is a coefficient position
   (often s in prose), while column_k in growth evaluation is the varying
   coefficient-table row / original triangle column. n can be an arch count, a
   Catalan ordinal, a coefficient-table row or a number of coefficient variables.
   cardinality, j and ordinalCatalan can carry the same Catalan ordinal.

6. antiCatalan / anti-Catalan / Anti catalan are local labels for the additive
   correction family D(j,a), not a negation or reciprocal of the Catalan numbers.
   工antiCatalan includes both a mixed basis and finite additive corrections.
   formulasAntiCatalan is an integer row count, not a collection of callables.

7. Representation order is not uniform. The stored Catalan residual row has TWO
   nested tuples; its getter flattens the Catalan tuple and keeps the binomial
   tuple last. analyze.工antiToFormula returns a binomial tuple FIRST followed by
   individual Catalan multipliers, in ascending basis order. This is a different
   representation, not merely another spelling of the stored row.

### Scalar Operations And Related Combinatorial Operations

binomial(top, degree) -> int

- Z0Z_research/catalanArches/_beDry.py:15
- Returns int(sympy_binomial(top, degree)). top is the upper argument; degree
- is the lower argument. The underlying import is sympy_binomial at line 8.
- Re-exported by catalanArches/__init__.py:23.

sympy_binomial and sympyBinomial

- _beDry.py:8 and formulaExperiments.py:13 respectively.
- Both import SymPy's binomial. The former supports numeric evaluation via the
- wrapper; the latter builds symbolic degree laws in formulaExperiments.py:174.
- These are spelling variants of the same external operation in different roles.

comb

- math.comb in archReducePrototype.py:8, archiveTriangle/A005315Binomial.py:16,
- A005315PolynomialCyclic.py:7, translateToBinomial.py:12, and the production
- mapFolding/oeis/A400429/_A005315.py:70.
- createPolynomialCyclicForm.py emits the quoted callable name 'comb' into
- generated ASTs and imports. Those strings are references, not local bindings.
- Repeated numeric comb terms dominate the raw match count in A005315Binomial.py.

binomial as a math alias

- Z0Z_research/catalanArches/counting/archCurses.py:6,117,136.
- The name resolves to math.comb here, not the research package wrapper.

catalan(n) -> int; sympy_catalan

- _beDry.py:30 and :8. Scalar Catalan-number evaluation, returned as int.
- The package re-exports catalan. Other scalar notations in local text are
- C(n), C_n, C_m, C(j), Cat(n-i), and A000108. See the notation section.

lobb(ordinal, archesOffset) -> int

- Z0Z_research/constraint/archReducePrototype.py:101.
- binomial(2*ordinal, ordinal-archesOffset)
  - binomial(2*ordinal, ordinal-archesOffset-1).
- The first argument is the Lobb row, the second the offset/column. The local
- note explain/lobb.md writes the same operation as L(m,r), with A039599.

combinations

- itertools.combinations enumerates selections rather than returning their count.
- Executable pair enumeration occurs in mapFolding/_e/dataBaskets.py:795,
- _e/reduceIt.py:127 and ./_e/p2上nDimensional/reduceIt.py:186.
- _e/algorithms/iff.py imports it at :60 but contains no combinations(...) call.
- Its comment at :76 writes (14choose2), 14c2, 21c2 and 18c2 as selection counts.
- Z0Z_notes/A000136.ipynb cells 8-11 call combinations(cells,r) or combinations
- on constructed cell collections. r is the selection size there; cc, symmetry
- and setOfSymmetry denote selected objects/collections, not numeric coefficients.
- Z0Z_notes/FunctionalProgramming.md:69 lists itertools.combinations as an API.
- Generic English "compatible combinations" in dataBaskets docstrings is not a
- guarantee that those passages describe an n-choose-r operation.

Dyck-related labels

- mapFolding/algorithms/catalanArch.py:43 defines advanceDyck and :13 labels its
- enumeration C(n) iterations. count(n) produces a generation histogram, not C_n
- as a scalar. doTheNeedful(n) produces cumulative reduction counts.
- mapFolding/algorithms/matrixMeanders.py:8 defines walkDyckPath; generated
- variants and an older archive implementation retain this name.
- mapFolding/kitAST/theSSOT.py:158 uses the string key 'Dyck' with value
- 'walkDyckPath'. Dyck language/path terminology describes the combinatorial
- objects/operations, not an alternative scalar n-choose-r API.

Additional unbranded binomial computations:

- counting/archCurses.py:109 sumProductSumSubSub_k uses alternating binomial
- weights against outlineSubSub_k. :123 get_sub_k uses a degree+1 binomial
- finite-difference recurrence beyond kPolynomial. Its labels k, indexSub_k,
- indexSubSub_k, kPolynomial, degree, outlineSubSub_k and valueCalculated do
- not contain binomial/Catalan. The strings sumProductSumSubSub_k and get_sub_k
- also name persisted cache layers. These belong in the inventory even though
- their function names would not be found by a binomial-only search.

Production versus archived A005315 representations:

- mapFolding/oeis/A400429/_A005315.py uses comb in A005315of0 and A005315of2.
- Its later A005315of3 through A005315of11 use numerator/denominator polynomial
- coefficients and crunchCoefficients (a rational generating-function recurrence).
- research/matrixMeanders/archiveTriangle/A005315Binomial.py instead stores
- repeated explicit comb expansions in A005315of3 through A005315of10.
- A005315PolynomialCyclic.py uses the same n-choose-r basis within cyclic waves.
- x is the input row; 次n is an input-origin translation parameter. A005315ofN
- selects a fixed formula. These names do not expose the chosen representation.
- numerator, denominator and 次coefficient are polynomial/series coefficient
- labels in the production version, not scalar n-choose-r arguments.

### Basis Sequences, Multipliers And Evaluators

工boxOfBasisBinomial(top, start, stop=None, step=1)

- _beDry.py:18. Produces evaluated binomial(top,r) values for r in range(...).
- The upper argument is fixed and the lower argument varies. start/stop/step
- index the lower argument; they do not vary top.

工boxOfBasisCatalan(start, stop=None, step=1)

- _beDry.py:24. Produces evaluated Catalan values for ordinals in range(...).
- boxOfBasisCatalan in cXc.py:24,30 holds a realized descending basis sequence.
- getCatalanBasis(n,row_n) in constraintPropagationPrototype.py:114 produces
- n values beginning at C(row_n), descending. Here n is the basis length.

These basis names denote evaluated numbers. The names basis and 工basis... in
other functions may instead denote a callable that evaluates one basis position.
For example analyze.basisToEquation accepts basis: Callable[[int],int].

Phrasing and generic multiplier labels already in use:

- boxOfNumbers: generic dot-product input; in dotProductCatalan it holds the
  - multipliers of Catalan basis values, despite the generic noun.
- boxOfMultiplicands: second dotProduct input (_beDry.py:44).
- boxOfIntegers: binomial multipliers or a mixed flat/nested correction record.
- boxOfCoefficients, coefficients, lookupCoefficients: different coefficient
  - families depending on their scope; see the coordinate/index appendix.
- multiplyThis: a locally bound CALLABLE, in both 工growthBinomial and
  - 工antiCatalan, rather than a multiplier value or a coefficient tuple.
- basisCoefficients: the fixed numeric weights passed to solver constraints;
  - boxOfCoefficients there contains the unknown coefficient variables.

At zero shifts, the following formulas show the represented objects. These are
explanatory notation for existing behavior, not proposed identifiers.

- `dotProductBinomial(T,d,(b_0,...,b_(q-1))) = sum_i b_i * binomial(T,d-i)`
- `dotProductCatalan(o,(c_0,...,c_(q-1))) = sum_i c_i * C(o-i)`
- `catalanMultiplyThis(j,a,(*c,b)) = sum_i c_i * C(j+a-i) + C(j) * sum_i b_i * binomial(j,floor(a/2)-i)`

Sources: _beDry.py:34,47,50. catalanMultiplyThis therefore represents a mixed
Catalan / Catalan-times-binomial basis. Its two shifts affect the respective
generated ranges. dotProductCatalan's stop remains based on unshifted ordinal;
record its actual range expression when reasoning about nonzero shifts.

工basisAntiCatalan(j,a,position), formulaExperiments.py:72, evaluates one element
of this mixed basis. Positions below a select shifted Catalan values; subsequent
positions select C(j) times descending binomial values. totalBasis at :84 counts
both families. AntiCatalanFormula is the Callable[[int,int],int] type at :22.

工coefficient_k(n,次k), formulasReduce.py:24, adds 工growthBinomial to 工antiCatalan.
工growthBinomial(column_k,次k), :27, evaluates the growth contribution with explicit
scaffolding plus getBinomialMultiplyThis(...). 工antiCatalan(j,次antiCatalan), :61,
evaluates the correction scaffolding plus getCatalanMultiplyThis and getCatalanAddThis.
The notebook cells 7 and 17 redefine the evaluator names locally for experiments.
Their references 工growthBinomialReference and 工antiCatalanReference are captured
callables; their names do not denote stored numeric reference tables.

### The Different Coefficient Families

Original Catalan-linear coefficient rows:

- theCoefficients.csv; 工theCoefficients; lookupTheCoefficients; lookupCoefficients;
- lookupX / lookupXTerms; c_{k,i} / c_{k,s}; boxOfTheCoefficients.
- The table row index is the original triangle column, while its entries weight
- a descending Catalan basis across the triangle row argument.
- _filesystem.py:79 reads the table. cXc.py compares its evaluations with A287548
- and obtains A331499 by reversing and differencing. formulasParents.py derives
- diagonals and partial sums. A287548 and A331499 are resulting count triangles,
- not synonyms for the coefficient table or Catalan sequence.

Whole growth polynomials:

- `lookupGrowthFormulas[次k][less]`, `lookupGrowthCoefficients` and `a_{s,less}`.
- analyze.py:22 defines the signed, anchored, descending-binomial basis.
- 工basisBinomialGrowth supports differenceOrder as well as before-anchor values.
- growthTermsToFormula, growthFormulaToTerms and growthToFormula fit/evaluate
- multipliers. GrowthFormula in formulaExperiments.py:21 is a callable type,
- while lookupGrowthFormulas is a mapping of numeric multipliers.

Growth residual coefficients:

- lookupResidual, lookupGrowthResiduals, lookupResidualByDegree;
- binomialMultiplyThis.csv, BinomialMultiplyThis, getBinomialMultiplyThis.
- growthFormulaToResidual and growthResidualToFormula remove/restore the shared
- growth scaffolding. These residuals depend on which explicit contributions
- remain in 工growthBinomial; their identity is not the whole growth polynomial.
- The terms "growth terms", "growth tuples", "residual tuples" and
- "binomial coefficients" refer to different stages in these operations.

Laws Across Polynomial Degrees:

- `lookupDegreeFormulas[parityEven,less][position]`, `lookupGrowthDegreeFormulas`,
- degreeFormula; b_{less,q} in explain/A CoefficientAnalysis.md.
- These weight binomial(degree-less,q), not the original binomial growth basis.
- analyze.py:198,215 fits/evaluates these laws. order in the symbolic notebook
- analysis is this lower binomial argument q, not the original growth degree.

Anti-Catalan correction VALUES:

- 工antiCatalan; `lookupAntiCatalan[a][j]`; lookupRecoveredAntiCatalan;
- lookup工anti / lookup工antiRow; lookupCorrections; countAntiCatalan.
- _工antiToLookup, columnTo工anti and 工antiToFormula manipulate these values or
- recover their multipliers. The numeric correction, its table row, its
- coefficient record and its evaluation operation have separate identities.

Anti-Catalan correction MULTIPLIERS:

- catalanCoefficients / binomialCoefficients (CSV header and getter locals);
- CatalanMultiplyThis / CatalanAddThis (mapping type aliases);
- catalanMultiplyThis.csv / catalanAddThis.csv (different stored artifacts).
- The latter is a finite table of additive values, not Catalan-basis weights.
- coefficientsCatalan in archReducePrototype.py:138 is an original-row result
- after converting Lobb coefficients; it is not the same family as the residual
- catalanCoefficients in catalanMultiplyThis.csv.

Lobb multipliers:

- coefficientsLobb, coefficientLobb, lookupLobbX, h_{k,r}.
- archReducePrototype.py collects/counts them and converts them to coefficientsCatalan.
- analyze.extendRowSumXLobb uses them as an intermediate basis to extend X rows.
- basisLobb contains evaluated Lobb numbers, not the unknown transformed weights.
- The solver variable label is lobbCoefficient[n,archesOffset] (:112), reversing
- the word order of coefficientLobb. The weighted-sum label includes literal Lobb.

Cyclic/quasipolynomial binomial multipliers:

- archiveTriangle/createPolynomialCyclicForm.py:102 _binomialCoefficients returns
- forward-difference-derived multipliers. ast_exprBinomialBasis (:171) represents
- an AST expression for one basis value. boxOfCoefficients contains its weights.
- translateToBinomial.py:20-24 uses labels (period,shift,degree) or ('constant',0,0)
- and basis(offset,label) = comb((offset+shift)//period,degree).
- These are a separate binomial-basis representation of A005315 formulas.

### Coordinates And Control Labels

top

- The upper n-choose-r argument T. In 工growthBinomial, T = column_k-kArchetype.
- top+shift is passed to the basis; it is not the coefficient row index itself.

degree

- Lower n-choose-r argument in binomial(top,degree).
- Highest lower argument / polynomial degree in dotProductBinomial and growth.
- Derived property IndexColumn_k.degree = (次k-1)//2.
- Degree of a residue polynomial in the cyclic-form generator.
- Lower argument for a chosen period/shift in translateToBinomial labels.

degreeLess; less; binomialDegree; position; order

- degreeLess = 次k.degree-less-differenceOrder in 工basisBinomialGrowth.
- less counts downward from the highest growth degree (zero-based position).
- binomialDegree is a lower argument in the Lobb-to-Catalan conversion.
- position indexes a mixed basis in 工basisAntiCatalan, or a degree-law weight.
- order is the lower argument in symbolic degree-law expansion, but also denotes
- a cyclotomic order/period in the cyclic-form generator. It is not one global unit.

ordinal; ordinalCatalan; cardinality; j

- ordinal is the scalar Catalan index in dotProductCatalan and a Lobb row in lobb.
- cardinality in catalanMultiplyThis is passed j by its callers.
- ordinalCatalan in 工antiToFormula is the key j of an observed correction row.
- j is also used for triangle indices or summation indices in explanatory notes.

IndexColumn_k / 次k; column_k; row_n; n; 次rowSum; 次diagonal

- 次k = coefficient position s in the growth/correction system.
- column_k = original count-triangle column / row of the coefficient table.
- row_n = count-triangle row in formulasParents; other functions use n or row.
- 工coefficient_k's n is the varying coefficient-table row supplied as column_k
- to 工growthBinomial. catalan's n is a Catalan ordinal. getCatalanBasis's n is
- the number of basis entries. The arch enumeration APIs use n for arch count.
- 次rowSum and 次diagonal select particular derived coefficient rows in
- formulasParents. Their stored values are multipliers, not triangle counts.

IndexColumn_k.sansAntiCatalan = 2*次k-4; 次kSansAntiCatalan = 4

- The former is a coordinate threshold/anchor component; the latter is an offset
- constant. kArchetype = max(次k,次k.sansAntiCatalan).
- 次k.j(n) = n-次k+1. 次k.次antiCatalan(n) = 2*次k-5-n.
- Thus j+a=次k-4 for a=次antiCatalan. See _semiotics.py:11-49.

次antiCatalan; 次Lobb; archesOffset; ordinalLobb

- 次antiCatalan indexes a correction-family row; j indexes entries within it.
- 次Lobb / archesOffset are Lobb column/offset coordinates.
- ordinalLobb in analyze.extendRowSumXLobb = row_n-column_k+1 (Lobb row).
- ordinal and offset must remain distinct even when their numeric values coincide.

parity; parityEven; 次k_isEven吗; isEven吗; isOdd吗

- Stored parity 0 means even 次k, and 1 means odd 次k.
- parityEven / 次k_isEven吗 are Booleans whose True means EVEN.
- getBinomialMultiplyThis indexes with not 次k_isEven吗; binomialLookupToTuple
- explicitly converts that to int. binomialDegreeToRecord reconstructs
- 次k = 2*degree+2-parity. The Boolean and stored-int encodings are opposite.

direction; degreeDirection; shift; shiftCatalan; shiftBinomial

- direction is the sign selected by the second-lowest bit of 次k.
- degreeDirection = (-1)**(次k+1), the parity-dependent shift sign.
- These encode signs, not physical directions on an arch diagram.
- shift in dotProductBinomial offsets its upper argument; shift in
- dotProductCatalan changes the start of its ordinal range. The two named shifts
- in catalanMultiplyThis affect different bases. Cyclic-form shift additionally
- occurs before floor division by period, so it is a different operation.

formulasAntiCatalan; totalCatalan; totalBasis; totalCoefficients

- formulasAntiCatalan = 次kMaximum-4 counts correction rows in generate.py.
- totalCatalan counts shifted Catalan basis entries in 工antiToFormula.
- totalBasis counts mixed basis entries in antiCatalanFormulasToRow.
- totalCoefficients is the number of weights recovered by an equation solve.
- boxOfAntiCatalanPositions and boxOfCatalanPositions are coordinate ranges;
- their elements are not Catalan values.

### Persisted Files, Column Labels, Types And Notebook State

All nine research CSVs live under Z0Z_research/catalanArches/data.

binomialMultiplyThis.csv

- Headerless rows: parity, degree, then descending-binomial residual weights.
- readBinomialMultiplyThis -> `dict[parity][degree]` -> tuple of weights.
- BinomialMultiplyThis is the corresponding Mapping type.
- Notebook display headers: parity, degree, 次k, then numeric less positions.
- read/get/writeBinomialMultiplyThis and pathFilenameBinomialMultiplyThis refer
- to this artifact. There is no module-level variable binomialMultiplyThis;
- the bare spelling occurs in filenames, prose and diagnostic strings.

catalanMultiplyThis.csv

- Explicit header: 次antiCatalan,catalanCoefficients,binomialCoefficients.
- readCatalanMultiplyThis -> dict[a] -> (tupleCatalan,tupleBinomial).
- getCatalanMultiplyThis -> (*tupleCatalan,tupleBinomial).
- CatalanMultiplyThis is the two-tuple mapping type in formulaExperiments.py.
- HTML headers: 次antiCatalan, Catalan coefficients (descending),
- binomial coefficients (descending).
- writeCatalanMultiplyThis and pathFilenameCatalanMultiplyThis retain the name.

catalanAddThis.csv

- Headerless rows: a, then additive corrections indexed by zero-based j.
- readCatalanAddThis -> dict[a] -> tuple of values; getCatalanAddThis(a,j)
- returns an individual value. CatalanAddThis is its mapping type.
- HTML headers are 次antiCatalan and j = 0, j = 1, ... .
- writeCatalanAddThis and pathFilenameCatalanAddThis identify the same artifact.

coefficientsLobb.csv

- Headerless k,h_(k,0),...,h_(k,k-1), as documented in explain/lobb.md.
- pathFilenameCoefficientsLobb is defined/re-exported. The current Python source
- contains no read call using that path; do not mistake the path name for an
- actively loaded lookupLobbX table.

theCoefficients.csv / theCoefficients_incomplete.csv

- Complete and sparse/incomplete original Catalan-linear coefficient rows.
- Paths: pathFilenameTheCoefficients / pathFilenameTheCoefficientsIncomplete.
- 工theCoefficients reads the complete table. These store coefficients, unlike
- A287548.csv / A331499.csv / A331499_incomplete.csv, which store count triangles.

pathDataCatalan

- _filesystem.py:25. Directory path for all of the above and archCurses caching.
- The directory label catalanArches therefore contains data of multiple bases.
- Singular catalanArch.py is the runtime enumeration module; plural
- catalanArches is the research package label. cXc.py is another file label.

Notebook source labels and bindings, formulaExperiments.ipynb:

- Headings: Binomial; Binomial — load CSV values; Binomial — calculate;
- Coefficient table; Analysis; Binomial — save CSV values;
- Catalan — load CSV values; Catalan — calculate;
- Catalan — inspect finite corrections; Catalan — save CSV values.
- lookupBinomialStored / lookupBinomialCandidate: captured vs recalculated tables.
- lookupCatalanStored / lookupCatalanCandidate: captured vs recalculated mixed weights.
- lookupCatalanAddStored / lookupCatalanAddCandidate: captured vs recalculated
  - finite additive corrections.
- paritySelection / degreeSelection / antiCatalanSelection select table rows.
- 工growthBinomialReference / 工antiCatalanReference are captured callables.
- The section called Catalan operates on the anti-Catalan mixed correction model.

Other output labels:

- formulaExperiments.py displays degree, less-column numbers, Δ, Δ², error,
- and comparisons labelled '`<position> @ -degree`', '`@ 0`', '`@ +degree`'.
- generate.py reports 'growth terms', 'growth tuples', 'growth changes',
- 'antiCatalan terms', 'antiCatalan changes', 'less = ...' and 'j = ...'.
- These labels can describe values, residual weights or changes to weights;
- consult the producing function rather than inferring identity from 'terms'.

### Mathematical Notation In Local Explanations

explain/A CoefficientAnalysis.md

- s = fixed coefficient position (次k); k = varying coefficient-row coordinate.
- d = floor((s-1)/2); K_s = max(s,2s-4); sigma_s = growth sign.
- g_s(k) = growth polynomial; a_(s,ell) = its binomial-basis weights.
- ell = less; p = difference order; D(j,a) = correction.
- c_(k,s) = g_s(k)+D(k-s+1,2s-5-k).
- epsilon_s = parity-dependent shift sign.
- r_ell(d) = residual law across degree; b_(ell,q) = its binomial weights.

explain/anti catalan.md

- C(j) / C(j+t) = scalar Catalan values; D(j,a) = correction family.
- r_a(j) = rational factor after dividing by C(j); P_a(j) = polynomial part.
- b_(a,t) = weights of shifted Catalan values. a = 次antiCatalan.
- The historical paper quantities M-bar and m-bar describe another statistic;
- the note itself distinguishes them from A287548's T(n,k).

explain/lobb.md

- L(m,r) = Lobb number, A039599; C_m = Catalan number.
- F_k(n) = Catalan-linear evaluation of coefficient row k.
- c_(k,i) = original weights; h_(k,r) = transformed Lobb weights.
- f_m = observations of F_k at successive n, unrelated to scalar f(k) in Sadé notes.
- p = known-prefix length; r = Lobb offset; a,b = Catalan identity arguments.

explain/A331499.md

- Cat(n-i) and C_(n-i) both mean Catalan values.
- D_r(n) = a right-edge diagonal of A331499, distinct from correction D(j,a).
- d_(r,i) = diagonal Catalan weights; C(z) = Catalan generating function.
- Q_r(z), P_k(z) = polynomial factors for count sequences.
- T(n,k) here concerns A331499; T(n,k) in anti catalan.md concerns A287548.

explain/shifting.md

- binomial notation expresses shift-induced basis changes.
- C_j(d) is an unshifted BASIS COEFFICIENT, not the Catalan number C_j.
- a_j(d) is a shifted contribution; s_i(d) a selected shift.
- 'Odd fourth', 'Even fourth', 'leading binomial', 'second-position contribution'
- and 'fourth integer' label multiplier positions rather than Catalan ordinals.

Additional representations:

- research/matrixMeanders/Status Semi-Meander Triangle.md:361 writes the
- partial-fraction coefficient factor binom(t+r-1,r-1); :625-627 describes
- Catalan totals and the A287548/A331499 count-triangle interpretations.
- Z0Z_literature/Sade_1949_working_reference.md:373 identifies Catalan values
- as f(k) / A000108. Citation URLs containing A000108 need not denote evaluation.
- mapFolding/oeis/_theSSOT.py:20 lists A000108 as a known/downloadable sequence.
- mapFolding/tests/test_oeis.py:38 labels its public dispatcher case
- 'unimplemented-sequence'. constraintPropagationPrototype.py resolves that ID
- directly with catalan(n); these are different access layers.

### Documentation-Only Or Historical Spellings

explain/A CoefficientAnalysis.md contains names not defined by current Python:

- 工lookupAntiCatalan, 工lookupGrowth, 工direction,
- reproduceCoefficientExtensions, verifyCoefficientAnalysis.
- The replay paths reproduceCoefficientExtensions.py and coefficientVerify.py
- referenced there are absent. 工multiplicand is explicitly described as removed.
- Current counterparts include direction as an IndexColumn_k property,
- _工antiToLookup, growthFormulaToTerms and inline correction-table construction;
- they are related operations, not automatically interchangeable replacements.

explain/anti catalan.md mentions verifyFormulas, inner5 and historical finite rows.
Those are documentary references rather than current bindings in formulasReduce.py.
The statement in A CoefficientAnalysis.md that discovery uses correction rows 0-8
also does not match generate.py's current range(formulasAntiCatalan), where the
count is 次kMaximum-4 (32-4). Treat numerical claims in those notes as historical
until checked against the current implementation/data. No such verification was
performed in this naming review.

### Boundary Cases That Should Not Be Merged Into One Concept

The test ID cancel-part-of-binomial at test_Z0Z_generatingFunction.py:36,69 refers
to cancellation involving a two-term polynomial (1+x), rather than an n-choose-r
value. It is a second mathematical use of the word binomial.

Ordinary factorial/permutation counts and factorial ratios occur in OEIS formulas,
elimination symmetry multipliers and easyRun/pinning.py. They are related
combinatorics, but they are not necessarily n-choose-r formulas. Their locations
are listed separately below. No separate Pascal, ballot, Narayana, nCr or nCk
API/binding was found in current source. Lobb formulas and combinations enumeration
are the principal additional explicit families beyond binomial and Catalan.

The remaining appendices provide the exact spellings and all matched locations.
Line ranges compress consecutive matching lines; gaps are never filled in.

## Exact-Spelling Index

Continued in development/nChooseR-inventoryAnnex.md
