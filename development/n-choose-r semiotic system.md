# n-choose-r semiotic system

Primary problem: I am ignorant of the many contexts in which nCr is used, and I am especially ignorant of synonymous notations, colloquialisms, and equivalent mathematical expressions.

As of 2026-10-19, my strongest understanding of nCr-related concepts comes from probability: therefore, I shall call everything "nCr", and I will intentionally build on my probability knowledge.

To cure my ignorance, I must be able to distinguish nCr from non-nCr. Therefore, in the absence of a more specific rule, some identifiers and other labels must be modified.

1. If an identifier or other label is related to nCr, prepend `Y7Y_` whether or not the identifier is clear. (`Y7Y_` is a temporary marker and limited to this scope.)
2. If an identifier or other label could be misinterpreted as nCr-related (e.g., "coefficient" seems to be a problem), prepend the identifier with the already established `Z0Z_`, which indicates that the object needs attention.

## How many combinations?

combinationsTotal = nCr = binomial(n, r) = n! / (r! * (n - r)!)

## Binomial Coefficient

Expanding (1 + x)ⁿ gives a series of addends starting with x^0 and increasing to x^n. Each power of x
has an integer _coefficient_. Instead of long multiplication, we can find each coefficient using nCr.
"n" is the same in both nCr and (1 + x)ⁿ. In x^d, x is the base, d is called the exponent, the power,
or the index. In the context of the binomial coefficient, "index" is the best word. The addends,
written in increasing order of the index are indexed with the same value as the index of x. The r in
nCr is the index in the expanded polynomial. Therefore, whether one wants to find the integer
coefficient of x with the rth index or find the rth addend, one uses nCr.

### Semiotic implication

In my opinion, the "coefficient" in a binomial coefficient resulting from nCr is a cyclist holding on
to an nCr box truck. It doesn't have a strong claim to "coefficient." If we were routinely expanding
(1 + x)ⁿ without any modifications, it would have a stronger claim, but that _never_ happens.

One should never assume this package uses "coefficient" in the sense of binomial coefficient.

### Decisions recorded 2026-10-21

In `Z0Z_research/Y7Y_catalanArches/formulasReduce.py`, identifiers containing
"coefficient" do not need `Z0Z_` or any other naming adjustment. This includes
`工coefficient_k`, `工boxOfCoefficients`, `coefficientFinder`, `boxOfCoefficients`,
`boxOfColumnCoefficients`, and the imported `coefficientsEqual吗`. The definition of
`coefficientsEqual吗` in `analyze.py` uses the same name.

In `mapFolding/oeis/A400429/_A005315.py`, remove `Z0Z_` from all coefficient terms.
`crunchCoefficients`, `次coefficient`, and `boxOfCoefficients` describe polynomial or
rational-series coefficients, rather than nCr values. Ordinary prose such as
"coefficient table" and "numerator coefficients" also has no prefix.

These decisions settle the coefficient names in these two modules. They take precedence over
the general ambiguity rule above; they do not settle coefficient names in other modules.
Docstrings use exact identifiers in backticks, without adding plurals or possessives.
