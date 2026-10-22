# n-choose-r semiotic system

Primary problem: I am ignorant of the many contexts in which nCr is used, and I am especially ignorant
of synonymous notations, colloquialisms, and equivalent mathematical expressions.

As of 2026-10-19, my strongest understanding of nCr-related concepts comes from probability:
therefore, I shall call everything "nCr", and I will intentionally build on my probability knowledge.

To cure my ignorance, I must be able to distinguish nCr from non-nCr. Therefore, in the absence of a
more specific rule, some identifiers and other labels must be modified.

1. If an identifier or other label is related to nCr, prepend `Y7Y_` whether or not the identifier is
   clear. (`Y7Y_` is a temporary marker and limited to this scope.)
2. If an identifier or other label could be misinterpreted as nCr-related (e.g., "coefficient" seems
   to be a problem), prepend the identifier with the already established `Z0Z_`, which indicates that
   the object needs attention.

## How many combinations?

combinationsTotal = nCr = math.comb(n, r) = n! / (r! * (n - r)!) = binomial(n, r)

where 0 <= r <= n

### Binomial Coefficient

Expanding (1 + x)ⁿ gives a series of addends starting with x⁰ and increasing to xⁿ. Each power of x
has an integer _coefficient_. Instead of long multiplication, we can find any individual coefficient
using nCr. "n" is the same in both nCr and (1 + x)ⁿ. Given xⁱ, x is the base and i is called the
exponent, the power, or the index. In the context of the binomial coefficient, "index" is the best
word.

Write the addend terms as a sequence in increasing order of the x index. The relative position of each
addend term in the sequence is also called its index. Conveniently, the index of an addend term is the
same value as the index of x in that addend term.

The r in nCr is the index in the expanded polynomial. Therefore, whether one wants to find the integer
coefficient of x with the rth index or find the rth addend, one can use nCr.

### Semiotic implication

One should not assume this package uses "coefficient" in the sense of binomial coefficient.

As of 2026-10-22, the uses of "coefficient" are stable except for Lobb coefficients.
`theCoefficients` refers to THE coefficients to couple with the Catalan numbers to discover
a closed form of A000682. These coefficients retain their specific identity throughout
`工theCoefficients`, `lookupTheCoefficients`, and the corresponding CSV filenames.

The mixed correction table uses `catalanMultiplicands` and `binomialMultiplicands`.
These names describe the multiplicands of its Catalan and binomial basis values.
The scalar names and parameters `binomial(top, degree)` and `catalan(n)` are stable.

## The binomial function

$$
{(x+y)^{n}} = {ax^{k}y^{m}}\\
{k+m=n}
$$

The binomial function is a superset of nCr. I avoid using expanded polynomials, and prefer summing a
series of binomials.

addend = coefficient * binomial(n, k)

where binomial(n, k) = 0 for k < 0.

Here, k is equivalent to r in nCr, but k is a more conventional variable to represent an index. I have
been using the identifier "degree" as a synonym for power, exponent, or index. The words power,
exponent, and index are overloaded and/or ambiguous, so I avoided them.

n is the greatest factor in the series n, n-1, ..., n-k+1.
