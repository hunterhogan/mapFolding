---
name: Docstring Standards
description: Standards for Python docstrings.
---
# Python Docstring Standards

Standards for Python docstrings conforming to NumPy style with enhanced precision, international accessibility, and machine parseability.

## Docstring Format Standards

### Summary Line: Voice and Mood

**For public identifiers** (functions, classes, methods, variables without leading underscore):

The summary line must be written as **a second-person declarative clause expressing dynamic modality with an implied subject**.

**What this means:**

- **Second person with implied subject**: Address the reader as "you" implicitly, omitting the explicit subject pronoun.
- **Declarative**: State a fact or capability, not issue a command.
- **Dynamic modality**: Express ability, possibility, or permission using modal verbs like "can", "may", "will", or present tense with capability implications.
- **Terse**: Keep the summary line concise by using Subject-Verb-Object structure with the subject implied.

**Summary line structure:**

- Start with the verb directly (implied "you can" or "you may").
- Follow with the object and any essential modifiers.
- End with a full stop.

**Examples:**

- ✅ Correct: "Express descending iteration or a reverse direction."
- ✅ Correct: "Compute the total number of distinct map foldings for a given shape."
- ✅ Correct: "Convert a pile range to an iterator of individual leaves."
- ✅ Correct: "Validate folding constraints before computation."

- ❌ Incorrect: "You can express descending iteration." (explicit subject - too verbose for summary)
- ❌ Incorrect: "Compute the total number of distinct map foldings." (imperative mood without modal - sounds commanding)
- ❌ Incorrect: "Computes foldings." (third person, not second person)
- ❌ Incorrect: "Returns the iterator." (too passive, not focused on reader capability)

**Why this matters:**

- Implied subject keeps the summary line concise and scannable.
- Dynamic modality (even when implied) respects the reader's autonomy.
- The tone remains informative and empowering without verbosity.

**Body text (extended description):**
After the terse summary line, start the body text with an **explicit subject** ("You can...", "The identifier...", "This function...") to provide context and smooth the transition from the concise summary to the detailed explanation.

**For private identifiers** (functions, classes, methods, variables with leading underscore):

The summary line must be written in **first-person descriptive voice** explaining how the identifier functions within the codebase architecture.

**What this means:**

- **First person**: Use "I use this..." from the developer's perspective.
- **Descriptive**: Explain the identifier's role in relation to other code, such as control flow, shared subroutines, or something else.
- **Architectural focus**: Clarify how this private component fits into the larger system.

**Examples:**

- ✅ Correct: "I use this shared subroutine for validating input across multiple public methods."
- ✅ Correct: "I use this to manage control flow between the preprocessing and computation phases."
- ✅ Correct: "I use this to cache intermediate results during recursive folding operations."

- ❌ Incorrect: "You can validate input." (second person - implies public API)
- ❌ Incorrect: "Validates input." (third person - too passive, doesn't explain architectural role)
- ❌ Incorrect: "Validate input before processing." (imperative - doesn't explain usage context)

**Why this matters:**

- Descriptive tone explains *how* and *why* this code exists in relation to the system, not just *what* it does.
- The purpose remains informative - helping the reader understand the codebase architecture.

### Opening and Closing

- Start the summary on the same line as the opening triple quotes.
- Use a full stop to end every sentence.

### AI-Generated Marker

When creating a new docstring, structure it as follows:

```python
def exampleFunction(parameterFirst: int, parameterSecond: str = "default") -> bool:
 """Validate input parameters and return success status.

 (AI generated docstring)

 You can use this function to validate input parameters before processing. The function
 checks each parameter against defined constraints and returns success status. Extended
 description uses explicit subjects to provide context. Cite references in text [1].

 Parameters
 ----------
 parameterFirst : int
  Description of the parameter.
 parameterSecond : str = "default"
  Description with default value noted after the type.

 Returns
 -------
 isValid : bool
  Description of the return value.

 References
 ----------
 [1] Relevant documentation or concept being referenced
  https://example.com/documentation
 """
```

When reformatting an existing docstring, do NOT add or remove "(AI generated docstring)".

### Mathematical Notation

- **No LaTeX**: Use Unicode characters for mathematical symbols in all sections.
- **Mathematical Basis Section**: For functions with a strong mathematical foundation, create a dedicated `Mathematical Basis` section.
  - **Audience**: Academic experts.
  - **Content**: Use precise, academic terms and standard notation (Unicode).
  - **Establishing Statements**: Use statements like "Let p be..." to bridge the gap between code identifiers and academic nomenclature.
  - **Constraint**: Do not simplify explanations in this section.
- **General Documentation**: Do not include complex mathematical derivations or dense notation in the general docstring sections (Summary, Parameters, etc.). Keep the general explanation accessible to a wider audience.

### Section Formatting (NumPy Style)

Use this exact structure:

```text
[blank line]
Section Name
------------
entry : datatype
 Definition or description.
entry : datatype = defaultValue
 Definition or description.
```

**Standard section names** (use exactly these):

- `Parameters` (never "Args" or "Arguments")
- `Returns`
- `Raises`
- `Yields`
- `Attributes`
- `Examples`
- `See Also`
- `References`

**Custom section names for specific content:**

- Never use a generic "Notes" section.
- Name sections based on their actual content:
  - `Mathematical Basis` (see Mathematical Notation section)
  - `Performance Considerations`
  - `Thread Safety`
  - `Memory Usage`
  - `Algorithm Details`
  - `Implementation Notes`
  - `Compatibility`
  - etc.

This specificity helps readers quickly find relevant information.

### Parameters Section

- Use the parameter name exactly as it appears in the signature.
- Include the type annotation from the signature.
- If a default value exists (in signature or logic), append `= defaultValue` after the type.
- Use four-space indentation for the description.

### Returns Section

- Use a meaningful identifier, not just the type.
- Format: `meaningfulName : returnType`

### References Section

Docstrings must provide navigation to related code and external resources. All references should be collected in a `References` section at the end of the docstring, numbered sequentially, and cited in the text using `[1]`, `[2]`, etc.

**What needs a reference:**

1. **External packages and APIs**: Any mention of third-party libraries or their components
   - `pandas.DataFrame`, `numpy.ndarray`, `requests.Session`
   - Use Context7 links when available for well-documented packages
   - Use official documentation links as fallback

2. **Standard library items** (selective):
   - Rarely used modules or functions
   - Edge cases or subtle behaviors
   - Complex APIs (e.g., `asyncio`, `collections.abc`, `operator`)
   - Skip universally known items (e.g., `list`, `dict`, `str`, `int`)

3. **Same-package references**:
   - Functions, classes, or methods defined elsewhere in the package
   - Related utilities
   - Algorithms or data structures used internally

4. **Theoretical foundations**:
   - Mathematical concepts (group theory, graph theory, combinatorics)
   - Algorithms (sorting, searching, optimization)
   - Academic papers or textbooks
   - Wikipedia articles for established concepts

5. **Standards and specifications**:
   - IETF RFCs, W3C specifications, PEPs
   - File formats (JSON, CSV, HDF5)
   - Protocols (HTTP, TCP, MQTT)

**Backticks are a heuristic**: Backticked identifiers often need references, but not always (skip if in signature). Conversely, some concepts need references even without backticks (e.g., "Chinese numerals" refers to a cultural/linguistic system).

**Reference format:**

```python
def exampleFunction(data: list[int]) -> int:
 """Compute the sum using this positional notation system [1].

 This function implements the algorithm described in [2] using `gmpy2` [3]
 for arbitrary-precision arithmetic. The approach is related to
 `mapFolding.basecamp.countFolds` [4].

 Parameters
 ----------
 data : list[int]
  Input values in positional notation.

 Returns
 -------
 total : int
  Sum of input values.

 References
 ----------
 [1] Positional notation - Wikipedia
  https://en.wikipedia.org/wiki/Positional_notation
 [2] Knuth, D. E. (1997). The Art of Computer Programming, Volume 2:
  Seminumerical Algorithms (3rd ed.). Addison-Wesley.
 [3] gmpy2 - Context7
  https://gmpy2.readthedocs.io/en/latest/
 [4] mapFolding.basecamp.countFolds
 """
```

**Reference citation guidelines:**

- **Verification Required**: You must verify every external link you add. When modifying an existing docstring, verify all existing links as a maintenance task.
- Cite references in the text using `[1]`, `[2]`, etc., immediately after the relevant term or concept
- Multiple references for one concept: `[1, 2]` or `[1][2]` (be consistent within a docstring)
- List references in order of first citation
- Include both title/description and URL for web resources
- For packages: prefer Context7 links when available
- For academic papers: include full citation (author, year, title, publication)
- For internal package references: use module path, note "Internal package reference"

**Common external packages:**

- `hunterMakesPy`: <https://context7.com/hunterhogan/huntermakespy>
- `astToolkit`: <https://context7.com/hunterhogan/asttoolkit>
- `numpy`: <https://numpy.org/doc/stable/reference/index.html>
- `pandas`: <https://pandas.pydata.org/docs/reference/index.html>
- `gmpy2`: <https://gmpy2.readthedocs.io/en/latest/>
- `numba`: <https://numba.readthedocs.io/en/stable/>

### Examples Section

**Private Identifiers**: Private identifiers (starting with `_`) do not require an `Examples` section.

**Real code only**: Examples must be drawn from actual usage in the codebase.

**Workflow for finding examples**:

1. Use code search to find real invocations of the documented function/class.
2. Select representative usage that demonstrates typical or important patterns.
3. Simplify if needed, but preserve the identifier style and conventions of the codebase.
4. If no real usage exists in the codebase, note this limitation.

**Never create contrived examples**. Real code demonstrates actual patterns and validates that the function works as documented.

- Examples must follow the identifier and style conventions of the codebase.
- Include context if needed for clarity (imports, setup, etc.).

## Terminology and Semiotic Rules

### Docstrings Are Not Prose

**Critical distinction**: Literary prose avoids repetition through synonyms and pronouns. Technical documentation requires consistency and precision.

**In docstrings:**

- **Repeat nouns instead of using pronouns**: Write "`parameterFirst`" every time, not "it" or "the parameter".
- **Repeat exact identifiers**: Don't vary between "`handler`", "the handler", or "this callback".
- **Avoid synonyms for precision**: Don't alternate between "function" and "routine", or "list" and "sequence" unless they mean different things.
- **Consistency aids comprehension**: Non-native speakers and machine translation tools rely on consistent terminology.

**Why this matters:**

- Pronouns create ambiguity: "it" can refer to multiple prior nouns.
- Synonyms suggest false distinctions: readers wonder if "handler" and "callback" mean different things.
- Repetition in technical writing establishes clear, unambiguous reference chains.

**Example:**

❌ **Bad** (prose-style with pronouns and synonyms):

> Process the data using the handler. It will transform the input
and pass the result to the callback, which processes it further.

✅ **Good** (technical style with repetition):

> Process `data` using `handler`. The `handler` transforms `data`
and passes `data` to `handler`, which processes `data` further.

### Backtick Usage

- Always use backticks when referring to:
  - Identifiers and other labels: `parameterName`, `className`
  - Python keywords: `try`, `except`, `class`
  - Types: `Exception`, `None`, `bool`
  - Code elements: `self`, `cls`
- Natural language discussing concepts uses standard English without backticks.
- Never write lowercase technical terms: write `Exception`, never "exception" when referring to the Python type.

### Vocabulary Prohibitions

- Never use "pipeline". Use "assembly line" if describing a sequence of transformations.
- Avoid end-of-line colons in prose (machine parsers interpret them as syntax).

### Technical Complexity Separation

- **General Sections (Summary, Parameters, Returns)**: Write for a competent developer who may not be a domain expert. Avoid implementation-specific jargon (e.g., "mutable object", "thread-safe", "atomic") unless it is the primary function of the code.
- **Technical Detail Sections**: Put implementation details, hardware constraints, or expert-level warnings in well-named custom sections (e.g., `Concurrency`, `Memory Management`, `Algorithm Details`).
  - **Audience**: Experts who need to know.
  - **Tone**: Efficient, precise, technical.
  - **Constraint**: Do not explain standard concepts (like "don't share mutable objects between threads") to experts. Only document non-obvious behaviors or critical constraints.

### Plurality and Identifiers

- **Do not force English plurality on identifiers**: `keyword` not `keywords` or `keyword`s.
- **Preserve parameter names exactly**: `handlers` (the field) not "the handlers parameter".
- **Use generalized nouns to avoid plurality**: Do not pluralize identifiers or add possessives to them. Instead, precede the strict identifier with a generalized noun or quantifier.
  - ❌ Incorrect: "Returns the `iterator`'s `leaves`." (Modified identifiers)
  - ✅ Correct: "Returns the `iterator` of `type` `int` `leaf`." (Noun "type int" precedes `leaf`)
  - ❌ Incorrect: "Enumerate `dimensionIndices`." (Pluralized identifier)
  - ✅ Correct: "Enumerate each `dimensionIndex`." (Quantifier "each" precedes `dimensionIndex`)
- **Reinforce meaning through context** rather than breaking identifiers for prose.

### Disambiguation

- Qualify ambiguous terms: "Python keyword `try`" not just "`try`".
- Maintain precision: "`class` `Exception`" rather than "the `Exception` `class`".
- Avoid compound hyphenation with technical terms: not "`Exception`-free" but "successful completion".

### Writing for International and Machine Translation Audiences

**Assume your audience:**

- Two-thirds of the humans are non-native English speakers
- Half will use machine translation tools to some degree (Google Translate, DeepL, etc.)
- Future readers will use the docstring to generate new code

**Guidelines:**

- **Use simple sentence structures**: Avoid complex subordinate clauses when possible.
- **Prefer active voice**: "This function returns X" rather than "X is returned by this function".
- **Avoid idioms and colloquialisms**: "fails quickly" not "bails out", "prevents errors" not "catches issues".
- **Use consistent terminology**: Machine translation works better with repeated exact phrases.
- **Minimize ambiguity**: Pronouns confuse both humans and translation systems.
- **Be explicit**: Don't rely on cultural context or implied knowledge.
- **Short, complete sentences**: Better for parsing and translation than long, flowing prose.

**Test your writing**: If a sentence would be difficult to translate word-by-word, rewrite it more explicitly.

## Scope-Specific Standards

### Module-Level Docstrings

Place at the very top of the file, before imports.

**Required components:**

1. Summary (second-person declarative with dynamic modality and implied subject)
2. Extended description of module purpose
3. **Table of contents listing all public identifiers**

**Format:**

```python
"""Perform map-folding computations and analysis.

(AI generated docstring)

You can use this module to compute distinct foldings of multi-dimensional maps. The module
provides the core algorithms and orchestrates computation workflows. Extended description
of module purpose and organization.

Contents
--------
Functions
 functionAlpha
  One-line description of functionAlpha.
 functionBeta
  One-line description of functionBeta.

Classes
 ClassAlpha
  One-line description of ClassAlpha.

"""
import ...
```

**Table of contents requirements:**

- List all public functions, classes, and other exported identifiers.
- If there are sections (`#======== Boolean filters ====`) and subsections (`#---- Specific filters ----`), reflect that hierarchy.
- Maintain alphabetical order within each section.
- Provide one-line descriptions for each item.
- Omit private identifiers (those starting with `_`).

### Package `__init__.py` Docstrings

Document what the package exposes and its organizational purpose.

**Required components:**

1. Summary (second-person declarative with dynamic modality and implied subject)
2. Extended description
3. **Table of contents of public modules and subpackages**

**Format:**

```python
"""Access map-folding algorithms and utilities.

(AI generated docstring)

You can use this package to compute distinct foldings of multi-dimensional maps. This package
provides implementations of various map-folding counting algorithms, state management for
parallel computations, and interfaces to OEIS sequences.

Modules
-------
basecamp
 Entry point for map-folding computations with algorithm selection.
dataBaskets
 State containers and data structures for folding computations.
filesystemToolkit
 Utilities for result persistence and caching.

Subpackages
-----------
algorithms
 Collection of map-folding algorithm implementations.
_e
 Experimental elimination-based algorithms (internal research code).
"""
```

**Table of contents requirements:**

- List public modules with brief descriptions
- List subpackages with brief descriptions
- Use section headers (`Modules`, `Subpackages`) with dashed underlines
- Omit private modules (starting with `_`) unless they are documented internal APIs

### Class Docstrings

```python
class ExampleClass:
 """Manage map folding computation state.

 (AI generated docstring)

 You can use this class to encapsulate and manage the state needed for parallel
 map-folding computations. Extended description of behavior and usage.

 Attributes
 ----------
 attributeName : type
  Description of the attribute.
 """
```

### Class and Instance Variable Docstrings

For class-level or instance-level variables that need documentation:

```python
class Configuration:
 """Store and manage configuration settings.

 (AI generated docstring)

 You can use this class to maintain configuration values across the application.

 Attributes
 ----------
 settingDefault : int
  The default setting value applied when no override is provided.
 """

 settingDefault: int = 42
 """The default setting value applied when no override is provided."""
```

## Type Annotation Requirements

While the primary task is docstrings, observe these rules when type annotations are encountered:

- Always use PEP 585 style (`list[int]`, not `List[int]`).
- Use PEP 604 style (`int | None`, not `Optional[int]`; `int | str`, not `Union[int, str]`).
