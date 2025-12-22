"""
Generating a constrained sequence of permutations.

1.	A column can hold one value from the full domain of all possible values.
	A.	The domain of the column is an independent copy of the full domain of all possible values.
	B.	Each column has a unique index corresponding to its position on a line.
	C.	The first column is at position 0 and its index is 0.
	D.	Position values and indices increase by one with each stop to the right.
2.	The carriage starts at position -1, which does not have a column, and moves to the right, which is position 0.
	A.	The scribe performs an action, and the carriage moves.
		1.	Actions: assign a value, assign no value, and collect values.
		2.	Moves: move to the right and move to the left.
	B.	The scribe checks if the position does or does not have a column.
		1.	If the position has a column, the scribe checks if the domain of the column does or does not have values.
			A.	If the domain of the column does not have values, the scribe assigns no value. The carriage moves to the left.
			B.	If the domain of the column has values, the scribe assigns a value from the domain of the column. The carriage moves to the right.
		2.	If the position does not have a column, the scribe checks if the positions/columns to the left have values or are empty.
			A.	If the positions to the left are empty, the scribe does nothing (because the carriage has moved to position -1). The carriage does not move.
			B.	If the positions to the left have values, the scribe records the sequence of values in the list of sequences. The carriage moves to the left.
3.	A filter reduces the domain of possible values for one or more columns.
	A.	The scope of the filter defines the columns that are affected by the filter.
	B.	An expiration index of a filter defines when the filter permanently stops affecting the columns. (See janitor.)
	C.	When a filter is created, an unchangeable expiration index corresponding to the position to the left of the carriage is automatically added.
	D.	A filter may have more than one expiration index.
	E.	An expiration index must not be in the scope of columns affected by the filter.
	F.	The filter must be independent; the filter application order must be irrelevant.
4.	When the carriage moves to a position, the janitor¹ deletes each filter that has an expiration index matching the index of the carriage's position and removes the column's assigned value.
5.	A bureaucrat creates a filter if the state of the system matches a prescribed condition.
	A.	Each column has a bureaucrat that monitors the domain of the column.
		1.	Condition: if the domain of the column is empty, and if the column does not have an assigned value, the bureaucrat creates a filter.
		2.	The filter's effect: remove all values from the column's domain. The scope: all columns to the right of the carriage.
	B.	Each column has a bureaucrat that monitors the assigned value of the column.
		1.	Condition: if the column has an assigned value, the bureaucrat creates a filter.
		2.	The filter's effect: remove the assigned value from a column's domain. The scope: the bureaucrat's column and all columns to the right of the column.
	C.	To restrict a column's value to exactly one possibility, for example, implement a bureaucrat to create a filter before the first move that restricts the domain of the column to the desired value.
6.	Steps occur in the following order and do not overlap with other steps:
	A.	Bureaucrats create filters.
	B.	The carriage moves.
	C.	The janitor acts.
	D.	The scribe acts.

Technical details
- The bureaucrats can and probably should work concurrently with each other.
- This is not a recursive or backtracking permutation generator. Cf., e.g., https://github.com/TheAlgorithms/Python/blob/master/data_structures/arrays/permutations.py.

Observations
- This complex system relies on simple rules and actions.
- The "actors" should be stupid, weak, and ignorant. Note that the scribe, the most sophisticated actor, for example, doesn't know the position index, the most important data point.
- Implement your constraints by "hiring" a bureaucrat to create a filter.
- The janitor is not a bureaucrat.
- Filters only restrict the domain.
- Nothing can add to the domain.
- The actors are not part of the state of the system.
- The filters are not part of the state of the system.
- After their creation, filters cannot be modified.
- Filters cannot be deleted: they expire automatically.
- Create concurrency by dividing the domain of one column into non-overlapping subdomains and running multiple instances of the system, each with one subdomain.

Question: is the state equal to: the assigned values and the domains of the columns?

¹ "Janitor" is closely linked to "Janus", the Roman "guardian god of portals, doors, and gates." See, e.g., https://www.etymonline.com/word/janitor.
"""
