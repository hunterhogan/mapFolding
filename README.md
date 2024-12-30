# A modular foundation for improvements

This version is not faster than the original version or the inefficient tasks version. But, because this version has stricter
separation of concerns, it is easier to use to prototype ideas and compare and contrast different methodologies.

The likely best place to make efficiency gains is the `while` loop of the algorithm. That logic is isolated in
"lovelace.py" and I aspire for the other modules to support optimization of that module.

## Side note

Despite [the word "algorithm" ultimately being derived from](https://www.etymonline.com/word/algorithm#etymonline_v_8145)
the name of the incomparable Muhammad ibn Musa [al-Khwarizmi](https://en.wikipedia.org/wiki/Al-Khwarizmi),
the concept of the computer algorithm was created by [Ada Lovelace](https://en.wikipedia.org/wiki/Ada_Lovelace) in approximately 1843.
It is difficult to overstate the magnitude of this accomplishment. Consider, for example, that in 1900,
[David Hilbert's](https://en.wikipedia.org/wiki/David_Hilbert) 23 Problems included a challenge to create what we call an algorithm, but
algorithmic analysis didn't exist and they did not have a word for it at the time.

Lovelace's accomplishments could not even be contextualized before the corpus
of literature produced by Kurt Gödel, Alonzo Church, Alan Turing, and their contemporaries. Before the rest of the world caught up to Lovelace, a few things happened: adoption of the wired telegraph, invention of electric networks and electrical devices, invention of the wireless telegraph (the radio), and the invention of powered flight, to name a few.

Therefore, the core of the `foldings` algorithm is in the module, "lovelace.py."
