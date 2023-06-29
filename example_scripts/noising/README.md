# Text noising for Icelandic

This submodule implements various ways to add spelling issues, grammatical noise and so on for Icelandic text.


## Usage
Provide a file of tokenized text, one sentence per line

```python
python generate_errors.py < input_file.txt
```

You can adjust the error rate using the ``--word-spelling-error-rate`` and ``--rule-chance-error-rate`` arguments (on a scale of 0 to 1).

For adapting this error generation code to other languages, you can disable the Icelandic-specific errors in ``generate_errors.py`` (``DativitisErrorRule``, ``MoodErrorRule``, ``NounCaseErrorRule``, ``SplitWordErrorRule``) and compile a list of common issues for the chosen language in the ``spelling`` module under ``rules`` (see ``regex.txt``, ``simple,txt`` and ``word_pairs.txt``).
