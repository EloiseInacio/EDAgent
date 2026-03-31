# Core questions

1. What is the downstream task or application?
2. What does one row represent?
3. What is the target label and its granularity?
4. What split or grouping variables should be respected?

# SQL database follow-up

Ask these only when the data source is a SQL database (not a CSV file):
- What is the database type and connection string? (e.g. `sqlite:///path/to/file.db`, `postgresql://user:pass@host/dbname`)
- Which table is the primary analysis target?
- Are there additional tables to join? If so: which tables, which columns to join on, and what join type (inner/left)?
- Should any rows be filtered (WHERE clause)? Should the analysis be limited to a specific column subset?

# Data access follow-up

Ask this only if asset files are referenced in the manifest:
- Are the asset paths absolute or relative? If relative, what is the base directory to resolve them from?

# Meta-data follow-up

Ask this only if relevant:
- Are metadata detector-generated or manually annotated?

# Audio data follow-up

Ask these only when audio data is detected:
- Is the audio speech, music, environmental sound, or mixed?
- What is the expected sample rate (e.g. 8 kHz telephony, 16 kHz speech, 44.1 kHz music)?
- Is audio mono or stereo? Should channels be analyzed independently?
- What type of annotation is provided: event labels, transcripts, continuous scores, or binary flags?
- Are there known recording conditions (microphone type, noise level) that may introduce domain-specific bias?