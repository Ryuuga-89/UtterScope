# UtterScope

**Local-first speech analytics for language learners.**

UtterScope is an open-source CLI tool for analyzing recorded conversations and language lessons.

Give it an audio file, and UtterScope transcribes the conversation, identifies speakers, extracts the learner's speech, and provides useful metrics such as speaking time, speaking rate, pauses, and fillers — while keeping audio processing local.

```bash
utterscope analyze lesson.mp3
```

```text
╭─ UtterScope ─────────────────────────────────╮
│ lesson.mp3                         25m 14s    │
╰──────────────────────────────────────────────╯

  ✓ Preparing audio
  ✓ Detecting speech
  ✓ Transcribing
  ✓ Identifying speakers
  ✓ Analyzing learner

Learner
───────────────────────────────────────────────
Speaking time                              10:32
Speaking ratio                             41.8%
Speaking rate                            104 WPM
Long pauses                                   13
Fillers                                       27

Report → ./report.html
```

## Why UtterScope?

Speaking practice is difficult to measure.

After a conversation lesson, you may remember that you struggled with a few expressions, but it is difficult to answer questions such as:

* How much did I actually speak?
* Am I speaking faster than a month ago?
* Am I pausing less frequently?
* Which filler words do I overuse?
* Are my utterances becoming longer?
* What expressions or grammatical patterns do I repeatedly struggle with?

UtterScope turns recorded conversations into structured data so that speaking progress can be observed over time.

## Features

### Local transcription

Audio is transcribed locally using Whisper-based speech recognition optimized for local hardware.

Your raw lesson audio does not need to be uploaded to an external transcription service.

### Speaker diarization

UtterScope separates speakers in a conversation and identifies the utterances that belong to the learner.

```text
Teacher  00:17  What do you think about living in a small apartment?

Learner  00:22  I think... it is a little narrow, so I feel pressure.

Teacher  00:29  I see. What would you change about it?
```

### Speaking metrics

UtterScope calculates deterministic metrics directly from the transcript and audio.

Examples include:

* speaking time
* speaking ratio
* words per minute
* mean utterance length
* turn count
* pause frequency
* long pauses
* filler frequency

These metrics do not require an LLM.

### Optional language analysis

LLM-based analysis can optionally provide higher-level feedback such as:

* grammar issues
* unnatural expressions
* vocabulary suggestions
* alternative expressions
* recurring mistakes

The core transcription and quantitative analysis remain usable without an LLM.

### Reports

Analysis results for each run are stored together in one directory:

```text
results/
└── 250919-1_lesson/
    ├── lesson.mp3          # copy of the input audio
    ├── transcript.json
    ├── analysis.json
    ├── feedback.json       # optional (--llm)
    ├── report.md           # human-readable summary
    └── report.html         # interactive timeline + audio
```

`report.md` and `report.html` are always written at the end of analyze (even with
`--no-llm`; the feedback sections note that LLM was skipped).
## How it works

```text
Audio
  │
  ▼
FFmpeg
  │
  ▼
Voice Activity Detection
  │
  ├───────────────┐
  ▼               ▼
ASR          Diarization
  │               │
  └───────┬───────┘
          ▼
 Structured Transcript
          │
     ┌────┴────┐
     ▼         ▼
  Metrics   LLM Analysis
             optional
     │         │
     └────┬────┘
          ▼
       Report
```

UtterScope is designed around interchangeable components so that transcription, diarization, and language-analysis backends can evolve independently.

## Installation

> UtterScope is currently under development.

Clone the repository:

```bash
git clone <repository-url>
cd utterscope
```

Install dependencies using [uv]:

```bash
uv sync
```

Run UtterScope:

```bash
uv run utterscope analyze path/to/lesson.mp3
```

For speaker diarization and optional LLM feedback, run the interactive setup once:

```bash
uv run utterscope setup
```

Setup can save `HF_TOKEN` (pyannote) and `GEMINI_API_KEY` (Gemini feedback). Without a Gemini key, pass `--no-llm` or analysis will exit with an error when `--llm` is on (the default).

## Usage

Run without arguments for the interactive flow (audio path → ASR model →
analyze → learner → optional LLM provider/model):

```bash
utterscope
```

Or analyze a recorded lesson with flags. When multiple speakers are detected, pick the learner with the ○/● radio UI:

```bash
utterscope analyze lesson.mp3
```

Skip interactive selection by passing a speaker id (required for non-interactive `analyze`):

```bash
utterscope analyze lesson.mp3 --learner SPEAKER_01
```

Specify an ASR model:

```bash
utterscope analyze lesson.mp3 --model large-v3-turbo
```

Change how long a pause must be to count as "long". The CLI flag wins; otherwise setup / `.env` (`UTTERSCOPE_LONG_PAUSE_THRESHOLD`) is used (default: 1.0s):

```bash
utterscope analyze lesson.mp3 --long-pause-threshold 1.5
```

You can also set the same value interactively with `utterscope setup`.

Run without LLM-based analysis:

```bash
utterscope analyze lesson.mp3 --no-llm
```

Export structured results. `--output` sets the **root**; each run creates a
timestamped folder `YYMMDD-n_<audio-stem>/` inside it (default root: `./results`):

```bash
utterscope analyze lesson.mp3 --learner SPEAKER_01 --output ./results
```

Outputs:

```text
./results/
└── 250919-1_lesson/
    ├── lesson.mp3
    ├── transcript.json   # diarized transcript
    ├── analysis.json     # learner metrics (time, WPM, pauses, fillers, …)
    ├── feedback.json     # optional LLM feedback (--llm)
    ├── report.md
    └── report.html
```

Configure a fixed absolute output root or “ask every time” (interactive only)
with `utterscope setup`.

## Architecture

UtterScope is primarily written in Python.

The current technology stack includes:

| Component                | Technology            |
| ------------------------ | --------------------- |
| CLI                      | Typer + Rich          |
| Package management       | uv                    |
| Data models              | Pydantic              |
| Audio processing         | FFmpeg                |
| Voice activity detection | Silero VAD            |
| Speech recognition       | MLX Whisper / Whisper |
| Speaker diarization      | pyannote.audio        |
| Report generation        | Jinja2                |
| Storage                  | JSON / SQLite         |
| Linting & formatting     | Ruff                  |
| Type checking            | ty                    |
| Testing                  | pytest                |

The project is designed so that major inference components can eventually be replaced by alternative backends.

## Design Principles

### Local first

Audio contains highly personal information.

UtterScope therefore aims to perform transcription, diarization, and quantitative analysis locally whenever possible.

### Deterministic where possible

An LLM should not calculate something that can be measured directly.

Metrics such as speaking time, WPM, pause duration, and filler frequency are computed deterministically.

LLMs are reserved for tasks where semantic understanding is useful.

### Backend agnostic

Speech recognition and language analysis should not depend permanently on one model or provider.

The long-term goal is to support interchangeable backends for:

* local Whisper implementations
* alternative ASR models
* local LLMs
* hosted LLM APIs

### CLI first

UtterScope is designed as a command-line application first.

The core analysis pipeline should remain usable without running a web server or installing a JavaScript frontend.

## Roadmap

### v0.1 — Transcription

* [x] Rich CLI
* [x] audio preprocessing
* [x] local Whisper transcription
* [x] timestamped transcript
* [x] JSON export

### v0.2 — Conversation analysis

* [x] voice activity detection
* [x] speaker diarization
* [x] learner-speaker selection
* [x] speaking time
* [x] speaking ratio
* [x] words per minute
* [x] pause analysis
* [x] filler detection

### v0.3 — Language feedback

* [x] optional LLM integration
* [x] grammar feedback
* [x] naturalness feedback
* [x] vocabulary analysis
* [x] recurring-error detection

### v0.4 — Progress tracking

* [x] HTML reports
* [ ] lesson history
* [ ] SQLite storage
* [ ] lesson comparison
* [ ] longitudinal speaking metrics

### Future

* [ ] additional ASR backends
* [ ] local LLM support
* [ ] interactive transcript
* [ ] audio-linked utterances
* [ ] pronunciation analysis
* [ ] web dashboard

## Privacy

UtterScope is designed to keep raw audio on the user's machine.

Some optional features may use external APIs. When enabled, the data sent to those services depends on the selected provider and configuration.

The goal is to make external processing explicit and optional.

## Contributing

UtterScope is in an early stage of development.

Issues, feature proposals, bug reports, and pull requests are welcome.

If you are interested in speech recognition, language learning, audio processing, or local AI, feel free to contribute.

## License

TBD
